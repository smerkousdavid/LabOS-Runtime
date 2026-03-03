#!/usr/bin/env python3
# IMPORTANT: Disable xformers BEFORE any other imports to avoid triton issues on Windows
import os
os.environ["XFORMERS_DISABLED"] = "1"

"""
LingBot-Depth Integration Module

Provides depth refinement and completion using the LingBot-Depth model.
Transforms sparse/noisy RealSense depth into high-quality, metric-accurate depth maps.

All depth values in this module are in MILLIMETERS for consistency with RealSense SDK.
The model internally works in meters, but conversion is handled automatically.

Usage:
    from depth import LingBotDepth
    
    # Initialize
    depth_refiner = LingBotDepth()
    
    # Refine depth (all values in mm)
    refined_depth_mm = depth_refiner.refine(rgb_image, raw_depth_mm, intrinsics)
    
    # With 3D point cloud
    refined_depth_mm, points_mm = depth_refiner.refine_with_points(rgb_image, raw_depth_mm, intrinsics)

References:
    - https://github.com/robbyant/lingbot-depth
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
import torch

# Add lingbot-depth to path
SCRIPT_DIR = Path(__file__).parent.resolve()
LINGBOT_PATH = SCRIPT_DIR / "lingbot-depth"
if LINGBOT_PATH.exists() and str(LINGBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LINGBOT_PATH))

# Lazy import flag - model is imported only when needed
LINGBOT_AVAILABLE = None  # None = not checked yet, True/False = checked
MDMModel = None  # Will be imported lazily


def _import_lingbot_model():
    """Lazily import the LingBot-Depth model to avoid xformers/triton issues at module load time."""
    global LINGBOT_AVAILABLE, MDMModel
    
    if LINGBOT_AVAILABLE is not None:
        return LINGBOT_AVAILABLE
    
    try:
        from mdm.model.v2 import MDMModel as _MDMModel
        MDMModel = _MDMModel
        LINGBOT_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"Warning: Could not import LingBot-Depth model: {e}")
        print("Make sure lingbot-depth is installed: pip install -e lingbot-depth/")
        LINGBOT_AVAILABLE = False
        return False
    except TypeError as e:
        # xformers/triton compatibility issue
        print(f"Warning: xformers/triton compatibility issue: {e}")
        print("Try: pip install xformers --upgrade")
        print("Or: pip uninstall xformers triton && pip install xformers triton")
        LINGBOT_AVAILABLE = False
        return False
    except Exception as e:
        print(f"Warning: Error loading LingBot-Depth: {e}")
        LINGBOT_AVAILABLE = False
        return False


class LingBotDepth:
    """
    LingBot-Depth wrapper for depth refinement and completion.
    
    This class handles:
    - Loading the model from local weights or HuggingFace
    - Converting between mm (RealSense) and meters (model)
    - Normalizing camera intrinsics
    - Running inference with proper tensor handling
    
    All public methods work with depth in MILLIMETERS.
    """
    
    DEFAULT_WEIGHTS_PATH = SCRIPT_DIR / "weights" / "lingbot.pt"
    
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        device: str = "auto",
        use_fp16: bool = True,
        resolution_level: int = 9
    ):
        """
        Initialize the LingBot-Depth model.
        
        Args:
            model_path: Path to model weights (.pt file) or HuggingFace model ID.
                       If None, uses default weights at version2/weights/lingbot.pt
            device: Device to use ('auto', 'cuda', 'cpu')
            use_fp16: Use mixed precision for faster inference
            resolution_level: Resolution level (0-9, higher = better quality but slower)
        """
        # Lazy import the model
        if not _import_lingbot_model():
            raise ImportError("LingBot-Depth model not available. See error messages above.")
        
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.use_fp16 = use_fp16
        self.resolution_level = resolution_level
        
        # Resolve model path
        if model_path is None:
            model_path = self.DEFAULT_WEIGHTS_PATH
        
        model_path = Path(model_path)
        
        print(f"Loading LingBot-Depth model...")
        print(f"  Device: {self.device}")
        
        if model_path.exists():
            print(f"  Weights: {model_path}")
            self.model = MDMModel.from_pretrained(str(model_path)).to(self.device)
        else:
            # Try as HuggingFace model ID
            print(f"  Model: {model_path} (HuggingFace)")
            self.model = MDMModel.from_pretrained(str(model_path)).to(self.device)
        
        # Use PyTorch native SDPA instead of xformers (avoids triton compatibility issues)
        self.model.enable_pytorch_native_sdpa()
        
        self.model.eval()
        print(f"  Model loaded successfully (using PyTorch native SDPA)!")
        
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory: {gpu_mem:.1f} GB")
    
    def _preprocess_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        """
        Preprocess RGB image for the model.
        
        Args:
            rgb: RGB image (H, W, 3) uint8 or float, BGR or RGB
            
        Returns:
            Tensor (1, 3, H, W) float32 normalized to [0, 1]
        """
        # Convert to float if needed
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        elif rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        
        # Ensure RGB order (assume input could be BGR from OpenCV)
        # User should pass RGB, but we'll document this clearly
        
        # Convert to tensor: (H, W, 3) -> (1, 3, H, W)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(dtype=torch.float32, device=self.device)
        
        return tensor
    
    def _preprocess_depth(self, depth_mm: np.ndarray) -> torch.Tensor:
        """
        Preprocess depth map for the model.
        
        Args:
            depth_mm: Depth in millimeters (H, W), any numeric type
            
        Returns:
            Tensor (1, H, W) float32 in METERS
        """
        # Convert to float32 meters
        depth_m = depth_mm.astype(np.float32) / 1000.0
        
        # Handle invalid values
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensor
        tensor = torch.from_numpy(depth_m).unsqueeze(0)
        tensor = tensor.to(dtype=torch.float32, device=self.device)
        
        return tensor
    
    def _normalize_intrinsics(
        self, 
        intrinsics: np.ndarray, 
        width: int, 
        height: int
    ) -> torch.Tensor:
        """
        Normalize camera intrinsics for the model.
        
        Args:
            intrinsics: 3x3 camera matrix with fx, fy, cx, cy in pixels
            width: Image width
            height: Image height
            
        Returns:
            Tensor (1, 3, 3) with normalized intrinsics
        """
        K = intrinsics.astype(np.float32).copy()
        
        # Normalize: fx/W, fy/H, cx/W, cy/H
        K[0, 0] /= width   # fx
        K[0, 2] /= width   # cx
        K[1, 1] /= height  # fy
        K[1, 2] /= height  # cy
        
        tensor = torch.from_numpy(K).unsqueeze(0)
        tensor = tensor.to(dtype=torch.float32, device=self.device)
        
        return tensor
    
    @torch.inference_mode()
    def refine(
        self,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics: np.ndarray,
        apply_mask: bool = True
    ) -> np.ndarray:
        """
        Refine/complete a sparse depth map.
        
        Args:
            rgb: RGB image (H, W, 3), uint8 [0-255] or float [0-1]
            depth_mm: Raw depth in MILLIMETERS (H, W)
            intrinsics: 3x3 camera intrinsics matrix (fx, fy, cx, cy in pixels)
            apply_mask: Apply validity mask to output
            
        Returns:
            Refined depth in MILLIMETERS (H, W), float32
        """
        h, w = rgb.shape[:2]
        
        # Preprocess inputs
        rgb_tensor = self._preprocess_rgb(rgb)
        depth_tensor = self._preprocess_depth(depth_mm)
        intrinsics_tensor = self._normalize_intrinsics(intrinsics, w, h)
        
        # Run inference
        output = self.model.infer(
            rgb_tensor,
            depth_in=depth_tensor,
            intrinsics=intrinsics_tensor,
            apply_mask=apply_mask,
            use_fp16=self.use_fp16,
            resolution_level=self.resolution_level
        )
        
        # Get refined depth (in meters)
        depth_refined_m = output['depth'].squeeze().cpu().numpy()
        
        # Convert back to millimeters
        depth_refined_mm = depth_refined_m * 1000.0
        
        # Handle invalid values
        depth_refined_mm = np.nan_to_num(depth_refined_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_refined_mm[~np.isfinite(depth_refined_mm)] = 0.0
        
        return depth_refined_mm.astype(np.float32)
    
    @torch.inference_mode()
    def refine_with_points(
        self,
        rgb: np.ndarray,
        depth_mm: np.ndarray,
        intrinsics: np.ndarray,
        apply_mask: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine depth and compute 3D point cloud.
        
        Args:
            rgb: RGB image (H, W, 3), uint8 [0-255] or float [0-1]
            depth_mm: Raw depth in MILLIMETERS (H, W)
            intrinsics: 3x3 camera intrinsics matrix (fx, fy, cx, cy in pixels)
            apply_mask: Apply validity mask to output
            
        Returns:
            Tuple of:
            - Refined depth in MILLIMETERS (H, W), float32
            - Point cloud in MILLIMETERS (H, W, 3), float32 [x, y, z]
        """
        h, w = rgb.shape[:2]
        
        # Preprocess inputs
        rgb_tensor = self._preprocess_rgb(rgb)
        depth_tensor = self._preprocess_depth(depth_mm)
        intrinsics_tensor = self._normalize_intrinsics(intrinsics, w, h)
        
        # Run inference
        output = self.model.infer(
            rgb_tensor,
            depth_in=depth_tensor,
            intrinsics=intrinsics_tensor,
            apply_mask=apply_mask,
            use_fp16=self.use_fp16,
            resolution_level=self.resolution_level
        )
        
        # Get refined depth (in meters)
        depth_refined_m = output['depth'].squeeze().cpu().numpy()
        
        # Get points (in meters)
        points_m = output.get('points', None)
        if points_m is not None:
            points_m = points_m.squeeze().cpu().numpy()
            points_mm = points_m * 1000.0
            points_mm = np.nan_to_num(points_mm, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Compute points manually if not returned
            points_mm = self._depth_to_points(depth_refined_m * 1000.0, intrinsics)
        
        # Convert depth to millimeters
        depth_refined_mm = depth_refined_m * 1000.0
        depth_refined_mm = np.nan_to_num(depth_refined_mm, nan=0.0, posinf=0.0, neginf=0.0)
        depth_refined_mm[~np.isfinite(depth_refined_mm)] = 0.0
        
        return depth_refined_mm.astype(np.float32), points_mm.astype(np.float32)
    
    def _depth_to_points(
        self,
        depth_mm: np.ndarray,
        intrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_mm: Depth in millimeters (H, W)
            intrinsics: 3x3 camera matrix (unnormalized, in pixels)
            
        Returns:
            Point cloud (H, W, 3) in millimeters [x, y, z]
        """
        h, w = depth_mm.shape
        
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # Create pixel coordinate grid
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)
        
        # Backproject to 3D
        z = depth_mm
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1)
        
        return points


def depth_to_colormap(
    depth: np.ndarray,
    vmin: float = None,
    vmax: float = None,
    colormap: int = cv2.COLORMAP_TURBO,
    invalid_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Convert depth map to colorized visualization.
    
    Args:
        depth: Depth map (H, W), any units
        vmin: Minimum depth for colormap (auto if None)
        vmax: Maximum depth for colormap (auto if None)
        colormap: OpenCV colormap (TURBO, JET, VIRIDIS, etc.)
        invalid_color: Color for invalid/zero depth pixels (BGR)
        
    Returns:
        Colorized depth (H, W, 3) BGR uint8
    """
    # Create valid mask
    valid_mask = np.isfinite(depth) & (depth > 0)
    depth_clean = depth.copy()
    depth_clean[~valid_mask] = 0
    
    # Auto-range
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1
    
    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    
    # Set invalid pixels
    depth_colored[~valid_mask] = invalid_color
    
    return depth_colored


def realsense_intrinsics_to_matrix(intrinsics) -> np.ndarray:
    """
    Convert RealSense intrinsics to 3x3 matrix.
    
    Args:
        intrinsics: pyrealsense2.intrinsics object
        
    Returns:
        3x3 numpy array camera matrix
    """
    return np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)


def compute_3d_distance(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """
    Compute Euclidean distance between two 3D points.
    
    Args:
        point1: First point [x, y, z]
        point2: Second point [x, y, z]
        
    Returns:
        Distance in same units as input
    """
    return float(np.linalg.norm(np.array(point2) - np.array(point1)))


def pixel_to_3d(
    u: int,
    v: int,
    depth: float,
    intrinsics: np.ndarray
) -> np.ndarray:
    """
    Convert pixel coordinates + depth to 3D point.
    
    Args:
        u: Pixel x coordinate
        v: Pixel y coordinate  
        depth: Depth value at pixel
        intrinsics: 3x3 camera matrix
        
    Returns:
        3D point [x, y, z] in same units as depth
    """
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.array([x, y, z], dtype=np.float32)


# Quick test
if __name__ == "__main__":
    print("LingBot-Depth Module")
    print("=" * 50)
    
    # Attempt lazy import
    if not _import_lingbot_model():
        print("LingBot-Depth not available!")
        sys.exit(1)
    
    # Check for weights
    weights_path = LingBotDepth.DEFAULT_WEIGHTS_PATH
    if weights_path.exists():
        print(f"Weights found: {weights_path}")
        print(f"Size: {weights_path.stat().st_size / 1e9:.2f} GB")
    else:
        print(f"Weights not found at: {weights_path}")
        print("Please download weights or specify a different path.")
        sys.exit(1)
    
    # Try to load model
    try:
        model = LingBotDepth()
        print("\nModel loaded successfully!")
        
        # Test with dummy data
        h, w = 480, 640
        rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        depth = np.random.rand(h, w).astype(np.float32) * 5000  # 0-5000mm
        intrinsics = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"\nTest input shapes:")
        print(f"  RGB: {rgb.shape}")
        print(f"  Depth: {depth.shape}")
        
        refined = model.refine(rgb, depth, intrinsics)
        print(f"  Output: {refined.shape}")
        print(f"  Depth range: {refined[refined > 0].min():.1f} - {refined.max():.1f} mm")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

