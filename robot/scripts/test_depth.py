#!/usr/bin/env python3
# IMPORTANT: Disable xformers BEFORE any other imports to avoid triton issues on Windows
import os
os.environ["XFORMERS_DISABLED"] = "1"

"""
Interactive Depth Testing Tool

Live RealSense streaming with depth refinement, cursor depth display,
and 3D distance measurement between clicked points.

Features:
    - Live color + depth preview (side-by-side)
    - Real-time depth at cursor position (mm)
    - Click two points to measure 3D distance
    - Toggle between raw and LingBot-refined depth
    - Visual feedback: crosshairs, measurement lines, distance text

Controls:
    Mouse Move  - Show depth at cursor
    Left Click  - Add measurement point (2 points for distance)
    R           - Reset measurement points
    T           - Toggle raw/refined depth mode
    C           - Cycle colormaps
    S           - Save current frame
    Q/ESC       - Quit

Usage:
    python test_depth.py                    # Default settings
    python test_depth.py --no-refine        # Raw depth only (no model)
    python test_depth.py --width 1280       # Custom resolution
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np
import cv2

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    print("Error: pyrealsense2 not installed. Install with: pip install pyrealsense2")
    REALSENSE_AVAILABLE = False

# Import depth module utilities (these don't require the model)
# Use fallback implementations to avoid xformers/triton import issues
DEPTH_MODULE_AVAILABLE = False
LingBotDepth = None

def depth_to_colormap(depth, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO, invalid_color=(0, 0, 0)):
    """Convert depth map to colorized visualization."""
    valid_mask = np.isfinite(depth) & (depth > 0)
    depth_clean = depth.copy()
    depth_clean[~valid_mask] = 0
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1
    depth_normalized = np.clip((depth_clean - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    depth_colored[~valid_mask] = invalid_color
    return depth_colored

def realsense_intrinsics_to_matrix(intrinsics):
    """Convert RealSense intrinsics to 3x3 matrix."""
    return np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

def pixel_to_3d(u, v, depth, intrinsics):
    """Convert pixel + depth to 3D point."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)

def compute_3d_distance(p1, p2):
    """Compute 3D Euclidean distance."""
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))

def _try_import_lingbot():
    """Attempt to import LingBot-Depth model (lazy import to avoid xformers issues)."""
    global DEPTH_MODULE_AVAILABLE, LingBotDepth
    if LingBotDepth is not None:
        return True
    try:
        from aira.vision.depth import LingBotDepth as _LingBotDepth
        LingBotDepth = _LingBotDepth
        DEPTH_MODULE_AVAILABLE = True
        return True
    except Exception as e:
        print(f"Note: LingBot-Depth model not available: {e}")
        print("  Running with raw depth only. Use --no-refine to suppress this message.")
        DEPTH_MODULE_AVAILABLE = False
        return False


# Colormap options
COLORMAPS = [
    ("TURBO", cv2.COLORMAP_TURBO),
    ("JET", cv2.COLORMAP_JET),
    ("VIRIDIS", cv2.COLORMAP_VIRIDIS),
    ("INFERNO", cv2.COLORMAP_INFERNO),
    ("MAGMA", cv2.COLORMAP_MAGMA),
    ("PLASMA", cv2.COLORMAP_PLASMA),
    ("HOT", cv2.COLORMAP_HOT),
    ("BONE", cv2.COLORMAP_BONE),
]


class DepthViewer:
    """Interactive depth viewer with measurement capabilities."""
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_refine: bool = True,
        model_path: str = None,
        save_dir: str = "depth_captures"
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_refine = use_refine  # Will be validated in start()
        self.model_path = model_path
        self.save_dir = Path(save_dir)
        
        # RealSense
        self.pipeline = None
        self.profile = None
        self.align = None
        self.intrinsics = None
        self.intrinsics_matrix = None
        
        # Depth refinement model
        self.depth_refiner = None
        
        # UI state
        self.cursor_pos = [None, None]  # [x, y]
        self.measurement_points = []    # List of (display_x, display_y, depth_x, depth_y, depth_mm, point_3d)
        self.colormap_idx = 0
        self.show_refined = True
        self.running = True
        
        # Frame data
        self.current_rgb = None
        self.current_depth_raw = None   # Raw RealSense depth (mm)
        self.current_depth_refined = None  # Refined depth (mm)
        
    def start(self) -> bool:
        """Initialize RealSense and depth model."""
        if not REALSENSE_AVAILABLE:
            print("RealSense not available!")
            return False
        
        # Initialize RealSense
        print("Initializing RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Try requested resolution with fallbacks
        resolutions = [
            (self.width, self.height),
            (1280, 720),
            (848, 480),
            (640, 480),
        ]
        
        started = False
        for w, h in resolutions:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self.fps)
                config.enable_stream(rs.stream.depth, w, h, rs.format.z16, self.fps)
                
                print(f"  Trying {w}x{h}...")
                self.profile = self.pipeline.start(config)
                self.width, self.height = w, h
                started = True
                break
            except RuntimeError:
                continue
        
        if not started:
            print("Failed to start RealSense with any resolution!")
            return False
        
        # Create alignment object
        self.align = rs.align(rs.stream.color)
        
        # Get intrinsics
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.intrinsics = color_profile.get_intrinsics()
        self.intrinsics_matrix = realsense_intrinsics_to_matrix(self.intrinsics)
        
        print(f"  Camera started: {self.width}x{self.height}")
        print(f"  Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
        
        # Warm up
        print("  Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        # Initialize depth refinement model (lazy import to avoid xformers issues)
        if self.use_refine:
            if _try_import_lingbot():
                try:
                    print("\nInitializing LingBot-Depth model...")
                    self.depth_refiner = LingBotDepth(
                        model_path=self.model_path,
                        device="auto",
                        use_fp16=True,
                        resolution_level=7  # Balance between quality and speed
                    )
                except Exception as e:
                    print(f"  Warning: Could not load depth model: {e}")
                    print("  Falling back to raw depth only")
                    self.depth_refiner = None
                    self.use_refine = False
            else:
                print("  LingBot-Depth not available, using raw depth")
                self.depth_refiner = None
                self.use_refine = False
        
        # Create save directory
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        return True
    
    def stop(self):
        """Stop RealSense pipeline."""
        if self.pipeline and self.profile:
            try:
                self.pipeline.stop()
                print("Camera stopped.")
            except:
                pass
    
    def get_frames(self) -> bool:
        """Get aligned color and depth frames."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return False
            
            # Convert to numpy
            self.current_rgb = np.asanyarray(color_frame.get_data())
            self.current_depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            
            return True
        except Exception as e:
            print(f"Frame error: {e}")
            return False
    
    def refine_depth(self):
        """Run depth refinement on current frame."""
        if self.depth_refiner is None or self.current_rgb is None or self.current_depth_raw is None:
            self.current_depth_refined = self.current_depth_raw
            return
        
        try:
            # Convert BGR to RGB for model
            rgb = cv2.cvtColor(self.current_rgb, cv2.COLOR_BGR2RGB)
            
            # Refine depth
            self.current_depth_refined = self.depth_refiner.refine(
                rgb, 
                self.current_depth_raw, 
                self.intrinsics_matrix
            )
        except Exception as e:
            print(f"Refinement error: {e}")
            self.current_depth_refined = self.current_depth_raw
    
    def get_depth_at_pixel(self, x: int, y: int) -> Tuple[float, float]:
        """Get raw and refined depth at pixel coordinates (in mm)."""
        raw_depth = 0.0
        refined_depth = 0.0
        
        if self.current_depth_raw is not None:
            h, w = self.current_depth_raw.shape
            if 0 <= x < w and 0 <= y < h:
                raw_depth = self.current_depth_raw[y, x]
        
        if self.current_depth_refined is not None:
            h, w = self.current_depth_refined.shape
            if 0 <= x < w and 0 <= y < h:
                refined_depth = self.current_depth_refined[y, x]
        
        return raw_depth, refined_depth
    
    def add_measurement_point(self, display_x: int, display_y: int):
        """Add a measurement point at display coordinates."""
        # Map display coordinates to depth coordinates
        # In side-by-side mode, we need to handle left (color) vs right (depth) side
        
        # For now, use the same coordinates (assuming depth is aligned)
        depth_x = display_x % self.width  # Handle side-by-side
        depth_y = display_y
        
        # Clamp to valid range
        depth_x = max(0, min(self.width - 1, depth_x))
        depth_y = max(0, min(self.height - 1, depth_y))
        
        # Get depth
        raw_depth, refined_depth = self.get_depth_at_pixel(depth_x, depth_y)
        depth_mm = refined_depth if (self.show_refined and refined_depth > 0) else raw_depth
        
        if depth_mm <= 0:
            print("  Invalid depth at this point. Try another location.")
            return
        
        # Compute 3D point
        point_3d = pixel_to_3d(depth_x, depth_y, depth_mm, self.intrinsics_matrix)
        
        # Store measurement point
        self.measurement_points.append({
            'display_x': display_x,
            'display_y': display_y,
            'depth_x': depth_x,
            'depth_y': depth_y,
            'depth_mm': depth_mm,
            'point_3d': point_3d
        })
        
        print(f"  Point {len(self.measurement_points)}: ({depth_x}, {depth_y}) -> "
              f"Depth: {depth_mm:.1f}mm, 3D: [{point_3d[0]:.1f}, {point_3d[1]:.1f}, {point_3d[2]:.1f}]mm")
        
        if len(self.measurement_points) == 2:
            # Compute distance
            p1 = self.measurement_points[0]['point_3d']
            p2 = self.measurement_points[1]['point_3d']
            dist = compute_3d_distance(p1, p2)
            print(f"  3D Distance: {dist:.1f}mm ({dist/25.4:.2f} inches)")
        elif len(self.measurement_points) > 2:
            # Reset and start new measurement
            self.measurement_points = [self.measurement_points[-1]]
            print("  Measurement reset. New point 1 set.")
    
    def reset_measurement(self):
        """Reset measurement points."""
        self.measurement_points = []
        print("  Measurement points cleared.")
    
    def save_frame(self):
        """Save current frame to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save RGB
        rgb_path = self.save_dir / f"rgb_{timestamp}.png"
        cv2.imwrite(str(rgb_path), self.current_rgb)
        
        # Save raw depth
        raw_path = self.save_dir / f"depth_raw_{timestamp}.npy"
        np.save(str(raw_path), self.current_depth_raw)
        
        # Save refined depth
        if self.current_depth_refined is not None:
            refined_path = self.save_dir / f"depth_refined_{timestamp}.npy"
            np.save(str(refined_path), self.current_depth_refined)
        
        # Save visualization
        vis_path = self.save_dir / f"vis_{timestamp}.png"
        # (would save current display)
        
        print(f"  Saved to {self.save_dir}/")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pos = [x, y]
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.add_measurement_point(x, y)
    
    def draw_ui(self, display: np.ndarray):
        """Draw UI elements on the display."""
        h, w = display.shape[:2]
        
        # Get current depth mode
        mode_text = "REFINED" if self.show_refined and self.depth_refiner else "RAW"
        colormap_name = COLORMAPS[self.colormap_idx][0]
        
        # Draw cursor depth
        if self.cursor_pos[0] is not None:
            cx, cy = self.cursor_pos
            
            # Get depth at cursor
            depth_x = cx % self.width
            depth_y = cy
            
            if 0 <= depth_x < self.width and 0 <= depth_y < self.height:
                raw_depth, refined_depth = self.get_depth_at_pixel(depth_x, depth_y)
                
                # Draw crosshairs
                cv2.line(display, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
                cv2.line(display, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)
                
                # Depth text
                if self.show_refined and refined_depth > 0:
                    depth_text = f"Depth: {refined_depth:.0f}mm ({refined_depth/25.4:.1f}in)"
                elif raw_depth > 0:
                    depth_text = f"Depth: {raw_depth:.0f}mm ({raw_depth/25.4:.1f}in)"
                else:
                    depth_text = "Depth: N/A"
                
                # Draw depth value near cursor
                text_x = min(cx + 20, w - 200)
                text_y = max(cy - 10, 20)
                cv2.putText(display, depth_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw measurement points and line
        for i, point in enumerate(self.measurement_points):
            px, py = point['display_x'], point['display_y']
            
            # Draw point marker
            cv2.circle(display, (px, py), 8, (0, 255, 0), -1)  # Filled green
            cv2.circle(display, (px, py), 10, (255, 255, 255), 2)  # White outline
            
            # Draw point number
            cv2.putText(display, str(i + 1), (px - 4, py + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Draw point info
            depth_mm = point['depth_mm']
            info_text = f"P{i+1}: {depth_mm:.0f}mm"
            cv2.putText(display, info_text, (px + 15, py + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw line between points
        if len(self.measurement_points) >= 2:
            p1 = self.measurement_points[0]
            p2 = self.measurement_points[1]
            
            # Draw line
            cv2.line(display, 
                    (p1['display_x'], p1['display_y']),
                    (p2['display_x'], p2['display_y']),
                    (0, 255, 255), 2)
            
            # Compute and draw distance
            dist = compute_3d_distance(p1['point_3d'], p2['point_3d'])
            
            # Draw distance at midpoint
            mid_x = (p1['display_x'] + p2['display_x']) // 2
            mid_y = (p1['display_y'] + p2['display_y']) // 2
            
            dist_text = f"{dist:.0f}mm ({dist/25.4:.1f}in)"
            
            # Background box for better visibility
            (text_w, text_h), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, 
                         (mid_x - 5, mid_y - text_h - 5),
                         (mid_x + text_w + 5, mid_y + 5),
                         (0, 0, 0), -1)
            cv2.putText(display, dist_text, (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw status bar
        status_text = f"Mode: {mode_text} | Colormap: {colormap_name} | [R]eset [T]oggle [C]olor [S]ave [Q]uit"
        cv2.putText(display, status_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display
    
    def run(self):
        """Main viewer loop."""
        print("\n" + "="*60)
        print("INTERACTIVE DEPTH VIEWER")
        print("="*60)
        print("Controls:")
        print("  Mouse Move  - Show depth at cursor")
        print("  Left Click  - Add measurement point")
        print("  R           - Reset measurement")
        print("  T           - Toggle raw/refined depth")
        print("  C           - Cycle colormaps")
        print("  S           - Save current frame")
        print("  Q/ESC       - Quit")
        print("="*60 + "\n")
        
        cv2.namedWindow('Depth Viewer', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Depth Viewer', self.mouse_callback)
        
        frame_count = 0
        fps_start = time.time()
        fps = 0.0
        
        refine_interval = 5  # Refine every N frames (for performance)
        
        try:
            while self.running:
                # Get frames
                if not self.get_frames():
                    continue
                
                # Refine depth periodically
                if self.depth_refiner and frame_count % refine_interval == 0:
                    self.refine_depth()
                elif self.current_depth_refined is None:
                    self.current_depth_refined = self.current_depth_raw
                
                # Select which depth to display
                if self.show_refined and self.current_depth_refined is not None:
                    depth_display = self.current_depth_refined
                else:
                    depth_display = self.current_depth_raw
                
                # Colorize depth
                colormap = COLORMAPS[self.colormap_idx][1]
                depth_colored = depth_to_colormap(depth_display, colormap=colormap)
                
                # Create side-by-side display
                display = np.hstack([self.current_rgb, depth_colored])
                
                # Draw UI
                display = self.draw_ui(display)
                
                # Draw FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    fps_start = time.time()
                
                cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show
                cv2.imshow('Depth Viewer', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    self.running = False
                elif key == ord('r') or key == ord('R'):
                    self.reset_measurement()
                elif key == ord('t') or key == ord('T'):
                    self.show_refined = not self.show_refined
                    mode = "REFINED" if self.show_refined else "RAW"
                    print(f"  Depth mode: {mode}")
                elif key == ord('c') or key == ord('C'):
                    self.colormap_idx = (self.colormap_idx + 1) % len(COLORMAPS)
                    print(f"  Colormap: {COLORMAPS[self.colormap_idx][0]}")
                elif key == ord('s') or key == ord('S'):
                    self.save_frame()
        
        finally:
            cv2.destroyAllWindows()
        
        print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Depth Viewer with 3D Measurement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Mouse Move  - Show depth at cursor position
  Left Click  - Add measurement point (2 points = 3D distance)
  R           - Reset measurement points
  T           - Toggle raw/refined depth mode
  C           - Cycle through colormaps
  S           - Save current frame
  Q/ESC       - Quit

Examples:
  python test_depth.py                    # Default settings with depth refinement
  python test_depth.py --no-refine        # Raw RealSense depth only
  python test_depth.py --width 1920 --height 1080  # Higher resolution
        """
    )
    
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frame rate (default: 30)')
    parser.add_argument('--no-refine', action='store_true',
                       help='Disable depth refinement (raw depth only)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights (default: weights/lingbot.pt)')
    parser.add_argument('--save-dir', type=str, default='depth_captures',
                       help='Directory to save captured frames (default: depth_captures)')
    
    args = parser.parse_args()
    
    if not REALSENSE_AVAILABLE:
        print("Error: pyrealsense2 not available!")
        return 1
    
    # Create viewer
    viewer = DepthViewer(
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_refine=not args.no_refine,
        model_path=args.model,
        save_dir=args.save_dir
    )
    
    try:
        # Start
        if not viewer.start():
            return 1
        
        # Run
        viewer.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        viewer.stop()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
