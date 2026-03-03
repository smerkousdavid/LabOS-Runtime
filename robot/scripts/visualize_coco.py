#!/usr/bin/env python3
"""
COCO Dataset Visualization Tool

Visualizes COCO format dataset with bounding boxes, segmentation masks,
and category labels.

Usage:
    python visualize_coco.py annotations.json
    python visualize_coco.py annotations.json --dataset-dir dataset
    python visualize_coco.py annotations.json --show-masks --show-keypoints
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from collections import defaultdict

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class COCOVisualizer:
    """Visualize COCO format dataset."""
    
    # Color palette for categories (BGR format for OpenCV)
    COLORS = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 255),    # Light Blue
        (128, 255, 0),    # Lime
    ]
    
    def __init__(
        self,
        json_path: Path,
        dataset_dir: Path,
        show_masks: bool = False,
        show_keypoints: bool = False,
        mask_alpha: float = 0.5
    ):
        self.json_path = json_path
        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir / "images"
        self.show_masks = show_masks
        self.show_keypoints = show_keypoints
        self.mask_alpha = mask_alpha
        
        # Load COCO data
        self.data = self.load_coco_json()
        self.categories = {cat['id']: cat for cat in self.data['categories']}
        self.images = {img['id']: img for img in self.data['images']}
        self.annotations = defaultdict(list)
        
        for ann in self.data['annotations']:
            self.annotations[ann['image_id']].append(ann)
        
        # Create color map for categories
        self.category_colors = {}
        for i, cat_id in enumerate(sorted(self.categories.keys())):
            self.category_colors[cat_id] = self.COLORS[i % len(self.COLORS)]
        
        print(f"Loaded dataset:")
        print(f"  Images: {len(self.images)}")
        print(f"  Annotations: {len(self.data['annotations'])}")
        print(f"  Categories: {len(self.categories)}")
    
    def load_coco_json(self) -> Dict:
        """Load COCO format JSON file."""
        print(f"Loading: {self.json_path}")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Invalid COCO format: missing '{key}'")
        
        return data
    
    def draw_bbox(self, image: np.ndarray, bbox: List[float], color: Tuple[int, int, int], 
                   label: str = "", thickness: int = 2) -> np.ndarray:
        """Draw bounding box on image."""
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        if label:
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - text_h - 10), (x + text_w + 5, y), color, -1)
            cv2.putText(image, label, (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def draw_segmentation(self, image: np.ndarray, segmentation: List, 
                         color: Tuple[int, int, int]) -> np.ndarray:
        """Draw segmentation mask on image."""
        if not segmentation:
            return image
        
        # Create mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Handle different segmentation formats
        if isinstance(segmentation[0], list):
            # Polygon format: [[x1, y1, x2, y2, ...], ...]
            for seg in segmentation:
                if len(seg) >= 6:  # At least 3 points
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 255)
        else:
            # RLE or other format - skip for now
            return image
        
        # Apply mask with transparency
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        
        image = cv2.addWeighted(image, 1.0, mask_colored, self.mask_alpha, 0)
        
        return image
    
    def draw_keypoints(self, image: np.ndarray, keypoints: List[float], 
                      num_keypoints: int, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw keypoints on image."""
        if not keypoints or num_keypoints == 0:
            return image
        
        # COCO keypoints format: [x1, y1, v1, x2, y2, v2, ...]
        # v: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
        for i in range(0, len(keypoints), 3):
            if i + 2 >= len(keypoints):
                break
            
            x, y, v = int(keypoints[i]), int(keypoints[i + 1]), int(keypoints[i + 2])
            
            if v > 0:  # Keypoint is labeled
                # Draw keypoint
                cv2.circle(image, (x, y), 5, color, -1)
                cv2.circle(image, (x, y), 5, (255, 255, 255), 2)
        
        return image
    
    def visualize_image(self, image_id: int) -> Optional[np.ndarray]:
        """Visualize a single image with all annotations."""
        if image_id not in self.images:
            print(f"Image ID {image_id} not found")
            return None
        
        image_info = self.images[image_id]
        image_path = self.images_dir / image_info['file_name']
        
        if not image_path.exists():
            print(f"Image file not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Get annotations for this image
        anns = self.annotations.get(image_id, [])
        
        # Draw annotations
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = self.categories[cat_id]['name']
            color = self.category_colors[cat_id]
            
            # Draw segmentation mask
            if self.show_masks and 'segmentation' in ann and ann['segmentation']:
                image = self.draw_segmentation(image, ann['segmentation'], color)
            
            # Draw bounding box
            if 'bbox' in ann and ann['bbox']:
                label = f"{cat_name} (ID: {ann['id']})"
                image = self.draw_bbox(image, ann['bbox'], color, label)
            
            # Draw keypoints
            if self.show_keypoints and 'keypoints' in ann:
                num_kp = ann.get('num_keypoints', 0)
                if num_kp > 0:
                    image = self.draw_keypoints(image, ann['keypoints'], num_kp, color)
        
        # Add image info
        info_text = f"Image ID: {image_id} | Annotations: {len(anns)}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def run_interactive(self):
        """Run interactive visualization."""
        print("\n" + "="*60)
        print("COCO DATASET VISUALIZER")
        print("="*60)
        print("Controls:")
        print("  N / →  - Next image")
        print("  P / ←  - Previous image")
        print("  G      - Go to image ID")
        print("  M      - Toggle masks")
        print("  K      - Toggle keypoints")
        print("  I      - Show image info")
        print("  S      - Save current image")
        print("  Q/ESC  - Quit")
        print("="*60 + "\n")
        
        image_ids = sorted(self.images.keys())
        if not image_ids:
            print("No images found in dataset!")
            return
        
        current_idx = 0
        
        cv2.namedWindow('COCO Visualizer', cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                image_id = image_ids[current_idx]
                image = self.visualize_image(image_id)
                
                if image is None:
                    current_idx = (current_idx + 1) % len(image_ids)
                    continue
                
                # Display
                cv2.imshow('COCO Visualizer', image)
                
                # Handle keys
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('n') or key == 83:  # N or Right arrow
                    current_idx = (current_idx + 1) % len(image_ids)
                elif key == ord('p') or key == 81:  # P or Left arrow
                    current_idx = (current_idx - 1) % len(image_ids)
                elif key == ord('g'):  # G - Go to image ID
                    try:
                        target_id = int(input("Enter image ID: "))
                        if target_id in image_ids:
                            current_idx = image_ids.index(target_id)
                        else:
                            print(f"Image ID {target_id} not found")
                    except ValueError:
                        print("Invalid image ID")
                elif key == ord('m'):  # M - Toggle masks
                    self.show_masks = not self.show_masks
                    print(f"Masks: {'ON' if self.show_masks else 'OFF'}")
                elif key == ord('k'):  # K - Toggle keypoints
                    self.show_keypoints = not self.show_keypoints
                    print(f"Keypoints: {'ON' if self.show_keypoints else 'OFF'}")
                elif key == ord('i'):  # I - Show info
                    image_info = self.images[image_id]
                    anns = self.annotations.get(image_id, [])
                    print(f"\nImage ID: {image_id}")
                    print(f"  File: {image_info['file_name']}")
                    print(f"  Size: {image_info.get('width', '?')}x{image_info.get('height', '?')}")
                    print(f"  Annotations: {len(anns)}")
                    for ann in anns:
                        cat_name = self.categories[ann['category_id']]['name']
                        print(f"    - {cat_name} (ID: {ann['id']})")
                elif key == ord('s'):  # S - Save
                    output_path = Path(f"viz_{image_id}.png")
                    cv2.imwrite(str(output_path), image)
                    print(f"Saved: {output_path}")
        
        finally:
            cv2.destroyAllWindows()
    
    def export_all(self, output_dir: str = "visualizations"):
        """Export all visualizations to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\nExporting visualizations to: {output_path.absolute()}")
        
        image_ids = sorted(self.images.keys())
        for i, image_id in enumerate(image_ids, 1):
            image = self.visualize_image(image_id)
            if image is not None:
                image_info = self.images[image_id]
                filename = f"{image_id:06d}_{Path(image_info['file_name']).stem}.png"
                output_file = output_path / filename
                cv2.imwrite(str(output_file), image)
                
                if i % 10 == 0:
                    print(f"  Exported {i}/{len(image_ids)} images...")
        
        print(f"  ✓ Exported {len(image_ids)} images to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize COCO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (interactive mode):
  N / left   - Next image
  P / right  - Previous image
  G          - Go to image ID
  M          - Toggle segmentation masks
  K          - Toggle keypoints
  I          - Show image info
  S          - Save current image
  Q/ESC      - Quit

Examples:
  python visualize_coco.py annotations.json
  python visualize_coco.py annotations.json --show-masks
  python visualize_coco.py annotations.json --export-all
        """
    )
    
    parser.add_argument('json_file', type=str,
                       help='COCO annotations JSON file')
    parser.add_argument('--dataset-dir', type=str, default='dataset',
                       help='Dataset directory (default: dataset)')
    parser.add_argument('--show-masks', action='store_true',
                       help='Show segmentation masks')
    parser.add_argument('--show-keypoints', action='store_true',
                       help='Show keypoints')
    parser.add_argument('--mask-alpha', type=float, default=0.5,
                       help='Mask transparency (0.0-1.0, default: 0.5)')
    parser.add_argument('--export-all', action='store_true',
                       help='Export all visualizations to files instead of interactive mode')
    parser.add_argument('--export-dir', type=str, default='visualizations',
                       help='Export directory (default: visualizations)')
    
    args = parser.parse_args()
    
    # Validate paths
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1
    
    # Create visualizer
    try:
        viz = COCOVisualizer(
            json_path=json_path,
            dataset_dir=dataset_dir,
            show_masks=args.show_masks,
            show_keypoints=args.show_keypoints,
            mask_alpha=args.mask_alpha
        )
    except Exception as e:
        print(f"Error initializing visualizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run visualization
    try:
        if args.export_all:
            viz.export_all(args.export_dir)
        else:
            viz.run_interactive()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

