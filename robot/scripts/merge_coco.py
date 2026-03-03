#!/usr/bin/env python3
"""
Merge COCO Datasets together

Takes multiple COCO format datasets and combines them into a single
unified COCO dataset. Each dataset has its own image source directory. Images
are read, converted to PNG, and saved with numbered filenames (1.png, 2.png, ...)
to avoid name collisions.

Usage:
    python merge_coco.py source1 source1_coco.json source2 source2_coco.json
    python merge_coco.py dir1 ann1.json dir2 ann2.json --output my_dataset
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

try:
    from PIL import Image
    from PIL import PngImagePlugin
    # Allow PNGs with large metadata chunks (e.g. zTXt from CVAT)
    PngImagePlugin.MAX_TEXT_CHUNK = 64 * 1024 * 1024  # 64 MB (handles PNGs with large metadata)
except ImportError:
    Image = None
    PngImagePlugin = None


class COCOMerger:
    """Merge multiple COCO format datasets with per-dataset image sources."""

    def __init__(
        self,
        output_dir: str = "dataset",
        image_subdir: str = "images"
    ):
        self.output_dir = Path(output_dir)
        self.image_subdir = self.output_dir / image_subdir

        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.image_subdir.mkdir(exist_ok=True, parents=True)

        # Trackers for merging (reset per-dataset in merge_dataset)
        self.image_id_map = {}  # old_id -> new_id (per dataset merge)
        self.category_id_map = {}  # old_id -> new_id
        self.annotation_id_counter = 1
        self.image_id_counter = 1
        self.category_id_counter = 1
        self.next_image_number = 1  # for output filenames 1.png, 2.png, ...

        # Unified dataset
        self.merged_categories = []
        self.merged_images = []
        self.merged_annotations = []

        # Track category names to avoid duplicates
        self.category_name_to_id = {}
    
    def load_coco_json(self, json_path: Path) -> Dict:
        """Load a COCO format JSON file."""
        print(f"Loading: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Validate COCO structure
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Invalid COCO format: missing '{key}' in {json_path}")
        
        print(f"  Images: {len(data['images'])}")
        print(f"  Annotations: {len(data['annotations'])}")
        print(f"  Categories: {len(data['categories'])}")
        
        return data
    
    def merge_categories(self, categories: List[Dict]) -> None:
        """Merge categories, avoiding duplicates by name."""
        for cat in categories:
            cat_name = cat['name']
            cat_supercategory = cat.get('supercategory', '')
            
            if cat_name not in self.category_name_to_id:
                # New category
                new_id = self.category_id_counter
                self.category_id_counter += 1
                
                new_category = {
                    'id': new_id,
                    'name': cat_name,
                    'supercategory': cat_supercategory
                }
                
                self.merged_categories.append(new_category)
                self.category_name_to_id[cat_name] = new_id
                self.category_id_map[cat['id']] = new_id
            else:
                # Category already exists, map old ID to existing new ID
                self.category_id_map[cat['id']] = self.category_name_to_id[cat_name]
    
    def find_image_file(self, image_filename: str, source_dir: Path) -> Optional[Path]:
        """Find image file in the given source directory."""
        # Try exact match first
        image_path = source_dir / image_filename
        if image_path.exists():
            return image_path

        # Try without extension variations
        base_name = Path(image_filename).stem
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            image_path = source_dir / f"{base_name}{ext}"
            if image_path.exists():
                return image_path

        # Try case-insensitive search
        if source_dir.exists():
            for file in source_dir.iterdir():
                if file.is_file() and file.stem.lower() == base_name.lower():
                    return file

        return None

    def save_image_as_numbered_png(self, source_path: Path) -> Tuple[str, int, int]:
        """Read image, convert to PNG, save as next number (e.g. 1.png). Returns (filename, width, height)."""
        if Image is None:
            raise RuntimeError("Pillow (PIL) is required. Install with: pip install Pillow")

        filename = f"{self.next_image_number}.png"
        self.next_image_number += 1
        dest_path = self.image_subdir / filename

        img = Image.open(source_path).convert("RGB")
        width, height = img.size
        img.save(dest_path, "PNG")

        return filename, width, height

    def merge_dataset(
        self,
        coco_data: Dict,
        image_source_dir: Path,
        dataset_name: str = ""
    ) -> Tuple[int, int]:
        """Merge a single COCO dataset into the unified dataset."""
        print(f"\nMerging dataset: {dataset_name} (images from {image_source_dir})")

        # Merge categories first
        self.merge_categories(coco_data['categories'])

        # Reset image_id_map for this dataset (maps this dataset's old ids -> new ids)
        self.image_id_map = {}

        # Process images
        images_added = 0
        images_missing = 0

        for image_info in coco_data['images']:
            old_image_id = image_info['id']
            image_filename = image_info.get('file_name', f"image_{old_image_id}.png")

            # Find image file in this dataset's source directory
            source_path = self.find_image_file(image_filename, image_source_dir)

            if source_path is None:
                print(f"  WARNING: Image not found: {image_filename}")
                images_missing += 1
                continue

            # Assign new image ID
            new_image_id = self.image_id_counter
            self.image_id_counter += 1
            self.image_id_map[old_image_id] = new_image_id

            # Read, convert, save as numbered PNG
            out_filename, width, height = self.save_image_as_numbered_png(source_path)

            # Create new image info
            new_image_info = {
                'id': new_image_id,
                'file_name': out_filename,
                'width': width,
                'height': height,
                'license': image_info.get('license', 0),
                'flickr_url': image_info.get('flickr_url', ''),
                'coco_url': image_info.get('coco_url', ''),
                'date_captured': image_info.get('date_captured', '')
            }

            self.merged_images.append(new_image_info)
            images_added += 1
        
        # Process annotations
        annotations_added = 0
        
        for ann in coco_data['annotations']:
            old_image_id = ann['image_id']
            
            # Skip if image wasn't found
            if old_image_id not in self.image_id_map:
                continue
            
            new_image_id = self.image_id_map[old_image_id]
            old_category_id = ann['category_id']
            new_category_id = self.category_id_map.get(old_category_id)
            
            if new_category_id is None:
                print(f"  WARNING: Category ID {old_category_id} not found in merged categories")
                continue
            
            # Create new annotation
            new_ann = {
                'id': self.annotation_id_counter,
                'image_id': new_image_id,
                'category_id': new_category_id,
                'segmentation': ann.get('segmentation', []),
                'area': ann.get('area', 0),
                'bbox': ann.get('bbox', []),
                'iscrowd': ann.get('iscrowd', 0)
            }
            
            # Copy additional fields if present
            for key in ['keypoints', 'num_keypoints', 'attributes']:
                if key in ann:
                    new_ann[key] = ann[key]
            
            self.merged_annotations.append(new_ann)
            self.annotation_id_counter += 1
            annotations_added += 1
        
        print(f"  Images added: {images_added}, missing: {images_missing}")
        print(f"  Annotations added: {annotations_added}")
        
        return images_added, annotations_added
    
    def save_merged_dataset(self, output_filename: str = "annotations.json") -> Path:
        """Save merged COCO dataset to JSON file."""
        output_path = self.output_dir / output_filename
        
        merged_data = {
            'info': {
                'description': 'Merged COCO dataset from CVAT',
                'version': '1.0',
                'year': 2024
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'Unknown',
                    'url': ''
                }
            ],
            'images': self.merged_images,
            'annotations': self.merged_annotations,
            'categories': self.merged_categories
        }
        
        print(f"\nSaving merged dataset to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"  Total images: {len(self.merged_images)}")
        print(f"  Total annotations: {len(self.merged_annotations)}")
        print(f"  Total categories: {len(self.merged_categories)}")
        
        return output_path
    
    def print_summary(self):
        """Print summary of merged dataset."""
        print("\n" + "="*60)
        print("MERGE SUMMARY")
        print("="*60)
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Images directory: {self.image_subdir.absolute()}")
        print(f"\nCategories ({len(self.merged_categories)}):")
        for cat in sorted(self.merged_categories, key=lambda x: x['id']):
            print(f"  [{cat['id']:3d}] {cat['name']}")
        print(f"\nTotal images: {len(self.merged_images)}")
        print(f"Total annotations: {len(self.merged_annotations)}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Merge COCO format datasets from CVAT (alternating image_dir, json_file)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_coco.py source1 source1_coco.json source2 source2_coco.json
  python merge_coco.py dir1 ann1.json dir2 ann2.json --output my_dataset
        """
    )

    parser.add_argument('pairs', nargs='+', type=str,
                        help='Alternating (image_source_dir, coco.json) pairs')
    parser.add_argument('--output', type=str, default='dataset',
                        help='Output directory (default: dataset)')
    parser.add_argument('--image-subdir', type=str, default='images',
                        help='Subdirectory for images in output (default: images)')
    parser.add_argument('--output-json', type=str, default='annotations.json',
                        help='Output JSON filename (default: annotations.json)')

    args = parser.parse_args()

    # Parse (image_dir, json_file) pairs
    if len(args.pairs) % 2 != 0:
        print("Error: Arguments must be alternating (image_source_dir, coco.json) pairs")
        print("Usage: python merge_coco.py source1 source1_coco.json source2 source2_coco.json")
        return 1

    pairs = []
    for i in range(0, len(args.pairs), 2):
        image_dir = Path(args.pairs[i])
        json_file = Path(args.pairs[i + 1])
        pairs.append((image_dir, json_file))

    # Validate
    for image_dir, json_path in pairs:
        if not image_dir.exists() or not image_dir.is_dir():
            print(f"Error: Image source directory not found: {image_dir}")
            return 1
        if not json_path.exists() or not json_path.is_file():
            print(f"Error: COCO JSON file not found: {json_path}")
            return 1

    print(f"Datasets to merge: {len(pairs)}")
    for i, (image_dir, json_path) in enumerate(pairs, 1):
        print(f"  {i}. {json_path.name} <- images from {image_dir}")

    # Create merger (no single capture_dir; each dataset has its own source)
    merger = COCOMerger(
        output_dir=args.output,
        image_subdir=args.image_subdir
    )

    # Merge each dataset with its image source
    total_images = 0
    total_annotations = 0

    for i, (image_source_dir, json_path) in enumerate(pairs, 1):
        try:
            coco_data = merger.load_coco_json(json_path)
            images_added, anns_added = merger.merge_dataset(
                coco_data,
                image_source_dir,
                f"Dataset {i} ({json_path.name})"
            )
            total_images += images_added
            total_annotations += anns_added
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Save merged dataset
    try:
        output_path = merger.save_merged_dataset(args.output_json)
        merger.print_summary()
        print(f"\n✓ Successfully merged datasets!")
        print(f"  Output: {output_path.absolute()}")
    except Exception as e:
        print(f"Error saving merged dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

