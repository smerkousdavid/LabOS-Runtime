#!/usr/bin/env python3
"""
RealSense Image Capture Script

Captures and saves color and depth images from RealSense camera.
Press SPACE to capture, Q/ESC to quit.

Usage:
    python capture.py                    # Default settings
    python capture.py --output my_images  # Custom output folder
    python capture.py --no-depth          # Color only
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    print("Error: pyrealsense2 not installed. Install with: pip install pyrealsense2")
    REALSENSE_AVAILABLE = False
    sys.exit(1)


class ImageCapture:
    """Simple RealSense image capture tool."""
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        output_dir: str = "capture",
        save_depth: bool = True
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.save_depth = save_depth
        self.capture_count = 0
        
        # RealSense
        self.pipeline = None
        self.profile = None
        self.align = None
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def start(self) -> bool:
        """Initialize RealSense camera."""
        if not REALSENSE_AVAILABLE:
            return False
        
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
                if self.save_depth:
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
        
        print(f"  Camera started: {self.width}x{self.height}")
        
        # Warm up
        print("  Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        return True
    
    def stop(self):
        """Stop RealSense pipeline."""
        if self.pipeline and self.profile:
            try:
                self.pipeline.stop()
                print("Camera stopped.")
            except:
                pass
    
    def get_frames(self):
        """Get aligned color and depth frames."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() if self.save_depth else None
            
            if not color_frame:
                return None, None
            
            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = None
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
        except Exception as e:
            print(f"Frame error: {e}")
            return None, None
    
    def save_image(self, color_image, depth_image=None):
        """Save captured images."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Save color image
        color_path = self.output_dir / f"color_{timestamp}.png"
        cv2.imwrite(str(color_path), color_image)
        print(f"  Saved: {color_path.name}")
        
        # Save depth image
        if depth_image is not None:
            depth_path = self.output_dir / f"depth_{timestamp}.png"
            # Save as 16-bit PNG for full depth range
            cv2.imwrite(str(depth_path), depth_image)
            print(f"  Saved: {depth_path.name}")
            
            # Also save as numpy array for full precision
            depth_npy_path = self.output_dir / f"depth_{timestamp}.npy"
            np.save(str(depth_npy_path), depth_image.astype(np.uint16))
            print(f"  Saved: {depth_npy_path.name}")
        
        self.capture_count += 1
        print(f"  Total captures: {self.capture_count}")
    
    def run(self):
        """Main capture loop."""
        print("\n" + "="*60)
        print("REALSENSE IMAGE CAPTURE")
        print("="*60)
        print("Controls:")
        print("  SPACE  - Capture image(s)")
        print("  Q/ESC  - Quit")
        print("="*60 + "\n")
        
        cv2.namedWindow('Capture', cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # Get frames
                color_image, depth_image = self.get_frames()
                if color_image is None:
                    continue
                
                # Display color image
                display = color_image.copy()
                
                # Add depth overlay if available
                if depth_image is not None:
                    # Create depth colormap for visualization
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    # Show side-by-side
                    display = np.hstack([color_image, depth_colormap])
                
                # Add instructions
                cv2.putText(display, "Press SPACE to capture, Q to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Captures: {self.capture_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Capture', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # SPACE
                    self.save_image(color_image, depth_image)
        
        finally:
            cv2.destroyAllWindows()
        
        print(f"\nCapture complete! Saved {self.capture_count} image(s) to {self.output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Capture RealSense images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  SPACE  - Capture color and depth images
  Q/ESC  - Quit

Examples:
  python capture.py                    # Default: save to capture/
  python capture.py --output my_images # Custom output folder
  python capture.py --no-depth         # Color images only
        """
    )
    
    parser.add_argument('--width', type=int, default=1280,
                       help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frame rate (default: 30)')
    parser.add_argument('--output', type=str, default='../capture',
                       help='Output directory (default: capture)')
    parser.add_argument('--no-depth', action='store_true',
                       help='Disable depth capture (color only)')
    
    args = parser.parse_args()
    
    if not REALSENSE_AVAILABLE:
        return 1
    
    # Create capture instance
    capture = ImageCapture(
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_dir=args.output,
        save_depth=not args.no_depth
    )
    
    try:
        # Start camera
        if not capture.start():
            return 1
        
        # Run capture loop
        capture.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        capture.stop()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

