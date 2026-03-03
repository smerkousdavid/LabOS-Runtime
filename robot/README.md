# Steps for Using Robot

## Setup
Please ensure you are using a UFactory XArm 6 with the Gripper 2 and an Intel RealSense D455. These scripts use the official xArm-Python-SDK to move the robot arm.

### Prerequisites
```bash
pip install pyrealsense2 opencv-python numpy
```

---

## Camera Calibration

There are two types of calibration for the RealSense D455:

| Calibration Type | Purpose | When to Run |
|------------------|---------|-------------|
| **Depth Calibration** | Improves depth sensor accuracy using on-device firmware | Once, or when depth seems inaccurate |
| **Intrinsics Calibration** | Computes camera matrix for ArUco/pose estimation | Once per camera resolution |

---

## 0. Depth Calibration (On-Device Self-Calibration)

This uses Intel's built-in self-calibration to improve depth accuracy. No special targets needed.

### Quick Check
```bash
python depth_calibration.py --preview
```
This shows live depth with noise statistics. Good calibration has RMS noise < 2%.

### Run On-Chip Calibration
```bash
python depth_calibration.py --on-chip
```

**Instructions:**
1. Point the camera at a flat, textured surface (NOT a plain white wall)
2. Keep the camera 0.5-2.0 meters from the surface
3. Hold steady during calibration (30-60 seconds)
4. Review the health score (< 0.25 is good)
5. Choose to save the calibration to the device

### Run Tare Calibration (Optional)
Use this if absolute distance accuracy is critical:
```bash
python depth_calibration.py --tare 500
```
Replace `500` with the exact distance to your target in millimeters.

### Interactive Mode
```bash
python depth_calibration.py
```
Provides a menu-driven interface for all calibration options.

### Reset to Factory
```bash
python depth_calibration.py --reset
```

---

## 1. Intrinsics Calibration (OpenCV Camera Calibration)

This computes the camera intrinsics matrix for ArUco marker detection and pose estimation.

### Prerequisite
Print a checkerboard calibration pattern:
- Default: 7x9 inner corners (8x10 squares)
- Recommended square size: 24mm
- Print on matte paper (avoid glossy/reflective surfaces)

### PART 1: Capture Calibration Images

```bash
python capture_calibration.py
```

**Controls:**
- `SPACE` - Capture image (when checkerboard is detected)
- `D` - Toggle depth overlay
- `C` - Toggle checkerboard detection preview
- `Q/ESC` - Quit

**Tips for good calibration:**
1. Capture 15-25 images with the checkerboard at various positions
2. Include images at the center AND corners of the frame
3. Vary the angle: tilt left/right, top/bottom, rotate
4. Vary the distance: near (fills 80% of frame) and far (fills 30%)
5. Ensure the checkerboard is fully visible in each image
6. The green "Checkerboard: FOUND" indicator confirms detection

### PART 2: Calculate Calibration

```bash
python calculate_calibration.py
```

This processes the images and outputs:
- `calibration_matrix.npy` - 3x3 camera intrinsics matrix
- `distortion_coefficients.npy` - Lens distortion parameters
- `intrinsics.txt` - Human-readable summary

**Quality Guidelines (RMS reprojection error):**
- < 0.5 pixels: Excellent
- 0.5-1.0 pixels: Good
- 1.0-2.0 pixels: Acceptable
- \> 2.0 pixels: Poor - recapture images

### Custom Options

```bash
# Custom checkerboard size (9x6 inner corners)
python capture_calibration.py --checkerboard 9 6

# Higher resolution
python capture_calibration.py --width 1920 --height 1080

# Custom square size (25mm)
python calculate_calibration.py --square-size 0.025

# Visualize detection during calculation
python calculate_calibration.py --visualize
```

---

## 2. Hand-Eye Calibration

After camera intrinsics are calibrated, perform hand-eye calibration to relate camera coordinates to robot tool coordinates. (See main project documentation)

---

## File Structure

```
version2/
├── README.md                    # This file
├── depth_calibration.py         # On-device depth calibration
├── capture_calibration.py       # Capture images for intrinsics calibration
├── calculate_calibration.py     # Compute intrinsics from images
└── calibration_images/          # Default folder for captured images
    ├── calib_*.png              # Color images
    ├── depth_*.png              # Depth images (optional)
    ├── calibration_matrix.npy   # Output: camera matrix
    ├── distortion_coefficients.npy  # Output: distortion
    └── intrinsics.txt           # Output: human-readable summary
```

---

## Troubleshooting

### Depth calibration fails
- Ensure USB 3.0 connection (check with `--preview`)
- Point at a textured surface (newspaper, carpet, etc.)
- Avoid reflective or transparent surfaces
- Hold camera completely still

### Checkerboard not detected
- Improve lighting (avoid shadows on the board)
- Ensure the entire board is visible
- Board might be too close or too far
- Try printing on matte paper

### High reprojection error
- Capture more images (15-25 recommended)
- Ensure varied positions and angles
- Remove blurry images from the calibration folder
- Check that the checkerboard is flat

### "This device does not support auto-calibration"
- Ensure you have the latest firmware
- D455 supports self-calibration; D415 may not
- Try updating via RealSense Viewer

---

## Robot Runtime

The **robot-runtime** connects to the NAT server via WebSocket and exposes all xArm tools as callable functions for the LLM agent.

### Standalone (on the robot machine)

```bash
cd robot/
pip install -r requirements.txt
./run.sh
# or with explicit args:
python robot_runtime.py --nat-url ws://labos-nat:8002/ws --xarm-ip 192.168.1.185
```

### Docker (in the LabOS compose stack)

1. Enable in `config/config.yaml`:
   ```yaml
   robot:
     enabled: true
     xarm_ip: "192.168.1.185"
   ```
2. Run `./run.sh` -- the robot-runtime container starts automatically.
3. For RealSense USB passthrough, uncomment the `devices:` block in `compose/runtime.j2`.

### WebSocket Protocol

The robot-runtime uses three message types on the existing NAT WebSocket connection:

| Direction | Type | Description |
|-----------|------|-------------|
| robot -> NAT | `robot_register` | Sent on connect with `tools` array (name, description, parameters) |
| NAT -> robot | `robot_execute` | Tool invocation: `{request_id, tool_name, arguments}` |
| robot -> NAT | `robot_result` | Execution result: `{request_id, tool_name, success, result}` |

### Available Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_status` | -- | Current protocol status (idle / running step) |
| `start_protocol` | `protocol_name` | Start a protocol by name |
| `get_protocols` | -- | List available protocols |
| `describe_protocol` | `protocol_name` | Describe a protocol's steps |
| `stop_robot` | -- | Stop protocol, return to position control |
| `get_object_definitions` | -- | List configured objects from `configs/objects.yaml` |
| `list_objects` | -- | List objects visible in camera with positions and colors |
| `move_to_object` | `object_name`, `target_px_x?`, `target_px_y?` | Move to a detected object |
| `gripper` | `position` | Control gripper: close/midway/open/0-800 |
| `z_level` | `level` | Move to height: low/medium/high/mm number |
| `is_holding_something` | -- | True if gripper is closed |
| `go_home` | -- | Send robot to home position |
| `see_object` | `object_name` | True if object is visible in camera |
| `manual_mode` | -- | Enable manual (teaching) mode |

### NAT Server Integration

The NAT agent should:
1. Listen for `robot_register` messages on the WebSocket
2. Parse the `tools` array to discover available robot capabilities
3. When the LLM decides to use a robot tool, send `robot_execute` with a unique `request_id`
4. Wait for the matching `robot_result` with the same `request_id`
5. Use `success` to determine if the call succeeded and `result` for the text response
