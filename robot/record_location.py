#!/usr/bin/env python3
"""
Record the current robot pose as a named location (joint + base cartesian) into version2/locations/.

Puts the robot into manual (teaching) mode so you can move it by hand to the desired pose,
then saves joint angles and cartesian pose to locations/<name>.json. Use arm().go_to('name')
to move the robot to that location later.

Usage:
    python record_location.py my_pose
    python record_location.py --name pick_above
    python record_location.py my_pose --ip 192.168.1.195
"""

import json
import sys
from pathlib import Path

# Ensure version2 is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Record current robot pose to locations/<name>.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "location",
        nargs="?",
        type=str,
        default=None,
        help="Location name (saved as locations/<name>.json)",
    )
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Location name (alternative to positional)")
    parser.add_argument("--ip", type=str, default=None,
                        help="Robot IP (default: from handeye_calibration_data.json)")
    args = parser.parse_args()

    location_name = (args.name or args.location or "").strip()
    if not location_name:
        parser.error("Location name required (e.g. record_location.py my_pose or --name my_pose)")

    try:
        from aira.robot import arm
        from aira.utils.paths import get_project_root
    except ImportError as e:
        print(f"Error: {e}")
        return 1

    locations_dir = get_project_root() / "locations"
    locations_dir.mkdir(parents=True, exist_ok=True)
    out_path = locations_dir / f"{location_name}.json"

    print("Connecting to robot...")
    try:
        a = arm(ip=args.ip)
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1

    print("Putting robot in MANUAL (teaching) mode. You can now move it by hand.")
    a.set_manual_mode()
    try:
        input("Move the robot to the desired pose, then press Enter to save... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        a.set_position_mode()
        return 0

    code, pose = a.get_position()
    if code != 0:
        print("Failed to read current position.")
        a.set_position_mode()
        return 1

    code_j, joints = a.get_joint_angles()
    if code_j != 0:
        print("Warning: could not read joint angles (cartesian only will be saved).")

    data = {
        "pose": [float(x) for x in pose],
        "position_mm": [float(pose[0]), float(pose[1]), float(pose[2])],
        "orientation_deg": [float(pose[3]), float(pose[4]), float(pose[5])],
    }
    if code_j == 0 and joints:
        data["joint_angles_deg"] = [float(j) for j in joints]

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved location '{location_name}' to {out_path}")
    print(f"  position_mm: {data['position_mm']}")
    print(f"  orientation_deg: {data['orientation_deg']}")
    if "joint_angles_deg" in data:
        print(f"  joint_angles_deg: {data['joint_angles_deg']}")
    print("Use: arm().go_to('" + location_name + "')")

    print("Restoring position control mode.")
    a.set_position_mode()
    return 0


if __name__ == "__main__":
    sys.exit(main())
