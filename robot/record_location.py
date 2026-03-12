#!/usr/bin/env python3
"""
Record the current robot pose as a named location (joint + base cartesian) into locations/.

Supports multi-arm setups via --arm left|right|both.

Usage:
    python record_location.py my_pose
    python record_location.py --name pick_above --arm left
    python record_location.py my_pose --arm both
    python record_location.py my_pose --ip 192.168.1.195
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _record_single_arm(a, arm_label):
    """Put arm in manual mode, wait for user, read pose + joints, return dict."""
    print(f"[{arm_label}] Putting robot in MANUAL (teaching) mode.")
    a.set_manual_mode()
    try:
        input(f"[{arm_label}] Move the robot to the desired pose, then press Enter... ")
    except KeyboardInterrupt:
        print("\nCancelled.")
        a.set_position_mode()
        return None

    code, pose = a.get_position()
    if code != 0:
        print(f"[{arm_label}] Failed to read current position.")
        a.set_position_mode()
        return None

    code_j, joints = a.get_joint_angles()

    data = {
        "pose": [float(x) for x in pose],
        "position_mm": [float(pose[0]), float(pose[1]), float(pose[2])],
        "orientation_deg": [float(pose[3]), float(pose[4]), float(pose[5])],
    }
    if code_j == 0 and joints:
        data["joint_angles_deg"] = [float(j) for j in joints]

    print(f"[{arm_label}] Restoring position control mode.")
    a.set_position_mode()
    return data


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
                        help="Robot IP (default: from robot_mapping.json or handeye_calibration_data.json)")
    parser.add_argument("--arm", type=str, default=None, choices=["left", "right", "both"],
                        help="Which arm to record: left, right, or both (default: default arm)")
    args = parser.parse_args()

    location_name = (args.name or args.location or "").strip()
    if not location_name:
        parser.error("Location name required (e.g. record_location.py my_pose or --name my_pose)")

    try:
        from aira.robot import arm, get_arm_names
        from aira.utils.paths import get_project_root
    except ImportError as e:
        print(f"Error: {e}")
        return 1

    locations_dir = get_project_root() / "locations"
    locations_dir.mkdir(parents=True, exist_ok=True)
    out_path = locations_dir / f"{location_name}.json"

    arm_choice = args.arm

    if arm_choice == "both":
        arm_names = get_arm_names()
        if len(arm_names) < 2:
            print(f"Error: --arm both requires at least 2 arms in robot_mapping.json (found: {arm_names})")
            return 1

        result = {"arm": "both"}
        for arm_name in arm_names:
            print(f"\nConnecting to arm '{arm_name}'...")
            try:
                a = arm(name=arm_name, ip=args.ip)
            except Exception as e:
                print(f"Connection to '{arm_name}' failed: {e}")
                return 1

            data = _record_single_arm(a, arm_name)
            if data is None:
                return 1
            result[arm_name] = data

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved bimanual location '{location_name}' to {out_path}")
        for arm_name in arm_names:
            sub = result[arm_name]
            print(f"  [{arm_name}] position_mm: {sub['position_mm']}")
        return 0

    print("Connecting to robot...")
    try:
        a = arm(name=arm_choice, ip=args.ip)
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1

    data = _record_single_arm(a, arm_choice or "default")
    if data is None:
        return 1

    if arm_choice:
        data["arm"] = arm_choice

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved location '{location_name}' to {out_path}")
    print(f"  position_mm: {data['position_mm']}")
    print(f"  orientation_deg: {data['orientation_deg']}")
    if "joint_angles_deg" in data:
        print(f"  joint_angles_deg: {data['joint_angles_deg']}")
    if arm_choice:
        print(f"  arm: {arm_choice}")
    print("Use: arm().go_to('" + location_name + "')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
