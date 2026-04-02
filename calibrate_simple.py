"""
Simple calibration: reads follower joint state and computes TCP position.
Run lerobot_teleoperate.py in another terminal to control the arm.
This script just reads the arm position and maps pixels to world coords.
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
from so101_kinematics import forward_kinematics_deg
from workspace_calibration import WorkspaceCalibration

CAMERA_INDEX   = 0
CAMERA_BACKEND = 700
FOLLOWER_PORT  = "COM7"
CALIB_DIR      = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower"
SAVE_PATH      = r"c:\DISK Z\Hakaton_HSE\cv_ik_pipeline\calibration.json"


def connect_follower():
    from pathlib import Path
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    cfg = SOFollowerRobotConfig(port=FOLLOWER_PORT, id="my_follower",
                                disable_torque_on_disconnect=True, cameras={},
                                calibration_dir=Path(CALIB_DIR))
    robot = SOFollower(cfg)
    robot.connect()
    # Disable torque so arm can be moved freely by hand
    try:
        robot.bus.disable_torque()
        print("Torque DISABLED — move arm freely by hand.")
    except Exception as e:
        print(f"Could not disable torque: {e}")
    return robot


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

def get_tcp(robot):
    obs = robot.get_observation()
    state = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
    pos, _ = forward_kinematics_deg(state)
    return pos


def main():
    print("=== Simple Workspace Calibration ===")
    print("Teleop should be running in another terminal.")
    print("1. Guide follower gripper on top of cube with leader")
    print("2. Press SPACE → capture TCP position")
    print("3. Click cube center in camera window")
    print("4. Repeat 4+ times at different cube positions")
    print("5. Press Q to save\n")

    print("Connecting follower (read-only)...")
    robot = connect_follower()
    print("Connected.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}")
        robot.disconnect()
        return

    cal = WorkspaceCalibration()
    pending_world = [None]
    current_tcp = [None]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and pending_world[0] is not None:
            world = pending_world[0]
            cal.add_point((x, y), world)
            print(f"  Saved #{len(cal.pixel_points)}: pixel=({x},{y}) "
                  f"world=({world[0]:.4f}, {world[1]:.4f}, {world[2]:.4f}) m")
            pending_world[0] = None

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    while True:
        # Read TCP position
        try:
            tcp = get_tcp(robot)
            current_tcp[0] = tcp
        except Exception:
            pass

        ret, frame = cap.read()
        if not ret:
            break

        # Draw saved points
        for px in cal.pixel_points:
            cv2.circle(frame, (int(px[0]), int(px[1])), 8, (0, 255, 0), -1)

        # Show TCP
        if current_tcp[0] is not None:
            t = current_tcp[0]
            cv2.putText(frame, f"TCP: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if pending_world[0] is not None:
            cv2.putText(frame, f"Pts:{len(cal.pixel_points)}  << CLICK ON CUBE >>",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Pts:{len(cal.pixel_points)}/4+  SPACE=capture  Q=done",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') and current_tcp[0] is not None:
            pending_world[0] = current_tcp[0].copy()
            t = pending_world[0]
            print(f"\nCaptured TCP: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")
            print("  Click on the cube in the camera window.")

    cap.release()
    cv2.destroyAllWindows()
    robot.disconnect()

    if len(cal.pixel_points) >= 3:
        print(f"\nFitting {len(cal.pixel_points)} points...")
        error = cal.compute()
        cal.save(SAVE_PATH)
        print(f"Saved! Error: {error:.4f} m")
    else:
        print(f"Need 3+ points (got {len(cal.pixel_points)}). Not saved.")


if __name__ == "__main__":
    main()
