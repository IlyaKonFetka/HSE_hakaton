"""
Main pick-and-place pipeline for SO-101.

Pipeline:
1. Capture frame from front camera
2. Detect cube via color segmentation
3. Map pixel coords → world coords via calibration
4. Compute joint angles via inverse kinematics
5. Execute pick-and-place sequence:
   a. Move above cube
   b. Lower to grab height
   c. Close gripper
   d. Lift
   e. Move above plate
   f. Lower
   g. Open gripper
   h. Return home
"""

import time
import sys
import threading
import cv2
import numpy as np
from pathlib import Path

from so101_kinematics import forward_kinematics_deg, inverse_kinematics_deg
from ultralytics import YOLOWorld as _YW
_yolo_model = _YW("yolov8s-worldv2.pt")
_yolo_model.set_classes(["cube", "wooden block", "small box"])

def detect_object(frame, draw=False):
    res = _yolo_model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    if draw:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
    return {"cx": cx, "cy": cy}
from robot_controller import SO101Controller
from workspace_calibration import WorkspaceCalibration


# ── Configuration ────────────────────────────────────────────────────
FRONT_CAMERA_INDEX = 0
CAMERA_BACKEND = 700  # CAP_DSHOW for Windows

ROBOT_PORT = "COM7"
CALIBRATION_DIR = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower"

# Heights (meters, relative to base frame which is on the table)
GRAB_HEIGHT_OFFSET = -0.02     # how far below table surface to reach
ABOVE_HEIGHT_OFFSET = 0.06     # how high above table for safe moves
LIFT_HEIGHT_OFFSET = 0.10      # height when carrying object

# Plate position (approximate, in robot base frame meters)
# Adjust after calibration!
PLATE_POS_XY = (0.0790, 0.0019)
PLATE_Z      = -0.0348   # table level at plate center

# Gripper values
GRIPPER_OPEN = 80.0
GRIPPER_CLOSED = 15.0

# Movement timing
MOVE_DURATION = 1.5
GRAB_DURATION = 0.8


def capture_frame(cap):
    """Grab a frame from the camera."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    return frame


def pick_and_place(robot, target_world_xyz, plate_xyz):
    """
    Execute the full pick-and-place sequence.

    Parameters
    ----------
    robot : SO101Controller
    target_world_xyz : (x, y, z) in meters, position of the cube
    plate_xyz : (x, y, z) in meters, position of the plate
    """
    x, y, z = target_world_xyz
    px, py, pz = plate_xyz

    # Get current state for IK initial guess
    current = robot.get_state()
    current_joints = current[:5]

    print(f"  Target cube position: ({x:.3f}, {y:.3f}, {z:.3f})")
    print(f"  Target plate position: ({px:.3f}, {py:.3f}, {pz:.3f})")

    # Step 1: Move above cube
    print("  [1/8] Moving above cube...")
    above_cube = np.array([x, y, z + ABOVE_HEIGHT_OFFSET])
    angles, ok, err = inverse_kinematics_deg(above_cube, current_joints)
    print(f"    IK above cube: ok={ok} err={err:.4f}m  angles={np.round(angles,1)}")
    robot.move_smooth(angles, gripper=GRIPPER_OPEN, duration=MOVE_DURATION)

    # Step 2: Open gripper wide
    print("  [2/8] Opening gripper...")
    robot.open_gripper(GRIPPER_OPEN)
    time.sleep(0.3)

    # Step 3: Lower to grab
    print("  [3/8] Lowering to grab...")
    grab_pos = np.array([x, y, z + GRAB_HEIGHT_OFFSET])
    angles_grab, _, _ = inverse_kinematics_deg(grab_pos, angles)
    robot.move_smooth(angles_grab, gripper=GRIPPER_OPEN, duration=GRAB_DURATION)
    time.sleep(0.2)

    # Step 4: Close gripper
    print("  [4/8] Closing gripper...")
    robot.close_gripper(GRIPPER_CLOSED)
    time.sleep(0.5)

    # Step 5: Lift
    print("  [5/8] Lifting...")
    lift_pos = np.array([x, y, z + LIFT_HEIGHT_OFFSET])
    angles_lift, _, _ = inverse_kinematics_deg(lift_pos, angles_grab)
    robot.move_smooth(angles_lift, duration=MOVE_DURATION)

    # Step 6: Move above plate
    print("  [6/8] Moving above plate...")
    above_plate = np.array([px, py, pz + LIFT_HEIGHT_OFFSET])
    angles_plate, _, _ = inverse_kinematics_deg(above_plate, angles_lift)
    robot.move_smooth(angles_plate, duration=MOVE_DURATION)

    # Step 7: Lower to plate
    print("  [7/8] Lowering to plate...")
    on_plate = np.array([px, py, pz + 0.02])
    angles_place, _, _ = inverse_kinematics_deg(on_plate, angles_plate)
    robot.move_smooth(angles_place, duration=GRAB_DURATION)
    time.sleep(0.2)

    # Step 8: Release
    print("  [8/8] Releasing...")
    robot.open_gripper(GRIPPER_OPEN)
    time.sleep(0.3)

    # Return above plate
    robot.move_smooth(angles_plate, duration=MOVE_DURATION)

    print("  Pick-and-place complete!")


def run_pipeline():
    """Main pipeline loop."""
    print("=" * 50)
    print("SO-101 Pick-and-Place Pipeline")
    print("=" * 50)

    # Load calibration
    cal_path = Path(__file__).parent / "calibration.json"
    cal = WorkspaceCalibration()
    if cal_path.exists():
        cal.load(str(cal_path))
    else:
        print(f"\nNo calibration found at {cal_path}")
        print("Run workspace_calibration.py first, or use manual mode.")
        print("Starting manual mode...\n")
        return run_manual_mode()

    # Connect robot
    print("\nConnecting robot...")
    robot = SO101Controller(port=ROBOT_PORT, calibration_dir=CALIBRATION_DIR)
    robot.connect()

    # Open camera
    print("Opening camera...")
    cap = cv2.VideoCapture(FRONT_CAMERA_INDEX, CAMERA_BACKEND)
    if not cap.isOpened():
        print("Failed to open camera!")
        robot.disconnect()
        return

    print("\nReady! Press SPACE to detect and pick, 'q' to quit.\n")

    robot_busy = threading.Event()

    def run_pick(world_pos, plate_pos):
        try:
            pick_and_place(robot, world_pos, plate_pos)
            print("Done! Press SPACE for another cycle.\n")
        except Exception as e:
            print(f"Pick-and-place error: {e}")
        finally:
            robot_busy.clear()

    try:
        while True:
            frame = capture_frame(cap)

            # Detect cube
            detection = detect_object(frame, draw=True)

            # Show frame
            status = "BUSY" if robot_busy.is_set() else "READY"
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if robot_busy.is_set() else (0, 255, 0), 2)
            if detection:
                world_pos = cal.pixel_to_world(detection["cx"], detection["cy"])
                info = f"Cube: ({world_pos[0]:.3f},{world_pos[1]:.3f},{world_pos[2]:.3f})"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Pipeline", frame)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                if robot_busy.is_set():
                    print("Robot is busy!")
                    continue
                if detection is None:
                    print("No cube detected!")
                    continue

                world_pos = cal.pixel_to_world(detection["cx"], detection["cy"])
                plate_pos = np.array([PLATE_POS_XY[0], PLATE_POS_XY[1], PLATE_Z])

                print(f"\nExecuting pick-and-place...")
                robot_busy.set()
                threading.Thread(target=run_pick, args=(world_pos, plate_pos), daemon=True).start()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()


def run_manual_mode():
    """
    Manual mode: move arm to positions using keyboard.
    Useful for testing IK and finding calibration points.
    """
    print("=" * 50)
    print("Manual Mode - Test IK and find positions")
    print("=" * 50)

    robot = SO101Controller(port=ROBOT_PORT, calibration_dir=CALIBRATION_DIR)
    robot.connect()

    print("\nCommands:")
    print("  'h'          - move to home position")
    print("  'p'          - print current state + FK position")
    print("  'g X Y Z'    - go to world position via IK (meters)")
    print("  'j A B C D E'- set joint angles directly (degrees)")
    print("  'o' / 'c'    - open/close gripper")
    print("  'q'          - quit")

    try:
        while True:
            cmd = input("\n> ").strip()
            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0].lower()

            if action == 'q':
                break
            elif action == 'h':
                print("Moving to home...")
                robot.move_smooth([0, 0, 0, 0, 0], gripper=50, duration=2.0)
            elif action == 'p':
                state = robot.get_state()
                print(f"  Joints (deg): {state[:5]}")
                print(f"  Gripper: {state[5]:.1f}")
                pos, rot = forward_kinematics_deg(state[:5])
                print(f"  TCP position (m): ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            elif action == 'g' and len(parts) >= 4:
                target = [float(x.strip(',')) for x in parts[1:4]]
                state = robot.get_state()
                angles, ok, err = inverse_kinematics_deg(np.array(target), state[:5])
                print(f"  IK: angles={angles}, success={ok}, error={err:.4f}m")
                if ok or err < 0.05:
                    robot.move_smooth(angles, duration=MOVE_DURATION)
                else:
                    print("  IK failed, not moving")
            elif action == 'j' and len(parts) >= 6:
                angles = [float(x) for x in parts[1:6]]
                robot.move_smooth(angles, duration=MOVE_DURATION)
            elif action == 'o':
                robot.open_gripper()
            elif action == 'c':
                robot.close_gripper()
            else:
                print("Unknown command")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manual":
        run_manual_mode()
    else:
        run_pipeline()
