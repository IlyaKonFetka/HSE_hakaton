"""
Workspace calibration using teleop (leader → follower) as a 3D pointer.

Procedure:
1. Script connects both arms and starts teleoperation
2. Use leader arm to guide follower gripper ON TOP of the cube
3. Press SPACE → captures follower TCP position
4. Click on the cube center in the camera window
5. Move cube to a new spot, repeat 4+ times
6. Press 'q' to finish and save calibration
"""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from so101_kinematics import forward_kinematics_deg
from workspace_calibration import WorkspaceCalibration

# Load YOLO for cube detection overlay
try:
    from ultralytics import YOLOWorld
    _yolo = YOLOWorld("yolov8s-worldv2.pt")
    _yolo.set_classes(["cube", "wooden block", "small box"])
    YOLO_OK = True
except Exception:
    YOLO_OK = False

# ── Config ──────────────────────────────────────────────
CAMERA_INDEX   = 0
CAMERA_BACKEND = 700
FOLLOWER_PORT  = "COM7"
LEADER_PORT    = "COM6"
CALIB_DIR_FOLLOWER = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower"
CALIB_DIR_LEADER   = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\teleoperators\so_leader"
SAVE_PATH      = r"c:\DISK Z\Hakaton_HSE\cv_ik_pipeline\calibration.json"
TELEOP_HZ      = 30
# ────────────────────────────────────────────────────────


def connect_arms():
    from pathlib import Path
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    follower_cfg = SOFollowerRobotConfig(
        port=FOLLOWER_PORT, id="my_follower",
        disable_torque_on_disconnect=False, cameras={},
        calibration_dir=Path(CALIB_DIR_FOLLOWER),
    )
    follower = SOFollower(follower_cfg)
    follower.connect()

    leader_cfg = SOLeaderTeleopConfig(
        port=LEADER_PORT, id="my_leader",
        calibration_dir=Path(CALIB_DIR_LEADER),
    )
    leader = SOLeader(leader_cfg)
    leader.connect()

    return follower, leader


MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

def get_tcp_pos(follower):
    obs = follower.get_observation()
    state = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
    pos, _ = forward_kinematics_deg(state)
    return pos, state


def teleop_step(follower, leader):
    """Send one leader→follower command."""
    action = leader.get_action()
    follower.send_action(action)


def main():
    print("=== Arm-assisted Workspace Calibration (with teleop) ===")
    print("1. Script starts teleop: leader arm controls follower")
    print("2. Guide follower gripper ON TOP of cube")
    print("3. Press SPACE → captures TCP position")
    print("4. Click on cube center in camera window")
    print("5. Move cube, repeat 4+ times | Press 'q' to finish\n")

    print("Connecting arms...")
    try:
        follower, leader = connect_arms()
        print("Arms connected. Teleop active.\n")
        robot_ok = True
    except Exception as e:
        print(f"Arm connection failed: {e}\n")
        follower = leader = None
        robot_ok = False

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}")
        return

    cal = WorkspaceCalibration()
    pending_world = [None]
    current_tcp = [None]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and pending_world[0] is not None:
            world = pending_world[0]
            cal.add_point((x, y), world)
            print(f"  Saved #{len(cal.pixel_points)}: pixel=({x},{y})  world=({world[0]:.4f}, {world[1]:.4f}, {world[2]:.4f}) m")
            pending_world[0] = None

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    dt = 1.0 / TELEOP_HZ

    while True:
        t0 = time.time()

        # Teleop step
        if robot_ok:
            try:
                teleop_step(follower, leader)
                tcp, _ = get_tcp_pos(follower)
                current_tcp[0] = tcp
            except Exception as e:
                print(f"Teleop error: {e}")

        ret, frame = cap.read()
        if not ret:
            break

        h_f, w_f = frame.shape[:2]

        # YOLO cube detection overlay
        cube_px = None
        if YOLO_OK:
            res = _yolo.predict(frame, conf=0.05, verbose=False)
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                best = int(boxes.conf.argmax())
                x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
                cube_px = ((x1+x2)//2, (y1+y2)//2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                cv2.putText(frame, "CUBE", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Draw previously saved calibration points
        for i, px in enumerate(cal.pixel_points):
            cv2.circle(frame, (int(px[0]), int(px[1])), 8, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (int(px[0])+8, int(px[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # ── Big TCP readout ──────────────────────────────────────────────
        if current_tcp[0] is not None:
            t = current_tcp[0]
            cv2.putText(frame, "FOLLOWER TCP (m):", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, f"  x={t[0]:.4f}  y={t[1]:.4f}  z={t[2]:.4f}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No TCP (arm not connected)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── State banner ─────────────────────────────────────────────────
        n_pts = len(cal.pixel_points)
        if pending_world[0] is not None:
            # waiting for click
            banner = f"[{n_pts}/4] TCP captured!  >> CLICK ON CUBE IN IMAGE <<"
            color  = (0, 0, 255)
        else:
            banner = f"[{n_pts}/4]  Guide gripper ONTO cube, then SPACE  |  Q=save+quit"
            color  = (255, 220, 0)

        cv2.rectangle(frame, (0, h_f-45), (w_f, h_f), (0,0,0), -1)
        cv2.putText(frame, banner, (10, h_f-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') and current_tcp[0] is not None:
            pending_world[0] = current_tcp[0].copy()
            t = pending_world[0]
            print(f"\nTCP captured: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")
            print("  Now CLICK on the cube in the camera window.")

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    if robot_ok:
        follower.disconnect()
        leader.disconnect()

    if len(cal.pixel_points) >= 3:
        print(f"\nFitting calibration with {len(cal.pixel_points)} points...")
        error = cal.compute()
        cal.save(SAVE_PATH)
        print(f"Saved! Mean error: {error:.4f} m")
    else:
        print(f"Need at least 3 points (got {len(cal.pixel_points)}). Not saved.")


if __name__ == "__main__":
    main()
