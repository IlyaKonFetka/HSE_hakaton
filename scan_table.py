"""
Table-surface scan via teleoperation.

Procedure:
  1. Script starts teleoperation (leader → follower)
  2. Move leader so follower fingertips touch the table surface
  3. Sweep slowly across the workspace
  4. Press SPACE to capture a point (or hold for auto-capture every 0.5s)
  5. Press 'q' to finish → saves point cloud + new calibration

Two overlays on the camera feed:
  RED dot   = where our camera model PREDICTS the TCP to be
  GREEN dot = where you CLICK to say where TCP actually is

If red dots land on the actual fingertip → model is correct.
If not → click the real position to build a correction calibration.

Output: table_scan.json  (point cloud)
        calibration.json (updated homography if ≥4 click-pairs collected)
"""

import sys, json, time, cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from so101_kinematics import forward_kinematics_deg
from workspace_calibration import WorkspaceCalibration

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
CAMERA_BACKEND = 700
FOLLOWER_PORT  = "COM7"
LEADER_PORT    = "COM6"
CALIB_DIR_F = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower"
CALIB_DIR_L = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\teleoperators\so_leader"

SCAN_FILE = Path(__file__).parent / "table_scan.json"
CAL_FILE  = Path(__file__).parent / "calibration.json"

# Camera model (single frontal camera)
CAM_POS = np.array([0.50, 0.00, -0.020])
F_PX    = 620.0
CX, CY  = 320.0, 240.0

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TELEOP_HZ = 30

# ── Camera projection ─────────────────────────────────────────────────────────
def robot_to_pixel(xyz):
    """Project robot-frame point to image pixel. Returns (u,v) or None."""
    d = xyz - CAM_POS
    fwd = -d[0]   # along -X (viewing direction)
    rgt = -d[1]   # along -Y (camera right)
    dwn = -d[2]   # along -Z (camera down)
    if fwd <= 0.01:
        return None
    u = int(CX + F_PX * rgt / fwd)
    v = int(CY + F_PX * dwn / fwd)
    return u, v


def connect_arms():
    from pathlib import Path as P
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.teleoperators.so_leader.so_leader import SOLeader
    from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

    f_cfg = SOFollowerRobotConfig(
        port=FOLLOWER_PORT, id="my_follower",
        disable_torque_on_disconnect=False, cameras={},
        calibration_dir=P(CALIB_DIR_F),
    )
    follower = SOFollower(f_cfg); follower.connect()

    l_cfg = SOLeaderTeleopConfig(
        port=LEADER_PORT, id="my_leader",
        calibration_dir=P(CALIB_DIR_L),
    )
    leader = SOLeader(l_cfg); leader.connect()
    return follower, leader


def get_tcp(follower):
    obs = follower.get_observation()
    joints = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
    pos, _ = forward_kinematics_deg(joints)
    return pos, joints


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Table Surface Scan ===")
    print("Connecting arms...")
    try:
        follower, leader = connect_arms()
        print("Arms connected.\n")
        robot_ok = True
    except Exception as e:
        print(f"Arm connection failed: {e}")
        follower = leader = None
        robot_ok = False

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    assert cap.isOpened(), "Cannot open camera"

    cal = WorkspaceCalibration()     # for click-based correction pairs
    scan_points = []                 # list of {x,y,z} from arm sweep
    current_tcp = [None]
    last_auto_t = [0.0]
    auto_capture = [False]

    def on_click(event, x, y, flags, param):
        """Click = mark where TCP fingertip ACTUALLY appears in image."""
        if event == cv2.EVENT_LBUTTONDOWN and current_tcp[0] is not None:
            tcp = current_tcp[0]
            cal.add_point((x, y), tcp)
            n = len(cal.pixel_points)
            print(f"  Click #{n}: pixel=({x},{y})  robot=({tcp[0]:.4f},{tcp[1]:.4f},{tcp[2]:.4f})")

    cv2.namedWindow("Table Scan")
    cv2.setMouseCallback("Table Scan", on_click)

    print("Controls:")
    print("  SPACE      = capture current TCP point")
    print("  H          = toggle auto-capture every 0.5s (hold and sweep)")
    print("  Left-click = mark where arm tip IS in image (correction pair)")
    print("  Q          = finish and save\n")

    dt = 1.0 / TELEOP_HZ

    while True:
        t0 = time.time()

        if robot_ok:
            try:
                action = leader.get_action()
                follower.send_action(action)
                tcp, _ = get_tcp(follower)
                current_tcp[0] = tcp
            except Exception as e:
                print(f"Teleop error: {e}")

        ret, frame = cap.read()
        if not ret:
            continue

        h_f, w_f = frame.shape[:2]
        tcp = current_tcp[0]

        # ── Draw projected TCP position (RED = predicted by camera model) ──
        if tcp is not None:
            pix = robot_to_pixel(tcp)
            if pix and 0 <= pix[0] < w_f and 0 <= pix[1] < h_f:
                cv2.circle(frame, pix, 10, (0, 0, 255), 2)
                cv2.circle(frame, pix, 2,  (0, 0, 255), -1)
                cv2.putText(frame, "model", (pix[0]+12, pix[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # ── Draw click-calibration points (GREEN) ──
        for i, px in enumerate(cal.pixel_points):
            cv2.circle(frame, (int(px[0]), int(px[1])), 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (int(px[0])+7, int(px[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # ── Draw scan points projected (YELLOW dots) ──
        for pt in scan_points[-50:]:   # show last 50
            xyz = np.array([pt["x"], pt["y"], pt["z"]])
            pix2 = robot_to_pixel(xyz)
            if pix2 and 0 <= pix2[0] < w_f and 0 <= pix2[1] < h_f:
                cv2.circle(frame, pix2, 3, (0, 220, 255), -1)

        # ── TCP readout ──
        if tcp is not None:
            cv2.putText(frame, f"TCP: x={tcp[0]:.3f}  y={tcp[1]:.3f}  z={tcp[2]:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Status bar ──
        n_scan  = len(scan_points)
        n_click = len(cal.pixel_points)
        mode_str = "AUTO" if auto_capture[0] else "manual"
        bar = (f"scan={n_scan}pts  clicks={n_click}  [{mode_str}]  "
               f"SPACE=cap  H=auto  click=correct  Q=save")
        cv2.rectangle(frame, (0, h_f-35), (w_f, h_f), (0, 0, 0), -1)
        cv2.putText(frame, bar, (6, h_f-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 0), 1)

        cv2.imshow("Table Scan", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Auto-capture while H is held ──
        now = time.time()
        if auto_capture[0] and tcp is not None and (now - last_auto_t[0]) >= 0.5:
            scan_points.append({"x": float(tcp[0]), "y": float(tcp[1]), "z": float(tcp[2])})
            last_auto_t[0] = now
            print(f"\r  Auto: ({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})  total={len(scan_points)}",
                  end="", flush=True)

        if key == ord('q'):
            break
        elif key == ord(' ') and tcp is not None:
            scan_points.append({"x": float(tcp[0]), "y": float(tcp[1]), "z": float(tcp[2])})
            print(f"  Saved #{len(scan_points)}: ({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})")
        elif key == ord('h'):
            auto_capture[0] = not auto_capture[0]
            print(f"\nAuto-capture: {'ON' if auto_capture[0] else 'OFF'}")

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    if robot_ok:
        follower.disconnect()
        leader.disconnect()

    # ── Save scan ─────────────────────────────────────────────────────────────
    print(f"\n\nScan complete: {len(scan_points)} points")

    if scan_points:
        SCAN_FILE.write_text(json.dumps(scan_points, indent=2))
        print(f"Point cloud saved to {SCAN_FILE}")

        pts = np.array([[p["x"], p["y"], p["z"]] for p in scan_points])
        print(f"\nWorkspace bounds from scan:")
        print(f"  X: {pts[:,0].min():.3f} … {pts[:,0].max():.3f} m")
        print(f"  Y: {pts[:,1].min():.3f} … {pts[:,1].max():.3f} m")
        print(f"  Z: {pts[:,2].min():.4f} … {pts[:,2].max():.4f} m  "
              f"(mean={pts[:,2].mean():.4f})")
        table_z = float(pts[:,2].mean())
        print(f"\n  → Estimated TABLE_Z = {table_z:.4f} m")

    # ── Save click calibration if enough points ───────────────────────────────
    if len(cal.pixel_points) >= 4:
        print(f"\nFitting calibration from {len(cal.pixel_points)} click-pairs...")
        err = cal.compute()
        cal.save(str(CAL_FILE))
        print(f"Calibration saved to {CAL_FILE}  (mean error: {err:.4f} m)")
    elif len(cal.pixel_points) > 0:
        print(f"\nOnly {len(cal.pixel_points)} click-pairs — need 4 to fit calibration.")
    else:
        print("\nNo click corrections — calibration unchanged.")

    # ── Print suggested TABLE_Z update ───────────────────────────────────────
    if scan_points:
        print(f"\nIf TABLE_Z is wrong, update in localize.py:")
        print(f"  TABLE_Z = {table_z:.4f}")
        print(f"  GRAB_Z  = {table_z + 0.025:.4f}  (table + half cube)")


if __name__ == "__main__":
    main()
