"""
Full pick-and-place pipeline:
  1. YOLO detects cube on CAM1
  2. Similar-triangles → (X, Y) in robot frame
  3. IK adjusts above_pick / pick to actual cube position
  4. Executes: home → above_pick → pick → close → above_pick → above_place → place → open → home
SPACE = start, Q = quit
"""
import sys, json, time, cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from ultralytics import YOLOWorld
from so101_kinematics import forward_kinematics_deg, inverse_kinematics_deg, JOINT_LIMITS_RAD
from robot_controller import SO101Controller

# ── Config ────────────────────────────────────────────────────────────────────
CAM1_INDEX  = 0
CAM2_INDEX  = 2
CAM_BACKEND = 700
K1     = 27.5    # CAM1: h_px * D_near_face
K2     = 27.72   # CAM2: h_px * D_near_face
F_PX   = 620.0
CX     = 320.0
CAM1_X = 0.50    # CAM1 is 50cm from robot in X
CAM2_Y = 0.35    # CAM2 is 35cm from robot center in Y

POSES_FILE = Path(__file__).parent / "poses.json"
MOTORS = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]

# ── Load poses ────────────────────────────────────────────────────────────────
poses = json.loads(POSES_FILE.read_text())
def joints(name):
    return np.array(poses[name][:5], dtype=float)
def gripper(name):
    return float(poses[name][5])

# ── Cube detection ────────────────────────────────────────────────────────────
def detect_best(model, frame):
    res = model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    i = int(boxes.conf.argmax())
    x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
    return x1, y1, x2, y2

def detect_cube(model, f1, f2):
    bbox1 = bbox2 = None
    X = Y = None

    det1 = detect_best(model, f1)
    if det1:
        x1,y1,x2,y2 = det1
        bbox1 = det1
        h_px = y2 - y1
        D1   = K1 / h_px
        X    = CAM1_X - D1          # CAM1 -> X only

    det2 = detect_best(model, f2)
    if det2:
        x1,y1,x2,y2 = det2
        bbox2 = det2
        h_px = y2 - y1
        D2   = K2 / h_px
        Y    = CAM2_Y - D2          # CAM2 -> Y only

    if X is None or Y is None:
        return None
    return X, Y, bbox1, bbox2

# ── IK with multi-start ───────────────────────────────────────────────────────
def solve_ik(target, hint_deg):
    best_j, best_e = None, np.inf
    for guess in [hint_deg] + [
        np.rad2deg([np.random.uniform(lo,hi) for lo,hi in JOINT_LIMITS_RAD])
        for _ in range(40)
    ]:
        j, _, e = inverse_kinematics_deg(target, initial_guess_deg=guess)
        if e < best_e:
            best_j, best_e = j, e
        if best_e < 0.003:
            break
    return best_j, best_e

# ── Pick-and-place sequence ───────────────────────────────────────────────────
def pick_and_place(ctrl, cube_x, cube_y):
    print(f"\n=== Pick-and-place: cube at X={cube_x:.3f} Y={cube_y:.3f} ===")

    # Offset 3cm radially outward from robot origin
    r = np.sqrt(cube_x**2 + cube_y**2)
    if r > 0.001:
        cube_x = cube_x - 0.03 * cube_x / r
        cube_y = cube_y - 0.03 * cube_y / r
        print(f"  Offset target: X={cube_x:.3f} Y={cube_y:.3f}")

    # Get Z heights from taught positions via FK
    pick_fk,  _ = forward_kinematics_deg(joints("pick"))
    above_fk, _ = forward_kinematics_deg(joints("above_pick"))
    pick_z  = float(pick_fk[2]) - 0.048  # grab 4.5cm lower
    above_z = float(above_fk[2]) + 0.05  # 5cm higher approach point
    print(f"  Pick Z={pick_z:.3f}  Above Z={above_z:.3f}")

    # Solve IK for actual cube position
    target_pick  = np.array([cube_x, cube_y, pick_z])
    target_above = np.array([cube_x, cube_y, above_z])

    j_pick,  e_pick  = solve_ik(target_pick,  joints("pick"))
    j_above, e_above = solve_ik(target_above, joints("above_pick"))
    print(f"  IK pick err={e_pick*1000:.1f}mm  above err={e_above*1000:.1f}mm")

    # Fall back to taught positions if IK is bad
    if e_pick > 0.020:
        print("  IK error too large for pick — using taught position")
        j_pick = joints("pick")
    if e_above > 0.020:
        print("  IK error too large for above_pick — using taught position")
        j_above = joints("above_pick")

    g_open  = gripper("above_pick")
    g_close = 10.0  # closed

    print("  → home"); ctrl.move_smooth(joints("home"), gripper("home"), duration=2.0)
    print("  → above pick"); ctrl.move_smooth(j_above, g_open, duration=2.0)
    print("  → pick (open gripper)"); ctrl.move_smooth(j_pick, g_open, duration=1.5)
    print("  → close gripper"); ctrl.close_gripper(g_close, duration=0.8)
    print("  → above pick"); ctrl.move_smooth(j_above, g_close, duration=1.5)
    print("  → above place"); ctrl.move_smooth(joints("above_place"), g_close, duration=2.0)
    print("  → place"); ctrl.move_smooth(joints("place"), g_close, duration=1.5)
    print("  → open gripper"); ctrl.open_gripper(g_open, duration=0.8)
    print("  → home"); ctrl.move_smooth(joints("home"), g_open, duration=2.0)
    print("  Done!\n")

# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading YOLO...")
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["cube", "wooden block", "small box"])
print("Connecting robot...")
ctrl = SO101Controller("COM7",
    r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower")
ctrl.connect()

cap1 = cv2.VideoCapture(CAM1_INDEX, CAM_BACKEND)
cap2 = cv2.VideoCapture(CAM2_INDEX, CAM_BACKEND)
assert cap1.isOpened() and cap2.isOpened()

print("\nCameras open. SPACE = pick cube, Q = quit\n")
running = False

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue

    det = detect_cube(model, f1, f2)
    if det:
        X, Y, bbox1, bbox2 = det
        if bbox1:
            cv2.rectangle(f1, bbox1[:2], bbox1[2:], (0,255,0), 2)
        if bbox2:
            cv2.rectangle(f2, bbox2[:2], bbox2[2:], (0,255,0), 2)
        cv2.putText(f1, f"X={X:.3f}  Y={Y:.3f}", (10, f1.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,80), 2)

    status = "RUNNING" if running else "SPACE=pick  Q=quit"
    cv2.putText(f1, "CAM1", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
    cv2.putText(f2, "CAM2", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
    cv2.putText(f1, status, (10, f1.shape[0]-45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255) if not running else (0,80,255), 2)

    cv2.imshow("Pick and Place", np.hstack([f1, f2]))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not running and det:
        X, Y = det[0], det[1]
        running = True
        try:
            pick_and_place(ctrl, X, Y)
        except Exception as e:
            print(f"Error: {e}")
        running = False

cap1.release(); cap2.release()
cv2.destroyAllWindows()
ctrl.disconnect()
