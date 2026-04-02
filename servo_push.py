
import sys, json, cv2, time
import numpy as np
from pathlib import Path
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from ultralytics import YOLOWorld
from so101_kinematics import forward_kinematics_deg, inverse_kinematics_deg, JOINT_LIMITS_RAD
from robot_controller import SO101Controller

CAM1_INDEX  = 0
CAM2_INDEX  = 2
CAM_BACKEND = 700
TARGET_FILE = Path(__file__).parent / "target.json"

K1 = 27.5;  CAM1_X = 0.50
K2 = 27.72; CAM2_Y = 0.35

KP        = 0.0008
DEADBAND  = 4
MAX_STEP  = 0.015
PUSH_Z_OFFSET = -0.02


assert TARGET_FILE.exists(), "Run capture_target.py first!"
target = json.loads(TARGET_FILE.read_text())
H1_TARGET = target["h1"]
H2_TARGET = target["h2"]
print(f"Target: CAM1 h={H1_TARGET}  CAM2 h={H2_TARGET}")


def solve_ik(target_pos, hint):
    best_j, best_e = None, np.inf
    for guess in [hint] + [
        np.rad2deg([np.random.uniform(lo,hi) for lo,hi in JOINT_LIMITS_RAD])
        for _ in range(20)
    ]:
        j,_,e = inverse_kinematics_deg(target_pos, initial_guess_deg=guess)
        if e < best_e: best_j, best_e = j, e
        if best_e < 0.003: break
    return best_j, best_e


def detect(model, frame):
    res = model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes
    if boxes and len(boxes):
        i = int(boxes.conf.argmax())
        x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        return x1,y1,x2,y2
    return None


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

print("\nPosition gripper near cube (use teleop if needed).")
print("SPACE = start servo  Q = quit\n")

servo_active = False
done = False

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue

    h1 = h2 = None
    d1 = detect(model, f1)
    if d1:
        x1,y1,x2,y2 = d1
        h1 = y2-y1
        err1 = H1_TARGET - h1
        col = (0,255,0) if abs(err1) < DEADBAND else (0,165,255)
        cv2.rectangle(f1,(x1,y1),(x2,y2),col,2)
        cv2.putText(f1, f"h={h1} err={err1:+d}", (x1,max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)

    d2 = detect(model, f2)
    if d2:
        x1,y1,x2,y2 = d2
        h2 = y2-y1
        err2 = H2_TARGET - h2
        col = (0,255,0) if abs(err2) < DEADBAND else (0,165,255)
        cv2.rectangle(f2,(x1,y1),(x2,y2),col,2)
        cv2.putText(f2, f"h={h2} err={err2:+d}", (x1,max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)


    if servo_active and h1 is not None and h2 is not None:
        err1 = H1_TARGET - h1
        err2 = H2_TARGET - h2

        if abs(err1) < DEADBAND and abs(err2) < DEADBAND:
            print("  TARGET REACHED!")
            servo_active = False
            done = True
        else:

            state = ctrl.get_state()
            tcp, _ = forward_kinematics_deg(state[:5])


            dx = np.clip(-KP * err1, -MAX_STEP, MAX_STEP)
            dy = np.clip(-KP * err2, -MAX_STEP, MAX_STEP)

            new_target = tcp + np.array([dx, dy, 0.0])
            joints_new, ik_err = solve_ik(new_target, state[:5])

            if ik_err < 0.015:
                ctrl.move_smooth(joints_new, duration=0.3, steps=8)

            print(f"\r  err1={err1:+4d}px  err2={err2:+4d}px  "
                  f"dx={dx*100:+.1f}cm  dy={dy*100:+.1f}cm", end="", flush=True)


    cv2.putText(f1, f"CAM1  target h={H1_TARGET}", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,220,255),2)
    cv2.putText(f2, f"CAM2  target h={H2_TARGET}", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,220,255),2)

    status = "SERVO ACTIVE" if servo_active else ("DONE - run scenario" if done else "SPACE=start  Q=quit")
    color  = (0,80,255) if servo_active else ((0,255,0) if done else (150,150,150))
    cv2.putText(f1, status, (10,f1.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Visual Servo", np.hstack([f1, f2]))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        servo_active = not servo_active
        done = False
        print(f"\nServo {'STARTED' if servo_active else 'STOPPED'}")

cap1.release(); cap2.release()
cv2.destroyAllWindows()
ctrl.disconnect()
