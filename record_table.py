
import sys, json, time, cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

from ultralytics import YOLOWorld
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

FOLLOWER_PORT = "COM7"
LEADER_PORT   = "COM6"
CALIB_F = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\robots\so_follower"
CALIB_L = r"C:\Users\simal\.cache\huggingface\lerobot\calibration\teleoperators\so_leader"

CAM1_INDEX  = 0
CAM2_INDEX  = 2
CAM_BACKEND = 700

POINTS_FILE = Path(__file__).parent / "table_points.json"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


print("Loading YOLO...")
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["cube", "wooden block", "small box"])

def detect(frame):
    res = model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes
    if boxes and len(boxes):
        i = int(boxes.conf.argmax())
        x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        return x1,y1,x2,y2
    return None


print("Connecting arms...")
f_cfg = SOFollowerRobotConfig(port=FOLLOWER_PORT, id="my_follower",
    disable_torque_on_disconnect=False, cameras={}, calibration_dir=Path(CALIB_F))
follower = SOFollower(f_cfg); follower.connect()

l_cfg = SOLeaderTeleopConfig(port=LEADER_PORT, id="my_leader", calibration_dir=Path(CALIB_L))
leader = SOLeader(l_cfg); leader.connect()

cap1 = cv2.VideoCapture(CAM1_INDEX, CAM_BACKEND)
cap2 = cv2.VideoCapture(CAM2_INDEX, CAM_BACKEND)
assert cap1.isOpened() and cap2.isOpened()


points = []
if POINTS_FILE.exists():
    points = json.loads(POINTS_FILE.read_text())
    print(f"Loaded {len(points)} existing points")

auto = False
last_auto_t = 0.0

print(f"\nTeleop active. Touch table vertically with fingertips.")
print("SPACE=record  H=auto-record  Q=save&quit\n")

while True:
    t0 = time.time()


    action = leader.get_action()
    follower.send_action(action)


    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue


    h1 = h2 = cx1 = cx2 = None
    d1 = detect(f1)
    if d1:
        x1,y1,x2,y2 = d1
        h1  = y2-y1
        cx1 = (x1+x2)//2
        cv2.rectangle(f1,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(f1,f"h={h1} cx={cx1}",(x1,max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    d2 = detect(f2)
    if d2:
        x1,y1,x2,y2 = d2
        h2  = y2-y1
        cx2 = (x1+x2)//2
        cv2.rectangle(f2,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(f2,f"h={h2} cx={cx2}",(x1,max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)


    obs = follower.get_observation()
    joints = [float(obs[f"{m}.pos"]) for m in MOTORS]

    def save_point():
        pt = {"joints": joints, "h1": h1, "cx1": cx1, "h2": h2, "cx2": cx2}
        points.append(pt)
        print(f"  #{len(points)}: h1={h1} h2={h2}  joints=[{joints[0]:.1f},{joints[1]:.1f},{joints[2]:.1f},{joints[3]:.1f},{joints[4]:.1f}]")


    cv2.putText(f1,"CAM1",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,255),2)
    cv2.putText(f2,"CAM2",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,255),2)
    mode_str = "AUTO" if auto else "manual"
    bar = f"pts={len(points)}  [{mode_str}]  SPACE=rec  H=auto  Q=save"
    cv2.rectangle(f1,(0,f1.shape[0]-30),(f1.shape[1],f1.shape[0]),(0,0,0),-1)
    cv2.putText(f1,bar,(6,f1.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,220,0),1)

    cv2.imshow("Record Table", np.hstack([f1, f2]))
    key = cv2.waitKey(1) & 0xFF


    now = time.time()
    if auto and (now - last_auto_t) >= 0.5:
        save_point()
        last_auto_t = now

    if key == ord('q'):
        break
    elif key == ord(' '):
        save_point()
    elif key == ord('h'):
        auto = not auto
        print(f"\nAuto-record: {'ON' if auto else 'OFF'}")

    elapsed = time.time() - t0
    if elapsed < 1/30:
        time.sleep(1/30 - elapsed)


cap1.release(); cap2.release()
cv2.destroyAllWindows()
follower.disconnect()
leader.disconnect()

POINTS_FILE.write_text(json.dumps(points, indent=2))
print(f"\nSaved {len(points)} points to {POINTS_FILE}")
