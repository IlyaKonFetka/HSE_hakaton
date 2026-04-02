
import sys, json, cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
from ultralytics import YOLOWorld

CAM1_INDEX = 0
CAM2_INDEX = 2
BACKEND    = 700
TARGET_FILE = Path(__file__).parent / "target.json"

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["cube", "wooden block", "small box"])

cap1 = cv2.VideoCapture(CAM1_INDEX, BACKEND)
cap2 = cv2.VideoCapture(CAM2_INDEX, BACKEND)
assert cap1.isOpened() and cap2.isOpened()

def detect(frame):
    res = model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes
    if boxes and len(boxes):
        i = int(boxes.conf.argmax())
        x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        return x1,y1,x2,y2
    return None

print("Place cube at the TARGET position. SPACE = capture, Q = quit.\n")

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue

    h1 = h2 = cx1 = cx2 = None

    d1 = detect(f1)
    if d1:
        x1,y1,x2,y2 = d1
        h1  = y2 - y1
        cx1 = (x1+x2)//2
        cv2.rectangle(f1,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(f1, f"h={h1}  cx={cx1}", (x1,max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    d2 = detect(f2)
    if d2:
        x1,y1,x2,y2 = d2
        h2  = y2 - y1
        cx2 = (x1+x2)//2
        cv2.rectangle(f2,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(f2, f"h={h2}  cx={cx2}", (x1,max(y1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(f1, "CAM1 -> X", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,220,255),2)
    cv2.putText(f2, "CAM2 -> Y", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,220,255),2)
    cv2.putText(f1, "SPACE=capture  Q=quit",
                (10,f1.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(150,150,150),1)

    cv2.imshow("Capture Target", np.hstack([f1, f2]))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if h1 is None or h2 is None:
            print("  Cube not detected on both cameras!"); continue
        data = {"h1": h1, "cx1": cx1, "h2": h2, "cx2": cx2}
        TARGET_FILE.write_text(json.dumps(data, indent=2))
        print(f"  Target saved: CAM1 h={h1} cx={cx1} | CAM2 h={h2} cx={cx2}")

        green = np.zeros_like(f1); green[:,:,1] = 80
        cv2.imshow("Capture Target", np.hstack([f1+green, f2+green]))
        cv2.waitKey(400)

cap1.release(); cap2.release()
cv2.destroyAllWindows()
