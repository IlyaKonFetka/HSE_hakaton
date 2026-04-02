"""
YOLO cube detection + pixel→robot coords via calibration homography.
Press 'q' to quit.
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
from workspace_calibration import WorkspaceCalibration
from ultralytics import YOLOWorld

CAMERA_INDEX   = 0
CAMERA_BACKEND = 700
CAL_PATH       = r"c:\DISK Z\Hakaton_HSE\cv_ik_pipeline\calibration.json"

cal = WorkspaceCalibration()
cal.load(CAL_PATH)

print("Loading YOLO...")
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["cube", "wooden block", "small box"])
print("Ready.\n")

cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
assert cap.isOpened(), "Cannot open camera"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    res   = model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes

    if boxes is not None and len(boxes) > 0:
        best = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        conf   = float(boxes.conf[best])

        rx, ry, rz = cal.pixel_to_world(cx, cy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        cv2.putText(frame, f"pixel:  ({cx}, {cy})  conf={conf:.2f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"robot:  x={rx:.3f}  y={ry:.3f}  z={rz:.3f} m",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        print(f"\r  px=({cx:4d},{cy:4d})  x={rx:.3f} y={ry:.3f} z={rz:.3f}m",
              end="", flush=True)
    else:
        cv2.putText(frame, "No cube", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Detect & Locate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print()
