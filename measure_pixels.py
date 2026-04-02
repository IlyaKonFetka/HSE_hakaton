"""
Live pixel-height measurement tool.
Place cube at known distances, read off h_px.
Press SPACE to log the current measurement.
"""
import sys, cv2, numpy as np
sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
from ultralytics import YOLOWorld

CAM1_INDEX = 0
CAM2_INDEX = 2
BACKEND    = 700
IMG_H      = 480

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["cube", "wooden block", "small box"])

cap1 = cv2.VideoCapture(CAM1_INDEX, BACKEND)
cap2 = cv2.VideoCapture(CAM2_INDEX, BACKEND)
assert cap1.isOpened(), "Cannot open CAM1"
assert cap2.isOpened(), "Cannot open CAM2"

log = []

print("Both cameras open. Q = quit.\n")

# Calibrated constants
# k = h_px * D_near_face  (distance to near face, as physically measured)
K1      = 55 * 0.50   # = 27.5  (55px at 50cm near face)
K2      = 44 * 0.63   # = 27.72 (44px at 63cm near face)
CAM1_X  = 0.50   # CAM1 is 50cm from robot in X
CAM2_Y  = 0.35   # CAM2 is 35cm from robot center in Y
F_PX    = 620.0  # focal length in pixels (k/cube_side = 29/0.05 ≈ 580, use 620)
CX      = 320.0  # image center

def detect(frame):
    res = model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes
    if boxes and len(boxes):
        i = int(boxes.conf.argmax())
        x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        return x1,y1,x2,y2
    return None

# Top-down map dimensions
MAP_W, MAP_H = 300, 400
X_MIN, X_MAX = 0.0, 0.55   # robot frame X range (m)
Y_MIN, Y_MAX = -0.30, 0.30  # robot frame Y range (m)

def make_map(X, Y):
    """Draw top-down workspace map with cube position."""
    m = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    # grid lines
    for xg in np.arange(0.0, 0.55, 0.10):
        ux = int((xg - X_MIN) / (X_MAX - X_MIN) * MAP_W)
        cv2.line(m, (ux,0), (ux,MAP_H), (40,40,40), 1)
        cv2.putText(m, f"{xg:.0f}", (ux+2, MAP_H-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60,60,60), 1)
    for yg in np.arange(-0.30, 0.31, 0.10):
        vy = int((1 - (yg - Y_MIN) / (Y_MAX - Y_MIN)) * MAP_H)
        cv2.line(m, (0,vy), (MAP_W,vy), (40,40,40), 1)
    # robot base
    rx = int((0.0 - X_MIN) / (X_MAX - X_MIN) * MAP_W)
    ry = int((1 - (0.0 - Y_MIN) / (Y_MAX - Y_MIN)) * MAP_H)
    cv2.drawMarker(m, (rx, ry), (0,100,255), cv2.MARKER_STAR, 20, 2)
    cv2.putText(m, "Robot", (rx+5, ry-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,255), 1)
    # camera position
    cx_m = int((CAM1_X - X_MIN) / (X_MAX - X_MIN) * MAP_W)
    cv2.drawMarker(m, (cx_m, ry), (200,200,0), cv2.MARKER_TRIANGLE_DOWN, 14, 2)
    cv2.putText(m, "CAM1", (cx_m+5, ry+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)
    # cube
    if X is not None:
        ux = int((X - X_MIN) / (X_MAX - X_MIN) * MAP_W)
        vy = int((1 - (Y - Y_MIN) / (Y_MAX - Y_MIN)) * MAP_H)
        ux = np.clip(ux, 5, MAP_W-5)
        vy = np.clip(vy, 5, MAP_H-5)
        cv2.rectangle(m, (ux-10, vy-10), (ux+10, vy+10), (0,255,0), 2)
        cv2.putText(m, f"({X:.2f},{Y:.2f})", (ux+12, vy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    # axes labels
    cv2.putText(m, "X->", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    cv2.putText(m, "Y", (MAP_W-20, MAP_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    return m


while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2: continue

    # ── CAM1: X only (from depth) ─────────────────────────────────────────
    det1 = detect(f1)
    X = None
    if det1:
        bx1,by1,bx2,by2 = det1
        h_px = by2 - by1
        D1   = K1 / h_px
        X    = CAM1_X - D1
        cv2.rectangle(f1,(bx1,by1),(bx2,by2),(0,255,0),2)
        cv2.putText(f1, f"h={h_px}", (bx1, max(by1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.putText(f1, f"X = {X:.3f} m", (10, f1.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,80), 2)

    # ── CAM2: Y only (from depth) ─────────────────────────────────────────
    det2 = detect(f2)
    Y = None
    if det2:
        bx1,by1,bx2,by2 = det2
        h_px2 = by2 - by1
        D2    = K2 / h_px2
        Y     = CAM2_Y - D2
        cv2.rectangle(f2,(bx1,by1),(bx2,by2),(0,255,0),2)
        cv2.putText(f2, f"h={h_px2}", (bx1, max(by1-8,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.putText(f2, f"Y = {Y:.3f} m", (10, f2.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,80), 2)

    cv2.putText(f1, "CAM1 -> X", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)
    cv2.putText(f2, "CAM2 -> Y", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,255), 2)

    cv2.imshow("Localization", np.hstack([f1, f2]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release(); cap2.release()
cv2.destroyAllWindows()
