"""
Stereo cube localization via similar triangles.

Camera 1 (index 0) - FRONTAL: looks in -X direction
    depth  (box height) -> X estimate   (medium trust)
    lateral (U pixel)   -> Y estimate   (HIGH trust)

Camera 2 (index 2) - SIDE: looks in -Y direction (from left side of workspace)
    depth  (box height) -> Y estimate   (medium trust)
    lateral (U pixel)   -> X estimate   (HIGH trust)

Calibration: place cube at known real distance, SPACE to capture.
Run:  python stereo_localize.py calibrate
      python stereo_localize.py
"""

import sys, json, cv2, time
import numpy as np
from pathlib import Path

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
from ultralytics import YOLOWorld

# ── Config ────────────────────────────────────────────────────────────────────
CAM1_INDEX = 0
CAM2_INDEX = 2
BACKEND    = 700

CUBE_SIDE  = 0.05   # real cube side length in meters (5 cm)
IMG_W, IMG_H = 640, 480
CX = IMG_W / 2      # image center X
CY = IMG_H / 2      # image center Y

# Camera positions in robot frame (meters)
# Cam1: frontal, 50cm ahead, centered
CAM1_POS = np.array([0.50, 0.00, -0.020])   # 5cm above table (table=-0.035)
# Cam2: side, positioned to LEFT of workspace, looking in -Y direction
# x=at robot base level, y=35cm to the left of robot center
CAM2_POS = np.array([0.00, 0.35, -0.020])

GRAB_Z = -0.010   # grab height = table + half cube

CAL_FILE = Path(__file__).parent / "stereo_cal.json"

# Trust weights (0..1): how much each estimate contributes
W_CAM1_Y = 0.70   # cam1 lateral → Y (high trust)
W_CAM2_X = 0.70   # cam2 lateral → X (high trust)
W_CAM1_X = 0.30   # cam1 depth   → X (lower trust)
W_CAM2_Y = 0.30   # cam2 depth   → Y (lower trust)

# ── Similar triangles math ────────────────────────────────────────────────────

class CamModel:
    """
    Single camera calibrated with similar triangles.

    Provides two estimates from a YOLO bounding box:
      depth_estimate   - distance along camera viewing axis
      lateral_estimate - offset perpendicular to viewing axis (from U pixel)
    """

    def __init__(self, name, pos, viewing_axis, right_axis):
        """
        pos          : camera position in robot frame (np array)
        viewing_axis : unit vector camera looks along (e.g. (-1,0,0) or (0,-1,0))
        right_axis   : unit vector = image right direction in robot frame
        """
        self.name   = name
        self.pos    = np.array(pos)
        self.vaxis  = np.array(viewing_axis, dtype=float)
        self.raxis  = np.array(right_axis,   dtype=float)
        self.k      = None   # similar-triangles constant: k = h_px * D_center

    def calibrate(self, h_px, u_px, d_to_near_face_m):
        """Record one calibration point. More points → average."""
        D_center = d_to_near_face_m + CUBE_SIDE / 2
        k_sample = h_px * D_center
        if self.k is None:
            self.k = k_sample
        else:
            self.k = 0.5 * (self.k + k_sample)   # running average
        print(f"  [{self.name}] h_px={h_px:.1f}  D_center={D_center:.3f}m  k={self.k:.1f}")

    @property
    def f_px(self):
        """Estimated focal length from calibration."""
        if self.k is None:
            return None
        return self.k / CUBE_SIDE

    def estimate(self, h_px, u_px):
        """
        Returns (depth_m, lateral_m) in camera frame:
          depth   = distance from camera to cube center along viewing axis
          lateral = offset of cube center perpendicular to viewing axis
                    positive = cube is to the RIGHT in camera image
        Returns None if not calibrated.
        """
        if self.k is None:
            return None, None
        D_near   = self.k / h_px
        D_center = D_near + CUBE_SIDE / 2
        f        = self.f_px
        lateral  = (u_px - CX) * D_center / f   # signed, + = image right
        return D_center, lateral

    def to_robot(self, D_center, lateral):
        """
        Convert (depth, lateral) → 3D point in robot frame.
        depth   : along self.vaxis (away from camera)
        lateral : along self.raxis (image right direction)
        """
        return self.pos + D_center * self.vaxis + lateral * self.raxis

    def save_dict(self):
        return {"name": self.name, "k": self.k,
                "pos": self.pos.tolist(),
                "vaxis": self.vaxis.tolist(),
                "raxis": self.raxis.tolist()}

    @classmethod
    def load_dict(cls, d):
        m = cls(d["name"], d["pos"], d["vaxis"], d["raxis"])
        m.k = d["k"]
        return m


# ── Camera setup ──────────────────────────────────────────────────────────────
# Cam1: frontal, looks in -X, image-right = -Y_robot
cam1 = CamModel("CAM1-frontal",
                pos  = CAM1_POS,
                viewing_axis = (-1, 0, 0),
                right_axis   = ( 0,-1, 0))

# Cam2: side, looks in -Y (from left), image-right = +X_robot
cam2 = CamModel("CAM2-side",
                pos  = CAM2_POS,
                viewing_axis = ( 0,-1, 0),
                right_axis   = ( 1, 0, 0))


def save_calibration():
    data = {"cam1": cam1.save_dict(), "cam2": cam2.save_dict()}
    CAL_FILE.write_text(json.dumps(data, indent=2))
    print(f"Calibration saved to {CAL_FILE}")


def load_calibration():
    if not CAL_FILE.exists():
        return False
    data = json.loads(CAL_FILE.read_text())
    global cam1, cam2
    cam1 = CamModel.load_dict(data["cam1"])
    cam2 = CamModel.load_dict(data["cam2"])
    print(f"Calibration loaded: cam1 k={cam1.k:.1f}  cam2 k={cam2.k:.1f}")
    return True


def detect_best(model, frame):
    """Run YOLO, return (cx, cy, box_h, x1,y1,x2,y2, conf) or None."""
    res   = model.predict(frame, conf=0.05, verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    i = int(boxes.conf.argmax())
    x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    bh = int(y2 - y1)
    return cx, cy, bh, x1, y1, x2, y2, float(boxes.conf[i])


def fuse(cam1_det, cam2_det):
    """
    Fuse 4 estimates into final (x, y) in robot frame.
    Returns (x, y, debug_dict) or None.
    """
    estimates_x, weights_x = [], []
    estimates_y, weights_y = [], []

    if cam1_det:
        cx1, cy1, bh1 = cam1_det
        D1, lat1 = cam1.estimate(bh1, cx1)
        if D1:
            p1 = cam1.to_robot(D1, lat1)
            # cam1 depth → X (vaxis is -X, so pos[0] + D*(-1) = pos[0]-D)
            x1_est = p1[0]
            # cam1 lateral → Y (raxis is -Y, so pos[1] + lat*(-1))
            y1_est = p1[1]
            estimates_x.append(x1_est); weights_x.append(W_CAM1_X)
            estimates_y.append(y1_est); weights_y.append(W_CAM1_Y)

    if cam2_det:
        cx2, cy2, bh2 = cam2_det
        D2, lat2 = cam2.estimate(bh2, cx2)
        if D2:
            p2 = cam2.to_robot(D2, lat2)
            # cam2 depth → Y (vaxis is -Y, so pos[1] + D*(-1) = pos[1]-D)
            x2_est = p2[0]
            y2_est = p2[1]
            estimates_x.append(x2_est); weights_x.append(W_CAM2_X)
            estimates_y.append(y2_est); weights_y.append(W_CAM2_Y)

    if not estimates_x:
        return None

    wx = np.array(weights_x); wx /= wx.sum()
    wy = np.array(weights_y); wy /= wy.sum()
    x_final = float(np.dot(wx, estimates_x))
    y_final = float(np.dot(wy, estimates_y))

    dbg = {}
    if cam1_det and cam1.k:
        D1, lat1 = cam1.estimate(cam1_det[2], cam1_det[0])
        dbg["cam1_x"] = cam1.to_robot(D1, lat1)[0]
        dbg["cam1_y"] = cam1.to_robot(D1, lat1)[1]
        dbg["cam1_bh"] = cam1_det[2]
    if cam2_det and cam2.k:
        D2, lat2 = cam2.estimate(cam2_det[2], cam2_det[0])
        dbg["cam2_x"] = cam2.to_robot(D2, lat2)[0]
        dbg["cam2_y"] = cam2.to_robot(D2, lat2)[1]
        dbg["cam2_bh"] = cam2_det[2]

    return x_final, y_final, dbg


# ── Calibration mode ──────────────────────────────────────────────────────────
def run_calibration(model):
    cap1 = cv2.VideoCapture(CAM1_INDEX, BACKEND)
    cap2 = cv2.VideoCapture(CAM2_INDEX, BACKEND)
    assert cap1.isOpened() and cap2.isOpened()

    print("\n=== CALIBRATION MODE ===")
    print("Place cube at KNOWN DISTANCE from camera, press SPACE.")
    print("Then type: distance in cm (to near face of cube).")
    print("Calibrate each camera with 1-3 points. Press 'q' to finish.\n")
    print("Keys: 1=calibrate cam1  2=calibrate cam2  q=save&quit\n")

    active_cam = 1

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2: continue

        d1 = detect_best(model, f1)
        d2 = detect_best(model, f2)

        for frame, det, cam, cidx in [(f1,d1,cam1,1),(f2,d2,cam2,2)]:
            if det:
                cx,cy,bh,x1,y1,x2,y2,conf = det
                col = (0,255,0) if cidx==active_cam else (100,100,100)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                k_str = f"k={cam.k:.0f}" if cam.k else "uncal"
                cv2.putText(frame,f"bh={bh}px {k_str}",(x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
            label = f"CAM{cidx} ({'ACTIVE' if cidx==active_cam else 'inactive'})"
            cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,
                        (0,255,255) if cidx==active_cam else (150,150,150),2)
            cv2.putText(frame,"1/2=select  SPACE=capture  q=done",
                        (10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,0),1)

        combined = np.hstack([f1, f2])
        cv2.imshow("Calibration", combined)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('1'):
            active_cam = 1; print("Active: CAM1 (frontal)")
        elif key == ord('2'):
            active_cam = 2; print("Active: CAM2 (side)")
        elif key == ord(' '):
            det = d1 if active_cam == 1 else d2
            if det is None:
                print("  No cube detected!"); continue
            cx,cy,bh = det[0], det[1], det[2]
            print(f"  Captured: bh={bh}px  cx={cx}")
            try:
                d_cm = float(input("  Distance to NEAR face of cube (cm): "))
                cam = cam1 if active_cam == 1 else cam2
                cam.calibrate(bh, cx, d_cm / 100.0)
            except ValueError:
                print("  Invalid input, skipped")

    cap1.release(); cap2.release()
    cv2.destroyAllWindows()
    save_calibration()


# ── Detection mode ────────────────────────────────────────────────────────────
def run_detection(model):
    if not load_calibration():
        print("No calibration found! Run:  python stereo_localize.py calibrate")
        return

    cap1 = cv2.VideoCapture(CAM1_INDEX, BACKEND)
    cap2 = cv2.VideoCapture(CAM2_INDEX, BACKEND)
    assert cap1.isOpened() and cap2.isOpened()
    print("\nRunning stereo detection. 'q' to quit.\n")

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2: continue

        d1 = detect_best(model, f1)
        d2 = detect_best(model, f2)

        c1 = (d1[0], d1[1], d1[2]) if d1 else None
        c2 = (d2[0], d2[1], d2[2]) if d2 else None

        result = fuse(c1, c2)

        for frame, det, label in [(f1, d1, "CAM1 frontal"), (f2, d2, "CAM2 side")]:
            if det:
                _,_,_,x1,y1,x2,y2,conf = det
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        if result:
            x, y, dbg = result
            txt = f"ROBOT: x={x:.3f}  y={y:.3f}  z={GRAB_Z:.3f}"
            cv2.putText(f1, txt, (10, f1.shape[0]-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
            if "cam1_x" in dbg:
                cv2.putText(f1, f"cam1: x={dbg['cam1_x']:.3f} y={dbg['cam1_y']:.3f} bh={dbg['cam1_bh']}",
                            (10, f1.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,150), 1)
            if "cam2_x" in dbg:
                cv2.putText(f2, f"cam2: x={dbg['cam2_x']:.3f} y={dbg['cam2_y']:.3f} bh={dbg['cam2_bh']}",
                            (10, f2.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,150), 1)
            print(f"\r  x={x:.3f} y={y:.3f}  "
                  f"[c1x={dbg.get('cam1_x',0):.3f} c1y={dbg.get('cam1_y',0):.3f} | "
                  f"c2x={dbg.get('cam2_x',0):.3f} c2y={dbg.get('cam2_y',0):.3f}]",
                  end="", flush=True)
        else:
            cv2.putText(f1, "No cube", (10, f1.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Stereo Localize", np.hstack([f1, f2]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release(); cap2.release()
    cv2.destroyAllWindows()
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading YOLO...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(["cube", "wooden block", "small box"])
    print("Ready.\n")

    mode = sys.argv[1] if len(sys.argv) > 1 else "detect"
    if mode == "calibrate":
        run_calibration(model)
    else:
        run_detection(model)
