"""
Step 2: Convert YOLO pixel detection → 3D robot frame coordinates.

Method: ray-plane intersection.
  - Camera ray through detected pixel (using K)
  - Intersect with table plane (known Z in robot frame)
  - Transform camera → robot frame using extrinsics (R, t)

Requires:
  - camera_params.json  (from camera_calibration.py)
  - calibration.json    (4+ points: pixel + robot TCP xyz)

The 4 calibration points are used via solvePnP to estimate camera pose
relative to robot base frame. No additional calibration step needed.
"""

import json
import numpy as np
import cv2
from pathlib import Path


TABLE_Z_ROBOT = -0.035   # table height in robot base frame (meters)
                          # measured: plate z = -0.035m ≈ table level


class CubeLocalizer:
    """
    Converts pixel (u,v) to robot frame (x,y,z) using camera model.
    """

    def __init__(self, camera_params_path, workspace_cal_path):
        self.K    = None
        self.dist = None
        self.R_cam2robot = None   # 3x3 rotation: camera → robot
        self.t_cam2robot = None   # 3x1 translation: camera → robot
        self.table_z = TABLE_Z_ROBOT
        self._load(camera_params_path, workspace_cal_path)

    def _load(self, cam_path, cal_path):
        # Camera intrinsics
        cam = json.loads(Path(cam_path).read_text())
        self.K    = np.array(cam["K"],    dtype=np.float64)
        self.dist = np.array(cam["dist"], dtype=np.float64)
        print(f"Camera K loaded, rms={cam['rms_error']:.3f}px")

        # Workspace calibration points: pixel → robot TCP xyz
        cal = json.loads(Path(cal_path).read_text())
        pixel_pts = np.array(cal["pixel_points"], dtype=np.float64)   # (N,2)
        world_pts = np.array(cal["world_points"],  dtype=np.float64)  # (N,3) in meters

        # solvePnP: find camera pose such that K * [R|t] * world_pt = pixel_pt
        # world_pts are in robot frame (meters → mm for solvePnP convention)
        # Note: solvePnP gives robot→camera transform, we invert it
        world_mm = world_pts * 1000.0   # meters → mm (same units as square_mm)

        ok, rvec, tvec = cv2.solvePnP(
            world_mm.reshape(-1, 1, 3),
            pixel_pts.reshape(-1, 1, 2),
            self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError("solvePnP failed — not enough calibration points")

        # rvec/tvec: transform from robot(world) frame to camera frame
        R_robot2cam, _ = cv2.Rodrigues(rvec)
        t_robot2cam = tvec.flatten()

        # Invert to get camera→robot
        self.R_cam2robot = R_robot2cam.T
        self.t_cam2robot = -R_robot2cam.T @ t_robot2cam

        # Verify reprojection
        errors = []
        for i in range(len(pixel_pts)):
            proj, _ = cv2.projectPoints(
                world_mm[i].reshape(1,1,3), rvec, tvec, self.K, self.dist)
            err = np.linalg.norm(proj.flatten() - pixel_pts[i])
            errors.append(err)
        print(f"PnP reprojection errors: "
              f"{np.mean(errors):.1f}px mean, {np.max(errors):.1f}px max")

    def pixel_to_robot(self, u, v, z_robot=None):
        """
        Convert pixel (u,v) to 3D position in robot base frame.

        z_robot: known Z of the target in robot frame (default: table height).
                 e.g. TABLE_Z_ROBOT + cube_height/2 for cube center.

        Returns np.array([x, y, z]) in meters.
        """
        if z_robot is None:
            z_robot = self.table_z

        # 1. Undistort pixel
        pt = cv2.undistortPoints(
            np.array([[[u, v]]], dtype=np.float64), self.K, self.dist, P=self.K)
        u_ud, v_ud = pt[0, 0]

        # 2. Back-project to normalized camera ray (direction in camera frame)
        ray_cam = np.linalg.inv(self.K) @ np.array([u_ud, v_ud, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # 3. Ray in robot frame
        ray_robot = self.R_cam2robot @ ray_cam
        cam_origin_robot = self.t_cam2robot  # camera center in robot frame (mm)
        cam_origin_m = cam_origin_robot / 1000.0  # convert mm → m

        # 4. Intersect ray with horizontal plane z = z_robot
        # cam_origin_m + t * ray_robot  →  z = z_robot
        if abs(ray_robot[2]) < 1e-6:
            raise ValueError("Ray is parallel to table plane")
        t = (z_robot - cam_origin_m[2]) / ray_robot[2]
        intersection = cam_origin_m + t * ray_robot

        return intersection  # [x, y, z_robot]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

    CAM_PARAMS = Path(__file__).parent / "camera_params.json"
    CAL_PATH   = Path(__file__).parent / "calibration.json"
    CAMERA_INDEX   = 0
    CAMERA_BACKEND = 700

    if not CAM_PARAMS.exists():
        print(f"No camera_params.json found.")
        print("Run: python camera_calibration.py first")
        sys.exit(1)

    localizer = CubeLocalizer(str(CAM_PARAMS), str(CAL_PATH))

    # Load YOLO detector
    from ultralytics import YOLOWorld
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(["cube", "wooden block", "small box"])

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    assert cap.isOpened()

    print("\nRunning. 'q' = quit\n")

    # Allow setting cube height above table
    cube_half_h = 0.0  # add half cube height if you want cube center Z

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame, conf=0.05, verbose=False)
        boxes = results[0].boxes
        h_f = frame.shape[0]

        if boxes is not None and len(boxes) > 0:
            best = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            conf = float(boxes.conf[best])

            try:
                z_grab = localizer.table_z + cube_half_h
                pos = localizer.pixel_to_robot(cx, cy, z_robot=z_grab)
                x, y, z = pos

                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"x={x:.3f} y={y:.3f} z={z:.3f}m  conf={conf:.2f}",
                            (10, h_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                print(f"\r  px=({cx},{cy})  robot: x={x:.3f} y={y:.3f} z={z:.3f}m",
                      end="", flush=True)
            except Exception as e:
                cv2.putText(frame, f"Error: {e}", (10, h_f - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        else:
            cv2.putText(frame, "No cube", (10, h_f - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Cube Localizer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()
