"""
Geometric pixel → robot frame coordinate transform.

Camera geometry (measured):
  - Camera is 50cm in front of robot base along X axis: cam_x = 0.5m
  - Camera is 5cm above table: cam_z = table_z + 0.05
  - Camera looks toward robot (-X direction)
  - Table height in robot frame: z = -0.035m

Camera intrinsics (DEXP DQ4M3FA1, 640x480, 120° diagonal FOV):
  - diagonal_px = sqrt(640²+480²) = 800
  - f = (800/2) / tan(60°) ≈ 231 px
  - cx=320, cy=240

Method: ray-plane intersection.
  pixel → normalized ray in camera → rotate to robot frame → intersect z=table plane
"""

import numpy as np

# ── Camera intrinsics ─────────────────────────────────────────────────────────
IMG_W, IMG_H = 640, 480
FOV_DIAG_DEG = 120.0

diag_px = np.sqrt(IMG_W**2 + IMG_H**2)          # 800 px
f_px    = (diag_px / 2) / np.tan(np.radians(FOV_DIAG_DEG / 2))  # 231 px
K = np.array([[f_px,   0,  IMG_W/2],
              [0,   f_px,  IMG_H/2],
              [0,      0,        1]], dtype=np.float64)

# ── Camera extrinsics (camera pose in robot base frame) ───────────────────────
TABLE_Z = -0.035   # table height in robot frame (m)
CAM_H   =  0.05   # camera height above table (m) — measured 5cm

# Camera 1 (index 0): 50cm in front of robot, centered
CAM1_POS = np.array([0.50, 0.00, TABLE_Z + CAM_H])

# Camera 2 (index 2): same depth as cam1 (50cm), offset 25cm laterally
# +Y = left of robot / right when looking from camera toward robot
# -Y = right of robot / left when looking from camera toward robot
CAM2_H   = 0.03        # camera 2 is 3cm above table
CAM2_POS = np.array([0.50, +0.25, TABLE_Z + CAM2_H])
# ↑ If cube Y coords are mirrored, flip sign: np.array([0.50, +0.25, ...])

CAM_POS = CAM1_POS  # default (used by pixel_to_robot single-camera)

# Both cameras look toward robot (-X direction)
# Camera axes in robot frame:
#   z_cam = (-1, 0, 0)  camera looks in -X_robot
#   x_cam = ( 0,-1, 0)  camera right = -Y_robot
#   y_cam = ( 0, 0,-1)  camera down  = -Z_robot
R_cam2robot = np.array([
    [ 0., -1.,  0.],
    [ 0.,  0., -1.],
    [-1.,  0.,  0.],
]).T


def ray_in_robot(u, v, cam_pos):
    """Return (origin, direction) of camera ray in robot frame."""
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    direction = R_cam2robot @ ray_cam
    return cam_pos.copy(), direction


def triangulate(u1, v1, u2, v2):
    """
    Stereo triangulation: find 3D point from two pixel observations.
    Uses least-squares closest point between two rays.

    Camera 1 = CAM1_POS, Camera 2 = CAM2_POS (both looking in -X direction).
    Returns xyz in robot frame (meters).
    """
    o1, d1 = ray_in_robot(u1, v1, CAM1_POS)
    o2, d2 = ray_in_robot(u2, v2, CAM2_POS)

    # Solve: o1 + t1*d1 ≈ o2 + t2*d2  (least squares)
    A = np.stack([d1, -d2], axis=1)  # (3,2)
    b = o2 - o1
    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t1, t2 = t

    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    # Weighted average: cam1 is more trusted (closer to robot, known geometry)
    w1, w2 = 0.85, 0.15
    return w1 * p1 + w2 * p2


def pixel_to_robot(u, v, target_z=TABLE_Z):
    """
    Convert pixel (u, v) to 3D position in robot base frame.

    Assumes the target point lies on the horizontal plane z = target_z.

    Parameters
    ----------
    u, v       : pixel coordinates
    target_z   : known Z of target in robot frame (default: table level)

    Returns
    -------
    xyz : np.array([x, y, z]) in robot base frame (meters)
    """
    # 1. Back-project pixel to normalized image coordinates
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])

    # 2. Rotate ray to robot frame
    ray_robot = R_cam2robot @ ray_cam   # direction in robot frame

    # 3. Ray-plane intersection: CAM_POS + t * ray_robot, solve for z = target_z
    if abs(ray_robot[2]) < 1e-9:
        raise ValueError("Ray is parallel to the table plane — check camera tilt")

    cam_z = CAM_POS[2]
    t = (target_z - cam_z) / ray_robot[2]
    if t < 0:
        raise ValueError(f"Intersection is behind camera (t={t:.3f})")

    point = CAM_POS + t * ray_robot
    return point   # [x, y, target_z]


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"K =\n{K}\n")
    print(f"Camera position in robot frame: {CAM_POS}")
    print(f"Table z = {TABLE_Z} m\n")

    # Robot base (0,0,0) projected to pixel — just to understand geometry
    # Point on table directly below robot base: (0, 0, TABLE_Z)
    test_pts = [
        (0.05,  0.00, TABLE_Z, "cube 5cm from robot, center"),
        (0.10,  0.00, TABLE_Z, "cube 10cm from robot, center"),
        (0.15,  0.00, TABLE_Z, "cube 15cm from robot, center"),
        (0.10,  0.05, TABLE_Z, "cube 10cm from robot, 5cm left"),
        (0.10, -0.05, TABLE_Z, "cube 10cm from robot, 5cm right"),
    ]

    print("Forward check (robot XYZ → pixel):")
    for x, y, z, label in test_pts:
        # Transform to camera frame
        p_robot = np.array([x, y, z])
        p_cam   = np.linalg.inv(R_cam2robot) @ (p_robot - CAM_POS)
        if p_cam[2] <= 0:
            print(f"  {label}: BEHIND camera")
            continue
        u = K[0,0] * p_cam[0] / p_cam[2] + K[0,2]
        v = K[1,1] * p_cam[1] / p_cam[2] + K[1,2]
        print(f"  {label}: pixel=({u:.0f}, {v:.0f})")

    print("\nInverse check (pixel → robot XYZ):")
    for x, y, z, label in test_pts:
        p_robot = np.array([x, y, z])
        p_cam   = np.linalg.inv(R_cam2robot) @ (p_robot - CAM_POS)
        if p_cam[2] <= 0:
            continue
        u = K[0,0] * p_cam[0] / p_cam[2] + K[0,2]
        v = K[1,1] * p_cam[1] / p_cam[2] + K[1,2]
        recovered = pixel_to_robot(u, v, TABLE_Z)
        err = np.linalg.norm(recovered - p_robot)
        print(f"  {label}: error={err*1000:.2f}mm  recovered=({recovered[0]:.3f}, {recovered[1]:.3f}, {recovered[2]:.3f})")
