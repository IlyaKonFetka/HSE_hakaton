"""
Single-camera cube localization via ray–plane intersection.

Camera 1 (frontal):
  - Position in robot frame: (0.50, 0.00, CAM_Z)
  - Looks in -X direction (toward robot base)
  - Image right (+u) = -Y_robot
  - Image down  (+v) = -Z_robot

Focal length calibrated empirically:
  k = h_px * D_center  (similar triangles constant, measured ~31)
  f_px = k / cube_side = 31 / 0.05 = 620 px
"""

import numpy as np

# ── Parameters (adjust if camera moves) ──────────────────────────────────────
CAM_POS  = np.array([0.50, 0.00, -0.020])   # camera position in robot frame (m)
TABLE_Z  = -0.035                             # table surface in robot frame (m)
GRAB_Z   = -0.010                             # grab height = table + half cube

F_PX     = 620.0    # focal length in pixels (from calibration: k=31, cube=5cm)
CX, CY   = 320.0, 240.0    # image principal point (640×480 camera)


def pixel_to_robot(u, v, target_z=TABLE_Z):
    """
    Back-project pixel (u, v) onto horizontal plane z = target_z.

    Returns np.array([x, y, z]) in robot base frame (meters).
    Raises ValueError if ray is parallel to the plane.
    """
    # Ray direction in robot frame:
    #   forward = -X_robot,  right = -Y_robot,  down = -Z_robot
    ray = np.array([
        -1.0,                  # -X  (camera looks toward robot)
        -(u - CX) / F_PX,     # -Y  (right pixel → negative Y)
        -(v - CY) / F_PX,     # -Z  (down pixel  → negative Z)
    ])

    if abs(ray[2]) < 1e-9:
        raise ValueError("Ray is parallel to the table plane (pixel at image horizon)")

    # Solve: CAM_POS + t * ray  s.t.  z-component = target_z
    t = (target_z - CAM_POS[2]) / ray[2]
    if t < 0:
        raise ValueError(f"Intersection behind camera (t={t:.3f})")

    return CAM_POS + t * ray


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"CAM_POS = {CAM_POS}")
    print(f"F_PX    = {F_PX}")
    print(f"TABLE_Z = {TABLE_Z}\n")

    # Forward check: known robot positions → expected pixels
    test_pts = [
        (0.23,  0.00, "cube 27cm from cam, center"),
        (0.15,  0.00, "cube 35cm from cam, center"),
        (0.23,  0.10, "cube 27cm from cam, 10cm left"),
        (0.23, -0.10, "cube 27cm from cam, 10cm right"),
        (0.30,  0.05, "cube 20cm from cam, 5cm left"),
    ]

    print("Forward (robot XYZ → pixel):")
    for x, y, label in test_pts:
        p = np.array([x, y, TABLE_Z])
        # Project to camera: delta in robot frame
        dx = p - CAM_POS   # vector from cam to point
        # In camera frame: forward=-X, right=-Y, down=-Z
        fwd  = -dx[0]
        rgt  = -dx[1]
        dwn  = -dx[2]
        u = CX + F_PX * rgt / fwd
        v = CY + F_PX * dwn / fwd
        print(f"  {label:35s}  pixel=({u:.0f}, {v:.0f})")

    print("\nInverse (pixel → robot XYZ):")
    for x, y, label in test_pts:
        p = np.array([x, y, TABLE_Z])
        dx = p - CAM_POS
        fwd = -dx[0]; rgt = -dx[1]; dwn = -dx[2]
        u = CX + F_PX * rgt / fwd
        v = CY + F_PX * dwn / fwd
        recovered = pixel_to_robot(u, v)
        err = np.linalg.norm(recovered - p)
        print(f"  {label:35s}  err={err*1000:.2f}mm  "
              f"got=({recovered[0]:.3f}, {recovered[1]:.3f}, {recovered[2]:.3f})")
