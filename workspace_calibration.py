"""
Workspace calibration: maps pixel coordinates to robot workspace coordinates.

Two approaches:
1. Manual calibration: move arm to known positions, record joint angles + pixel coords.
2. Simple linear mapping: assume front camera looks at workspace from a fixed position.

For hackathon: we use a simple planar mapping.
The workspace is roughly a plane at table height, and we map pixel (u,v) from
the front camera to (x,y) in the robot base frame.
"""

import json
import cv2
import numpy as np
from pathlib import Path


class WorkspaceCalibration:
    """
    Maps pixel coordinates from front camera to 3D workspace coordinates.

    Uses a set of calibration points (pixel_uv → world_xyz) and fits
    a simple affine transform.
    """

    def __init__(self):
        self.pixel_points = []   # list of (u, v)
        self.world_points = []   # list of (x, y, z)
        self.transform = None    # 3x3 affine matrix (pixel → world XY)
        self.fixed_z = None      # table height Z

    def add_point(self, pixel_uv, world_xyz):
        """Add a calibration point."""
        self.pixel_points.append(list(pixel_uv))
        self.world_points.append(list(world_xyz))

    def compute(self):
        """
        Fit homography from pixel coords to world XY.
        Needs at least 4 calibration points.
        """
        n = len(self.pixel_points)
        if n < 4:
            raise ValueError(f"Need at least 4 calibration points, got {n}")

        px = np.array(self.pixel_points, dtype=np.float64)
        wd = np.array(self.world_points, dtype=np.float64)

        # Fixed Z: median is more robust than mean against outliers
        self.fixed_z = float(np.median(wd[:, 2]))

        # Fit homography: pixel (u,v) → world (x,y)
        # Homography is correct model for perspective camera + flat plane
        self.transform, mask = cv2.findHomography(px, wd[:, :2], cv2.RANSAC, 0.005)

        if self.transform is None:
            raise RuntimeError("Homography fit failed")

        # Compute reprojection error
        ones = np.ones((n, 1))
        px_h = np.hstack([px, ones])
        errors = []
        for i in range(n):
            p = np.array([px[i, 0], px[i, 1], 1.0])
            q = self.transform @ p
            q /= q[2]
            errors.append(np.linalg.norm(q[:2] - wd[i, :2]))
        error = float(np.mean(errors))
        print(f"Calibration fitted with {n} points, mean error: {error:.4f} m")
        return error

    def pixel_to_world(self, u, v):
        """
        Convert pixel coordinates to world coordinates using homography.

        Returns
        -------
        world_xyz : ndarray, shape (3,)
            Position in robot base frame (x,y from homography, z fixed).
        """
        if self.transform is None:
            raise RuntimeError("Calibration not computed yet. Call compute() first.")

        p = np.array([u, v, 1.0], dtype=np.float64)
        q = self.transform @ p
        q /= q[2]
        return np.array([q[0], q[1], self.fixed_z])

    def save(self, path):
        """Save calibration to JSON."""
        data = {
            "pixel_points": self.pixel_points,
            "world_points": self.world_points,
            "fixed_z": self.fixed_z,
            "transform": self.transform.tolist() if self.transform is not None else None,
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Calibration saved to {path}")

    def load(self, path):
        """Load calibration from JSON."""
        data = json.loads(Path(path).read_text())
        self.pixel_points = data["pixel_points"]
        self.world_points = data["world_points"]
        self.fixed_z = data["fixed_z"]
        if data.get("transform"):
            self.transform = np.array(data["transform"])
        print(f"Calibration loaded from {path} ({len(self.pixel_points)} points)")


def interactive_calibrate(camera_index=1, backend=700):
    """
    Interactive calibration procedure.

    Instructions:
    1. Move the robot arm's gripper to a visible position
    2. Click on the gripper tip in the camera view
    3. Enter the world coordinates (read from robot state)
    4. Repeat for at least 4 points at different positions
    5. Press 'q' to finish and save

    In practice for the hackathon:
    - Place the cube at known positions on the table
    - Click the cube center in the camera
    - Enter approximate world coordinates
    """
    import cv2

    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    cal = WorkspaceCalibration()
    click_pos = [None]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)
            print(f"  Clicked pixel: ({x}, {y})")

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    print("=== Workspace Calibration ===")
    print("1. Place cube at a known position on the table")
    print("2. Click on the cube in the camera view")
    print("3. Enter world X Y Z coordinates (meters, space-separated)")
    print("4. Repeat for 4+ positions")
    print("5. Press 'q' to finish\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw existing calibration points
        for px in cal.pixel_points:
            cv2.circle(frame, (int(px[0]), int(px[1])), 5, (0, 255, 0), -1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if click_pos[0] is not None:
            u, v = click_pos[0]
            click_pos[0] = None
            try:
                coords_str = input("  Enter world X Y Z (meters): ")
                xyz = [float(x) for x in coords_str.split()]
                if len(xyz) == 3:
                    cal.add_point((u, v), xyz)
                    print(f"  Added point #{len(cal.pixel_points)}: pixel=({u},{v}) world={xyz}")
                elif len(xyz) == 2:
                    # If only X Y given, use default Z (table height)
                    cal.add_point((u, v), [xyz[0], xyz[1], 0.0])
                    print(f"  Added point #{len(cal.pixel_points)}: pixel=({u},{v}) world={xyz + [0.0]}")
                else:
                    print("  Need 2 or 3 coordinates")
            except (ValueError, EOFError):
                print("  Invalid input, skipping")

    cap.release()
    cv2.destroyAllWindows()

    if len(cal.pixel_points) >= 3:
        cal.compute()
        save_path = str(Path(__file__).parent / "calibration.json")
        cal.save(save_path)
    else:
        print("Not enough points for calibration")

    return cal


if __name__ == "__main__":
    interactive_calibrate()
