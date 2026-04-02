"""
Step 1: Camera intrinsic calibration using a checkerboard.

1. Print checkerboard_9x6.png (generated below) on A4
2. Run this script
3. Move the checkerboard in front of camera, covering different angles/positions
4. Press SPACE to capture a frame, collect 15-20 frames
5. Press 'c' to calibrate, 'q' to quit

Result saved to camera_params.json: K matrix + distortion coefficients.
"""

import cv2
import numpy as np
import json
from pathlib import Path

CAMERA_INDEX   = 0
CAMERA_BACKEND = 700

BOARD_W = 9   # inner corners per row
BOARD_H = 6   # inner corners per col
SQUARE_MM = 25.0  # physical square size in mm (measure after printing!)

OUT_PATH = Path(__file__).parent / "camera_params.json"


def generate_checkerboard():
    """Save a printable checkerboard PNG."""
    sq = 80  # pixels per square in the saved image
    rows, cols = BOARD_H + 1, BOARD_W + 1
    img = np.zeros((rows * sq, cols * sq), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = 255
    p = Path(__file__).parent / "checkerboard_print.png"
    cv2.imwrite(str(p), img)
    print(f"Checkerboard saved to {p} — print this on A4")


def run_calibration():
    generate_checkerboard()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
    objp *= SQUARE_MM  # in mm

    obj_points = []
    img_points = []

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    assert cap.isOpened(), "Cannot open camera"

    n_captured = 0
    print(f"\nCalibrating camera. Board: {BOARD_W}x{BOARD_H} inner corners")
    print("SPACE = capture frame | c = compute calibration | q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)

        display = frame.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (BOARD_W, BOARD_H), corners2, found)
            cv2.putText(display, "BOARD FOUND — press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Point camera at checkerboard",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {n_captured}  (need 15+)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        cv2.imshow("Camera Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') and found:
            obj_points.append(objp)
            img_points.append(corners2)
            n_captured += 1
            print(f"  Captured #{n_captured}")
        elif key == ord('c'):
            if n_captured < 6:
                print(f"Need at least 6 frames, have {n_captured}")
                continue
            print(f"\nComputing calibration from {n_captured} frames...")
            h, w = gray.shape
            ret_val, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, (w, h), None, None)
            print(f"Reprojection error: {ret_val:.3f} px (good if < 1.0)")
            print(f"K =\n{K}")

            # Save
            data = {
                "K": K.tolist(),
                "dist": dist.tolist(),
                "rms_error": float(ret_val),
                "image_size": [w, h],
                "board": {"w": BOARD_W, "h": BOARD_H, "square_mm": SQUARE_MM},
            }
            OUT_PATH.write_text(json.dumps(data, indent=2))
            print(f"\nSaved to {OUT_PATH}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
