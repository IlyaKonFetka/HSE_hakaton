"""
Cube detection using OpenCV color segmentation.

Detects colored objects (cube, ball) via HSV thresholding.
Returns bounding box center in pixel coordinates.
"""

import cv2
import numpy as np

# Pre-defined HSV ranges for common object colors.
# Each entry: (lower_hsv, upper_hsv)
# Tune these by running calibrate_hsv() interactively.
COLOR_PRESETS = {
    "red1":   ((0, 100, 80),   (10, 255, 255)),
    "red2":   ((160, 100, 80), (180, 255, 255)),
    "orange": ((10, 100, 100), (25, 255, 255)),
    "yellow": ((25, 100, 100), (35, 255, 255)),
    "green":  ((35, 80, 80),   (85, 255, 255)),
    "blue":   ((90, 80, 80),   (130, 255, 255)),
    "brown":  ((5, 50, 50),    (20, 200, 200)),
    # Calibrated at hackathon venue:
    "cube_front":  ((0, 27, 215),  (38, 255, 233)),   # camera 0
    "cube_side":   ((17, 16, 208), (29, 44, 228)),     # camera 2
}

# ROI: crop to bottom N% of frame (where the table is)
# Set to 1.0 to use full frame.
ROI_TOP_FRACTION = 0.45   # ignore top 45% of frame (background/ceiling)


def detect_object(
    frame_bgr,
    color_ranges=None,
    min_area=500,
    max_area=80000,
    draw=False,
    roi_top_fraction=ROI_TOP_FRACTION,
):
    """
    Detect the largest colored object in the frame.

    Parameters
    ----------
    frame_bgr : ndarray (H, W, 3) BGR
        Input image from OpenCV camera.
    color_ranges : list of (lower_hsv, upper_hsv) tuples
        HSV ranges to threshold. Defaults to brown/red (wooden cube).
    min_area : int
        Minimum contour area in pixels.
    max_area : int
        Maximum contour area in pixels.
    draw : bool
        If True, draw detection on frame (modifies in-place).

    Returns
    -------
    detection : dict or None
        {"cx": int, "cy": int, "w": int, "h": int, "area": int}
        Pixel coordinates of object center, or None if not found.
    """
    if color_ranges is None:
        color_ranges = [
            COLOR_PRESETS["cube_front"],
            COLOR_PRESETS["cube_side"],
        ]

    # Crop to ROI (bottom portion = table workspace)
    h_full, w_full = frame_bgr.shape[:2]
    roi_y = int(h_full * roi_top_fraction)
    roi_frame = frame_bgr[roi_y:, :]

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Combine masks from all color ranges
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area and area > best_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 0.3 < aspect < 3.0:
                # Convert back to full-frame coordinates
                best = {
                    "cx": x + w // 2,
                    "cy": roi_y + y + h // 2,   # add ROI offset
                    "w": w, "h": h, "area": area,
                    "cx_roi": x + w // 2,
                    "cy_roi": y + h // 2,
                }
                best_area = area

    if draw and best is not None:
        cx, cy, w, h = best["cx"], best["cy"], best["w"], best["h"]
        cv2.rectangle(frame_bgr, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 255, 0), 2)
        cv2.circle(frame_bgr, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame_bgr, f"({cx},{cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw ROI boundary
        cv2.line(frame_bgr, (0, roi_y), (w_full, roi_y), (255, 0, 0), 1)

    return best


def calibrate_hsv(camera_index=1, backend=700):
    """
    Interactive HSV calibration tool.
    Opens a window with trackbars to adjust HSV thresholds.
    Press 'q' to quit and print the selected range.
    """
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    cv2.namedWindow("HSV Calibration")
    cv2.createTrackbar("H_lo", "HSV Calibration", 0, 180, lambda x: None)
    cv2.createTrackbar("S_lo", "HSV Calibration", 50, 255, lambda x: None)
    cv2.createTrackbar("V_lo", "HSV Calibration", 50, 255, lambda x: None)
    cv2.createTrackbar("H_hi", "HSV Calibration", 30, 180, lambda x: None)
    cv2.createTrackbar("S_hi", "HSV Calibration", 255, 255, lambda x: None)
    cv2.createTrackbar("V_hi", "HSV Calibration", 255, 255, lambda x: None)

    print("Adjust sliders. Press 'q' to quit and save values.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_lo = cv2.getTrackbarPos("H_lo", "HSV Calibration")
        s_lo = cv2.getTrackbarPos("S_lo", "HSV Calibration")
        v_lo = cv2.getTrackbarPos("V_lo", "HSV Calibration")
        h_hi = cv2.getTrackbarPos("H_hi", "HSV Calibration")
        s_hi = cv2.getTrackbarPos("S_hi", "HSV Calibration")
        v_hi = cv2.getTrackbarPos("V_hi", "HSV Calibration")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]), np.array([h_hi, s_hi, v_hi]))

        result = cv2.bitwise_and(frame, frame, mask=mask)

        display = np.hstack([frame, result])
        cv2.imshow("HSV Calibration", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\nSelected HSV range:")
            print(f"  Lower: ({h_lo}, {s_lo}, {v_lo})")
            print(f"  Upper: ({h_hi}, {s_hi}, {v_hi})")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        cam_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        calibrate_hsv(camera_index=cam_idx)
    else:
        # Quick test: grab frame and detect
        cap = cv2.VideoCapture(1, 700)
        ret, frame = cap.read()
        cap.release()
        if ret:
            det = detect_object(frame, draw=True)
            print(f"Detection: {det}")
            cv2.imshow("Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
