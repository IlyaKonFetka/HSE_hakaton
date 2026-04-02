"""
Cube detector v2: HSV mask + contour quad detection + optional Hough lines.

Pipeline:
  1. Loose HSV mask to suppress dynamic background
  2. Canny edges on masked region
  3. Find contours → approxPolyDP → keep 4-sided convex quads
  4. Score by squareness + solidity + area
  5. (optional) HoughLinesP confirmation

Run standalone to tune parameters interactively with trackbars.
"""

import cv2
import numpy as np

# ── Default parameters (tune via trackbars in standalone mode) ────────────────
DEFAULT_PARAMS = dict(
    # HSV range (wide by default — cube is light/beige)
    h_lo=0,   h_hi=40,
    s_lo=0,   s_hi=80,    # low saturation = desaturated/beige colors
    v_lo=160, v_hi=255,
    # Edge detection
    canny_lo=20, canny_hi=80,
    # Shape
    min_area=800,
    max_area=80000,
    min_squareness=40,   # percent (0-100)
    poly_eps=4,          # approxPolyDP epsilon (% of perimeter)
    roi_top=35,          # ignore top X% of frame
)

_params = dict(DEFAULT_PARAMS)


def set_params(**kwargs):
    _params.update(kwargs)


def detect_cube(frame, draw=False, use_hough=False):
    """
    Returns dict {cx, cy, w, h, area, quad} or None.
    quad: 4 corner points of detected rectangle (in full-frame coords).
    """
    h_frame, w_frame = frame.shape[:2]
    roi_top = int(h_frame * _params["roi_top"] / 100)

    roi = frame[roi_top:].copy()

    # ── 1. HSV color mask ─────────────────────────────────────────────────────
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lo = np.array([_params["h_lo"], _params["s_lo"], _params["v_lo"]])
    hi = np.array([_params["h_hi"], _params["s_hi"], _params["v_hi"]])
    color_mask = cv2.inRange(hsv, lo, hi)

    # Morphological cleanup
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,  k3)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k7)

    # ── 2. Canny edges on the masked region ───────────────────────────────────
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=color_mask)
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)
    edges = cv2.Canny(blurred, _params["canny_lo"], _params["canny_hi"])

    # Also run Canny on full gray (not just mask) and combine
    edges_full = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0),
                           _params["canny_lo"], _params["canny_hi"])
    # Keep only edges where color_mask is active (or very nearby)
    color_dilated = cv2.dilate(color_mask, k7, iterations=2)
    edges_full = cv2.bitwise_and(edges_full, edges_full, mask=color_dilated)
    edges = cv2.bitwise_or(edges, edges_full)

    edges = cv2.dilate(edges, k3, iterations=1)

    # ── 3. Hough lines (optional — helps confirm straight edges) ─────────────
    hough_mask = np.zeros_like(edges)
    if use_hough:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=30, minLineLength=20, maxLineGap=10)
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)
        edges = cv2.bitwise_or(edges, hough_mask)

    # ── 4. Find contours → look for 4-sided convex quads ─────────────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = _params["min_area"]
    max_area = _params["max_area"]
    min_sq   = _params["min_squareness"] / 100.0
    eps_pct  = _params["poly_eps"] / 100.0

    best = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps_pct * peri, True)

        # Accept 4-sided shapes; also try bounding rect as fallback
        if len(approx) == 4:
            quad = approx
        elif 3 <= len(approx) <= 6:
            # Fall back to bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            quad = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
        else:
            continue

        if not cv2.isContourConvex(quad):
            # Try convex hull
            quad = cv2.convexHull(quad)
            if len(quad) != 4:
                continue

        x, y, w, h = cv2.boundingRect(quad)
        if w == 0 or h == 0:
            continue

        squareness = min(w, h) / max(w, h)
        if squareness < min_sq:
            continue

        # Color consistency: fraction of quad interior that matches color mask
        qmask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(qmask, [quad], 0, 255, -1)
        color_inside = cv2.bitwise_and(color_mask, color_mask, mask=qmask)
        quad_px = cv2.countNonZero(qmask)
        color_frac = cv2.countNonZero(color_inside) / quad_px if quad_px > 0 else 0

        score = area * squareness * (0.3 + 0.7 * color_frac)

        if score > best_score:
            best_score = score
            cx = x + w // 2
            cy = y + h // 2
            quad_full = quad.copy()
            quad_full[:, :, 1] += roi_top
            best = {
                "cx": cx, "cy": cy + roi_top,
                "w": w, "h": h, "area": area,
                "quad": quad_full,
                "color_frac": color_frac,
                "squareness": squareness,
            }

    if draw:
        # Debug: small edge/mask preview in corner
        h_roi = h_frame - roi_top
        preview_w = w_frame // 5
        preview_h = int(h_roi * preview_w / w_frame)

        mask_small = cv2.resize(color_mask, (preview_w, preview_h))
        edge_small = cv2.resize(edges,      (preview_w, preview_h))
        combo = np.zeros((preview_h, preview_w * 2, 3), dtype=np.uint8)
        combo[:, :preview_w] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        combo[:, preview_w:] = cv2.cvtColor(edge_small, cv2.COLOR_GRAY2BGR)
        cv2.putText(combo, "mask | edges", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
        frame[roi_top:roi_top + preview_h, :preview_w * 2] = combo

        if best:
            cv2.drawContours(frame, [best["quad"]], 0, (0, 255, 0), 2)
            cv2.circle(frame, (best["cx"], best["cy"]), 6, (0, 0, 255), -1)
            label = (f"sq={best['squareness']:.2f} "
                     f"col={best['color_frac']:.2f} "
                     f"a={best['area']:.0f}")
            cv2.putText(frame, label, (best["cx"] - 60, best["cy"] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return best


# ── Standalone interactive tuner ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
    from workspace_calibration import WorkspaceCalibration

    CAMERA_INDEX   = 0
    CAMERA_BACKEND = 700

    cal = WorkspaceCalibration()
    try:
        cal.load(r"c:\DISK Z\Hakaton_HSE\cv_ik_pipeline\calibration.json")
        has_cal = True
        print("Calibration loaded.")
    except Exception:
        has_cal = False

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    assert cap.isOpened(), "Cannot open camera"

    WIN = "Cube Detector v2"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # Trackbars
    def tb(name, val, mx): cv2.createTrackbar(name, WIN, val, mx, lambda x: None)

    tb("H lo",          _params["h_lo"],           179)
    tb("H hi",          _params["h_hi"],           179)
    tb("S lo",          _params["s_lo"],           255)
    tb("S hi",          _params["s_hi"],           255)
    tb("V lo",          _params["v_lo"],           255)
    tb("V hi",          _params["v_hi"],           255)
    tb("Canny lo",      _params["canny_lo"],       200)
    tb("Canny hi",      _params["canny_hi"],       200)
    tb("Min area/100",  _params["min_area"] // 100, 100)
    tb("Squareness%",   _params["min_squareness"],  100)
    tb("Poly eps%",     _params["poly_eps"],         20)
    tb("ROI top%",      _params["roi_top"],          80)
    tb("Use Hough",     0,                            1)

    def read_params():
        _params["h_lo"]           = cv2.getTrackbarPos("H lo",         WIN)
        _params["h_hi"]           = cv2.getTrackbarPos("H hi",         WIN)
        _params["s_lo"]           = cv2.getTrackbarPos("S lo",         WIN)
        _params["s_hi"]           = cv2.getTrackbarPos("S hi",         WIN)
        _params["v_lo"]           = cv2.getTrackbarPos("V lo",         WIN)
        _params["v_hi"]           = cv2.getTrackbarPos("V hi",         WIN)
        _params["canny_lo"]       = cv2.getTrackbarPos("Canny lo",     WIN)
        _params["canny_hi"]       = cv2.getTrackbarPos("Canny hi",     WIN)
        _params["min_area"]       = cv2.getTrackbarPos("Min area/100", WIN) * 100 + 100
        _params["min_squareness"] = cv2.getTrackbarPos("Squareness%",  WIN)
        _params["poly_eps"]       = cv2.getTrackbarPos("Poly eps%",    WIN) + 1
        _params["roi_top"]        = cv2.getTrackbarPos("ROI top%",     WIN)

    use_hough = False
    print("\nTrackbars to tune. Press 'h'=toggle Hough, 's'=save params, 'q'=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        read_params()
        use_hough = bool(cv2.getTrackbarPos("Use Hough", WIN))

        result = detect_cube(frame, draw=True, use_hough=use_hough)

        h, w = frame.shape[:2]
        if result:
            cx, cy = result["cx"], result["cy"]
            if has_cal:
                wx, wy, wz = cal.pixel_to_world(cx, cy)
                txt = f"x={wx:.3f} y={wy:.3f} z={wz:.3f}m"
                cv2.putText(frame, txt, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"\r  px=({cx},{cy})  {txt}   ", end="", flush=True)
        else:
            cv2.putText(frame, "No cube detected", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\nSaved params: {_params}")

    cap.release()
    cv2.destroyAllWindows()
    print()
