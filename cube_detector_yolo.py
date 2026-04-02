"""
Cube detector using YOLO-World (zero-shot, no training needed).
Detects cube by text description.

pip install ultralytics
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")

CAMERA_INDEX   = 0
CAMERA_BACKEND = 700
CAL_PATH       = r"c:\DISK Z\Hakaton_HSE\cv_ik_pipeline\calibration.json"

# Text prompts — YOLO-World finds these objects in the frame
CLASSES = ["cube", "wooden block", "small box", "cardboard box"]
CONF    = 0.05   # low threshold — raise if too many false positives


def load_model():
    from ultralytics import YOLOWorld
    print("Loading YOLO-World model (downloads ~30MB on first run)...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(CLASSES)
    print(f"Model ready. Searching for: {CLASSES}")
    return model


def detect_cube(model, frame):
    """Returns {cx, cy, w, h, conf} of best detection, or None."""
    results = model.predict(frame, conf=CONF, verbose=False, imgsz=640)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Pick highest confidence detection
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    conf = float(boxes.conf[best_idx])
    cls  = int(boxes.cls[best_idx])

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    w  = int(x2 - x1)
    h  = int(y2 - y1)

    return {"cx": cx, "cy": cy, "w": w, "h": h,
            "conf": conf, "cls": CLASSES[cls] if cls < len(CLASSES) else "?",
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}


def main():
    from workspace_calibration import WorkspaceCalibration

    model = load_model()

    cal = WorkspaceCalibration()
    try:
        cal.load(CAL_PATH)
        has_cal = True
        print("Calibration loaded.")
    except Exception:
        has_cal = False
        print("No calibration — showing pixels only.")

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    assert cap.isOpened(), "Cannot open camera"

    print("\nRunning. Press 'q' to quit, '+'/'-' to adjust confidence.\n")
    conf = CONF

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result = detect_cube(model, frame)
        h_f, w_f = frame.shape[:2]

        if result:
            cv2.rectangle(frame, (result["x1"], result["y1"]),
                                 (result["x2"], result["y2"]), (0, 255, 0), 2)
            cv2.circle(frame, (result["cx"], result["cy"]), 5, (0, 0, 255), -1)

            label = f"{result['cls']} {result['conf']:.2f}"
            cv2.putText(frame, label, (result["x1"], result["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if has_cal:
                wx, wy, wz = cal.pixel_to_world(result["cx"], result["cy"])
                world_txt = f"x={wx:.3f} y={wy:.3f} z={wz:.3f}m"
                cv2.putText(frame, world_txt, (10, h_f - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"\r  {label}  px=({result['cx']},{result['cy']})  {world_txt}   ",
                      end="", flush=True)
        else:
            cv2.putText(frame, "No cube", (10, h_f - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"conf>{conf:.2f}  +/- to adjust", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("YOLO-World Cube Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            conf = min(0.9, conf + 0.05)
            model.predictor = None  # reset predictor to apply new conf
            print(f"\n  conf -> {conf:.2f}")
        elif key == ord('-'):
            conf = max(0.01, conf - 0.05)
            model.predictor = None
            print(f"\n  conf -> {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
