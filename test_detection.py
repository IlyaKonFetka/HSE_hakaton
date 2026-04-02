"""
Test module 1: cube detection + coordinate mapping.
Shows live camera with detected cube and world coordinates.
Press 'q' to quit.
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, r"c:\DISK Z\Hakaton_HSE\lerobot\src")
from cube_detector import detect_object
from workspace_calibration import WorkspaceCalibration
from so101_kinematics import inverse_kinematics_deg

CAMERA_INDEX = 0
CAMERA_BACKEND = 700  # CAP_DSHOW
CAL_PATH = r"c:\DISK Z\Hakaton_HSE\cv_ik_pipeline\calibration.json"

cal = WorkspaceCalibration()
cal.load(CAL_PATH)
print(f"Calibration loaded: {len(cal.pixel_points)} points")
print("Calibration points:")
for i, (px, wd) in enumerate(zip(cal.pixel_points, cal.world_points)):
    print(f"  [{i+1}] pixel=({px[0]:.0f},{px[1]:.0f})  world=({wd[0]:.4f}, {wd[1]:.4f}, {wd[2]:.4f})")

cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
if not cap.isOpened():
    print("Cannot open camera!")
    sys.exit(1)

print("\nCamera open. Move cube around. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    detection = detect_object(frame, draw=True)

    if detection:
        cx, cy = detection["cx"], detection["cy"]
        world = cal.pixel_to_world(cx, cy)
        x, y, z = world

        # Try IK to see if position is reachable
        angles, ok, err = inverse_kinematics_deg(np.array([x, y, z]))

        text_color = (0, 255, 0) if (ok and err < 0.01) else (0, 100, 255)

        cv2.putText(frame, f"Pixel: ({cx}, {cy})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"World: x={x:.3f} y={y:.3f} z={z:.3f} m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(frame, f"IK: ok={ok}  err={err*1000:.1f}mm", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        if ok:
            cv2.putText(frame, f"Angles: {np.round(angles, 1)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Print to console too
        reachable = "OK" if (ok and err < 0.01) else "BAD"
        print(f"\r  pixel=({cx:4d},{cy:4d})  x={x:.3f} y={y:.3f} z={z:.3f}  IK={reachable} err={err*1000:.1f}mm",
              end="", flush=True)
    else:
        cv2.putText(frame, "No cube detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Detection Test", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print()
