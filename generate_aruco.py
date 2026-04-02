"""Run once to generate and save ArUco marker as PNG. Then print and stick on cube."""
import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_img = np.zeros((300, 300), dtype=np.uint8)
cv2.aruco.generateImageMarker(aruco_dict, 0, 300, marker_img, 1)

# Add white border so it's easier to detect
bordered = cv2.copyMakeBorder(marker_img, 30, 30, 30, 30,
                               cv2.BORDER_CONSTANT, value=255)
cv2.imwrite("aruco_marker_0.png", bordered)
print("Saved aruco_marker_0.png — print this and stick on top of cube")
cv2.imshow("ArUco Marker (print me)", bordered)
cv2.waitKey(0)
cv2.destroyAllWindows()
