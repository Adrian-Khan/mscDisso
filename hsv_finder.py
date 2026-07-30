"""
hsv_finder.py
find the colour of alligator clips (or whatever tracking material) to allow opencv to detect it specifically

"""

import cv2
import numpy as np

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        print(f"x={x}, y={y} → H={h}, S={s}, V={v}")

cv2.namedWindow('HSV Picker')
cv2.setMouseCallback('HSV Picker', mouse_callback)

frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('HSV Picker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()