"""
test hsv detection is correct

"""

# test red detection
import cv2
import numpy as np

cap = cv2.VideoCapture(2)
RED_LOW  = np.array([165, 120, 120])
RED_HIGH = np.array([180, 180, 165])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOW, RED_HIGH)
    cv2.imshow('Red mask', mask)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()