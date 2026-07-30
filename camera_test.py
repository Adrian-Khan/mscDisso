"""
camera_test.py

test if external webcam (or internal) is connected, and at what port - also can frame the sensors correctly and adjust positioning if needed here 

"""

import cv2

cap = cv2.VideoCapture(2)  # for logitech turns out its on 2, both 0 and 1 are just my own integrated webcam
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()