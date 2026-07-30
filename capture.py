"""
capture.py 

this is the main file for capturing the dataset for training.

input is from the arduino to get voltage, adc, resistance.
input is also from the camera to get yellow and blue x and y , pixel distance, resting distance

output is the above + time in ms, and strain ratio as a csv file 
"""

import cv2
import serial
import csv
import time
import numpy as np

SERIAL_PORT = 'COM7'
BAUD_RATE = 115200
CAMERA_INDEX = 2
OUTPUT_FILE = 'session_002.csv'

# hsv ranges found in hsv_finder.py for my alligator clips 
# yellow clip (left)
YELLOW_LOW  = np.array([20,  80,  140])
YELLOW_HIGH = np.array([35,  255, 255])

# blue clip (right)
BLUE_LOW  = np.array([100, 200, 100])
BLUE_HIGH = np.array([124, 255, 200])

def find_clip(frame, lower, upper):
    # find the centre of the clip
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # take largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

# run main
print("Connecting to Arduino")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
ser.reset_input_buffer()
print("Arduino connected.")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting capture. Press Q to stop.")
print(f"Saving to {OUTPUT_FILE}")

start = time.time()
last_adc = None
last_voltage = None
last_resistance = None

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp_ms',
        'adc_raw',
        'voltage',
        'resistance_ohm',
        'yellow_x', 'yellow_y',
        'blue_x', 'blue_y',
        'pixel_distance',
        'resting_distance',
        'strain_ratio'
    ])

    # capture resting distance first
    print("Hold sensor at rest for 3 seconds to set baseline")
    resting_distances = []
    baseline_start = time.time()

    while time.time() - baseline_start < 3:
        ret, frame = cap.read()
        if not ret:
            continue

        yellow = find_clip(frame, YELLOW_LOW, YELLOW_HIGH)
        blue   = find_clip(frame, BLUE_LOW,   BLUE_HIGH)

        if yellow and blue:
            d = np.sqrt((blue[0]-yellow[0])**2 +
                        (blue[1]-yellow[1])**2)
            resting_distances.append(d)

            # show detection during rest phase
            cv2.circle(frame, yellow, 12, (0, 255, 255), -1)
            cv2.circle(frame, blue,   12, (255, 100, 0), -1)
            cv2.line(frame, yellow, blue, (255, 255, 255), 2)
            cv2.putText(frame, "REST PHASE - hold still",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

        cv2.imshow('Sensor Capture', frame)
        cv2.waitKey(1)

    if len(resting_distances) == 0:
        print("WARNING: Could not detect both clips during rest phase.")
        print("Check lighting and clip colours are visible.")
        cap.release()
        ser.close()
        cv2.destroyAllWindows()
        exit()

    resting_distance = np.mean(resting_distances)
    print(f"Resting distance set: {resting_distance:.1f} pixels")
    print("Begin stretching. Press Q when done.")

    # main capture loop and logic
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = int((time.time() - start) * 1000)

        # read latest serial line
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                parts = line.split(',')
                if len(parts) == 4:
                    last_adc        = int(parts[1])
                    last_voltage    = float(parts[2])
                    last_resistance = float(parts[3])
            except:
                pass

        # find both clips
        yellow = find_clip(frame, YELLOW_LOW, YELLOW_HIGH)
        blue   = find_clip(frame, BLUE_LOW,   BLUE_HIGH)

        pixel_dist  = None
        strain_ratio = None

        if yellow and blue:
            pixel_dist = np.sqrt((blue[0]-yellow[0])**2 +
                                  (blue[1]-yellow[1])**2)
            strain_ratio = pixel_dist / resting_distance

            # draw tracking
            cv2.circle(frame, yellow, 12, (0, 255, 255), -1)
            cv2.circle(frame, blue,   12, (255, 100, 0), -1)
            cv2.line(frame, yellow, blue, (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Strain: {strain_ratio:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame,
                        "WARNING: clip not detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        if last_resistance is not None:
            cv2.putText(frame,
                        f"R: {last_resistance:.1f} ohm",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # write row
        writer.writerow([
            t_ms,
            last_adc,
            last_voltage,
            last_resistance,
            yellow[0] if yellow else None,
            yellow[1] if yellow else None,
            blue[0]   if blue   else None,
            blue[1]   if blue   else None,
            pixel_dist,
            resting_distance,
            strain_ratio
        ])

        cv2.imshow('Sensor Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
ser.close()
cv2.destroyAllWindows()
print(f"Done. Data saved to {OUTPUT_FILE}")