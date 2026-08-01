import cv2
import serial
import csv
import time
import numpy as np

# setup
SERIAL_PORT = 'COM7'      # my arduino port 
BAUD_RATE = 115200
CAMERA_INDEX = 2 # for the logitech camera 
OUTPUT_FILE = 'session_001.csv'

#  colour range for detection
S_LOW, V_LOW = 120, 70
H_LOW1, H_HIGH1 = 0, 10
H_LOW2, H_HIGH2 = 170, 180

def find_dot_centres(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (H_LOW1, S_LOW, V_LOW), (H_HIGH1, 255, 255))
    mask2 = cv2.inRange(hsv, (H_LOW2, S_LOW, V_LOW), (H_HIGH2, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for c in contours:
        if cv2.contourArea(c) > 50:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centres.append((cx, cy))
    return sorted(centres, key=lambda p: p[0])


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
        'dot1_x', 'dot1_y',
        'dot2_x', 'dot2_y',
        'pixel_distance',
        'resting_distance',
        'strain_ratio'
    ])

    # capture resting distance first
    print("Hold sensor at REST for 3 seconds to set baseline.")
    resting_distances = []
    baseline_start = time.time()

    while time.time() - baseline_start < 3:
        ret, frame = cap.read()
        if not ret:
            continue
        centres = find_dot_centres(frame)
        if len(centres) >= 2:
            d = np.sqrt((centres[1][0]-centres[0][0])**2 +
                        (centres[1][1]-centres[0][1])**2)
            resting_distances.append(d)
        cv2.imshow('Sensor Capture - REST PHASE', frame)
        cv2.waitKey(1)

    if len(resting_distances) == 0:
        print("WARNING: Could not detect dots during rest phase.")
        print("Check red dots are visible and try again.")
        cap.release()
        ser.close()
        cv2.destroyAllWindows()
        exit()

    resting_distance = np.mean(resting_distances)
    print(f"Resting distance set: {resting_distance:.1f} pixels")
    print("Begin stretching. Press Q when done.")

    # main capture loop 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = int((time.time() - start) * 1000)

        # read serial
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                parts = line.split(',')
                if len(parts) == 4:
                    last_adc = int(parts[1])
                    last_voltage = float(parts[2])
                    last_resistance = float(parts[3])
            except:
                pass

        # find dots
        centres = find_dot_centres(frame)
        dot1 = centres[0] if len(centres) > 0 else (None, None)
        dot2 = centres[1] if len(centres) > 1 else (None, None)

        pixel_dist = None
        strain_ratio = None

        if dot1[0] is not None and dot2[0] is not None:
            pixel_dist = np.sqrt((dot2[0]-dot1[0])**2 +
                                  (dot2[1]-dot1[1])**2)
            strain_ratio = pixel_dist / resting_distance

            # draw on frame
            cv2.line(frame, dot1, dot2, (0, 255, 0), 2)
            cv2.circle(frame, dot1, 10, (0, 255, 0), -1)
            cv2.circle(frame, dot2, 10, (0, 255, 0), -1)
            cv2.putText(frame,
                        f"Strain: {strain_ratio:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

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
            dot1[0], dot1[1],
            dot2[0], dot2[1],
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