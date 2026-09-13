import serial
import time

ser = serial.Serial('COM7', 115200, timeout=2) 
time.sleep(2)
ser.reset_input_buffer()

for i in range(500):
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        print(line)
    else:
        print("nothing in buffer")
    time.sleep(0.1)

ser.close()