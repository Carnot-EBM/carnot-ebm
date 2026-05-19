import serial
import time
try:
    with serial.Serial('/dev/ttyACM1', 115200, timeout=1) as ser:
        ser.write(b"SAMPLE\n")
        time.sleep(0.5)
        print("Read:", ser.read(100))
except Exception as e:
    print("Error:", e)
