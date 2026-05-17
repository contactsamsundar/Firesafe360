import serial
import time

# --- Serial Configuration ---
# Change this to your specific COM port (e.g., 'COM4', '/dev/ttyUSB0', '/dev/cu.usbserial-XXXX')
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200

def main():
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✅ Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        print("Waiting for sensor data...\n")
        
        # Flush the buffer to ignore old data
        ser.reset_input_buffer() 

        while True:
            # Check if there is data waiting to be read
            if ser.in_waiting > 0:
                # Read the line, decode it to a string, and strip the newline characters
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                # Skip empty lines or the CSV header
                if not line or "Temp,Humidity" in line:
                    continue

                # Parse the CSV line
                data = line.split(',')

                if len(data) == 5:
                    # Extract the values from the split CSV list
                    temp = data[0] if data[0] != 'NaN' else "Error"
                    humidity = data[1] if data[1] != 'NaN' else "Error"
                    flame = data[2]
                    smoke = data[3]
                    alarm = "ACTIVE 🚨" if data[4] == '1' else "INACTIVE OK ✅"
                    
                    # --- Print Formatted Sensor Data ---
                    print("=" * 45)
                    print("📡 SENSOR DATA RECEIVED:")
                    print("-" * 45)
                    print(f"Temperature: {temp}°C")
                    print(f"Humidity:    {humidity}%")
                    print(f"Flame Raw:   {flame}")
                    print(f"Smoke Raw:   {smoke}")
                    print(f"Alarm State: {alarm}")
                    print("=" * 45 + "\n")
                    
                else:
                    print(f"⚠️ Received malformed data: {line}")

            # Small sleep to prevent high CPU utilization
            time.sleep(0.01)

    except serial.SerialException as e:
        print(f"\n[ERROR] Could not open port {SERIAL_PORT}.")
        print("1. Check if the port name is correct.")
        print("2. Ensure the Arduino IDE Serial Monitor is CLOSED.")
        print(f"Detailed error: {e}")
        
    except KeyboardInterrupt:
        print("\nExiting program by user request.")
        
    finally:
        # Clean up and close the port when the program exits
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == '__main__':
    main()
