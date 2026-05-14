import serial
import time
import json
import requests

# --- Serial Configuration ---
# Change this to your specific COM port (e.g., 'COM4', '/dev/ttyUSB0', '/dev/cu.usbserial-XXXX')
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200

# --- Ollama Configuration ---
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "tinyllama" # Change to gemma:2b or llama3:8b if preferred

SYSTEM_PROMPT = """You are an automated Emergency Response AI analyzing ESP32 environmental sensor data.
When provided with the JSON payload, you must output a concise, easy-to-read emergency bulletin.

Your response MUST follow this structure exactly:
1. STATUS: What is happening right now (summarize the temperature, smoke, and flame sensors).
2. ACTION REQUIRED: You must explicitly state ONE of the following three actions:
   - "EVACUATE" (If flame is detected, smoke is critical, or temperature is dangerously high).
   - "STAY CAUTIOUS" (If smoke is elevated but not critical, or temperature is slightly abnormal).
   - "ALL OK" (If sensors are normal).
3. DETAILS: A brief, one-sentence explanation of why you chose that action based on the data."""

def ask_ollama(sensor_json_str):
    """Sends the sensor data to the local Ollama LLM and returns the response."""
    payload = {
        "model": MODEL_NAME,
        "system": SYSTEM_PROMPT,
        "prompt": f"Analyse this JSON data:\n\n{sensor_json_str}",
        "stream": False
    }
    
    try:
        # 120-second timeout matches the Node.js timeout config
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "(empty response)")
    except requests.exceptions.RequestException as e:
        return f"[LLM ERROR] Could not connect to Ollama: {e}"

def main():
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✅ Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        print(f"🤖 Connected to Ollama model: {MODEL_NAME}")
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
                    # Create a dictionary to match the JSON expectation of the LLM prompt
                    sensor_dict = {
                        "temperature_C": data[0] if data[0] != 'NaN' else "Error",
                        "humidity_percent": data[1] if data[1] != 'NaN' else "Error",
                        "flame_raw": int(data[2]) if data[2].isdigit() else data[2],
                        "smoke_raw": int(data[3]) if data[3].isdigit() else data[3],
                        "hardware_alarm_active": True if data[4] == '1' else False
                    }
                    
                    # Convert to JSON string
                    sensor_json = json.dumps(sensor_dict, indent=2)

                    # --- 1. Print Raw Input ---
                    print("=" * 60)
                    print("📡 RAW SENSOR DATA RECEIVED:")
                    print("-" * 60)
                    print(sensor_json)
                    print("-" * 60)
                    
                    # --- 2. Pass to LLM ---
                    print(f"🧠 Sending to {MODEL_NAME} for analysis... please wait.")
                    llm_response = ask_ollama(sensor_json)
                    
                    # --- 3. Print LLM Output ---
                    print("\n🚨 AI EMERGENCY RESPONSE BULLETIN:")
                    print("-" * 60)
                    print(llm_response.strip())
                    print("=" * 60 + "\n")
                    
                else:
                    print(f"⚠️ Received malformed data: {line}")

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