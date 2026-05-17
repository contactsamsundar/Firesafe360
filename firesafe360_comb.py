"""
🔊📷🌡️ Triple Multimodal Surveillance System (Audio + Vision + Sensors)
========================================================================
Controls (Ensure the Webcam window is active!):
  F1      Capture frame, record 10s audio, snapshot serial data, and analyse
  TAB     Clear / reset display
  Q/ESC   Quit
"""

import os
import sys
import json
import time
import threading
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
def check_deps():
    missing = []
    deps = [
        ("tensorflow", "tensorflow"), ("numpy", "numpy"), 
        ("sounddevice", "sounddevice"), ("blessed", "blessed"),
        ("opencv-python", "cv2"), ("pandas", "pandas"), 
        ("ultralytics", "ultralytics"), ("pyserial", "serial")
    ]
    for pkg, imp in deps:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[!] Please install missing packages:\n    pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import cv2
import numpy as np
import pandas as pd
import sounddevice as sd
import tensorflow as tf
import blessed
import serial
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
# Hardware / Serial Settings
SERIAL_PORT = 'COM3'  # Update this to your Arduino's COM port
BAUD_RATE   = 115200

# Audio YAMNet Settings
MODEL_DIR    = Path("yamnet_local")
SAVED_MODEL  = MODEL_DIR / "saved_model"
CLASS_MAP    = MODEL_DIR / "class_names.json"
SAMPLE_RATE  = 16000
CLIP_SECONDS = 10

# Vision YOLO Settings
CONF_THRESHOLD = 0.10
CAMERA_INDEX   = 0
CAPTURE_DIR    = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

F1_KEYS = [7340032, 65470, 63236, 190, 0x700000]

def get_yolo_models():
    return [
        {"name": "SHOU-ISD fire-and-smoke YOLOv8", "local_path": r"models_yolo_fire\SHOU-ISD__fire-and-smoke\yolov8n_1.pt", "status": "OK"},
        {"name": "YOLOv8m smoke detection", "local_path": r"models_yolo_fire\kittendev__YOLOv8m-smoke-detection\best.pt", "status": "OK"},
        {"name": "YOLOv8s forest fire detection", "local_path": r"models_yolo_fire\touati-kamel__yolov8s-forest-fire-detection\model.pt", "status": "OK"},
        {"name": "YOLO11 fire/smoke detector", "local_path": r"models_yolo_fire\leeyunjai__yolo11-firedetect\firedetect-11s.pt", "status": "OK"},
        {"name": "YOLOv26 fire detection", "local_path": r"models_yolo_fire\SalahALHaismawi__yolov26-fire-detection\best.pt", "status": "OK"},
        {"name": "ProFSAM YOLOv11n fire detector", "local_path": r"models_yolo_fire\UEmmanuel5__ProFSAM-Fire-Detector\Fire_best.pt", "status": "OK"}
    ]

# ── Sensor Processing (Arduino Serial) ───────────────────────────────────────
class SensorMonitor:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.connected = False
        self.latest_data = {}
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while True:
            try:
                ser = serial.Serial(self.port, self.baud, timeout=1)
                self.connected = True
                ser.reset_input_buffer()
                
                while self.connected:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if not line or "Temp,Humidity" in line:
                            continue
                        
                        data = line.split(',')
                        if len(data) == 5:
                            temp = data[0] + " °C" if data[0] != 'NaN' else "Sensor Error"
                            hum = data[1] + " %" if data[1] != 'NaN' else "Sensor Error"
                            alarm_status = "ACTIVE 🚨" if data[4] == '1' else "CLEAR ✅"
                            
                            self.latest_data = {
                                "temp": temp,
                                "hum": hum,
                                "flame": data[2],
                                "smoke": data[3],
                                "alarm": alarm_status
                            }
            except serial.SerialException:
                self.connected = False
                self.latest_data = {"error": "Connection Lost / Port Closed"}
                time.sleep(2)  # Wait before attempting to reconnect
            except Exception:
                pass


# ── Audio Processing (YAMNet) ────────────────────────────────────────────────
class YAMNetClassifier:
    def __init__(self):
        if not SAVED_MODEL.exists():
            print(f"[!] Model not found at {SAVED_MODEL}.")
            sys.exit(1)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel("ERROR")
        self.model = tf.saved_model.load(str(SAVED_MODEL))

        try:
            with open(CLASS_MAP) as f:
                self.class_names = json.load(f)
        except FileNotFoundError:
            print(f"[!] Could not find {CLASS_MAP}.")
            sys.exit(1)

    def predict(self, waveform: np.ndarray):
        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)

        top5_idx = mean_scores.argsort()[-5:][::-1]
        top5     = [(self.class_names[i], float(mean_scores[i])) for i in top5_idx]
        return top5[0][0], top5[0][1], top5

def record_clip(duration=CLIP_SECONDS, sr=SAMPLE_RATE, progress_cb=None):
    frames = int(sr * duration)
    audio  = sd.rec(frames, samplerate=sr, channels=1, dtype='float32')
    for i in range(duration):
        time.sleep(1)
        if progress_cb: progress_cb(i + 1, duration)
    sd.wait()
    wav = audio.flatten()
    peak = np.abs(wav).max()
    if peak > 1e-6: wav = wav / peak
    return wav

# ── Vision Processing (YOLO) ─────────────────────────────────────────────────
def infer_one_model(model_row, image_path):
    result = {
        "model_name": model_row["name"],
        "status": "FAILED",
        "detections": "",
        "max_confidence": "",
        "max_detection_label": "", 
        "error": "",
    }
    try:
        model = YOLO(model_row["local_path"])
        outputs = model.predict(source=str(image_path), conf=CONF_THRESHOLD, verbose=False)
        
        detections = []
        max_conf, max_label = 0.0, "NONE"

        for output in outputs:
            names = output.names
            if output.boxes is None: continue

            for box in output.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names.get(cls_id, str(cls_id))
                detections.append(f"{label}:{conf:.2f}")
                
                if conf > max_conf:
                    max_conf = conf
                    max_label = label

        result["status"] = "OK"
        result["detections"] = ", ".join(detections) if detections else "NO DETECTION"
        result["max_confidence"] = f"{max_conf:.2f}" if detections else "0.00"
        result["max_detection_label"] = max_label 

    except Exception as e:
        result["error"] = str(e)

    return result

# ── Unified Terminal Display ─────────────────────────────────────────────────
class UnifiedDisplay:
    def __init__(self, sensor_monitor):
        self.term = blessed.Terminal()
        self.lock = threading.Lock()
        self.sensor_monitor = sensor_monitor
        
        self.audio_state = "idle"
        self.audio_result = None
        self.audio_elapsed = 0
        
        self.vision_state = "idle"
        self.vision_result_df = None
        self.vision_current_model = ""

        self.sensor_state = "idle"
        self.sensor_snapshot = {}

    def render(self):
        with self.lock:
            t = self.term
            print(t.home + t.clear, end="")
            
            # Header
            print(t.bold + t.cyan + "🔊📷🌡️ Triple Multimodal System" + t.normal)
            print("Controls: " + t.yellow + "F1" + t.normal + " = Scan All | " + 
                  t.yellow + "TAB" + t.normal + " = Clear | " + 
                  t.yellow + "Q/ESC" + t.normal + " = Quit")
            print(t.red + "[!] Note: Click the WEBCAM WINDOW for keys to register!" + t.normal)
            print("-" * 75)

            # --- SENSOR STATUS BLOCK ---
            print(t.bold + "\n🌡️ SENSOR MODULE" + t.normal)
            if self.sensor_state == "result":
                if not self.sensor_monitor.connected:
                    print("Status: " + t.red + "OFFLINE - NO DATA CAPTURED" + t.normal)
                else:
                    print("Status: " + t.green + "SNAPSHOT CAPTURED" + t.normal)
                
                d = self.sensor_snapshot
                print(f"  Temperature  : {d.get('temp', 'N/A')}")
                print(f"  Humidity     : {d.get('hum', 'N/A')}")
                print(f"  Flame Raw    : {d.get('flame', 'N/A')}")
                print(f"  Smoke Raw    : {d.get('smoke', 'N/A')}")
                print(f"  Alarm Status : {d.get('alarm', 'N/A')}")
                
            elif not self.sensor_monitor.connected:
                print("Status: " + t.red + f"OFFLINE (Searching for {SERIAL_PORT}...)" + t.normal)
            elif self.sensor_state == "idle":
                print("Status: READY (Monitoring in background)")


            # --- VISION STATUS BLOCK ---
            print(t.bold + "\n📷 VISION MODULE" + t.normal)
            if self.vision_state == "idle":
                print("Status: READY")
            elif self.vision_state == "analysing":
                print(f"Status: ANALYSING... (Running: {t.cyan}{self.vision_current_model}{t.normal})")
            elif self.vision_state == "result":
                print("Status: " + t.green + "SCAN COMPLETE" + t.normal)
                if self.vision_result_df is not None:
                    print("\n" + self.vision_result_df.to_string(index=False))

            # --- AUDIO STATUS BLOCK ---
            print(t.bold + "\n🎤 AUDIO MODULE" + t.normal)
            if self.audio_state == "idle":
                print("Status: READY")
            elif self.audio_state == "recording":
                print(f"Status: RECORDING... ({self.audio_elapsed} / {CLIP_SECONDS}s) - Make noise!")
            elif self.audio_state == "analysing":
                print("Status: ANALYSING with YAMNet...")
            elif self.audio_state == "result":
                print("Status: " + t.green + "SCAN COMPLETE" + t.normal)
                top_class, conf, top5 = self.audio_result
                print("\n" + t.bold + t.green + f"🎯 Top Audio Match: {top_class.upper()} ({conf*100:.1f}%)" + t.normal)
                print("Top 5 Detected:")
                for i, (name, score) in enumerate(top5):
                    color = t.green if i == 0 else t.white
                    print(color + f"  {i+1}. {name:<20} {score*100:>5.1f}%" + t.normal)

            sys.stdout.flush()

# ── Main Event Loop ──────────────────────────────────────────────────────────
def main():
    print("Loading AI Models and opening Serial port, please wait...")
    yamnet_detector = YAMNetClassifier()
    yolo_models = [m for m in get_yolo_models() if m.get("status") == "OK" and m.get("local_path")]
    sensor_monitor = SensorMonitor(SERIAL_PORT, BAUD_RATE)
    
    display = UnifiedDisplay(sensor_monitor)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    audio_busy = False
    vision_busy = False

    def audio_task():
        nonlocal audio_busy
        audio_busy = True
        try:
            def on_progress(elapsed, total):
                display.audio_elapsed = elapsed
                display.audio_state = "recording"
                display.render()

            waveform = record_clip(progress_cb=on_progress)
            display.audio_state = "analysing"
            display.render()
            
            top_class, conf, top5 = yamnet_detector.predict(waveform)
            
            display.audio_result = (top_class, conf, top5)
            display.audio_state = "result"
            display.render()
        finally:
            audio_busy = False

    def vision_task(frame_to_process, image_path):
        nonlocal vision_busy
        vision_busy = True
        try:
            cv2.imwrite(str(image_path), frame_to_process)
            all_results = []
            
            for m in yolo_models:
                display.vision_state = "analysing"
                display.vision_current_model = m['name']
                display.render()
                
                res = infer_one_model(m, image_path)
                all_results.append(res)
                
            display.vision_result_df = pd.DataFrame(all_results)
            display.vision_state = "result"
            display.render()
        finally:
            vision_busy = False

    # Initial Draw
    display.render()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        cv2.imshow("Webcam - F1: Scan All | TAB: Clear | Q/Esc: Quit", frame)

        key = cv2.waitKeyEx(1)
        if key == -1:
            continue

        # F1 Pressed
        if key in F1_KEYS:
            if not audio_busy and not vision_busy:
                # 1. Snapshot Sensor Data Instantly (Even if Offline)
                if sensor_monitor.connected:
                    display.sensor_snapshot = sensor_monitor.latest_data.copy()
                else:
                    display.sensor_snapshot = {} # Will default to "N/A" on render
                display.sensor_state = "result"

                # 2. Dispatch Background CV & Audio Threads
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = CAPTURE_DIR / f"capture_{timestamp}.jpg"
                
                threading.Thread(target=audio_task, daemon=True).start()
                threading.Thread(target=vision_task, args=(frame.copy(), image_path), daemon=True).start()

                # Force render to show immediate snapshot while threads spool up
                display.render()

        # TAB Pressed (Clear/Reset)
        elif key == 9:
            if not audio_busy and not vision_busy:
                display.audio_state = "idle"
                display.vision_state = "idle"
                display.sensor_state = "idle"
                
                display.audio_result = None
                display.vision_result_df = None
                display.sensor_snapshot = {}
                display.render()

        # Q or Esc Pressed
        elif key in [ord("q"), ord("Q"), 27]:
            print("\nExiting program...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()