import csv
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

MODEL_DIR = Path("models_yolo_fire")
MANIFEST = MODEL_DIR / "manifest.csv"

CONF_THRESHOLD = 0.25
CAMERA_INDEX = 0

def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}. Run download_yolo_fire_models.py first."
        )

    rows = []
    with MANIFEST.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "OK" and row.get("local_path"):
                rows.append(row)
    return rows

def infer_one_model(model_row, image_path):
    name = model_row["name"]
    year = model_row["year"]
    path = model_row["local_path"]

    result = {
        "model_name": name,
        "year_released": year,
        "status": "FAILED",
        "detections": "",
        "max_confidence": "",
        "error": "",
    }

    try:
        model = YOLO(path)
        outputs = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        detections = []
        max_conf = 0.0

        for output in outputs:
            names = output.names

            if output.boxes is None:
                continue

            for box in output.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names.get(cls_id, str(cls_id))
                detections.append(f"{label}:{conf:.2f}")
                max_conf = max(max_conf, conf)

        result["status"] = "OK"
        result["detections"] = ", ".join(detections) if detections else "NO DETECTION"
        result["max_confidence"] = f"{max_conf:.2f}" if detections else "0.00"

    except Exception as e:
        result["error"] = str(e)

    return result

def main():
    models = load_manifest()

    if not models:
        print("No downloaded models found in manifest.")
        return

    print(f"Loaded {len(models)} models.")
    print("Press TAB to capture a frame and run all models.")
    print("Press Q to quit.")

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    capture_dir = Path("captures")
    capture_dir.mkdir(exist_ok=True)

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read frame from camera.")
            time.sleep(0.5)
            continue

        cv2.imshow("Webcam - Press TAB to test, Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q")]:
            break

        # TAB key = 9
        if key == 9:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = capture_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)

            print(f"\nCaptured: {image_path}")
            print("Running models...")

            all_results = []

            for m in models:
                print(f"Running: {m['name']}")
                r = infer_one_model(m, image_path)
                all_results.append(r)

            df = pd.DataFrame(all_results)
            print("\n=== Results ===")
            print(df.to_string(index=False))

            results_path = capture_dir / f"results_{timestamp}.csv"
            df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()