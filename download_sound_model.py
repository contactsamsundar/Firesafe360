"""
FireGuard — YAMNet Model Downloader
=====================================
Downloads Google's YAMNet audio classification model from TensorFlow Hub
and saves it locally so fire_detector.py runs fully OFFLINE afterwards.

YAMNet facts:
  • 3.7M parameters (MobileNetV1)  ~13 MB saved
  • 521 AudioSet classes including:
      "Fire"  "Crackling"  "Smoke detector, smoke alarm"
      "Fire alarm"  "Alarm"  "Hiss"  "White noise"
  • Input: 16 kHz mono float32 waveform (any length ≥ 0.975 s)
  • Inference: ~0.03s per 10s clip on CPU  ← fast on Raspberry Pi too

Usage (needs internet, run ONCE):
    pip install tensorflow tensorflow-hub numpy sounddevice blessed
    python download_model.py
"""

import os, sys, json, platform, shutil
from pathlib import Path

MODEL_SAVE_DIR = Path("yamnet_local")
META_FILE      = MODEL_SAVE_DIR / "meta.json"

IS_PI = platform.machine().startswith("aarch") or \
        "raspberrypi" in platform.node().lower()

# ── Dependency check ──────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg, imp in [("tensorflow","tensorflow"), ("tensorflow-hub","tensorflow_hub"),
                     ("numpy","numpy"), ("sounddevice","sounddevice"),
                     ("blessed","blessed")]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        print(f"    Run:  pip install {' '.join(missing)}\n")
        sys.exit(1)
    print("[✓] All dependencies present")

check_deps()

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ── Fire-relevant YAMNet class names ─────────────────────────────────────────
# These are the display_name values in yamnet_class_map.csv that we aggregate.
FIRE_CLASSES = {
    "Fire",
    "Crackling",
    "Fire alarm",
    "Smoke detector, smoke alarm",
    "Hiss",                    # gas leak hiss / pressure release
}

ALARM_CLASSES = {
    "Smoke detector, smoke alarm",
    "Fire alarm",
    "Alarm",
    "Buzzer",
    "Beep, bleep",
}

# ── Download & save ───────────────────────────────────────────────────────────
def download_and_save():
    print("\n[*] Downloading YAMNet from TensorFlow Hub ...")
    print("    (This is ~13 MB and only happens once)\n")

    # Set cache dir so TF Hub downloads here
    os.environ["TFHUB_CACHE_DIR"] = str(MODEL_SAVE_DIR / "tfhub_cache")

    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("[✓] Model loaded from TF Hub")

    # Save as TF SavedModel for offline use
    saved_path = MODEL_SAVE_DIR / "saved_model"
    tf.saved_model.save(model, str(saved_path))
    print(f"[✓] Saved to {saved_path}")

    # Extract and save class names
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_names = []
    with open(class_map_path, "r") as f:
        next(f)   # skip header
        for line in f:
            parts = line.strip().split(",")
            class_names.append(parts[2].strip('"'))

    with open(MODEL_SAVE_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[✓] Saved {len(class_names)} class labels")

    # Save fire/alarm index sets for fast lookup
    fire_indices  = [i for i, n in enumerate(class_names) if n in FIRE_CLASSES]
    alarm_indices = [i for i, n in enumerate(class_names) if n in ALARM_CLASSES]
    print(f"[✓] Fire classes  : {[class_names[i] for i in fire_indices]}")
    print(f"[✓] Alarm classes : {[class_names[i] for i in alarm_indices]}")

    # Write meta
    meta = {
        "model":         "YAMNet (Google)",
        "source":        "https://tfhub.dev/google/yamnet/1",
        "num_classes":   len(class_names),
        "sample_rate":   16000,
        "fire_indices":  fire_indices,
        "alarm_indices": alarm_indices,
        "fire_classes":  [class_names[i] for i in fire_indices],
        "alarm_classes": [class_names[i] for i in alarm_indices],
        "platform":      platform.system(),
        "saved_model":   str(saved_path),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = sum(f.stat().st_size for f in MODEL_SAVE_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\n[✓] Total saved size: {size_mb:.1f} MB")


# ── Quick sanity-check inference ──────────────────────────────────────────────
def verify():
    print("\n[*] Verifying saved model with dummy audio ...")
    model = tf.saved_model.load(str(MODEL_SAVE_DIR / "saved_model"))
    dummy = np.zeros(16000, dtype=np.float32)   # 1 second of silence
    scores, embeddings, spectrogram = model(dummy)
    print(f"[✓] Output shape: scores={scores.shape}  embeddings={embeddings.shape}")
    print(f"[✓] Top class index: {int(scores.numpy().mean(axis=0).argmax())}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("""
╔══════════════════════════════════════════════════╗
║  🔥  FireGuard — YAMNet Model Downloader  🔥     ║
║     Google YAMNet · 521 classes · 3.7M params   ║
╚══════════════════════════════════════════════════╝
""")
    print(f"[i] Platform  : {platform.system()} {platform.machine()}")
    print(f"[i] Python    : {sys.version.split()[0]}")
    print(f"[i] TF        : {tf.__version__}")
    print(f"[i] Save dir  : {MODEL_SAVE_DIR.resolve()}\n")

    if MODEL_SAVE_DIR.exists() and (MODEL_SAVE_DIR / "saved_model").exists():
        print("[i] Model already downloaded. Delete 'yamnet_local/' to re-download.")
        verify()
    else:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        download_and_save()
        verify()

    print("\n" + "─" * 52)
    print("✅  Done. Run:  python fire_detector.py")
    print("   Then press  TAB  to start a 10-second recording.")
    print("─" * 52 + "\n")


if __name__ == "__main__":
    main()
