import cv2
import torch
import time
import os
import json
import logging
import winsound
import numpy as np
from datetime import datetime
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel,
    ViTImageProcessor, ViTForImageClassification
)
from ultralytics import YOLO
import torchvision.models as models

# ==========================================
# COLOUR PALETTE FOR ON-SCREEN OVERLAY
# ==========================================
C = {
    "bg_dark":    (15,  15,  25),
    "panel":      (30,  30,  50),
    "white":      (255, 255, 255),
    "yellow":     (0,   220, 255),
    "green":      (50,  230, 120),
    "red":        (60,   60, 240),
    "orange":     (30,  140, 255),
    "cyan":       (230, 200,  40),
    "grey":       (160, 160, 160),
    "fire_red":   (0,    0,  255),
    "fire_orng":  (0,   100, 255),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

# ==========================================
# LOGGING – console AND a scrolling log list
# ==========================================
log_lines = []          # displayed on screen (newest last, capped at 18)
step_lines = []         # current-frame step-by-step analysis lines

def log(level, msg):
    """Unified logger: writes to console AND screen."""
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "[ INFO ]", "WARN": "[  ⚠️  ]", "STEP": "[ STEP ]",
              "FIRE": "[ 🔥 FIRE 🔥 ]", "SAFE": "[ ✅ SAFE ]", "DATA": "[ DATA ]"}.get(level, "[  ...  ]")
    line = f"{ts}  {prefix}  {msg}"
    print(line)
    log_lines.append((level, line))
    if len(log_lines) > 18:
        log_lines.pop(0)

def step(n, msg):
    """Record a numbered analysis step for the current frame."""
    text = f"  STEP {n:02d} » {msg}"
    print(text)
    step_lines.append(text)

# ==========================================
# OVERLAY HELPERS
# ==========================================
def draw_panel(img, x, y, w, h, alpha=0.72, color=None):
    """Draw a semi-transparent dark panel."""
    color = color or C["panel"]
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def put(img, text, x, y, scale=0.48, color=None, thickness=1):
    color = color or C["white"]
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

def put_mono(img, text, x, y, scale=1.0, color=None):
    color = color or C["white"]
    cv2.putText(img, text, (x, y), FONT_MONO, scale, color, 1, cv2.LINE_AA)

def bar(img, x, y, w, pct, fg, label=""):
    """Draw a percentage bar."""
    h = 12
    cv2.rectangle(img, (x, y), (x+w, y+h), C["grey"], 1)
    fill = int(w * min(pct, 100) / 100)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), fg, -1)
    if label:
        put(img, label, x+w+6, y+9, 0.38, C["white"])

def section_header(img, x, y, w, title, color):
    draw_panel(img, x, y-14, w, 18, alpha=0.85, color=color)
    put(img, title, x+6, y, 0.44, C["white"], 1)

# ==========================================
# BOOT SEQUENCE
# ==========================================
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S')

print("\n" + "="*70)
print("  🔬  FIRE DETECTION AI LAB  –  SCIENCE EXHIBITION EDITION")
print("="*70 + "\n")

MAIN_DIR = "science_fair_models"
os.makedirs("Fire_Evidence", exist_ok=True)
os.makedirs("Safe_Captures", exist_ok=True)
JSONL = "full_dataset_log.jsonl"

# ==========================================
# LOAD MODELS  (with step logging)
# ==========================================
log("INFO", "Loading all 8 AI models into RAM …")

log("STEP", "1/8  Loading CLIP-Base (zero-shot vision-language model)")
clip_base  = CLIPModel.from_pretrained(f"./{MAIN_DIR}/clip_base")
proc_base  = CLIPProcessor.from_pretrained(f"./{MAIN_DIR}/clip_base")

log("STEP", "2/8  Loading CLIP-Large (higher-capacity zero-shot model)")
clip_large = CLIPModel.from_pretrained(f"./{MAIN_DIR}/clip_large")
proc_large = CLIPProcessor.from_pretrained(f"./{MAIN_DIR}/clip_large")

log("STEP", "3/8  Loading SigLIP (Google sigmoid vision-language model)")
siglip      = AutoModel.from_pretrained(f"./{MAIN_DIR}/siglip_base")
proc_siglip = AutoProcessor.from_pretrained(f"./{MAIN_DIR}/siglip_base")

log("STEP", "4/8  Loading YOLOv8 General (real-time object detector)")
yolo_general = YOLO(f"./{MAIN_DIR}/yolo_general/yolov8n.pt")

log("STEP", "5/8  Loading YOLOv8 Fire-Specialist (fire-trained detector)")
yolo_fire = YOLO(f"./{MAIN_DIR}/yolo_fire/model.pt")

log("STEP", "6/8  Loading MobileNetV3-Small (lightweight CNN classifier)")
mobilenet = models.mobilenet_v3_small()
mobilenet.load_state_dict(torch.load(f"./{MAIN_DIR}/mobilenet/mobilenet_v3_small.pth", weights_only=True))
mobilenet.eval()
imagenet_categories = models.MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
mob_preprocess = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()

log("STEP", "7/8  Loading ResNet-50 (deep residual CNN classifier)")
resnet50 = models.resnet50()
resnet50.load_state_dict(torch.load(f"./{MAIN_DIR}/resnet/resnet50.pth", weights_only=True))
resnet50.eval()
res_preprocess = models.ResNet50_Weights.DEFAULT.transforms()

log("STEP", "8/8  Loading Google ViT-Base (Vision Transformer classifier)")
vit_model     = ViTForImageClassification.from_pretrained(f"./{MAIN_DIR}/vit_base")
vit_processor = ViTImageProcessor.from_pretrained(f"./{MAIN_DIR}/vit_base")

log("INFO", "✅  ALL 8 MODELS READY — Continuous logging enabled")
print("\n" + "="*70 + "\n")

# ==========================================
# STATE
# ==========================================
clip_labels = ["a photograph of a fire or flame", "a normal room"]
last_data   = {}          # holds results from last analysis
frame_count = 0
last_analysis_time = None
analyzing   = False

cap = cv2.VideoCapture(0)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # ── Canvas: widen frame to fit side panels ──────────────────────────────
    SIDE = 370
    LOG_H = 320
    canvas = np.zeros((H + LOG_H, W + SIDE, 3), dtype=np.uint8)
    canvas[:H, :W] = frame                      # camera feed top-left

    # ── Fill background panels ───────────────────────────────────────────────
    draw_panel(canvas, W, 0, SIDE, H + LOG_H, alpha=0.97, color=(18, 18, 32))
    draw_panel(canvas, 0, H, W,    LOG_H,     alpha=0.97, color=(12, 12, 22))

    # ======================================================
    #  LEFT COLUMN: camera feed with overlaid fire boxes
    # ======================================================
    fire_detected = last_data.get("is_fire", False)

    # Animated fire border when fire is active
    if fire_detected:
        t = time.time()
        blink = int((np.sin(t * 8) + 1) * 0.5 * 255)
        border_c = (0, int(blink * 0.4), blink)
        cv2.rectangle(canvas, (0, 0), (W-1, H-1), border_c, 6)
        put(canvas, "🔥  FIRE DETECTED  🔥", W//2 - 130, H - 18, 0.7, C["fire_red"], 2)

    # Status badge top-left of camera
    status_txt = "FIRE DETECTED" if fire_detected else "SCENE SAFE"
    status_col = C["fire_red"] if fire_detected else C["green"]
    draw_panel(canvas, 4, 4, 200, 28, alpha=0.82, color=status_col)
    put(canvas, status_txt, 10, 23, 0.6, C["white"], 2)

    # Frame counter + fps hint
    put(canvas, f"Frame: {frame_count:05d}   Press TAB to analyse", 6, H - 40, 0.42, C["cyan"])

    # ======================================================
    #  RIGHT PANEL: model results
    # ======================================================
    rx = W + 8
    ry = 12

    # ── Title ────────────────────────────────────────────────────────────────
    draw_panel(canvas, W, 0, SIDE, 36, alpha=0.9, color=(40, 10, 80))
    put(canvas, "AI FIRE DETECTION LAB", rx, 24, 0.62, C["yellow"], 2)

    ry = 50

    if last_data:
        d = last_data

        # ── CLIP MODELS ──────────────────────────────────────────────────────
        section_header(canvas, W+4, ry, SIDE-8, "ZERO-SHOT LANGUAGE-VISION", (60, 30, 100))
        ry += 10

        for label, prob, col in [
            ("CLIP Base",  d["p_base"],  C["cyan"]),
            ("CLIP Large", d["p_large"], C["orange"]),
            ("SigLIP",     d["p_sig"],   C["green"]),
        ]:
            ry += 16
            put(canvas, label, rx, ry, 0.44, C["grey"])
            pct_c = C["fire_red"] if prob > 65 else col
            put(canvas, f"{prob:.1f}%", rx + 120, ry, 0.44, pct_c)
            bar(canvas, rx + 175, ry - 10, 140, prob, pct_c)
            ry += 4

        ry += 12
        # ── YOLO DETECTORS ───────────────────────────────────────────────────
        section_header(canvas, W+4, ry, SIDE-8, "REAL-TIME OBJECT DETECTORS (YOLO)", (0, 60, 80))
        ry += 14

        for label, dets in [("YOLO General", d["det_gen"]), ("YOLO Fire", d["det_fire"])]:
            ry += 16
            color = C["fire_red"] if (label == "YOLO Fire" and dets) else C["white"]
            obj_str = ", ".join(dets[:4]) if dets else "— nothing detected"
            put(canvas, f"{label}:", rx, ry, 0.42, C["grey"])
            ry += 14
            put(canvas, f"  {obj_str[:42]}", rx, ry, 0.40, color)

        ry += 16
        # ── CNN CLASSIFIERS ──────────────────────────────────────────────────
        section_header(canvas, W+4, ry, SIDE-8, "CNN IMAGE CLASSIFIERS", (0, 80, 40))
        ry += 14

        for label, guess in [
            ("MobileNetV3", d["mob"]),
            ("ResNet-50",   d["res"]),
            ("Google ViT",  d["vit"]),
        ]:
            ry += 16
            put(canvas, f"{label}:", rx, ry, 0.42, C["grey"])
            ry += 14
            put(canvas, f"  {guess[:42]}", rx, ry, 0.40, C["white"])

        ry += 18
        # ── INFERENCE STATS ──────────────────────────────────────────────────
        section_header(canvas, W+4, ry, SIDE-8, "PERFORMANCE", (80, 50, 0))
        ry += 20
        put(canvas, f"Inference time :  {d['time']:.3f} s", rx, ry, 0.44, C["yellow"])
        ry += 16
        put(canvas, f"Last analysed  :  {d['ts']}", rx, ry, 0.41, C["grey"])
        ry += 16
        entries = sum(1 for _ in open(JSONL)) if os.path.exists(JSONL) else 0
        put(canvas, f"Dataset entries:  {entries}", rx, ry, 0.44, C["cyan"])
        ry += 14
        put(canvas, f"Saved to: {d['folder']}/cap_{d['ts_file']}.jpg", rx, ry, 0.37, C["grey"])

        ry += 14
        # ── VERDICT BOX ──────────────────────────────────────────────────────
        verd_col = (0, 30, 160) if fire_detected else (0, 100, 30)
        draw_panel(canvas, W+4, ry, SIDE-8, 30, alpha=0.92, color=verd_col)
        verd_txt = "🔥  VERDICT:  FIRE DETECTED" if fire_detected else "✅  VERDICT:  SCENE IS SAFE"
        put(canvas, verd_txt, rx+6, ry+20, 0.52, C["white"], 2)

    else:
        put(canvas, "Press  TAB  to start analysis", rx, H//2, 0.55, C["grey"])
        put(canvas, "All 8 models loaded & ready.", rx, H//2 + 28, 0.45, C["grey"])

    # ======================================================
    #  BOTTOM PANEL: live log feed
    # ======================================================
    log_x  = 8
    log_y0 = H + 18
    put(canvas, "LIVE SYSTEM LOG", log_x, log_y0, 0.52, C["yellow"])
    log_y0 += 8

    for i, (lvl, line) in enumerate(log_lines[-14:]):
        lc = {
            "FIRE": C["fire_red"],
            "SAFE": C["green"],
            "WARN": C["orange"],
            "DATA": C["cyan"],
            "STEP": C["grey"],
        }.get(lvl, C["white"])
        put_mono(canvas, line[:110], log_x, log_y0 + 18 + i * 20, 0.78, lc)

    # ── Step-by-step analysis in the bottom-right corner ─────────────────────
    if step_lines:
        sx = W + 8
        sy = H + 20
        put(canvas, "ANALYSIS STEPS (current frame)", sx, sy, 0.47, C["yellow"])
        for j, s in enumerate(step_lines[-12:]):
            put_mono(canvas, s[:50], sx, sy + 22 + j * 18, 0.75, C["grey"])

    # ======================================================
    # KEY: TAB  →  full analysis
    # ======================================================
    cv2.imshow("🔬 Fire Detection AI Lab — Science Exhibition", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:   # ESC
        break

    elif key == 9:  # TAB
        step_lines.clear()
        log("INFO", "TAB pressed — starting full 8-model analysis …")

        ts_file  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        total_t0 = time.time()

        # Convert frame once
        step(1, "Capturing frame from webcam")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(rgb_frame)
        log("STEP", f"Frame captured  ({frame.shape[1]}×{frame.shape[0]} px)")

        with torch.no_grad():

            # ── CLIP Base ────────────────────────────────────────────────────
            step(2, "Running CLIP-Base (zero-shot language-vision)")
            t0 = time.time()
            in_b  = proc_base(text=clip_labels, images=pil_img, return_tensors="pt", padding=True)
            p_base = clip_base(**in_b).logits_per_image.softmax(dim=1)[0][0].item() * 100
            log("STEP", f"CLIP-Base  → fire prob = {p_base:.2f}%  ({time.time()-t0:.3f}s)")

            # ── CLIP Large ───────────────────────────────────────────────────
            step(3, "Running CLIP-Large (zero-shot language-vision)")
            t0 = time.time()
            in_l   = proc_large(text=clip_labels, images=pil_img, return_tensors="pt", padding=True)
            p_large = clip_large(**in_l).logits_per_image.softmax(dim=1)[0][0].item() * 100
            log("STEP", f"CLIP-Large → fire prob = {p_large:.2f}%  ({time.time()-t0:.3f}s)")

            # ── SigLIP ───────────────────────────────────────────────────────
            step(4, "Running SigLIP (Google sigmoid vision-language)")
            t0 = time.time()
            in_s   = proc_siglip(text=clip_labels, images=pil_img, return_tensors="pt", padding="max_length")
            p_sig  = torch.sigmoid(siglip(**in_s).logits_per_image)[0][0].item() * 100
            log("STEP", f"SigLIP     → fire prob = {p_sig:.2f}%  ({time.time()-t0:.3f}s)")

            # ── YOLO General ─────────────────────────────────────────────────
            step(5, "Running YOLOv8 General (real-time object detector)")
            t0 = time.time()
            res_gen = yolo_general(rgb_frame, verbose=False)[0]
            det_gen = [yolo_general.names[int(b.cls)] for b in res_gen.boxes]
            log("STEP", f"YOLO-Gen   → {len(det_gen)} objects: {det_gen[:5]}  ({time.time()-t0:.3f}s)")

            # ── YOLO Fire ────────────────────────────────────────────────────
            step(6, "Running YOLOv8 Fire-Specialist (fire-trained)")
            t0 = time.time()
            res_fire = yolo_fire(rgb_frame, verbose=False)[0]
            det_fire = [yolo_fire.names[int(b.cls)] for b in res_fire.boxes]
            log("STEP", f"YOLO-Fire  → {len(det_fire)} detections: {det_fire}  ({time.time()-t0:.3f}s)")

            # ── MobileNet ────────────────────────────────────────────────────
            step(7, "Running MobileNetV3-Small (lightweight CNN)")
            t0 = time.time()
            in_mob   = mob_preprocess(pil_img).unsqueeze(0)
            mob_guess = imagenet_categories[mobilenet(in_mob).argmax(-1).item()]
            log("STEP", f"MobileNet  → '{mob_guess}'  ({time.time()-t0:.3f}s)")

            # ── ResNet-50 ────────────────────────────────────────────────────
            step(8, "Running ResNet-50 (deep residual CNN)")
            t0 = time.time()
            in_res   = res_preprocess(pil_img).unsqueeze(0)
            res_guess = imagenet_categories[resnet50(in_res).argmax(-1).item()]
            log("STEP", f"ResNet-50  → '{res_guess}'  ({time.time()-t0:.3f}s)")

            # ── ViT ──────────────────────────────────────────────────────────
            step(9, "Running Google ViT-Base (Vision Transformer)")
            t0 = time.time()
            vit_in   = vit_processor(images=pil_img, return_tensors="pt")
            vit_guess = vit_model.config.id2label[vit_model(**vit_in).logits.argmax(-1).item()]
            log("STEP", f"ViT-Base   → '{vit_guess}'  ({time.time()-t0:.3f}s)")

        total_time = time.time() - total_t0
        step(10, f"Total inference complete in {total_time:.3f}s")

        # ── DECISION LOGIC ───────────────────────────────────────────────────
        step(11, "Applying decision thresholds (65% confidence)")
        is_fire = (p_large > 65.0 or p_sig > 65.0 or
                   "fire" in [d.lower() for d in det_fire])

        reasons = []
        if p_large > 65.0:  reasons.append(f"CLIP-Large={p_large:.1f}%")
        if p_sig   > 65.0:  reasons.append(f"SigLIP={p_sig:.1f}%")
        if "fire" in [d.lower() for d in det_fire]: reasons.append("YOLO-Fire triggered")
        step(12, f"Decision: {'FIRE (' + ', '.join(reasons) + ')' if is_fire else 'SAFE — all thresholds OK'}")

        # ── SAVE IMAGE ───────────────────────────────────────────────────────
        step(13, "Saving annotated image to disk")
        folder   = "Fire_Evidence" if is_fire else "Safe_Captures"
        save_img = res_fire.plot() if (is_fire and res_fire.boxes) else frame
        photo_path = f"{folder}/cap_{ts_file}.jpg"
        cv2.imwrite(photo_path, save_img)

        # ── LOG ENTRY ────────────────────────────────────────────────────────
        step(14, f"Writing JSON-L entry to {JSONL}")
        log_entry = {
            "timestamp":      datetime.now().isoformat(),
            "detected_fire":  bool(is_fire),
            "inference_time": round(total_time, 3),
            "image_path":     photo_path,
            "decision_reasons": reasons,
            "results": {
                "clip_base_prob":      round(p_base,  2),
                "clip_large_prob":     round(p_large, 2),
                "siglip_prob":         round(p_sig,   2),
                "yolo_fire_objects":   det_fire,
                "yolo_general_objects": det_gen,
                "mobilenet_class":     mob_guess,
                "resnet50_class":      res_guess,
                "vit_class":           vit_guess,
            }
        }
        with open(JSONL, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # ── ALERT / CONFIRMATION ─────────────────────────────────────────────
        if is_fire:
            log("FIRE", f"🚨 FIRE! Triggers: {reasons}  |  saved → {photo_path}")
            winsound.Beep(2500, 1000)
        else:
            log("SAFE", f"Scene safe. Highest prob = {max(p_base, p_large, p_sig):.1f}%  |  saved → {photo_path}")

        log("DATA", f"Entry logged. Inference: {total_time:.3f}s | Total records: "
                    f"{sum(1 for _ in open(JSONL))}")

        # ── Store state for overlay ───────────────────────────────────────────
        last_data = {
            "p_base": p_base, "p_large": p_large, "p_sig": p_sig,
            "det_gen": det_gen, "det_fire": det_fire,
            "mob": mob_guess, "res": res_guess, "vit": vit_guess,
            "time": total_time, "is_fire": is_fire,
            "ts": datetime.now().strftime("%H:%M:%S"),
            "ts_file": ts_file, "folder": folder,
        }

cap.release()
cv2.destroyAllWindows()
print("\n[EXIT] Camera released. Dataset saved to:", JSONL)