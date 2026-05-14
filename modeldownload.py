import os
import torch
import torchvision.models as models
from transformers import (
    CLIPProcessor, CLIPModel, 
    AutoProcessor, AutoModel,
    ViTImageProcessor, ViTForImageClassification
)
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

print("="*60)
print("🚀 BOOTING SMART MODEL DOWNLOADER")
print("Checking local storage to prevent duplicate downloads...")
print("="*60 + "\n")

MAIN_DIR = "science_fair_models"
os.makedirs(MAIN_DIR, exist_ok=True)

# Helper function to keep our code clean
def skip_check(file_path, name):
    if os.path.exists(file_path):
        print(f"⏭️  SKIPPED: [{name}] is already downloaded!")
        return True
    return False

# ==========================================
# 1. ZERO-SHOT TRANSFORMERS
# ==========================================
print("\n--- PHASE 1: Zero-Shot Transformers ---")

os.makedirs(f"{MAIN_DIR}/clip_base", exist_ok=True)
if not skip_check(f"{MAIN_DIR}/clip_base/config.json", "CLIP Base"):
    print("[1/8] Downloading CLIP Base (OpenAI)...")
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(f"./{MAIN_DIR}/clip_base")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(f"./{MAIN_DIR}/clip_base")

os.makedirs(f"{MAIN_DIR}/clip_large", exist_ok=True)
if not skip_check(f"{MAIN_DIR}/clip_large/config.json", "CLIP Large"):
    print("[2/8] Downloading CLIP Large (OpenAI)...")
    CLIPModel.from_pretrained("openai/clip-vit-large-patch14").save_pretrained(f"./{MAIN_DIR}/clip_large")
    CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14").save_pretrained(f"./{MAIN_DIR}/clip_large")

os.makedirs(f"{MAIN_DIR}/siglip_base", exist_ok=True)
if not skip_check(f"{MAIN_DIR}/siglip_base/config.json", "SigLIP"):
    print("[3/8] Downloading SigLIP (Google)...")
    AutoModel.from_pretrained("google/siglip-base-patch16-224").save_pretrained(f"./{MAIN_DIR}/siglip_base")
    AutoProcessor.from_pretrained("google/siglip-base-patch16-224").save_pretrained(f"./{MAIN_DIR}/siglip_base")

os.makedirs(f"{MAIN_DIR}/vit_base", exist_ok=True)
if not skip_check(f"{MAIN_DIR}/vit_base/config.json", "Vision Transformer"):
    print("[4/8] Downloading Vision Transformer (Google)...")
    ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").save_pretrained(f"./{MAIN_DIR}/vit_base")
    ViTImageProcessor.from_pretrained("google/vit-base-patch16-224").save_pretrained(f"./{MAIN_DIR}/vit_base")


# ==========================================
# 2. REAL-TIME OBJECT DETECTORS (YOLO)
# ==========================================
print("\n--- PHASE 2: Real-Time Object Detectors ---")

os.makedirs(f"{MAIN_DIR}/yolo_general", exist_ok=True)
if not skip_check(f"{MAIN_DIR}/yolo_general/yolov8n.pt", "YOLOv8 Nano"):
    print("[5/8] Downloading YOLOv8 Nano (General Objects)...")
    YOLO("yolov8n.pt") 
    if os.path.exists("yolov8n.pt"):
        os.replace("yolov8n.pt", f"{MAIN_DIR}/yolo_general/yolov8n.pt")

os.makedirs(f"{MAIN_DIR}/yolo_fire", exist_ok=True)
# CHANGE 1: Look for model.pt in the skip check
if not skip_check(f"{MAIN_DIR}/yolo_fire/model.pt", "Fire YOLO"):
    print("[6/8] Downloading Specialist Fire YOLO...")
    hf_hub_download(
        repo_id="touati-kamel/yolov8s-forest-fire-detection", 
        filename="model.pt", # CHANGE 2: Download model.pt instead of best.pt
        local_dir=f"./{MAIN_DIR}/yolo_fire"
    )


# ==========================================
# 3. CLASSIC & EDGE CNNs
# ==========================================
print("\n--- PHASE 3: Edge Computing & Classic CNNs ---")

os.makedirs(f"{MAIN_DIR}/mobilenet", exist_ok=True)
if not skip_check(f"{MAIN_DIR}/mobilenet/mobilenet_v3_small.pth", "MobileNet"):
    print("[7/8] Downloading MobileNet V3 Small...")
    mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    torch.save(mobilenet.state_dict(), f"{MAIN_DIR}/mobilenet/mobilenet_v3_small.pth")

os.makedirs(f"{MAIN_DIR}/resnet", exist_ok=True)
if not skip_check(f"{MAIN_DIR}/resnet/resnet50.pth", "ResNet-50"):
    print("[8/8] Downloading ResNet-50...")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    torch.save(resnet50.state_dict(), f"{MAIN_DIR}/resnet/resnet50.pth")

print("\n" + "="*60)
print("✅ SYSTEM CHECK COMPLETE!")
print("All models are securely stored on your local hard drive.")
print("="*60)