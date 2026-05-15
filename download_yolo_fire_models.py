from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

MODELS = [
    {
        "name": "SHOU-ISD fire-and-smoke YOLOv8",
        "repo": "SHOU-ISD/fire-and-smoke",
        "year": "2023/2024",
        "preferred_files": ["fire-and-smoke.pt", "yolov8n_1.pt"],
    },
    {
        "name": "YOLOv8m smoke detection",
        "repo": "kittendev/YOLOv8m-smoke-detection",
        "year": "2023",
        "preferred_files": ["best.pt"],
    },
    {
        "name": "YOLOv8s forest fire detection",
        "repo": "touati-kamel/yolov8s-forest-fire-detection",
        "year": "2025",
        "preferred_files": ["best.pt", "model.pt"],
    },
    {
        "name": "YOLOv10 fire and smoke detection",
        "repo": "TommyNgx/YOLOv10-Fire-and-Smoke-Detection",
        "year": "2024/2025",
        "preferred_files": ["best.pt"],
    },
    {
        "name": "YOLO11 fire/smoke detector",
        "repo": "leeyunjai/yolo11-firedetect",
        "year": "2025",
        "preferred_files": ["best.pt", "model.pt"],
    },
    {
        "name": "YOLOv26 fire detection",
        "repo": "SalahALHaismawi/yolov26-fire-detection",
        "year": "2026",
        "preferred_files": ["best.pt", "model.pt"],
    },
    {
        "name": "ProFSAM YOLOv11n fire detector",
        "repo": "UEmmanuel5/ProFSAM-Fire-Detector",
        "year": "2025",
        "preferred_files": ["Fire_best.pt", "best.pt"],
    },
]

OUT_DIR = Path("models_yolo_fire")
OUT_DIR.mkdir(exist_ok=True)

def find_pt_file(repo_id: str, preferred_files: list[str]) -> str | None:
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"[ERROR] Cannot list files for {repo_id}: {e}")
        return None

    for f in preferred_files:
        if f in files:
            return f

    pt_files = [f for f in files if f.lower().endswith(".pt")]
    return pt_files[0] if pt_files else None

def main():
    manifest_lines = ["name,repo,year,local_path,status,error"]

    for m in MODELS:
        name = m["name"]
        repo = m["repo"]
        year = m["year"]

        print(f"\n=== Downloading: {name} | {repo} ===")

        try:
            filename = find_pt_file(repo, m["preferred_files"])

            if not filename:
                raise RuntimeError("No .pt file found in repo")

            local_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=OUT_DIR / repo.replace("/", "__"),
                local_dir_use_symlinks=False,
            )

            print(f"[OK] {name} -> {local_path}")
            manifest_lines.append(
                f'"{name}","{repo}","{year}","{local_path}","OK",""'
            )

        except Exception as e:
            print(f"[FAILED] {name}: {e}")
            manifest_lines.append(
                f'"{name}","{repo}","{year}","","FAILED","{str(e).replace(chr(34), chr(39))}"'
            )

    manifest_path = OUT_DIR / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"\nManifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()