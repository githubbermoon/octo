
import os

def try_mount_drive() -> bool:
    """Mount Google Drive when running inside an interactive Colab kernel."""
    try:
        from google.colab import drive
        from IPython import get_ipython
    except Exception:
        return False

    ip = get_ipython()
    if ip is None or not hasattr(ip, "kernel"):
        return False

    drive.mount("/content/drive")
    return True


mounted = try_mount_drive()

BASE_DIR = "/content/drive/MyDrive/UHI_Project"

folders = [
    "raw_data",
    "processed/sentinel2",
    "processed/landsat",
    "processed/modis",
    "processed/stacks",
    "processed/tables",
    "models/rf",
    "models/cnn",
    "results/maps",
    "results/plots",
    "figures_for_paper",
]

for folder in folders:
    path = os.path.join(BASE_DIR, folder)
    os.makedirs(path, exist_ok=True)
    print(f"[ok] Created: {path}")

if not mounted and not os.path.isdir(os.path.join(BASE_DIR, "raw_data")):
    print("Note: Drive is not mounted. Run drive.mount('/content/drive') in a notebook cell.")
