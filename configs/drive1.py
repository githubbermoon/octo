
from google.colab import drive
import os

drive.mount("/content/drive")

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
    print(f"✔ Created: {path}")
