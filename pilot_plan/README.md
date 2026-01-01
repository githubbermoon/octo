# Pilot Pipeline - UHI v1

Scripts for running the complete UHI analysis pipeline on a small subset of data.

## Usage (Colab)

```bash
# Step 1: Prepare data
!python pilot_plan/01_prepare_data.py \
    --aligned-dir /content/drive/MyDrive/UHI_Project/processed/stacks/aligned/ \
    --worldcover /content/drive/MyDrive/UHI_Project/raw_data/worldcover/worldcover_2021.tif \
    --output /content/drive/MyDrive/UHI_Project/pilot_train.csv

# Step 2: Train XGBoost
!python pilot_plan/02_train_xgb.py \
    --data /content/drive/MyDrive/UHI_Project/pilot_train.csv \
    --output /content/drive/MyDrive/UHI_Project/models/

# Step 3: SHAP analysis
!python pilot_plan/03_shap_analysis.py \
    --model /content/drive/MyDrive/UHI_Project/models/xgb_pilot.json \
    --data /content/drive/MyDrive/UHI_Project/pilot_train.csv \
    --output /content/drive/MyDrive/UHI_Project/figures/

# Step 4: Visualize & UHI intensity
!python pilot_plan/04_visualize_uhi.py \
    --model /content/drive/MyDrive/UHI_Project/models/xgb_pilot.json \
    --aligned-dir /content/drive/MyDrive/UHI_Project/processed/stacks/aligned/ \
    --output /content/drive/MyDrive/UHI_Project/figures/
```

## Success Criteria

| Metric        | Target |
| ------------- | ------ |
| RMSE          | < 3 K  |
| R²            | > 0.5  |
| UHI intensity | 2-5 K  |

## Files

- `01_prepare_data.py` - Flatten tiles to CSV with WorldCover features
- `02_train_xgb.py` - Train XGBoost with spatial validation
- `03_shap_analysis.py` - Feature importance & dependence plots
- `04_visualize_uhi.py` - Prediction maps & UHI calculation
