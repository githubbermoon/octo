# 📋 UHI v1 — Ongoing Progress Log

> **Purpose:** Shared reference for all agents (Antigravity, Codex, etc.)  
> **Last Updated:** 2025-12-31 17:59 IST

---

## ✅ Completed Steps

### Step 0: EE Export (DONE)

- **Script:** `N1/ee_export_seasonal_scenes.py`
- **Output:** `Research_Data_Seasonal_Scenes/` on Drive
- **Sensors:** Sentinel-2, Landsat-8, Landsat-9, MODIS
- **Seasons:** Summer (Mar-May), Winter (Nov-Jan)
- **Years:** 2020-2025
- **Total exports:** ~498 images logged in `N1/export_log.csv`

### Step 1: Organize Exports (DONE)

- Files organized into sensor subfolders:
  ```
  Research_Data_Seasonal_Scenes/
  ├── Sentinel-2/
  ├── Landsat-8/
  ├── Landsat-9/
  └── MODIS/
  ```

### Step 2: Preprocessing (NEEDS RE-RUN)

- **Script:** `scripts/preprocess.py`
- **Config:** `configs/configD1.yaml`
- **Parameters:**
  - `tile_size: 256`
  - `overlap: 32`
  - `min_valid_frac: 0.7`
  - `clip_percentiles: [2, 98]`
  - `normalize: true` (skip for Landsat LST)

| Sensor     | Scenes | Tiles | Skipped | Output Path                        |
| ---------- | ------ | ----- | ------- | ---------------------------------- |
| Landsat-8  | 34     | 102   | 34      | `UHI_Project/processed/landsat8/`  |
| Landsat-9  | 21     | 62    | 22      | `UHI_Project/processed/landsat9/`  |
| Sentinel-2 | 71     | 3,222 | 257     | `UHI_Project/processed/sentinel2/` |
| MODIS      | 360    | 0     | 0       | Skipped (validation only)          |

**⚠️ BUGS FOUND (Jan 2026):**

| Bug                          | Cause                                     | Fix                                                     |
| ---------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| `bounds: scene-level`        | Saved `src.bounds` instead of tile bounds | Now computes tile-specific bounds from `tile_transform` |
| `LST normalized to 0-1`      | Applied min-max scaling to all data       | Added `skip_normalize_lst` (detects "landsat" in path)  |
| `transform stored as string` | numpy saved Affine as str                 | `fix_tile_bounds.py` parses string                      |

**Re-run preprocessing:**

```bash
# Clear old tiles
!rm -rf /content/drive/MyDrive/UHI_Project/processed/landsat8/tiles/*
!rm -rf /content/drive/MyDrive/UHI_Project/processed/landsat9/tiles/*

# Re-preprocess with fixes
!python scripts/preprocess.py --input /content/drive/MyDrive/Research_Data_Seasonal_Scenes/Landsat-8/ --output /content/drive/MyDrive/UHI_Project/processed/landsat8/
!python scripts/preprocess.py --input /content/drive/MyDrive/Research_Data_Seasonal_Scenes/Landsat-9/ --output /content/drive/MyDrive/UHI_Project/processed/landsat9/
```

**Total Landsat tiles:** 164 (target: LST in Kelvin)  
**Total Sentinel-2 tiles:** 3,222 (features: NDVI, NDBI normalized)

---

## 🔄 In Progress

### Step 3: Feature Alignment (UPDATED)

- **Script:** `scripts/align_features.py`
- **Purpose:** SOTA alignment of Sentinel-2 features with Landsat LST
- **Output:** Training pairs in `UHI_Project/processed/stacks/aligned/`

**Features (v2):**
| Feature | Status |
|---------|--------|
| Season matching (summer→summer) | ✅ |
| Year matching (2023→2023) | ✅ |
| Spatial overlap filtering | ✅ |
| Pixel-level resampling (area-weighted) | ✅ |
| LST band selection by name | ✅ |
| Verbose debug output | ✅ |
| `--limit N` flag for testing | ✅ |

**Run in Colab:**

```bash
# Test with 3 tiles
!python scripts/align_features.py --config configs/configD1.yaml --limit 3

# Full run
!python scripts/align_features.py --config configs/configD1.yaml
```

### Step 3b: Verify Alignment (NEW)

- **Script:** `scripts/inspect_aligned.py`
- **Purpose:** Verify aligned tile quality before training
- **Checks:**
  - Value ranges (NDVI, NDBI, LST)
  - Correlation NDVI-LST (should be negative)
  - Correlation NDBI-LST (should be positive)
  - NaN percentage

**Run in Colab:**

```bash
!python scripts/inspect_aligned.py --dir /content/drive/MyDrive/UHI_Project/processed/stacks/aligned/ --sample 5 --plot
```

### Step 3c: WorldCover Downloaded ✅

- **Script:** `scripts/download_worldcover.py`
- **Output:** `/content/drive/MyDrive/UHI_Project/raw_data/worldcover/worldcover_2021.tif`
- **Analysis:** `scripts/analyze_worldcover.py`

```
CLASS DISTRIBUTION:
  🔴 Urban (Built-up):     205,059 px (66.7%)
  🌿 Vegetation:            89,262 px (29.0%)
  💧 Water:                  3,773 px ( 1.2%)
✅ UHI analysis viable!
```

---

## 📋 TODO

### Step 4: Dataset Split

- Split into train/val/test (spatial block split)
- Create PyTorch DataLoader

### Step 5: Train XGBoost (Primary Baseline)

- Features: NDVI, NDBI, lat, lon
- Target: LST (Kelvin)
- Validation: Spatial CV (5-fold)

### Step 6: Train Random Forest (SHAP)

- For explainability plots

### Step 7: Train CNN (Novelty)

- Spatial context model
- Requires GPU

### Step 8: SHAP Analysis

- Feature importance
- Dependence plots

### Step 9: Validation

- Spatial CV metrics (RMSE, R², MAE)
- MODIS trend comparison
- WorldCover land cover validation

---

## 🔧 Key Configuration

```yaml
# From configs/configD1.yaml
spatial:
  crs: EPSG:32643
  base_resolution: 30

modeling:
  features: [NDVI, NDBI, latitude, longitude]
  target: LST_K
  # MODIS_LST is NOT a feature!

  xgboost: # PRIMARY
    n_estimators: 500
    max_depth: 8

  random_forest: # SHAP
    n_estimators: 300

  cnn: # NOVELTY
    epochs: 30
    batch_size: 16
```

---

## 📁 File Locations (Drive)

```
/content/drive/MyDrive/
├── Research_Data_Seasonal_Scenes/   # Raw EE exports (organized)
│   ├── Sentinel-2/
│   ├── Landsat-8/
│   ├── Landsat-9/
│   └── MODIS/
└── UHI_Project/
    ├── raw_data/                    # Alternative raw storage
    ├── processed/
    │   ├── landsat8/tiles/          # 102 .npz tiles
    │   ├── landsat9/tiles/          # 62 .npz tiles
    │   ├── sentinel2/tiles/         # 3222 .npz tiles
    │   ├── stacks/aligned/          # Feature-aligned pairs (after align_features.py)
    │   └── modis/                   # MODIS seasonal means (for validation)
    ├── models/
    │   ├── xgb/
    │   ├── rf/
    │   └── cnn/
    ├── validation/                  # Validation outputs
    │   ├── pixel_wise/              # Level 1 metrics
    │   └── regional/                # Level 2 MODIS comparison
    └── results/
```

**Quick Path Reference:**

- Raw exports: `/content/drive/MyDrive/Research_Data_Seasonal_Scenes/`
- Processed tiles: `/content/drive/MyDrive/UHI_Project/processed/`
- MODIS raw: `/content/drive/MyDrive/Research_Data_Seasonal_Scenes/MODIS/`

---

## ⚠️ Critical Rules (from AGENT_CONTEXT.md)

1. **MODIS = validation only** (NOT a training feature)
2. **Model order:** XGBoost → RF → CNN
3. **Seasons:** Summer + Winter only (no monsoon)
4. **CRS:** EPSG:32643 (UTM 43N)
5. **AOI:** Bangalore [77.55, 12.90, 77.70, 13.05]

---

## 📝 Agent Notes

> Add your notes here when making changes:

**[Antigravity 2025-12-31 18:00]:** Created preprocessing pipeline, organized exports, validated tile counts. Created `align_features.py` for spatial matching of S2→Landsat tiles.
**[Codex 2025-12-31 19:05]:** Added `scripts/preprocess.py`, wired preprocessing settings in `configs/configD1.yaml` and `params.yaml`, updated `README.md`, `PROJECT_STRUCTURE.md`, `.gitignore`, `requirements.txt`, and removed stale MLflow/CLI references (`pyproject.toml`, docs). Fixed AOI bounds in `params.yaml` to Bangalore and made `configs/drive1.py` safe for Colab mounting.
**[Codex 2025-12-31 19:14]:** Updated `scripts/download_worldcover.py` to default-save into Drive at `UHI_Project/raw_data/worldcover/` when no output/config is provided.

---

## 🤖 Instructions for Codex/Claude

**READ BEFORE MAKING ANY CHANGES:**

### 1. Always Check This File First

```
Before coding, review:
- ✅ Completed Steps (don't redo)
- 🔄 In Progress (coordinate)
- 📋 TODO (pick next task)
```

### 2. Update This File After Changes

Add a note in format:

```
**[YourName YYYY-MM-DD HH:MM]:** What you did, what files you changed.
```

### 3. Critical Rules (NEVER VIOLATE)

| Rule                            | Consequence if Violated      |
| ------------------------------- | ---------------------------- |
| MODIS is NOT a feature          | Data leakage, paper rejected |
| Model order: XGBoost → RF → CNN | Wrong baseline comparison    |
| Seasons: Summer + Winter only   | Data mismatch                |
| CRS: EPSG:32643                 | Spatial misalignment         |

### 4. Current Next Step

**Run feature alignment in Colab:**

```bash
!python scripts/align_features.py --config configs/configD1.yaml
```

**Then build:** `scripts/train_xgboost.py` (Step 5)

### 5. File Conventions

- Scripts: `scripts/<name>.py`
- Configs: `configs/<name>.yaml`
- Docs: `agentG/<name>.md`
- Logs: Update `ongoing.md`

### 6. When Creating Training Scripts

```python
# Required structure
features = ["NDVI", "NDBI", "lat", "lon"]  # From aligned tiles
target = "LST"  # Landsat LST in Kelvin
validation = "spatial_block_cv"  # NOT random split!
```

### 7. Reference Documents

- `AGENT_CONTEXT.md` — Master constraints
- `ongoing.md` — This file (progress log)
- `configs/configD1.yaml` — All parameters
- `agentG/plan1.md` — Execution plan
