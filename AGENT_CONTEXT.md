# 🔴 AGENT CONTEXT — READ THIS FIRST

> **Project:** Urban Heat Island (UHI) v1 Research Paper  
> **Target:** IEEE-quality SOTA publication  
> **Last Updated:** 2025-12-31

> ⚠️ **EDITING RULE:** APPEND new information — NEVER delete existing context!

---

## ⚠️ CRITICAL CONSTRAINTS — DO NOT VIOLATE

### 1. MODIS is VALIDATION ONLY

```
❌ WRONG: Using MODIS_LST as a model input feature
✅ RIGHT: MODIS_LST is reserved for temporal trend validation
```

**Why:** Using MODIS as input then validating against MODIS = data leakage

### 2. Seasons are Summer & Winter ONLY

```
❌ WRONG: Including monsoon season
✅ RIGHT: Summer (Mar-May), Winter (Nov-Jan)
```

**Current pipeline only exports summer/winter data**

### 3. Model Order: XGBoost → RF → CNN

```
1. XGBoost (PRIMARY) — Fastest, run first
2. Random Forest (SHAP) — For interpretability
3. CNN (NOVELTY) — Spatial context, requires GPU
```

### 4. AOI is Bangalore, India

```yaml
bounds: [77.55, 12.90, 77.70, 13.05] # [west, south, east, north]
crs: EPSG:32643 # UTM 43N
resolution: 30m
```

### 5. WorldCover Downloaded ✅

```
AOI bounds: [77.55, 12.9, 77.7, 13.05]
Target CRS: EPSG:32643
Target resolution: 30m
WorldCover tile: ESA_WorldCover_10m_2021_v200_N12E075
Output: /content/drive/MyDrive/UHI_Project/raw_data/worldcover/worldcover_2021.tif
Shape: (1, 560, 549)
```

### 5b. WorldCover Analysis ✅

```
CLASS DISTRIBUTION:
  🔴 Urban (Built-up):     205,059 px (66.7%)
  🌿 Vegetation:            89,262 px (29.0%)
  💧 Water:                  3,773 px ( 1.2%)

✅ GOOD: Both urban and vegetation present!
   → Can compute UHI intensity (urban LST - vegetation LST)
```

### 6. Feature Stack (MODEL INPUTS)

```
[ NDVI, NDBI, latitude, longitude ]
```

**Target:** Landsat_LST (Kelvin)

---

## 📁 Project Structure

```
octo/
├── N1/                    # EE Export Scripts (data acquisition)
│   ├── ee_export_seasonal_scenes.py
│   └── export_log.csv
├── configs/
│   ├── configD1.yaml     # Main project config
│   └── drive1.py         # Colab Drive setup
├── agentG/
│   ├── plan1.md          # Execution plan (ALIGNED)
│   └── valid1.md         # Validation strategy (ALIGNED)
├── scripts/
│   ├── preprocess.py     # Tiling & normalization
│   ├── align_features.py # SOTA S2-Landsat alignment ← NEW
│   ├── inspect_aligned.py# Verify aligned outputs ← NEW
│   ├── download_worldcover.py # ESA WorldCover download
│   ├── analyze_worldcover.py  # Class distribution
│   └── validate_modis.py # MODIS trend validation
├── params.yaml           # DVC parameters
├── ongoing.md            # Progress log for agents
└── AGENT_CONTEXT.md      # THIS FILE
```

---

## 🗂️ Drive Folder Structure (Source of Truth)

```
/content/drive/MyDrive/UHI_Project
├── raw_data/
│   ├── Sentinel-2/
│   ├── Landsat-8/
│   ├── Landsat-9/
│   ├── MODIS/
│   └── worldcover/
├── processed/
│   ├── sentinel2/tiles/
│   ├── landsat8/tiles/
│   ├── landsat9/tiles/
│   ├── modis/tiles/
│   └── stacks/aligned/
├── models/
├── results/
└── figures_for_paper/
```

**IMPORTANT:** Raw data lives in `UHI_Project/raw_data/*` (NOT `Research_Data_Seasonal_Scenes`).  
Use this for preprocessing inputs:

```
/content/drive/MyDrive/UHI_Project/raw_data/<Sensor-Name>
```

---

## 📝 Recent Operations

- **Landsat-8 re-preprocessing (2025-12-31):**

  - Re-ran `scripts/preprocess.py` with debug enabled.
  - Confirmed 2x2 tiling per scene (`r0_c0`, `r0_c224`, `r224_c0`, `r224_c224`) given `tile_size=256`, `overlap=32`.
  - Tile transforms saved and verified; `band_names` = `LST_K`.
  - **Normalization:** skipped for Landsat LST tiles (per debug logs).
  - Output path: `/content/drive/MyDrive/UHI_Project/processed/landsat8`.

- **Sentinel-2 re-preprocessing (2026-01-01):**

  - 70/71 scenes processed (1 corrupted file skipped).
  - Total tiles: 3173
  - **Normalization FIX:** Reflectance bands normalized, NDVI/NDBI kept as-is [-1,1]

- **Feature Alignment Verified (2026-01-01):**

  - Pilot run with `--limit 3` successful.
  - LST range: 301-323 K ✅
  - corr(LST, NDVI): -0.321 ✅ (negative = more vegetation → cooler)
  - corr(LST, NDBI): +0.324 ✅ (positive = more built-up → hotter)
  - **Status:** Ready for XGBoost training!

---

## 🧪 Pilot Scripts Scope (from `pilot_scripts.md`)

- `align_features.py` is **pilot/debug only**; do not use for paper‑final metrics.
- `align_seasonal.py` union‑bounds + fixed 256×256 is **acceptable for Phase‑1 city‑scale analysis**, but must be revisited for pixel‑level ML.
- `validate_modis.py` is **validation only**; fallback logic is acceptable.
- Pilot plan assumes:
  - inputs from `align_features.py`
  - 256×256 tiles
  - NDVI/NDBI in native [-1, 1]
  - LST in Celsius
- Pilot outputs are **not paper‑final**.
- **Pipeline Fixes Applied (2026-01-02):**

  - `preprocess.py`: Skip NDVI/NDBI normalization (INDEX_BANDS detection)
  - `align_features.py`: nanmax for NDVI, nanmedian for NDBI
  - `align_features.py`: Temporal filter ±15 days of Landsat date
  - `02_train_xgb.py`: Spatial Block CV (GroupKFold)
  - **Reference:** See `PIPELINE_RULES.md` for canonical rules

- **Seasonal Aggregation Pipeline (2026-01-02):**
  - **NEW SCRIPT:** `scripts/align_seasonal.py` (preferred over `align_features.py`)
  - One output per (year, season) — NOT per Landsat date
  - Same temporal window for NDVI/NDBI and LST
  - `n_obs_lst` layer for quality control (min 3 observations)
  - WorldCover masks: `is_urban`, `is_vegetation`, `is_water`
  - `validate_features.py` for assertion-based validation

---

## 🎯 Data Pipeline

```
EE Export (done)           →  raw GeoTIFFs on Drive
                               ↓
Preprocess (scripts/)      →  256×256 tiles, normalized
                               ↓
Seasonal Alignment         →  One .npz per (year, season)
(align_seasonal.py)            with NDVI/NDBI/LST/n_obs
                               ↓
Validation                 →  validate_features.py
                               ↓
XGBoost/RF/CNN             →  Trained models
                               ↓
SHAP (RF only)             →  Explainability plots
                               ↓
Validation                 →  Spatial CV + MODIS trend comparison
```

---

## ✅ Config Reference

### configD1.yaml (Source of Truth)

| Section                                | Key Settings          |
| -------------------------------------- | --------------------- |
| `spatial.crs`                          | EPSG:32643            |
| `modeling.features`                    | NDVI, NDBI, lat, lon  |
| `modeling.xgboost`                     | Primary baseline      |
| `modeling.random_forest`               | SHAP interpretability |
| `validation.external.modis_comparison` | true                  |

### params.yaml

| Section                     | Key Settings          |
| --------------------------- | --------------------- |
| `download.aoi.bounds`       | Bangalore coordinates |
| `preprocess.tile_size`      | 256                   |
| `preprocess.min_valid_frac` | 0.7                   |

---

## 🚫 Common Mistakes to Avoid

| Mistake                     | Correct Approach                            |
| --------------------------- | ------------------------------------------- |
| Using California AOI bounds | Use Bangalore: [77.55, 12.90, 77.70, 13.05] |
| Adding MODIS_LST as feature | MODIS is for validation only                |
| Skipping XGBoost            | XGBoost first, RF second, CNN third         |
| Using monsoon season        | Only summer + winter                        |
| Random train-test split     | Use spatial block CV                        |
| Pointwise MODIS validation  | Trend agreement only (1 km vs 30 m)         |

---

## 📋 Agent Prompts

### When Starting Work

> "Before making changes, I will review AGENT_CONTEXT.md to ensure alignment with project constraints."

### When Adding Features

> "I confirm this feature is in the approved list: [NDVI, NDBI, lat, lon]. MODIS_LST is NOT used as a feature."

### When Modifying Modeling

> "Model order is XGBoost → RF → CNN. XGBoost is PRIMARY baseline."

### When Changing Validation

> "MODIS is used for temporal trend validation only, not pixel-wise comparison."

---

## 📚 Reference Documents

| Document                  | Purpose                        |
| ------------------------- | ------------------------------ |
| `agentG/plan1.md`         | Full execution plan            |
| `agentG/valid1.md`        | Validation strategy            |
| `configs/configD1.yaml`   | Project configuration          |
| `N1/SESSION_CHANGELOG.md` | EE export script documentation |

---

**REMEMBER:** When in doubt, check this file. Consistency is critical for SOTA research.
