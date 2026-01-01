# UHI Pipeline Rules - Ground Truth Reference

**Last Updated:** 2026-01-02  
**Status:** CANONICAL - All scripts must comply

---

## 1. Sensor Roles

| Sensor      | Product | Variable                | Role                                 |
| ----------- | ------- | ----------------------- | ------------------------------------ |
| Landsat-8/9 | L2 C2   | LST (ST_B10)            | **Training target**                  |
| Sentinel-2  | L2A     | Reflectance + NDVI/NDBI | **Training features**                |
| MODIS       | LST     | LST                     | **Validation ONLY** (never training) |

---

## 2. Normalization Rules

### ✅ Sentinel-2

| Band Type            | Normalize? | Method             | Why                        |
| -------------------- | ---------- | ------------------ | -------------------------- |
| Reflectance (B2-B12) | **YES**    | min-max or z-score | ML needs comparable scales |
| NDVI                 | **NO**     | Keep as-is [-1, 1] | Physical index, bounded    |
| NDBI                 | **NO**     | Keep as-is [-1, 1] | Physical index, bounded    |

### ✅ Landsat-8/9

| Band Type    | Normalize?    | Method               | Why                  |
| ------------ | ------------- | -------------------- | -------------------- |
| LST (ST_B10) | **NO**        | Convert to K/°C only | Physical temperature |
| Reflectance  | YES (if used) | Same as S2           | N/A currently        |

### ❌ MODIS

**DO NOT TOUCH** - Validation baseline only.

---

## 3. Compositing Rules

| Variable    | Method         | Reason                       | Code Location           |
| ----------- | -------------- | ---------------------------- | ----------------------- |
| NDVI        | `np.nanmax`    | Peak vegetation, least cloud | `align_features.py:329` |
| NDBI        | `np.nanmedian` | Robust to SWIR noise         | `align_features.py:330` |
| LST         | mean/median    | Physically meaningful        | EE export               |
| Reflectance | median         | Robust to noise              | N/A currently           |

**❌ NEVER use `nanmean` for NDVI**

---

## 4. Processing Order

```
Sentinel-2 L2A
├─ 1. Cloud mask (EE: SCL)
├─ 2. Compute NDVI, NDBI
├─ 3. Export at native resolution
├─ 4. Preprocess: normalize reflectance ONLY
├─ 5. NDVI max composite, NDBI median composite
└─ 6. Align to Landsat grid

Landsat L2
├─ 1. Cloud mask (EE: QA_PIXEL)
├─ 2. ST_B10 → LST (scale + offset)
├─ 3. Export at 30m
├─ 4. NO normalization
└─ 5. Mean/median composite

MODIS
└─ Untouched → validation only
```

---

## 5. Script Compliance Checklist

### `scripts/preprocess.py` ✅

- [x] Landsat: skip all normalization
- [x] Sentinel-2: normalize reflectance, skip NDVI/NDBI
- [x] INDEX_BANDS = {"NDVI", "NDBI"} detection

### `scripts/align_seasonal.py` ✅ (NEW - Preferred)

- [x] One output per (year, season) — NOT per date
- [x] Same temporal window for NDVI/NDBI and LST
- [x] NDVI: `np.nanmax` (peak vegetation)
- [x] NDBI: `np.nanmedian` (robust)
- [x] LST: `np.nanmean` (seasonal mean)
- [x] `n_obs_lst` layer for quality control
- [x] Pixels with < min_observations masked
- [x] WorldCover integration (urban/rural masks)
- [x] `lst_aggregation` metadata stored

### `scripts/align_features.py` (DEPRECATED)

- [x] Per-date alignment (use align_seasonal.py instead)

### `pilot_plan/02_train_xgb.py` ✅

- [x] Spatial block CV (GroupKFold)
- [x] No additional normalization

---

## 6. Sanity Checks (Run Every Time)

```python
# After alignment, verify:
assert ndvi.max() >= 0.6, "NDVI max too low - normalization bug?"
assert 0.15 <= ndvi.mean() <= 0.30, "NDVI mean unexpected"
assert ndbi.max() >= 0.2, "NDBI max too low"
assert lst.min() >= 270, "LST not in Kelvin"
assert corr(ndvi, lst) < 0, "NDVI-LST should be negative"
assert corr(ndbi, lst) > 0, "NDBI-LST should be positive"
```

---

## 7. Common Failure Modes (NEVER REPEAT)

| ❌ Error                 | Why It's Wrong            |
| ------------------------ | ------------------------- |
| Normalizing NDVI/NDBI    | Destroys physical meaning |
| `nanmean` for NDVI       | Smooths vegetation signal |
| Normalizing after mosaic | Creates tile seams        |
| MODIS as training data   | Validation baseline only  |
| Random CV split          | Spatial leakage           |

---

## 8. Paper Language

> "MODIS LST was used solely for independent validation and not as a model input."

> "NDVI and NDBI were computed from Sentinel-2 L2A surface reflectance and retained in their native [-1, 1] range without additional normalization."

> "LST was derived from Landsat Collection 2 Level 2 thermal band (ST_B10) using the official scale factor and offset."
