#!/usr/bin/env python3
"""
Research-Grade Feature Validation Module for UHI Pipeline.

This module provides strict, assertion-based validation for all EO features
in the Urban Heat Island analysis pipeline. It catches silent pipeline errors
that visual inspection often misses.

Author: UHI Research Team
Last Updated: 2026-01-02

Usage:
    from scripts.validate_features import run_all_validations
    
    features = {"ndvi": ndvi_arr, "ndbi": ndbi_arr, "lst": lst_arr}
    run_all_validations(features, season="summer")

Design Philosophy:
    - Fail fast on hard errors
    - Warn on soft anomalies
    - All assertions have human-readable messages
    - Season-aware thresholds
    - No silent passes
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

# =============================================================================
# SEASON-SPECIFIC RULES
# =============================================================================

@dataclass
class SeasonRules:
    """Season-specific validation thresholds."""
    # NDVI thresholds
    ndvi_max_min: float  # Minimum acceptable max NDVI
    ndvi_mean_min: float  # Minimum acceptable mean NDVI
    ndvi_mean_max: float  # Maximum acceptable mean NDVI (too high = bug)
    
    # NDBI thresholds
    ndbi_max_min: float  # Minimum acceptable max NDBI (built-up signal)
    ndbi_mean_min: float  # Minimum acceptable mean NDBI
    ndbi_mean_max: float  # Maximum acceptable mean NDBI
    
    # LST thresholds (Kelvin)
    lst_min_k: float  # Minimum realistic LST
    lst_max_k: float  # Maximum realistic LST
    lst_urban_min_k: float  # Minimum urban hotspot temperature
    lst_dynamic_range_min: float  # Minimum LST range (detect over-smoothing)


SEASON_RULES: dict[str, SeasonRules] = {
    "summer": SeasonRules(
        # Summer in Bangalore: hot, pre-monsoon stress, lower vegetation
        ndvi_max_min=0.50,  # Parks should still have NDVI > 0.5
        ndvi_mean_min=0.10,
        ndvi_mean_max=0.50,  # If mean > 0.5, likely a bug
        ndbi_max_min=0.15,  # Built-up areas must show
        ndbi_mean_min=0.00,
        ndbi_mean_max=0.30,
        lst_min_k=290.0,  # ~17°C minimum
        lst_max_k=330.0,  # ~57°C maximum (hot asphalt)
        lst_urban_min_k=305.0,  # ~32°C (urban daytime minimum)
        lst_dynamic_range_min=8.0,  # At least 8K variation
    ),
    "winter": SeasonRules(
        # Winter in Bangalore: cooler, less vegetation stress
        ndvi_max_min=0.55,  # Slightly higher vegetation
        ndvi_mean_min=0.12,
        ndvi_mean_max=0.55,
        ndbi_max_min=0.15,
        ndbi_mean_min=0.00,
        ndbi_mean_max=0.30,
        lst_min_k=280.0,  # ~7°C minimum
        lst_max_k=315.0,  # ~42°C maximum
        lst_urban_min_k=295.0,  # ~22°C
        lst_dynamic_range_min=6.0,
    ),
    "post_monsoon": SeasonRules(
        # Post-monsoon: lush vegetation, moderate temps
        ndvi_max_min=0.65,  # High vegetation after rain
        ndvi_mean_min=0.20,
        ndvi_mean_max=0.60,
        ndbi_max_min=0.12,
        ndbi_mean_min=-0.05,
        ndbi_mean_max=0.25,
        lst_min_k=285.0,  # ~12°C
        lst_max_k=320.0,  # ~47°C
        lst_urban_min_k=298.0,  # ~25°C
        lst_dynamic_range_min=5.0,
    ),
}


def _get_rules(season: str) -> SeasonRules:
    """Get season rules with fallback."""
    if season not in SEASON_RULES:
        warnings.warn(f"Unknown season '{season}', using 'summer' defaults")
        return SEASON_RULES["summer"]
    return SEASON_RULES[season]


# =============================================================================
# CORE VALIDATORS
# =============================================================================

class ValidationError(AssertionError):
    """Raised when a hard validation fails."""
    pass


class ValidationWarning(UserWarning):
    """Issued for soft validation anomalies."""
    pass


def validate_ndvi(
    arr: np.ndarray,
    season: str = "summer",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate NDVI array for physical plausibility.
    
    Checks:
        1. Not all NaN
        2. Range within [-1, 1]
        3. Max NDVI >= season threshold (vegetation exists)
        4. Mean NDVI >= season threshold (not collapsed to zero)
        5. Warn if too uniform (normalization bug)
    
    Args:
        arr: 2D NDVI array
        season: One of "summer", "winter", "post_monsoon"
        strict: If True, raise on failure. If False, return status.
    
    Returns:
        Dict with validation results
    
    Raises:
        ValidationError: On hard failure (if strict=True)
    """
    rules = _get_rules(season)
    results = {"valid": True, "errors": [], "warnings": []}
    
    # Flatten and get valid values
    valid = arr[np.isfinite(arr)]
    
    # Check 1: Not all NaN
    if len(valid) == 0:
        msg = "NDVI: Array is entirely NaN/invalid"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
        return results
    
    nan_frac = 1 - len(valid) / arr.size
    if nan_frac > 0.5:
        msg = f"NDVI: {nan_frac:.1%} NaN values (>50%)"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 2: Range within [-1, 1]
    arr_min, arr_max = float(valid.min()), float(valid.max())
    
    if arr_min < -1.0 or arr_max > 1.0:
        msg = f"NDVI: Out of bounds [{arr_min:.3f}, {arr_max:.3f}], expected [-1, 1]"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 2b: Detect double-normalization (compressed to [0, ~0.3])
    if arr_min >= 0 and arr_max < 0.35 and arr_max > 0.1:
        msg = f"NDVI: Suspiciously compressed range [{arr_min:.3f}, {arr_max:.3f}] — possible double normalization"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 3: Max NDVI (vegetation must exist)
    if arr_max < rules.ndvi_max_min:
        msg = f"NDVI: Max={arr_max:.3f} < {rules.ndvi_max_min} — no vegetation detected ({season})"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 4: Mean NDVI
    arr_mean = float(valid.mean())
    
    if arr_mean < rules.ndvi_mean_min:
        msg = f"NDVI: Mean={arr_mean:.3f} < {rules.ndvi_mean_min} — signal collapsed ({season})"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    if arr_mean > rules.ndvi_mean_max:
        msg = f"NDVI: Mean={arr_mean:.3f} > {rules.ndvi_mean_max} — unrealistic for urban area ({season})"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 5: Uniformity (std too low = normalization bug)
    arr_std = float(valid.std())
    if arr_std < 0.03:
        msg = f"NDVI: Std={arr_std:.4f} — suspiciously uniform (possible bug)"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Store stats
    results["stats"] = {
        "min": arr_min,
        "max": arr_max,
        "mean": arr_mean,
        "std": arr_std,
        "nan_frac": nan_frac,
    }
    
    return results


def validate_ndbi(
    arr: np.ndarray,
    season: str = "summer",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate NDBI array for physical plausibility.
    
    Checks:
        1. Not all NaN
        2. Range within [-1, 1]
        3. Max NDBI >= threshold (built-up areas exist)
        4. Mean NDBI >= threshold (signal not collapsed)
        5. Built-up signal not uniformly low
    
    Args:
        arr: 2D NDBI array
        season: Season identifier
        strict: Raise on failure if True
    
    Returns:
        Dict with validation results
    """
    rules = _get_rules(season)
    results = {"valid": True, "errors": [], "warnings": []}
    
    valid = arr[np.isfinite(arr)]
    
    # Check 1: Not all NaN
    if len(valid) == 0:
        msg = "NDBI: Array is entirely NaN/invalid"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
        return results
    
    nan_frac = 1 - len(valid) / arr.size
    
    # Check 2: Range
    arr_min, arr_max = float(valid.min()), float(valid.max())
    
    if arr_min < -1.0 or arr_max > 1.0:
        msg = f"NDBI: Out of bounds [{arr_min:.3f}, {arr_max:.3f}], expected [-1, 1]"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 2b: Detect double-normalization
    if arr_min >= 0 and arr_max < 0.35 and arr_max > 0.1:
        msg = f"NDBI: Suspiciously compressed range [{arr_min:.3f}, {arr_max:.3f}] — possible double normalization"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 3: Max NDBI (built-up must exist in urban area)
    if arr_max < rules.ndbi_max_min:
        msg = f"NDBI: Max={arr_max:.3f} < {rules.ndbi_max_min} — no built-up signal ({season})"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 4: Mean NDBI
    arr_mean = float(valid.mean())
    
    if arr_mean < rules.ndbi_mean_min:
        msg = f"NDBI: Mean={arr_mean:.3f} < {rules.ndbi_mean_min} — signal collapsed ({season})"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 5: Std
    arr_std = float(valid.std())
    if arr_std < 0.02:
        msg = f"NDBI: Std={arr_std:.4f} — suspiciously uniform"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    results["stats"] = {
        "min": arr_min,
        "max": arr_max,
        "mean": arr_mean,
        "std": arr_std,
        "nan_frac": nan_frac,
    }
    
    return results


def validate_lst(
    arr: np.ndarray,
    season: str = "summer",
    unit: Literal["K", "C"] = "K",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate LST array for physical plausibility.
    
    Checks:
        1. Not all NaN
        2. Realistic min/max for season
        3. Urban daytime minimum temperature
        4. Dynamic range (detect over-smoothing)
        5. No impossible values
    
    Args:
        arr: 2D LST array
        season: Season identifier
        unit: "K" for Kelvin, "C" for Celsius
        strict: Raise on failure if True
    
    Returns:
        Dict with validation results
    """
    rules = _get_rules(season)
    results = {"valid": True, "errors": [], "warnings": []}
    
    # Convert to Kelvin if needed
    arr_k = arr if unit == "K" else arr + 273.15
    
    valid = arr_k[np.isfinite(arr_k)]
    
    # Check 1: Not all NaN
    if len(valid) == 0:
        msg = "LST: Array is entirely NaN/invalid"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
        return results
    
    nan_frac = 1 - len(valid) / arr_k.size
    if nan_frac > 0.3:
        msg = f"LST: {nan_frac:.1%} NaN values (>30%)"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    arr_min, arr_max = float(valid.min()), float(valid.max())
    arr_mean = float(valid.mean())
    
    # Check 2: Realistic range
    if arr_min < rules.lst_min_k:
        msg = f"LST: Min={arr_min:.1f}K < {rules.lst_min_k}K — too cold for {season}"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    if arr_max > rules.lst_max_k:
        msg = f"LST: Max={arr_max:.1f}K > {rules.lst_max_k}K — unrealistically hot"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 3: Urban minimum
    if arr_max < rules.lst_urban_min_k:
        msg = f"LST: Max={arr_max:.1f}K < {rules.lst_urban_min_k}K — no urban hotspot detected ({season})"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 4: Dynamic range
    dynamic_range = arr_max - arr_min
    if dynamic_range < rules.lst_dynamic_range_min:
        msg = f"LST: Dynamic range={dynamic_range:.1f}K < {rules.lst_dynamic_range_min}K — over-smoothed?"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 5: Possible Kelvin/Celsius confusion
    if arr_min < 200:  # Clearly not Kelvin
        msg = f"LST: Min={arr_min:.1f} — values seem like Celsius but unit={unit}"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    if arr_max > 400:  # Too hot for Earth
        msg = f"LST: Max={arr_max:.1f} — impossible surface temperature"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    results["stats"] = {
        "min_k": arr_min,
        "max_k": arr_max,
        "mean_k": arr_mean,
        "dynamic_range_k": dynamic_range,
        "nan_frac": nan_frac,
    }
    
    return results


def validate_reflectance(
    arr: np.ndarray,
    band_name: str = "unknown",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate reflectance band for normalization correctness.
    
    Checks:
        1. Values within [0, 1] (normalized)
        2. Not normalized twice (compressed variance)
        3. Reasonable variance (not flat)
        4. No tile-edge spikes
    
    Args:
        arr: 2D reflectance array
        band_name: Band identifier for error messages
        strict: Raise on failure if True
    
    Returns:
        Dict with validation results
    """
    results = {"valid": True, "errors": [], "warnings": []}
    
    valid = arr[np.isfinite(arr)]
    
    if len(valid) == 0:
        msg = f"Reflectance ({band_name}): Array is entirely NaN"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
        return results
    
    arr_min, arr_max = float(valid.min()), float(valid.max())
    arr_mean = float(valid.mean())
    arr_std = float(valid.std())
    
    # Check 1: Values in [0, 1]
    if arr_min < 0 or arr_max > 1:
        # Check if raw DN (not normalized)
        if arr_max > 10000:
            msg = f"Reflectance ({band_name}): Range [{arr_min:.0f}, {arr_max:.0f}] — appears to be raw DN, not normalized"
        else:
            msg = f"Reflectance ({band_name}): Range [{arr_min:.3f}, {arr_max:.3f}] — out of expected [0, 1]"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    
    # Check 2: Not double-normalized (very low variance)
    if arr_std < 0.01:
        msg = f"Reflectance ({band_name}): Std={arr_std:.4f} — suspiciously low (double normalization?)"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 3: Not flat
    if arr_max - arr_min < 0.05:
        msg = f"Reflectance ({band_name}): Range only {arr_max - arr_min:.3f} — flat image?"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 4: Edge spikes (check border vs interior)
    if arr.ndim == 2 and arr.shape[0] > 10 and arr.shape[1] > 10:
        border = np.concatenate([
            arr[0, :].flatten(),
            arr[-1, :].flatten(),
            arr[:, 0].flatten(),
            arr[:, -1].flatten(),
        ])
        interior = arr[5:-5, 5:-5].flatten()
        
        border_valid = border[np.isfinite(border)]
        interior_valid = interior[np.isfinite(interior)]
        
        if len(border_valid) > 0 and len(interior_valid) > 0:
            border_mean = np.mean(border_valid)
            interior_mean = np.mean(interior_valid)
            
            if abs(border_mean - interior_mean) > 0.15:
                msg = f"Reflectance ({band_name}): Border mean={border_mean:.3f}, interior={interior_mean:.3f} — edge artifact?"
                results["warnings"].append(msg)
                warnings.warn(msg, ValidationWarning)
    
    results["stats"] = {
        "min": arr_min,
        "max": arr_max,
        "mean": arr_mean,
        "std": arr_std,
    }
    
    return results


# =============================================================================
# TILE SEAM DETECTION
# =============================================================================

def detect_tile_seams(
    arr: np.ndarray,
    threshold_ratio: float = 3.0,
    edge_width: int = 2,
) -> dict[str, Any]:
    """
    Detect Sentinel-2 tile boundary artifacts.
    
    Looks for abrupt step changes caused by:
        - Mosaic date mismatch
        - Atmospheric differences between tiles
        - Improper resampling
    
    Algorithm:
        1. Compute horizontal and vertical gradients
        2. Find high-gradient rows/columns
        3. Compare seam gradients to interior gradients
        4. Flag if seam contrast > threshold_ratio × interior contrast
    
    Args:
        arr: 2D array (NDVI, NDBI, or reflectance)
        threshold_ratio: Seam/interior gradient ratio to trigger failure
        edge_width: Width of edge to check (pixels)
    
    Returns:
        Dict with seam detection results and mask
    
    Raises:
        ValidationError: If significant seams detected
    """
    results = {"has_seams": False, "seam_rows": [], "seam_cols": [], "errors": [], "warnings": []}
    
    if arr.ndim != 2:
        results["warnings"].append("Seam detection requires 2D array")
        return results
    
    h, w = arr.shape
    if h < 20 or w < 20:
        results["warnings"].append("Array too small for seam detection")
        return results
    
    # Replace NaN with local mean for gradient computation
    arr_filled = arr.copy()
    nan_mask = ~np.isfinite(arr_filled)
    if nan_mask.any():
        arr_filled[nan_mask] = np.nanmean(arr_filled)
    
    # Compute gradients
    grad_h = np.abs(np.diff(arr_filled, axis=0))  # Horizontal seams (row differences)
    grad_v = np.abs(np.diff(arr_filled, axis=1))  # Vertical seams (col differences)
    
    # Compute interior gradient statistics (exclude borders)
    interior_h = grad_h[10:-10, 10:-10]
    interior_v = grad_v[10:-10, 10:-10]
    
    interior_h_mean = float(np.nanmean(interior_h)) if interior_h.size > 0 else 0.01
    interior_v_mean = float(np.nanmean(interior_v)) if interior_v.size > 0 else 0.01
    
    # Avoid division by zero
    interior_h_mean = max(interior_h_mean, 0.001)
    interior_v_mean = max(interior_v_mean, 0.001)
    
    # Find rows/columns with high gradient (potential seams)
    seam_rows = []
    seam_cols = []
    
    # Check each row for horizontal seams
    for row_idx in range(grad_h.shape[0]):
        row_grad = float(np.nanmean(grad_h[row_idx, :]))
        if row_grad > threshold_ratio * interior_h_mean:
            seam_rows.append(row_idx)
    
    # Check each column for vertical seams
    for col_idx in range(grad_v.shape[1]):
        col_grad = float(np.nanmean(grad_v[:, col_idx]))
        if col_grad > threshold_ratio * interior_v_mean:
            seam_cols.append(col_idx)
    
    results["seam_rows"] = seam_rows
    results["seam_cols"] = seam_cols
    results["interior_h_mean"] = interior_h_mean
    results["interior_v_mean"] = interior_v_mean
    
    # Create seam mask
    seam_mask = np.zeros(arr.shape, dtype=bool)
    for r in seam_rows:
        seam_mask[r:r+edge_width, :] = True
    for c in seam_cols:
        seam_mask[:, c:c+edge_width] = True
    
    results["seam_mask"] = seam_mask
    
    # Determine severity
    n_seams = len(seam_rows) + len(seam_cols)
    
    if n_seams > 5:
        msg = f"Tile seams: {len(seam_rows)} horizontal, {len(seam_cols)} vertical seams detected"
        results["errors"].append(msg)
        results["has_seams"] = True
    elif n_seams > 0:
        msg = f"Tile seams: {n_seams} potential seam(s) detected (may be natural features)"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    return results


# =============================================================================
# CROSS-VARIABLE CONSISTENCY
# =============================================================================

def validate_physical_consistency(
    ndvi: np.ndarray,
    ndbi: np.ndarray,
    lst: np.ndarray,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Validate physical consistency between NDVI, NDBI, and LST.
    
    Physical expectations:
        1. NDVI vs LST → NEGATIVE correlation (more vegetation → cooler)
        2. NDBI vs LST → POSITIVE correlation (more built-up → hotter)
        3. Water pixels should not be the hottest
        4. High-vegetation pixels should not be the hottest
    
    Args:
        ndvi: 2D NDVI array
        ndbi: 2D NDBI array
        lst: 2D LST array (Kelvin)
        strict: Raise on failure if True
    
    Returns:
        Dict with consistency check results
    
    Raises:
        ValidationError: If physical relationships are violated
    """
    results = {"valid": True, "errors": [], "warnings": []}
    
    # Flatten and create valid mask
    valid = (
        np.isfinite(ndvi.flatten()) &
        np.isfinite(ndbi.flatten()) &
        np.isfinite(lst.flatten())
    )
    
    if valid.sum() < 100:
        msg = "Physical consistency: Too few valid pixels for correlation"
        results["warnings"].append(msg)
        return results
    
    ndvi_flat = ndvi.flatten()[valid]
    ndbi_flat = ndbi.flatten()[valid]
    lst_flat = lst.flatten()[valid]
    
    # Compute correlations
    corr_ndvi_lst = float(np.corrcoef(ndvi_flat, lst_flat)[0, 1])
    corr_ndbi_lst = float(np.corrcoef(ndbi_flat, lst_flat)[0, 1])
    
    results["correlations"] = {
        "ndvi_lst": corr_ndvi_lst,
        "ndbi_lst": corr_ndbi_lst,
    }
    
    # Check 1: NDVI vs LST should be negative
    if corr_ndvi_lst > 0.1:  # Allow small positive (noise)
        msg = f"Physical consistency: corr(NDVI, LST)={corr_ndvi_lst:.3f} > 0 — should be NEGATIVE (more vegetation → cooler)"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    elif corr_ndvi_lst > -0.1:
        msg = f"Physical consistency: corr(NDVI, LST)={corr_ndvi_lst:.3f} — weak negative (expected < -0.2)"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 2: NDBI vs LST should be positive
    if corr_ndbi_lst < -0.1:  # Allow small negative (noise)
        msg = f"Physical consistency: corr(NDBI, LST)={corr_ndbi_lst:.3f} < 0 — should be POSITIVE (more built-up → hotter)"
        results["errors"].append(msg)
        results["valid"] = False
        if strict:
            raise ValidationError(msg)
    elif corr_ndbi_lst < 0.1:
        msg = f"Physical consistency: corr(NDBI, LST)={corr_ndbi_lst:.3f} — weak positive (expected > 0.2)"
        results["warnings"].append(msg)
        warnings.warn(msg, ValidationWarning)
    
    # Check 3: Water pixels (low NDBI, low NDVI) should not be hottest
    # Approximate water: NDVI < 0.1 and NDBI < -0.1
    water_mask = (ndvi_flat < 0.1) & (ndbi_flat < -0.1)
    
    if water_mask.sum() > 10:
        water_lst = lst_flat[water_mask]
        other_lst = lst_flat[~water_mask]
        
        if np.mean(water_lst) > np.percentile(other_lst, 90):
            msg = f"Physical consistency: Water pixels (mean LST={np.mean(water_lst):.1f}K) hotter than 90th percentile of land"
            results["warnings"].append(msg)
            warnings.warn(msg, ValidationWarning)
    
    # Check 4: High-vegetation pixels should not be hottest
    veg_mask = ndvi_flat > 0.5
    
    if veg_mask.sum() > 10:
        veg_lst = lst_flat[veg_mask]
        other_lst = lst_flat[~veg_mask]
        
        if np.mean(veg_lst) > np.mean(other_lst):
            msg = f"Physical consistency: Vegetation (mean LST={np.mean(veg_lst):.1f}K) hotter than other pixels ({np.mean(other_lst):.1f}K)"
            results["warnings"].append(msg)
            warnings.warn(msg, ValidationWarning)
    
    return results


# =============================================================================
# MAIN DRIVER
# =============================================================================

def run_all_validations(
    features: dict[str, np.ndarray],
    season: str = "summer",
    strict: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run all validation checks on a feature dictionary.
    
    Args:
        features: Dict with keys like "ndvi", "ndbi", "lst", "B2", "B3", etc.
        season: Season identifier
        strict: Raise on hard failures if True
        verbose: Print diagnostics
    
    Returns:
        Dict with all validation results
    
    Raises:
        ValidationError: On hard failure (if strict=True)
    """
    results = {
        "season": season,
        "strict": strict,
        "all_valid": True,
        "n_errors": 0,
        "n_warnings": 0,
        "validators": {},
    }
    
    if verbose:
        print("=" * 60)
        print(f"FEATURE VALIDATION — Season: {season.upper()}")
        print("=" * 60)
    
    all_errors = []
    all_warnings = []
    
    # Normalize keys to lowercase
    features_lower = {k.lower(): v for k, v in features.items()}
    
    # 1. Validate NDVI
    if "ndvi" in features_lower:
        if verbose:
            print("\n📗 Validating NDVI...")
        res = validate_ndvi(features_lower["ndvi"], season, strict=False)
        results["validators"]["ndvi"] = res
        all_errors.extend(res["errors"])
        all_warnings.extend(res["warnings"])
        if verbose:
            if res["valid"]:
                stats = res["stats"]
                print(f"   ✅ Range: [{stats['min']:.3f}, {stats['max']:.3f}], Mean: {stats['mean']:.3f}")
            else:
                for err in res["errors"]:
                    print(f"   ❌ {err}")
    
    # 2. Validate NDBI
    if "ndbi" in features_lower:
        if verbose:
            print("\n📙 Validating NDBI...")
        res = validate_ndbi(features_lower["ndbi"], season, strict=False)
        results["validators"]["ndbi"] = res
        all_errors.extend(res["errors"])
        all_warnings.extend(res["warnings"])
        if verbose:
            if res["valid"]:
                stats = res["stats"]
                print(f"   ✅ Range: [{stats['min']:.3f}, {stats['max']:.3f}], Mean: {stats['mean']:.3f}")
            else:
                for err in res["errors"]:
                    print(f"   ❌ {err}")
    
    # 3. Validate LST
    if "lst" in features_lower:
        if verbose:
            print("\n🌡️  Validating LST...")
        res = validate_lst(features_lower["lst"], season, strict=False)
        results["validators"]["lst"] = res
        all_errors.extend(res["errors"])
        all_warnings.extend(res["warnings"])
        if verbose:
            if res["valid"]:
                stats = res["stats"]
                print(f"   ✅ Range: [{stats['min_k']:.1f}K, {stats['max_k']:.1f}K], Dynamic range: {stats['dynamic_range_k']:.1f}K")
            else:
                for err in res["errors"]:
                    print(f"   ❌ {err}")
    
    # 4. Validate reflectance bands
    reflectance_bands = [k for k in features_lower if k.startswith("b") and k[1:].isdigit()]
    for band in reflectance_bands:
        if verbose:
            print(f"\n🔵 Validating {band.upper()}...")
        res = validate_reflectance(features_lower[band], band.upper(), strict=False)
        results["validators"][band] = res
        all_errors.extend(res["errors"])
        all_warnings.extend(res["warnings"])
        if verbose:
            if res["valid"]:
                stats = res["stats"]
                print(f"   ✅ Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            else:
                for err in res["errors"]:
                    print(f"   ❌ {err}")
    
    # 5. Tile seam detection (on NDVI as representative)
    if "ndvi" in features_lower:
        if verbose:
            print("\n🔲 Checking for tile seams...")
        res = detect_tile_seams(features_lower["ndvi"])
        results["validators"]["tile_seams"] = res
        if res["has_seams"]:
            all_errors.extend(res["errors"])
        all_warnings.extend(res["warnings"])
        if verbose:
            if not res["has_seams"]:
                print(f"   ✅ No significant seams detected")
            else:
                print(f"   ❌ Seams detected: {len(res['seam_rows'])} rows, {len(res['seam_cols'])} cols")
    
    # 6. Physical consistency
    if all(k in features_lower for k in ["ndvi", "ndbi", "lst"]):
        if verbose:
            print("\n🔗 Checking physical consistency...")
        res = validate_physical_consistency(
            features_lower["ndvi"],
            features_lower["ndbi"],
            features_lower["lst"],
            strict=False,
        )
        results["validators"]["physical_consistency"] = res
        all_errors.extend(res["errors"])
        all_warnings.extend(res["warnings"])
        if verbose:
            corr = res.get("correlations", {})
            if res["valid"]:
                print(f"   ✅ corr(NDVI,LST)={corr.get('ndvi_lst', 0):.3f}, corr(NDBI,LST)={corr.get('ndbi_lst', 0):.3f}")
            else:
                for err in res["errors"]:
                    print(f"   ❌ {err}")
    
    # Summary
    results["n_errors"] = len(all_errors)
    results["n_warnings"] = len(all_warnings)
    results["all_valid"] = len(all_errors) == 0
    
    if verbose:
        print("\n" + "=" * 60)
        if results["all_valid"]:
            print(f"✅ ALL VALIDATIONS PASSED ({results['n_warnings']} warnings)")
        else:
            print(f"❌ VALIDATION FAILED: {results['n_errors']} errors, {results['n_warnings']} warnings")
        print("=" * 60)
    
    # Raise if strict mode and errors exist
    if strict and not results["all_valid"]:
        raise ValidationError(f"Validation failed with {results['n_errors']} errors: {all_errors[0]}")
    
    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validate(ndvi: np.ndarray, ndbi: np.ndarray, lst: np.ndarray, season: str = "summer") -> bool:
    """
    Quick validation check - returns True if all pass, False otherwise.
    
    Does not raise exceptions.
    """
    try:
        results = run_all_validations(
            {"ndvi": ndvi, "ndbi": ndbi, "lst": lst},
            season=season,
            strict=False,
            verbose=False,
        )
        return results["all_valid"]
    except Exception:
        return False


if __name__ == "__main__":
    # Self-test with synthetic data
    print("Running self-test with synthetic data...\n")
    
    # Create realistic synthetic arrays
    np.random.seed(42)
    h, w = 256, 256
    
    # Synthetic NDVI (urban area with some vegetation)
    ndvi = np.random.uniform(0.0, 0.3, (h, w))
    ndvi[100:150, 100:150] = np.random.uniform(0.5, 0.7, (50, 50))  # Park
    
    # Synthetic NDBI (inverse of NDVI roughly)
    ndbi = 0.2 - ndvi * 0.3 + np.random.uniform(-0.05, 0.05, (h, w))
    
    # Synthetic LST (correlated with NDBI)
    lst = 300 + ndbi * 30 + np.random.uniform(-2, 2, (h, w))
    
    # Run validation
    results = run_all_validations(
        {"ndvi": ndvi, "ndbi": ndbi, "lst": lst},
        season="summer",
        strict=False,
        verbose=True,
    )
    
    print(f"\nSelf-test complete. Valid: {results['all_valid']}")
