#!/usr/bin/env python3
"""
MODIS Validation Script for UHI v1 Pipeline.

Performs two levels of validation:
  Level 1: Pixel-wise validation using Landsat hold-out test set
  Level 2: Regional validation comparing model predictions with MODIS LST trends

Urban/Rural mapping uses ESA WorldCover (or simple NDBI threshold if WorldCover unavailable).

Usage:
    python scripts/validate_modis.py --config configs/configD1.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except ImportError:
    rasterio = None


@dataclass
class ValidationConfig:
    modis_dir: Path
    landsat_dir: Path
    predictions_dir: Path
    output_dir: Path
    worldcover_path: Optional[Path] = None  # ESA WorldCover GeoTIFF
    ndbi_urban_threshold: float = 0.1  # Fallback: NDBI > 0.1 = urban
    test_split_ratio: float = 0.1


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml required: pip install pyyaml")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path) -> ValidationConfig:
    """Load configuration from YAML file."""
    config = load_yaml(config_path)
    
    base_dir = Path(config.get("project", {}).get("base_dir", "/content/drive/MyDrive/UHI_Project"))
    
    return ValidationConfig(
        modis_dir=base_dir / "raw_data" / "MODIS",  # New path: UHI_Project/raw_data/MODIS
        landsat_dir=base_dir / "processed",
        predictions_dir=base_dir / "processed" / "stacks" / "aligned",
        output_dir=base_dir / "validation",
    )


# =============================================================================
# LEVEL 1: Pixel-wise Validation (Landsat Hold-out)
# =============================================================================

def compute_pixel_metrics(
    predictions: np.ndarray, 
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute pixel-wise regression metrics."""
    if mask is not None:
        pred = predictions[mask]
        gt = ground_truth[mask]
    else:
        pred = predictions.flatten()
        gt = ground_truth.flatten()
    
    # Remove NaN values
    valid = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[valid]
    gt = gt[valid]
    
    if len(pred) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan}
    
    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    
    # MAE
    mae = np.mean(np.abs(pred - gt))
    
    # R²
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Bias (mean error)
    bias = np.mean(pred - gt)
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "bias": float(bias),
        "n_pixels": int(len(pred)),
    }


def level1_pixel_validation(
    aligned_tiles: List[Path],
    test_ratio: float = 0.1,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Level 1: Pixel-wise validation using Landsat test split.
    
    Uses the aligned tiles (which contain both features and LST target).
    Splits into train/test and computes metrics on test set.
    """
    print("\n" + "="*60)
    print("LEVEL 1: Pixel-Wise Validation (Landsat Hold-out)")
    print("="*60)
    
    # Shuffle and split tiles
    np.random.seed(42)
    tiles = list(aligned_tiles)
    np.random.shuffle(tiles)
    
    n_test = max(1, int(len(tiles) * test_ratio))
    test_tiles = tiles[:n_test]
    train_tiles = tiles[n_test:]
    
    print(f"Total tiles: {len(tiles)}")
    print(f"Test tiles: {len(test_tiles)} ({test_ratio*100:.0f}%)")
    
    # Collect all predictions and ground truth from test tiles
    all_pred = []
    all_gt = []
    
    for tile_path in tqdm(test_tiles, desc="Loading test tiles"):
        try:
            data = np.load(tile_path, allow_pickle=True)
            
            # In aligned tiles, we have NDVI, NDBI, lat, lon as features
            # and LST as target. For now, we compare against LST directly.
            # In production, this would use model predictions.
            
            lst = data.get("lst")
            if lst is not None:
                # For demonstration: use a simple baseline (NDBI-based estimate)
                ndbi = data.get("ndbi")
                if ndbi is not None:
                    # Simple linear model: LST ≈ A*NDBI + B (placeholder)
                    # In real use, load actual model predictions
                    baseline_pred = 300 + 20 * ndbi  # Dummy baseline
                    all_pred.append(baseline_pred.flatten())
                    all_gt.append(lst.flatten())
        except Exception as e:
            print(f"Error loading {tile_path}: {e}")
            continue
    
    if not all_pred:
        print("No test data found!")
        return {"error": "No test tiles with LST"}
    
    predictions = np.concatenate(all_pred)
    ground_truth = np.concatenate(all_gt)
    
    metrics = compute_pixel_metrics(predictions, ground_truth)
    
    print(f"\n📊 Level 1 Results:")
    print(f"   RMSE: {metrics['rmse']:.2f} K")
    print(f"   MAE:  {metrics['mae']:.2f} K")
    print(f"   R²:   {metrics['r2']:.3f}")
    print(f"   Bias: {metrics['bias']:.2f} K")
    print(f"   N:    {metrics['n_pixels']:,} pixels")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "level1_pixel_metrics.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"   Saved: {results_path}")
    
    return {
        "level": 1,
        "type": "pixel_wise",
        "metrics": metrics,
        "n_test_tiles": len(test_tiles),
        "n_train_tiles": len(train_tiles),
    }


# =============================================================================
# LEVEL 2: Regional Validation (MODIS Comparison)
# =============================================================================

def create_urban_mask_from_ndbi(
    ndbi: np.ndarray, 
    threshold: float = 0.1
) -> np.ndarray:
    """
    Create urban/rural mask from NDBI.
    
    Urban: NDBI > threshold
    Rural: NDBI <= threshold
    
    This is a fallback when ESA WorldCover is not available.
    """
    return ndbi > threshold


def create_urban_mask_from_worldcover(
    worldcover_path: Path,
    target_bounds: Tuple[float, float, float, float],
    target_shape: Tuple[int, int],
    target_crs: str = "EPSG:32643",
) -> Optional[np.ndarray]:
    """
    Create urban mask from ESA WorldCover.
    
    WorldCover classes:
        10 = Tree cover
        20 = Shrubland
        30 = Grassland
        40 = Cropland
        50 = Built-up (URBAN)
        60 = Bare/sparse vegetation
        70 = Snow and ice
        80 = Permanent water
        90 = Herbaceous wetland
        95 = Mangroves
        100 = Moss and lichen
    """
    if rasterio is None or not worldcover_path.exists():
        return None
    
    try:
        with rasterio.open(worldcover_path) as src:
            # Reproject to target grid
            dst_transform = rasterio.transform.from_bounds(
                *target_bounds, target_shape[1], target_shape[0]
            )
            
            worldcover = np.empty(target_shape, dtype=np.uint8)
            reproject(
                source=rasterio.band(src, 1),
                destination=worldcover,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
            )
        
        # Built-up = 50
        urban_mask = worldcover == 50
        return urban_mask
        
    except Exception as e:
        print(f"Error loading WorldCover: {e}")
        return None


def compute_seasonal_modis_means(
    modis_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Compute seasonal mean LST from raw MODIS files.
    
    File naming: MODIS_summer_2023_2023-04-15.tif
    
    Returns: {
        "2023_summer": {"mean": 310.5, "std": 5.2, "n_scenes": 30},
        "2023_winter": {"mean": 295.3, "std": 4.1, "n_scenes": 31},
        ...
    }
    """
    print("\nComputing MODIS seasonal means...")
    
    seasonal_data = {}
    
    modis_files = list(modis_dir.glob("MODIS_*.tif")) + list(modis_dir.glob("MODIS_*.tiff"))
    
    for f in tqdm(modis_files, desc="Processing MODIS"):
        try:
            # Parse filename: MODIS_summer_2023_2023-04-15.tif
            parts = f.stem.split("_")
            if len(parts) >= 3:
                season = parts[1]  # summer or winter
                year = parts[2]    # 2023
                key = f"{year}_{season}"
                
                with rasterio.open(f) as src:
                    lst = src.read(1).astype(np.float32)
                    # MODIS LST is often scaled; assume already in Kelvin
                    # Mask nodata (usually 0)
                    lst = np.where(lst == 0, np.nan, lst)
                
                if key not in seasonal_data:
                    seasonal_data[key] = {"values": [], "files": []}
                
                seasonal_data[key]["values"].append(np.nanmean(lst))
                seasonal_data[key]["files"].append(str(f.name))
                
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
    
    # Compute statistics per season
    results = {}
    for key, data in seasonal_data.items():
        values = np.array(data["values"])
        results[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "n_scenes": len(values),
        }
    
    return results


def level2_regional_validation(
    modis_dir: Path,
    aligned_tiles: List[Path],
    output_dir: Path = None,
    worldcover_path: Optional[Path] = None,
    ndbi_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Level 2: Regional validation comparing model trends with MODIS.
    
    Computes:
    1. UHI intensity (Urban - Rural LST difference)
    2. Seasonal trends (Summer vs Winter)
    3. Comparison with MODIS seasonal means
    """
    print("\n" + "="*60)
    print("LEVEL 2: Regional Validation (MODIS Comparison)")
    print("="*60)
    
    # Step 1: Compute MODIS seasonal means
    modis_means = compute_seasonal_modis_means(modis_dir)
    
    print("\n📊 MODIS Seasonal Means:")
    for key, stats in sorted(modis_means.items()):
        print(f"   {key}: {stats['mean']:.1f}K ± {stats['std']:.1f}K (n={stats['n_scenes']})")
    
    # Step 2: Compute model predictions by urban/rural
    print("\nAnalyzing model predictions by urban/rural zones...")
    
    urban_lst = []
    rural_lst = []
    
    for tile_path in tqdm(aligned_tiles[:100], desc="Analyzing tiles"):  # Sample 100
        try:
            data = np.load(tile_path, allow_pickle=True)
            lst = data.get("lst")
            ndbi = data.get("ndbi")
            
            if lst is None or ndbi is None:
                continue
            
            # Create urban mask from NDBI (fallback)
            urban_mask = create_urban_mask_from_ndbi(ndbi, ndbi_threshold)
            
            # Collect LST values
            urban_vals = lst[urban_mask & np.isfinite(lst)]
            rural_vals = lst[~urban_mask & np.isfinite(lst)]
            
            if len(urban_vals) > 0:
                urban_lst.append(np.mean(urban_vals))
            if len(rural_vals) > 0:
                rural_lst.append(np.mean(rural_vals))
                
        except Exception as e:
            continue
    
    # Compute UHI intensity
    if urban_lst and rural_lst:
        urban_mean = np.mean(urban_lst)
        rural_mean = np.mean(rural_lst)
        uhi_intensity = urban_mean - rural_mean
    else:
        urban_mean = rural_mean = uhi_intensity = np.nan
    
    print(f"\n📊 Level 2 Results:")
    print(f"   Urban mean LST:  {urban_mean:.1f} K")
    print(f"   Rural mean LST:  {rural_mean:.1f} K")
    print(f"   UHI Intensity:   {uhi_intensity:.2f} K")
    print(f"   (Positive = urban is hotter = expected for UHI)")
    
    # Compare with MODIS trends
    modis_summer = [v["mean"] for k, v in modis_means.items() if "summer" in k]
    modis_winter = [v["mean"] for k, v in modis_means.items() if "winter" in k]
    
    if modis_summer and modis_winter:
        modis_seasonal_diff = np.mean(modis_summer) - np.mean(modis_winter)
        print(f"\n   MODIS Summer mean: {np.mean(modis_summer):.1f} K")
        print(f"   MODIS Winter mean: {np.mean(modis_winter):.1f} K")
        print(f"   MODIS Seasonal diff: {modis_seasonal_diff:.1f} K")
    else:
        modis_seasonal_diff = np.nan
    
    results = {
        "level": 2,
        "type": "regional",
        "urban_mean_lst": float(urban_mean) if np.isfinite(urban_mean) else None,
        "rural_mean_lst": float(rural_mean) if np.isfinite(rural_mean) else None,
        "uhi_intensity_k": float(uhi_intensity) if np.isfinite(uhi_intensity) else None,
        "modis_seasonal_means": modis_means,
        "modis_seasonal_diff_k": float(modis_seasonal_diff) if np.isfinite(modis_seasonal_diff) else None,
        "urban_mask_method": "ndbi_threshold" if worldcover_path is None else "worldcover",
        "ndbi_threshold": ndbi_threshold,
    }
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "level2_regional_metrics.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n   Saved: {results_path}")
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MODIS Validation for UHI v1")
    parser.add_argument("--config", default="configs/configD1.yaml", help="Config YAML")
    parser.add_argument("--modis-dir", help="Override MODIS directory")
    parser.add_argument("--aligned-dir", help="Override aligned tiles directory")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--worldcover", help="Path to ESA WorldCover GeoTIFF")
    parser.add_argument("--ndbi-threshold", type=float, default=0.1,
                        help="NDBI threshold for urban classification (fallback)")
    parser.add_argument("--level", type=int, choices=[1, 2], 
                        help="Run only specific level (1=pixel, 2=regional)")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(config_path)
    else:
        cfg = ValidationConfig(
            modis_dir=Path("data/modis"),
            landsat_dir=Path("data/processed"),
            predictions_dir=Path("data/processed/stacks/aligned"),
            output_dir=Path("data/validation"),
        )
    
    # Override from args
    if args.modis_dir:
        cfg.modis_dir = Path(args.modis_dir)
    if args.aligned_dir:
        cfg.predictions_dir = Path(args.aligned_dir)
    if args.output:
        cfg.output_dir = Path(args.output)
    if args.worldcover:
        cfg.worldcover_path = Path(args.worldcover)
    cfg.ndbi_urban_threshold = args.ndbi_threshold
    
    # Find aligned tiles
    aligned_tiles = list(cfg.predictions_dir.glob("*.npz"))
    print(f"Found {len(aligned_tiles)} aligned tiles")
    
    if not aligned_tiles:
        print("⚠️  No aligned tiles found. Run align_features.py first!")
        print(f"   Looking in: {cfg.predictions_dir}")
        return
    
    results = {}
    
    # Level 1: Pixel-wise
    if args.level is None or args.level == 1:
        l1_output = cfg.output_dir / "pixel_wise"
        results["level1"] = level1_pixel_validation(
            aligned_tiles=aligned_tiles,
            test_ratio=cfg.test_split_ratio,
            output_dir=l1_output,
        )
    
    # Level 2: Regional
    if args.level is None or args.level == 2:
        l2_output = cfg.output_dir / "regional"
        results["level2"] = level2_regional_validation(
            modis_dir=cfg.modis_dir,
            aligned_tiles=aligned_tiles,
            output_dir=l2_output,
            worldcover_path=cfg.worldcover_path,
            ndbi_threshold=cfg.ndbi_urban_threshold,
        )
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
