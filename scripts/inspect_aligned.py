#!/usr/bin/env python3
"""
Inspect and verify aligned tile outputs.

Checks:
1. Metadata (year, season, sources)
2. Value ranges (NDVI: -1 to 1, NDBI: -1 to 1, LST: 270-350K)
3. Correlation between NDVI and LST (should be negative for UHI)
4. Visual plots (if matplotlib available)

Usage:
    python scripts/inspect_aligned.py --tile /path/to/aligned_tile.npz
    python scripts/inspect_aligned.py --dir /path/to/aligned/ --sample 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load_aligned_tile(tile_path: Path) -> dict:
    """Load aligned tile and return contents."""
    data = np.load(tile_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def check_value_ranges(tile_data: dict) -> dict:
    """Check if values are in expected ranges."""
    checks = {}
    
    # NDVI: should be -1 to 1
    if "ndvi" in tile_data:
        ndvi = tile_data["ndvi"]
        checks["ndvi"] = {
            "min": float(np.nanmin(ndvi)),
            "max": float(np.nanmax(ndvi)),
            "mean": float(np.nanmean(ndvi)),
            "valid": -1.5 <= np.nanmin(ndvi) and np.nanmax(ndvi) <= 1.5,
            "expected_range": "[-1, 1]",
        }
    
    # NDBI: should be -1 to 1
    if "ndbi" in tile_data:
        ndbi = tile_data["ndbi"]
        checks["ndbi"] = {
            "min": float(np.nanmin(ndbi)),
            "max": float(np.nanmax(ndbi)),
            "mean": float(np.nanmean(ndbi)),
            "valid": -1.5 <= np.nanmin(ndbi) and np.nanmax(ndbi) <= 1.5,
            "expected_range": "[-1, 1]",
        }
    
    # LST: should be ~270-350 K (reasonable Earth surface temps)
    if "lst" in tile_data:
        lst = tile_data["lst"]
        checks["lst"] = {
            "min": float(np.nanmin(lst)),
            "max": float(np.nanmax(lst)),
            "mean": float(np.nanmean(lst)),
            "valid": 250 <= np.nanmean(lst) <= 400,
            "expected_range": "[270, 350] K",
        }
    
    return checks


def compute_correlations(tile_data: dict) -> dict:
    """Compute correlations between features and LST."""
    correlations = {}
    
    if "lst" not in tile_data:
        return correlations
    
    lst = tile_data["lst"].flatten()
    
    for key in ["ndvi", "ndbi"]:
        if key in tile_data:
            arr = tile_data[key].flatten()
            
            # Remove NaN for correlation
            mask = np.isfinite(lst) & np.isfinite(arr)
            if mask.sum() > 100:
                corr = np.corrcoef(lst[mask], arr[mask])[0, 1]
                correlations[f"corr_lst_{key}"] = float(corr)
    
    return correlations


def inspect_tile(tile_path: Path, plot: bool = False) -> dict:
    """Full inspection of one aligned tile."""
    print(f"\n{'='*60}")
    print(f"INSPECTING: {tile_path.name}")
    print(f"{'='*60}")
    
    data = load_aligned_tile(tile_path)
    
    # Metadata
    print("\n📋 METADATA")
    print("-"*40)
    meta_keys = ["year", "season", "date", "landsat_source", "n_s2_matched", 
                 "n_s2_used", "alignment_method", "bounds"]
    for key in meta_keys:
        if key in data:
            val = data[key]
            if isinstance(val, np.ndarray):
                val = val.item() if val.ndim == 0 else list(val)
            print(f"  {key}: {val}")
    
    # Shapes
    print("\n📐 ARRAY SHAPES")
    print("-"*40)
    for key in ["ndvi", "ndbi", "lat", "lon", "lst"]:
        if key in data:
            print(f"  {key}: {data[key].shape}")
    
    # Value ranges
    print("\n📊 VALUE RANGES")
    print("-"*40)
    checks = check_value_ranges(data)
    for key, info in checks.items():
        status = "✅" if info["valid"] else "❌"
        print(f"  {key}: min={info['min']:.3f}, max={info['max']:.3f}, "
              f"mean={info['mean']:.3f} {status} (expected: {info['expected_range']})")
    
    # Correlations
    print("\n📈 CORRELATIONS (UHI sanity check)")
    print("-"*40)
    corrs = compute_correlations(data)
    for key, val in corrs.items():
        if "ndvi" in key:
            # For UHI: NDVI-LST should be NEGATIVE (more vegetation = cooler)
            expected = "negative" if "ndvi" in key else "positive"
            status = "✅" if (val < 0 and "ndvi" in key) or (val > 0 and "ndbi" in key) else "⚠️"
            print(f"  {key}: {val:.3f} {status} (expected: {expected} for UHI)")
        elif "ndbi" in key:
            # NDBI-LST should be POSITIVE (more built-up = hotter)
            status = "✅" if val > 0 else "⚠️"
            print(f"  {key}: {val:.3f} {status} (expected: positive for UHI)")
    
    # NaN check
    print("\n🔍 DATA QUALITY")
    print("-"*40)
    for key in ["ndvi", "ndbi", "lst"]:
        if key in data:
            arr = data[key]
            nan_pct = 100 * np.isnan(arr).sum() / arr.size
            print(f"  {key}: {nan_pct:.1f}% NaN")
    
    # Optional plotting
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # NDVI
            im = axes[0].imshow(data["ndvi"], cmap="RdYlGn", vmin=-0.5, vmax=0.8)
            axes[0].set_title("NDVI (S2 composite)")
            plt.colorbar(im, ax=axes[0])
            
            # NDBI
            im = axes[1].imshow(data["ndbi"], cmap="RdYlBu_r", vmin=-0.5, vmax=0.5)
            axes[1].set_title("NDBI (S2 composite)")
            plt.colorbar(im, ax=axes[1])
            
            # LST
            im = axes[2].imshow(data["lst"], cmap="hot")
            axes[2].set_title("LST (Landsat)")
            plt.colorbar(im, ax=axes[2], label="Kelvin")
            
            # NDVI vs LST scatter
            ndvi_flat = data["ndvi"].flatten()
            lst_flat = data["lst"].flatten()
            mask = np.isfinite(ndvi_flat) & np.isfinite(lst_flat)
            axes[3].scatter(ndvi_flat[mask][::10], lst_flat[mask][::10], alpha=0.3, s=1)
            axes[3].set_xlabel("NDVI")
            axes[3].set_ylabel("LST (K)")
            axes[3].set_title("NDVI vs LST (expect negative trend)")
            
            plt.tight_layout()
            
            out_path = tile_path.with_suffix(".png")
            plt.savefig(out_path, dpi=150)
            print(f"\n📸 Plot saved: {out_path}")
            plt.close()
            
        except ImportError:
            print("\n⚠️  matplotlib not available for plotting")
    
    return {
        "checks": checks,
        "correlations": corrs,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect aligned tiles")
    parser.add_argument("--tile", help="Path to single aligned tile .npz")
    parser.add_argument("--dir", help="Directory of aligned tiles (random sample)")
    parser.add_argument("--sample", type=int, default=3, help="Number of tiles to sample")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()
    
    if args.tile:
        tile_path = Path(args.tile)
        if tile_path.exists():
            inspect_tile(tile_path, plot=args.plot)
        else:
            print(f"❌ File not found: {tile_path}")
    
    elif args.dir:
        tiles_dir = Path(args.dir)
        tiles = list(tiles_dir.glob("*.npz"))
        
        if not tiles:
            print(f"❌ No .npz files found in: {tiles_dir}")
            return
        
        print(f"Found {len(tiles)} aligned tiles")
        
        # Random sample
        sample_size = min(args.sample, len(tiles))
        np.random.seed(42)
        sample_tiles = np.random.choice(tiles, sample_size, replace=False)
        
        all_results = []
        for tile_path in sample_tiles:
            result = inspect_tile(tile_path, plot=args.plot)
            all_results.append(result)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY ACROSS SAMPLED TILES")
        print("="*60)
        
        # Average correlations
        ndvi_corrs = [r["correlations"].get("corr_lst_ndvi", np.nan) for r in all_results]
        ndbi_corrs = [r["correlations"].get("corr_lst_ndbi", np.nan) for r in all_results]
        
        print(f"  Avg corr(LST, NDVI): {np.nanmean(ndvi_corrs):.3f} (should be negative)")
        print(f"  Avg corr(LST, NDBI): {np.nanmean(ndbi_corrs):.3f} (should be positive)")
        
        # Check if ready for XGBoost
        if np.nanmean(ndvi_corrs) < 0 and np.nanmean(ndbi_corrs) > 0:
            print("\n✅ ALIGNMENT LOOKS CORRECT — Ready for XGBoost!")
        else:
            print("\n⚠️  UNUSUAL CORRELATIONS — Review before modeling")
    
    else:
        # Default: check aligned dir from config
        default_dir = Path("/content/drive/MyDrive/UHI_Project/processed/stacks/aligned")
        if default_dir.exists():
            tiles = list(default_dir.glob("*.npz"))
            if tiles:
                print(f"Found {len(tiles)} tiles in default location")
                inspect_tile(tiles[0], plot=args.plot)
        else:
            print("Usage: python inspect_aligned.py --tile /path/to/tile.npz")
            print("   or: python inspect_aligned.py --dir /path/to/aligned/ --sample 5")


if __name__ == "__main__":
    main()
