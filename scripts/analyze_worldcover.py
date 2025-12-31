#!/usr/bin/env python3
"""
Analyze ESA WorldCover to check urban/rural class distribution.

Shows what land cover classes are available in your AOI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import rasterio
except ImportError:
    raise SystemExit("rasterio required: pip install rasterio")


# ESA WorldCover 2021 class definitions
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland", 
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",        # URBAN
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Classification for UHI analysis
URBAN_CLASSES = {50}  # Built-up
VEGETATION_CLASSES = {10, 20, 30, 40}  # Trees, shrubs, grass, crops
WATER_CLASSES = {80}
OTHER_CLASSES = {60, 70, 90, 95, 100}


def analyze_worldcover(worldcover_path: Path) -> dict:
    """Analyze WorldCover class distribution."""
    
    with rasterio.open(worldcover_path) as src:
        data = src.read(1)
        crs = src.crs
        resolution = src.res
        bounds = src.bounds
        
    # Count pixels per class
    unique, counts = np.unique(data, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    total_pixels = data.size
    
    print("="*60)
    print("ESA WORLDCOVER ANALYSIS")
    print("="*60)
    print(f"File: {worldcover_path}")
    print(f"CRS: {crs}")
    print(f"Resolution: {resolution[0]:.1f}m x {resolution[1]:.1f}m")
    print(f"Bounds: {bounds}")
    print(f"Shape: {data.shape}")
    print(f"Total pixels: {total_pixels:,}")
    print()
    
    # Class breakdown
    print("CLASS DISTRIBUTION")
    print("-"*60)
    
    urban_pixels = 0
    vegetation_pixels = 0
    water_pixels = 0
    
    for class_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if class_id == 0:
            class_name = "NoData"
        else:
            class_name = WORLDCOVER_CLASSES.get(class_id, f"Unknown({class_id})")
        
        pct = (count / total_pixels) * 100
        
        # Categorize
        marker = ""
        if class_id in URBAN_CLASSES:
            marker = " 🔴 URBAN"
            urban_pixels += count
        elif class_id in VEGETATION_CLASSES:
            marker = " 🌿 VEGETATION"
            vegetation_pixels += count
        elif class_id in WATER_CLASSES:
            marker = " 💧 WATER"
            water_pixels += count
        
        print(f"  {class_id:3d}: {class_name:25s} → {count:8,} px ({pct:5.1f}%){marker}")
    
    print()
    print("SUMMARY FOR UHI ANALYSIS")
    print("-"*60)
    
    urban_pct = (urban_pixels / total_pixels) * 100
    veg_pct = (vegetation_pixels / total_pixels) * 100
    water_pct = (water_pixels / total_pixels) * 100
    
    print(f"  🔴 Urban (Built-up):    {urban_pixels:8,} px ({urban_pct:5.1f}%)")
    print(f"  🌿 Vegetation:          {vegetation_pixels:8,} px ({veg_pct:5.1f}%)")
    print(f"  💧 Water:               {water_pixels:8,} px ({water_pct:5.1f}%)")
    
    print()
    
    # UHI feasibility check
    if urban_pct > 5 and veg_pct > 5:
        print("✅ GOOD: Both urban and vegetation classes present!")
        print("   → Can compute UHI intensity (urban LST - vegetation LST)")
    elif urban_pct > 5 and veg_pct < 5:
        print("⚠️  WARNING: Very little vegetation (<5%)")
        print("   → Consider using NDVI-based cool zones instead")
    elif urban_pct < 5:
        print("⚠️  WARNING: Very little urban coverage (<5%)")
        print("   → May need to expand AOI or use NDVI-based hot zones")
    else:
        print("❌ ISSUE: Unclear class distribution")
    
    print("="*60)
    
    return {
        "urban_pct": urban_pct,
        "vegetation_pct": veg_pct,
        "water_pct": water_pct,
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze WorldCover class distribution")
    parser.add_argument("--worldcover", 
                        default="/content/drive/MyDrive/UHI_Project/raw_data/worldcover/worldcover_2021.tif",
                        help="Path to WorldCover GeoTIFF")
    args = parser.parse_args()
    
    wc_path = Path(args.worldcover)
    if not wc_path.exists():
        # Try local fallback
        local_fallback = Path("worldcover_2021.tif")
        if local_fallback.exists():
            wc_path = local_fallback
        else:
            print(f"❌ WorldCover not found: {wc_path}")
            print("   Run scripts/download_worldcover.py first")
            return
    
    analyze_worldcover(wc_path)


if __name__ == "__main__":
    main()
