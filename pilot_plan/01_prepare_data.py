#!/usr/bin/env python3
"""
Step 1: Prepare training data from aligned tiles.

Flattens aligned .npz stacks to a pandas DataFrame for XGBoost training.
Adds WorldCover land cover class and urban mask.

Usage:
    python pilot_plan/01_prepare_data.py --aligned-dir /path/to/aligned/ --output pilot_train.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
except ImportError:
    rasterio = None


def load_worldcover(wc_path: str, bounds: tuple, size: int = 256) -> np.ndarray:
    """Load and resample WorldCover to tile grid."""
    if rasterio is None:
        print("Warning: rasterio not available, skipping WorldCover")
        return np.zeros((size, size), dtype=np.uint8)
    
    with rasterio.open(wc_path) as src:
        wc_resampled = np.empty((size, size), dtype=np.uint8)
        dst_transform = from_bounds(*bounds, size, size)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=wc_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:32643",
            resampling=Resampling.nearest
        )
    return wc_resampled


def prepare_data(
    aligned_dir: Path,
    worldcover_path: str,
    output_path: Path,
) -> pd.DataFrame:
    """Convert aligned tiles to tabular format."""
    
    tiles = list(aligned_dir.glob("*.npz"))
    print(f"Found {len(tiles)} aligned tiles")
    
    all_rows = []
    
    for tile_path in tqdm(tiles, desc="Processing tiles"):
        data = np.load(tile_path, allow_pickle=True)
        
        # Extract arrays
        ndvi = data['ndvi'].flatten()
        ndbi = data['ndbi'].flatten()
        lst = data['lst'].flatten()
        lat = data['lat'].flatten() if 'lat' in data else np.zeros_like(ndvi)
        lon = data['lon'].flatten() if 'lon' in data else np.zeros_like(ndvi)
        bounds = data['bounds']
        
        # === SAFETY GUARDRAILS ===
        # These prevent misuse with seasonal union-bound tiles or normalized indices
        
        # Assert tile is exactly 256x256 (pilot expects fixed grid)
        raw_shape = data['ndvi'].shape
        assert raw_shape == (256, 256), (
            f"Tile {tile_path.name} has shape {raw_shape}, expected (256, 256). "
            "This script expects per-date aligned tiles, not seasonal union-bound outputs."
        )
        
        # Assert NDVI/NDBI are in physical range [-1, 1] (not normalized)
        ndvi_min, ndvi_max = np.nanmin(ndvi), np.nanmax(ndvi)
        assert -1.0 <= ndvi_min and ndvi_max <= 1.0, (
            f"NDVI range [{ndvi_min:.3f}, {ndvi_max:.3f}] outside [-1, 1]. "
            "NDVI may have been incorrectly normalized. Check preprocess.py."
        )
        
        ndbi_min, ndbi_max = np.nanmin(ndbi), np.nanmax(ndbi)
        assert -1.0 <= ndbi_min and ndbi_max <= 1.0, (
            f"NDBI range [{ndbi_min:.3f}, {ndbi_max:.3f}] outside [-1, 1]. "
            "NDBI may have been incorrectly normalized. Check preprocess.py."
        )
        
        # Load WorldCover
        try:
            wc = load_worldcover(worldcover_path, tuple(bounds), 256).flatten()
        except Exception as e:
            print(f"  Warning: Could not load WorldCover for {tile_path.name}: {e}")
            wc = np.zeros_like(ndvi, dtype=np.uint8)
        
        # Create DataFrame for this tile
        tile_df = pd.DataFrame({
            'ndvi': ndvi,
            'ndbi': ndbi,
            'lat': lat,
            'lon': lon,
            'lst': lst,
            'landcover': wc,
            'is_urban': (wc == 50).astype(int),
            'is_water': (wc == 80).astype(int),
            'is_vegetation': ((wc == 10) | (wc == 20) | (wc == 30) | (wc == 40)).astype(int),
            'tile': tile_path.stem,
        })
        
        # Remove NaN rows
        tile_df = tile_df.dropna()
        all_rows.append(tile_df)
        
    # Combine all tiles
    df = pd.concat(all_rows, ignore_index=True)
    
    # Convert LST to Celsius for easier interpretation
    df['lst_c'] = df['lst'] - 273.15
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} rows to {output_path}")
    
    # Summary stats
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total rows: {len(df):,}")
    print(f"Tiles: {df['tile'].nunique()}")
    print(f"\nLST range: {df['lst_c'].min():.1f} - {df['lst_c'].max():.1f} °C")
    print(f"NDVI range: {df['ndvi'].min():.3f} - {df['ndvi'].max():.3f}")
    print(f"NDBI range: {df['ndbi'].min():.3f} - {df['ndbi'].max():.3f}")
    
    print(f"\nLand cover distribution:")
    for lc, count in df['landcover'].value_counts().head(5).items():
        pct = 100 * count / len(df)
        names = {10:'Trees', 20:'Shrub', 30:'Grass', 40:'Crops', 
                 50:'Built-up', 60:'Barren', 80:'Water'}
        print(f"  {names.get(lc, f'Class {lc}')}: {count:,} ({pct:.1f}%)")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from aligned tiles")
    parser.add_argument("--aligned-dir", required=True, help="Directory with aligned .npz files")
    parser.add_argument("--worldcover", default="/content/drive/MyDrive/UHI_Project/raw_data/worldcover/worldcover_2021.tif")
    parser.add_argument("--output", default="pilot_train.csv", help="Output CSV path")
    args = parser.parse_args()
    
    aligned_dir = Path(args.aligned_dir)
    output_path = Path(args.output)
    
    if not aligned_dir.exists():
        print(f"Error: {aligned_dir} not found")
        return
    
    prepare_data(aligned_dir, args.worldcover, output_path)


if __name__ == "__main__":
    main()
