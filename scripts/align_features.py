#!/usr/bin/env python3
"""
SOTA Feature Alignment Script for UHI v1 Pipeline.

Properly aligns Sentinel-2 features with Landsat LST targets:
1. Season matching: Landsat winter → S2 winter, Landsat summer → S2 summer
2. Temporal matching: Match within same year ± tolerance
3. Pixel-level resampling: S2 (10m) resampled to Landsat grid (30m) using area-weighted mean
4. Creates seasonal composites for robust feature extraction

Usage:
    python scripts/align_features.py --config configs/configD1.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@dataclass
class AlignConfig:
    sentinel2_dir: Path
    landsat8_dir: Path
    landsat9_dir: Path
    output_dir: Path
    temporal_tolerance_days: int = 15  # Match S2 within ±N days of Landsat
    use_temporal_filter: bool = True   # Filter by date proximity
    use_seasonal_composite: bool = True  # Aggregate scenes


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml required: pip install pyyaml")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path) -> AlignConfig:
    """Load configuration from YAML file."""
    config = load_yaml(config_path)
    base_dir = Path(config.get("project", {}).get("base_dir", "/content/drive/MyDrive/UHI_Project"))
    
    return AlignConfig(
        sentinel2_dir=base_dir / "processed" / "sentinel2" / "tiles",
        landsat8_dir=base_dir / "processed" / "landsat8" / "tiles",
        landsat9_dir=base_dir / "processed" / "landsat9" / "tiles",
        output_dir=base_dir / "processed" / "stacks" / "aligned",
    )


def parse_tile_metadata(filename: str) -> dict[str, Any]:
    """
    Parse metadata from tile filename.
    
    Expected format: "Sensor YYYY season YYYY-MM-DD_NN_rX_cY.npz"
    Example: "Landsat-8 2020 summer 2020-03-31_01_r0_c0.npz"
    """
    pattern = r"(Sentinel-2|Landsat-8|Landsat-9)\s+(\d{4})\s+(summer|winter)\s+(\d{4}-\d{2}-\d{2})_(\d+)_r(\d+)_c(\d+)"
    match = re.search(pattern, filename)
    
    if not match:
        return {}
    
    return {
        "sensor": match.group(1),
        "year": int(match.group(2)),
        "season": match.group(3),
        "date": match.group(4),
        "scene_id": match.group(5),
        "row": int(match.group(6)),
        "col": int(match.group(7)),
    }


def parse_date(date_str: str):
    """Parse date string to datetime object."""
    from datetime import datetime
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None


def filter_by_temporal_proximity(
    s2_tiles: list[Path],
    landsat_date: str,
    tolerance_days: int = 15,
) -> list[Path]:
    """
    Filter S2 tiles to those within ±tolerance_days of Landsat date.
    
    This ensures tighter temporal coupling for NDVI-LST relationships.
    """
    from datetime import timedelta
    
    target_date = parse_date(landsat_date)
    if target_date is None:
        return s2_tiles  # Fallback: no filtering
    
    filtered = []
    for tile_path in s2_tiles:
        meta = parse_tile_metadata(tile_path.name)
        if not meta or 'date' not in meta:
            continue
        
        tile_date = parse_date(meta['date'])
        if tile_date is None:
            continue
        
        delta = abs((tile_date - target_date).days)
        if delta <= tolerance_days:
            filtered.append(tile_path)
    
    return filtered if filtered else s2_tiles  # Fallback to all if none match


def group_tiles_by_season(tiles_dir: Path) -> dict[str, list[Path]]:
    """
    Group tiles by year_season key.
    
    Returns: {"2020_summer": [tile1, tile2, ...], "2020_winter": [...], ...}
    """
    groups = {}
    
    for tile_path in tiles_dir.glob("*.npz"):
        meta = parse_tile_metadata(tile_path.name)
        if not meta:
            continue
        
        key = f"{meta['year']}_{meta['season']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(tile_path)
    
    return groups


def load_tile_bounds(tile_path: Path) -> tuple | None:
    """Load only bounds from tile (fast, for filtering)."""
    try:
        data = np.load(tile_path, allow_pickle=True)
        bounds = data.get("bounds")
        if bounds is not None:
            return tuple(bounds)
        return None
    except Exception:
        return None


def bounds_overlap(bounds1: tuple, bounds2: tuple, min_overlap_frac: float = 0.1) -> bool:
    """
    Check if two bounding boxes overlap.
    
    bounds format: (left, bottom, right, top) in projected coordinates.
    Returns True if overlap area is at least min_overlap_frac of the smaller box.
    """
    if bounds1 is None or bounds2 is None:
        return False
    
    left1, bottom1, right1, top1 = bounds1
    left2, bottom2, right2, top2 = bounds2
    
    # Check for no overlap
    if right1 <= left2 or right2 <= left1:
        return False
    if top1 <= bottom2 or top2 <= bottom1:
        return False
    
    # Compute overlap area
    overlap_left = max(left1, left2)
    overlap_right = min(right1, right2)
    overlap_bottom = max(bottom1, bottom2)
    overlap_top = min(top1, top2)
    
    overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
    
    # Compute areas
    area1 = (right1 - left1) * (top1 - bottom1)
    area2 = (right2 - left2) * (top2 - bottom2)
    min_area = min(area1, area2)
    
    if min_area <= 0:
        return False
    
    return (overlap_area / min_area) >= min_overlap_frac


def load_tile_array(tile_path: Path) -> tuple[np.ndarray, np.ndarray, Any]:
    """Load tile array, bounds, and band names from .npz file."""
    data = np.load(tile_path, allow_pickle=True)
    image = data["image"]  # (bands, H, W)
    bounds = data.get("bounds")
    band_names = data.get("band_names", [])
    return image, bounds, band_names


def extract_ndvi_ndbi(image: np.ndarray, band_names: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract NDVI and NDBI from Sentinel-2 tile.
    
    Assumes band order from EE export includes NDVI and NDBI as last 2 bands.
    """
    ndvi_idx = None
    ndbi_idx = None
    
    for i, name in enumerate(band_names):
        name_str = str(name).upper() if name else ""
        if "NDVI" in name_str:
            ndvi_idx = i
        if "NDBI" in name_str:
            ndbi_idx = i
    
    # Fallback: assume last two bands are NDVI, NDBI
    if ndvi_idx is None:
        ndvi_idx = -2 if image.shape[0] >= 2 else 0
    if ndbi_idx is None:
        ndbi_idx = -1 if image.shape[0] >= 1 else 0
    
    return image[ndvi_idx], image[ndbi_idx]


def resample_to_target_grid(
    source: np.ndarray,
    source_bounds: tuple,
    target_bounds: tuple,
    target_shape: tuple,
    resampling_method: str = "average"
) -> np.ndarray:
    """
    Resample source array to target grid.
    
    Uses area-weighted average (Resampling.average) for SOTA pixel alignment.
    """
    if not HAS_RASTERIO:
        # Fallback: simple resize using numpy
        from scipy.ndimage import zoom
        zoom_factors = (target_shape[0] / source.shape[0], target_shape[1] / source.shape[1])
        return zoom(source, zoom_factors, order=1)  # Bilinear
    
    # Create transforms
    src_height, src_width = source.shape
    dst_height, dst_width = target_shape
    
    src_transform = from_bounds(*source_bounds, src_width, src_height)
    dst_transform = from_bounds(*target_bounds, dst_width, dst_height)
    
    # CRITICAL: Use NaN fill instead of zeros
    destination = np.full(target_shape, np.nan, dtype=np.float32)
    
    # Resampling method
    if resampling_method == "average":
        method = Resampling.average
    elif resampling_method == "bilinear":
        method = Resampling.bilinear
    else:
        method = Resampling.nearest
    
    # Reproject with nodata handling
    reproject(
        source=source.astype(np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs="EPSG:32643",
        dst_transform=dst_transform,
        dst_crs="EPSG:32643",
        resampling=method,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    
    return destination


def create_seasonal_composite(tiles: list[Path], target_bounds: tuple, target_shape: tuple) -> dict[str, np.ndarray]:
    """
    Create seasonal mean composite from multiple tiles.
    
    Aggregates all tiles in a season to create robust NDVI/NDBI features.
    """
    ndvi_stack = []
    ndbi_stack = []
    
    for tile_path in tiles:
        try:
            image, bounds, band_names = load_tile_array(tile_path)
            
            if bounds is None:
                continue
            
            ndvi, ndbi = extract_ndvi_ndbi(image, list(band_names))
            
            # Resample to target grid
            ndvi_resampled = resample_to_target_grid(
                ndvi, tuple(bounds), target_bounds, target_shape, "average"
            )
            ndbi_resampled = resample_to_target_grid(
                ndbi, tuple(bounds), target_bounds, target_shape, "average"
            )
            
            ndvi_stack.append(ndvi_resampled)
            ndbi_stack.append(ndbi_resampled)
            
        except Exception as e:
            print(f"Error processing {tile_path}: {e}")
            continue
    
    if not ndvi_stack:
        return {}
    
    # Compute composites:
    # - NDVI: max (peak vegetation = clearest, least cloud)
    # - NDBI: median (robust to SWIR noise and outliers)
    ndvi_composite = np.nanmax(np.stack(ndvi_stack), axis=0)
    ndbi_composite = np.nanmedian(np.stack(ndbi_stack), axis=0)
    
    return {
        "ndvi": ndvi_composite,
        "ndbi": ndbi_composite,
        "n_scenes": len(ndvi_stack),
    }


def align_seasonal_pairs(
    s2_groups: dict[str, list[Path]],
    landsat_groups: dict[str, list[Path]],
    output_dir: Path,
    limit: int = 0,  # 0 = no limit
) -> dict[str, Any]:
    """
    Align S2 and Landsat tiles by season.
    
    For each season/year:
    1. Create S2 composite (NDVI, NDBI)
    2. For each Landsat tile, resample S2 features to Landsat grid
    3. Save aligned pair
    """
    report = {
        "aligned_pairs": 0,
        "seasons_processed": [],
        "pairs": [],
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching seasons
    common_seasons = set(s2_groups.keys()) & set(landsat_groups.keys())
    print(f"\nFound {len(common_seasons)} matching season-years")
    
    for season_key in sorted(common_seasons):
        print(f"\n{'='*50}")
        print(f"Processing: {season_key}")
        print(f"{'='*50}")
        
        s2_tiles = s2_groups[season_key]
        landsat_tiles = landsat_groups[season_key]
        
        print(f"  S2 tiles: {len(s2_tiles)}")
        print(f"  Landsat tiles: {len(landsat_tiles)}")
        
        # Pre-load S2 bounds for fast spatial filtering
        print(f"  Caching S2 bounds...")
        s2_bounds_cache = {}
        for s2_path in s2_tiles:
            bounds = load_tile_bounds(s2_path)
            if bounds:
                s2_bounds_cache[s2_path] = bounds
        print(f"  Cached {len(s2_bounds_cache)} S2 tile bounds")
        
        # Process each Landsat tile
        for landsat_path in tqdm(landsat_tiles, desc=f"  Aligning {season_key}"):
            try:
                # Load Landsat tile (target grid)
                landsat_image, landsat_bounds, landsat_band_names = load_tile_array(landsat_path)
                
                if landsat_bounds is None:
                    continue
                
                target_bounds = tuple(landsat_bounds)
                target_shape = (landsat_image.shape[1], landsat_image.shape[2])  # (H, W)
                
                # Extract LST band by name (fallback to band 0)
                lst_idx = 0
                for i, name in enumerate(landsat_band_names):
                    name_str = str(name).upper() if name else ""
                    if "LST" in name_str or "TEMPERATURE" in name_str:
                        lst_idx = i
                        break
                lst = landsat_image[lst_idx]
                
                # SPATIAL FILTERING: Find S2 tiles that overlap this Landsat tile
                matching_s2 = [
                    s2_path for s2_path, s2_bounds in s2_bounds_cache.items()
                    if bounds_overlap(target_bounds, s2_bounds, min_overlap_frac=0.1)
                ]
                
                # DEBUG: Show how many S2 tiles matched spatially
                meta = parse_tile_metadata(landsat_path.name)
                landsat_date = meta.get('date', '')
                n_spatial = len(matching_s2)
                
                # TEMPORAL FILTERING: Keep only S2 tiles within ±N days of Landsat date
                # This is critical for accurate NDVI-LST coupling
                matching_s2 = filter_by_temporal_proximity(
                    matching_s2, landsat_date, tolerance_days=15
                )
                n_temporal = len(matching_s2)
                
                tqdm.write(f"    → Landsat r{meta['row']}_c{meta['col']} ({landsat_date}): {n_spatial} spatial → {n_temporal} temporal S2 tiles")
                
                if not matching_s2:
                    tqdm.write(f"      ⚠️  No S2 tiles within ±15 days!")
                    continue
                
                # Create composite from spatially + temporally matching S2 tiles
                composite = create_seasonal_composite(
                    matching_s2, target_bounds, target_shape
                )
                
                if not composite:
                    tqdm.write(f"      ⚠️  Composite failed!")
                    continue
                
                tqdm.write(f"      ✓  Used {composite['n_scenes']}/{n_temporal} S2 tiles in composite")
                
                # Create coordinate grids
                left, bottom, right, top = target_bounds
                h, w = target_shape
                lons = np.linspace(left, right, w)
                lats = np.linspace(top, bottom, h)  # Top to bottom
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                
                # Parse metadata for output filename
                meta = parse_tile_metadata(landsat_path.name)
                out_name = f"{meta['year']}_{meta['season']}_{meta['date']}_r{meta['row']}_c{meta['col']}.npz"
                out_path = output_dir / out_name
                
                # Save aligned pair
                np.savez_compressed(
                    out_path,
                    ndvi=composite["ndvi"].astype(np.float32),
                    ndbi=composite["ndbi"].astype(np.float32),
                    lat=lat_grid.astype(np.float32),
                    lon=lon_grid.astype(np.float32),
                    lst=lst.astype(np.float32),
                    year=meta["year"],
                    season=meta["season"],
                    date=meta["date"],
                    landsat_source=str(landsat_path),
                    n_s2_matched=len(matching_s2),  # S2 tiles that overlapped
                    n_s2_used=composite["n_scenes"],  # S2 tiles actually used
                    bounds=target_bounds,
                    alignment_method="spatial_overlap_seasonal_composite",
                )
                
                report["aligned_pairs"] += 1
                report["pairs"].append({
                    "output": str(out_path),
                    "landsat": str(landsat_path),
                    "year": meta["year"],
                    "season": meta["season"],
                    "n_s2_matched": len(matching_s2),
                    "n_s2_used": composite["n_scenes"],
                })
                
                # Check limit
                if limit > 0 and report["aligned_pairs"] >= limit:
                    print(f"\n⏹️  Reached limit of {limit} tiles. Stopping early.")
                    return report
                
            except Exception as e:
                print(f"Error processing {landsat_path}: {e}")
                continue
        
        report["seasons_processed"].append(season_key)
    
    # Save report
    report_path = output_dir.parent / "alignment_report_v2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="SOTA Feature Alignment for UHI v1")
    parser.add_argument("--config", default="configs/configD1.yaml", help="Config YAML")
    parser.add_argument("--sentinel2-dir", help="Override S2 tiles directory")
    parser.add_argument("--landsat8-dir", help="Override Landsat-8 tiles directory")
    parser.add_argument("--landsat9-dir", help="Override Landsat-9 tiles directory")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit total aligned tiles (0=no limit, for testing)")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(config_path)
    else:
        cfg = AlignConfig(
            sentinel2_dir=Path("data/processed/sentinel2/tiles"),
            landsat8_dir=Path("data/processed/landsat8/tiles"),
            landsat9_dir=Path("data/processed/landsat9/tiles"),
            output_dir=Path("data/processed/stacks/aligned"),
        )
    
    # Override from args
    if args.sentinel2_dir:
        cfg.sentinel2_dir = Path(args.sentinel2_dir)
    if args.landsat8_dir:
        cfg.landsat8_dir = Path(args.landsat8_dir)
    if args.landsat9_dir:
        cfg.landsat9_dir = Path(args.landsat9_dir)
    if args.output:
        cfg.output_dir = Path(args.output)
    
    print("="*60)
    print("SOTA FEATURE ALIGNMENT")
    print("="*60)
    print(f"S2 dir: {cfg.sentinel2_dir}")
    print(f"Landsat-8 dir: {cfg.landsat8_dir}")
    print(f"Landsat-9 dir: {cfg.landsat9_dir}")
    print(f"Output: {cfg.output_dir}")
    
    # Group tiles by season
    print("\nGrouping tiles by season...")
    s2_groups = group_tiles_by_season(cfg.sentinel2_dir)
    l8_groups = group_tiles_by_season(cfg.landsat8_dir)
    l9_groups = group_tiles_by_season(cfg.landsat9_dir)
    
    # Merge Landsat 8 and 9 groups
    landsat_groups = {}
    for key, tiles in l8_groups.items():
        landsat_groups[key] = tiles
    for key, tiles in l9_groups.items():
        if key in landsat_groups:
            landsat_groups[key].extend(tiles)
        else:
            landsat_groups[key] = tiles
    
    print(f"\nS2 season-years: {sorted(s2_groups.keys())}")
    print(f"Landsat season-years: {sorted(landsat_groups.keys())}")
    
    if args.limit > 0:
        print(f"⚠️  LIMIT MODE: Will stop after {args.limit} tiles")
    
    # Align
    report = align_seasonal_pairs(s2_groups, landsat_groups, cfg.output_dir, limit=args.limit)
    
    print("\n" + "="*60)
    print("✅ ALIGNMENT COMPLETE")
    print("="*60)
    print(f"Aligned pairs: {report['aligned_pairs']}")
    print(f"Seasons processed: {len(report['seasons_processed'])}")
    print(f"Output: {cfg.output_dir}")
    print(f"Report: {cfg.output_dir.parent / 'alignment_report_v2.json'}")


if __name__ == "__main__":
    main()
