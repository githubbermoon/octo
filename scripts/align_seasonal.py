#!/usr/bin/env python3
"""
Season-Level Feature Alignment for UHI v1 Pipeline.

SCIENTIFIC DESIGN:
    - One output per (year, season) — NOT per Landsat date
    - Same temporal window for NDVI, NDBI, and LST
    - LST aggregated via mean over all clear Landsat observations
    - n_observations layer for quality control
    - Excludes pixels with < min_observations

Output structure:
    {year}_{season}.npz containing:
        - ndvi: max composite over season
        - ndbi: median composite over season
        - lst: mean LST over season
        - n_obs_lst: count of valid LST observations per pixel
        - lat, lon: coordinate grids
        - metadata: aggregation info

Usage:
    python scripts/align_seasonal.py --config configs/configD1.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
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


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SeasonConfig:
    """Configuration for seasonal alignment."""
    sentinel2_dir: Path
    landsat8_dir: Path
    landsat9_dir: Path
    output_dir: Path
    worldcover_path: Path = None
    min_observations: int = 3  # Minimum LST observations per pixel
    tile_size: int = 256
    lst_aggregation: str = "mean"  # Only mean for now


# Season date ranges (inclusive)
SEASON_WINDOWS = {
    "summer": (3, 1, 5, 31),   # March 1 - May 31
    "winter": (11, 1, 1, 31),  # Nov 1 - Jan 31 (crosses year boundary)
}


def get_season_window(year: int, season: str) -> tuple[datetime, datetime]:
    """Get start/end dates for a season window."""
    if season == "summer":
        start = datetime(year, 3, 1)
        end = datetime(year, 5, 31)
    elif season == "winter":
        # Winter crosses year boundary: Nov Y to Jan Y+1
        start = datetime(year, 11, 1)
        end = datetime(year + 1, 1, 31)
    else:
        raise ValueError(f"Unknown season: {season}")
    return start, end


def date_in_season(date_str: str, year: int, season: str) -> bool:
    """Check if a date falls within the season window."""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return False
    
    start, end = get_season_window(year, season)
    return start <= date <= end


# =============================================================================
# TILE PARSING
# =============================================================================

def parse_tile_metadata(filename: str) -> dict[str, Any]:
    """Parse metadata from tile filename."""
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


def load_tile_array(tile_path: Path) -> tuple[np.ndarray, tuple, list]:
    """Load tile data, bounds, and band names."""
    data = np.load(tile_path, allow_pickle=True)
    image = data["image"]
    bounds = tuple(data["bounds"]) if "bounds" in data else None
    band_names = list(data.get("band_names", []))
    return image, bounds, band_names


def get_union_bounds(bounds_list: list[tuple]) -> tuple:
    """Compute union of multiple bounding boxes."""
    if not bounds_list:
        return None
    
    lefts = [b[0] for b in bounds_list]
    bottoms = [b[1] for b in bounds_list]
    rights = [b[2] for b in bounds_list]
    tops = [b[3] for b in bounds_list]
    
    return (min(lefts), min(bottoms), max(rights), max(tops))


# =============================================================================
# SENTINEL-2 PROCESSING
# =============================================================================

def extract_ndvi_ndbi(image: np.ndarray, band_names: list) -> tuple[np.ndarray, np.ndarray]:
    """Extract NDVI and NDBI from multi-band image."""
    ndvi = ndbi = None
    
    for i, name in enumerate(band_names):
        name_upper = str(name).upper() if name else ""
        if "NDVI" in name_upper:
            ndvi = image[i]
        elif "NDBI" in name_upper:
            ndbi = image[i]
    
    # Fallback: assume last two bands are NDVI, NDBI
    if ndvi is None and image.shape[0] >= 2:
        ndvi = image[-2]
    if ndbi is None and image.shape[0] >= 1:
        ndbi = image[-1]
    
    return ndvi, ndbi


def create_s2_seasonal_composite(
    tiles: list[Path],
    target_bounds: tuple,
    target_shape: tuple,
) -> dict[str, np.ndarray]:
    """
    Create seasonal composite from all S2 tiles in a season.
    
    NDVI: nanmax (peak vegetation)
    NDBI: nanmedian (robust)
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
            ndvi_resampled = resample_to_target_grid(ndvi, bounds, target_bounds, target_shape)
            ndbi_resampled = resample_to_target_grid(ndbi, bounds, target_bounds, target_shape)
            
            ndvi_stack.append(ndvi_resampled)
            ndbi_stack.append(ndbi_resampled)
            
        except Exception as e:
            continue
    
    if not ndvi_stack:
        return {}
    
    # Aggregate: max for NDVI, median for NDBI
    ndvi_composite = np.nanmax(np.stack(ndvi_stack), axis=0)
    ndbi_composite = np.nanmedian(np.stack(ndvi_stack), axis=0)
    
    return {
        "ndvi": ndvi_composite,
        "ndbi": ndbi_composite,
        "n_scenes": len(ndvi_stack),
    }


# =============================================================================
# LANDSAT LST PROCESSING
# =============================================================================

def create_lst_seasonal_composite(
    tiles: list[Path],
    target_bounds: tuple,
    target_shape: tuple,
    min_observations: int = 3,
) -> dict[str, np.ndarray]:
    """
    Create seasonal LST composite from all Landsat tiles.
    
    Returns:
        lst: mean LST (Kelvin)
        n_obs: count of valid observations per pixel
    """
    lst_stack = []
    
    for tile_path in tiles:
        try:
            image, bounds, band_names = load_tile_array(tile_path)
            if bounds is None:
                continue
            
            # Extract LST (first band or by name)
            lst_idx = 0
            for i, name in enumerate(band_names):
                name_upper = str(name).upper() if name else ""
                if "LST" in name_upper or "TEMP" in name_upper:
                    lst_idx = i
                    break
            
            lst = image[lst_idx]
            
            # Resample to target grid
            lst_resampled = resample_to_target_grid(lst, bounds, target_bounds, target_shape)
            lst_stack.append(lst_resampled)
            
        except Exception as e:
            continue
    
    if not lst_stack:
        return {}
    
    # Stack and compute mean + count
    lst_cube = np.stack(lst_stack, axis=0)  # (N, H, W)
    
    # Count valid observations per pixel
    n_obs = np.sum(np.isfinite(lst_cube), axis=0)
    
    # Mean LST
    lst_mean = np.nanmean(lst_cube, axis=0)
    
    # Mask pixels with insufficient observations
    lst_mean[n_obs < min_observations] = np.nan
    
    return {
        "lst": lst_mean,
        "n_obs": n_obs,
        "n_scenes": len(lst_stack),
    }


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_to_target_grid(
    source: np.ndarray,
    src_bounds: tuple,
    dst_bounds: tuple,
    dst_shape: tuple,
    method: str = "average",
) -> np.ndarray:
    """Resample source array to target grid."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required for resampling")
    
    src_height, src_width = source.shape
    src_transform = from_bounds(*src_bounds, src_width, src_height)
    
    dst_height, dst_width = dst_shape
    dst_transform = from_bounds(*dst_bounds, dst_width, dst_height)
    
    destination = np.empty(dst_shape, dtype=np.float32)
    
    resampling = Resampling.average if method == "average" else Resampling.bilinear
    
    reproject(
        source=source.astype(np.float32),
        destination=destination,
        src_transform=src_transform,
        src_crs="EPSG:32643",
        dst_transform=dst_transform,
        dst_crs="EPSG:32643",
        resampling=resampling,
    )
    
    return destination


# =============================================================================
# WORLDCOVER INTEGRATION
# =============================================================================

def load_worldcover_for_bounds(
    wc_path: Path,
    bounds: tuple,
    shape: tuple,
) -> np.ndarray:
    """
    Load and resample WorldCover to target grid.
    
    Returns class array (uint8):
        10=Trees, 20=Shrub, 30=Grass, 40=Crops,
        50=Built-up, 60=Barren, 80=Water
    """
    if not HAS_RASTERIO or not wc_path or not wc_path.exists():
        return None
    
    try:
        with rasterio.open(wc_path) as src:
            wc_resampled = np.empty(shape, dtype=np.uint8)
            dst_transform = from_bounds(*bounds, shape[1], shape[0])
            
            reproject(
                source=rasterio.band(src, 1),
                destination=wc_resampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:32643",
                resampling=Resampling.nearest,
            )
        return wc_resampled
    except Exception:
        return None


def compute_urban_rural_masks(worldcover: np.ndarray) -> dict[str, np.ndarray]:
    """
    Create urban and rural masks from WorldCover.
    
    Urban: class 50 (built-up)
    Rural: classes 10, 20, 30, 40 (vegetation)
    Water: class 80
    """
    if worldcover is None:
        return {}
    
    return {
        "is_urban": (worldcover == 50).astype(np.uint8),
        "is_vegetation": np.isin(worldcover, [10, 20, 30, 40]).astype(np.uint8),
        "is_water": (worldcover == 80).astype(np.uint8),
        "landcover": worldcover,
    }


# =============================================================================
# MAIN ALIGNMENT PIPELINE
# =============================================================================

def collect_tiles_by_year_season(
    tiles_dir: Path,
    sensor_prefix: str = "",
) -> dict[tuple[int, str], list[Path]]:
    """
    Group tiles by (year, season).
    
    Returns: {(2020, "summer"): [tile1, tile2, ...], ...}
    """
    groups = {}
    
    for tile_path in tiles_dir.glob("*.npz"):
        meta = parse_tile_metadata(tile_path.name)
        if not meta:
            continue
        if sensor_prefix and sensor_prefix not in meta.get("sensor", ""):
            continue
        
        key = (meta["year"], meta["season"])
        if key not in groups:
            groups[key] = []
        groups[key].append(tile_path)
    
    return groups


def align_seasonal(
    cfg: SeasonConfig,
    years: list[int] = None,
    seasons: list[str] = None,
) -> dict[str, Any]:
    """
    Main alignment function: one output per (year, season).
    
    Scientific guarantees:
        - Same temporal window for NDVI/NDBI and LST
        - LST aggregated via mean
        - n_observations layer included
        - Pixels with < min_observations masked
    """
    
    print("=" * 60)
    print("SEASONAL FEATURE ALIGNMENT")
    print("=" * 60)
    print(f"Output: {cfg.output_dir}")
    print(f"Min LST observations: {cfg.min_observations}")
    print(f"LST aggregation: {cfg.lst_aggregation}")
    
    # Collect tiles
    s2_groups = collect_tiles_by_year_season(cfg.sentinel2_dir)
    l8_groups = collect_tiles_by_year_season(cfg.landsat8_dir, "Landsat-8")
    l9_groups = collect_tiles_by_year_season(cfg.landsat9_dir, "Landsat-9")
    
    # Merge Landsat-8 and Landsat-9
    landsat_groups = {}
    for key in set(l8_groups.keys()) | set(l9_groups.keys()):
        landsat_groups[key] = l8_groups.get(key, []) + l9_groups.get(key, [])
    
    # Determine years and seasons to process
    all_keys = set(s2_groups.keys()) & set(landsat_groups.keys())
    
    if years:
        all_keys = {k for k in all_keys if k[0] in years}
    if seasons:
        all_keys = {k for k in all_keys if k[1] in seasons}
    
    print(f"\nFound {len(all_keys)} (year, season) combinations with both S2 and Landsat")
    
    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "outputs": [],
        "errors": [],
    }
    
    for year, season in tqdm(sorted(all_keys), desc="Processing seasons"):
        print(f"\n--- {year} {season} ---")
        
        s2_tiles = s2_groups.get((year, season), [])
        landsat_tiles = landsat_groups.get((year, season), [])
        
        print(f"  S2 tiles: {len(s2_tiles)}")
        print(f"  Landsat tiles: {len(landsat_tiles)}")
        
        if not s2_tiles or not landsat_tiles:
            report["errors"].append(f"{year}_{season}: Missing tiles")
            continue
        
        # Determine target grid from first Landsat tile
        _, first_bounds, _ = load_tile_array(landsat_tiles[0])
        if first_bounds is None:
            report["errors"].append(f"{year}_{season}: No bounds in Landsat tiles")
            continue
        
        # Use union of all Landsat bounds for this season
        all_landsat_bounds = []
        for t in landsat_tiles:
            _, b, _ = load_tile_array(t)
            if b:
                all_landsat_bounds.append(b)
        
        target_bounds = get_union_bounds(all_landsat_bounds) if all_landsat_bounds else first_bounds
        target_shape = (cfg.tile_size, cfg.tile_size)
        
        # Create S2 composite (NDVI max, NDBI median)
        s2_result = create_s2_seasonal_composite(s2_tiles, target_bounds, target_shape)
        if not s2_result:
            report["errors"].append(f"{year}_{season}: S2 composite failed")
            continue
        
        print(f"  S2 composite: {s2_result['n_scenes']} scenes used")
        
        # Create Landsat composite (LST mean + n_obs)
        lst_result = create_lst_seasonal_composite(
            landsat_tiles, target_bounds, target_shape, cfg.min_observations
        )
        if not lst_result:
            report["errors"].append(f"{year}_{season}: LST composite failed")
            continue
        
        print(f"  Landsat composite: {lst_result['n_scenes']} scenes, mean n_obs={np.nanmean(lst_result['n_obs']):.1f}")
        
        # Load WorldCover if available
        wc_masks = {}
        if cfg.worldcover_path and cfg.worldcover_path.exists():
            wc = load_worldcover_for_bounds(cfg.worldcover_path, target_bounds, target_shape)
            wc_masks = compute_urban_rural_masks(wc)
            print(f"  WorldCover: loaded")
        
        # Create coordinate grids
        left, bottom, right, top = target_bounds
        h, w = target_shape
        lons = np.linspace(left, right, w)
        lats = np.linspace(top, bottom, h)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Output filename
        out_name = f"{year}_{season}.npz"
        out_path = cfg.output_dir / out_name
        
        # Save
        output_data = {
            # Core features
            "ndvi": s2_result["ndvi"].astype(np.float32),
            "ndbi": s2_result["ndbi"].astype(np.float32),
            "lst": lst_result["lst"].astype(np.float32),
            "n_obs_lst": lst_result["n_obs"].astype(np.uint8),
            
            # Coordinates
            "lat": lat_grid.astype(np.float32),
            "lon": lon_grid.astype(np.float32),
            
            # Metadata
            "year": year,
            "season": season,
            "bounds": np.array(target_bounds),
            "n_s2_scenes": s2_result["n_scenes"],
            "n_landsat_scenes": lst_result["n_scenes"],
            "lst_aggregation": cfg.lst_aggregation,
            "min_observations": cfg.min_observations,
        }
        
        # Add WorldCover masks if available
        for k, v in wc_masks.items():
            output_data[k] = v
        
        np.savez_compressed(out_path, **output_data)
        
        report["outputs"].append({
            "path": str(out_path),
            "year": year,
            "season": season,
            "n_s2": s2_result["n_scenes"],
            "n_landsat": lst_result["n_scenes"],
        })
        
        print(f"  ✓ Saved: {out_name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Outputs created: {len(report['outputs'])}")
    print(f"Errors: {len(report['errors'])}")
    
    if report["errors"]:
        for err in report["errors"]:
            print(f"  ⚠️  {err}")
    
    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Seasonal Feature Alignment for UHI")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--years", type=int, nargs="+", help="Filter by years")
    parser.add_argument("--seasons", nargs="+", choices=["summer", "winter"], help="Filter by seasons")
    parser.add_argument("--worldcover", help="Path to WorldCover GeoTIFF")
    parser.add_argument("--min-obs", type=int, default=3, help="Min LST observations per pixel")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    
    yaml_cfg = {}
    try:
        import yaml
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
    except:
        pass
    
    base_dir = Path(yaml_cfg.get("project", {}).get("base_dir", "/content/drive/MyDrive/UHI_Project"))
    
    cfg = SeasonConfig(
        sentinel2_dir=base_dir / "processed" / "sentinel2" / "tiles",
        landsat8_dir=base_dir / "processed" / "landsat8" / "tiles",
        landsat9_dir=base_dir / "processed" / "landsat9" / "tiles",
        output_dir=base_dir / "processed" / "stacks" / "seasonal",
        worldcover_path=Path(args.worldcover) if args.worldcover else base_dir / "raw_data" / "worldcover" / "worldcover_2021.tif",
        min_observations=args.min_obs,
    )
    
    align_seasonal(cfg, years=args.years, seasons=args.seasons)


if __name__ == "__main__":
    main()
