#!/usr/bin/env python3
"""
Feature Alignment Script for UHI v1 Pipeline.

Spatially aligns Sentinel-2 feature tiles with Landsat LST target tiles
to create training pairs: [NDVI, NDBI, lat, lon] → LST

Usage:
    python scripts/align_features.py --config configs/configD1.yaml
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
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


@dataclass
class AlignConfig:
    sentinel2_dir: Path
    landsat_dir: Path
    output_dir: Path
    max_distance_m: float = 1000.0  # Max spatial distance for matching (meters)
    crs: str = "EPSG:32643"


def load_yaml(path: Path) -> Dict[str, Any]:
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
    paths = config.get("paths", {}).get("processed", {})
    
    sentinel2_dir = base_dir / "processed" / "sentinel2" / "tiles"
    landsat_dir = base_dir / "processed"  # Will combine landsat8 + landsat9
    output_dir = base_dir / "processed" / "stacks"
    
    crs = config.get("spatial", {}).get("crs", "EPSG:32643")
    
    return AlignConfig(
        sentinel2_dir=sentinel2_dir,
        landsat_dir=landsat_dir,
        output_dir=output_dir,
        crs=crs,
    )


def load_tile_metadata(tiles_dir: Path) -> List[Dict[str, Any]]:
    """Load metadata.json from a tiles directory."""
    metadata_path = tiles_dir.parent / "metadata.json"
    if not metadata_path.exists():
        return []
    with open(metadata_path) as f:
        return json.load(f)


def parse_bounds_from_npz(npz_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Extract bounds from a tile .npz file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        bounds = data.get("bounds")
        if bounds is not None:
            # bounds is a BoundingBox(left, bottom, right, top)
            return tuple(bounds)
        return None
    except Exception:
        return None


def get_tile_center(bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get center coordinates from bounds (left, bottom, right, top)."""
    left, bottom, right, top = bounds
    return ((left + right) / 2, (bottom + top) / 2)


def find_tiles(tiles_dir: Path) -> List[Path]:
    """Find all .npz tile files in directory."""
    if not tiles_dir.exists():
        return []
    return list(tiles_dir.glob("*.npz"))


def extract_features_from_tile(tile_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Extract NDVI, NDBI and coordinates from Sentinel-2 tile."""
    try:
        data = np.load(tile_path, allow_pickle=True)
        image = data["image"]  # Shape: (bands, H, W)
        band_names = list(data.get("band_names", []))
        bounds = data.get("bounds")
        
        if bounds is None:
            return None
        
        # Find NDVI and NDBI bands (computed during EE export)
        # Band order from EE script: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDBI
        ndvi_idx = None
        ndbi_idx = None
        
        for i, name in enumerate(band_names):
            if name and "NDVI" in str(name).upper():
                ndvi_idx = i
            if name and "NDBI" in str(name).upper():
                ndbi_idx = i
        
        # If not found by name, assume last two bands
        if ndvi_idx is None:
            ndvi_idx = -2 if image.shape[0] >= 2 else 0
        if ndbi_idx is None:
            ndbi_idx = -1 if image.shape[0] >= 1 else 0
        
        ndvi = image[ndvi_idx]
        ndbi = image[ndbi_idx]
        
        # Create lat/lon grids
        left, bottom, right, top = bounds
        h, w = ndvi.shape
        lons = np.linspace(left, right, w)
        lats = np.linspace(top, bottom, h)  # top to bottom
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return {
            "ndvi": ndvi,
            "ndbi": ndbi,
            "lat": lat_grid,
            "lon": lon_grid,
            "bounds": bounds,
        }
    except Exception as e:
        print(f"Error loading {tile_path}: {e}")
        return None


def extract_lst_from_tile(tile_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Extract LST from Landsat tile."""
    try:
        data = np.load(tile_path, allow_pickle=True)
        image = data["image"]  # Shape: (bands, H, W)
        band_names = list(data.get("band_names", []))
        bounds = data.get("bounds")
        
        if bounds is None:
            return None
        
        # For Landsat, LST is typically a single band or in thermal band
        # The preprocessing normalizes values, so we use the first band
        # In actual implementation, you'd identify the LST band specifically
        lst = image[0] if image.shape[0] >= 1 else None
        
        if lst is None:
            return None
        
        return {
            "lst": lst,
            "bounds": bounds,
        }
    except Exception as e:
        print(f"Error loading {tile_path}: {e}")
        return None


def align_tiles(
    sentinel2_tiles: List[Path],
    landsat_tiles: List[Path],
    output_dir: Path,
    max_distance: float = 1000.0,
) -> Dict[str, Any]:
    """
    Align Sentinel-2 feature tiles with Landsat LST tiles.
    
    For each Landsat tile, find the nearest Sentinel-2 tile(s) and
    create a training sample.
    """
    report = {
        "aligned_pairs": 0,
        "unmatched_landsat": 0,
        "unmatched_sentinel2": 0,
        "pairs": [],
    }
    
    # Build spatial index of Sentinel-2 tile centers
    s2_centers = []
    s2_bounds_map = {}
    
    print("Indexing Sentinel-2 tiles...")
    for s2_path in tqdm(sentinel2_tiles, desc="Indexing S2"):
        bounds = parse_bounds_from_npz(s2_path)
        if bounds:
            center = get_tile_center(bounds)
            s2_centers.append(center)
            s2_bounds_map[len(s2_centers) - 1] = (s2_path, bounds)
    
    if not s2_centers:
        print("No Sentinel-2 tiles with valid bounds found!")
        return report
    
    # Build KD-tree for fast spatial lookup
    if cKDTree is not None:
        s2_tree = cKDTree(s2_centers)
    else:
        # Fallback: brute force matching
        s2_tree = None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = output_dir / "aligned"
    aligned_dir.mkdir(exist_ok=True)
    
    print("Aligning Landsat tiles with Sentinel-2...")
    for ls_path in tqdm(landsat_tiles, desc="Aligning"):
        ls_bounds = parse_bounds_from_npz(ls_path)
        if not ls_bounds:
            report["unmatched_landsat"] += 1
            continue
        
        ls_center = get_tile_center(ls_bounds)
        
        # Find nearest Sentinel-2 tile
        if s2_tree is not None:
            dist, idx = s2_tree.query(ls_center, k=1)
        else:
            # Brute force
            dists = [
                np.sqrt((c[0] - ls_center[0])**2 + (c[1] - ls_center[1])**2)
                for c in s2_centers
            ]
            idx = np.argmin(dists)
            dist = dists[idx]
        
        if dist > max_distance:
            report["unmatched_landsat"] += 1
            continue
        
        s2_path, s2_bounds = s2_bounds_map[idx]
        
        # Extract features and target
        features = extract_features_from_tile(s2_path)
        target = extract_lst_from_tile(ls_path)
        
        if features is None or target is None:
            report["unmatched_landsat"] += 1
            continue
        
        # Save aligned pair
        pair_id = f"{ls_path.stem}__aligned"
        out_path = aligned_dir / f"{pair_id}.npz"
        
        np.savez_compressed(
            out_path,
            ndvi=features["ndvi"].astype(np.float32),
            ndbi=features["ndbi"].astype(np.float32),
            lat=features["lat"].astype(np.float32),
            lon=features["lon"].astype(np.float32),
            lst=target["lst"].astype(np.float32),
            landsat_source=str(ls_path),
            sentinel2_source=str(s2_path),
            landsat_bounds=ls_bounds,
            sentinel2_bounds=s2_bounds,
        )
        
        report["aligned_pairs"] += 1
        report["pairs"].append({
            "output": str(out_path),
            "landsat": str(ls_path),
            "sentinel2": str(s2_path),
            "distance_m": float(dist),
        })
    
    # Save alignment report
    report_path = output_dir / "alignment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Align Sentinel-2 features with Landsat LST")
    parser.add_argument("--config", default="configs/configD1.yaml", help="Config YAML")
    parser.add_argument("--sentinel2-dir", help="Override Sentinel-2 tiles directory")
    parser.add_argument("--landsat-dir", help="Override Landsat tiles directory")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--max-distance", type=float, default=1000.0, 
                        help="Max matching distance in meters")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(config_path)
    else:
        cfg = AlignConfig(
            sentinel2_dir=Path("data/processed/sentinel2/tiles"),
            landsat_dir=Path("data/processed"),
            output_dir=Path("data/processed/stacks"),
        )
    
    # Override from args
    if args.sentinel2_dir:
        cfg.sentinel2_dir = Path(args.sentinel2_dir)
    if args.landsat_dir:
        cfg.landsat_dir = Path(args.landsat_dir)
    if args.output:
        cfg.output_dir = Path(args.output)
    cfg.max_distance_m = args.max_distance
    
    # Find tiles
    print(f"Looking for Sentinel-2 tiles in: {cfg.sentinel2_dir}")
    sentinel2_tiles = find_tiles(cfg.sentinel2_dir)
    print(f"Found {len(sentinel2_tiles)} Sentinel-2 tiles")
    
    # Combine Landsat-8 and Landsat-9 tiles
    landsat_tiles = []
    for sensor in ["landsat8", "landsat9"]:
        tiles_dir = cfg.landsat_dir / sensor / "tiles"
        tiles = find_tiles(tiles_dir)
        landsat_tiles.extend(tiles)
        print(f"Found {len(tiles)} {sensor} tiles")
    
    print(f"Total Landsat tiles: {len(landsat_tiles)}")
    
    if not sentinel2_tiles or not landsat_tiles:
        print("Error: No tiles found. Check paths.")
        return
    
    # Align tiles
    report = align_tiles(
        sentinel2_tiles=sentinel2_tiles,
        landsat_tiles=landsat_tiles,
        output_dir=cfg.output_dir,
        max_distance=cfg.max_distance_m,
    )
    
    print(f"\n✅ Alignment complete!")
    print(f"   Aligned pairs: {report['aligned_pairs']}")
    print(f"   Unmatched Landsat: {report['unmatched_landsat']}")
    print(f"   Output: {cfg.output_dir}/aligned/")


if __name__ == "__main__":
    main()
