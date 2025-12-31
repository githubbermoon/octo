#!/usr/bin/env python3
"""
Download and clip ESA WorldCover to the AOI bounds from params.yaml.

Reprojects to project CRS (EPSG:32643) and resamples to 30m resolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping

try:
    import planetary_computer
    import pystac_client
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("pystac-client and planetary-computer are required") from exc


# Project defaults
DST_CRS = "EPSG:32643"  # UTM 43N (Bangalore)
DST_RESOLUTION = 30  # meters (match Landsat)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import guard
        raise SystemExit("pyyaml is required to read config files") from exc
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def resolve_base_dir(config_path: Optional[Path]) -> Optional[Path]:
    if config_path and config_path.exists():
        config_data = load_yaml(config_path)
        base_dir = config_data.get("project", {}).get("base_dir")
        if base_dir:
            return Path(base_dir)
    drive_base = Path("/content/drive/MyDrive/UHI_Project")
    return drive_base if drive_base.exists() else None


def resolve_aoi_bounds(params_path: Optional[Path]) -> list[float]:
    if not params_path or not params_path.exists():
        raise SystemExit("params.yaml is required to resolve AOI bounds")
    params_data = load_yaml(params_path)
    bounds = params_data.get("download", {}).get("aoi", {}).get("bounds")
    if not bounds or len(bounds) != 4:
        raise SystemExit("AOI bounds not found in params.yaml")
    return [float(b) for b in bounds]


def resolve_crs_and_resolution(config_path: Optional[Path]) -> tuple[str, int]:
    """Read CRS and resolution from config, or use defaults."""
    crs = DST_CRS
    resolution = DST_RESOLUTION
    if config_path and config_path.exists():
        config_data = load_yaml(config_path)
        spatial = config_data.get("spatial", {})
        crs = spatial.get("crs", crs)
        resolution = int(spatial.get("base_resolution", resolution))
    return crs, resolution


def pick_worldcover_item(bounds: list[float]):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["esa-worldcover"], bbox=bounds)
    items = list(search.get_items())
    if not items:
        raise SystemExit("No ESA WorldCover items found for the AOI")
    items.sort(key=lambda item: item.datetime or item.properties.get("datetime"), reverse=True)
    return items[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESA WorldCover for AOI.")
    parser.add_argument("--config", default="configs/configD1.yaml", help="Config YAML path")
    parser.add_argument("--params", default="params.yaml", help="Params YAML path")
    parser.add_argument("--output", help="Output GeoTIFF path")
    parser.add_argument("--asset", default="map", help="WorldCover asset key (default: map)")
    parser.add_argument("--resolution", type=int, help="Output resolution in meters (default: 30)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    params_path = Path(args.params) if args.params else None

    base_dir = resolve_base_dir(config_path)
    bounds = resolve_aoi_bounds(params_path)
    dst_crs, dst_resolution = resolve_crs_and_resolution(config_path)
    
    # Override resolution from args if provided
    if args.resolution:
        dst_resolution = args.resolution
    
    if args.output:
        output_path = Path(args.output)
        if base_dir and not output_path.is_absolute():
            output_path = base_dir / output_path
    else:
        if base_dir:
            output_path = base_dir / "raw_data/worldcover/worldcover_2021.tif"
        else:
            output_path = Path("worldcover_2021.tif")

    print(f"AOI bounds: {bounds}")
    print(f"Target CRS: {dst_crs}")
    print(f"Target resolution: {dst_resolution}m")

    item = pick_worldcover_item(bounds)
    if args.asset not in item.assets:
        raise SystemExit(f"Asset '{args.asset}' not found in WorldCover item")
    href = item.assets[args.asset].href
    print(f"Downloading from: {href}")

    # Step 1: Clip to AOI
    geom = mapping(box(*bounds))
    with rasterio.open(href) as src:
        clipped_image, clipped_transform = mask(src, [geom], crop=True)
        src_crs = src.crs
        src_dtype = src.dtypes[0]
        
        print(f"Clipped shape: {clipped_image.shape}")
        print(f"Source CRS: {src_crs}")

        # Step 2: Calculate transform for reprojection + resampling
        # Using explicit resolution for 30m output
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs,
            clipped_image.shape[2],
            clipped_image.shape[1],
            *rasterio.transform.array_bounds(
                clipped_image.shape[1], 
                clipped_image.shape[2], 
                clipped_transform
            ),
            resolution=dst_resolution,  # <-- This forces 30m output
        )
        
        print(f"Output shape: ({dst_height}, {dst_width})")
        
        # Step 3: Reproject to target CRS at target resolution
        dst_data = np.empty((1, dst_height, dst_width), dtype=src_dtype)
        
        reproject(
            source=clipped_image,
            destination=dst_data,
            src_transform=clipped_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,  # Nearest for categorical data
        )
        
        # Step 4: Prepare output metadata
        out_meta = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": src_dtype,
            "crs": dst_crs,
            "transform": dst_transform,
            "compress": "lzw",
        }

    # Step 5: Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(dst_data)

    print(f"\n✅ Saved WorldCover to {output_path}")
    print(f"   CRS: {dst_crs}")
    print(f"   Resolution: {dst_resolution}m")
    print(f"   Shape: {dst_data.shape}")


if __name__ == "__main__":
    main()

