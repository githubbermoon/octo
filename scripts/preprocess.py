#!/usr/bin/env python3
"""
Tile-based preprocessing for local geospatial rasters.

This script scans an input folder for rasters (GeoTIFF by default), tiles them,
optionally normalizes each tile, and writes compressed .npz tiles with metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit("rasterio is required for preprocessing") from exc


RASTER_EXTS = {".tif", ".tiff"}


@dataclass
class PreprocessConfig:
    input_dir: Path
    output_dir: Path
    tile_size: int = 256
    overlap: int = 32
    min_valid_frac: float = 0.7
    normalize: bool = True
    skip_normalize_lst: bool = True  # Don't normalize LST bands (Landsat)
    clip_percentiles: Tuple[float, float] = (2.0, 98.0)
    max_tiles_per_scene: Optional[int] = None
    dry_run: bool = False
    debug: bool = False


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import guard
        raise SystemExit("pyyaml is required to read config files") from exc
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def resolve_paths(
    config_path: Optional[Path],
    params_path: Optional[Path],
    input_arg: Optional[str],
    output_arg: Optional[str],
) -> PreprocessConfig:
    config_data: Dict[str, Any] = {}
    params_data: Dict[str, Any] = {}

    if config_path and config_path.exists():
        config_data = load_yaml(config_path)
    if params_path and params_path.exists():
        params_data = load_yaml(params_path)

    base_dir = None
    if config_data.get("project", {}).get("base_dir"):
        base_dir = Path(config_data["project"]["base_dir"])

    paths_cfg = config_data.get("paths", {})
    default_input = Path(paths_cfg.get("raw_data", "data/raw"))
    default_output = Path(paths_cfg.get("processed", {}).get("stacks", "data/processed"))

    if input_arg:
        input_dir = Path(input_arg)
        if base_dir and not input_dir.is_absolute():
            input_dir = base_dir / input_dir
    else:
        input_dir = default_input
        if base_dir and not input_dir.is_absolute():
            input_dir = base_dir / input_dir

    if output_arg:
        output_dir = Path(output_arg)
        if base_dir and not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    else:
        output_dir = default_output
        if base_dir and not output_dir.is_absolute():
            output_dir = base_dir / output_dir

    preprocess_cfg = params_data.get("preprocess", {})
    tile_size = int(preprocess_cfg.get("tile_size", 256))
    overlap = int(preprocess_cfg.get("overlap", 32))
    normalize = bool(preprocess_cfg.get("normalize", True))
    min_valid_frac = float(preprocess_cfg.get("min_valid_frac", 0.7))
    clip_percentiles = tuple(preprocess_cfg.get("clip_percentiles", [2, 98]))

    if "preprocess" in config_data:
        preprocess_cfg = config_data["preprocess"]
        tile_size = int(preprocess_cfg.get("tile_size", tile_size))
        overlap = int(preprocess_cfg.get("overlap", overlap))
        normalize = bool(preprocess_cfg.get("normalize", normalize))
        min_valid_frac = float(preprocess_cfg.get("min_valid_frac", min_valid_frac))
        clip_percentiles = tuple(preprocess_cfg.get("clip_percentiles", clip_percentiles))

    return PreprocessConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap=overlap,
        normalize=normalize,
        min_valid_frac=min_valid_frac,
        clip_percentiles=(float(clip_percentiles[0]), float(clip_percentiles[1])),
    )


def find_rasters(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    rasters: List[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in RASTER_EXTS:
            rasters.append(path)
    return rasters


def normalize_tile(tile: np.ndarray, clip_percentiles: Tuple[float, float]) -> np.ndarray:
    # Per-band percentile clipping followed by min-max scaling to [0, 1].
    out = np.empty_like(tile, dtype=np.float32)
    for b in range(tile.shape[0]):
        band = tile[b].astype(np.float32)
        lo, hi = np.nanpercentile(band, clip_percentiles)
        band = np.clip(band, lo, hi)
        min_v = np.nanmin(band)
        max_v = np.nanmax(band)
        denom = max_v - min_v
        if denom < 1e-8:
            out[b] = 0.0
        else:
            out[b] = (band - min_v) / denom
    return out


def tile_indices(height: int, width: int, tile_size: int, overlap: int) -> Iterable[Tuple[int, int]]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    max_row = max(0, height - tile_size)
    max_col = max(0, width - tile_size)
    row = 0
    while row <= max_row:
        col = 0
        while col <= max_col:
            yield row, col
            col += stride
        row += stride


def process_scene(
    path: Path,
    cfg: PreprocessConfig,
    tiles_dir: Path,
    metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scene_summary = {
        "source": str(path),
        "tiles_written": 0,
        "tiles_skipped": 0,
    }

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            image = np.where(image == nodata, np.nan, image)
        height, width = image.shape[1], image.shape[2]

        band_names = list(src.descriptions) if src.descriptions else []
        if not band_names or all(b is None for b in band_names):
            band_names = [f"B{idx:02d}" for idx in range(1, image.shape[0] + 1)]

        tile_count = 0
        debug_samples = 0
        for row, col in tile_indices(height, width, cfg.tile_size, cfg.overlap):
            tile = image[:, row:row + cfg.tile_size, col:col + cfg.tile_size]
            if tile.shape[1] != cfg.tile_size or tile.shape[2] != cfg.tile_size:
                continue

            valid_mask = np.all(np.isfinite(tile), axis=0)
            valid_frac = float(valid_mask.mean())
            if valid_frac < cfg.min_valid_frac:
                scene_summary["tiles_skipped"] += 1
                continue

            # Normalization rules:
            # - Landsat: skip ALL (LST should stay in Kelvin)
            # - Sentinel-2: normalize reflectance bands, but SKIP NDVI/NDBI (indices)
            is_landsat = "landsat" in str(path).lower()
            is_sentinel2 = "sentinel" in str(path).lower()
            
            # Index bands that should NOT be normalized (already bounded physical indices)
            INDEX_BANDS = {"NDVI", "NDBI", "ndvi", "ndbi"}
            
            if is_landsat and cfg.skip_normalize_lst:
                # Skip all normalization for Landsat
                if cfg.debug and tile_count == 0:
                    print(f"  [debug] Skipping normalization for Landsat: {path.name}")
            elif is_sentinel2 and cfg.normalize:
                # For Sentinel-2: normalize per-band, but skip NDVI/NDBI
                tile_normalized = tile.copy()
                for band_idx, band_name in enumerate(band_names):
                    if band_name in INDEX_BANDS:
                        # Keep NDVI/NDBI as-is (physical indices)
                        if cfg.debug and tile_count == 0:
                            print(f"  [debug] Skipping normalization for {band_name} (index band)")
                    else:
                        # Normalize reflectance bands
                        band_data = tile[band_idx]
                        p_low, p_high = cfg.clip_percentiles
                        valid = band_data[np.isfinite(band_data)]
                        if len(valid) > 0:
                            low = np.percentile(valid, p_low)
                            high = np.percentile(valid, p_high)
                            if high > low:
                                clipped = np.clip(band_data, low, high)
                                tile_normalized[band_idx] = (clipped - low) / (high - low)
                tile = tile_normalized
            elif cfg.normalize:
                # Default: normalize all bands
                tile = normalize_tile(tile, cfg.clip_percentiles)

            tile_id = f"{path.stem}_r{row}_c{col}"
            out_path = tiles_dir / f"{tile_id}.npz"
            tile_window = Window(col, row, cfg.tile_size, cfg.tile_size)
            tile_transform = window_transform(tile_window, src.transform)
            
            # Compute tile-specific bounds from tile_transform
            # tile_transform gives origin at (col, row), tile covers tile_size pixels
            a = tile_transform.a  # pixel width
            e = tile_transform.e  # pixel height (negative)
            c = tile_transform.c  # tile left
            f = tile_transform.f  # tile top
            tile_left = c
            tile_top = f
            tile_right = c + cfg.tile_size * a
            tile_bottom = f + cfg.tile_size * e
            if tile_bottom > tile_top:
                tile_bottom, tile_top = tile_top, tile_bottom
            tile_bounds = (tile_left, tile_bottom, tile_right, tile_top)
            
            if not cfg.dry_run:
                np.savez_compressed(
                    out_path,
                    image=tile.astype(np.float32),
                    transform=str(src.transform),
                    tile_transform=str(tile_transform),
                    crs=str(src.crs),
                    bounds=np.array(tile_bounds),  # Tile-specific bounds!
                    scene_bounds=src.bounds,  # Keep scene bounds for reference
                    band_names=band_names,
                    source=str(path),
                    row=row,
                    col=col,
                )

            metadata.append(
                {
                    "tile_id": tile_id,
                    "source": str(path),
                    "row": row,
                    "col": col,
                    "tile_size": cfg.tile_size,
                    "valid_frac": valid_frac,
                    "band_names": band_names,
                }
            )

            tile_count += 1
            scene_summary["tiles_written"] += 1
            if cfg.debug and debug_samples < 3:
                debug_samples += 1
                print(
                    f"[debug] {tile_id} row={row} col={col} "
                    f"tile_transform={tile_transform}"
                )
            if cfg.max_tiles_per_scene and tile_count >= cfg.max_tiles_per_scene:
                break

    return scene_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess local raster data into tiles.")
    parser.add_argument("--config", default="configs/configD1.yaml", help="Config YAML path")
    parser.add_argument("--params", default="params.yaml", help="Params YAML path")
    parser.add_argument("--input", help="Input directory (overrides config)")
    parser.add_argument("--output", help="Output directory (overrides config)")
    parser.add_argument("--tile-size", type=int, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, help="Tile overlap in pixels")
    parser.add_argument("--min-valid-frac", type=float, default=0.7, help="Min valid pixel fraction")
    parser.add_argument("--no-normalize", action="store_true", help="Disable per-tile normalization")
    parser.add_argument("--clip-percentiles", nargs=2, type=float, default=[2.0, 98.0])
    parser.add_argument("--max-tiles-per-scene", type=int, help="Limit tiles per scene")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report only")
    parser.add_argument("--debug", action="store_true", help="Print debug samples per scene")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    params_path = Path(args.params) if args.params else None
    cfg = resolve_paths(config_path, params_path, args.input, args.output)

    if args.tile_size:
        cfg.tile_size = args.tile_size
    if args.overlap is not None:
        cfg.overlap = args.overlap
    cfg.min_valid_frac = args.min_valid_frac
    cfg.normalize = not args.no_normalize
    cfg.clip_percentiles = (args.clip_percentiles[0], args.clip_percentiles[1])
    cfg.max_tiles_per_scene = args.max_tiles_per_scene
    cfg.dry_run = args.dry_run
    cfg.debug = args.debug

    tiles_dir = cfg.output_dir / "tiles"
    metadata_path = cfg.output_dir / "metadata.json"
    report_path = cfg.output_dir / "preprocess_report.json"

    if not cfg.dry_run:
        tiles_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    rasters = find_rasters(cfg.input_dir)
    if not rasters:
        raise SystemExit(f"No rasters found in {cfg.input_dir}")

    metadata: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "tile_size": cfg.tile_size,
        "overlap": cfg.overlap,
        "min_valid_frac": cfg.min_valid_frac,
        "normalize": cfg.normalize,
        "clip_percentiles": list(cfg.clip_percentiles),
        "scenes": [],
        "total_tiles_written": 0,
        "total_tiles_skipped": 0,
    }

    for raster in tqdm(rasters, desc="Preprocessing scenes"):
        summary = process_scene(raster, cfg, tiles_dir, metadata)
        report["scenes"].append(summary)
        report["total_tiles_written"] += summary["tiles_written"]
        report["total_tiles_skipped"] += summary["tiles_skipped"]

    if not cfg.dry_run:
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

    print(
        f"Done. Tiles: {report['total_tiles_written']}, "
        f"skipped: {report['total_tiles_skipped']}, "
        f"output: {cfg.output_dir}"
    )


if __name__ == "__main__":
    main()
