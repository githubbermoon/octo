#!/usr/bin/env python3
"""
Fix tile bounds metadata.

The preprocessing saved scene-level bounds for all tiles instead of
tile-specific bounds. This script recalculates correct bounds using
the transform and row/col values.

Usage:
    python scripts/fix_tile_bounds.py --tiles-dir /path/to/tiles/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    from rasterio.transform import array_bounds
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False


def _bounds_from_transform(transform_obj, height: int, width: int) -> tuple | None:
    """Compute bounds from Affine transform."""
    if transform_obj is None:
        return None
    
    # Try rasterio first
    if HAS_RASTERIO:
        try:
            # Parse string if needed
            if isinstance(transform_obj, str):
                import re
                from rasterio.transform import Affine
                numbers = re.findall(r'[-+]?\d*\.?\d+', transform_obj)
                if len(numbers) >= 6:
                    transform_obj = Affine(
                        float(numbers[0]), float(numbers[1]), float(numbers[2]),
                        float(numbers[3]), float(numbers[4]), float(numbers[5])
                    )
            return array_bounds(height, width, transform_obj)
        except Exception:
            pass
    
    # Manual parsing
    if hasattr(transform_obj, 'a'):
        a, b, c = transform_obj.a, transform_obj.b, transform_obj.c
        d, e, f = transform_obj.d, transform_obj.e, transform_obj.f
    elif isinstance(transform_obj, str):
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', transform_obj)
        if len(numbers) >= 6:
            a, b, c = float(numbers[0]), float(numbers[1]), float(numbers[2])
            d, e, f = float(numbers[3]), float(numbers[4]), float(numbers[5])
        else:
            return None
    elif hasattr(transform_obj, '__iter__') and not isinstance(transform_obj, str):
        vals = list(transform_obj)[:6]
        if len(vals) < 6:
            return None
        a, b, c, d, e, f = vals
    else:
        return None
    
    left = float(c)
    top = float(f)
    right = float(left + width * a)
    bottom = float(top + height * e)
    if bottom > top:
        bottom, top = top, bottom
    return (left, bottom, right, top)


def calculate_tile_bounds(
    scene_transform: tuple,
    row: int,
    col: int,
    tile_size: int = 256,
) -> tuple:
    """
    Calculate tile-specific bounds from scene transform and tile position.
    
    Affine transform format: (a, b, c, d, e, f)
    For typical UTM:
      - a = pixel_width (30m for Landsat)
      - e = pixel_height (negative, -30m)
      - c = origin_x (left edge)
      - f = origin_y (top edge)
    
    Returns: (left, bottom, right, top) in projected coordinates
    """
    # Parse Affine transform
    # Rasterio gives: Affine(a, b, c, d, e, f)
    # x = a * col + b * row + c
    # y = d * col + e * row + f
    
    if hasattr(scene_transform, 'a'):
        # It's an Affine object
        a, b, c, d, e, f = (scene_transform.a, scene_transform.b, scene_transform.c,
                            scene_transform.d, scene_transform.e, scene_transform.f)
    elif len(scene_transform) >= 6:
        a, b, c, d, e, f = scene_transform[:6]
    else:
        # Default for 30m Landsat
        a, e = 30, -30
        c, f = 776550, 1444230
        b, d = 0, 0
    
    # Tile corners in pixel space (relative to scene origin)
    # col offset is pixel column, row offset is pixel row
    
    # Top-left corner of tile
    tile_left = c + col * a  # a is pixel width (30m)
    tile_top = f + row * e   # e is negative (-30m), so tile_top decreases with row
    
    # Bottom-right corner of tile
    tile_right = tile_left + tile_size * a   # 256 * 30 = 7680m right
    tile_bottom = tile_top + tile_size * e   # 256 * (-30) = -7680m down
    
    # Ensure bounds format: (left, bottom, right, top) where bottom < top
    if tile_bottom > tile_top:
        tile_bottom, tile_top = tile_top, tile_bottom
    
    return (float(tile_left), float(tile_bottom), float(tile_right), float(tile_top))


def fix_tiles_in_directory(tiles_dir: Path, overwrite: bool = True) -> dict:
    """Fix bounds for all tiles in directory."""
    
    tiles = list(tiles_dir.glob("*.npz"))
    print(f"Found {len(tiles)} tiles in {tiles_dir}")
    
    fixed = 0
    errors = []
    
    for tile_path in tqdm(tiles, desc="Fixing bounds"):
        try:
            # Load tile
            data = dict(np.load(tile_path, allow_pickle=True))
            
            # Get image dimensions
            image = data.get("image")
            if image is None:
                errors.append(f"{tile_path.name}: no image")
                continue
            height, width = image.shape[1], image.shape[2]
            
            # Try tile_transform first (most accurate)
            tile_transform_raw = data.get("tile_transform")
            tile_transform_obj = None
            if tile_transform_raw is not None:
                if isinstance(tile_transform_raw, np.ndarray) and tile_transform_raw.ndim == 0:
                    tile_transform_obj = tile_transform_raw.item()
                else:
                    tile_transform_obj = tile_transform_raw
            
            new_bounds = None
            if tile_transform_obj is not None:
                new_bounds = _bounds_from_transform(tile_transform_obj, height, width)
            
            # Fallback: compute from scene transform + row/col
            if new_bounds is None:
                transform_raw = data.get("transform")
                if transform_raw is None:
                    errors.append(f"{tile_path.name}: no transform")
                    continue
                
                # Extract transform obj
                if isinstance(transform_raw, np.ndarray) and transform_raw.ndim == 0:
                    transform_obj = transform_raw.item()
                else:
                    transform_obj = transform_raw
                
                # Get row/col
                row_raw = data.get("row", 0)
                col_raw = data.get("col", 0)
                row = int(row_raw.item() if isinstance(row_raw, np.ndarray) else row_raw)
                col = int(col_raw.item() if isinstance(col_raw, np.ndarray) else col_raw)
                
                # Parse transform and compute bounds
                import re
                if isinstance(transform_obj, str):
                    numbers = re.findall(r'[-+]?\d*\.?\d+', transform_obj)
                    if len(numbers) >= 6:
                        a = float(numbers[0])
                        e = float(numbers[4])
                        c = float(numbers[2])
                        f = float(numbers[5])
                    else:
                        errors.append(f"{tile_path.name}: could not parse transform")
                        continue
                elif hasattr(transform_obj, 'a'):
                    a, e, c, f = transform_obj.a, transform_obj.e, transform_obj.c, transform_obj.f
                else:
                    errors.append(f"{tile_path.name}: unknown transform type")
                    continue
                
                tile_left = float(c + col * a)
                tile_top = float(f + row * e)
                tile_right = float(tile_left + width * a)
                tile_bottom = float(tile_top + height * e)
                if tile_bottom > tile_top:
                    tile_bottom, tile_top = tile_top, tile_bottom
                new_bounds = (tile_left, tile_bottom, tile_right, tile_top)
            
            old_bounds = tuple(data.get("bounds", []))
            
            # Update bounds
            data["bounds"] = np.array(new_bounds)
            
            # Save (overwrite or to new location)
            if overwrite:
                np.savez_compressed(tile_path, **data)
            else:
                out_path = tile_path.parent / f"fixed_{tile_path.name}"
                np.savez_compressed(out_path, **data)
            
            fixed += 1
            
            # Log first few for verification
            if fixed <= 3:
                print(f"\n  {tile_path.name}:")
                print(f"    row={row}, col={col}")
                print(f"    old bounds: {old_bounds}")
                print(f"    new bounds: {new_bounds}")
            
        except Exception as e:
            errors.append(f"{tile_path.name}: {e}")
    
    return {"fixed": fixed, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Fix tile bounds metadata")
    parser.add_argument("--tiles-dir", required=True, help="Directory containing .npz tiles")
    parser.add_argument("--no-overwrite", action="store_true", help="Don't overwrite, create fixed_* files")
    args = parser.parse_args()
    
    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.exists():
        print(f"❌ Directory not found: {tiles_dir}")
        return
    
    print("="*60)
    print("FIXING TILE BOUNDS")
    print("="*60)
    print(f"Directory: {tiles_dir}")
    print(f"Overwrite: {not args.no_overwrite}")
    
    result = fix_tiles_in_directory(tiles_dir, overwrite=not args.no_overwrite)
    
    print("\n" + "="*60)
    print(f"✅ Fixed: {result['fixed']} tiles")
    if result['errors']:
        print(f"❌ Errors: {len(result['errors'])}")
        for err in result['errors'][:5]:
            print(f"   {err}")


if __name__ == "__main__":
    main()
