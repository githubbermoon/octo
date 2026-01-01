#!/usr/bin/env python3
"""
Step 4: Visualize predictions and compute UHI intensity.

⚠️  PILOT SCRIPT WARNING ⚠️
This script:
  - Uses HEURISTIC urban/rural masks (NDBI/NDVI thresholds)
  - Is for PILOT VISUALIZATION ONLY
  - MUST NOT be used for final UHI reporting

For paper-ready UHI analysis, use WorldCover-based masks in align_seasonal.py outputs.

Usage:
    python pilot_plan/04_visualize_uhi.py --model models/xgb_pilot.json --aligned-dir /path/to/aligned/
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Runtime warning on import
warnings.warn(
    "04_visualize_uhi.py uses heuristic NDBI/NDVI masks. "
    "This is for PILOT visualization only. Do NOT use for final UHI reporting.",
    UserWarning,
)

try:
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as e:
    raise SystemExit(f"Required package not found: {e}")


# Paper-ready colormap
LST_COLORS = ['#313695', '#4575b4', '#74add1', '#abd9e9', 
              '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
LST_CMAP = LinearSegmentedColormap.from_list('lst_paper', LST_COLORS, N=256)


def load_tile_features(tile_path: Path) -> tuple:
    """Load tile and prepare features."""
    data = np.load(tile_path, allow_pickle=True)
    
    ndvi = data['ndvi']
    ndbi = data['ndbi']
    lst = data['lst'] - 273.15  # to Celsius
    lat = data['lat'] if 'lat' in data else np.zeros_like(ndvi)
    lon = data['lon'] if 'lon' in data else np.zeros_like(ndvi)
    
    # Create feature array (flat)
    shape = ndvi.shape
    features = pd.DataFrame({
        'ndvi': ndvi.flatten(),
        'ndbi': ndbi.flatten(),
        'lat': lat.flatten(),
        'lon': lon.flatten(),
        'is_urban': np.zeros(ndvi.size),  # Will be filled if WorldCover available
        'is_water': np.zeros(ndvi.size),
        'is_vegetation': np.zeros(ndvi.size),
    })
    
    return features, lst, shape


def visualize_predictions(
    model_path: Path,
    aligned_dir: Path,
    output_dir: Path,
) -> dict:
    """Predict LST and visualize results."""
    
    print("="*60)
    print("VISUALIZATION & UHI ANALYSIS")
    print("="*60)
    
    # Load model
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    
    # Get one tile for visualization
    tiles = list(aligned_dir.glob("*.npz"))
    if not tiles:
        print("No tiles found!")
        return {}
    
    # Use last tile (validation tile)
    tile_path = tiles[-1]
    print(f"Visualizing: {tile_path.name}")
    
    # Load tile
    data = np.load(tile_path, allow_pickle=True)
    ndvi = data['ndvi']
    ndbi = data['ndbi']
    lst_actual = data['lst'] - 273.15
    
    # Prepare features
    features, _, shape = load_tile_features(tile_path)
    
    # Predict
    available_features = ['ndvi', 'ndbi', 'lat', 'lon', 'is_urban', 'is_water', 'is_vegetation']
    X = features[[c for c in available_features if c in features.columns]]
    
    # Handle NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    lst_pred = np.full(X.shape[0], np.nan)
    lst_pred[valid_mask] = model.predict(X[valid_mask])
    lst_pred = lst_pred.reshape(shape)
    
    # Residuals
    residuals = lst_actual - lst_pred
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fixed color range
    vmin, vmax = 20, 50
    
    # Figure 1: Actual vs Predicted
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(lst_actual, cmap=LST_CMAP, vmin=vmin, vmax=vmax)
    axes[0].set_title('Actual LST (°C)')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(lst_pred, cmap=LST_CMAP, vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted LST (°C)')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(residuals, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[2].set_title('Residuals (Actual - Pred)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'XGBoost Prediction: {tile_path.stem}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    valid = ~np.isnan(lst_actual.flatten()) & ~np.isnan(lst_pred.flatten())
    ax.scatter(lst_actual.flatten()[valid], lst_pred.flatten()[valid], alpha=0.1, s=1)
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', label='1:1 line')
    ax.set_xlabel('Actual LST (°C)')
    ax.set_ylabel('Predicted LST (°C)')
    ax.set_title('Actual vs Predicted LST')
    ax.legend()
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    plt.savefig(output_dir / "scatter_actual_vs_pred.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics
    rmse = np.sqrt(np.nanmean(residuals**2))
    mae = np.nanmean(np.abs(residuals))
    
    print(f"\nPrediction Metrics:")
    print(f"  RMSE: {rmse:.2f} °C")
    print(f"  MAE:  {mae:.2f} °C")
    
    # UHI Intensity calculation
    print("\n" + "="*60)
    print("UHI INTENSITY")
    print("="*60)
    
    # Simple urban detection: high NDBI, low NDVI
    urban_mask = (ndbi > np.nanmedian(ndbi)) & (ndvi < np.nanmedian(ndvi))
    rural_mask = (ndvi > np.nanmedian(ndvi)) & (ndbi < np.nanmedian(ndbi))
    
    urban_lst = np.nanmean(lst_actual[urban_mask])
    rural_lst = np.nanmean(lst_actual[rural_mask])
    uhi = urban_lst - rural_lst
    
    print(f"Urban LST (NDBI high, NDVI low): {urban_lst:.1f} °C ({np.sum(urban_mask)} pixels)")
    print(f"Rural LST (NDVI high, NDBI low): {rural_lst:.1f} °C ({np.sum(rural_mask)} pixels)")
    print(f"UHI Intensity: {uhi:.1f} K")
    
    # Check success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    print(f"UHI 2-5 K: {'✅' if 2 <= uhi <= 5 else '⚠️'} ({uhi:.1f} K)")
    
    print(f"\nFigures saved to {output_dir}/")
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "uhi_intensity": float(uhi),
        "urban_lst": float(urban_lst),
        "rural_lst": float(rural_lst),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions and compute UHI")
    parser.add_argument("--model", required=True, help="Path to XGBoost model")
    parser.add_argument("--aligned-dir", required=True, help="Directory with aligned tiles")
    parser.add_argument("--output", default="figures", help="Output directory")
    args = parser.parse_args()
    
    visualize_predictions(
        Path(args.model),
        Path(args.aligned_dir),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
