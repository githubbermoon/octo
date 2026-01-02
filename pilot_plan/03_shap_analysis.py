#!/usr/bin/env python3
"""
Step 3: SHAP analysis for feature importance visualization.

Usage:
    python pilot_plan/03_shap_analysis.py --model models/xgb_pilot.json --data pilot_train.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    import shap
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(f"Required package not found: {e}")


def shap_analysis(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    sample_size: int = 1000,
) -> None:
    """Generate SHAP analysis plots."""
    
    print("="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    # Load model
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    print(f"Loaded model: {model_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Features (must match training features)
    feature_cols = ['ndvi', 'ndbi', 'lat', 'lon', 'is_urban', 'is_water', 'is_vegetation', 'is_summer']
    available_features = [c for c in feature_cols if c in df.columns]
    
    X = df[available_features]
    
    # Sample for faster SHAP computation
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} rows for SHAP")
    else:
        X_sample = X
    
    # SHAP values
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary plot (bar)
    print("Generating summary plot (bar)...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Summary plot (beeswarm)
    print("Generating summary plot (beeswarm)...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Values Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dependence plots for top features
    print("Generating dependence plots...")
    top_features = ['ndvi', 'ndbi', 'is_urban']
    for feat in top_features:
        if feat in available_features:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(feat, shap_values, X_sample, show=False)
            plt.title(f"SHAP Dependence: {feat}")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_dependence_{feat}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nSHAP plots saved to {output_dir}/")
    
    # Print summary
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    # Mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    for feat, val in sorted(zip(available_features, mean_shap), key=lambda x: -x[1]):
        print(f"  {feat}: {val:.3f}")
    
    print("\n📊 Key findings:")
    ndvi_idx = available_features.index('ndvi') if 'ndvi' in available_features else None
    ndbi_idx = available_features.index('ndbi') if 'ndbi' in available_features else None
    
    if ndvi_idx is not None:
        ndvi_effect = np.corrcoef(X_sample['ndvi'], shap_values[:, ndvi_idx])[0, 1]
        print(f"  NDVI effect: {'cooling ✅' if ndvi_effect < 0 else 'warming ⚠️'}")
    
    if ndbi_idx is not None:
        ndbi_effect = np.corrcoef(X_sample['ndbi'], shap_values[:, ndbi_idx])[0, 1]
        print(f"  NDBI effect: {'warming ✅' if ndbi_effect > 0 else 'cooling ⚠️'}")


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for XGBoost model")
    parser.add_argument("--model", required=True, help="Path to XGBoost model (.json)")
    parser.add_argument("--data", required=True, help="Training data CSV")
    parser.add_argument("--output", default="figures", help="Output directory for plots")
    parser.add_argument("--sample", type=int, default=1000, help="Sample size for SHAP")
    args = parser.parse_args()
    
    shap_analysis(
        Path(args.model),
        Path(args.data),
        Path(args.output),
        args.sample,
    )


if __name__ == "__main__":
    main()
