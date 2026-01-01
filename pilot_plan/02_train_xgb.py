#!/usr/bin/env python3
"""
Step 2: Train XGBoost model on prepared data.

Usage:
    python pilot_plan/02_train_xgb.py --data pilot_train.csv --output models/xgb_pilot.json
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError as e:
    raise SystemExit(f"Required package not found: {e}")


def train_xgboost(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    spatial_split: bool = True,
) -> dict:
    """Train XGBoost regressor for LST prediction."""
    
    print("="*60)
    print("XGBOOST TRAINING")
    print("="*60)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")
    
    # Features and target
    feature_cols = ['ndvi', 'ndbi', 'lat', 'lon', 'is_urban', 'is_water', 'is_vegetation']
    target_col = 'lst_c'  # Celsius
    
    # Check which features are available
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"Features: {available_features}")
    
    X = df[available_features]
    y = df[target_col]
    
    # Split data
    if spatial_split and 'tile' in df.columns:
        # Use one tile for validation (spatial split)
        tiles = df['tile'].unique()
        print(f"Tiles: {list(tiles)}")
        
        if len(tiles) >= 2:
            val_tile = tiles[-1]  # Last tile for validation
            train_mask = df['tile'] != val_tile
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[~train_mask]
            y_val = y[~train_mask]
            print(f"Spatial split: train on {len(X_train):,}, validate on {len(X_val):,} ({val_tile})")
        else:
            # Random split if only 1 tile
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Val:   {len(X_val):,} samples")
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
    )
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Train RMSE: {train_rmse:.2f} °C")
    print(f"Val RMSE:   {val_rmse:.2f} °C")
    print(f"Val MAE:    {val_mae:.2f} °C")
    print(f"Val R²:     {val_r2:.3f}")
    
    # Feature importance
    print("\nFeature Importance:")
    importance = dict(zip(available_features, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgb_pilot.json"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")
    
    # Save metrics
    metrics = {
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "val_r2": float(val_r2),
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "n_train": len(X_train),
        "n_val": len(X_val),
        "features": available_features,
    }
    
    metrics_path = output_dir / "xgb_pilot_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Check success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    print(f"RMSE < 3 K: {'✅' if val_rmse < 3 else '❌'} ({val_rmse:.2f})")
    print(f"R² > 0.5:   {'✅' if val_r2 > 0.5 else '❌'} ({val_r2:.3f})")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost for LST prediction")
    parser.add_argument("--data", required=True, help="Training CSV from prepare_data.py")
    parser.add_argument("--output", default="models", help="Output directory for model")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    
    data_path = Path(args.data)
    output_dir = Path(args.output)
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return
    
    train_xgboost(data_path, output_dir, args.test_size)


if __name__ == "__main__":
    main()
