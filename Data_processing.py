"""
SmartWaste AI - Step 1: Data Preprocessing & Demand Forecasting Model
========================================================================
Purpose:
    - Load raw restaurant sales CSV data
    - Clean & aggregate to daily quantity-sold-per-item level
    - Engineer time-series features (lags, rolling averages, day-of-week, etc.)
    - Train a baseline demand-forecasting model per menu item
    - Persist trained models + metadata to disk for the API layer (Step 2)

Expected raw CSV columns (adjust COLUMN_MAP below if your dataset differs):
    date        -> transaction date (e.g. "2024-01-15")
    item_name   -> name of the menu item (e.g. "Chicken Biryani")
    quantity    -> units sold in that transaction/row

Output artifacts (saved in ./artifacts/):
    models.pkl          -> dict of {item_name: trained sklearn model}
    daily_sales.csv      -> cleaned, aggregated daily sales table
    item_metadata.json   -> per-item stats (avg demand, std dev, last date, etc.)

Run:
    python data_preprocessing.py --csv path/to/restaurant_sales.csv
"""

import os
import json
import argparse
import warnings
from datetime import timedelta

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG - tweak these if your CSV headers are named differently
# ---------------------------------------------------------------------------
COLUMN_MAP = {

    "date": "date",
    "item_name": "item_name",
    "quantity": "quantity_sold",
}

ARTIFACTS_DIR = "artifacts"
MIN_ROWS_PER_ITEM = 15   # need enough history to train a meaningful model
LAG_DAYS = [1, 2, 3, 7]  # yesterday, 2 days ago, 3 days ago, same day last week
ROLLING_WINDOWS = [3, 7]  # 3-day and 7-day rolling averages


# ---------------------------------------------------------------------------
# STEP 1A: LOAD & CLEAN RAW DATA
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw sales CSV and standardize column names/types."""
    df = pd.read_csv(csv_path)

    # Rename columns to our internal standard names
    reverse_map = {v: k for k, v in COLUMN_MAP.items()}
    df = df.rename(columns=reverse_map)

    required_cols = {"date", "item_name", "quantity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Update COLUMN_MAP at the top of this script to match your CSV headers."
        )

    # Type cleanup
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["item_name"] = df["item_name"].astype(str).str.strip()

    # Drop rows we couldn't parse
    before = len(df)
    df = df.dropna(subset=["date", "item_name", "quantity"])
    dropped = before - len(df)
    if dropped:
        print(f"[load_data] Dropped {dropped} unparseable/invalid rows.")

    return df


# ---------------------------------------------------------------------------
# STEP 1B: AGGREGATE TO DAILY ITEM-LEVEL DEMAND
# ---------------------------------------------------------------------------
def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw transaction rows into one row per (date, item_name)
    representing total quantity sold that day.
    """
    daily = (
        df.groupby(["item_name", pd.Grouper(key="date", freq="D")])["quantity"]
        .sum()
        .reset_index()
    )

    # Fill in missing calendar days per item with 0 sales (important for
    # accurate time-series lag/rolling features - no sale != missing data)
    filled_frames = []
    for item, group in daily.groupby("item_name"):
        group = group.set_index("date").sort_index()
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(full_range, fill_value=0)
        group["item_name"] = item
        group.index.name = "date"
        filled_frames.append(group.reset_index())

    daily_filled = pd.concat(filled_frames, ignore_index=True)
    daily_filled = daily_filled[["date", "item_name", "quantity"]]
    daily_filled = daily_filled.sort_values(["item_name", "date"]).reset_index(drop=True)

    return daily_filled


# ---------------------------------------------------------------------------
# STEP 1C: FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-item time-series features:
      - calendar features (day of week, weekend flag, month)
      - lag features (demand N days ago)
      - rolling average features (smoothed recent demand trend)
    """
    frames = []
    for item, group in daily.groupby("item_name"):
        group = group.sort_values("date").copy()

        # Calendar features
        group["day_of_week"] = group["date"].dt.dayofweek  # 0=Mon ... 6=Sun
        group["is_weekend"] = group["day_of_week"].isin([5, 6]).astype(int)
        group["month"] = group["date"].dt.month
        group["day_of_month"] = group["date"].dt.day

        # Lag features
        for lag in LAG_DAYS:
            group[f"lag_{lag}"] = group["quantity"].shift(lag)

        # Rolling average features (shifted by 1 to avoid leaking current day)
        for window in ROLLING_WINDOWS:
            group[f"rolling_avg_{window}"] = (
                group["quantity"].shift(1).rolling(window=window).mean()
            )

        frames.append(group)

    features_df = pd.concat(frames, ignore_index=True)

    # Drop early rows per item that don't have full lag/rolling history yet
    features_df = features_df.dropna().reset_index(drop=True)

    return features_df


# ---------------------------------------------------------------------------
# STEP 1D: TRAIN ONE FORECASTING MODEL PER ITEM
# ---------------------------------------------------------------------------
def train_models(features_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Train a separate RandomForestRegressor per menu item (simple, robust
    baseline for a hackathon - easy to explain, no tuning required).

    Returns:
        models   -> {item_name: trained_model}
        metadata -> {item_name: {mae, avg_demand, std_demand, last_date, n_samples}}
    """
    feature_cols = (
        ["day_of_week", "is_weekend", "month", "day_of_month"]
        + [f"lag_{lag}" for lag in LAG_DAYS]
        + [f"rolling_avg_{w}" for w in ROLLING_WINDOWS]
    )

    models = {}
    metadata = {}

    for item, group in features_df.groupby("item_name"):
        if len(group) < MIN_ROWS_PER_ITEM:
            print(f"[train_models] Skipping '{item}' - only {len(group)} rows (need {MIN_ROWS_PER_ITEM}+).")
            continue

        X = group[feature_cols]
        y = group["quantity"]

        # Simple chronological split (no shuffling - respects time order)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Quick validation metric for the demo/pitch
        if len(X_test) > 0:
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
        else:
            mae = None

        models[item] = model
        metadata[item] = {
            "mae": round(float(mae), 2) if mae is not None else None,
            "avg_demand": round(float(group["quantity"].mean()), 2),
            "std_demand": round(float(group["quantity"].std()), 2),
            "last_date": group["date"].max().strftime("%Y-%m-%d"),
            "n_samples": int(len(group)),
            "feature_cols": feature_cols,
        }

        print(f"[train_models] Trained '{item}': MAE={metadata[item]['mae']}, "
              f"avg_demand={metadata[item]['avg_demand']}, n={metadata[item]['n_samples']}")

    return models, metadata


# ---------------------------------------------------------------------------
# STEP 1E: PERSIST ARTIFACTS
# ---------------------------------------------------------------------------
def save_artifacts(models: dict, metadata: dict, daily_df: pd.DataFrame) -> None:
    """Save trained models, metadata, and cleaned daily sales table to disk."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(models, os.path.join(ARTIFACTS_DIR, "models.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "item_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    daily_df.to_csv(os.path.join(ARTIFACTS_DIR, "daily_sales.csv"), index=False)

    print(f"\n[save_artifacts] Saved {len(models)} models -> {ARTIFACTS_DIR}/models.pkl")
    print(f"[save_artifacts] Saved metadata -> {ARTIFACTS_DIR}/item_metadata.json")
    print(f"[save_artifacts] Saved cleaned daily sales -> {ARTIFACTS_DIR}/daily_sales.csv")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline(csv_path: str) -> None:
    print(f"[pipeline] Loading data from: {csv_path}")
    raw_df = load_data(csv_path)
    print(f"[pipeline] Loaded {len(raw_df)} raw rows across {raw_df['item_name'].nunique()} items.")

    print("[pipeline] Aggregating to daily item-level demand...")
    daily_df = aggregate_daily_sales(raw_df)
    print(f"[pipeline] Aggregated to {len(daily_df)} daily item rows.")

    print("[pipeline] Engineering time-series features...")
    features_df = engineer_features(daily_df)
    print(f"[pipeline] {len(features_df)} rows remain after feature engineering (lag warm-up dropped).")

    print("[pipeline] Training per-item forecasting models...")
    models, metadata = train_models(features_df)

    print("[pipeline] Saving artifacts...")
    save_artifacts(models, metadata, daily_df)

    print("\n[pipeline] Done. Ready for Step 2 (API layer to serve forecasts).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartWaste AI - Data Preprocessing & Training")
    parser.add_argument("--csv", type=str, required=True, help="Path to restaurant sales CSV file")
    args = parser.parse_args()

    run_pipeline(args.csv)
