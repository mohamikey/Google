import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

ARTIFACTS_DIR = "artifacts"
LAG_DAYS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7]

def clean_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize messy POS export column names to standard internal schema safely."""
    df = df.dropna(how='all')
    
    col_map = {}
    assigned_targets = set()
    
    # First pass: precise matching
    for col in df.columns:
        c_lower = str(col).strip().lower()
        if c_lower in ['date', 'order_date', 'sales_date', 'transaction_date'] and 'date' not in assigned_targets:
            col_map[col] = 'date'
            assigned_targets.add('date')
        elif c_lower in ['item_name', 'product', 'dish', 'item', 'product_name'] and 'item_name' not in assigned_targets:
            col_map[col] = 'item_name'
            assigned_targets.add('item_name')
        elif c_lower in ['quantity', 'qty', 'sold', 'count', 'units'] and 'quantity' not in assigned_targets:
            col_map[col] = 'quantity'
            assigned_targets.add('quantity')

    # Second pass: safe substring matching for remaining unassigned targets
    for col in df.columns:
        if col in col_map:
            continue
        c_lower = str(col).strip().lower()
        if 'date' not in assigned_targets and 'date' in c_lower and 'time' not in c_lower:
            col_map[col] = 'date'
            assigned_targets.add('date')
        elif 'item_name' not in assigned_targets and ('item' in c_lower or 'product' in c_lower) and 'price' not in c_lower and 'type' not in c_lower:
            col_map[col] = 'item_name'
            assigned_targets.add('item_name')
        elif 'quantity' not in assigned_targets and ('qty' in c_lower or 'quantity' in c_lower) and 'price' not in c_lower and 'amount' not in c_lower:
            col_map[col] = 'quantity'
            assigned_targets.add('quantity')

    df = df.rename(columns=col_map)
    
    required = ['date', 'item_name', 'quantity']
    missing = [req for req in required if req not in df.columns]
    
    if missing:
        raise ValueError(
            f"Data validation failed. Missing mapped columns: {', '.join(missing)}. "
            f"Found columns: {', '.join(df.columns.tolist())}"
        )
        
    try:
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['date'])
    except Exception as e:
        raise ValueError(f"Failed to parse dates in the 'date' column: {str(e)}")
        
    # Guarantee df['quantity'] is a 1D Series if multiple columns matched
    if isinstance(df['quantity'], pd.DataFrame):
        df['quantity'] = df['quantity'].iloc[:, 0]
        
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    
    df = df.groupby(['date', 'item_name'], as_index=False)['quantity'].sum()
    return df[['date', 'item_name', 'quantity']].sort_values(['item_name', 'date'])

def build_features(group_df: pd.DataFrame) -> pd.DataFrame:
    """Generate lag and rolling features for a single item timeseries."""
    df = group_df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day

    for lag in LAG_DAYS:
        df[f'lag_{lag}'] = df['quantity'].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f'rolling_avg_{window}'] = df['quantity'].shift(1).rolling(window=window).mean()

    return df.dropna()


def run_pipeline(csv_path: str) -> dict:
    """Main execution pipeline: ingests CSV, trains per-item models, saves artifacts."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    try:
        raw_df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")
        
    if raw_df.empty:
        raise ValueError("The uploaded CSV file is empty.")
        
    # Map headers and validate data
    df = clean_and_map_columns(raw_df)
    
    # Save cleaned daily sales for backend feature builder reference
    daily_path = os.path.join(ARTIFACTS_DIR, "daily_sales.csv")
    df.to_csv(daily_path, index=False)
    
    models = {}
    metadata = {}
    
    for item_name, group in df.groupby('item_name'):
        processed = build_features(group)
        
        # Skip items with insufficient history to build lag features
        if len(processed) < max(LAG_DAYS) + 1:
            continue 
            
        feature_cols = [c for c in processed.columns if c not in ['date', 'item_name', 'quantity']]
        X = processed[feature_cols]
        y = processed['quantity']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        models[item_name] = model
        metadata[item_name] = {
            "feature_cols": feature_cols,
            "avg_demand": float(y.mean()),
            "std_demand": float(y.std()) if len(y) > 1 else 0.0
        }
        
    if not models:
        raise ValueError("Not enough historical data to train models for any items. Provide a longer sales history.")
        
    joblib.dump(models, os.path.join(ARTIFACTS_DIR, "models.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "item_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return {
        "status": "success",
        "items_trained": len(models),
        "message": f"Successfully trained and updated models for {len(models)} items."
    }

if __name__ == "__main__":
    # Local testing execution
    test_file = "data/daily_sales.csv" if os.path.exists("data/daily_sales.csv") else "artifacts/daily_sales.csv"
    if os.path.exists(test_file):
        try:
            print(run_pipeline(test_file))
        except Exception as e:
            print(f"Pipeline Error: {e}")