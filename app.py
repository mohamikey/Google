import os
import io
import json
from datetime import datetime, timedelta
from typing import Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Data_processing import run_pipeline

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
LAG_DAYS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7]

app = FastAPI(title="SmartWaste AI API", version="0.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Ingredient breakdown reference table
# ---------------------------------------------------------------------------
INGREDIENTS = {
    "Grilled Chicken": [
        {"name": "Chicken breast", "share": "60%"},
        {"name": "Marinade & spices", "share": "15%"},
        {"name": "Grilling oil", "share": "10%"},
        {"name": "Garnish", "share": "15%"},
    ],
    "Beef Koshari": [
        {"name": "Rice", "share": "35%"},
        {"name": "Lentils", "share": "20%"},
        {"name": "Pasta", "share": "15%"},
        {"name": "Tomato sauce", "share": "20%"},
        {"name": "Fried onions", "share": "10%"},
    ],
    "Falafel Sandwich": [
        {"name": "Falafel mix", "share": "45%"},
        {"name": "Pita bread", "share": "25%"},
        {"name": "Tahini sauce", "share": "15%"},
        {"name": "Salad & pickles", "share": "15%"},
    ],
    "Grilled Fish": [
        {"name": "Fish fillet", "share": "65%"},
        {"name": "Lemon & spices", "share": "10%"},
        {"name": "Grilling oil", "share": "10%"},
        {"name": "Rice side", "share": "15%"},
    ],
    "Molokhia": [
        {"name": "Molokhia leaves", "share": "50%"},
        {"name": "Chicken stock", "share": "30%"},
        {"name": "Garlic & coriander", "share": "10%"},
        {"name": "Rice side", "share": "10%"},
    ],
    "Vegetable Salad": [
        {"name": "Mixed vegetables", "share": "70%"},
        {"name": "Olive oil & lemon", "share": "15%"},
        {"name": "Herbs & seasoning", "share": "15%"},
    ],
    "Pasta Alfredo": [
        {"name": "Pasta", "share": "45%"},
        {"name": "Cheese", "share": "25%"},
        {"name": "Cream sauce", "share": "20%"},
        {"name": "Butter & oil", "share": "10%"},
    ],
    "Fresh Juice": [
        {"name": "Fruit", "share": "80%"},
        {"name": "Water", "share": "15%"},
        {"name": "Sugar / sweetener", "share": "5%"},
    ],
}

DEFAULT_INGREDIENTS = [
    {"name": "Primary ingredient", "share": "60%"},
    {"name": "Sauce / seasoning", "share": "25%"},
    {"name": "Garnish", "share": "15%"},
]


def get_ingredients(item_name: str) -> List[dict]:
    return INGREDIENTS.get(item_name, DEFAULT_INGREDIENTS)


MOCK_ITEMS = [
    {"item_name": "Grilled Chicken",  "predicted_demand": 64, "recommended_prep_quantity": 70,  "waste_risk": "Medium"},
    {"item_name": "Beef Koshari",      "predicted_demand": 58, "recommended_prep_quantity": 64,  "waste_risk": "Low"},
    {"item_name": "Falafel Sandwich", "predicted_demand": 91, "recommended_prep_quantity": 100, "waste_risk": "Low"},
    {"item_name": "Grilled Fish",      "predicted_demand": 22, "recommended_prep_quantity": 24,  "waste_risk": "High"},
    {"item_name": "Vegetable Salad",  "predicted_demand": 33, "recommended_prep_quantity": 36,  "waste_risk": "Low"},
    {"item_name": "Pasta Alfredo",    "predicted_demand": 27, "recommended_prep_quantity": 30,  "waste_risk": "Medium"},
    {"item_name": "Molokhia",          "predicted_demand": 41, "recommended_prep_quantity": 45,  "waste_risk": "High"},
    {"item_name": "Fresh Juice",       "predicted_demand": 73, "recommended_prep_quantity": 80,  "waste_risk": "Low"},
]


# ---------------------------------------------------------------------------
# Global State & Artifact Loading
# ---------------------------------------------------------------------------
_models: dict = {}
_metadata: dict = {}
_daily_sales: Optional[pd.DataFrame] = None


def _load_artifacts() -> None:
    global _models, _metadata, _daily_sales

    models_path = os.path.join(ARTIFACTS_DIR, "models.pkl")
    metadata_path = os.path.join(ARTIFACTS_DIR, "item_metadata.json")
    daily_path = os.path.join(ARTIFACTS_DIR, "daily_sales.csv")

    if os.path.exists(models_path) and os.path.exists(metadata_path) and os.path.exists(daily_path):
        try:
            _models = joblib.load(models_path)
            with open(metadata_path) as f:
                _metadata = json.load(f)
            _daily_sales = pd.read_csv(daily_path, parse_dates=["date"])
            print(f"[app] Loaded {len(_models)} trained models from '{ARTIFACTS_DIR}/'.")
        except Exception as exc:
            print(f"[app] Failed to load artifacts ({exc}) - falling back to demo data.")
            _models, _metadata, _daily_sales = {}, {}, None
    else:
        print(f"[app] No trained artifacts found in '{ARTIFACTS_DIR}/' - serving demo data.")


_load_artifacts()


# ---------------------------------------------------------------------------
# Forecasting & Fallback Logic
# ---------------------------------------------------------------------------
def _build_feature_row(item_name: str) -> Optional[pd.DataFrame]:
    if _daily_sales is None:
        return None

    history = _daily_sales[_daily_sales["item_name"] == item_name].sort_values("date")
    feature_cols = _metadata.get(item_name, {}).get("feature_cols")

    if history.empty or not feature_cols:
        return None

    max_lag = max(LAG_DAYS)
    quantities = history["quantity"].values
    next_date = history["date"].max() + timedelta(days=1)

    # If history is shorter than required lags, pad or use available history safely
    if len(history) < max_lag + 1:
        padded_qty = pd.Series(quantities).reindex(range(max_lag), method='ffill').fillna(quantities[0]).values
        row = {
            "day_of_week": next_date.dayofweek,
            "is_weekend": int(next_date.dayofweek in (5, 6)),
            "month": next_date.month,
            "day_of_month": next_date.day,
        }
        for idx, lag in enumerate(LAG_DAYS):
            row[f"lag_{lag}"] = padded_qty[-(idx + 1)]
        for window in ROLLING_WINDOWS:
            row[f"rolling_avg_{window}"] = quantities[-window:].mean() if len(quantities) >= window else quantities.mean()
        return pd.DataFrame([row])[feature_cols]

    row = {
        "day_of_week": next_date.dayofweek,
        "is_weekend": int(next_date.dayofweek in (5, 6)),
        "month": next_date.month,
        "day_of_month": next_date.day,
    }
    for lag in LAG_DAYS:
        row[f"lag_{lag}"] = quantities[-lag]
    for window in ROLLING_WINDOWS:
        row[f"rolling_avg_{window}"] = quantities[-window:].mean()

    return pd.DataFrame([row])[feature_cols]


def _waste_risk_from_gap(predicted: float, prep: float) -> str:
    if predicted <= 0:
        return "Low"
    gap_pct = (prep - predicted) / predicted
    if gap_pct >= 0.18:
        return "High"
    if gap_pct >= 0.08:
        return "Medium"
    return "Low"


def _forecast_item(item_name: str) -> Optional[dict]:
    if _daily_sales is None or _daily_sales[_daily_sales["item_name"] == item_name].empty:
        return None

    history = _daily_sales[_daily_sales["item_name"] == item_name]
    predicted = None

    if item_name in _models:
        feature_row = _build_feature_row(item_name)
        if feature_row is not None:
            try:
                predicted = max(0.0, float(_models[item_name].predict(feature_row)[0]))
            except Exception:
                pass

    # Fallback to recent average if model prediction isn't available
    if predicted is None or predicted == 0:
        predicted = float(history["quantity"].tail(7).mean())

    predicted = max(1.0, predicted) # Ensure demand is at least 1
    meta = _metadata.get(item_name, {})
    avg = meta.get("avg_demand") or predicted or 1
    std = meta.get("std_demand") or 0
    volatility = min(std / avg, 0.5) if avg else 0.1
    buffer_pct = 0.06 + volatility * 0.3
    recommended_prep = predicted * (1 + buffer_pct)

    return {
        "item_name": item_name,
        "predicted_demand": round(predicted),
        "recommended_prep_quantity": round(recommended_prep),
        "waste_risk": _waste_risk_from_gap(predicted, recommended_prep),
        "ingredients": get_ingredients(item_name),
    }


def _mock_item(item_name: str) -> Optional[dict]:
    for entry in MOCK_ITEMS:
        if entry["item_name"].lower() == item_name.lower():
            return {**entry, "ingredients": get_ingredients(entry["item_name"])}
    return None


def _resolve_item(item_name: str) -> Optional[dict]:
    result = _forecast_item(item_name)
    if result is None:
        result = _mock_item(item_name)
    return result


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------
class Ingredient(BaseModel):
    name: str
    share: str


class ItemForecast(BaseModel):
    item_name: str
    predicted_demand: int
    recommended_prep_quantity: int
    waste_risk: str
    ingredients: List[Ingredient]


class DashboardSummary(BaseModel):
    generated_at: str
    items: List[ItemForecast]


class UploadResult(BaseModel):
    status: str
    rows_parsed: int
    message: str
    filename: str


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.post("/upload-sales", response_model=UploadResult)
async def upload_sales(file: UploadFile = File(...)):
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file exported from your POS.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    temp_path = os.path.join(ARTIFACTS_DIR, "temp_upload.csv")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(raw_bytes)

    try:
        # Run preprocessing pipeline and retrain
        pipeline_result = run_pipeline(temp_path)
        
        # Reload artifacts into memory immediately
        _load_artifacts()
        
        df = pd.read_csv(temp_path)
        rows_parsed = len(df)

        msg = pipeline_result.get("message", "Sales history ingested successfully.") if isinstance(pipeline_result, dict) \
              else f"Sales history ingested successfully. Models trained: {pipeline_result}"

        return {
            "status": "success",
            "rows_parsed": rows_parsed,
            "message": msg,
            "filename": filename,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal pipeline error: {str(exc)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary():
    # If daily sales exist from uploaded data, use those items. Otherwise fallback to mock items.
    if _daily_sales is not None and not _daily_sales.empty:
        item_names = _daily_sales["item_name"].unique().tolist()
    else:
        item_names = [m["item_name"] for m in MOCK_ITEMS]

    items = []
    for name in item_names:
        result = _forecast_item(name)
        if result is None:
            result = _mock_item(name)
        if result:
            items.append(result)

    return {"generated_at": datetime.now().isoformat(), "items": items}


@app.get("/predict/{item_name}", response_model=ItemForecast)
def predict_item(item_name: str):
    result = _resolve_item(item_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No forecast available for '{item_name}'")
    return result


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(_models),
        "using_demo_data": len(_models) == 0,
        "deployment": {
            "compute": "google-cloud-run",
            "compute_ready": True,
            "storage": "google-cloud-storage",
            "storage_ready": True,
            "notes": "Stateless container; PORT env var respected.",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)