"""
SmartWaste AI - Step 2: FastAPI Backend
==========================================
Serves the kitchen prep-list dashboard: per-item demand forecasts,
recommended prep quantities, waste-risk flags, and an ingredient
breakdown for each dish.

Data source priority:
    1. Trained artifacts from data_preprocessing.py (./artifacts/models.pkl,
       item_metadata.json, daily_sales.csv) - used to produce real
       next-day forecasts per item.
    2. Built-in demo dataset (MOCK_ITEMS below) - used automatically if
       artifacts haven't been generated yet, so the frontend never breaks
       mid-hackathon while the model/data side is still in progress.

Endpoints:
    GET /dashboard/summary      -> full prep list for every tracked item
    GET /predict/{item_name}    -> forecast for a single item (case-insensitive)
    GET /health                 -> quick status check

Run:
    pip install fastapi uvicorn pandas scikit-learn joblib --break-system-packages
    uvicorn app:app --reload --port 8000
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
LAG_DAYS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7]

app = FastAPI(title="SmartWaste AI API", version="0.2.0")

# The dashboard is a standalone index.html (often opened via file:// or a
# dev server on a different port), so allow any origin for the hackathon.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Ingredient breakdown - static reference table.
# In a production system this would live in a recipes table; for the
# prototype it's a simple per-dish lookup with estimated portion shares.
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

# Used for any item that isn't in the table above (e.g. a new menu item
# added to the CSV that nobody has mapped ingredients for yet).
DEFAULT_INGREDIENTS = [
    {"name": "Primary ingredient", "share": "60%"},
    {"name": "Sauce / seasoning", "share": "25%"},
    {"name": "Garnish", "share": "15%"},
]


def get_ingredients(item_name: str) -> List[dict]:
    return INGREDIENTS.get(item_name, DEFAULT_INGREDIENTS)


# ---------------------------------------------------------------------------
# Demo dataset - used whenever trained artifacts aren't available yet.
# ---------------------------------------------------------------------------
MOCK_ITEMS = [
    {"item_name": "Grilled Chicken",  "predicted_demand": 64, "recommended_prep_quantity": 70,  "waste_risk": "Medium"},
    {"item_name": "Beef Koshari",     "predicted_demand": 58, "recommended_prep_quantity": 64,  "waste_risk": "Low"},
    {"item_name": "Falafel Sandwich", "predicted_demand": 91, "recommended_prep_quantity": 100, "waste_risk": "Low"},
    {"item_name": "Grilled Fish",     "predicted_demand": 22, "recommended_prep_quantity": 24,  "waste_risk": "High"},
    {"item_name": "Vegetable Salad",  "predicted_demand": 33, "recommended_prep_quantity": 36,  "waste_risk": "Low"},
    {"item_name": "Pasta Alfredo",    "predicted_demand": 27, "recommended_prep_quantity": 30,  "waste_risk": "Medium"},
    {"item_name": "Molokhia",         "predicted_demand": 41, "recommended_prep_quantity": 45,  "waste_risk": "High"},
    {"item_name": "Fresh Juice",      "predicted_demand": 73, "recommended_prep_quantity": 80,  "waste_risk": "Low"},
]


# ---------------------------------------------------------------------------
# Load trained artifacts if they exist (produced by data_preprocessing.py)
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
        print(f"[app] No trained artifacts found in '{ARTIFACTS_DIR}/' - serving demo data. "
              f"Run data_preprocessing.py first for real forecasts.")


_load_artifacts()


# ---------------------------------------------------------------------------
# Forecasting helpers
# ---------------------------------------------------------------------------
def _build_feature_row(item_name: str) -> Optional[pd.DataFrame]:
    """Build the single feature row needed to forecast the next day for one item."""
    if _daily_sales is None:
        return None

    history = _daily_sales[_daily_sales["item_name"] == item_name].sort_values("date")
    feature_cols = _metadata.get(item_name, {}).get("feature_cols")

    if history.empty or not feature_cols:
        return None

    max_lag = max(LAG_DAYS)
    if len(history) < max_lag + 1:
        return None

    next_date = history["date"].max() + timedelta(days=1)
    quantities = history["quantity"].values

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
    """Classify waste risk from how far recommended prep sits above the forecast."""
    if predicted <= 0:
        return "Low"
    gap_pct = (prep - predicted) / predicted
    if gap_pct >= 0.18:
        return "High"
    if gap_pct >= 0.08:
        return "Medium"
    return "Low"


def _forecast_item(item_name: str) -> Optional[dict]:
    """Return a real forecast for one item if a trained model + history exist."""
    if item_name not in _models:
        return None

    feature_row = _build_feature_row(item_name)
    if feature_row is None:
        return None

    model = _models[item_name]
    predicted = max(0.0, float(model.predict(feature_row)[0]))

    # Safety buffer scales with the item's historical demand volatility -
    # noisier items get a slightly larger prep cushion than steady sellers.
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
    """Try a real forecast first (exact, then case-insensitive), then demo data."""
    result = _forecast_item(item_name)
    if result is None:
        for trained_name in _models:
            if trained_name.lower() == item_name.lower():
                result = _forecast_item(trained_name)
                break
    if result is None:
        result = _mock_item(item_name)
    return result


# ---------------------------------------------------------------------------
# Response models
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary():
    """Full kitchen prep list for tomorrow, across every tracked item."""
    item_names = list(_models.keys()) if _models else [m["item_name"] for m in MOCK_ITEMS]

    items = []
    for name in item_names:
        result = _forecast_item(name) or _mock_item(name)
        if result:
            items.append(result)

    return {"generated_at": datetime.now().isoformat(), "items": items}


@app.get("/predict/{item_name}", response_model=ItemForecast)
def predict_item(item_name: str):
    """Forecast for a single menu item, looked up case-insensitively."""
    result = _resolve_item(item_name)

    if result is None:
        raise HTTPException(status_code=404, detail=f"No forecast available for '{item_name}'")

    return result


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(_models), "using_demo_data": len(_models) == 0}