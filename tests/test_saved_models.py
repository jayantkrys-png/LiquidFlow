import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from liquidflow.features import calendar_features

MODEL_ROOT = Path("artifacts/models")


@pytest.mark.skipif(not (MODEL_ROOT / "sarimax_liquid.pkl").exists(), reason="Fit models first")
def test_saved_sarimax_reproduces_raw_live_forecast():
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    forecasts = pd.read_csv("artifacts/raw_predictions.csv")
    latest = forecasts.origin.max()
    for category in ["Liquid", "Overnight"]:
        model = SARIMAXResults.load(MODEL_ROOT / f"sarimax_{category.lower()}.pkl")
        months = pd.period_range(pd.Period(latest) + 1, periods=4)
        cal = pd.DataFrame([calendar_features(m) for m in months], index=months)
        rows = forecasts[(forecasts.origin == latest) & (forecasts.category == category) & (forecasts.model == "sarimax")].sort_values("horizon")
        actual = model.get_forecast(4, exog=cal).predicted_mean.to_numpy() * rows.origin_aum.iloc[0]
        # atol=0.1 crore: statsmodels/scipy minor versions produce O(1e-5) differences.
        np.testing.assert_allclose(actual, rows.raw_point.to_numpy(), rtol=1e-6, atol=0.1)


@pytest.mark.skipif(not (MODEL_ROOT / "catboost_full.cbm").exists(), reason="Fit models first")
def test_native_catboost_model_reloads():
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(MODEL_ROOT / "catboost_full.cbm"))
    names = json.loads((MODEL_ROOT / "catboost_full_features.json").read_text())
    features = pd.read_parquet("data/processed/features.parquet")
    current = features[features.origin == features.origin.max()].copy()
    predicted = model.predict(current[names].astype(float)) * current.origin_aum.to_numpy()
    raw = pd.read_csv("artifacts/raw_predictions.csv")
    expected = raw[(raw.origin == raw.origin.max()) & (raw.model == "catboost_full")].sort_values(["category", "horizon"]).raw_point.to_numpy()
    np.testing.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-6)