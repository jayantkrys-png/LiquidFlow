"""Candidate models and the expanding-window refit loop."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
import hashlib
import joblib

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from liquidflow.features import CALENDAR, calendar_features, feature_columns, prepare_flows


def fit_regressor(train: pd.DataFrame, current: pd.DataFrame, model_name: str, protocol: dict):
    scope = "core" if model_name.endswith("_core") else "full"
    exclude = model_name.removeprefix("ridge_no_") if model_name.startswith("ridge_no_") else None
    names = feature_columns(train, scope, exclude)
    x = train[names].astype(float)
    y = train.label_rate.astype(float)
    if model_name.startswith("ridge"):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(), Ridge(alpha=protocol["ridge_alpha"]),
        )
        model.fit(x, y)
    else:
        model = CatBoostRegressor(
            **protocol["catboost"], verbose=False, allow_writing_files=False,
            nan_mode="Min",
        )
        model.fit(x, y)
    prediction = model.predict(current[names].astype(float))
    return np.asarray(prediction) * current.origin_aum.to_numpy(), model, names


def sarimax_forecast(data: pd.DataFrame, origin: str, horizons: list[int], protocol: dict, model_path=None):
    known = data.loc[:pd.Period(origin, "M")]
    rates = (known.net_flow_crore / known.aum_crore.shift(1)).iloc[1:]
    cal = pd.DataFrame([calendar_features(m) for m in rates.index], index=rates.index)
    future_index = pd.period_range(rates.index[-1] + 1, periods=max(horizons), freq="M")
    future_cal = pd.DataFrame([calendar_features(m) for m in future_index], index=future_index)
    spec = protocol["sarimax"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = SARIMAX(
            rates.astype(float), exog=cal.astype(float),
            order=tuple(spec["order"]), seasonal_order=tuple(spec["seasonal_order"]),
            trend=spec["trend"], enforce_stationarity=True, enforce_invertibility=True,
        ).fit(disp=False, maxiter=spec["maximum_iterations"])
        predicted = fit.get_forecast(len(future_index), exog=future_cal).predicted_mean
    if not np.isfinite(predicted).all():
        raise ValueError(f"SARIMAX produced nonfinite forecasts at {origin}")
    if model_path is not None:
        fit.save(model_path)
    return np.asarray(predicted) * known.aum_crore.iloc[-1], {
        "origin": origin, "converged": bool(fit.mle_retvals.get("converged", False)),
        "warnings": sorted(set(str(w.message) for w in caught)),
    }


def rolling_predictions(flows: pd.DataFrame, features: pd.DataFrame, protocol: dict, output: Path):
    candidates = protocol["candidates"]
    # Ablations use the exact same Ridge settings and origin rows.
    ablations = [f"ridge_no_{g}" for g in ("calendar", "rates", "equity", "liquidity")]
    all_models = candidates + ablations
    series = prepare_flows(flows)
    first_origin = protocol["warmup_origin_start"]
    origins = sorted(features.loc[features.origin >= first_origin, "origin"].unique())
    rows, diagnostics = [], []
    final_models = {}
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    for step, origin in enumerate(origins):
        train = features.loc[
            (features.target_month <= origin) & features.actual.notna()
        ].copy()
        if train.empty or train.target_month.max() > origin:
            raise ValueError("Training includes an unknown future label")
        current = features.loc[features.origin == origin].copy()
        if len(current) != len(protocol["categories"]) * len(protocol["horizons"]):
            raise ValueError("Incomplete category/horizon prediction batch")
        predictions = {
            "seasonal_naive": current.seasonal_naive.to_numpy(),
            "seasonal_scaled": current.seasonal_scaled.to_numpy(),
        }
        sarimax_values = np.empty(len(current))
        for category, data in series.items():
            model_path = model_dir / f"sarimax_{category.lower()}.pkl" if origin == origins[-1] else None
            pred, diag = sarimax_forecast(data, origin, protocol["horizons"], protocol, model_path)
            diag["category"] = category
            diagnostics.append(diag)
            positions = np.flatnonzero(current.category.to_numpy() == category)
            for p in positions:
                sarimax_values[p] = pred[int(current.iloc[p].horizon) - 1]
        predictions["sarimax"] = sarimax_values
        for name in all_models:
            if name in predictions:
                continue
            pred, model, names = fit_regressor(train, current, name, protocol)
            predictions[name] = pred
            if origin == origins[-1]:
                final_models[name] = (model, names)
                if isinstance(model, CatBoostRegressor):
                    model.save_model(str(model_dir / f"{name}.cbm"))
                else:
                    joblib.dump(model, model_dir / f"{name}.joblib")
                (model_dir / f"{name}_features.json").write_text(json.dumps(names, indent=2))
        for name in all_models:
            predicted = predictions[name]
            if not np.isfinite(predicted).all():
                raise ValueError(f"{name}: nonfinite prediction at {origin}")
            for pos, (_, record) in enumerate(current.iterrows()):
                row = record[[
                    "origin", "target_month", "category", "horizon",
                    "origin_aum", "actual", "mase_scale", "seasonal_naive",
                ]].to_dict()
                row.update({
                    "model": name, "raw_point": float(predicted[pos]),
                    "training_rows": len(train), "training_label_max": train.target_month.max(),
                })
                rows.append(row)
        if step % 8 == 0 or step == len(origins) - 1:
            print(f"Refit {step + 1}/{len(origins)}: known AMFI through {origin}, train rows {len(train)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    result = pd.DataFrame(rows)
    result.to_csv(output / "raw_predictions.csv", index=False)
    manifest = {
        "known_amfi_through": origins[-1],
        "protocol_sha256": hashlib.sha256(Path("protocol.json").read_bytes()).hexdigest(),
        "flow_sha256": hashlib.sha256(Path("data/processed/amfi_flows.csv").read_bytes()).hexdigest(),
        "macro_sha256": hashlib.sha256(Path("data/processed/macro_monthly.csv").read_bytes()).hexdigest(),
        "raw_predictions_sha256": hashlib.sha256((output / "raw_predictions.csv").read_bytes()).hexdigest(),
    }
    (output / "fit_manifest.json").write_text(json.dumps(manifest, indent=2))
    return result, final_models