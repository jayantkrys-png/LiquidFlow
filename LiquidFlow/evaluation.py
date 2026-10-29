"""Forecast intervals, scoring, model selection and cash-buffer costs."""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975)
QCOLS = {q: f"q{int(q * 1000):03d}" for q in QUANTILES}


def calibrate(raw: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """A residual is usable only once its actual month has been released."""
    result = raw.copy()
    for col in [*QCOLS.values(), "point", "calibration_n", "calibration_target_max"]:
        result[col] = np.nan if col != "calibration_target_max" else ""
    lookback = protocol["intervals"]["lookback_errors_per_category_horizon"]
    for (_, _, _), group in raw.groupby(["model", "category", "horizon"], sort=False):
        group = group.sort_values("origin")
        for idx, row in group.iterrows():
            eligible = group.loc[
                (group.target_month <= row.origin)
                & group.actual.notna()
                & (group.origin < row.origin)
            ].sort_values("target_month").tail(lookback)
            errors = (eligible.actual - eligible.raw_point) / eligible.origin_aum
            offsets = np.quantile(errors, QUANTILES, method="linear") if len(errors) else np.zeros(len(QUANTILES))
            values = row.raw_point + offsets * row.origin_aum
            if np.any(np.diff(values) < -1e-8):
                raise AssertionError("Quantile crossing")
            for q, value in zip(QUANTILES, values):
                result.at[idx, QCOLS[q]] = float(value)
            result.at[idx, "point"] = float(values[3])
            result.at[idx, "calibration_n"] = len(eligible)
            result.at[idx, "calibration_target_max"] = (
                eligible.target_month.max() if len(eligible) else ""
            )
    result["calibration_n"] = result.calibration_n.astype(int)
    result["split"] = "warmup"
    dev = result.target_month.between(
        protocol["development_target_start"], protocol["development_target_end"]
    )
    test = result.target_month.between(protocol["test_target_start"], protocol["test_target_end"])
    result.loc[dev, "split"] = "development"
    result.loc[test, "split"] = "test"
    result.loc[result.actual.isna(), "split"] = "live"
    return result


def metric_record(group: pd.DataFrame) -> dict:
    y = group.actual.to_numpy(float)
    p = group.point.to_numpy(float)
    error = np.abs(y - p)
    nonzero = np.abs(y) > 0
    pinballs = []
    for q, column in QCOLS.items():
        residual = y - group[column].to_numpy(float)
        pinballs.append(np.maximum(q * residual, (q - 1) * residual))
    return {
        "n": int(len(group)),
        "mae_crore": float(error.mean()),
        "rmse_crore": float(np.sqrt(np.mean((y - p) ** 2))),
        "mape_pct": float((error[nonzero] / np.abs(y[nonzero])).mean() * 100) if nonzero.any() else None,
        "zero_actuals_excluded_from_mape": int((~nonzero).sum()),
        "near_zero_actuals_under_1_crore": int((np.abs(y) < 1).sum()),
        "aum_scaled_mae_pct": float(np.mean(error / group.origin_aum.to_numpy(float)) * 100),
        "mase": float(np.mean(error / group.mase_scale.to_numpy(float))),
        "pinball_crore": float(np.mean(pinballs)),
        "coverage80_pct": float(np.mean((y >= group.q100) & (y <= group.q900)) * 100),
        "coverage95_pct": float(np.mean((y >= group.q025) & (y <= group.q975)) * 100),
        "width80_crore": float((group.q900 - group.q100).mean()),
        "width95_crore": float((group.q975 - group.q025).mean()),
        "calibration_n_min": int(group.calibration_n.min()),
    }


def score_predictions(predictions: pd.DataFrame, protocol: dict):
    measured = predictions.loc[predictions.split.isin(["development", "test"])].copy()
    details = []
    for keys, group in measured.groupby(["split", "model", "category", "horizon"]):
        details.append(dict(zip(["split", "model", "category", "horizon"], keys)) | metric_record(group))
    detail = pd.DataFrame(details)
    aggregate = []
    for keys, group in measured.loc[
        measured.horizon.isin(protocol["primary_horizons"])
    ].groupby(["split", "model"]):
        aggregate.append(dict(zip(["split", "model"], keys)) | metric_record(group))
    return detail, pd.DataFrame(aggregate)


def choose_model(aggregate: pd.DataFrame, protocol: dict) -> tuple[str, pd.DataFrame]:
    development = aggregate.loc[
        (aggregate.split == "development")
        & aggregate.model.isin(protocol["candidates"])
    ].sort_values(["aum_scaled_mae_pct", "model"])
    if len(development) != len(protocol["candidates"]):
        raise ValueError("Missing candidate on the earlier development split")
    return str(development.iloc[0].model), development


def cash_cost(buffer, realized_outflow, days, annual_gap=0.005, penalty=3.0, charge_all=False):
    buffer = np.asarray(buffer, dtype=float)
    outflow = np.asarray(realized_outflow, dtype=float)
    if np.any(buffer < 0) or np.any(outflow < 0) or annual_gap < 0 or np.any(np.asarray(penalty) < 0):
        raise ValueError("Cash-cost inputs must be nonnegative")
    shortfall = np.maximum(outflow - buffer, 0)
    idle = buffer if charge_all else np.maximum(buffer - outflow, 0)
    drag = annual_gap * np.asarray(days) / 365.0 * idle
    return drag + penalty * shortfall, drag, shortfall


def policy_backtest(predictions: pd.DataFrame, selected: str, protocol: dict):
    # h1 only: no repeated charging of overlapping h2/h3/h4 forecasts.
    data = predictions.loc[
        (predictions.model == selected)
        & (predictions.horizon == protocol["policy"]["scored_horizon"])
        & (predictions.split == "test")
    ].copy()
    data["outflow_crore"] = np.maximum(0, -data.actual)
    data["uncapped_buffer_crore"] = np.maximum(0, -data.q050)
    data["q95_buffer_crore"] = np.minimum(data.uncapped_buffer_crore, data.origin_aum)
    data["naive_buffer_crore"] = np.minimum(np.maximum(0, -data.seasonal_naive), data.origin_aum)
    data["days"] = data.target_month.map(lambda m: calendar.monthrange(int(m[:4]), int(m[5:]))[1])
    scenarios, summary = [], []
    for gap in protocol["policy"]["sensitivity_yield_gaps"]:
        for penalty in protocol["policy"]["sensitivity_penalties"]:
            for name, col in [("fixed_q95", "q95_buffer_crore"), ("seasonal_naive", "naive_buffer_crore")]:
                total, drag, shortfall = cash_cost(data[col], data.outflow_crore, data.days, gap, penalty)
                scenarios.append({
                    "policy": name, "annual_gap": gap, "penalty": penalty,
                    "cost_crore": float(total.sum()), "drag_crore": float(drag.sum()),
                    "shortfall_crore": float(shortfall.sum()),
                })
    for category, group in data.groupby("category"):
        for name, col in [("fixed_q95", "q95_buffer_crore"), ("seasonal_naive", "naive_buffer_crore")]:
            total, drag, shortfall = cash_cost(
                group[col], group.outflow_crore, group.days,
                protocol["policy"]["annual_yield_gap"], protocol["policy"]["shortfall_penalty_multiplier"],
            )
            conservative, _, _ = cash_cost(
                group[col], group.outflow_crore, group.days,
                protocol["policy"]["annual_yield_gap"], protocol["policy"]["shortfall_penalty_multiplier"],
                charge_all=True,
            )
            summary.append({
                "category": category, "policy": name, "n": len(group),
                "cost_crore": float(total.sum()),
                "all_buffer_drag_cost_crore": float(conservative.sum()),
                "drag_crore": float(drag.sum()), "shortfall_crore": float(shortfall.sum()),
                "shortfall_months": int((shortfall > 0).sum()),
                "mean_buffer_crore": float(group[col].mean()),
                "mean_buffer_aum_pct": float((group[col] / group.origin_aum).mean() * 100),
                "aum_cap_hits": int((group.uncapped_buffer_crore > group.origin_aum).sum()) if name == "fixed_q95" else 0,
            })
    return data, pd.DataFrame(summary), pd.DataFrame(scenarios)


def policy_sanity_checks(predictions: pd.DataFrame, selected: str, protocol: dict):
    """Compare buffer rules under both readings of the shortfall penalty.

    Includes a seasonal-naive 95% band (no model) and holding all assets in
    cash, as reference points for the selected model's buffer.
    """
    base = predictions.loc[
        (predictions.model == selected)
        & (predictions.horizon == 1)
        & (predictions.split == "test")
    ].copy().set_index(["category", "target_month"])
    naive = predictions.loc[
        (predictions.model == "seasonal_naive")
        & (predictions.horizon == 1)
        & (predictions.split == "test")
    ].set_index(["category", "target_month"]).reindex(base.index)
    if naive.q050.isna().any():
        raise ValueError("Missing aligned seasonal-naive quantile forecast")
    outflow = np.maximum(0, -base.actual.to_numpy())
    days = np.asarray([
        calendar.monthrange(int(month[:4]), int(month[5:]))[1]
        for _, month in base.index
    ])
    aum = base.origin_aum.to_numpy()
    buffers = {
        "prior_year_outflow": np.minimum(np.maximum(0, -base.seasonal_naive.to_numpy()), aum),
        "selected_model_q95": np.minimum(np.maximum(0, -base.q050.to_numpy()), aum),
        "seasonal_naive_q95": np.minimum(np.maximum(0, -naive.q050.to_numpy()), aum),
        "all_origin_aum_reference": aum,
    }
    rows = []
    gap = protocol["policy"]["annual_yield_gap"]
    for interpretation in ("literal_3_rupees_per_rupee", "three_times_monthly_yield_gap"):
        penalty = 3.0 if interpretation.startswith("literal") else 3 * gap * days / 365
        for name, buffer in buffers.items():
            cost, drag, shortfall = cash_cost(buffer, outflow, days, gap, penalty)
            rows.append({
                "interpretation": interpretation, "policy": name, "n": len(base),
                "cost_crore": float(cost.sum()), "drag_crore": float(drag.sum()),
                "shortfall_crore": float(shortfall.sum()),
                "shortfall_category_months": int((shortfall > 0).sum()),
                "mean_buffer_aum_pct": float(np.mean(buffer / aum) * 100),
                "mean_buffer_crore": float(np.mean(buffer)),
            })
    return pd.DataFrame(rows)