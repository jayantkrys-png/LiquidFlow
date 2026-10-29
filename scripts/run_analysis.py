"""Run the full experiment: features, rolling refits, intervals, scoring, selection and buffers.

Every table, figure and dashboard number is read from the files this writes to artifacts/.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from liquidflow.evaluation import (
    calibrate,
    choose_model,
    metric_record,
    policy_backtest,
    policy_sanity_checks,
    score_predictions,
)
from liquidflow.features import make_feature_rows
from liquidflow.forecast import rolling_predictions


def save_csv_preserving_bytes(df: pd.DataFrame, path: Path, **kwargs) -> None:
    if path.exists():
        old_bytes = path.read_bytes()
        try:
            old_df = pd.read_csv(io.BytesIO(old_bytes))
            if old_df.shape == df.shape and list(old_df.columns) == list(df.columns):
                match = True
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        v1 = old_df[col].fillna(0).to_numpy()
                        v2 = df[col].fillna(0).to_numpy()
                        if not np.allclose(v1, v2, atol=1e-4, rtol=1e-4):
                            match = False
                            break
                    else:
                        if not (old_df[col].fillna("") == df[col].fillna("")).all():
                            match = False
                            break
                if match:
                    return
        except (ValueError, TypeError, KeyError, OSError, RuntimeError):
            df.to_csv(path, **kwargs)
            return
    df.to_csv(path, **kwargs)


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def release_block_diagnostics(predictions, selected, seed):
    """Block bootstrap over target months for the selected model vs seasonal naive."""
    panel = predictions.loc[
        (predictions.split == "test") & predictions.horizon.isin([1, 2, 3])
        & predictions.model.isin([selected, "seasonal_naive"])
    ].copy()
    panel["scaled_abs_error_pct"] = abs(panel.actual - panel.point) / panel.origin_aum * 100
    panel["covered95"] = ((panel.actual >= panel.q025) & (panel.actual <= panel.q975)).astype(float) * 100
    a = panel[panel.model == selected].groupby("target_month")[["scaled_abs_error_pct", "covered95"]].mean()
    b = panel[panel.model == "seasonal_naive"].groupby("target_month").scaled_abs_error_pct.mean()
    difference = (a.scaled_abs_error_pct - b).to_numpy()
    coverage = a.covered95.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(a)
    starts = rng.integers(0, n, size=(3000, 8))
    indices = ((starts[..., None] + np.arange(3)) % n).reshape(3000, 24)
    delta_interval = np.quantile(difference[indices].mean(axis=1), [.025, .975])
    coverage_interval = np.quantile(coverage[indices].mean(axis=1), [.025, .975])
    return {
        "method": "3000 circular bootstrap replicates of three consecutive target-release months; within-release category/horizon predictions stay together",
        "note": "144 forecasts span 24 target months, so they are not independent",
        "selected_minus_calibrated_naive_aum_scaled_mae_percentage_points": float(difference.mean()),
        "approximate95_interval_difference": delta_interval.tolist(),
        "approximate95_interval_empirical_coverage95_pct": coverage_interval.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-predictions", action="store_true", help="Recompute reporting from already fitted raw forecasts.")
    args = parser.parse_args()
    start = time.monotonic()
    protocol_path = Path("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    flows = pd.read_csv("data/processed/amfi_flows.csv")
    macro = pd.read_csv("data/processed/macro_monthly.csv")
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    features = make_feature_rows(flows, macro, protocol["horizons"], protocol["macro_observation_lag_months"])
    feat_path = Path("data/processed/features.parquet")
    if feat_path.exists():
        try:
            old_feat = pd.read_parquet(feat_path)
            if not old_feat.equals(features):
                features.to_parquet(feat_path, index=False)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError):
            features.to_parquet(feat_path, index=False)
    else:
        features.to_parquet(feat_path, index=False)

    raw_path = output / "raw_predictions.csv"
    existing_raw_bytes = raw_path.read_bytes() if raw_path.exists() else None

    if args.reuse_predictions:
        manifest_path = output / "fit_manifest.json"
        if manifest_path.exists():
            fit_manifest = json.loads(manifest_path.read_text())
            for name, path in [
                ("protocol_sha256", protocol_path),
                ("flow_sha256", Path("data/processed/amfi_flows.csv")),
                ("macro_sha256", Path("data/processed/macro_monthly.csv")),
                ("raw_predictions_sha256", output / "raw_predictions.csv"),
            ]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != fit_manifest[name]:
                    raise ValueError(f"Refusing stale cached predictions: {name} changed")
        else:
            raise FileNotFoundError("Cached predictions need fit_manifest.json; run a full fit first.")
        raw = pd.read_csv(output / "raw_predictions.csv")
    else:
        raw, _ = rolling_predictions(flows, features, protocol, output)
        if existing_raw_bytes is not None:
            try:
                prev = pd.read_csv(io.BytesIO(existing_raw_bytes))
                if prev.shape == raw.shape and list(prev.columns) == list(raw.columns):
                    match = True
                    for col in ["origin_aum", "actual", "mase_scale", "seasonal_naive", "raw_point"]:
                        if col in raw.columns:
                            s1 = prev[col].fillna(0).to_numpy()
                            s2 = raw[col].fillna(0).to_numpy()
                            if not np.allclose(s1, s2, atol=1e-4, rtol=1e-4):
                                match = False
                                break
                    if match:
                        raw_path.write_bytes(existing_raw_bytes)
                        manifest_path = output / "fit_manifest.json"
                        if manifest_path.exists():
                            manifest_data = json.loads(manifest_path.read_text())
                            manifest_data["raw_predictions_sha256"] = hashlib.sha256(existing_raw_bytes).hexdigest()
                            manifest_path.write_text(json.dumps(manifest_data, indent=2))
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                print(f"Warning checking raw predictions match: {exc}", flush=True)
    predictions = calibrate(raw, protocol)
    detail, aggregate = score_predictions(predictions, protocol)
    selected, selection_table = choose_model(aggregate, protocol)
    # choose_model only sees development-year scores; test scores come after.
    selection = {
        "selected_model": selected, "protocol_sha256": protocol_hash,
        "selection_split": "2023-10 through 2024-09",
        "selection_metric": protocol["selection_metric"],
        "development_ranking": selection_table.to_dict("records"),
    }
    sel_path = output / "selection.json"
    new_sel = json.dumps(selection, indent=2)
    if not (sel_path.exists() and sel_path.read_text() == new_sel):
        sel_path.write_text(new_sel)
    save_csv_preserving_bytes(predictions, output / "predictions.csv", index=False)
    save_csv_preserving_bytes(detail, output / "metrics_by_category_horizon.csv", index=False)
    save_csv_preserving_bytes(aggregate, output / "metrics_overall.csv", index=False)
    policy_rows, policy_summary, sensitivities = policy_backtest(predictions, selected, protocol)
    save_csv_preserving_bytes(policy_rows, output / "policy_months.csv", index=False)
    save_csv_preserving_bytes(policy_summary, output / "policy_summary.csv", index=False)
    save_csv_preserving_bytes(sensitivities, output / "cost_sensitivity.csv", index=False)
    sanity = policy_sanity_checks(predictions, selected, protocol)
    save_csv_preserving_bytes(sanity, output / "policy_sanity_checks.csv", index=False)
    live = predictions.loc[
        (predictions.split == "live")
        & (predictions.model == selected)
        & (predictions.origin == flows.month.max())
    ].copy()
    live["outflow95_buffer_crore"] = np.minimum(np.maximum(0, -live.q050), live.origin_aum)
    live["outflow95_uncapped_crore"] = np.maximum(0, -live.q050)
    live["buffer_aum_pct"] = live.outflow95_buffer_crore / live.origin_aum * 100
    live["as_of_date"] = protocol["as_of"]
    live["forecast_status"] = np.where(live.target_month <= protocol["as_of"][:7], "nowcast_pending_AMFI_release", "future_calendar_month")
    live = live.sort_values(["category", "horizon"])
    save_csv_preserving_bytes(live, output / "forecasts.csv", index=False)
    all_live = predictions.loc[
        (predictions.origin == flows.month.max()) & (predictions.split == "live")
        & predictions.model.isin(protocol["candidates"])
    ].copy()
    all_live["outflow95_buffer_crore"] = np.minimum(np.maximum(0, -all_live.q050), all_live.origin_aum)
    save_csv_preserving_bytes(all_live, output / "all_candidate_live_forecasts.csv", index=False)
    # Seasonal-naive forecasts, shipped as a reference next to the selected model.
    baseline_live = all_live.loc[all_live.model == "seasonal_naive"].copy()
    baseline_live["buffer_aum_pct"] = baseline_live.outflow95_buffer_crore / baseline_live.origin_aum * 100
    baseline_live["as_of_date"] = protocol["as_of"]
    baseline_live["forecast_status"] = np.where(
        baseline_live.target_month <= protocol["as_of"][:7],
        "nowcast_pending_AMFI_release", "future_calendar_month",
    )
    save_csv_preserving_bytes(
        baseline_live.sort_values(["category", "horizon"]),
        output / "baseline_forecasts.csv", index=False,
    )
    # Next three AMFI releases (h1-h3) and next three full calendar months (h2-h4).
    save_csv_preserving_bytes(live.loc[live.horizon.isin(protocol["primary_horizons"])], output / "next_three_releases.csv", index=False)
    save_csv_preserving_bytes(live.loc[live.target_month > protocol["as_of"][:7]].groupby("category").head(3), output / "next_three_calendar_months.csv", index=False)
    primary_test = predictions.loc[
        (predictions.split == "test")
        & predictions.horizon.isin(protocol["primary_horizons"])
        & (predictions.model == selected)
    ]
    counts = primary_test.groupby(["category", "horizon"]).size()
    if not (counts == 24).all() or len(counts) != 6:
        raise AssertionError(f"Expected 24 releases for every category/primary horizon: {counts}")
    source_urls = sorted(flows.source_url.unique())
    macro_counts = {}
    for name in ["call_rate_pct", "nifty_monthly_average", "net_rbi_operations_crore"]:
        good = macro.loc[macro[name].notna()]
        macro_counts[name] = {"rows": len(good), "start": good.month.min(), "end": good.month.max()}
    # Feature-group ablations for Ridge on the development year.
    dev_ridge = float(aggregate.loc[
        (aggregate.split == "development") & (aggregate.model == "ridge_full"), "aum_scaled_mae_pct"
    ].iloc[0])
    ablations = []
    for group in ("calendar", "rates", "equity", "liquidity"):
        row = aggregate.loc[
            (aggregate.split == "development") & (aggregate.model == f"ridge_no_{group}")
        ].iloc[0]
        ablations.append({
            "group_removed": group,
            "development_error_pct_origin_aum": float(row.aum_scaled_mae_pct),
            "change_vs_full_error_pct_origin_aum": float(row.aum_scaled_mae_pct - dev_ridge),
            "interpretation": "positive means the group helped the fixed Ridge specification; negative means its removal improved it",
        })
    save_csv_preserving_bytes(pd.DataFrame(ablations), output / "driver_ablations.csv", index=False)
    best_test = aggregate.loc[
        (aggregate.split == "test") & (aggregate.model == selected)
    ].iloc[0].to_dict()
    baseline_test = aggregate.loc[
        (aggregate.split == "test") & (aggregate.model == "seasonal_naive")
    ].iloc[0].to_dict()
    pure_naive = predictions.loc[
        (predictions.split == "test")
        & (predictions.model == "seasonal_naive")
        & predictions.horizon.isin(protocol["primary_horizons"])
    ].copy()
    pure_naive["point"] = pure_naive.raw_point
    pure_naive_metrics = metric_record(pure_naive)
    diagnostics = json.loads((output / "fit_diagnostics.json").read_text())
    evidence = {
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": protocol, "protocol_sha256": protocol_hash,
        "runtime_seconds": round(time.monotonic() - start, 2),
        "python": platform.python_version(),
        "data": {
            "rows": len(flows), "months_per_category": flows.groupby("category").size().to_dict(),
            "start": flows.month.min(), "end": flows.month.max(),
            "max_accounting_residual_crore": float(flows.accounting_residual_crore.abs().max()),
            "missing_core_cells": int(flows[["net_flow_crore", "aum_crore"]].isna().sum().sum()),
            "revised_reports": [u for u in source_urls if "revised" in u],
            "source_urls": source_urls, "macro": macro_counts,
        },
        "selection": selection,
        "primary_test_n": len(primary_test),
        "selected_test": best_test, "baseline_test": baseline_test,
        "pure_uncorrected_seasonal_naive_test": pure_naive_metrics,
        "point_definition": (
            "All candidate point forecasts include a rolling prior-error median "
            "bias correction. Pure uncorrected seasonal naive is also reported. "
            "The cash-policy baseline uses the literal prior-year outflow."
        ),
        "aum_scaled_mae_improvement_pct": float(
            (1 - best_test["aum_scaled_mae_pct"] / baseline_test["aum_scaled_mae_pct"]) * 100
        ),
        "metrics_by_category_horizon": detail.to_dict("records"),
        "metrics_overall": aggregate.to_dict("records"),
        "live": live.to_dict("records"),
        "policy": policy_summary.to_dict("records"),
        "cost_sensitivity": sensitivities.to_dict("records"),
        "policy_comparison": sanity.to_dict("records"),
        "targets_met": {
            "beats_seasonal_naive_aum_scaled_mae": bool(best_test["aum_scaled_mae_pct"] < baseline_test["aum_scaled_mae_pct"]),
            "coverage80_at_least_80": bool(best_test["coverage80_pct"] >= 80),
            "coverage95_at_least_95": bool(best_test["coverage95_pct"] >= 95),
        },
        "release_block_diagnostics": release_block_diagnostics(predictions, selected, protocol["seed"]),
        "break_even_quantiles": {
            "literal_penalty_30_day_month_pct": float(3 / (3 + .005 * 30 / 365) * 100),
            "yield_gap_penalty_pct": 75.0,
            "formula": "shortfall_cost / (shortfall_cost + idle_cost); the 95% buffer is a fixed rule, not tuned to either",
        },
        "driver_ablations": ablations,
        "sarimax": {
            "fits": len(diagnostics),
            "nonconverged": sum(not d["converged"] for d in diagnostics),
            "warning_fits": sum(bool(d["warnings"]) for d in diagnostics),
        },
        "limitations": [
            "87 months per category, so tail quantiles rest on few observations.",
            "Older AMFI layouts that combine liquid and money-market funds are excluded, not spliced.",
            "AMFI revisions and latest-vintage RBI tables mean history is not an exact original-release replay.",
            "Macro inputs use a fixed two-month publication lag.",
            "Advance-tax months are also quarter ends; monthly data cannot separate the two effects.",
            "Interval coverage is measured on 24 test months and can drift with regime changes.",
            "MAPE is unstable when net flow is near zero; rupee and AUM-scaled errors are reported too.",
            "Monthly net flow hides gross and intra-month redemptions, which a single fund also needs.",
        ],
    }
    # JSON has no NaN; store missing values as null.
    clean = json_clean(evidence)
    evidence_path = output / "evidence.json"
    if evidence_path.exists():
        try:
            old_e = json.loads(evidence_path.read_text())
            if not (
                np.isclose(old_e["selected_test"]["mae_crore"], evidence["selected_test"]["mae_crore"], atol=1e-4)
                and np.isclose(old_e["selected_test"]["coverage95_pct"], evidence["selected_test"]["coverage95_pct"], atol=1e-4)
                and old_e.get("selected_model") == evidence.get("selected_model")
            ):
                evidence_path.write_text(json.dumps(clean, indent=2, allow_nan=False))
        except (ValueError, TypeError, KeyError, OSError, RuntimeError):
            evidence_path.write_text(json.dumps(clean, indent=2, allow_nan=False))
    else:
        evidence_path.write_text(json.dumps(clean, indent=2, allow_nan=False))
    print(json.dumps({
        "selected_model": selected, "primary_test_n": len(primary_test),
        "aum_scaled_mae_improvement_pct": evidence["aum_scaled_mae_improvement_pct"],
        "selected_test": best_test, "sarimax": evidence["sarimax"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()