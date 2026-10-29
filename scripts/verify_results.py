"""Recompute headline numbers from the exported CSVs, without the metric code."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify_results(
    root: Path | str = "artifacts",
    flows_path: Path | str = "data/processed/amfi_flows.csv",
    write_output: bool = True,
) -> dict:
    """Verify headline evaluation numbers and data integrity checks."""
    root = Path(root)
    e = json.loads((root / "evidence.json").read_text())
    p = pd.read_csv(root / "predictions.csv")
    primary = p[
        (p.model == e["selection"]["selected_model"])
        & (p.split == "test")
        & p.horizon.isin([1, 2, 3])
    ]
    assert len(primary) == 144
    assert primary.groupby(["category", "horizon"]).size().eq(24).all()
    y, forecast = primary.actual.to_numpy(), primary.point.to_numpy()
    mae = sum(abs(float(a) - float(b)) for a, b in zip(y, forecast, strict=False)) / len(y)
    assert np.isclose(mae, e["selected_test"]["mae_crore"])
    coverage = (
        sum(float(lo) <= float(a) <= float(hi) for a, lo, hi in zip(y, primary.q025, primary.q975, strict=False))
        / len(y)
        * 100
    )
    assert np.isclose(coverage, e["selected_test"]["coverage95_pct"])
    assert (p.training_label_max <= p.origin).all()
    calibrated = p[p.calibration_n > 0]
    assert (calibrated.calibration_target_max <= calibrated.origin).all()
    live = pd.read_csv(root / "forecasts.csv")
    assert len(live) == 8 and live.groupby("category").size().eq(4).all()
    assert live.origin.nunique() == 1
    assert live.target_month.min() == "2026-10" and live.target_month.max() == "2027-01"
    assert (live.q025 <= live.q100).all() and (live.q100 <= live.point).all()
    assert (live.point <= live.q900).all() and (live.q900 <= live.q975).all()
    policy = pd.read_csv(root / "policy_months.csv")
    assert len(policy) == 48 and policy.horizon.eq(1).all()
    g = e["protocol"]["policy"]["annual_yield_gap"]
    penalty = e["protocol"]["policy"]["shortfall_penalty_multiplier"]
    computed_cost = 0.0
    for row in policy.itertuples():
        shortfall = max(float(row.outflow_crore - row.q95_buffer_crore), 0)
        idle = max(float(row.q95_buffer_crore - row.outflow_crore), 0)
        computed_cost += g * float(row.days) / 365 * idle + penalty * shortfall
    recorded_cost = sum(r["cost_crore"] for r in e["policy"] if r["policy"] == "fixed_q95")
    assert np.isclose(computed_cost, recorded_cost)
    data = pd.read_csv(flows_path)
    assert abs(data.subscriptions_crore - data.redemptions_crore - data.net_flow_crore).max() < 0.01
    assert data.groupby("category").size().eq(87).all()
    result = {
        "status": "PASS",
        "independent_mae_crore": mae,
        "independent_coverage95_pct": coverage,
        "independent_policy_cost_crore": computed_cost,
        "checks": [
            "24 target release months per category and primary horizon",
            "independent MAE and 95% coverage arithmetic",
            "training and calibration target cutoffs",
            "complete latest-origin live forecast grid and quantile order",
            "one-step non-overlapping policy costs",
            "source flow accounting identity and monthly counts",
        ],
    }
    if write_output:
        (root / "verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    res = verify_results()
    print(json.dumps(res, indent=2))
