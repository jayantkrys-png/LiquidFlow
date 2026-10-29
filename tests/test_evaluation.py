import numpy as np
import pandas as pd
import pytest

from liquidflow.evaluation import QCOLS, calibrate, cash_cost, choose_model


def test_calibration_waits_for_target_to_be_known():
    rows = []
    for m in pd.period_range("2021-01", "2022-12", freq="M"):
        rows.append({
            "origin": str(m), "target_month": str(m + 3), "horizon": 3,
            "model": "test", "category": "Liquid", "raw_point": 0,
            "actual": m.ordinal % 17, "origin_aum": 100,
        })
    raw = pd.DataFrame(rows)
    p = {"intervals": {"lookback_errors_per_category_horizon": 24},
         "development_target_start": "2022-01", "development_target_end": "2022-06",
         "test_target_start": "2022-07", "test_target_end": "2022-12"}
    before = calibrate(raw, p)
    raw.loc[raw.target_month > "2022-03", "actual"] = 1e9
    after = calibrate(raw, p)
    columns = list(QCOLS.values())
    a = before.loc[before.origin == "2022-03", columns].to_numpy(float)
    b = after.loc[after.origin == "2022-03", columns].to_numpy(float)
    np.testing.assert_array_equal(a, b)
    assert (before.calibration_target_max <= before.origin).all()
    assert (np.diff(before[columns].to_numpy(float), axis=1) >= 0).all()


def test_selector_ignores_final_test_winner():
    scores = pd.DataFrame([
        {"split": "development", "model": "a", "aum_scaled_mae_pct": 1},
        {"split": "development", "model": "b", "aum_scaled_mae_pct": 2},
        {"split": "test", "model": "a", "aum_scaled_mae_pct": 100},
        {"split": "test", "model": "b", "aum_scaled_mae_pct": 0},
    ])
    assert choose_model(scores, {"candidates": ["a", "b"]})[0] == "a"


def test_literal_cash_cost_units():
    cost, _drag, shortfall = cash_cost(100, 80, 30, 0.005, 3)
    assert cost == pytest.approx(20 * .005 * 30 / 365)
    cost, _drag, shortfall = cash_cost(50, 80, 30, .005, 3)
    assert cost == 90
    assert shortfall == 30
    with pytest.raises(ValueError):
        cash_cost(-1, 1, 30)