import numpy as np
import pandas as pd
import pytest

from liquidflow.features import (
    calendar_features,
    feature_columns,
    make_feature_rows,
    prepare_flows,
)


def fixture_data():
    months = pd.period_range("2019-07", "2023-12", freq="M").astype(str)
    flows = pd.DataFrame([
        {"month": m, "category": c, "net_flow_crore": (i % 7 - 3) * 100,
         "aum_crore": 10000 + 10 * i}
        for c in ["Liquid", "Overnight"] for i, m in enumerate(months)
    ])
    macro = pd.DataFrame({
        "month": months, "call_rate_pct": np.arange(len(months)) / 20 + 3,
        "nifty_average_change_pct": np.sin(np.arange(len(months))),
        "net_rbi_operations_crore": np.arange(len(months)) * 100,
    })
    return flows, macro


def test_future_data_cannot_change_origin_features():
    flows, macro = fixture_data()
    original = make_feature_rows(flows, macro)
    flows.loc[flows.month > "2022-06", ["net_flow_crore", "aum_crore"]] *= 99
    macro.loc[macro.month > "2022-04", ["call_rate_pct", "net_rbi_operations_crore"]] *= 99
    changed = make_feature_rows(flows, macro)
    names = feature_columns(original)
    a = original.loc[original.origin == "2022-06", names].reset_index(drop=True)
    b = changed.loc[changed.origin == "2022-06", names].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_origin_denominator_and_seasonal_date():
    flows, macro = fixture_data()
    frame = make_feature_rows(flows, macro)
    row = frame[(frame.origin == "2022-06") & (frame.horizon == 3) & (frame.category == "Liquid")].iloc[0]
    assert row.target_month == "2022-09"
    assert row.macro_cutoff == "2022-04"
    expected = flows[(flows.month == "2022-09") & (flows.category == "Liquid")].iloc[0].net_flow_crore
    assert row.label_rate == expected / row.origin_aum
    assert "actual" not in feature_columns(frame)
    assert "origin_aum" not in feature_columns(frame)


def test_missing_month_fails_instead_of_zero_imputation():
    flows, _ = fixture_data()
    with pytest.raises(ValueError, match="missing"):
        prepare_flows(flows.drop(index=10))


def test_quarter_tax_calendar_is_combined_not_causal():
    assert calendar_features(pd.Period("2027-03"))["cal_tax_quarter"] == 1
    assert calendar_features(pd.Period("2027-03"))["cal_march"] == 1
    assert calendar_features(pd.Period("2027-04"))["cal_tax_quarter"] == 0