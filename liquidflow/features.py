"""Direct multi-horizon feature rows.

Amounts are in INR crore. Labels are divided by the AUM known at the forecast
origin, and no feature uses data from after the origin month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ("Liquid", "Overnight")
METADATA = {
    "origin", "target_month", "category", "horizon", "origin_aum",
    "actual", "label_rate", "seasonal_naive", "seasonal_scaled",
    "mase_scale", "macro_cutoff",
}
CALENDAR = [
    "cal_sin1", "cal_cos1", "cal_sin2", "cal_cos2",
    "cal_tax_quarter", "cal_march", "cal_april",
]
MACRO_GROUPS = {
    "rates": ["macro_call_rate", "macro_call_change", "macro_call_age"],
    "equity": ["macro_nifty_change", "macro_nifty_change3", "macro_nifty_age"],
    "liquidity": ["macro_liquidity", "macro_liquidity_change", "macro_liquidity_age"],
}


def calendar_features(month: pd.Period) -> dict[str, float]:
    """Compute harmonic and quarter-end calendar features for a given month."""
    m = month.month
    return {
        "cal_sin1": float(np.sin(2 * np.pi * m / 12)),
        "cal_cos1": float(np.cos(2 * np.pi * m / 12)),
        "cal_sin2": float(np.sin(4 * np.pi * m / 12)),
        "cal_cos2": float(np.cos(4 * np.pi * m / 12)),
        # Advance-tax months are also quarter ends, so one flag covers both.
        "cal_tax_quarter": float(m in (3, 6, 9, 12)),
        "cal_march": float(m == 3),
        "cal_april": float(m == 4),
    }


def _last_macro(macro: pd.DataFrame, name: str, cutoff: pd.Period):
    good = macro.loc[(macro.index <= cutoff) & macro[name].notna(), name]
    if good.empty:
        return np.nan, np.nan
    last = good.index[-1]
    return float(good.iloc[-1]), float(cutoff.ordinal - last.ordinal)


def macro_features(
    macro: pd.DataFrame, origin: pd.Period, lag: int = 2
) -> dict[str, float]:
    """Compute lagged macro indicators and availability ages for an origin month."""
    cutoff = origin - lag
    call, call_age = _last_macro(macro, "call_rate_pct", cutoff)
    call_old, _ = _last_macro(macro, "call_rate_pct", cutoff - 3)
    equity, equity_age = _last_macro(macro, "nifty_average_change_pct", cutoff)
    equity3 = []
    for k in range(3):
        value, _ = _last_macro(macro, "nifty_average_change_pct", cutoff - k)
        equity3.append(value)
    liquidity, liquidity_age = _last_macro(macro, "net_rbi_operations_crore", cutoff)
    liquidity_old, _ = _last_macro(macro, "net_rbi_operations_crore", cutoff - 3)
    return {
        "macro_call_rate": call,
        "macro_call_change": call - call_old,
        "macro_call_age": call_age,
        "macro_nifty_change": equity,
        "macro_nifty_change3": float(np.mean(equity3)),
        "macro_nifty_age": equity_age,
        "macro_liquidity": liquidity / 100000,
        "macro_liquidity_change": (liquidity - liquidity_old) / 100000,
        "macro_liquidity_age": liquidity_age,
    }


def prepare_flows(flows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split, validate, and index flow records by fund category and period."""
    result = {}
    for category in CATEGORIES:
        data = flows.loc[flows.category == category].copy()
        data.index = pd.PeriodIndex(data.month, freq="M")
        data = data.sort_index()
        expected = pd.period_range(data.index.min(), data.index.max(), freq="M")
        if not data.index.equals(expected):
            raise ValueError(f"{category}: missing or duplicate months")
        if (data.aum_crore <= 0).any():
            raise ValueError("Nonpositive AUM cannot normalize a flow")
        result[category] = data
    if not result[CATEGORIES[0]].index.equals(result[CATEGORIES[1]].index):
        raise ValueError("Categories have different month coverage")
    return result


def make_feature_rows(
    flows: pd.DataFrame,
    macro: pd.DataFrame,
    horizons: tuple[int, ...] | list[int] = (1, 2, 3, 4),
    macro_lag: int = 2,
) -> pd.DataFrame:
    """Construct multi-horizon training and inference feature rows for all origins."""
    series = prepare_flows(flows)
    macro = macro.copy()
    macro.index = pd.PeriodIndex(macro.month, freq="M")
    macro = macro.sort_index()
    rows = []
    for category, data in series.items():
        other = series["Overnight" if category == "Liquid" else "Liquid"]
        for pos in range(13, len(data)):
            origin = data.index[pos]
            aum = float(data.iloc[pos].aum_crore)
            past = data.iloc[:pos + 1]
            macro_row = macro_features(macro, origin, macro_lag)
            mase_scale = float(past.net_flow_crore.diff(12).abs().dropna().mean())
            for h in horizons:
                target = origin + h
                seasonal_month = target - 12
                seasonal_flow = float(data.loc[seasonal_month, "net_flow_crore"])
                seasonal_aum = float(data.loc[seasonal_month - 1, "aum_crore"])
                actual = float(data.loc[target, "net_flow_crore"]) if target in data.index else np.nan
                calendar = calendar_features(target)
                row = {
                    "origin": str(origin), "target_month": str(target),
                    "category": category, "horizon": h, "origin_aum": aum,
                    "actual": actual, "label_rate": actual / aum,
                    "seasonal_naive": seasonal_flow,
                    "seasonal_scaled": seasonal_flow / seasonal_aum * aum,
                    "mase_scale": mase_scale, "macro_cutoff": str(origin - macro_lag),
                    "is_overnight": float(category == "Overnight"),
                    "horizon_number": float(h),
                    "lag0": float(data.iloc[pos].net_flow_crore / aum),
                    "lag1": float(data.iloc[pos - 1].net_flow_crore / aum),
                    "lag2": float(data.iloc[pos - 2].net_flow_crore / aum),
                    "lag3": float(data.iloc[pos - 3].net_flow_crore / aum),
                    "lag4": float(data.iloc[pos - 4].net_flow_crore / aum),
                    "lag5": float(data.iloc[pos - 5].net_flow_crore / aum),
                    "rolling_mean3": float(past.net_flow_crore.tail(3).mean() / aum),
                    "rolling_std6": float(past.net_flow_crore.tail(6).std(ddof=0) / aum),
                    "seasonal_flow_rate": seasonal_flow / seasonal_aum,
                    "aum_growth3": float(aum / data.iloc[pos - 3].aum_crore - 1),
                    "other_category_flow_rate": float(
                        other.loc[origin, "net_flow_crore"] / other.loc[origin, "aum_crore"]
                    ),
                    **calendar, **macro_row,
                }
                # Linear models get explicit category/season interactions.
                for name, value in calendar.items():
                    row[f"{name}_overnight"] = value * row["is_overnight"]
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["origin", "category", "horizon"]).reset_index(drop=True)


def feature_columns(
    frame: pd.DataFrame, scope: str = "full", exclude_group: str | None = None
) -> list[str]:
    """Return selected model feature column names based on scope and ablation group."""
    names = [n for n in frame.columns if n not in METADATA]
    if scope == "core":
        names = [n for n in names if not n.startswith("macro_")]
    if exclude_group == "calendar":
        names = [n for n in names if not n.startswith("cal_")]
    elif exclude_group in MACRO_GROUPS:
        names = [n for n in names if n not in MACRO_GROUPS[exclude_group]]
    return names