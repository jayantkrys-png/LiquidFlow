from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ARTIFACTS = Path("artifacts")
needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "evidence.json").exists(), reason="Run the analysis first"
)


def _markdown(app) -> str:
    return "\n".join(block.value for block in app.markdown)


@needs_artifacts
def test_dashboard_renders_both_categories():
    from streamlit.testing.v1 import AppTest

    forecasts = pd.read_csv(ARTIFACTS / "forecasts.csv")
    app = AppTest.from_file("app.py").run(timeout=60)
    assert not app.exception
    assert app.radio[0].options == ["Liquid", "Overnight"]
    assert [tab.label for tab in app.tabs] == [
        "Forecast", "Backtest", "Tax-date pattern", "Models", "Method and data",
    ]
    for category in ["Liquid", "Overnight"]:
        app.radio[0].set_value(category).run(timeout=60)
        assert not app.exception
        peak = forecasts[forecasts.category == category].outflow95_buffer_crore.max()
        assert f"₹{peak:,.0f} crore" in _markdown(app)


@needs_artifacts
def test_dashboard_multi_month_horizon_slider():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py").run(timeout=60)
    assert not app.exception
    assert app.slider[0].value == 4
    # Test extending to 6 months (through March 2027)
    app.slider[0].set_value(6).run(timeout=60)
    assert not app.exception
    md = _markdown(app)
    # Peak buffer for Liquid in March 2027 is ₹195,924 crore
    assert "₹195,924 crore" in md
    # Test extending to 12 months
    app.slider[0].set_value(12).run(timeout=60)
    assert not app.exception


@needs_artifacts
def test_forecast_grid_and_buffers():
    forecasts = pd.read_csv(ARTIFACTS / "forecasts.csv")
    assert len(forecasts) == 8
    assert set(forecasts.category) == {"Liquid", "Overnight"}
    assert sorted(forecasts.horizon.unique()) == [1, 2, 3, 4]
    cols = ["point", "q025", "q050", "q100", "q900", "q975", "outflow95_buffer_crore"]
    assert not forecasts[cols].isna().any().any()
    assert (forecasts.q025 <= forecasts.q050).all()
    assert (forecasts.q050 <= forecasts.q100).all()
    assert (forecasts.q100 <= forecasts.point).all()
    assert (forecasts.point <= forecasts.q900).all()
    assert (forecasts.q900 <= forecasts.q975).all()
    expected = np.minimum(np.maximum(0, -forecasts.q050), forecasts.origin_aum)
    np.testing.assert_allclose(forecasts.outflow95_buffer_crore, expected)
    pct = forecasts.outflow95_buffer_crore / forecasts.origin_aum * 100
    np.testing.assert_allclose(forecasts.buffer_aum_pct, pct)


@needs_artifacts
def test_baseline_grid():
    baseline = pd.read_csv(ARTIFACTS / "baseline_forecasts.csv")
    assert len(baseline) == 8
    assert baseline.model.eq("seasonal_naive").all()
