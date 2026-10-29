import pandas as pd
import pytest

from liquidflow.data import identify_header, normalized_text, numeric


def test_numeric_and_whitespace():
    assert numeric("(1,234.50)") == -1234.5
    assert normalized_text("  Overnight\u00a0 Fund  ") == "overnight fund"


def test_header_mapping_excludes_average_and_segregated_aum():
    frame = pd.DataFrame([
        ["Scheme Name", "Funds Mobilized (INR in crore)", "Repurchase/Redemption",
         "Net Inflow (+ve)/Outflow (-ve)", "Net Assets Under Management",
         "Average Net Assets Under Management", "Net Assets Under segregated"],
        ["Liquid Fund", 100, 80, 20, 1000, 900, 0],
    ])
    found = identify_header(frame)
    assert found["subscriptions_crore"] == 1
    assert found["redemptions_crore"] == 2
    assert found["net_flow_crore"] == 3
    assert found["aum_crore"] == 4


def test_non_numeric_is_not_silently_zero():
    with pytest.raises(ValueError):
        numeric("not published")


def test_older_aum_heading():
    frame = pd.DataFrame([
        ["Scheme Name", "Funds Mobilized", "Repurchase/Redemption",
         "Net Inflow (+ve)/Outflow (-ve)", "AUM as on 31-Jan-2020",
         "Average Net Assets Under Management"],
        ["Liquid Fund", 100, 80, 20, 1000, 900],
    ])
    assert identify_header(frame)["aum_crore"] == 4