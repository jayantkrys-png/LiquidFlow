
import pandas as pd

from liquidflow.macro import financial_year_matrix, parse_liquidity


def test_fiscal_year_conversion_and_missing(tmp_path):
    rows = [["2024-25", *range(1, 13), 9], ["2025-26", 5, "-", *["-"] * 10, "-"]]
    # Extra header lines make this a realistic table shape.
    rows = [["Year", *["month"] * 12, "annual"]] * 4 + rows
    path = tmp_path / "table.html"
    path.write_text(pd.DataFrame(rows).to_html(index=False, header=False))
    result = financial_year_matrix(path).set_index("month")["value"]
    assert result["2024-04"] == 1
    assert result["2025-03"] == 12
    assert result["2025-04"] == 5
    assert "2025-05" not in result


def test_signed_liquidity_not_zero_filled(tmp_path):
    rows = [
        ["Date", "repo", "reverse", "MSF", "SDF", "OMO", "Net injection"],
        ["Jul. 1, 2026", "-", "-", "-", "-", "-", "-100"],
        ["Jul. 2, 2026", "-", "-", "-", "-", "-", "-"],
    ]
    path = tmp_path / "liq.html"
    path.write_text(pd.DataFrame(rows).to_html(index=False, header=False))
    data = parse_liquidity(path, "https://example.org/source")
    assert len(data) == 1
    assert data[0]["month"] == "2026-07"
    assert data[0]["net_rbi_operations_crore"] == -100