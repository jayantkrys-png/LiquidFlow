"""LiquidFlow dashboard. Reads the saved experiment outputs; makes no network calls.

Run with: streamlit run app.py
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from liquidflow.features import calendar_features

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"
FONTS = ROOT / "assets" / "fonts"
REPORT = ROOT / "report" / "LiquidFlow_Team_Angry_Bird_PS03.pdf"

NAVY, BLUE, TEXT, MUTED = "#142C46", "#1769B5", "#243B53", "#52677D"
ICE, LINE, GRID = "#EEF5FC", "#D9E4EF", "#E2E9F1"
ORANGE, ORANGE_DARK, BAND80, BAND95, GREY = "#D9822B", "#A8541A", "#A9C8EA", "#DCE8F6", "#C9D3DF"
SANS, MONO = "Inter", "JetBrains Mono"

MODEL_NAMES = {
    "seasonal_naive": "Seasonal naive",
    "seasonal_scaled": "Scaled seasonal",
    "sarimax": "SARIMAX",
    "ridge_core": "Ridge",
    "ridge_full": "Ridge + macro",
    "catboost_core": "CatBoost",
    "catboost_full": "CatBoost + macro",
}
POLICY_NAMES = {
    "prior_year_outflow": "Last year's same-month outflow",
    "selected_model_q95": "LiquidFlow buffer",
    "seasonal_naive_q95": "Seasonal 95% band",
    "all_origin_aum_reference": "Hold all assets in cash",
}

CALIBRATION_OFFSETS = {
    "Liquid": {
        "1": [-0.08745428, -0.07843416, -0.07283115, -0.00837086, 0.14951411, 0.15569136, 0.15664161],
        "2": [-0.12859254, -0.11357683, -0.10599690, 0.00000652, 0.12921183, 0.13280420, 0.13978755],
        "3": [-0.13287738, -0.12593796, -0.10578718, -0.00627689, 0.12982292, 0.13245269, 0.13348339],
        "4": [-0.12392885, -0.11052659, -0.09936028, -0.00271373, 0.15111181, 0.15549593, 0.17194438],
        "5": [-0.13207957, -0.11386203, -0.10075262, -0.00260673, 0.12273614, 0.12473068, 0.12983201],
        "6": [-0.14231364, -0.13735940, -0.11433376, -0.00892747, 0.12552760, 0.13488888, 0.14521533],
        "7": [-0.12333944, -0.10841778, -0.10280179, -0.00362246, 0.16970247, 0.17517111, 0.18072922],
        "8": [-0.13291418, -0.11647676, -0.11153440, -0.00158505, 0.12750201, 0.13261955, 0.14170611],
        "9": [-0.14822868, -0.13667778, -0.10808113, -0.00624822, 0.13750793, 0.14754229, 0.14987172],
        "10": [-0.12429553, -0.10889494, -0.10646609, 0.00465074, 0.16958998, 0.18668002, 0.19286291],
        "11": [-0.14051914, -0.12507625, -0.11059076, 0.00155364, 0.13368796, 0.15472262, 0.15775262],
        "12": [-0.17426699, -0.14702579, -0.12549834, -0.00575977, 0.13880818, 0.15213424, 0.16268534],
        "13": [-0.13117068, -0.11676901, -0.11303946, 0.01194025, 0.17824168, 0.20317498, 0.20930387],
        "14": [-0.18006663, -0.17122126, -0.12526849, 0.00661595, 0.13571451, 0.15576670, 0.17181874],
        "15": [-0.18067922, -0.14397811, -0.12734744, -0.00615396, 0.13803842, 0.14303914, 0.16303481],
        "16": [-0.17212171, -0.14824641, -0.11390409, 0.00909647, 0.18817362, 0.20154474, 0.20570275],
        "17": [-0.18207864, -0.16352868, -0.12474029, 0.00305714, 0.14695526, 0.15017269, 0.15979506],
        "18": [-0.18821795, -0.15633092, -0.14039578, -0.00462112, 0.15205048, 0.15900809, 0.16097467],
        "19": [-0.16612942, -0.15519999, -0.11767643, 0.00646153, 0.19863347, 0.20295633, 0.20637991],
        "20": [-0.16229017, -0.15313150, -0.12591126, 0.00415834, 0.15937998, 0.16603773, 0.17021388],
        "21": [-0.17506158, -0.16014548, -0.14269011, -0.00657612, 0.16374696, 0.17012172, 0.18224091],
        "22": [-0.15764064, -0.14109574, -0.12032748, -0.00017509, 0.19832818, 0.22420859, 0.23439344],
        "23": [-0.18033936, -0.15159718, -0.13273074, 0.00197814, 0.16807887, 0.17077328, 0.18513034],
        "24": [-0.24125629, -0.20298115, -0.14104009, -0.00615856, 0.17613023, 0.19012166, 0.19404101],
    },
    "Overnight": {
        "1": [-0.39278428, -0.37884709, -0.34699149, -0.02460152, 0.24093362, 0.43527480, 0.48348235],
        "2": [-0.48412344, -0.44228710, -0.35209854, -0.02361128, 0.20281457, 0.38640890, 0.46740549],
        "3": [-0.50931054, -0.47774703, -0.43959134, -0.04530790, 0.26045613, 0.31495109, 0.33332387],
        "4": [-0.52352731, -0.46730619, -0.36223504, -0.01169110, 0.29061048, 0.40807152, 0.46397963],
        "5": [-0.55250622, -0.47503601, -0.37700725, -0.00749106, 0.21121926, 0.30511774, 0.37707559],
        "6": [-0.53115972, -0.48162344, -0.41609122, -0.03127979, 0.22628672, 0.27731069, 0.37078736],
        "7": [-0.48629282, -0.43762491, -0.37336716, -0.03337278, 0.33129717, 0.46185462, 0.51725078],
        "8": [-0.57671563, -0.47594773, -0.46646371, -0.01471008, 0.27023198, 0.46143212, 0.49437872],
        "9": [-0.52862801, -0.51699625, -0.48940916, -0.03130548, 0.22456101, 0.30578587, 0.37494482],
        "10": [-0.49693371, -0.47513616, -0.40128199, -0.02821089, 0.26689209, 0.39035268, 0.51673140],
        "11": [-0.51505145, -0.43689127, -0.40682002, -0.04376522, 0.24250211, 0.39744866, 0.42761165],
        "12": [-0.57406165, -0.53310544, -0.48111631, -0.05098961, 0.22493650, 0.38644591, 0.43518205],
        "13": [-0.52860955, -0.51588978, -0.46868535, -0.04307280, 0.32689493, 0.51171104, 0.54533598],
        "14": [-0.56206301, -0.48323797, -0.46309928, -0.04842219, 0.24602690, 0.37860885, 0.43648228],
        "15": [-0.55856782, -0.53035017, -0.47484702, -0.06853658, 0.25164480, 0.38864712, 0.42398495],
        "16": [-0.53794735, -0.46638291, -0.44715660, -0.05167555, 0.28310238, 0.55414462, 0.61029420],
        "17": [-0.60086219, -0.53579800, -0.47273230, -0.05163542, 0.21458033, 0.38529968, 0.43293377],
        "18": [-0.56898589, -0.50456352, -0.43501756, -0.05260612, 0.21650500, 0.37893787, 0.48053419],
        "19": [-0.52701200, -0.50400794, -0.43071433, -0.05559134, 0.32055384, 0.49269923, 0.56870525],
        "20": [-0.65498869, -0.59443394, -0.46817142, -0.05449533, 0.26354484, 0.38472581, 0.41894753],
        "21": [-0.65332795, -0.61502011, -0.40776724, -0.06329108, 0.27153938, 0.39510908, 0.44257958],
        "22": [-0.50950258, -0.48774302, -0.43085905, -0.05507441, 0.30602187, 0.54059126, 0.62155176],
        "23": [-0.55593632, -0.52724328, -0.46258709, -0.06429269, 0.22442225, 0.41861213, 0.51845659],
        "24": [-0.62539105, -0.54271232, -0.40296772, -0.06056245, 0.23038872, 0.42298712, 0.50492989],
    },
}

st.set_page_config(page_title="LiquidFlow", layout="wide")


def font_face(family: str, file: str, weight: int) -> str:
    data = base64.b64encode((FONTS / file).read_bytes()).decode()
    return f"@font-face{{font-family:'{family}';src:url(data:font/ttf;base64,{data}) format('truetype');font-weight:{weight};font-display:swap;}}"


@st.cache_resource
def stylesheet() -> str:
    faces = "".join([
        font_face(SANS, "Inter-Regular.ttf", 400),
        font_face(SANS, "Inter-Medium.ttf", 500),
        font_face(SANS, "Inter-SemiBold.ttf", 600),
        font_face(SANS, "Inter-Bold.ttf", 700),
        font_face("Inter Display", "InterDisplay-ExtraBold.ttf", 800),
        font_face(MONO, "JetBrainsMono-Regular.ttf", 400),
        font_face(MONO, "JetBrainsMono-Bold.ttf", 700),
    ])
    return f"""<style>{faces}
html, body, [data-testid="stApp"], [data-testid="stMarkdownContainer"], button, input, label, p, li {{
  font-family: '{SANS}', system-ui, sans-serif; color: {TEXT};
}}
.block-container {{ max-width: 1160px; padding-top: 3.2rem; padding-bottom: 4rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4, [data-testid="stHeading"] * {{ font-family: '{SANS}', sans-serif !important; color: {NAVY}; letter-spacing: -0.02em; }}
h3 {{ font-weight: 700 !important; font-size: 1.35rem !important; }}
.lf-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 0.3rem 0 0.4rem 0; }}
.lf-table th, .lf-table td {{ border-left: none !important; border-right: none !important; border-top: none !important; }}
.lf-table th {{ font-weight: 600; color: {MUTED}; text-align: left; padding: 8px 10px; border-bottom: 1.5px solid {NAVY}; white-space: nowrap; }}
.lf-table td {{ padding: 8px 10px; border-bottom: 1px solid {LINE}; color: {TEXT}; }}
.lf-table .n {{ text-align: right; font-family: '{MONO}', monospace; font-size: 0.85rem; white-space: nowrap; }}
.lf-table th.n {{ font-family: '{SANS}', sans-serif; font-size: 0.92rem; }}
.lf-table tr.hl td {{ background: #FDF1E6; }}
.lf-table tr.sel td {{ background: {ICE}; font-weight: 600; }}
.lf-scroll {{ overflow-x: auto; }}
.lf-kicker {{ font-family: '{MONO}', monospace; font-size: 0.78rem; letter-spacing: 0.06em; color: {BLUE};
  text-transform: uppercase; margin-bottom: 0.2rem; }}
.lf-title {{ font-family: 'Inter Display', '{SANS}', sans-serif; font-weight: 800; font-size: 2.9rem; line-height: 1.05;
  color: {NAVY}; letter-spacing: -0.03em; margin: 0.1rem 0 0.7rem 0; }}
.lf-lead {{ font-size: 1.12rem; line-height: 1.55; color: {NAVY}; max-width: 780px; }}
.lf-meta {{ font-size: 0.85rem; color: {MUTED}; margin-top: 0.3rem; }}
.lf-card {{ background: {ICE}; border-radius: 10px; padding: 16px 18px; height: 100%; margin-bottom: 8px; }}
.lf-card .v {{ font-family: '{MONO}', monospace; font-weight: 700; font-size: 1.65rem; color: {BLUE}; line-height: 1.15; }}
.lf-card .v.warn {{ color: {ORANGE_DARK}; }}
.lf-card .l {{ font-size: 0.86rem; color: {TEXT}; margin-top: 6px; line-height: 1.4; }}
.lf-note {{ background: {ICE}; border-left: 3px solid {BLUE}; border-radius: 6px; padding: 12px 16px; font-size: 0.95rem; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
[data-testid="stTab"] p {{ font-weight: 600; }}
code {{ font-family: '{MONO}', monospace; color: {NAVY} !important; background: {ICE} !important; }}
@media (max-width: 640px) {{ .lf-title {{ font-size: 2.1rem; }} .lf-card .v {{ font-size: 1.35rem; }} }}
</style>"""


def theme() -> dict:
    axis = {
        "labelFont": MONO, "labelFontSize": 11, "labelColor": MUTED, "titleFont": SANS,
        "titleFontSize": 11, "titleFontWeight": 400, "titleColor": MUTED,
        "gridColor": GRID, "domain": False, "ticks": False,
    }
    return {
        "config": {
            "font": SANS, "background": "white",
            "view": {"stroke": None},
            "axis": axis, "axisX": {"grid": False, "labelFont": SANS, "labelPadding": 8},
            "legend": {"labelFont": SANS, "labelFontSize": 12, "labelColor": TEXT, "titleFont": SANS,
                       "orient": "top", "title": None, "symbolSize": 120, "labelLimit": 0},
            "title": {"font": SANS, "fontSize": 14, "fontWeight": 700, "color": NAVY, "anchor": "start", "offset": 10},
            "text": {"font": MONO},
        }
    }


alt.theme.register("liquidflow", enable=True)(theme)


def snapshot_signature() -> tuple:
    names = ["evidence.json", "forecasts.csv", "policy_months.csv", "metrics_overall.csv"]
    return tuple((ART / n).stat().st_mtime_ns for n in names)


@st.cache_data
def load(signature: tuple) -> dict:
    flows = pd.read_csv(ROOT / "data" / "processed" / "amfi_flows.csv")
    flows["date"] = pd.to_datetime(flows.month)
    flows = flows.sort_values(["category", "date"])
    flows["pct_prev_aum"] = flows.net_flow_crore / flows.groupby("category").aum_crore.shift(1) * 100
    forecasts = pd.read_csv(ART / "forecasts.csv")
    forecasts["date"] = pd.to_datetime(forecasts.target_month)
    policy = pd.read_csv(ART / "policy_months.csv")
    policy["date"] = pd.to_datetime(policy.target_month)
    return {
        "evidence": json.loads((ART / "evidence.json").read_text()),
        "flows": flows,
        "forecasts": forecasts,
        "baseline": pd.read_csv(ART / "baseline_forecasts.csv"),
        "policy": policy,
        "costs": pd.read_csv(ART / "policy_sanity_checks.csv"),
        "metrics": pd.read_csv(ART / "metrics_overall.csv"),
        "metrics_by_category": pd.read_csv(ART / "metrics_by_category_horizon.csv"),
        "ablations": pd.read_csv(ART / "driver_ablations.csv"),
    }


@st.cache_data
def get_extended_forecast(category: str, n_months: int, base_fc: pd.DataFrame) -> pd.DataFrame:
    cat_base = base_fc[base_fc.category == category].sort_values("horizon")
    if n_months <= 4:
        return cat_base.iloc[:n_months].copy()

    aum = float(cat_base.origin_aum.iloc[0])
    origin = cat_base.origin.iloc[0]

    model = SARIMAXResults.load(ART / "models" / f"sarimax_{category.lower()}.pkl")
    months = pd.period_range(pd.Period(origin) + 1, periods=n_months, freq="M")
    cal = pd.DataFrame([calendar_features(m) for m in months], index=months)
    raw_points = model.get_forecast(n_months, exog=cal).predicted_mean.to_numpy() * aum

    cat_offsets = CALIBRATION_OFFSETS.get(category, {})
    rows = []
    for i in range(min(4, n_months)):
        rows.append(cat_base.iloc[i].to_dict())

    for h in range(5, n_months + 1):
        target = str(months[h - 1])
        raw_pt = float(raw_points[h - 1])
        offs = cat_offsets.get(str(h), cat_offsets.get(str(min(h, 24)), [0.0] * 7))
        q_vals = [raw_pt + off * aum for off in offs]
        row = {
            "origin": origin,
            "target_month": target,
            "category": category,
            "horizon": h,
            "origin_aum": aum,
            "actual": np.nan,
            "mase_scale": cat_base.mase_scale.iloc[0],
            "seasonal_naive": np.nan,
            "model": "sarimax",
            "raw_point": raw_pt,
            "training_rows": cat_base.training_rows.iloc[0],
            "training_label_max": cat_base.training_label_max.iloc[0],
            "q025": q_vals[0],
            "q050": q_vals[1],
            "q100": q_vals[2],
            "q500": q_vals[3],
            "q900": q_vals[4],
            "q950": q_vals[5],
            "q975": q_vals[6],
            "point": q_vals[3],
            "calibration_n": 24,
            "calibration_target_max": "2026-09",
            "split": "live",
            "outflow95_buffer_crore": min(max(0.0, -q_vals[1]), aum),
            "outflow95_uncapped_crore": max(0.0, -q_vals[1]),
            "buffer_aum_pct": min(max(0.0, -q_vals[1]), aum) / aum * 100,
            "as_of_date": cat_base.as_of_date.iloc[0],
            "forecast_status": "nowcast_pending_AMFI_release" if target <= cat_base.as_of_date.iloc[0][:7] else "future_calendar_month",
            "date": pd.to_datetime(target),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def table(frame: pd.DataFrame, text_columns: int = 1, row_class: dict | None = None) -> None:
    """Render a small table as HTML so it picks up the page fonts. Columns after
    the first `text_columns` are right-aligned and set in the mono face."""
    row_class = row_class or {}
    head = "".join(
        f'<th class="{"n" if i >= text_columns else ""}">{html.escape(str(c))}</th>' for i, c in enumerate(frame.columns)
    )
    body = []
    for r, values in enumerate(frame.itertuples(index=False)):
        cells = "".join(
            f'<td class="{"n" if i >= text_columns else ""}">{html.escape(str(v))}</td>' for i, v in enumerate(values)
        )
        body.append(f'<tr class="{row_class.get(r, "")}">{cells}</tr>')
    st.markdown(
        f'<div class="lf-scroll"><table class="lf-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def crore(x: float) -> str:
    return f"{x:,.0f}".replace("-", "−")


def month_label(date: pd.Timestamp) -> str:
    return date.strftime("%b %Y")


def card(column, value: str, label: str, warn: bool = False) -> None:
    column.markdown(
        f'<div class="lf-card"><div class="v{" warn" if warn else ""}">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def forecast_chart(history: pd.DataFrame, fc: pd.DataFrame) -> alt.LayerChart:
    hist = history.tail(24).assign(kind=lambda d: d.net_flow_crore.lt(0).map({True: "Outflow", False: "Inflow"}))
    fc = fc.assign(
        label=fc.date.map(month_label),
        buffer_level=-fc.outflow95_buffer_crore,
        central=fc.point.map(crore),
        band80=fc.q100.map(crore) + " to " + fc.q900.map(crore),
        band95=fc.q025.map(crore) + " to " + fc.q975.map(crore),
        buffer=fc.outflow95_buffer_crore.map(crore),
    )
    domain = [(hist.date.min() - pd.Timedelta(days=20)).isoformat(), (fc.date.max() + pd.Timedelta(days=25)).isoformat()]
    x = alt.X("date:T", title=None, scale=alt.Scale(domain=domain), axis=alt.Axis(format="%b %y", labelAngle=0, tickCount=10))
    y_title = "₹ crore, net flow per month"
    tips = [
        alt.Tooltip("label:N", title="Month"), alt.Tooltip("central:N", title="Central (₹ cr)"),
        alt.Tooltip("band80:N", title="80% range"), alt.Tooltip("band95:N", title="95% range"),
        alt.Tooltip("buffer:N", title="Buffer (₹ cr)"),
    ]
    line = alt.Chart(hist).mark_line(color=NAVY, strokeWidth=1.6).encode(
        x=x, y=alt.Y("net_flow_crore:Q", title=y_title, axis=alt.Axis(format="~s"))
    )
    dots = alt.Chart(hist).mark_circle(size=34, opacity=1).encode(
        x="date:T", y="net_flow_crore:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Inflow", "Outflow"], range=[NAVY, ORANGE]), legend=None),
        tooltip=[alt.Tooltip("yearmonth(date):T", title="Month"), alt.Tooltip("net_flow_crore:Q", title="Net flow (₹ cr)", format=",.0f")],
    )
    base = alt.Chart(fc)
    band95 = base.mark_bar(width=22, color=BAND95).encode(x="date:T", y="q025:Q", y2="q975:Q", tooltip=tips)
    band80 = base.mark_bar(width=22, color=BAND80).encode(x="date:T", y="q100:Q", y2="q900:Q", tooltip=tips)
    point = base.mark_point(shape="circle", size=70, filled=True, color="white", stroke=BLUE, strokeWidth=2.5).encode(
        x="date:T", y="point:Q", tooltip=tips)
    buffer = base.mark_tick(color=ORANGE_DARK, thickness=2.5, size=26).encode(x="date:T", y="buffer_level:Q", tooltip=tips)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=NAVY, strokeWidth=0.8).encode(y="y:Q")
    split = alt.Chart(pd.DataFrame({"date": [history.date.max() + pd.Timedelta(days=15)]})).mark_rule(
        color=MUTED, strokeDash=[3, 3], strokeWidth=0.8).encode(x="date:T")
    return (zero + split + band95 + band80 + line + dots + buffer + point).properties(height=360)


def backtest_chart(rows: pd.DataFrame) -> alt.LayerChart:
    rows = rows.assign(
        short=rows.outflow_crore > rows.q95_buffer_crore,
        status=lambda d: d.short.map({True: "Outflow larger than the buffer", False: "Actual net outflow"}),
        label=rows.date.map(month_label),
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=0, tickCount=8))
    tips = [
        alt.Tooltip("label:N", title="Month"),
        alt.Tooltip("outflow_crore:Q", title="Actual outflow (₹ cr)", format=",.0f"),
        alt.Tooltip("q95_buffer_crore:Q", title="LiquidFlow buffer (₹ cr)", format=",.0f"),
        alt.Tooltip("naive_buffer_crore:Q", title="Last year's rule (₹ cr)", format=",.0f"),
    ]
    bars = alt.Chart(rows).mark_bar(width=12).encode(
        x=x, y=alt.Y("outflow_crore:Q", title="₹ crore", axis=alt.Axis(format="~s")),
        color=alt.Color("status:N", scale=alt.Scale(
            domain=["Actual net outflow", "Outflow larger than the buffer"], range=[GREY, ORANGE])),
        tooltip=tips,
    )
    lines = alt.Chart(rows).transform_fold(
        ["q95_buffer_crore", "naive_buffer_crore"], as_=["rule", "value"]
    ).transform_calculate(
        rule="datum.rule == 'q95_buffer_crore' ? 'LiquidFlow buffer' : 'Last year\\'s outflow rule'"
    ).mark_line(interpolate="step", strokeWidth=1.8).encode(
        x="date:T", y="value:Q",
        stroke=alt.Stroke("rule:N", scale=alt.Scale(domain=["LiquidFlow buffer", "Last year's outflow rule"], range=[BLUE, NAVY]), title=None),
        strokeDash=alt.StrokeDash("rule:N", scale=alt.Scale(domain=["LiquidFlow buffer", "Last year's outflow rule"], range=[[1, 0], [3, 2]]), legend=None),
    )
    return alt.layer(bars, lines).resolve_scale(color="independent", stroke="independent").properties(height=320)


def heatmap(flows: pd.DataFrame) -> alt.LayerChart:
    d = flows.dropna(subset=["pct_prev_aum"]).assign(
        year=lambda f: f.date.dt.year.astype(str), month=lambda f: f.date.dt.strftime("%b"),
        value=lambda f: f.pct_prev_aum.round(1),
    )
    d["label"] = d.value.round(0).astype(int).map(lambda v: f"{v:+d}".replace("-", "−") if v else "0")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    base = alt.Chart(d).encode(
        x=alt.X("month:O", sort=months, title=None, axis=alt.Axis(orient="top", labelFont=SANS, labelAngle=0)),
        y=alt.Y("year:O", title=None),
    )
    rect = base.mark_rect(stroke="white", strokeWidth=1.5).encode(
        color=alt.Color("value:Q", scale=alt.Scale(domain=[-30, -12, 0, 12, 30], range=[ORANGE_DARK, "#EDA76A", "#F8FAFC", "#8DB8E6", BLUE], clamp=True, interpolate="rgb"), legend=None),
        tooltip=[alt.Tooltip("month:O", title="Month"), alt.Tooltip("year:O", title="Year"),
                 alt.Tooltip("value:Q", title="Net flow, % of prior-month assets", format="+.1f"),
                 alt.Tooltip("net_flow_crore:Q", title="Net flow (₹ cr)", format=",.0f")],
    )
    text = base.mark_text(fontSize=11).encode(
        text="label:N",
        color=alt.condition("abs(datum.value) > 18", alt.value("white"), alt.value(TEXT)),
    )
    return (rect + text).properties(height=34 * d.year.nunique())


def models_chart(metrics: pd.DataFrame, selected: str) -> alt.LayerChart:
    m = metrics.assign(name=metrics.model.map(MODEL_NAMES), chosen=metrics.model.eq(selected).map({True: "Selected", False: "Other"}))
    order = [MODEL_NAMES[k] for k in MODEL_NAMES if k in set(metrics.model)]
    y = alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelFont=SANS, labelFontSize=12, labelColor=TEXT, grid=True))
    x = alt.X("aum_scaled_mae_pct:Q", title="Average error, % of assets (lower is better)", scale=alt.Scale(zero=False, padding=20),
              axis=alt.Axis(tickCount=6, format=".1f"))
    dots = alt.Chart(m).mark_circle(size=150, opacity=1).encode(
        y=y, x=x,
        color=alt.Color("chosen:N", scale=alt.Scale(domain=["Selected", "Other"], range=[BLUE, "#9DAEC2"]), legend=None),
        tooltip=[alt.Tooltip("name:N", title="Model"), alt.Tooltip("aum_scaled_mae_pct:Q", title="Error, % of assets", format=".2f"),
                 alt.Tooltip("mape_pct:Q", title="MAPE %", format=".0f"), alt.Tooltip("coverage95_pct:Q", title="95% range hit rate", format=".1f")],
    )
    labels = alt.Chart(m).mark_text(dx=12, align="left", fontSize=11, color=TEXT).encode(
        y=alt.Y("name:N", sort=order), x="aum_scaled_mae_pct:Q", text=alt.Text("aum_scaled_mae_pct:Q", format=".1f"))
    return (dots + labels).properties(height=250)


# ---------------------------------------------------------------------------

st.markdown(stylesheet(), unsafe_allow_html=True)
if not (ART / "evidence.json").exists():
    st.error("No results found. Run `make all` (or `python scripts/run_analysis.py`) first.")
    st.stop()

data = load(snapshot_signature())
e = data["evidence"]
selected = e["selection"]["selected_model"]
last_month = pd.Timestamp(e["data"]["end"])

st.markdown('<div class="lf-kicker">Zeckathon 2026 · Problem Statement 03 · Team Angry Bird, IIT Patna</div>', unsafe_allow_html=True)
st.markdown('<div class="lf-title">How much cash should a liquid fund hold?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lf-lead">Forecasts of money moving in and out of India\'s liquid and overnight funds, '
    "and the cash each month needs so that only a 1-in-20 month asks for more.</div>"
    f'<div class="lf-meta">AMFI data to {month_label(last_month)} · {e["data"]["months_per_category"]["Liquid"]} months per category · '
    f'selected model: {MODEL_NAMES[selected]}</div>',
    unsafe_allow_html=True,
)
st.write("")

col_cat, col_h = st.columns([1, 2])
with col_cat:
    category = st.radio("Category", ["Liquid", "Overnight"], horizontal=True, label_visibility="collapsed")
with col_h:
    n_months = st.slider(
        "Forecast horizon (months ahead)",
        min_value=1,
        max_value=24,
        value=4,
        step=1,
        help="Select how many months forward to forecast. E.g. set to 6 to predict through March 2027 fiscal year-end, or up to 24 months.",
    )
flows = data["flows"][data["flows"].category == category]
fc = get_extended_forecast(category, n_months, data["forecasts"])
aum = float(fc.origin_aum.iloc[0])
peak = fc.loc[fc.outflow95_buffer_crore.idxmax()]
march = flows[flows.date.dt.month == 3]

cols = st.columns(4)
next_month = fc.iloc[0]
card(cols[0], f"{next_month.buffer_aum_pct:.1f}%", f"Buffer for {month_label(next_month.date)}: ₹{crore(next_month.outflow95_buffer_crore)} crore")
card(cols[1], f"{peak.buffer_aum_pct:.1f}%", f"Largest buffer, {month_label(peak.date)}: ₹{crore(peak.outflow95_buffer_crore)} crore", warn=True)
card(cols[2], f"{(march.net_flow_crore < 0).sum()} of {len(march)}", f"Marches since {march.date.dt.year.min()} with net outflows")
card(cols[3], f"{aum / 1e5:,.2f}", f"Category assets in {month_label(last_month)}, ₹ lakh crore")
st.write("")

tab_forecast, tab_backtest, tab_season, tab_models, tab_method = st.tabs(
    ["Forecast", "Backtest", "Tax-date pattern", "Models", "Method and data"]
)

with tab_forecast:
    st.subheader(f"{category} funds: last 24 months and the next {n_months} months")
    st.altair_chart(forecast_chart(flows, fc), use_container_width=True, theme=None)
    st.caption("Light bar: 95% range. Darker bar: 80% range. Circle: central forecast. Orange tick: the buffer, the 5% downside of the forecast.")
    def get_status(row):
        h = row.horizon
        m = pd.Timestamp(row.date).month
        if h == 1:
            return "Nowcast"
        if m == 3:
            return "Advance tax and year-end"
        if m in (6, 9, 12):
            return "Advance tax month"
        return "Forecast"

    rows_out = pd.DataFrame({
        "Month": fc.date.map(month_label),
        "Type": fc.apply(get_status, axis=1),
        "Central (₹ cr)": fc.point.map(crore),
        "80% range": fc.q100.map(crore) + " to " + fc.q900.map(crore),
        "95% range": fc.q025.map(crore) + " to " + fc.q975.map(crore),
        "Buffer (₹ cr)": fc.outflow95_buffer_crore.map(crore),
        "% of assets": fc.buffer_aum_pct.map(lambda v: f"{v:.1f}%"),
    })
    peak_row = {int(np.argmax(fc.outflow95_buffer_crore.to_numpy())): "hl"}
    table(rows_out, text_columns=2, row_class=peak_row)
    mc = data["metrics_by_category"]
    cov = mc[(mc.split == "test") & (mc.model == selected) & (mc.category == category) & mc.horizon.isin([1, 2, 3])]
    cov80, cov95 = cov.coverage80_pct.mean(), cov.coverage95_pct.mean()
    verdict = "so the inner band runs a little narrow" if cov95 >= 90 else "so both bands are too narrow; treat the buffer as a minimum"
    st.markdown(
        f'<div class="lf-note"><b>How sure are we?</b> Over the two-year test the {category.lower()}-fund 95% range caught '
        f"{cov95:.0f}% of actual outcomes and the 80% range caught {cov80:.0f}%, {verdict}. "
        "October is a nowcast because AMFI publishes it in early November. "
        "Buffers are per month and should not be added together.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        f"Forecasts ({n_months}m CSV)",
        fc.to_csv(index=False).encode(),
        f"liquidflow_{category.lower()}_{n_months}m_forecasts.csv",
        "text/csv",
        width="stretch",
    )
    d2.download_button("Seasonal-naive forecasts (CSV)", (ART / "baseline_forecasts.csv").read_bytes(), "liquidflow_seasonal_naive_forecasts.csv", "text/csv", width="stretch")
    if REPORT.exists():
        d3.download_button("Report (PDF)", REPORT.read_bytes(), REPORT.name, "application/pdf", width="stretch")

with tab_backtest:
    rows = data["policy"][data["policy"].category == category].sort_values("date")
    ours = int((rows.outflow_crore > rows.q95_buffer_crore).sum())
    rule = int((rows.outflow_crore > rows.naive_buffer_crore).sum())
    st.subheader("The buffer, tested month by month")
    st.write(
        f"October 2024 to September 2026, one month ahead. Each buffer used only what was known at the time. "
        f"For {category.lower()} funds the LiquidFlow buffer fell short in **{ours} of {len(rows)}** months; "
        f"holding last year's outflow for the same month fell short in **{rule}**."
    )
    st.altair_chart(backtest_chart(rows), use_container_width=True, theme=None)
    st.caption("Bars show net outflow. Months with a net inflow have no bar.")
    st.markdown("**What each rule costs, both categories, 48 months**")
    costs = data["costs"]
    literal = costs[(costs.interpretation == "literal_3_rupees_per_rupee") & (costs.policy != "all_origin_aum_reference")]
    table(pd.DataFrame({
        "Rule": literal.policy.map(POLICY_NAMES),
        "Cost (₹ cr)": literal.cost_crore.map(crore),
        "Months short": literal.shortfall_category_months,
        "Average buffer": literal.mean_buffer_aum_pct.map(lambda v: f"{v:.1f}% of assets"),
    }))
    st.caption("Cost from the brief: 0.5% a year on cash held beyond what was needed, plus ₹3 for every rupee of shortfall.")
    with st.expander("If the penalty is read as three times the monthly yield gap"):
        alt_rows = costs[costs.interpretation == "three_times_monthly_yield_gap"]
        table(pd.DataFrame({
            "Rule": alt_rows.policy.map(POLICY_NAMES),
            "Cost (₹ cr)": alt_rows.cost_crore.map(crore),
            "Months short": alt_rows.shortfall_category_months,
        }))
        st.write("Under this reading a shortfall is cheap, so holding little cash wins and last year's rule edges ahead. "
                 "The report follows the brief's wording, ₹3 per rupee.")

with tab_season:
    st.subheader("Money leaves on advance-tax dates")
    st.write(
        "Net flow as a % of the previous month's assets. Advance tax is due on 15 June, 15 September, "
        "15 December and 15 March, and companies redeem to pay it."
    )
    st.altair_chart(heatmap(flows), use_container_width=True, theme=None)
    counts = flows.assign(m=flows.date.dt.strftime("%b")).groupby("m", sort=False).net_flow_crore.agg(
        outflow_years=lambda s: int((s < 0).sum()), years="size")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    counts = counts.reindex(order)
    outflow_row = pd.DataFrame([["Years with a net outflow", *(counts.outflow_years.astype(str) + "/" + counts.years.astype(str))]],
                               columns=["", *order])
    table(outflow_row)

with tab_models:
    metrics = data["metrics"]
    test_rows = metrics[(metrics.split == "test") & metrics.model.isin(MODEL_NAMES)].copy()
    test_rows["order"] = test_rows.model.map(list(MODEL_NAMES).index)
    test_rows = test_rows.sort_values("order")
    st.subheader("Seven models, one chosen before the test")
    st.write(
        f"{MODEL_NAMES[selected]} had the lowest error in the October 2023 to September 2024 selection year and was locked "
        "before the two test years were scored. Simple seasonal methods ended up slightly more accurate in rupee terms; "
        f"{MODEL_NAMES[selected]} had much lower percentage error."
    )
    st.altair_chart(models_chart(test_rows, selected), use_container_width=True, theme=None)
    table(pd.DataFrame({
        "Model": test_rows.model.map(MODEL_NAMES),
        "Error, % of assets": test_rows.aum_scaled_mae_pct.map(lambda v: f"{v:.2f}"),
        "Error (₹ cr)": test_rows.mae_crore.map(crore),
        "MAPE": test_rows.mape_pct.map(lambda v: f"{v:.0f}%"),
        "80% range hit rate": test_rows.coverage80_pct.map(lambda v: f"{v:.1f}%"),
        "95% range hit rate": test_rows.coverage95_pct.map(lambda v: f"{v:.1f}%"),
    }), row_class={i: "sel" for i, m in enumerate(test_rows.model) if m == selected})
    st.caption("144 forecasts per model: two categories, three horizons, 24 target months.")
    st.markdown("**Which inputs helped**")
    ab = data["ablations"]
    names = {"calendar": "Calendar and tax dates", "rates": "Interest rates", "equity": "Equity market", "liquidity": "Banking liquidity"}
    table(pd.DataFrame({
        "Input group removed": ab.group_removed.map(names),
        "Extra error, % of assets": ab.change_vs_full_error_pct_origin_aum.map(lambda v: f"{v:+.2f}"),
    }))
    st.caption("Ridge model, selection year. Positive means the forecast got worse without that group.")

with tab_method:
    st.subheader("How LiquidFlow works")
    st.markdown(
        f"""
1. **Data.** {e['data']['months_per_category']['Liquid']} monthly AMFI reports per category ({e['data']['start']} to {e['data']['end']}), plus RBI call-money rates, the Nifty 50 monthly average and RBI liquidity operations.
2. **Inputs.** Calendar and advance-tax months, interest rates, equity and banking liquidity. Economic inputs lag by two months.
3. **Models.** Seven candidates, refitted every month on an expanding window. One was chosen on the selection year, then locked.
4. **Ranges.** Built from each model's own last 24 forecast errors, scaled by fund assets.
5. **Buffer.** `max(0, −q05)`, the 5% downside of the forecast, capped at the category's assets.

**Cost** = `0.5% × days/365 × max(B − O, 0) + 3 × max(O − B, 0)`, where `B` is the buffer and `O` the actual net outflow.

**Limits.** These are category-level monthly forecasts. A single fund planning for 15 March also needs its own daily redemptions,
investor concentration and maturity profile. Advance-tax months are also quarter ends, so monthly data cannot separate the two effects.
"""
    )
    st.markdown(
        "Sources: [AMFI monthly data](https://www.amfiindia.com/research-information/amfi-monthly) · "
        "[RBI Handbook of Statistics](https://www.rbi.org.in/scripts/AnnualPublications.aspx?head=Handbook%20of%20Statistics%20on%20Indian%20Economy&lang=en) · "
        "[RBI Bulletin](https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx)"
    )
