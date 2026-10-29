"""RBI inputs: call-money rate, Nifty 50 monthly average and daily liquidity operations."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from liquidflow.data import USER_AGENT, get_bytes, numeric

BASE = "https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx"
HANDBOOK = {
    "call_rate_pct": "https://www.rbi.org.in/scripts/PublicationsView.aspx?id=23999",
    "nifty_monthly_average": "https://www.rbi.org.in/scripts/PublicationsView.aspx?id=24007",
}


def financial_year_matrix(path: Path, average_rows: bool = False) -> pd.DataFrame:
    """RBI rows are Indian fiscal years, columns April through March."""
    tables = pd.read_html(path)
    table = next(t for t in tables if t.shape[1] == 14 and len(t) > 5)
    observations = []
    fiscal_year = None
    for _, row in table.iterrows():
        label = str(row.iloc[0]).strip()
        year_match = re.fullmatch(r"(\d{4})-\d{2}", label)
        if year_match:
            fiscal_year = int(year_match[1])
            if average_rows:
                continue
        elif not (average_rows and label.lower() == "average"):
            continue
        if fiscal_year is None:
            continue
        for col in range(1, 13):
            month = (col + 2) % 12 + 1
            year = fiscal_year + (month < 4)
            try:
                value = numeric(row.iloc[col])
            except (ValueError, TypeError):
                continue
            if np.isfinite(value):
                observations.append({"month": f"{year}-{month:02d}", "value": value})
    result = pd.DataFrame(observations)
    if result.duplicated("month").any():
        raise ValueError(f"Repeated fiscal-year cell in {path}")
    return result.sort_values("month")


def parse_liquidity(path: Path, source_url: str) -> list[dict]:
    """Daily net RBI liquidity operations (injection minus absorption)."""
    tables = pd.read_html(path)
    result = []
    for table in tables:
        if table.shape[1] < 7:
            continue
        heading = " ".join(table.head(5).astype(str).values.ravel()).lower()
        if "injection" not in heading and "absorption" not in heading:
            continue
        for _, row in table.iterrows():
            label = str(row.iloc[0]).strip()
            if not re.search(r"\d{4}", label) or not re.search(r"\d{1,2},", label):
                continue
            try:
                date = pd.Timestamp(pd.to_datetime(label.replace(".", ""), format="mixed"))
                value = numeric(row.iloc[-1])
            except (ValueError, TypeError):
                continue
            if np.isfinite(value):
                result.append({
                    "date": str(date.date()),
                    "month": str(date.to_period("M")),
                    "net_rbi_operations_crore": value,
                    "source_url": source_url,
                })
    return result


def acquire_macro(raw_root: Path, output_dir: Path) -> pd.DataFrame:
    """Download, extract, and assemble RBI macro indicators and liquidity operations."""
    raw_root.mkdir(parents=True, exist_ok=True)
    macro_dir = raw_root / "rbi"
    macro_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    series = []
    for name, url in HANDBOOK.items():
        path = macro_dir / f"{name}.html"
        if path.exists():
            body = path.read_bytes()
            meta = {"url": url, "sha256": hashlib.sha256(body).hexdigest(), "cache_reused": True}
        else:
            body, meta = get_bytes(url)
            path.write_bytes(body)
        manifest.append(meta)
        table = financial_year_matrix(path, average_rows=name == "call_rate_pct")
        table = table.rename(columns={"value": name}).set_index("month")
        series.append(table)

    # Use the archive's actual annual forms and copied table links.
    index_path = raw_root / "rbi_bulletin_index.html"
    if not index_path.exists():
        body, _ = get_bytes(BASE)
        index_path.write_bytes(body)
    soup = BeautifulSoup(index_path.read_bytes(), "lxml")
    form = soup.find("form")
    initial = {i.get("name"): i.get("value", "") for i in form.find_all("input") if i.get("name")}
    urls: set[str] = set()

    def annual(year: int) -> tuple[int, list[str]]:
        path = macro_dir / f"bulletin_archive_{year}.html"
        if not path.exists():
            payload = initial.copy()
            payload.update({
                "hdnYear": str(year), "hdnMonth": "0",
                "__EVENTTARGET": "rptBulletinHead$ctl03$lnkBulletinHead",
                "__EVENTARGUMENT": "",
            })
            payload.pop("btn", None)
            response = requests.post(
                BASE, data=payload, timeout=60, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            path.write_bytes(response.content)
        parsed = BeautifulSoup(path.read_bytes(), "lxml")
        links = [
            urljoin(BASE, a["href"]) for a in parsed.find_all("a", href=True)
            if "liquidity" in a.get_text().lower()
            and "operation" in a.get_text().lower()
            and "Id=" in a["href"]
        ]
        return year, links

    with ThreadPoolExecutor(max_workers=4) as pool:
        for year, links in pool.map(annual, range(2019, 2027)):
            print(f"RBI archive {year}: {len(set(links))} liquidity tables", flush=True)
            urls.update(links)
    if len(urls) < 24:
        raise RuntimeError(f"Only {len(urls)} liquidity source tables discovered")

    def daily_source(url: str) -> tuple[list[dict], dict]:
        source_id = re.search(r"Id=(\d+)", url)[1]
        path = macro_dir / f"liquidity_{source_id}.html"
        if path.exists():
            body = path.read_bytes()
            meta = {"url": url, "sha256": hashlib.sha256(body).hexdigest(), "cache_reused": True}
        else:
            body, meta = get_bytes(url)
            path.write_bytes(body)
        rows = parse_liquidity(path, url)
        meta.update({"rows": len(rows), "source_id": source_id})
        return rows, meta

    daily_rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(daily_source, url): url for url in sorted(urls)}
        for job in as_completed(jobs):
            try:
                rows, meta = job.result()
                daily_rows.extend(rows)
                manifest.append(meta)
            except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
                failures.append({"url": jobs[job], "error": str(exc)})
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("No dated liquidity observations were parsed")
    # Some archive pages repeat dates. Keep one value per date when the copies
    # agree and report any that conflict.
    conflicts = daily.groupby("date")["net_rbi_operations_crore"].nunique()
    conflict_dates = conflicts[conflicts > 1].index.tolist()
    if conflict_dates:
        raise ValueError(f"Conflicting RBI vintages for {conflict_dates[:5]}")
    daily = daily.sort_values(["date", "source_url"]).drop_duplicates("date")
    daily.to_csv(output_dir / "rbi_liquidity_daily.csv", index=False)
    monthly = daily.groupby("month").agg(
        net_rbi_operations_crore=("net_rbi_operations_crore", "mean"),
        liquidity_observed_days=("date", "size"),
        liquidity_source_urls=("source_url", lambda x: "|".join(sorted(set(x)))),
    )
    # Months with too few daily observations are left missing.
    monthly.loc[monthly.liquidity_observed_days < 20, "net_rbi_operations_crore"] = np.nan
    series.append(monthly)
    result = pd.concat(series, axis=1).sort_index()
    # This is a monthly-average index change, not a closing-price return.
    result["nifty_average_change_pct"] = (
        result.nifty_monthly_average.pct_change(fill_method=None) * 100
    )
    result.index.name = "month"
    result.reset_index().to_csv(output_dir / "macro_monthly.csv", index=False)
    (output_dir / "macro_manifest.json").write_text(json.dumps({
        "sources": manifest, "failures": failures,
        "liquidity_definition": (
            "Monthly arithmetic mean of RBI Bulletin Table 3 net injection (+) / "
            "absorption (-) daily operations, including the table's OMO components. "
            "A direction proxy, not the stock of surplus reserves or net LAF outstanding."
        ),
        "release_assumption": "Use observations at least two months older than the known-AMFI origin month.",
        "vintage_limit": "Handbook historical values are latest available, not original release vintages.",
    }, indent=2))
    return result.reset_index()