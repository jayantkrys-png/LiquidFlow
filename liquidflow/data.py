"""AMFI monthly report download, parsing and accounting checks.

Raw reports are stored exactly as downloaded. Every parsed row keeps its source
URL and the SHA-256 of the raw file it came from.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

AMFI_ARCHIVE = "https://www.amfiindia.com/research-information/amfi-monthly"
USER_AGENT = "LiquidFlow/1.0 (research; public aggregate data)"
MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
TARGETS = {"overnight fund": "Overnight", "liquid fund": "Liquid"}


def get_bytes(url: str, retries: int = 2) -> tuple[bytes, dict]:
    """Fetch a public URL with a small number of retries."""
    error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=40
            )
            response.raise_for_status()
            body = response.content
            return body, {
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "http_last_modified": response.headers.get("Last-Modified"),
            }
        except requests.RequestException as exc:
            error = exc
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Unable to download {url}: {error}")


def normalized_text(value: object) -> str:
    """Normalize string whitespace and convert to lowercase."""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip().lower()


def numeric(value: object) -> float:
    """Convert numeric string or number to float, handling commas and parentheses."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    return float(text)


def discover_amfi(start: str, end: str, raw_root: Path) -> list[dict]:
    """Find AMFI report download links for months between start and end."""
    body, metadata = get_bytes(AMFI_ARCHIVE)
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "amfi_archive.html").write_bytes(body)
    soup = BeautifulSoup(body, "html.parser")
    sources: dict[tuple[str, str], dict] = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(AMFI_ARCHIVE, anchor["href"])
        match = re.search(r"/am([a-z]{3})(\d{4})repo(?:revised)?\.(xls|pdf)$", url, re.IGNORECASE)
        if not match:
            continue
        month = f"{match[2]}-{MONTHS[match[1].lower()]:02d}"
        if start <= month <= end:
            ext = match[3].lower()
            sources[(month, ext)] = {"month": month, "format": ext, "url": url}
    (raw_root / "archive_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    result = []
    for period in pd.period_range(start, end, freq="M"):
        month = str(period)
        source = sources.get((month, "xls"), sources.get((month, "pdf")))
        if source:
            result.append(source)
    return result


def identify_header(frame: pd.DataFrame) -> dict[str, int]:
    """Identify columns by their source headings, not by a fixed column number."""
    patterns = {
        "subscriptions_crore": ("funds mobil", "mobilisation", "mobilization"),
        "redemptions_crore": ("repurchase", "redemption"),
        "net_flow_crore": ("net inflow", "net inflows", "net flow"),
        "aum_crore": ("net assets under", "net assets as", "net assets at", "aum as on"),
        "average_aum_crore": ("average net", "average assets"),
        "schemes": ("no. of schemes", "number of schemes"),
        "folios": ("no. of folio", "number of folio"),
    }
    result = {}
    # Some old reports have split headings. Combining their first eight rows
    # also handles those without changing or guessing the source's units.
    for col in frame.columns:
        texts = [normalized_text(v) for v in frame[col].head(8) if pd.notna(v)]
        combined = " ".join(texts)
        for name, terms in patterns.items():
            if any(term in combined for term in terms):
                if name == "aum_crore" and (
                    "average" in combined or "segregated" in combined
                ):
                    continue
                if name not in result:
                    result[name] = int(col)
    return result


def parse_amfi_xls(path: Path, month: str, url: str, digest: str) -> list[dict]:
    """Parse monthly fund flow rows from an AMFI Excel workbook."""
    # AMFI serves both OLE/BIFF and OOXML workbooks with a .xls suffix.
    # Pandas' signature-based engine detection handles the real file format.
    workbook = pd.ExcelFile(path)
    rows = []
    seen = set()
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet, header=None)
        header = identify_header(frame)
        for _, row in frame.iterrows():
            names = [normalized_text(v) for v in row if pd.notna(v)]
            hits = [TARGETS[name] for name in names if name in TARGETS]
            if not hits:
                continue
            category = hits[0]
            if category in seen:
                continue
            required = {
                "subscriptions_crore", "redemptions_crore",
                "net_flow_crore", "aum_crore",
            }
            if not required.issubset(header):
                # The name can appear in a note or another non-data worksheet.
                continue
            values = {}
            for name, col in header.items():
                try:
                    values[name] = numeric(row.iloc[col])
                except (ValueError, TypeError):
                    values[name] = np.nan
            if any(not np.isfinite(values[k]) for k in required):
                continue
            residual = (
                values["subscriptions_crore"] - values["redemptions_crore"]
                - values["net_flow_crore"]
            )
            if abs(residual) > 1.0:
                raise ValueError(
                    f"Accounting mismatch {month} {category}: {residual} crore"
                )
            if values["aum_crore"] <= 0:
                raise ValueError(f"Nonpositive AUM: {month} {category}")
            rows.append({
                "month": month,
                "category": category,
                **values,
                "accounting_residual_crore": residual,
                "source_url": url,
                "source_sha256": digest,
                "source_sheet": sheet,
            })
            seen.add(category)
    return rows


def parse_amfi_pdf(path: Path, month: str, url: str, digest: str) -> list[dict]:
    """Fallback for archive months that only have a PDF report."""
    import fitz
    text = "\n".join(page.get_text(sort=True) for page in fitz.open(path))
    rows = []
    for name, category in TARGETS.items():
        # Restrict the match to one source row, ending before the next named
        # category. Extract the seven established columns in source order.
        match = re.search(
            rf"(?i){re.escape(name)}\s+"
            r"((?:[-−]?\d[\d,]*(?:\.\d+)?\s+){7,12})", text
        )
        if not match:
            continue
        values = [
            float(x.replace(",", "").replace("−", "-"))
            for x in re.findall(r"[-−]?\d[\d,]*(?:\.\d+)?", match[1])
        ]
        if len(values) < 7:
            continue
        schemes, folios, subscriptions, redemptions, flow, aum, aaum = values[:7]
        residual = subscriptions - redemptions - flow
        if abs(residual) > 1.0 or aum <= 0:
            raise ValueError(f"PDF accounting mismatch {month} {category}")
        rows.append({
            "month": month, "category": category, "schemes": schemes,
            "folios": folios, "subscriptions_crore": subscriptions,
            "redemptions_crore": redemptions, "net_flow_crore": flow,
            "aum_crore": aum, "average_aum_crore": aaum,
            "accounting_residual_crore": residual,
            "source_url": url, "source_sha256": digest,
            "source_sheet": "PDF text; seven source columns reconciled",
        })
    return rows


def acquire_amfi(raw_root: Path, output_dir: Path,
                 start: str = "2018-04", end: str = "2026-09") -> pd.DataFrame:
    """Download, verify, and parse AMFI monthly reports into tabular flows."""
    sources = discover_amfi(start, end, raw_root)
    reports = raw_root / "amfi"
    reports.mkdir(parents=True, exist_ok=True)
    manifest, observations, failures = [], [], []

    def download(source: dict):
        path = reports / source["url"].rsplit("/", 1)[-1]
        if path.exists():
            body = path.read_bytes()
            meta = {
                "url": source["url"], "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "cache_reused": True,
            }
        else:
            body, meta = get_bytes(source["url"])
            path.write_bytes(body)
        return source, path, meta

    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = [pool.submit(download, source) for source in sources]
        for job in as_completed(jobs):
            try:
                source, path, metadata = job.result()
                metadata.update({"month": source["month"], "format": source["format"]})
                manifest.append(metadata)
                if source["format"] == "xls":
                    observations.extend(parse_amfi_xls(
                        path, source["month"], source["url"], metadata["sha256"]
                    ))
                else:
                    observations.extend(parse_amfi_pdf(
                        path, source["month"], source["url"], metadata["sha256"]
                    ))
            except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
                failures.append({"reason": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(observations)
    if data.empty:
        raise RuntimeError("No comparable Liquid or Overnight category rows parsed")
    data = data.sort_values(["month", "category"]).reset_index(drop=True)
    if data.duplicated(["month", "category"]).any():
        raise ValueError("Duplicate category/month observations")
    data.to_csv(output_dir / "amfi_flows.csv", index=False, float_format="%.10f")
    data.to_parquet(output_dir / "amfi_flows.parquet", index=False)
    (output_dir / "amfi_manifest.json").write_text(
        json.dumps({
            "archive": AMFI_ARCHIVE, "requested_start": start,
            "requested_end": end, "reports": sorted(manifest, key=lambda r: r["month"]),
            "failures": failures,
            "publication_note": (
                "Download and HTTP modification times are not the dates AMFI "
                "first published each report."
            ),
        }, indent=2), encoding="utf-8"
    )
    return data