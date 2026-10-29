"""Check every raw source file against its recorded hash and list excluded months."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parents[1])
parser = argparse.ArgumentParser()
parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
args = parser.parse_args()
flow = pd.read_csv("data/processed/amfi_flows.csv")
manifest = json.loads(Path("data/processed/amfi_manifest.json").read_text())
included_urls = set(flow.source_url)
exclusions, included = [], []
for source in manifest["reports"]:
    raw_path = args.raw_root / "amfi" / source["url"].rsplit("/", 1)[-1]
    assert raw_path.exists(), raw_path
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert digest == source["sha256"], source["url"]
    if source["url"] in included_urls:
        values = flow.loc[flow.source_url == source["url"]]
        assert set(values.category) == {"Liquid", "Overnight"}
        assert values.source_sha256.eq(digest).all()
        included.append(source["month"])
    else:
        exclusions.append({
            "month": source["month"], "url": source["url"],
            "reason": "Layout before July 2019 does not report liquid and overnight funds as a comparable pair; not spliced into the series.",
        })
audit = {
    "reports_downloaded": len(manifest["reports"]),
    "included_reports": len(included), "excluded_reports": exclusions,
    "acquisition_failures": manifest["failures"],
    "core_source_hash_verification": "PASS",
    "primary_key": ["month", "category"],
    "original_release_vintages": False,
}
macro_manifest = json.loads(Path("data/processed/macro_manifest.json").read_text())
verified_macro = 0
for source in macro_manifest["sources"]:
    if source.get("source_id"):
        raw_path = args.raw_root / "rbi" / f"liquidity_{source['source_id']}.html"
    elif "23999" in source["url"]:
        raw_path = args.raw_root / "rbi" / "call_rate_pct.html"
    elif "24007" in source["url"]:
        raw_path = args.raw_root / "rbi" / "nifty_monthly_average.html"
    else:
        raise ValueError(f"Unknown macro source mapping: {source['url']}")
    assert raw_path.exists(), raw_path
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == source["sha256"], source["url"]
    verified_macro += 1
daily = pd.read_csv("data/processed/rbi_liquidity_daily.csv")
monthly = pd.read_csv("data/processed/macro_monthly.csv").set_index("month")
independent_mean = daily.groupby("month").net_rbi_operations_crore.mean()
for month, value in independent_mean.items():
    recorded = monthly.loc[month, "net_rbi_operations_crore"]
    if pd.notna(recorded):
        assert abs(value - recorded) < .000001
audit["macro_source_hashes_verified"] = verified_macro
audit["macro_acquisition_failures"] = macro_manifest["failures"]
audit["liquidity_observed_days_range"] = [
    int(daily.groupby("month").size().min()), int(daily.groupby("month").size().max())
]
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/source_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps({k: v for k, v in audit.items() if k != "excluded_reports"}, indent=2))
print("Excluded source months:", ", ".join(r["month"] for r in exclusions))