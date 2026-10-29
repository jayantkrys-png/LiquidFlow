"""Download RBI inputs into data/processed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from liquidflow.macro import acquire_macro

parser = argparse.ArgumentParser()
parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
args = parser.parse_args()
result = acquire_macro(args.raw_root, Path("data/processed"))
for name in ["call_rate_pct", "nifty_monthly_average", "net_rbi_operations_crore"]:
    good = result[result[name].notna()]
    print(name, "rows", len(good), "start", good.month.min(), "end", good.month.max())