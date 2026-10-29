"""Download and parse AMFI monthly reports into data/processed."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liquidflow.data import acquire_amfi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--start", default="2018-04")
    parser.add_argument("--end", default="2026-09")
    args = parser.parse_args()
    data = acquire_amfi(
        args.raw_root, Path("data/processed"), args.start, args.end
    )
    print(data.groupby("category").agg(
        rows=("month", "size"), start=("month", "min"), end=("month", "max"),
        max_accounting_error=("accounting_residual_crore",
                              lambda x: float(x.abs().max()))
    ).to_string())


if __name__ == "__main__":
    main()