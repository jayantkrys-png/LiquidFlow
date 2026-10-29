# Data

All data here is public and aggregate. There are no investor-level records.

## Sources

| Folder | Source | What it holds |
| --- | --- | --- |
| `raw/amfi/` | [AMFI monthly reports](https://www.amfiindia.com/research-information/amfi-monthly) | 102 monthly category reports, as downloaded |
| `raw/rbi/` | [RBI Handbook of Statistics](https://www.rbi.org.in/scripts/AnnualPublications.aspx?head=Handbook%20of%20Statistics%20on%20Indian%20Economy&lang=en), Tables 174 and 182 | Call-money rate and Nifty 50 monthly averages |
| `raw/rbi/` | [RBI Bulletin](https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx) | Daily liquidity operations tables |
| `processed/` | Built from the files above | Monthly series used by the models |

## Processed files

- `amfi_flows.csv` (and `.parquet`): monthly subscriptions, redemptions, net flow and AUM for the Liquid and Overnight categories, July 2019 to September 2026, with the source URL and SHA-256 of each report.
- `macro_monthly.csv`: monthly call-money rate, Nifty 50 average change and net RBI liquidity operations.
- `rbi_liquidity_daily.csv`: daily net RBI liquidity operations parsed from the Bulletin tables.
- `features.parquet`: model input rows, one per forecast origin, category and horizon.
- `amfi_manifest.json`, `macro_manifest.json`: every source URL with its SHA-256.

`make audit` checks every raw file against these hashes and writes `artifacts/source_audit.json`.

## Notes

- 87 months per category are used. The 15 oldest AMFI reports combine liquid and money-market funds in one line, so they are kept in `raw/` but not used.
- Dec 2021 and Jan 2022 use AMFI's revised reports.
- RBI Handbook tables give the latest revised values, not the values first published.
- Net RBI liquidity operations show the direction of banking liquidity, not the size of the surplus. Months with fewer than 20 daily observations are marked incomplete.
- AMFI and RBI own the source data. The copies are included so the results can be reproduced, with attribution. Check their terms before any other use.

`make data` downloads the sources again (needs internet). To add newer AMFI months, run `python scripts/acquire_data.py --end YYYY-MM`, then `make all`. New months will change the results.
