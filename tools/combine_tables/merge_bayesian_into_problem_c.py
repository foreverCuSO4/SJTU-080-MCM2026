"""Merge Bayesian Monte Carlo columns into 2026_MCM_Problem_C_Data.csv.

Source:
  tools/combine_tables/monte_carlo_results_1M_all_hit_0201.csv
Columns to merge (per week):
  - Est_Fan%_Mean_Bayesian
  - Est_Fan%_Std_Bayesian
  - Acceptance_Rate%_Bayesian

Target:
  tools/combine_tables/2026_MCM_Problem_C_Data.csv

Output:
  tools/combine_tables/2026_MCM_Problem_C_Data_with_bayesian_weekly.csv

Strategy:
  The target table is one row per (season, celebrity_name) with week1..week11 judge scores.
  The Monte Carlo table is one row per (Season, Week, Celebrity, ...).
  We pivot the three Bayesian columns to wide format by week, producing columns like:
    week1_Est_FanPct_Mean_Bayesian, week1_Est_FanPct_Std_Bayesian, week1_Acceptance_RatePct_Bayesian
    week2_...
  Then we left-merge into the target on (season, celebrity_name) after normalizing names.

Notes:
  - If the Monte Carlo table has duplicate rows for the same (Season, Week, Celebrity),
    we average the three numeric columns.
  - We print a brief unmatched report to help spot name mismatches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_TARGET = HERE / "2026_MCM_Problem_C_Data.csv"
DEFAULT_SOURCE = HERE / "monte_carlo_results_1M_all_hit_0201.csv"
DEFAULT_OUT = HERE / "2026_MCM_Problem_C_Data_with_bayesian_weekly.csv"


def _normalize_name(s: object) -> str:
    """Normalize celebrity names for joining.

    - converts to string
    - strips leading/trailing whitespace
    - collapses internal whitespace
    """

    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> None:
    target_path = DEFAULT_TARGET
    source_path = DEFAULT_SOURCE
    out_path = DEFAULT_OUT

    if not target_path.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    target = pd.read_csv(target_path)
    source = pd.read_csv(source_path)

    # Validate required columns
    required_source_cols = {
        "Season",
        "Week",
        "Celebrity",
        "Est_Fan%_Mean_Bayesian",
        "Est_Fan%_Std_Bayesian",
        "Acceptance_Rate%_Bayesian",
    }
    missing = required_source_cols - set(source.columns)
    if missing:
        raise ValueError(f"Source is missing columns: {sorted(missing)}")

    required_target_cols = {"season", "celebrity_name"}
    missing_t = required_target_cols - set(target.columns)
    if missing_t:
        raise ValueError(f"Target is missing columns: {sorted(missing_t)}")

    # Normalize join keys
    target = target.copy()
    source = source.copy()

    target["_season"] = pd.to_numeric(target["season"], errors="coerce").astype("Int64")
    target["_celebrity"] = target["celebrity_name"].map(_normalize_name)

    source["_season"] = pd.to_numeric(source["Season"], errors="coerce").astype("Int64")
    source["_week"] = pd.to_numeric(source["Week"], errors="coerce").astype("Int64")
    source["_celebrity"] = source["Celebrity"].map(_normalize_name)

    # Keep only relevant columns
    val_cols = [
        "Est_Fan%_Mean_Bayesian",
        "Est_Fan%_Std_Bayesian",
        "Acceptance_Rate%_Bayesian",
    ]

    # Coerce values to numeric where possible
    for c in val_cols:
        source[c] = pd.to_numeric(source[c], errors="coerce")

    # Deduplicate by averaging if there are duplicates for same (season, week, celebrity)
    grouped = (
        source[["_season", "_week", "_celebrity", *val_cols]]
        .dropna(subset=["_season", "_week", "_celebrity"])
        .groupby(["_season", "_week", "_celebrity"], as_index=False)
        .mean(numeric_only=True)
    )

    # Pivot to wide format by week
    rename_metric = {
        "Est_Fan%_Mean_Bayesian": "Est_FanPct_Mean_Bayesian",
        "Est_Fan%_Std_Bayesian": "Est_FanPct_Std_Bayesian",
        "Acceptance_Rate%_Bayesian": "Acceptance_RatePct_Bayesian",
    }

    wide_parts = []
    for metric in val_cols:
        tmp = grouped.pivot(index=["_season", "_celebrity"], columns="_week", values=metric)
        tmp.columns = [f"week{int(w)}_{rename_metric[metric]}" for w in tmp.columns]
        wide_parts.append(tmp)

    wide = pd.concat(wide_parts, axis=1).reset_index()

    # Merge
    merged = target.merge(wide, how="left", on=["_season", "_celebrity"])

    # Report unmatched
    target_keys = set(zip(target["_season"].astype("Int64"), target["_celebrity"]))
    source_keys = set(zip(wide["_season"].astype("Int64"), wide["_celebrity"]))

    missing_in_source = sorted(target_keys - source_keys)
    extra_in_source = sorted(source_keys - target_keys)

    print(f"Target rows: {len(target):,}")
    print(f"Source unique (season, celebrity) rows after pivot: {len(wide):,}")
    print(f"Merged rows: {len(merged):,}\n")

    if missing_in_source:
        print(f"WARNING: {len(missing_in_source)} target (season, celebrity) keys not found in source.")
        print("First 20 missing keys:")
        for s, name in missing_in_source[:20]:
            print(f"  season={s}, celebrity={name!r}")
        print()

    if extra_in_source:
        print(f"NOTE: {len(extra_in_source)} source (season, celebrity) keys not present in target.")
        print("First 20 extra keys:")
        for s, name in extra_in_source[:20]:
            print(f"  season={s}, celebrity={name!r}")
        print()

    # Drop helper cols and write
    merged = merged.drop(columns=["_season", "_celebrity"], errors="ignore")
    merged.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
