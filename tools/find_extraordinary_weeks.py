"""Find extraordinary weeks based on Acceptance_Rate%_Bayesian.

This script:
1) Reads the Monte Carlo results CSV.
2) Aggregates by (Season, Week) and computes the minimum Acceptance_Rate%_Bayesian.
3) Prints all weeks where the min acceptance rate is below a threshold.
4) Writes those weeks to a CSV.

Default paths assume running from the repository root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_weeks_below_threshold(
	df: pd.DataFrame,
	*,
	threshold: float,
	acceptance_col: str = "Acceptance_Rate%_Bayesian",
) -> pd.DataFrame:
	"""Return a per-week table for weeks whose min acceptance rate < threshold."""
	required_cols = {"Season", "Week", acceptance_col}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {sorted(missing)}")

	work = df[["Season", "Week", acceptance_col]].copy()
	work[acceptance_col] = pd.to_numeric(work[acceptance_col], errors="coerce")

	weekly = (
		work.groupby(["Season", "Week"], as_index=False)
		.agg(
			Min_Acceptance_Rate_Bayesian=(acceptance_col, "min"),
			Rows_In_Week=(acceptance_col, "size"),
			NonNull_Acceptance_Values=(acceptance_col, lambda s: int(s.notna().sum())),
		)
		.sort_values(["Season", "Week"], kind="stable")
	)

	result = weekly[weekly["Min_Acceptance_Rate_Bayesian"] < threshold].copy()
	return result


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Print (Season, Week) where Acceptance_Rate%_Bayesian < threshold "
			"(using per-week minimum), and write those weeks to a CSV."
		)
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("tools") / "monte_carlo_results_1M_all_hit_0201.csv",
		help="Input CSV path.",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=5.0,
		help="Threshold for Acceptance_Rate%%_Bayesian.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("tools") / "weeks_acceptance_rate_bayesian_lt5.csv",
		help="Output CSV path to write the filtered weeks.",
	)
	args = parser.parse_args()

	if not args.input.exists():
		raise FileNotFoundError(f"Input CSV not found: {args.input.resolve()}")

	df = pd.read_csv(args.input)
	weeks = find_weeks_below_threshold(df, threshold=args.threshold)

	if weeks.empty:
		print(
			"No weeks found with min Acceptance_Rate%_Bayesian "
			f"< {args.threshold}."
		)
	else:
		print(
			f"Weeks with min Acceptance_Rate%_Bayesian < {args.threshold} "
			f"(count={len(weeks)}):"
		)
		for r in weeks.itertuples(index=False):
			print(
				f"  Season {int(r.Season)}, Week {int(r.Week)}: "
				f"min={r.Min_Acceptance_Rate_Bayesian:.4g} "
				f"(rows={int(r.Rows_In_Week)}, nonnull={int(r.NonNull_Acceptance_Values)})"
			)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	weeks.to_csv(args.output, index=False)
	print(f"Saved weeks CSV to: {args.output.resolve()}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

