
"""Visualize Google Trends popularity shares for a DWTS season.

Definition used here (per your spec):
- For each celebrity CSV, popularity = sum of the 2nd column (index 1).
- Normalize across all celebrities so shares sum to 1.

Example:
	python visualization/visualize_google_trends.py --season 34
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PopularityResult:
	name: str
	raw_sum: float


def _safe_read_csv(csv_path: Path) -> pd.DataFrame | None:
	"""Read a season CSV robustly.

	Returns None if the file cannot be parsed as a tabular CSV.
	"""

	try:
		df = pd.read_csv(csv_path)
	except Exception:
		return None
	if df is None or df.shape[1] < 2 or df.shape[0] == 0:
		return None
	return df


def compute_popularity_shares(season_dir: Path) -> pd.DataFrame:
	"""Compute per-celebrity popularity and normalized share.

	Returns a DataFrame with columns: name, raw_sum, share.
	"""

	season_dir = Path(season_dir)
	csv_paths = sorted(season_dir.glob("*.csv"))
	if not csv_paths:
		raise FileNotFoundError(f"No CSV files found in: {season_dir}")

	results: list[PopularityResult] = []
	for p in csv_paths:
		df = _safe_read_csv(p)
		if df is None:
			continue

		# Per spec: sum of the 2nd column (index 1).
		col = pd.to_numeric(df.iloc[:, 1], errors="coerce")
		raw_sum = float(np.nansum(col.to_numpy(dtype=float, copy=False)))

		name = p.stem
		# If the 2nd column has a real header that differs from filename, prefer it.
		try:
			second_col_name = str(df.columns[1])
			if second_col_name and second_col_name.lower() not in {"0", "1"}:
				name = second_col_name
		except Exception:
			pass

		results.append(PopularityResult(name=name, raw_sum=raw_sum))

	if not results:
		raise ValueError(f"No valid CSV tables found in: {season_dir}")

	out = pd.DataFrame([r.__dict__ for r in results])
	out = out.groupby("name", as_index=False)["raw_sum"].sum()

	total = float(out["raw_sum"].sum())
	if not np.isfinite(total) or total <= 0:
		raise ValueError("Total popularity sum is non-positive; cannot normalize.")

	out["share"] = out["raw_sum"] / total
	out = out.sort_values("share", ascending=False).reset_index(drop=True)
	return out


def _try_import_seaborn():
	try:
		import seaborn as sns  # type: ignore

		return sns
	except Exception:
		return None


def plot_popularity_shares(
	df: pd.DataFrame,
	*,
	title: str,
	output_png: Path,
	top_n: int | None = None,
	figsize: tuple[float, float] | None = None,
) -> None:
	"""Create a clean horizontal bar chart for popularity shares."""

	import matplotlib.pyplot as plt

	plot_df = df.copy()
	if top_n is not None:
		plot_df = plot_df.head(int(top_n))

	names = plot_df["name"].astype(str).tolist()
	shares = plot_df["share"].to_numpy(dtype=float)

	if figsize is None:
		# Scale height with number of bars for readability.
		figsize = (11.5, max(4.8, 0.42 * len(plot_df) + 1.8))

	sns = _try_import_seaborn()
	if sns is not None:
		sns.set_theme(style="whitegrid", context="talk")

	fig, ax = plt.subplots(figsize=figsize, dpi=200)

	# Choose a tasteful single-hue palette.
	if sns is not None:
		colors = sns.color_palette("Blues_r", n_colors=len(plot_df))
	else:
		# Fallback gradient-ish palette.
		colors = [(0.15, 0.35, 0.65, 0.95)] * len(plot_df)

	y = np.arange(len(plot_df))
	bars = ax.barh(y, shares, color=colors, edgecolor=(0, 0, 0, 0.08))
	ax.set_yticks(y, labels=names)
	ax.invert_yaxis()

	ax.set_title(title, pad=14, weight="bold")
	ax.set_xlabel("Popularity share (normalized, sum = 1)")

	# Show percentages on bars.
	for rect, s in zip(bars, shares, strict=False):
		ax.text(
			rect.get_width() + 0.002,
			rect.get_y() + rect.get_height() / 2,
			f"{s * 100:.2f}%",
			va="center",
			ha="left",
			fontsize=10,
			color="#1f2937",
		)

	ax.set_xlim(0, max(0.12, float(shares.max()) * 1.18))
	ax.grid(axis="x", linestyle="-", alpha=0.25)
	ax.grid(axis="y", visible=False)
	for spine in ("top", "right", "left"):
		ax.spines[spine].set_visible(False)

	fig.tight_layout()

	output_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_png, bbox_inches="tight")
	plt.close(fig)


def _default_season_dir(season: int) -> Path:
	# Repo layout: Google_trends/get_data/season_XX
	here = Path(__file__).resolve()
	repo_root = here.parents[1]
	return repo_root / "Google_trends" / "get_data" / f"season_{season}"  # type: ignore[return-value]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Visualize season celebrity popularity shares.")
	p.add_argument("--season", type=int, default=34, help="Season number (default: 34).")
	p.add_argument(
		"--season-dir",
		type=Path,
		default=None,
		help="Folder containing per-celebrity CSVs (overrides --season).",
	)
	p.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output PNG path (default: visualization/season_<N>_popularity.png).",
	)
	p.add_argument(
		"--top",
		type=int,
		default=None,
		help="Only plot top N celebrities (default: plot all).",
	)
	p.add_argument(
		"--export-csv",
		type=Path,
		default=None,
		help="Optional: export computed shares to CSV.",
	)
	return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
	args = parse_args(argv)

	season_dir = args.season_dir or _default_season_dir(args.season)
	df = compute_popularity_shares(season_dir)

	if args.export_csv is not None:
		args.export_csv.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(args.export_csv, index=False)

	output_png = args.output
	if output_png is None:
		# Save next to this script by default.
		output_png = Path(__file__).resolve().parent / f"season_{args.season}_popularity.png"

	title = f"Season {args.season} — Celebrity popularity share (Google Trends)"
	plot_popularity_shares(df, title=title, output_png=output_png, top_n=args.top)

	print(f"Saved figure: {output_png}")
	if args.export_csv is not None:
		print(f"Saved table:  {args.export_csv}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

