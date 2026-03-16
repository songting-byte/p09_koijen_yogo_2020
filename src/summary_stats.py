"""summary_stats.py
==================
Generates summary statistics for the Koijen & Yogo (2020) replication data.

This module reads the tidy parquets produced by tidy_data.py and outputs:
  1. A LaTeX table (``_output/summary_stats_table.tex``) summarising the
     cross-sectional distribution of outstanding amounts by asset class and
     the total across the 37-country sample for the most recent year.
  2. A static PNG chart (``_output/summary_stats_chart.png``) showing the
     time-series of total financial-asset amounts outstanding broken down
     by asset class (short-term debt, long-term debt, equity).

Both outputs are embedded in ``reports/report_summary.tex`` with descriptive
LaTeX captions explaining what the reader should take away from each exhibit.

Run:
    python summary_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / doit runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import config

DATA_DIR: Path = config("DATA_DIR")
OUTPUT_DIR: Path = config("OUTPUT_DIR")

_TYPE_LABEL = {1: "Short-term debt", 2: "Long-term debt", 3: "Equity"}
_TYPE_COLOR = {1: "#2166ac", 2: "#4dac26", 3: "#d01c8b"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_amounts(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "tidy_amounts.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"tidy_amounts.parquet not found at {path}. "
            "Run tidy_data.py first."
        )
    return pd.read_parquet(path)


def _latest_year(df: pd.DataFrame) -> int:
    return int(df["year"].dropna().max())


# ---------------------------------------------------------------------------
# Table: cross-sectional summary for the latest year
# ---------------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return a summary DataFrame for *year* with one row per asset-class type.

    Columns: N (countries), Mean (USD bn), Median, Std Dev, Min, Max, Total.
    """
    sub = df[df["year"] == year].copy()
    rows = []
    for tp, label in _TYPE_LABEL.items():
        vals = sub[sub["type"] == tp]["outstand_bn"].dropna()
        if vals.empty:
            continue
        rows.append({
            "Asset class": label,
            "Countries": len(vals),
            "Mean (USD bn)": vals.mean(),
            "Median (USD bn)": vals.median(),
            "Std dev (USD bn)": vals.std(),
            "Min (USD bn)": vals.min(),
            "Max (USD bn)": vals.max(),
            "Total (USD bn)": vals.sum(),
        })
    total_all = sub["outstand_bn"].dropna().sum()
    rows.append({
        "Asset class": "All assets",
        "Countries": sub["iso3"].nunique(),
        "Mean (USD bn)": np.nan,
        "Median (USD bn)": np.nan,
        "Std dev (USD bn)": np.nan,
        "Min (USD bn)": np.nan,
        "Max (USD bn)": np.nan,
        "Total (USD bn)": total_all,
    })
    return pd.DataFrame(rows)


def write_latex_table(summary: pd.DataFrame, year: int, out_path: Path) -> None:
    """Write *summary* as a LaTeX longtable to *out_path*."""
    float_cols = [c for c in summary.columns if "(USD bn)" in c]
    fmt = {c: "{:,.1f}".format for c in float_cols}
    fmt["Countries"] = "{:d}".format

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Summary statistics for outstanding financial-asset amounts "
            f"in {year} (USD billions). "
            r"The table reports the cross-sectional distribution across the "
            r"37-country sample used in the Koijen \& Yogo (2020) replication. "
            r"Long-term debt is the dominant asset class by total size, "
            r"while equity shows the widest dispersion across countries, "
            r"reflecting large differences in stock-market development. "
            r"The \emph{All assets} row sums all three classes.}"
        ),
        r"\label{tab:summary_stats}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline\hline",
    ]
    # Header
    header_cells = ["Asset class", "N", "Mean", "Median", "Std dev",
                    "Min", "Max", "Total"]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r" & (countries) & (USD bn) & (USD bn) & (USD bn) "
                 r"& (USD bn) & (USD bn) & (USD bn) \\")
    lines.append(r"\hline")

    for _, row in summary.iterrows():
        def _fmt(col, val):
            if pd.isna(val):
                return "---"
            if col == "Countries":
                return f"{int(val):d}"
            return f"{val:,.1f}"

        cells = [
            str(row["Asset class"]),
            _fmt("Countries", row["Countries"]),
            _fmt("Mean", row["Mean (USD bn)"]),
            _fmt("Median", row["Median (USD bn)"]),
            _fmt("Std dev", row["Std dev (USD bn)"]),
            _fmt("Min", row["Min (USD bn)"]),
            _fmt("Max", row["Max (USD bn)"]),
            _fmt("Total", row["Total (USD bn)"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"  LaTeX table → {out_path}")


# ---------------------------------------------------------------------------
# Chart: total outstanding by asset class over time
# ---------------------------------------------------------------------------

def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total outstanding (USD bn) by year and asset-class type."""
    return (
        df.dropna(subset=["year", "type", "outstand_bn"])
        .groupby(["year", "type"], as_index=False)["outstand_bn"]
        .sum()
    )


def write_chart(ts: pd.DataFrame, out_path: Path) -> None:
    """Save a stacked-area / line chart of totals by asset class to *out_path*."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for tp, label in _TYPE_LABEL.items():
        sub = ts[ts["type"] == tp].sort_values("year")
        if sub.empty:
            continue
        ax.plot(
            sub["year"].astype(int),
            sub["outstand_bn"] / 1_000,   # → USD trillions
            label=label,
            color=_TYPE_COLOR[tp],
            linewidth=2,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Outstanding (USD trillions)")
    ax.set_title(
        "Total Financial-Asset Amounts Outstanding by Asset Class\n"
        "(37-country sample, Koijen & Yogo 2020 replication)"
    )
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tidy amounts …")
    df = _load_amounts(DATA_DIR)
    year = _latest_year(df)
    print(f"  Latest year: {year}  |  rows: {len(df):,}")

    print("Building summary table …")
    summary = build_summary_table(df, year)
    write_latex_table(summary, year, OUTPUT_DIR / "summary_stats_table.tex")

    print("Building time-series chart …")
    ts = build_time_series(df)
    write_chart(ts, OUTPUT_DIR / "summary_stats_chart.png")


if __name__ == "__main__":
    main()
