"""summary_stats.py
==================
Generates summary statistics for the Koijen & Yogo (2020) replication data.

This module reads the tidy parquets produced by tidy_data.py and outputs:
  1. A LaTeX table (``_output/summary_stats_table.tex``) summarising the
     cross-sectional distribution of outstanding amounts by asset class and
     the total across the 37-country sample for the most recent year.
  2. A time-series line chart (``_output/summary_stats_chart.png``) showing
     total outstanding by asset class over the full sample period.
  3. A stacked horizontal bar chart (``_output/summary_stats_country_bar.png``)
     of each country's total outstanding in the latest year, broken down by
     asset class, to show which economies dominate the sample.
  4. A dot chart (``_output/summary_stats_foreign_share.png``) of the foreign
     ownership share of total outstanding by country and asset class, directly
     illustrating the cross-border demand patterns central to the paper.

All outputs are embedded in ``reports/report_summary.tex`` with descriptive
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
import matplotlib.ticker as mticker
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


def _load_bilateral(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "tidy_bilateral.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"tidy_bilateral.parquet not found at {path}. "
            "Run tidy_data.py first."
        )
    return pd.read_parquet(path)


def _latest_year(df: pd.DataFrame) -> int:
    return int(df["year"].dropna().max())


# ---------------------------------------------------------------------------
# Table: cross-sectional summary for the latest year
# ---------------------------------------------------------------------------

def build_summary_table(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return a summary DataFrame for *year* with one row per asset-class type."""
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
    """Write *summary* as a LaTeX table to *out_path*."""
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
# Chart 1: total outstanding by asset class over time
# ---------------------------------------------------------------------------

def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total outstanding (USD bn) by year and asset-class type."""
    return (
        df.dropna(subset=["year", "type", "outstand_bn"])
        .groupby(["year", "type"], as_index=False)["outstand_bn"]
        .sum()
    )


def write_timeseries_chart(ts: pd.DataFrame, out_path: Path) -> None:
    """Line chart of total outstanding by asset class over time."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for tp, label in _TYPE_LABEL.items():
        sub = ts[ts["type"] == tp].sort_values("year")
        if sub.empty:
            continue
        ax.plot(
            sub["year"].astype(int),
            sub["outstand_bn"] / 1_000,
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
    print(f"  Chart 1 → {out_path}")


# ---------------------------------------------------------------------------
# Chart 2: country-level stacked bar (latest year)
# ---------------------------------------------------------------------------

def write_country_bar_chart(df: pd.DataFrame, year: int, out_path: Path) -> None:
    """Horizontal stacked bar of outstanding by country and asset class."""
    sub = df[df["year"] == year].copy()

    # Pivot: rows = countries, columns = asset types
    pivot = (
        sub.groupby(["iso3", "type"])["outstand_bn"]
        .sum()
        .unstack(fill_value=0.0)
    )
    # Sort by total descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    # Convert to trillions for readability
    pivot = pivot / 1_000

    fig, ax = plt.subplots(figsize=(7, max(5, len(pivot) * 0.28)))

    left = np.zeros(len(pivot))
    for tp, label in _TYPE_LABEL.items():
        if tp not in pivot.columns:
            continue
        vals = pivot[tp].values
        ax.barh(pivot.index, vals, left=left, color=_TYPE_COLOR[tp],
                label=label, height=0.7)
        left += vals

    ax.set_xlabel("Outstanding (USD trillions)")
    ax.set_title(
        f"Outstanding Financial Assets by Country ({year})\n"
        "(37-country sample, Koijen & Yogo 2020 replication)"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart 2 → {out_path}")


# ---------------------------------------------------------------------------
# Chart 3: foreign ownership share by country (latest year)
# ---------------------------------------------------------------------------

def build_foreign_shares(
    amounts: pd.DataFrame,
    bilateral: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Compute foreign holdings as % of outstanding for each (iso3, type).

    Foreign holdings = total inflows from all foreign investors into a given
    issuer country, summed across all investor countries.
    """
    bil_yr = bilateral[bilateral["year"] == year].copy()
    # Sum all foreign investors' holdings in each issuer country by type
    foreign = (
        bil_yr[bil_yr["investor"] != bil_yr["issuer"]]
        .groupby(["issuer", "type"], as_index=False)["value_bn"]
        .sum()
        .rename(columns={"issuer": "iso3"})
    )

    amt_yr = (
        amounts[amounts["year"] == year]
        .groupby(["iso3", "type"], as_index=False)["outstand_bn"]
        .sum()
    )

    merged = amt_yr.merge(foreign, on=["iso3", "type"], how="left")
    merged["value_bn"] = merged["value_bn"].fillna(0.0)
    merged["foreign_share"] = (merged["value_bn"] / merged["outstand_bn"]).clip(0, 1)
    return merged.dropna(subset=["foreign_share"])


def write_foreign_share_chart(
    shares: pd.DataFrame,
    year: int,
    out_path: Path,
) -> None:
    """Dot/scatter chart of foreign ownership share by country and asset class."""
    # Determine country order by average foreign share across types
    order = (
        shares.groupby("iso3")["foreign_share"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(7, max(5, len(order) * 0.28)))

    markers = {1: "o", 2: "s", 3: "^"}
    for tp, label in _TYPE_LABEL.items():
        sub = shares[shares["type"] == tp].set_index("iso3")
        y_pos = [order.index(c) for c in sub.index if c in order]
        x_vals = [sub.loc[c, "foreign_share"] * 100 for c in sub.index if c in order]
        ax.scatter(x_vals, y_pos, color=_TYPE_COLOR[tp], marker=markers[tp],
                   s=40, label=label, zorder=3)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("Foreign ownership share (%)")
    ax.set_title(
        f"Foreign Ownership Share by Country and Asset Class ({year})\n"
        "(Koijen & Yogo 2020 replication)"
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart 3 → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tidy data …")
    amounts = _load_amounts(DATA_DIR)
    bilateral = _load_bilateral(DATA_DIR)
    year = _latest_year(amounts)
    print(f"  Latest year: {year}  |  amounts rows: {len(amounts):,}  "
          f"bilateral rows: {len(bilateral):,}")

    print("Building summary table …")
    summary = build_summary_table(amounts, year)
    write_latex_table(summary, year, OUTPUT_DIR / "summary_stats_table.tex")

    print("Building time-series chart …")
    ts = build_time_series(amounts)
    write_timeseries_chart(ts, OUTPUT_DIR / "summary_stats_chart.png")

    print("Building country bar chart …")
    write_country_bar_chart(amounts, year, OUTPUT_DIR / "summary_stats_country_bar.png")

    print("Building foreign ownership share chart …")
    shares = build_foreign_shares(amounts, bilateral, year)
    write_foreign_share_chart(shares, year, OUTPUT_DIR / "summary_stats_foreign_share.png")


if __name__ == "__main__":
    main()
