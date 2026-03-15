"""figure1_rep.py
================
Replicate Figure 1: scatter plots of short-term debt (x-axis) vs.
interest-rate differential (y-axis) for Euro, Japan, Switzerland,
and the United Kingdom relative to the United States.

Inputs (read from DATA_DIR):
  FIGURE1_XAXIS_FILE    x-axis series from figure_1_xaxis.py
  ST_RATE_WIDE_FILE     wide-format IR3TIB rates from pull_st_rate.py

Outputs (written to OUTPUT_DIR):
  FIGURE1_PNG_FILE      replicated figure (PNG)
  FIGURE1_DATA_FILE     merged replication data (parquet)

Configuration (via settings.py / .env / CLI):
  DATA_DIR              input data directory        (default: _data/)
  OUTPUT_DIR            figure output directory     (default: _output/)
  FIGURE1_XAXIS_FILE    (default: figure1_xaxis.parquet)
  ST_RATE_WIDE_FILE     (default: oecd_ir3tib_wide.parquet)
  FIGURE1_PNG_FILE      (default: figure1_replicated.png)
  FIGURE1_DATA_FILE     (default: figure1_reproduced_data.parquet)

Dependencies: pip install pandas numpy matplotlib pyarrow python-decouple
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import config

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")

FIGURE1_XAXIS_FILE = config("FIGURE1_XAXIS_FILE", default="figure1_xaxis.parquet",            cast=str)
ST_RATE_WIDE_FILE  = config("ST_RATE_WIDE_FILE",  default="oecd_ir3tib_wide.parquet",          cast=str)
FIGURE1_PNG_FILE   = config("FIGURE1_PNG_FILE",   default="figure1_replicated.png",            cast=str)
FIGURE1_DATA_FILE  = config("FIGURE1_DATA_FILE",  default="figure1_reproduced_data.parquet",   cast=str)

# ── Country mappings ──────────────────────────────────────────────────────────

# x-axis file uses ISO3 codes; rate file uses OECD codes
ISO3_TO_SHORT = {"JPN": "JP", "CHE": "CH", "GBR": "UK", "EA": "EA"}
SHORT_TO_OECD = {"US": "USA", "EA": "EA19", "JP": "JPN", "CH": "CHE", "UK": "GBR"}

PANEL_TITLES = {
    "EA": "A. Euro",
    "JP": "B. Japan",
    "CH": "C. Switzerland",
    "UK": "D. United Kingdom",
}
PANEL_ORDER = ["EA", "JP", "CH", "UK"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_xaxis(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / FIGURE1_XAXIS_FILE)
    df["country"] = df["country"].replace(ISO3_TO_SHORT)
    return df[["country", "year", "x_raw", "x"]]


def load_yaxis(data_dir: Path) -> pd.DataFrame:
    """Build y = US minus foreign short-term rate (December, demeaned)."""
    rate_df = pd.read_parquet(data_dir / ST_RATE_WIDE_FILE)

    rate_df["TIME_PERIOD"] = pd.to_datetime(rate_df["TIME_PERIOD"])
    rate_df["year"]  = rate_df["TIME_PERIOD"].dt.year
    rate_df["month"] = rate_df["TIME_PERIOD"].dt.month

    # Year-end (December) observation
    rate_dec = rate_df[rate_df["month"] == 12].copy()

    # Long format
    long_rows = []
    for c_short, c_oecd in SHORT_TO_OECD.items():
        tmp = rate_dec[["year", c_oecd]].copy()
        tmp.columns = ["year", "rate"]
        tmp["country"] = c_short
        long_rows.append(tmp)
    rate_long = pd.concat(long_rows, ignore_index=True)

    # Wide → compute US minus foreign → demean
    rate_wide = rate_long.pivot(index="year", columns="country", values="rate").reset_index()

    y_rows = []
    for c in PANEL_ORDER:
        tmp = rate_wide[["year"]].copy()
        tmp["y_raw"] = rate_wide["US"] - rate_wide[c]
        tmp["y"]     = tmp["y_raw"] - tmp["y_raw"].mean()
        tmp["country"] = c
        y_rows.append(tmp)

    return pd.concat(y_rows, ignore_index=True)


def build_plot_data(data_dir: Path) -> pd.DataFrame:
    x_df = load_xaxis(data_dir)
    y_df = load_yaxis(data_dir)
    plot_df = x_df.merge(y_df, on=["country", "year"], how="inner")

    print("Merged observations by country:")
    print(plot_df.groupby("country").size())
    return plot_df.sort_values(["country", "year"]).reset_index(drop=True)


def plot_figure1(plot_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, c in zip(axes, PANEL_ORDER):
        d = plot_df[plot_df["country"] == c].sort_values("year")

        ax.scatter(d["x"], d["y"], s=18)

        for _, row in d.iterrows():
            ax.text(
                row["x"], row["y"],
                str(row["year"])[-2:],
                fontsize=9, ha="center", va="center",
            )

        if len(d) >= 2:
            coef = np.polyfit(d["x"], d["y"], 1)
            xx = np.linspace(d["x"].min(), d["x"].max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], linewidth=1.5)

        ax.set_title(PANEL_TITLES[c], fontsize=12)
        ax.axhline(0, linewidth=0.6, alpha=0.5)
        ax.axvline(0, linewidth=0.6, alpha=0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-4, 4)
        ax.set_xticks([-1, -0.8, -0.4, 0, 0.4, 0.8, 1])
        ax.set_yticks([-4, -2, 0, 2, 4])

    fig.supxlabel(
        "Log face value of short-term government debt (foreign minus US)",
        fontsize=12,
    )
    fig.supylabel(
        "Short-term interest rate differential (US minus foreign, percentage points)",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir   = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = build_plot_data(data_dir)

    plot_figure1(plot_df, output_dir / FIGURE1_PNG_FILE)

    data_out = data_dir / FIGURE1_DATA_FILE
    plot_df.to_parquet(data_out, index=False)
    print(f"Saved data:   {data_out}")
    # print(plot_df.head(20))