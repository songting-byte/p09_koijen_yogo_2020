"""figure2_rep.py
================
Replicate Figure 2: scatter plots of long-term debt (x-axis) vs.
long-term yield differential (y-axis) for Germany, Japan, Switzerland,
and the United Kingdom relative to the United States.

Both axes use 5Y zero-coupon yields derived from 10Y par yields via
Nelson-Siegel (β₁ = β₂ = 0), consistent with figure2_x.py and the paper:
  "We estimate a Nelson and Siegel (1987) zero-coupon yield curve for each
   country, assuming that the ten-year benchmark yield is the par yield."

Inputs (read from DATA_DIR):
  FIGURE2_XAXIS_FILE    x-axis series       from figure2_x.py
  IR10Y_WIDE_FILE       10Y par yields      from figure2_interestrate.py

Outputs:
  FIGURE2_PNG_FILE      replicated figure (PNG)          → OUTPUT_DIR
  FIGURE2_DATA_FILE     merged replication data           → DATA_DIR

Configuration (via settings.py / .env / CLI):
  DATA_DIR              input data directory       (default: _data/)
  OUTPUT_DIR            figure output directory    (default: _output/)
  FIGURE2_XAXIS_FILE    (default: figure2_xaxis.parquet)
  IR10Y_WIDE_FILE       (default: figure2_yaxis.parquet)
  FIGURE2_PNG_FILE      (default: figure2_replicated.png)
  FIGURE2_DATA_FILE     (default: figure2_reproduced_data.parquet)
  NS_TAU                Nelson-Siegel τ parameter  (default: 1.5)

Dependencies: pip install pandas numpy matplotlib scipy pyarrow python-decouple
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from settings import config

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")

FIGURE2_XAXIS_FILE = config("FIGURE2_XAXIS_FILE", default="figure2_xaxis.parquet",           cast=str)
IR10Y_WIDE_FILE    = config("IR10Y_WIDE_FILE",     default="figure2_yaxis.parquet",           cast=str)
FIGURE2_PNG_FILE   = config("FIGURE2_PNG_FILE",    default="figure2_replicated.png",          cast=str)
FIGURE2_DATA_FILE  = config("FIGURE2_DATA_FILE",   default="figure2_reproduced_data.parquet", cast=str)
NS_TAU             = config("NS_TAU",               default=1.5,                              cast=float)

# ── Country mappings ──────────────────────────────────────────────────────────

PANEL_TITLES = {
    "DEU": "A. Germany",
    "JPN": "B. Japan",
    "CHE": "C. Switzerland",
    "GBR": "D. United Kingdom",
}
PANEL_ORDER  = ["DEU", "JPN", "CHE", "GBR"]
IR_COUNTRIES = ["DEU", "JPN", "CHE", "GBR"]   # foreign countries for y-axis

# ── Nelson-Siegel (mirrors figure2_x.py) ──────────────────────────────────────

def _ns_zero(m: float, beta0: float, beta1: float, beta2: float,
             tau: float = NS_TAU) -> float:
    """Nelson-Siegel spot rate at maturity m, continuously compounded decimal."""
    x = m / tau
    return (
        beta0
        + beta1 * (1 - np.exp(-x)) / x
        + beta2 * ((1 - np.exp(-x)) / x - np.exp(-x))
    )


def _par_yield_from_ns(beta0: float, beta1: float, beta2: float,
                       tau: float = NS_TAU, maturity: int = 10) -> float:
    """Par yield implied by NS curve (annual coupons, continuous discounting)."""
    mats  = np.arange(1, maturity + 1, dtype=float)
    spots = np.array([_ns_zero(t, beta0, beta1, beta2, tau) for t in mats])
    dfs   = np.exp(-spots * mats)
    return (1 - dfs[-1]) / dfs.sum()


def par_to_5y_zero(par_yield_10y_pct: float, tau: float = NS_TAU) -> float:
    """Convert 10Y par yield (%) to 5Y zero-coupon yield (%) via Nelson-Siegel.

    Sets β₁ = β₂ = 0 (flat curve identification with a single data point),
    solves for β₀ with Brent's method, then evaluates the curve at m = 5.
    """
    if np.isnan(par_yield_10y_pct):
        return np.nan
    par = par_yield_10y_pct / 100

    def residual(b0):
        return _par_yield_from_ns(b0, 0.0, 0.0, tau, maturity=10) - par

    try:
        beta0 = brentq(residual, a=0.0, b=0.30, xtol=1e-10)
    except ValueError:
        beta0 = par   # fallback: flat curve ≈ par yield

    return _ns_zero(5, beta0, 0.0, 0.0, tau) * 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_xaxis(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / FIGURE2_XAXIS_FILE)[
        ["country", "year", "x_raw", "x"]
    ]


def load_yaxis(data_dir: Path, tau: float = NS_TAU) -> pd.DataFrame:
    """Build y = US minus foreign 5Y zero-coupon yield (December, demeaned).

    Applies the same Nelson-Siegel par→zero conversion as figure2_x.py so
    that both axes are expressed in 5Y zero-coupon terms.
    """
    ir = pd.read_parquet(data_dir / IR10Y_WIDE_FILE)
    ir["TIME_PERIOD"] = pd.to_datetime(ir["TIME_PERIOD"])
    ir["year"]  = ir["TIME_PERIOD"].dt.year
    ir["month"] = ir["TIME_PERIOD"].dt.month

    ir_dec = ir[ir["month"] == 12].copy()

    # Convert all 10Y par yields → 5Y zero-coupon yields
    for col in ["USA"] + IR_COUNTRIES:
        ir_dec[f"{col}_5y"] = ir_dec[col].apply(
            lambda r: par_to_5y_zero(r, tau=tau)
        )

    y_rows = []
    for c in IR_COUNTRIES:
        tmp = ir_dec[["year"]].copy()
        tmp["y_raw"]   = ir_dec["USA_5y"].values - ir_dec[f"{c}_5y"].values
        tmp["y"]       = tmp["y_raw"] - tmp["y_raw"].mean()
        tmp["country"] = c
        y_rows.append(tmp.reset_index(drop=True))

    return pd.concat(y_rows, ignore_index=True)


def build_plot_data(data_dir: Path, tau: float = NS_TAU) -> pd.DataFrame:
    x_df = load_xaxis(data_dir)
    y_df = load_yaxis(data_dir, tau=tau)
    plot_df = x_df.merge(y_df, on=["country", "year"], how="inner")

    print("Merged observations by country:")
    print(plot_df.groupby("country").size())
    return plot_df.sort_values(["country", "year"]).reset_index(drop=True)


def plot_figure2(plot_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, c in zip(axes, PANEL_ORDER):
        d = plot_df[plot_df["country"] == c].sort_values("year")

        ax.scatter(d["x"], d["y"], s=18, zorder=3)

        for _, row in d.iterrows():
            ax.text(
                row["x"], row["y"],
                str(int(row["year"]))[-2:],
                fontsize=9, ha="center", va="center",
            )

        if len(d) >= 2:
            coef = np.polyfit(d["x"], d["y"], 1)
            xx = np.linspace(d["x"].min(), d["x"].max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], linewidth=1.5, zorder=2)

        ax.axhline(0, linewidth=0.6, alpha=0.5)
        ax.axvline(0, linewidth=0.6, alpha=0.5)
        ax.set_title(PANEL_TITLES[c], fontsize=12)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
        ax.set_yticks([-2, -1, 0, 1, 2])

    fig.supxlabel(
        "Log face value of long-term government debt (foreign minus US)",
        fontsize=12,
    )
    fig.supylabel(
        "Long-term interest rate differential (US minus foreign, percentage points)",
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

    plot_df = build_plot_data(data_dir, tau=NS_TAU)

    plot_figure2(plot_df, output_dir / FIGURE2_PNG_FILE)

    data_out = data_dir / FIGURE2_DATA_FILE
    plot_df.to_parquet(data_out, index=False)
    print(f"Saved data:   {data_out}")
    # print(plot_df.head(20))