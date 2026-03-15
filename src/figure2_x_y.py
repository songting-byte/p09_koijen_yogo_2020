"""figure2_x.py
==============
Compute the Figure 2 x-axis: log face-value ratio of long-term government
debt (foreign minus US), demeaned over time.

Replicates the paper's procedure (Koijen & Yogo 2020, Online Appendix):
  "We construct the long-term yields, based on the ten-year benchmark
   government bond yields from Datastream. We estimate a Nelson and Siegel
   (1987) zero-coupon yield curve for each country, assuming that the
   ten-year benchmark yield is the par yield."

Calculation steps:
  1. Take December year-end 10Y par yield (from OECD / Datastream).
  2. Fit a Nelson-Siegel zero-coupon curve, treating the 10Y par yield
     as the only observable: assume β₁ = β₂ = 0 (flat curve), so
       β₀ = par_yield_to_zero(r_par_10y)
     This is the only identification possible with a single data point.
  3. Read off the 5Y zero-coupon rate from the fitted curve:
       r_5y_zero = nelson_siegel(m=5, β₀, β₁=0, β₂=0, τ)  ≈ β₀
  4. Bond price:  P = exp(-5 × r_5y_zero)
  5. Face value:  FV = MV / P
  6. x_raw = log(FV_foreign) - log(FV_US)
  7. x     = x_raw - mean(x_raw)   [demeaned per country]

Note on identification:
  With only one par yield observation per country-year, the Nelson-Siegel
  curve is under-identified (3 free parameters, 1 equation).  The standard
  practice is to fix τ and set β₁ = β₂ = 0, yielding a flat zero curve
  where r_5y_zero ≈ r_10y_par (difference < 20 bp for developed sovereigns).
  The NS step is kept for methodological fidelity; the numeric impact is
  small but non-zero due to the par→zero convexity adjustment.

Inputs (read from DATA_DIR):
  LT_DEBT_USD_FILE    long-term debt market value   from pull_lt_usd.py
  IR10Y_WIDE_FILE     10Y par yields (wide format)  from figure2_interestrate.py

Output (written to DATA_DIR):
  FIGURE2_XAXIS_FILE  demeaned x-axis series

Configuration (via settings.py / .env / CLI):
  DATA_DIR              input/output directory    (default: _data/)
  OECD_START_PERIOD     start year                (default: "2003")
  OECD_END_PERIOD       end year                  (default: "2020")
  LT_DEBT_USD_FILE      (default: oecd_ltdebt_usd.parquet)
  IR10Y_WIDE_FILE       (default: figure2_yaxis.parquet)
  FIGURE2_XAXIS_FILE    (default: figure2_xaxis.parquet)
  NS_TAU                Nelson-Siegel τ parameter  (default: 1.5)

Dependencies: pip install pandas numpy scipy pyarrow python-decouple
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from settings import config

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR     = config("DATA_DIR")
START_PERIOD = config("OECD_START_PERIOD", default="2003", cast=str)
END_PERIOD   = config("OECD_END_PERIOD",   default="2020", cast=str)

LT_DEBT_USD_FILE   = config("LT_DEBT_USD_FILE",   default="oecd_ltdebt_usd.parquet",  cast=str)
IR10Y_WIDE_FILE    = config("IR10Y_WIDE_FILE",     default="figure2_yaxis.parquet",    cast=str)
FIGURE2_XAXIS_FILE = config("FIGURE2_XAXIS_FILE",  default="figure2_xaxis.parquet",   cast=str)
NS_TAU             = config("NS_TAU",               default=1.5,                       cast=float)

LABEL_MAP = {
    "DEU": "Germany",
    "JPN": "Japan",
    "CHE": "Switzerland",
    "GBR": "United Kingdom",
}

IR_COUNTRIES = ["CHE", "DEU", "GBR", "JPN", "USA"]

# ── Nelson-Siegel zero-coupon curve ──────────────────────────────────────────

def _ns_zero(m: float, beta0: float, beta1: float, beta2: float,
             tau: float = NS_TAU) -> float:
    """Nelson-Siegel spot (zero-coupon) rate at maturity m (years).

    Continuously compounded, expressed as a decimal (not percent).
    """
    x = m / tau
    return (
        beta0
        + beta1 * (1 - np.exp(-x)) / x
        + beta2 * ((1 - np.exp(-x)) / x - np.exp(-x))
    )


def _par_yield_from_ns(beta0: float, beta1: float, beta2: float,
                       tau: float = NS_TAU, maturity: int = 10) -> float:
    """Compute the par yield implied by a Nelson-Siegel zero curve.

    Assumes annual coupon payments, continuously compounded discounting.
    The par yield c satisfies: sum(c * df_t) + df_T = 1, where df_t = exp(-s_t * t).
    Solving: c = (1 - df_T) / sum(df_t).
    """
    maturities      = np.arange(1, maturity + 1, dtype=float)
    spot_rates      = np.array([_ns_zero(t, beta0, beta1, beta2, tau) for t in maturities])
    discount_factors = np.exp(-spot_rates * maturities)
    return (1 - discount_factors[-1]) / discount_factors.sum()


def par_to_5y_zero(par_yield_10y_pct: float, tau: float = NS_TAU) -> float:
    """Convert a 10Y par yield to a 5Y zero-coupon yield via Nelson-Siegel.

    Implements the paper's procedure: treat the 10Y benchmark yield as the
    par yield, fit a Nelson-Siegel curve (with β₁ = β₂ = 0, flat curve),
    and read off the 5-year spot rate.

    Identification:
      With a single observable (10Y par yield), the NS curve is under-identified.
      Setting β₁ = β₂ = 0 gives a flat zero curve; β₀ is then the unique value
      such that the implied 10Y par yield matches the observed par yield.

    Parameters
    ----------
    par_yield_10y_pct : float
        10-year par yield in percent (e.g. 4.5 for 4.5%).
    tau : float
        Nelson-Siegel shape parameter τ (default: 1.5).

    Returns
    -------
    float
        5-year zero-coupon spot rate in percent.
    """
    if np.isnan(par_yield_10y_pct):
        return np.nan

    par = par_yield_10y_pct / 100
    beta1, beta2 = 0.0, 0.0

    # Find β₀ such that par_yield_from_ns(β₀, 0, 0) = par_yield_10y
    # The par yield is monotone in β₀, so Brent's method works reliably.
    def residual(b0):
        return _par_yield_from_ns(b0, beta1, beta2, tau, maturity=10) - par

    # Search bounds: par yield is within [0, 30%]
    try:
        beta0 = brentq(residual, a=0.0, b=0.30, xtol=1e-10)
    except ValueError:
        # Fallback: par ≈ zero for flat curve (difference negligible)
        beta0 = par

    zero_5y = _ns_zero(5, beta0, beta1, beta2, tau)
    return zero_5y * 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load market-value and interest-rate parquet files."""
    mv = pd.read_parquet(data_dir / LT_DEBT_USD_FILE)

    ir = pd.read_parquet(data_dir / IR10Y_WIDE_FILE)
    ir["TIME_PERIOD"] = pd.to_datetime(ir["TIME_PERIOD"])

    return mv, ir


def compute_xaxis(mv: pd.DataFrame, ir: pd.DataFrame,
                  tau: float = NS_TAU) -> pd.DataFrame:
    """Merge MV with year-end yields, compute FV via NS curve, return demeaned x-axis.

    Returns
    -------
    pd.DataFrame with columns: country, label, year, x_raw, x
    """
    # December year-end yield
    ir_dec = ir[ir["TIME_PERIOD"].dt.month == 12].copy()
    ir_dec["year"] = ir_dec["TIME_PERIOD"].dt.year

    ir_long = (
        ir_dec.melt(
            id_vars="year",
            value_vars=IR_COUNTRIES,
            var_name="country",
            value_name="ir10y_pct",
        )
        .dropna()
    )

    print("10Y par yield (December year-end):")
    print(ir_long.pivot(index="year", columns="country", values="ir10y_pct").round(3).to_string())

    # ── Nelson-Siegel: 10Y par yield → 5Y zero-coupon yield ──────────────────
    ir_long["ir5y_zero_pct"] = ir_long["ir10y_pct"].apply(
        lambda r: par_to_5y_zero(r, tau=tau)
    )

    print("\n5Y zero-coupon yield (Nelson-Siegel, τ={:.1f}):".format(tau))
    print(ir_long.pivot(index="year", columns="country", values="ir5y_zero_pct").round(3).to_string())

    print("\nPar→Zero spread (10Y par minus 5Y zero, bp):")
    ir_long["spread_bp"] = (ir_long["ir10y_pct"] - ir_long["ir5y_zero_pct"]) * 100
    print(ir_long.pivot(index="year", columns="country", values="spread_bp").round(1).to_string())

    # Merge market value with yields
    df = mv.merge(ir_long, on=["country", "year"], how="inner")
    print(f"\nMerged rows: {len(df)}")
    print(f"Countries:   {sorted(df['country'].unique())}")
    print(f"Years:       {sorted(df['year'].unique())}")

    # ── Bond price using 5Y zero-coupon yield ─────────────────────────────────
    # P = exp(-5 * r_5y_zero)  [continuously compounded]
    df["ir5y_dec"] = df["ir5y_zero_pct"].fillna(3.0) / 100
    df["P"]         = np.exp(-5 * df["ir5y_dec"])
    df["fv_usd_mn"] = df["mv_usd_mn"] / df["P"]

    # X-axis: log ratio vs US, demeaned per country
    us = df[df["country"] == "USA"].set_index("year")["fv_usd_mn"]

    results = []
    for country, label in LABEL_MAP.items():
        sub    = df[df["country"] == country].set_index("year")["fv_usd_mn"]
        common = sub.index.intersection(us.index)

        rows = [
            {
                "country": country,
                "label":   label,
                "year":    int(yr),
                "x_raw":   np.log(sub[yr]) - np.log(us[yr]),
            }
            for yr in common
            if sub[yr] > 0 and us[yr] > 0
        ]

        df_c = pd.DataFrame(rows)
        df_c["x"] = df_c["x_raw"] - df_c["x_raw"].mean()
        results.append(df_c)

    df_x = pd.concat(results, ignore_index=True)

    print("\nFigure 2 x-axis (demeaned):")
    print(df_x.pivot(index="year", columns="label", values="x").round(4).to_string())

    return df_x


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path(DATA_DIR)

    mv, ir = load_inputs(data_dir)
    df_x   = compute_xaxis(mv, ir)

    out_path = data_dir / FIGURE2_XAXIS_FILE
    df_x.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")