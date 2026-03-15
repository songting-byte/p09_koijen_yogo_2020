"""figure2_lt_usd.py
================
Pull long-term government debt market value in USD from the OECD API.
Used for the Figure 2 x-axis calculation.

Key structure (mirrors pull_oecd.py, maturity S → L):
  A.N.{COUNTRIES}.W.S1.S1.N.L.LE.F3.L.USD._T.S.V.N.T0720._Z

Countries: DEU, JPN, CHE, GBR, USA  (Germany individually, not Euro Area)

Output (written to DATA_DIR):
  LT_DEBT_USD_FILE    long-format parquet  (country, year, mv_usd_mn)

Configuration (via settings.py / .env / CLI):
  DATA_DIR              output directory     (default: _data/)
  OECD_START_PERIOD     start year           (default: "2003")
  OECD_END_PERIOD       end year             (default: "2020")
  LT_DEBT_COUNTRIES     plus-separated list  (default: DEU+JPN+CHE+GBR+USA)
  LT_DEBT_USD_FILE      output filename      (default: oecd_ltdebt_usd.parquet)

Dependencies: pip install requests pandas pyarrow python-decouple
"""

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from settings import config

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR     = config("DATA_DIR")
START_PERIOD = config("OECD_START_PERIOD", default="2003", cast=str)
END_PERIOD   = config("OECD_END_PERIOD",   default="2020", cast=str)

LT_DEBT_COUNTRIES = config(
    "LT_DEBT_COUNTRIES", default="DEU+JPN+CHE+GBR+USA", cast=str
)
LT_DEBT_USD_FILE = config(
    "LT_DEBT_USD_FILE", default="oecd_ltdebt_usd.parquet", cast=str
)

# ── OECD endpoint ─────────────────────────────────────────────────────────────

FLOW_REF = "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1"
BASE_URL  = f"https://sdmx.oecd.org/public/rest/data/{FLOW_REF}"

# ── Fetch helper ──────────────────────────────────────────────────────────────

def fetch_ltdebt_usd(
    countries: str = LT_DEBT_COUNTRIES,
    start_period: str = START_PERIOD,
    end_period: str = END_PERIOD,
) -> pd.DataFrame:
    """Fetch long-term debt market value (USD) for the given countries."""
    # F3.L.USD = long-term bonds, USD-denominated (OECD handles FX conversion)
    key = (
        f"A.N.{countries}.W.S1.S1.N.L.LE.F3.L.USD._T.S.V.N.T0720._Z"
        f"?startPeriod={start_period}&endPeriod={end_period}"
        f"&dimensionAtObservation=AllDimensions"
        f"&format=csvfilewithlabels"
    )
    url = f"{BASE_URL}/{key}"
    headers = {
        "Accept":     "text/csv",
        "User-Agent": "python-requests (OECD SDMX client)",
    }
    print(f"Requesting URL:\n{url}\n")
    r = requests.get(url, headers=headers, timeout=120)
    if r.status_code in (403, 429):
        raise RuntimeError(f"Rate-limited (HTTP {r.status_code})")
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def pull_ltdebt_usd(
    countries: str = LT_DEBT_COUNTRIES,
    start_period: str = START_PERIOD,
    end_period: str = END_PERIOD,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Pull and tidy long-term debt market value.

    Returns
    -------
    pd.DataFrame with columns: country, year, mv_usd_mn
    """
    for attempt in range(max_retries):
        try:
            raw = fetch_ltdebt_usd(countries, start_period, end_period)
            print(f"OK, rows={len(raw)}")
            break
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {exc}, retrying…")
            time.sleep(5)

    # Column names vary slightly with csvfilewithlabels format
    area_col = next(c for c in raw.columns if c in ("REF_AREA", "Reference area"))
    time_col = next(c for c in raw.columns if c in ("TIME_PERIOD", "Time period"))
    val_col  = next(c for c in raw.columns if c in ("OBS_VALUE", "Observation value"))

    df = raw[[area_col, time_col, val_col]].copy()
    df.columns = ["country", "year", "mv_usd_mn"]
    df["year"]      = df["year"].astype(int)
    df["mv_usd_mn"] = pd.to_numeric(df["mv_usd_mn"], errors="coerce")
    df = (
        df.dropna(subset=["mv_usd_mn"])
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )

    print("\nLong-term debt market value (USD, millions):")
    print(df.pivot(index="year", columns="country", values="mv_usd_mn").round(0).to_string())

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pull_ltdebt_usd()

    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / LT_DEBT_USD_FILE
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
