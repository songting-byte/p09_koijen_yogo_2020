"""figure_1_xaxis.py
===================
Pull short-term debt market value in USD from the OECD API and compute
the x-axis (log market-value ratio relative to the US, demeaned) used
in Figure 1.

Approach:
  DF_T720R_A supports UNIT_MEASURE = USD, so OECD handles the
  local-currency → USD conversion internally. No separate exchange-rate
  pull is needed.

Key difference from pull_oecd.py:
  XDC  →  USD  in the dimension string.

Configuration (via settings.py / .env / CLI):
  DATA_DIR            where to write output files (default: _data/)
  OECD_START_PERIOD   start year  (default: "2003")
  OECD_END_PERIOD     end year    (default: "2020")
  FIGURE1_BASE_URL    full SDMX key URL (optional override)
  FIGURE1_MV_FILE     parquet/csv filename for raw USD market values
  FIGURE1_XAXIS_FILE  csv filename for the demeaned x-axis series

Dependencies: pip install requests pandas pyarrow python-decouple
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

from settings import config

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR          = config("DATA_DIR")
START_PERIOD      = config("OECD_START_PERIOD", default="2003", cast=str)
END_PERIOD        = config("OECD_END_PERIOD",   default="2020", cast=str)

COUNTRIES = (
    "USA+JPN+CHE+GBR+"
    "AUT+BEL+FIN+FRA+DEU+ITA+NLD+PRT+ESP+GRC"
)

_DEFAULT_BASE_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1/"
    f"A.N.{COUNTRIES}.W.S1.S1.N.L.LE.F3.S.USD._T.S.V.N.T0720._Z"
    f"?startPeriod={START_PERIOD}&endPeriod={END_PERIOD}"
)

FIGURE1_BASE_URL  = config("FIGURE1_BASE_URL",  default=_DEFAULT_BASE_URL, cast=str)
FIGURE1_MV_FILE   = config("FIGURE1_MV_FILE",   default="figure1_stdebt_usd.parquet", cast=str)
FIGURE1_XAXIS_FILE = config("FIGURE1_XAXIS_FILE", default="figure1_xaxis.parquet", cast=str)

EURO_AREA = {"AUT", "BEL", "FIN", "FRA", "DEU",
             "ITA", "NLD", "PRT", "ESP", "GRC"}

LABEL_MAP = {
    "EA":  "Euro",
    "JPN": "Japan",
    "CHE": "Switzerland",
    "GBR": "United Kingdom",
}

# ── SDMX helpers (mirrors pull_oecd.py) ──────────────────────────────────────

def _parse_base_url(base_url: str):
    parsed = urlparse(base_url)
    path_parts = parsed.path.split("/data/")
    if len(path_parts) != 2:
        raise ValueError("Base URL must contain a '/data/' segment")
    flow_ref, key = path_parts[1].split("/", 1)
    key_template = key.split("?")[0]
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    return flow_ref, key_template, params


def _get_json(session: requests.Session, url: str, params: dict) -> dict:
    headers = {"Accept": "application/vnd.sdmx.data+json;version=2.0"}
    r = session.get(url, params=params, headers=headers, timeout=120)
    r.raise_for_status()
    return r.json()


def _sdmx_json_to_tidy(payload: dict) -> pd.DataFrame:
    data_block   = payload.get("data", payload)
    structures   = data_block.get("structures", [])
    datasets     = data_block.get("dataSets", [])
    if not structures or not datasets:
        return pd.DataFrame()

    observations = datasets[0].get("observations", {})
    obs_dims     = structures[0].get("dimensions", {}).get("observation", [])
    dim_ids      = [d["id"] for d in obs_dims]
    dim_values   = {
        d["id"]: [{"id": v.get("id"), "name": v.get("name")}
                  for v in d.get("values", [])]
        for d in obs_dims
    }

    rows = []
    for key, obs in observations.items():
        indices = [int(i) for i in key.split(":")]
        row = {}
        for dim_id, idx in zip(dim_ids, indices):
            vals = dim_values.get(dim_id, [])
            v    = vals[idx] if idx < len(vals) else {}
            row[dim_id] = v.get("id")
        row["value"] = obs[0] if isinstance(obs, list) and obs else obs
        rows.append(row)

    return pd.DataFrame(rows)


# ── Main logic ────────────────────────────────────────────────────────────────

def pull_and_compute_xaxis(
    base_url: str = FIGURE1_BASE_URL,
    start_period: str = START_PERIOD,
    end_period: str = END_PERIOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull USD-denominated short-term debt and compute the demeaned x-axis.

    Returns
    -------
    df_x : pd.DataFrame
        Columns: country, label, year, x_raw, x
    df_mv : pd.DataFrame
        Columns: country, year, mv_usd_mn
    """
    flow_ref, key, base_params = _parse_base_url(base_url)
    params = {
        **base_params,
        "startPeriod":            start_period,
        "endPeriod":              end_period,
        "dimensionAtObservation": "AllDimensions",
        "format":                 "jsondata",
    }

    url = f"https://sdmx.oecd.org/public/rest/data/{flow_ref}/{key}"
    print(f"Requesting URL:\n{url}\n")

    with requests.Session() as session:
        payload = _get_json(session, url, params)

    df_raw = _sdmx_json_to_tidy(payload)
    print(f"Raw rows: {len(df_raw)}")

    # ── Tidy ─────────────────────────────────────────────────────────────────
    df = df_raw[["REF_AREA", "TIME_PERIOD", "value"]].copy()
    df.columns = ["country", "year", "mv_usd_mn"]
    df["year"]      = pd.to_numeric(df["year"],      errors="coerce").astype("Int64")
    df["mv_usd_mn"] = pd.to_numeric(df["mv_usd_mn"], errors="coerce")
    df = df.dropna()

    print(f"Countries: {sorted(df['country'].unique())}")
    print(df.pivot(index="year", columns="country", values="mv_usd_mn").to_string())

    # ── Euro area aggregate ───────────────────────────────────────────────────
    euro_sum = (
        df[df["country"].isin(EURO_AREA)]
        .groupby("year", as_index=False)["mv_usd_mn"]
        .sum()
        .assign(country="EA")
    )
    df = (
        pd.concat([df[~df["country"].isin(EURO_AREA)], euro_sum], ignore_index=True)
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )

    # ── X-axis: log ratio vs US, demeaned ────────────────────────────────────
    # USD pricing means no exchange-rate adjustment is needed.
    # We use market value ≈ face value (P ≈ 1 for short maturities);
    # a later correction with short-rate data can refine this.
    us = df[df["country"] == "USA"].set_index("year")["mv_usd_mn"]

    results = []
    for country, label in LABEL_MAP.items():
        sub = df[df["country"] == country].set_index("year")["mv_usd_mn"]
        common_years = sub.index.intersection(us.index)

        rows = [
            {
                "country": country,
                "label":   label,
                "year":    int(yr),
                "x_raw":   np.log(sub[yr]) - np.log(us[yr]),
            }
            for yr in common_years
        ]
        df_c = pd.DataFrame(rows)
        df_c["x"] = df_c["x_raw"] - df_c["x_raw"].mean()
        results.append(df_c)

    df_x = pd.concat(results, ignore_index=True)

    print("\n── X-axis (demeaned) ──")
    print(df_x.pivot(index="year", columns="label", values="x").round(4).to_string())

    return df_x, df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_x, df_mv = pull_and_compute_xaxis()

    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    mv_path    = output_dir / FIGURE1_MV_FILE
    xaxis_path = output_dir / FIGURE1_XAXIS_FILE

    # Save market-value data as parquet (consistent with pull_oecd.py)
    df_mv.to_parquet(mv_path, index=False)
    df_x.to_parquet(xaxis_path, index=False)

    print(f"\nSaved:\n  {mv_path}\n  {xaxis_path}")