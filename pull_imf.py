# -*- coding: utf-8 -*-
"""
IMF PIP (formerly CPIS) pull script for Table C1 issuer set + currency denomination aggregates.

Requires:
  pip install sdmx1 pandas pyarrow

Outputs:
  data/pip_bilateral_positions.parquet
  data/pip_currency_aggregates.parquet
  plus CSV versions for convenience
"""

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd
import sdmx

# Silence the harmless parser warnings you saw
warnings.filterwarnings(
    "ignore",
    message=r"xml\.Reader got no structure=.*StructureSpecificData"
)

IMF = sdmx.Client("IMF_DATA")

# ----------------------------
# Hard-won "known good" codes from YOUR run
# ----------------------------
ACCOUNTING_ENTRY_ASSETS = "A"
SECTOR_TOTAL = "S1"
CP_SECTOR_TOTAL = "S1"
FREQ_ANNUAL = "A"

# “positions” indicators you already verified
POS_INDICATORS = {
    "ST_DEBT": "P_F3_S_P_USD",
    "LT_DEBT": "P_F3_L_P_USD",
    "EQUITY":  "P_F51_P_USD",
}

# Denomination indicators pattern you already verified exist for many currencies
DIC_TEMPLATE = {
    "ST_DEBT": "P_F3_S_DIC_{cur}_P_USD",
    "LT_DEBT": "P_F3_L_DIC_{cur}_P_USD",
    "EQUITY":  "P_F51_DIC_{cur}_P_USD",
}

# The five currencies mentioned in your excerpt
BASE_CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF"]

# Paper's OFC investor list (drop as investors)
OFC_INVESTORS = {"BMU", "CYM", "GGY", "IRL", "IMN", "JEY", "LUX", "ANT"}  # Netherlands Antilles = ANT

# ----------------------------
# Table C1 issuer list (ISO3) + start year
# ----------------------------
ISSUER_START_YEAR: Dict[str, int] = {
    # Developed: North America
    "CAN": 2003,
    "USA": 2003,
    # Developed: Europe
    "AUT": 2003,
    "BEL": 2003,
    "DNK": 2003,
    "FIN": 2003,
    "FRA": 2003,
    "DEU": 2003,
    "ISR": 2003,
    "ITA": 2003,
    "NLD": 2003,
    "NOR": 2003,
    "PRT": 2003,
    "ESP": 2003,
    "SWE": 2003,
    "CHE": 2003,
    "GBR": 2003,
    # Developed: Pacific
    "AUS": 2003,
    "HKG": 2003,
    "JPN": 2003,
    "NZL": 2003,
    "SGP": 2003,
    # Emerging
    "BRA": 2003,
    "CHN": 2015,
    "COL": 2007,
    "CZE": 2003,
    "GRC": 2003,
    "HUN": 2003,
    "IND": 2004,
    "MYS": 2005,
    "MEX": 2003,
    "PHL": 2009,
    "POL": 2003,
    "RUS": 2004,
    "ZAF": 2003,
    "KOR": 2003,
    "THA": 2003,
}

ISSUERS = sorted(ISSUER_START_YEAR.keys())

# ----------------------------
# Optional: issuer -> local currency mapping (for local vs foreign later)
# (Only needed for the currency-cap allocation step.)
# You can expand this, but the allocator will gracefully treat unknowns as foreign.
# ----------------------------
EURO_AREA = {
    "AUT","BEL","FIN","FRA","DEU","ITA","NLD","PRT","ESP","GRC"
    # (You can add the rest if you later include more issuers)
}
ISSUER_TO_LOCAL_CCY = {
    **{c: "EUR" for c in EURO_AREA},
    "USA": "USD",
    "GBR": "GBP",
    "CHE": "CHF",
    "JPN": "JPY",
    "CAN": "CAD",
    "AUS": "AUD",
    "NZL": "NZD",
    "DNK": "DKK",
    "NOR": "NOK",
    "SWE": "SEK",
    "ISR": "ILS",
    "HKG": "HKD",
    "SGP": "SGD",
    "BRA": "BRL",
    "CHN": "CNY",
    "COL": "COP",
    "CZE": "CZK",
    "HUN": "HUF",
    "IND": "INR",
    "MYS": "MYR",
    "MEX": "MXN",
    "PHL": "PHP",
    "POL": "PLN",
    "RUS": "RUB",
    "ZAF": "ZAR",
    "KOR": "KRW",
    "THA": "THB",
}

# ----------------------------
# SDMX helpers
# ----------------------------
def pip_key(country="", accounting_entry="", indicator="", sector="", counterpart_sector="", counterpart_country="", frequency="A") -> str:
    # COUNTRY.ACCOUNTING_ENTRY.INDICATOR.SECTOR.COUNTERPART_SECTOR.COUNTERPART_COUNTRY.FREQUENCY
    return ".".join([country, accounting_entry, indicator, sector, counterpart_sector, counterpart_country, frequency])


def fetch_data(dataset: str, key: str, params: Optional[Dict[str, str]] = None, retries: int = 6):
    params = params or {}
    last_err = None
    for i in range(retries):
        try:
            return IMF.data(dataset, key=key, params=params)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Failed after {retries} tries for key={key}. Last error: {last_err}")


def to_long_df(msg) -> pd.DataFrame:
    df = sdmx.to_pandas(msg).reset_index()
    if "value" not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 1:
            df = df.rename(columns={num_cols[0]: "value"})
    return df


def get_pip_flow():
    return IMF.dataflow("PIP")


def get_cl_pip_indicator() -> pd.Series:
    flow = get_pip_flow()
    if "CL_PIP_INDICATOR" not in flow.codelist:
        raise RuntimeError("CL_PIP_INDICATOR not present in PIP flow; cannot validate indicator codes.")
    s = sdmx.to_pandas(flow.codelist["CL_PIP_INDICATOR"])
    if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
        s = s.iloc[:, 0]
    return s


def scale_to_usd(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMF returns 'value' plus 'SCALE'. Sometimes SCALE is blank.
    Convert value to USD units as: value_usd = value * 10^SCALE.
    If SCALE is missing/blank, assume 0.
    """
    out = df.copy()

    # value sometimes comes in as string depending on parser; coerce
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if "SCALE" in out.columns:
        scale = pd.to_numeric(out["SCALE"], errors="coerce").fillna(0).astype(int)
        out["SCALE"] = scale
        out["value_usd"] = out["value"] * (10 ** out["SCALE"])
    else:
        out["value_usd"] = out["value"]

    return out

# ----------------------------
# Pull 1: Bilateral positions for a single issuer (ALL investors)
# ----------------------------
def pull_bilateral_for_issuer(issuer: str, start_year: int, end_year: int) -> pd.DataFrame:
    ind_sel = "+".join(POS_INDICATORS.values())

    # COUNTRY wildcard => all investors
    key = pip_key(
        country="",
        accounting_entry=ACCOUNTING_ENTRY_ASSETS,
        indicator=ind_sel,
        sector=SECTOR_TOTAL,
        counterpart_sector=CP_SECTOR_TOTAL,
        counterpart_country=issuer,
        frequency=FREQ_ANNUAL,
    )
    msg = fetch_data("PIP", key, params={"startPeriod": str(start_year), "endPeriod": str(end_year)})
    df = to_long_df(msg)

    # Tag issuer explicitly (it is already in COUNTERPART_COUNTRY, but keep a stable name)
    df["issuer"] = issuer

    # Map to asset class
    inv_map = {v: k for k, v in POS_INDICATORS.items()}
    df["asset_class"] = df["INDICATOR"].map(inv_map).fillna(df["INDICATOR"])

    return df


# ----------------------------
# Pull 2: Currency aggregates (COUNTERPART_COUNTRY = G001)
# ----------------------------
def build_currency_indicator_list(
    currencies: List[str],
    cl_ind: pd.Series,
) -> List[Tuple[str, str, str]]:
    """
    Returns list of tuples (asset_class, currency, indicator_code),
    keeping only codes that exist in CL_PIP_INDICATOR.
    """
    rows = []
    for asset, tmpl in DIC_TEMPLATE.items():
        for cur in currencies:
            code = tmpl.format(cur=cur)
            if code in cl_ind.index:
                rows.append((asset, cur, code))
    return rows


def pull_currency_aggregates_all_investors(
    currencies: List[str],
    start_year: int,
    end_year: int,
    year_chunks: List[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Pulls currency denomination aggregates across issuers:
      COUNTRY = all investors (wildcard)
      COUNTERPART_COUNTRY = G001
    """
    cl_ind = get_cl_pip_indicator()

    rows = build_currency_indicator_list(currencies, cl_ind)
    if not rows:
        raise RuntimeError("No DIC indicators found for the chosen currency list. Expand currencies or verify CL_PIP_INDICATOR.")

    ind_sel = "+".join([r[2] for r in rows])
    rev = {code: (asset, cur) for asset, cur, code in rows}

    # Default chunking (helps avoid huge single responses)
    if year_chunks is None:
        year_chunks = [(start_year, min(2009, end_year)), (2010, min(2015, end_year)), (2016, end_year)]
        year_chunks = [yc for yc in year_chunks if yc[0] <= yc[1]]

    out = []
    for ys, ye in year_chunks:
        key = pip_key(
            country="",
            accounting_entry=ACCOUNTING_ENTRY_ASSETS,
            indicator=ind_sel,
            sector=SECTOR_TOTAL,
            counterpart_sector=CP_SECTOR_TOTAL,
            counterpart_country="G001",
            frequency=FREQ_ANNUAL,
        )
        msg = fetch_data("PIP", key, params={"startPeriod": str(ys), "endPeriod": str(ye)})
        df = to_long_df(msg)
        df["asset_class"] = df["INDICATOR"].map(lambda x: rev.get(x, ("", ""))[0])
        df["currency"] = df["INDICATOR"].map(lambda x: rev.get(x, ("", ""))[1])
        out.append(df)

    return pd.concat(out, ignore_index=True)


# ----------------------------
# Currency-cap allocator (general)
# ----------------------------
def allocate_local_foreign_by_currency_caps(
    bilateral: pd.DataFrame,
    ccy_agg: pd.DataFrame,
    issuer_to_local_ccy: Dict[str, str],
    investor_col: str = "COUNTRY",
    issuer_col: str = "COUNTERPART_COUNTRY",
    year_col: str = "TIME_PERIOD",
    value_col: str = "value_usd",
) -> pd.DataFrame:
    """
    For each issuer, treat holdings as 'local currency candidate' in issuer's currency bucket.
    For each (investor, asset_class, year, currency), cap candidates at ccy_agg totals and scale down proportionally.
    """
    df = bilateral.copy()
    df["local_currency"] = df[issuer_col].map(issuer_to_local_ccy)
    df["value_total"] = df[value_col]
    df["value_local_candidate"] = df["value_total"].where(df["local_currency"].notna(), 0.0)

    cand = (
        df[df["local_currency"].notna()]
        .groupby([investor_col, "asset_class", year_col, "local_currency"], as_index=False)["value_local_candidate"]
        .sum()
        .rename(columns={"local_currency": "currency", "value_local_candidate": "cand_local_sum"})
    )

    agg = (
        ccy_agg.groupby([investor_col, "asset_class", year_col, "currency"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "agg_ccy_total"})
    )

    caps = cand.merge(agg, on=[investor_col, "asset_class", year_col, "currency"], how="left")
    caps["cap_ratio"] = (caps["agg_ccy_total"] / caps["cand_local_sum"]).fillna(1.0).clip(upper=1.0)

    df = df.merge(
        caps[[investor_col, "asset_class", year_col, "currency", "cap_ratio"]],
        left_on=[investor_col, "asset_class", year_col, "local_currency"],
        right_on=[investor_col, "asset_class", year_col, "currency"],
        how="left",
    )
    df["cap_ratio"] = df["cap_ratio"].fillna(1.0)

    df["value_local"] = df["value_total"].where(df["local_currency"].notna(), 0.0) * df["cap_ratio"]
    df["value_foreign"] = df["value_total"] - df["value_local"]

    keep = [
        investor_col, issuer_col, "asset_class", year_col,
        "local_currency", "value_total", "value_local", "value_foreign"
    ]
    return df[keep]


# ----------------------------
# Main runner
# ----------------------------
def main(out_dir: str = "data", end_year: int = 2020):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Pull bilateral for the Table C1 issuers only (ALL investors)
    bilat_parts = []
    for issuer in ISSUERS:
        sy = ISSUER_START_YEAR[issuer]
        print(f"Pulling bilateral for issuer={issuer}, years={sy}-{end_year} ...")
        df_i = pull_bilateral_for_issuer(issuer, sy, end_year)
        bilat_parts.append(df_i)

    df_bilat = pd.concat(bilat_parts, ignore_index=True)

    # Drop OFC investors (paper step)
    if "COUNTRY" in df_bilat.columns:
        before = len(df_bilat)
        df_bilat = df_bilat[~df_bilat["COUNTRY"].isin(OFC_INVESTORS)].copy()
        print(f"Dropped OFC investors: {before - len(df_bilat):,} rows removed.")

    # Scale
    df_bilat = scale_to_usd(df_bilat)

    # Save bilateral
    bilat_parquet = os.path.join(out_dir, "pip_bilateral_positions.parquet")
    bilat_csv = os.path.join(out_dir, "pip_bilateral_positions.csv")
    df_bilat.to_parquet(bilat_parquet, index=False)
    df_bilat.to_csv(bilat_csv, index=False)
    print(f"Saved bilateral to:\n  {bilat_parquet}\n  {bilat_csv}")

    # 2) Pull currency aggregates for all investors (COUNTERPART=G001)
    # Build a currency list that includes BASE + any issuer local currencies that appear in DIC indicators
    cl_ind = get_cl_pip_indicator()
    desired_currencies = sorted(set(BASE_CURRENCIES + list(set(ISSUER_TO_LOCAL_CCY.values()))))

    # Keep only currencies that actually have any DIC indicator codes in CL_PIP_INDICATOR
    valid_rows = build_currency_indicator_list(desired_currencies, cl_ind)
    valid_currencies = sorted(set([cur for _, cur, _ in valid_rows]))

    print(f"Pulling currency aggregates for currencies={valid_currencies} ...")
    df_ccy = pull_currency_aggregates_all_investors(
        currencies=valid_currencies,
        start_year=2003,
        end_year=end_year,
    )
    df_ccy = scale_to_usd(df_ccy)

    ccy_parquet = os.path.join(out_dir, "pip_currency_aggregates.parquet")
    ccy_csv = os.path.join(out_dir, "pip_currency_aggregates.csv")
    df_ccy.to_parquet(ccy_parquet, index=False)
    df_ccy.to_csv(ccy_csv, index=False)
    print(f"Saved currency aggregates to:\n  {ccy_parquet}\n  {ccy_csv}")

    # 3) OPTIONAL: local-vs-foreign allocation with caps (paper logic)
    alloc = allocate_local_foreign_by_currency_caps(
        bilateral=df_bilat,
        ccy_agg=df_ccy,
        issuer_to_local_ccy=ISSUER_TO_LOCAL_CCY,
        value_col="value_usd",
    )
    alloc_parquet = os.path.join(out_dir, "pip_local_foreign_allocated.parquet")
    alloc_csv = os.path.join(out_dir, "pip_local_foreign_allocated.csv")
    alloc.to_parquet(alloc_parquet, index=False)
    alloc.to_csv(alloc_csv, index=False)
    print(f"Saved local/foreign allocation to:\n  {alloc_parquet}\n  {alloc_csv}")


if __name__ == "__main__":
    main(out_dir="data", end_year=2020)