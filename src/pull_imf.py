# -*- coding: utf-8 -*-
"""
IMF PIP (formerly CPIS) pull script for Table C1 issuer set + currency denomination aggregates.

Requires:
  pip install sdmx1 pandas pyarrow

Outputs:
  data/pip_bilateral_positions.parquet
    data/pip_bilateral_positions_reserve.parquet
  data/pip_currency_aggregates.parquet
  plus CSV versions for convenience
"""

import os
import csv
import re
import time
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple

import pandas as pd
import sdmx

from settings import config as _settings_config

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
}

# Denomination indicators pattern you already verified exist for many currencies
DIC_TEMPLATE = {
    "ST_DEBT": "P_F3_S_DIC_{cur}_P_USD",
    "LT_DEBT": "P_F3_L_DIC_{cur}_P_USD",
}
DEBT_DIC_TOTAL_TEMPLATE = "P_F3_DIC_{cur}_P_USD"

# The IMF aggregate denomination buckets used in the data instruction.
# Local-currency caps must be computed in these reported buckets only.
BASE_CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "OTHC"]
CORE_REPORTED_CURRENCIES = {"USD", "EUR", "JPY", "GBP", "CHF"}
# Keep a uniform global relaxation to mitigate under-coverage of local-currency
# buckets in IMF currency aggregates while preserving cap-based allocation.
CAP_RELAX_MULT = float(os.getenv("PIP_CAP_RELAX_MULT", "1.0"))
DISABLE_CAP = os.getenv("PIP_DISABLE_CAP", "0") == "1"
RESERVE_SECTOR_CODE_FILE = os.path.join("_data", "imf_pip_reserve_sector_code.txt")
RESERVE_SECTOR_MODE = os.getenv("PIP_RESERVE_SECTOR_MODE", "broad").strip().lower()

# Paper's OFC investor list (drop as investors)
OFC_INVESTORS = {"BMU", "CYM", "GGY", "IRL", "IMN", "JEY", "LUX", "ANT"}  # Netherlands Antilles = ANT
TIC_FALLBACK_PATHS = [
    os.path.join("data", "us_tic_holdings.parquet"),
    os.path.join("data", "us_tic_holdings.csv"),
    os.path.join("_data", "us_tic_holdings.parquet"),
    os.path.join("_data", "us_tic_holdings.csv"),
]

US_TREASURY_DIR_CANDIDATES = [
    os.path.join("manual added data", "feds_note_nationalitydata_to2024"),
    os.path.join("src", "feds_note_nationalitydata_to2024"),
    os.path.join("data", "feds_note_nationalitydata_to2024"),
    os.path.join("_data", "feds_note_nationalitydata_to2024"),
]

US_TREASURY_REQUIRED_FILES = {
    "corporate": "corporate_bonds_data_table.csv",
    "government": "gov_bonds_data_table.csv",
    "equity": "common_stock_data_table.csv",
}

US_SHC_APPENDIX_FILES = {
    "st_debt": "shc_app07_2020.csv",
    "lt_debt": "shc_app08_2020.csv",
}

US_SHC_DIR_CANDIDATES = [
    "manual added data",
    os.path.join("src", "shca2020_appendix"),
    "data_manual",
]

RESTATEMENT_MATRICES_PATH = os.path.join("manual added data", "Restatement_Matrices.dta")

# Paper methodology assignment for restatement matrices.
EURO_AREA_INVESTORS = {"AUT", "BEL", "FIN", "FRA", "DEU", "GRC", "ITA", "NLD", "PRT", "ESP"}
FUND_HOLDINGS_INVESTORS = EURO_AREA_INVESTORS.union({"AUS", "CAN", "DNK", "SWE", "CHE", "GBR"})
ENHANCED_FUND_HOLDINGS_INVESTORS = {"NOR", "USA"}

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

ISO3_TO_US_TREASURY_NAME = {
    "AUT": "AUSTRIA",
    "AUS": "AUSTRALIA",
    "BEL": "BELGIUM",
    "BRA": "BRAZIL",
    "CAN": "CANADA",
    "CHE": "SWITZERLAND",
    "CHN": "CHINA",
    "COL": "COLOMBIA",
    "CZE": "CZECH REPUBLIC",
    "DEU": "GERMANY",
    "DNK": "DENMARK",
    "ESP": "SPAIN",
    "FIN": "FINLAND",
    "FRA": "FRANCE",
    "GBR": "UNITED KINGDOM",
    "GRC": "GREECE",
    "HKG": "HONG KONG",
    "HUN": "HUNGARY",
    "IND": "INDIA",
    "ISR": "ISRAEL",
    "ITA": "ITALY",
    "JPN": "JAPAN",
    "KOR": "KOREA SOUTH",
    "MEX": "MEXICO",
    "MYS": "MALAYSIA",
    "NLD": "NETHERLANDS",
    "NOR": "NORWAY",
    "NZL": "NEW ZEALAND",
    "PHL": "PHILIPPINES",
    "POL": "POLAND",
    "PRT": "PORTUGAL",
    "RUS": "RUSSIA",
    "SGP": "SINGAPORE",
    "SWE": "SWEDEN",
    "THA": "THAILAND",
    "USA": "UNITED STATES",
    "ZAF": "SOUTH AFRICA",
}

US_TREASURY_NAME_TO_ISO3 = {
    v: k for k, v in ISO3_TO_US_TREASURY_NAME.items()
}

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


def map_local_ccy_to_imf_bucket(local_ccy: Optional[str]) -> Optional[str]:
    """
    IMF PIP currency aggregates are used as cap buckets.
    Keep the five explicitly reported currencies and map all other known
    local currencies to OTHC (other currencies) per instruction.
    """
    if local_ccy is None or pd.isna(local_ccy):
        return None
    c = str(local_ccy)
    return c if c in CORE_REPORTED_CURRENCIES else "OTHC"

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
    """Return the PIP indicator codelist as a Series.

    The IMF API has changed the codelist key over time.  Try known names in
    order; if none are present, build a permissive Series from the hard-coded
    indicator names used throughout this script so callers don't crash.
    """
    flow = get_pip_flow()

    # Try known key names in order of preference
    candidates = ["CL_PIP_INDICATOR", "CL_INDICATOR", "CL_PIP_IND"]
    for key in candidates:
        if key in flow.codelist:
            s = sdmx.to_pandas(flow.codelist[key])
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                s = s.iloc[:, 0]
            if isinstance(s, pd.Series) and not s.empty:
                return s

    # Fallback: search for any codelist whose key contains "INDICATOR"
    indicator_keys = [k for k in flow.codelist if "INDICATOR" in str(k).upper()]
    if indicator_keys:
        key = indicator_keys[0]
        print(f"  [pull_imf] CL_PIP_INDICATOR not found; using '{key}' instead.")
        s = sdmx.to_pandas(flow.codelist[key])
        if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
            s = s.iloc[:, 0]
        if isinstance(s, pd.Series) and not s.empty:
            return s

    # Last resort: return a permissive Series containing all known indicator
    # codes so callers can proceed without codelist validation.
    print(
        "  [pull_imf] WARNING: No indicator codelist found in PIP flow. "
        f"Available codelists: {list(flow.codelist)}. "
        "Proceeding with permissive indicator set (no validation)."
    )
    known = [
        "P_F3_S_P_USD", "P_F3_L_P_USD", "P_F5_P_USD", "P_F51_P_USD",
    ]
    # Add DIC templates for each base currency
    for cur in BASE_CURRENCIES:
        known += [
            f"P_F3_S_DIC_{cur}_P_USD",
            f"P_F3_L_DIC_{cur}_P_USD",
            f"P_F3_DIC_{cur}_P_USD",
            f"P_F5_DIC_{cur}_P_USD",
            f"P_F51_DIC_{cur}_P_USD",
        ]
    return pd.Series(known, index=known)


def load_reserve_sector_code_from_txt(txt_path: str = RESERVE_SECTOR_CODE_FILE) -> Optional[str]:
    """
    Read reserve-sector code from an external txt file.
    Expected line format: reserve_sector_code=<CODE>
    """
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue
            if line.lower().startswith("reserve_sector_code="):
                code = line.split("=", 1)[1].strip()
                return code or None

    return None


def load_reserve_sector_codes_from_txt(txt_path: str = RESERVE_SECTOR_CODE_FILE) -> List[str]:
    """
    Read reserve-sector codes from an external txt file.

    Accepted formats:
      reserve_sector_code=S121
      reserve_sector_codes=S121,S122
    """
    if not os.path.exists(txt_path):
        return []

    out: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith("#"):
                continue

            if line.lower().startswith("reserve_sector_codes="):
                rhs = line.split("=", 1)[1].strip()
                vals = [x.strip().upper() for x in rhs.split(",") if x.strip()]
                out.extend(vals)
            elif line.lower().startswith("reserve_sector_code="):
                rhs = line.split("=", 1)[1].strip().upper()
                if rhs:
                    out.append(rhs)

    dedup = []
    seen = set()
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def _safe_codelist_to_series(flow, key: str) -> pd.Series:
    if key not in flow.codelist:
        return pd.Series(dtype="object")
    s = sdmx.to_pandas(flow.codelist[key])
    if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
        s = s.iloc[:, 0]
    if not isinstance(s, pd.Series):
        return pd.Series(dtype="object")
    return s


def discover_reserve_sector_candidates() -> List[str]:
    """
    Discover plausible reserve-sector investor codes from PIP codelists.

    Priority is central-bank-like sectors first; keyword matches are appended.
    """
    flow = get_pip_flow()
    sector_series = pd.Series(dtype="object")
    for k in ["CL_PIP_SECTOR", "CL_SECTOR", "CL_REF_SECTOR"]:
        sector_series = _safe_codelist_to_series(flow, k)
        if not sector_series.empty:
            break

    preferred = ["S121", "S122", "S1XA", "S1X", "S1KK"]
    out: List[str] = []
    for c in preferred:
        out.append(c)

    if not sector_series.empty:
        for code, label in sector_series.items():
            code_u = str(code).upper()
            label_s = str(label).lower()
            if (
                ("central bank" in label_s)
                or ("monetary" in label_s)
                or ("reserve" in label_s)
            ):
                out.append(code_u)

    # Remove total-sector aggregate to avoid accidentally pulling all investors.
    out = [c for c in out if c != SECTOR_TOTAL]

    dedup = []
    seen = set()
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def probe_reserve_sector_codes(
    sector_codes: List[str],
    equity_indicator: str,
    end_year: int,
) -> pd.DataFrame:
    """
    Probe candidate sector codes quickly and keep objective coverage diagnostics.
    """
    rows = []
    if not sector_codes:
        return pd.DataFrame(columns=["sector_code", "rows", "investors", "issuers", "value_usd"])

    sample_issuer = "USA" if "USA" in ISSUERS else ISSUERS[0]
    start_year = max(2003, int(end_year) - 2)
    for code in sector_codes:
        try:
            df = pull_bilateral_for_issuer(
                issuer=sample_issuer,
                start_year=start_year,
                end_year=end_year,
                equity_indicator=equity_indicator,
                sector_code=code,
            )
            dfx = scale_to_usd(df) if not df.empty else df
            rows.append(
                {
                    "sector_code": code,
                    "rows": int(len(dfx)),
                    "investors": int(dfx["COUNTRY"].nunique()) if (not dfx.empty and "COUNTRY" in dfx.columns) else 0,
                    "issuers": int(dfx["COUNTERPART_COUNTRY"].nunique()) if (not dfx.empty and "COUNTERPART_COUNTRY" in dfx.columns) else 0,
                    "value_usd": float(pd.to_numeric(dfx.get("value_usd", pd.Series(dtype=float)), errors="coerce").sum()) if not dfx.empty else 0.0,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "sector_code": code,
                    "rows": 0,
                    "investors": 0,
                    "issuers": 0,
                    "value_usd": 0.0,
                    "error": str(e),
                }
            )

    out = pd.DataFrame(rows)
    if "error" not in out.columns:
        out["error"] = pd.NA
    return out.sort_values(["rows", "value_usd"], ascending=False).reset_index(drop=True)


def consolidate_reserve_sector_rows(df_reserve: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate reserve pulls across multiple sector codes to avoid double counting.

    Rule:
    - Preferred source order: S121, then S122, then any other sector code.
    - For a given (investor, issuer, year, asset), keep the preferred available sector.
    - This preserves broad coverage while preventing additive overlap between sectors.
    """
    if df_reserve.empty:
        return df_reserve
    if "reserve_sector_code" not in df_reserve.columns:
        return df_reserve

    key_cols = ["COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"]
    if not set(key_cols).issubset(set(df_reserve.columns)):
        return df_reserve

    x = df_reserve.copy()
    x["value_usd"] = pd.to_numeric(x.get("value_usd", pd.Series(dtype=float)), errors="coerce")
    x["reserve_sector_code"] = x["reserve_sector_code"].astype(str).str.upper()

    # Aggregate within key+sector first.
    g = (
        x.groupby(key_cols + ["reserve_sector_code"], as_index=False)["value_usd"]
        .sum(min_count=1)
    )

    overlap = g.groupby(key_cols, as_index=False)["reserve_sector_code"].nunique()
    overlap_n = int((overlap["reserve_sector_code"] > 1).sum())

    rank_map = {"S121": 0, "S122": 1}
    g["sector_rank"] = g["reserve_sector_code"].map(rank_map).fillna(9).astype(int)
    g["abs_value"] = pd.to_numeric(g["value_usd"], errors="coerce").abs()

    g = g.sort_values(key_cols + ["sector_rank", "abs_value"], ascending=[True, True, True, True, True, False])
    kept = g.drop_duplicates(subset=key_cols, keep="first").copy()

    print(
        "Consolidated reserve sectors (anti-double-count): "
        f"input_rows={len(df_reserve):,}, key_sector_rows={len(g):,}, output_rows={len(kept):,}, "
        f"overlap_keys={overlap_n:,}"
    )

    # Keep a compact, stable schema for downstream aggregation.
    # Preserve `issuer` for backward compatibility with table notebooks.
    kept["issuer"] = kept["COUNTERPART_COUNTRY"]
    keep_cols = ["issuer"] + key_cols + ["reserve_sector_code", "value_usd"]
    return kept[keep_cols]


def resolve_equity_indicators(cl_ind: pd.Series) -> Tuple[str, str]:
    """
    Prefer total equity+fund shares (F5) for CPIS/PIP consistency.
    Fall back to common equity (F51) if F5 is unavailable.

    Returns:
      (equity_position_indicator, equity_denom_indicator_template)
    """
    pos_candidates = ["P_F5_P_USD", "P_F51_P_USD"]
    dic_candidates = ["P_F5_DIC_{cur}_P_USD", "P_F51_DIC_{cur}_P_USD"]

    pos = next((code for code in pos_candidates if code in cl_ind.index), None)
    dic = next((tmpl for tmpl in dic_candidates if tmpl.format(cur="USD") in cl_ind.index), None)

    if pos is None:
        raise RuntimeError("No equity position indicator found in CL_PIP_INDICATOR (expected F5 or F51).")
    if dic is None:
        raise RuntimeError("No equity denomination indicator template found in CL_PIP_INDICATOR (expected F5_DIC or F51_DIC).")

    if pos != "P_F5_P_USD" or dic != "P_F5_DIC_{cur}_P_USD":
        print(
            "WARNING: IMF F5 equity indicators are unavailable; "
            f"falling back to {pos} / {dic}. "
            "This is a data-availability fallback, not the preferred instruction path."
        )

    return pos, dic


def scale_to_usd(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMF PIP values are already USD-valued in this extraction.
    Keep `SCALE` for diagnostics, but do not multiply `value` again.
    """
    out = df.copy()

    # value sometimes comes in as string depending on parser; coerce
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    if "SCALE" in out.columns:
        scale = pd.to_numeric(out["SCALE"], errors="coerce").fillna(0).astype(int)
        out["SCALE"] = scale

    out["value_usd"] = out["value"]

    return out


def _find_us_treasury_dir() -> Optional[str]:
    for d in US_TREASURY_DIR_CANDIDATES:
        if not os.path.isdir(d):
            continue
        required = [os.path.join(d, fn) for fn in US_TREASURY_REQUIRED_FILES.values()]
        if all(os.path.exists(p) for p in required):
            return d
    return None


def _find_shc_appendix_file(file_name: str) -> Optional[str]:
    for d in US_SHC_DIR_CANDIDATES:
        p = os.path.join(d, file_name)
        if os.path.exists(p):
            return p
    return None


def _normalize_country_label(label: str) -> str:
    s = str(label).upper().strip()
    # SHC appendix labels often include footnote markers like "(1)".
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _country_name_to_iso3_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    # Start with the Treasury-name map, normalized.
    for iso3, name in ISO3_TO_US_TREASURY_NAME.items():
        mapping[_normalize_country_label(name)] = iso3

    # Add explicit aliases seen in SHC appendix labels.
    mapping.update(
        {
            "CHINA MAINLAND": "CHN",
            "KOREA SOUTH": "KOR",
            "CZECH REPUBLIC": "CZE",
            "HONG KONG": "HKG",
            "UNITED KINGDOM": "GBR",
        }
    )
    return mapping


def _read_shc_appendix_country_totals(csv_path: str) -> pd.DataFrame:
    """
    Parse SHC appendix CSVs (A7/A8) using raw CSV rows.

    Structure is non-tabular at the top and variable-width in notes,
    so we intentionally avoid header-based parsing.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"])

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return pd.DataFrame(columns=["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"])

    # Detect year from preamble line such as "As of End-December 2020".
    year: Optional[str] = None
    for r in rows[:25]:
        text = " ".join([str(x) for x in r if x is not None])
        m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
        if m:
            year = m.group(1)
            break
    if year is None:
        return pd.DataFrame(columns=["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"])

    name_to_iso3 = _country_name_to_iso3_map()

    # Find the start of country rows.
    header_idx: Optional[int] = None
    for i, r in enumerate(rows):
        c0 = (r[0] if len(r) > 0 else "")
        if str(c0).strip().upper() == "COUNTRY OR REGION OF ISSUER":
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame(columns=["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"])

    out_rows = []
    for r in rows[header_idx + 1 :]:
        if not r:
            continue
        raw_name = str(r[0]).strip() if len(r) > 0 else ""
        if not raw_name:
            continue

        raw_name_upper = raw_name.upper()
        # Stop before regional totals / notes block.
        if raw_name_upper in {"TOTAL", "TOTALS BY REGION:"} or raw_name_upper.startswith("TOTAL "):
            continue
        if raw_name_upper.startswith("1.") or raw_name_upper.startswith("2."):
            break

        total_cell = str(r[1]).strip() if len(r) > 1 else ""
        if (not total_cell) or ("*" in total_cell):
            continue

        val = pd.to_numeric(total_cell.replace(",", ""), errors="coerce")
        if pd.isna(val):
            continue

        iso3 = name_to_iso3.get(_normalize_country_label(raw_name))
        if iso3 is None:
            continue

        out_rows.append(
            {
                "COUNTERPART_COUNTRY": iso3,
                "TIME_PERIOD": str(year),
                "value_million_usd": float(val),
            }
        )

    if not out_rows:
        return pd.DataFrame(columns=["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"])

    return (
        pd.DataFrame(out_rows)
        .groupby(["COUNTERPART_COUNTRY", "TIME_PERIOD"], as_index=False)["value_million_usd"]
        .sum()
    )


def load_us_shc_appendix_holdings(end_year: int) -> pd.DataFrame:
    """
    Load USA bilateral holdings from SHC Appendix A7/A8 when available.

    A7 supplies short-term debt totals by issuer; A8 supplies long-term debt totals.
    These files are annual snapshots (e.g., 2020) and are used to improve strict
    replication for U.S. investor debt classes.
    """
    st_path = _find_shc_appendix_file(US_SHC_APPENDIX_FILES["st_debt"])
    lt_path = _find_shc_appendix_file(US_SHC_APPENDIX_FILES["lt_debt"])

    frames = []

    if st_path:
        st = _read_shc_appendix_country_totals(st_path)
        if not st.empty:
            st = st[st["TIME_PERIOD"].astype(str).astype(int) <= int(end_year)].copy()
            st["asset_class"] = "ST_DEBT"
            frames.append(st)

    if lt_path:
        lt = _read_shc_appendix_country_totals(lt_path)
        if not lt.empty:
            lt = lt[lt["TIME_PERIOD"].astype(str).astype(int) <= int(end_year)].copy()
            lt["asset_class"] = "LT_DEBT"
            frames.append(lt)

    if not frames:
        return pd.DataFrame(columns=["COUNTRY", "issuer", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class", "value_usd"])

    out = pd.concat(frames, ignore_index=True)
    out["COUNTRY"] = "USA"
    out["issuer"] = out["COUNTERPART_COUNTRY"]
    out["TIME_PERIOD"] = out["TIME_PERIOD"].astype(str)
    out["value_usd"] = pd.to_numeric(out["value_million_usd"], errors="coerce").fillna(0.0) * 1_000_000.0

    out = out[["COUNTRY", "issuer", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class", "value_usd"]].copy()
    return (
        out.groupby(["COUNTRY", "issuer", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"], as_index=False)["value_usd"]
        .sum()
    )


def _read_us_treasury_nationality_long(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    val_cols = [c for c in raw.columns if c.startswith("market_value_nationality_")]
    if not val_cols:
        return pd.DataFrame(columns=["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"])

    keep_cols = ["cntry_name"] + val_cols
    raw = raw[keep_cols].copy()
    raw["cntry_name"] = raw["cntry_name"].astype(str).str.strip().str.upper()

    long = raw.melt(id_vars=["cntry_name"], value_vars=val_cols, var_name="series", value_name="value_million_usd")
    long["TIME_PERIOD"] = long["series"].str.extract(r"(\d{4})$")
    long["value_million_usd"] = pd.to_numeric(long["value_million_usd"], errors="coerce")
    long = long.dropna(subset=["TIME_PERIOD", "value_million_usd"]).copy()
    long["COUNTERPART_COUNTRY"] = long["cntry_name"].map(US_TREASURY_NAME_TO_ISO3)
    long = long.dropna(subset=["COUNTERPART_COUNTRY"]).copy()

    return (
        long.groupby(["COUNTERPART_COUNTRY", "TIME_PERIOD"], as_index=False)["value_million_usd"]
        .sum()
    )


def load_us_treasury_holdings(end_year: int) -> pd.DataFrame:
    """
    Load U.S. Treasury nationality estimates (Fed note tables) and return
        USA investor bilateral holdings.

        Base source:
            - Fed-note nationality tables for LT_DEBT and EQUITY.

        Optional strict-replication overlays:
            - SHC appendix A7 for ST_DEBT.
            - SHC appendix A8 for LT_DEBT (takes precedence for overlapping keys).
    """
    base = _find_us_treasury_dir()
    if base is None:
        out = pd.DataFrame(columns=["COUNTRY", "issuer", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class", "value_usd"])
    else:
        corp = _read_us_treasury_nationality_long(os.path.join(base, US_TREASURY_REQUIRED_FILES["corporate"]))
        govt = _read_us_treasury_nationality_long(os.path.join(base, US_TREASURY_REQUIRED_FILES["government"]))
        eq = _read_us_treasury_nationality_long(os.path.join(base, US_TREASURY_REQUIRED_FILES["equity"]))

        debt = corp.merge(govt, on=["COUNTERPART_COUNTRY", "TIME_PERIOD"], how="outer", suffixes=("_corp", "_gov"))
        debt["value_million_usd"] = debt["value_million_usd_corp"].fillna(0.0) + debt["value_million_usd_gov"].fillna(0.0)
        debt = debt[["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"]].copy()
        debt["asset_class"] = "LT_DEBT"

        eq = eq[["COUNTERPART_COUNTRY", "TIME_PERIOD", "value_million_usd"]].copy()
        eq["asset_class"] = "EQUITY"

        out = pd.concat([debt, eq], ignore_index=True)
        out["TIME_PERIOD"] = out["TIME_PERIOD"].astype(str)
        out = out[out["TIME_PERIOD"].str.fullmatch(r"\d{4}", na=False)].copy()
        out = out[out["TIME_PERIOD"].astype(int) <= int(end_year)].copy()
        out["COUNTRY"] = "USA"
        out["issuer"] = out["COUNTERPART_COUNTRY"]
        out["value_usd"] = pd.to_numeric(out["value_million_usd"], errors="coerce").fillna(0.0) * 1_000_000.0
        out = out[["COUNTRY", "issuer", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class", "value_usd"]].copy()

        out = (
            out.groupby(["COUNTRY", "issuer", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"], as_index=False)["value_usd"]
            .sum()
        )

    # Overlay SHC appendix debt data when present (strict replication for U.S. debt classes).
    shc = load_us_shc_appendix_holdings(end_year=end_year)
    if not shc.empty:
        key_cols = ["COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"]
        shc_keys = shc[key_cols].drop_duplicates()

        marked = out.merge(shc_keys.assign(_drop=1), on=key_cols, how="left")
        dropped = int(marked["_drop"].notna().sum())
        out = marked[marked["_drop"].isna()].drop(columns=["_drop"])

        if out.empty:
            out = shc.copy()
        else:
            out = pd.concat([out, shc], ignore_index=True)
        print(
            "Loaded SHC appendix overlay for USA debt: "
            f"rows={len(shc):,}, replaced_rows={dropped:,}"
        )

    return out


def apply_us_treasury_override(df_bilat: pd.DataFrame, us_treasury: pd.DataFrame) -> pd.DataFrame:
    if us_treasury.empty:
        return df_bilat

    df = df_bilat.copy()
    for col in ["COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    key_cols = ["COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"]
    repl_keys = us_treasury[key_cols].drop_duplicates()

    merged = df.merge(repl_keys.assign(_repl=1), on=key_cols, how="left")
    before = len(merged)
    merged = merged[merged["_repl"].isna()].drop(columns=["_repl"])

    append = us_treasury.copy()
    for c in merged.columns:
        if c not in append.columns:
            append[c] = pd.NA
    append = append[merged.columns]

    if merged.empty:
        out = append.copy()
    elif append.empty:
        out = merged.copy()
    else:
        out = pd.concat([merged, append], ignore_index=True)
    print(
        "Applied US-source override for USA holdings (Treasury/SHC): "
        f"replaced rows={before - len(merged):,}, appended rows={len(append):,}"
    )
    return out


def _preferred_restatement_methodology(investor_iso3: str) -> str:
    inv = str(investor_iso3).upper()
    if inv in ENHANCED_FUND_HOLDINGS_INVESTORS:
        return "Enhanced Fund Holdings"
    if inv in FUND_HOLDINGS_INVESTORS:
        return "Fund Holdings"
    return "Issuance"


def _fallback_methodology_order(preferred: str) -> List[str]:
    all_methods = ["Enhanced Fund Holdings", "Fund Holdings", "Issuance"]
    return [preferred] + [m for m in all_methods if m != preferred]


def _extrapolate_restatement_to_early_years(mat: pd.DataFrame, target_years: set) -> pd.DataFrame:
    """
    If matrices start after requested years (typically start at 2007),
    extend the earliest available matrix backward so requested early years are covered.
    """
    if mat.empty or "Year" not in mat.columns:
        return mat

    year_vals = pd.to_numeric(mat["Year"], errors="coerce").dropna().astype(int)
    if year_vals.empty:
        return mat

    anchor_year = int(year_vals.min())
    need_years = sorted(y for y in target_years if int(y) < anchor_year)
    if not need_years:
        return mat

    anchor = mat[mat["Year"] == anchor_year].copy()
    if anchor.empty:
        return mat

    copies = []
    for y in need_years:
        c = anchor.copy()
        c["Year"] = int(y)
        copies.append(c)

    out = pd.concat([mat] + copies, ignore_index=True)
    print(
        "Extrapolated restatement matrices backward: "
        f"anchor_year={anchor_year}, added_years={need_years}"
    )
    return out


def round_up_holdings_to_reporting_minimum(
    df_bilat: pd.DataFrame,
    value_col: str = "value_usd",
    minimum_usd: float = 1_000.0,
) -> pd.DataFrame:
    """
    Apply paper rule: round positive restated holdings up to the IMF reporting minimum.
    """
    if df_bilat.empty or value_col not in df_bilat.columns:
        return df_bilat

    out = df_bilat.copy()
    v = pd.to_numeric(out[value_col], errors="coerce")
    pos = v > 0
    rounded = v.copy()
    rounded.loc[pos] = np.ceil(v.loc[pos] / float(minimum_usd)) * float(minimum_usd)
    changed = int((rounded.fillna(v) != v).sum())
    out[value_col] = rounded
    print(
        "Applied reporting-minimum rounding: "
        f"minimum_usd={minimum_usd:.0f}, changed_rows={changed:,}"
    )
    return out


def apply_gcap_restatement_matrices(df_bilat: pd.DataFrame, matrices_path: str = RESTATEMENT_MATRICES_PATH) -> pd.DataFrame:
    """
    Restate issuer residency -> nationality using GCAP restatement matrices.
    Applies to ST/LT debt (asset code B) and equity (asset code E).
    """
    if df_bilat.empty:
        return df_bilat
    if not os.path.exists(matrices_path):
        print(f"Restatement matrices not found at {matrices_path}; skipping restatement step.")
        return df_bilat

    work = df_bilat.copy()
    for c in ["COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"]:
        if c in work.columns:
            work[c] = work[c].astype(str)

    work["asset_code"] = work["asset_class"].map({"ST_DEBT": "B", "LT_DEBT": "B", "EQUITY": "E"})
    target = work[work["asset_code"].notna()].copy()
    passthrough = work[work["asset_code"].isna()].drop(columns=["asset_code"], errors="ignore").copy()

    if target.empty:
        return df_bilat

    cols = ["Methodology", "Year", "Investor", "Asset_Class_Code", "Destination", "Destination_Restated", "Value"]
    mat = pd.read_stata(matrices_path, columns=cols)
    mat["Investor"] = mat["Investor"].astype(str).str.upper()
    mat["Asset_Class_Code"] = mat["Asset_Class_Code"].astype(str).str.upper()
    mat["Destination"] = mat["Destination"].astype(str).str.upper()
    mat["Destination_Restated"] = mat["Destination_Restated"].astype(str).str.upper()
    mat["Year"] = pd.to_numeric(mat["Year"], errors="coerce").astype("Int64")
    mat["Value"] = pd.to_numeric(mat["Value"], errors="coerce")

    years = set(pd.to_numeric(target["TIME_PERIOD"], errors="coerce").dropna().astype(int).tolist())
    mat = _extrapolate_restatement_to_early_years(mat, years)
    investors = set(target["COUNTRY"].astype(str).str.upper().unique().tolist())
    destinations = set(target["COUNTERPART_COUNTRY"].astype(str).str.upper().unique().tolist())
    asset_codes = set(target["asset_code"].astype(str).str.upper().unique().tolist())

    mat = mat[
        mat["Year"].isin(list(years))
        & mat["Investor"].isin(list(investors))
        & mat["Destination"].isin(list(destinations))
        & mat["Asset_Class_Code"].isin(list(asset_codes))
    ].copy()

    if mat.empty:
        print("Restatement matrices have no overlapping rows for current bilateral sample; skipping restatement.")
        out = pd.concat([target.drop(columns=["asset_code"], errors="ignore"), passthrough], ignore_index=True)
        return out

    pref_df = pd.DataFrame(
        {
            "Investor": sorted(investors),
            "PreferredMethodology": [_preferred_restatement_methodology(i) for i in sorted(investors)],
        }
    )
    mat = mat.merge(pref_df, on="Investor", how="left")

    # Keep preferred methodology if available; otherwise deterministic fallback per investor.
    kept = []
    for inv, g in mat.groupby("Investor", sort=False):
        preferred = g["PreferredMethodology"].iloc[0]
        methods_available = set(g["Methodology"].dropna().astype(str).tolist())
        chosen = None
        for m in _fallback_methodology_order(preferred):
            if m in methods_available:
                chosen = m
                break
        if chosen is None:
            continue
        kept.append(g[g["Methodology"] == chosen].copy())
    if kept:
        mat = pd.concat(kept, ignore_index=True)
    else:
        out = pd.concat([target.drop(columns=["asset_code"], errors="ignore"), passthrough], ignore_index=True)
        return out

    target["YearInt"] = pd.to_numeric(target["TIME_PERIOD"], errors="coerce").astype("Int64")
    target["COUNTRY"] = target["COUNTRY"].astype(str).str.upper()
    target["COUNTERPART_COUNTRY"] = target["COUNTERPART_COUNTRY"].astype(str).str.upper()

    merged = target.merge(
        mat,
        left_on=["COUNTRY", "YearInt", "asset_code", "COUNTERPART_COUNTRY"],
        right_on=["Investor", "Year", "Asset_Class_Code", "Destination"],
        how="left",
    )

    # If no matrix row exists, keep original destination with weight 1.
    missing = merged["Destination_Restated"].isna()
    merged.loc[missing, "Destination_Restated"] = merged.loc[missing, "COUNTERPART_COUNTRY"]
    merged.loc[missing, "Value"] = 1.0
    merged["Value"] = pd.to_numeric(merged["Value"], errors="coerce").fillna(1.0)

    merged["COUNTERPART_COUNTRY"] = merged["Destination_Restated"].astype(str)
    merged["issuer"] = merged["COUNTERPART_COUNTRY"]

    if "value_usd" in merged.columns:
        merged["value_usd"] = pd.to_numeric(merged["value_usd"], errors="coerce") * merged["Value"]
    if "value" in merged.columns:
        merged["value"] = pd.to_numeric(merged["value"], errors="coerce") * merged["Value"]

    drop_cols = [
        "asset_code", "YearInt", "Methodology", "Year", "Investor", "Asset_Class_Code",
        "Destination", "Destination_Restated", "PreferredMethodology", "Value",
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")

    group_cols = [c for c in merged.columns if c not in {"value_usd", "value"}]
    agg = {}
    if "value_usd" in merged.columns:
        agg["value_usd"] = "sum"
    if "value" in merged.columns:
        agg["value"] = "sum"
    restated = merged.groupby(group_cols, as_index=False).agg(agg)

    out = pd.concat([restated, passthrough], ignore_index=True)
    print(f"Applied GCAP restatement matrices: rows {len(df_bilat):,} -> {len(out):,}")
    return out

# ----------------------------
# Pull 1: Bilateral positions for a single issuer (ALL investors)
# ----------------------------
def pull_bilateral_for_issuer(
    issuer: str,
    start_year: int,
    end_year: int,
    equity_indicator: str,
    sector_code: str = SECTOR_TOTAL,
) -> pd.DataFrame:
    indicators = {**POS_INDICATORS, "EQUITY": equity_indicator}
    ind_sel = "+".join(indicators.values())

    # COUNTRY wildcard => all investors
    key = pip_key(
        country="",
        accounting_entry=ACCOUNTING_ENTRY_ASSETS,
        indicator=ind_sel,
        sector=sector_code,
        counterpart_sector=CP_SECTOR_TOTAL,
        counterpart_country=issuer,
        frequency=FREQ_ANNUAL,
    )
    msg = fetch_data("PIP", key, params={"startPeriod": str(start_year), "endPeriod": str(end_year)})
    df = to_long_df(msg)

    # Tag issuer explicitly (it is already in COUNTERPART_COUNTRY, but keep a stable name)
    df["issuer"] = issuer

    # Map to asset class
    inv_map = {v: k for k, v in indicators.items()}
    df["asset_class"] = df["INDICATOR"].map(inv_map).fillna(df["INDICATOR"])

    return df


# ----------------------------
# Pull 2: Currency aggregates (COUNTERPART_COUNTRY = G001)
# ----------------------------
def build_currency_indicator_list(
    currencies: List[str],
    cl_ind: pd.Series,
    equity_dic_template: str,
) -> List[Tuple[str, str, str]]:
    """
    Returns list of tuples (asset_class, currency, indicator_code),
    keeping only codes that exist in CL_PIP_INDICATOR.
    """
    rows = []
    templates = {
        **DIC_TEMPLATE,
        "DEBT_TOTAL": DEBT_DIC_TOTAL_TEMPLATE,
        "EQUITY": equity_dic_template,
    }
    for asset, tmpl in templates.items():
        for cur in currencies:
            code = tmpl.format(cur=cur)
            if code in cl_ind.index:
                rows.append((asset, cur, code))
    return rows


def pull_currency_aggregates_all_investors(
    currencies: List[str],
    start_year: int,
    end_year: int,
    equity_dic_template: str,
    year_chunks: List[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Pulls currency denomination aggregates across issuers:
      COUNTRY = all investors (wildcard)
      COUNTERPART_COUNTRY = G001
    """
    cl_ind = get_cl_pip_indicator()

    rows = build_currency_indicator_list(currencies, cl_ind, equity_dic_template)
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


def pull_debt_maturity_totals_all_investors(
    start_year: int,
    end_year: int,
    year_chunks: List[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Pull total debt by maturity (not currency-split) for all investors against world counterpart (G001).
    Used only to split F3 DIC total currency buckets when S/L DIC is unavailable in IMF metadata.
    """
    ind_sel = "+".join(POS_INDICATORS.values())

    if year_chunks is None:
        year_chunks = [(start_year, min(2009, end_year)), (2010, min(2015, end_year)), (2016, end_year)]
        year_chunks = [yc for yc in year_chunks if yc[0] <= yc[1]]

    rev = {v: k for k, v in POS_INDICATORS.items()}
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
        df["asset_class"] = df["INDICATOR"].map(rev)
        out.append(df)

    if not out:
        return pd.DataFrame(columns=["COUNTRY", "TIME_PERIOD", "asset_class", "value_usd"])
    return pd.concat(out, ignore_index=True)


def apply_debt_dic_total_fallback(
    ccy_agg: pd.DataFrame,
    debt_maturity_totals: pd.DataFrame,
) -> pd.DataFrame:
    """
    IMF structure-consistent fallback:
    - If currency-specific ST/LT DIC code is missing for a (COUNTRY, TIME_PERIOD, currency),
      but debt total DIC exists, split debt total using same-year ST/LT debt shares from P_F3_S_P_USD / P_F3_L_P_USD.
    - No country-specific or ad-hoc scaling is introduced.
    """
    required_cols = {"COUNTRY", "TIME_PERIOD", "asset_class", "currency", "value_usd"}
    if ccy_agg.empty or not required_cols.issubset(set(ccy_agg.columns)):
        return ccy_agg

    out = ccy_agg.copy()
    out["asset_class"] = out["asset_class"].astype(str)

    totals = out[out["asset_class"] == "DEBT_TOTAL"].copy()
    if totals.empty:
        return out[out["asset_class"] != "DEBT_TOTAL"].copy()

    mat = debt_maturity_totals.copy()
    if mat.empty:
        return out[out["asset_class"] != "DEBT_TOTAL"].copy()

    mat = mat[mat["asset_class"].isin(["ST_DEBT", "LT_DEBT"])].copy()
    mat["value_usd"] = pd.to_numeric(mat["value_usd"], errors="coerce")
    mat_p = (
        mat.groupby(["COUNTRY", "TIME_PERIOD", "asset_class"], as_index=False)["value_usd"]
        .sum()
        .pivot(index=["COUNTRY", "TIME_PERIOD"], columns="asset_class", values="value_usd")
        .reset_index()
    )
    if "ST_DEBT" not in mat_p.columns:
        mat_p["ST_DEBT"] = 0.0
    if "LT_DEBT" not in mat_p.columns:
        mat_p["LT_DEBT"] = 0.0
    mat_p["debt_total"] = mat_p["ST_DEBT"].fillna(0.0) + mat_p["LT_DEBT"].fillna(0.0)
    mat_p = mat_p[mat_p["debt_total"] > 0].copy()
    if mat_p.empty:
        return out[out["asset_class"] != "DEBT_TOTAL"].copy()

    mat_p["st_share"] = mat_p["ST_DEBT"] / mat_p["debt_total"]
    mat_p["lt_share"] = mat_p["LT_DEBT"] / mat_p["debt_total"]
    share_cols = ["COUNTRY", "TIME_PERIOD", "st_share", "lt_share"]

    existing_sl = out[out["asset_class"].isin(["ST_DEBT", "LT_DEBT"])][
        ["COUNTRY", "TIME_PERIOD", "currency", "asset_class"]
    ].drop_duplicates()

    fill_rows = []
    for ac, share_col in [("ST_DEBT", "st_share"), ("LT_DEBT", "lt_share")]:
        tmp = totals.merge(mat_p[share_cols], on=["COUNTRY", "TIME_PERIOD"], how="left")
        tmp = tmp[tmp[share_col].notna()].copy()
        if tmp.empty:
            continue
        tmp["asset_class"] = ac
        tmp["value_usd"] = pd.to_numeric(tmp["value_usd"], errors="coerce") * pd.to_numeric(tmp[share_col], errors="coerce")
        tmp = tmp.merge(
            existing_sl.assign(_exists=1),
            on=["COUNTRY", "TIME_PERIOD", "currency", "asset_class"],
            how="left",
        )
        tmp = tmp[tmp["_exists"].isna()].drop(columns=["_exists", "st_share", "lt_share"], errors="ignore")
        if not tmp.empty:
            fill_rows.append(tmp)

    out = out[out["asset_class"] != "DEBT_TOTAL"].copy()
    if fill_rows:
        out = pd.concat([out] + fill_rows, ignore_index=True)

    out = (
        out.groupby(["COUNTRY", "TIME_PERIOD", "asset_class", "currency"], as_index=False)["value_usd"]
        .sum()
    )
    return out


# ----------------------------
# Currency-cap allocator (general)
# ----------------------------
def allocate_local_foreign_by_currency_caps(
    bilateral: pd.DataFrame,
    ccy_agg: pd.DataFrame,
    issuer_to_local_ccy: Dict[str, str],
    cap_relax_mult: float = 1.0,
    disable_cap: bool = False,
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
    df["value_total"] = pd.to_numeric(df[value_col], errors="coerce")

    agg = (
        ccy_agg.groupby([investor_col, "asset_class", year_col, "currency"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "agg_ccy_total"})
    )

    has_local = df["local_currency"].notna()
    df["value_local_candidate"] = df["value_total"].where(has_local, 0.0)

    cand = (
        df[has_local]
        .groupby([investor_col, "asset_class", year_col, "local_currency"], as_index=False)["value_local_candidate"]
        .sum()
        .rename(columns={"value_local_candidate": "cand_local_sum"})
    )

    agg_exact = agg.rename(columns={"currency": "local_currency", "agg_ccy_total": "cap_exact"})
    agg_othc = (
        agg[agg["currency"].astype(str) == "OTHC"]
        [[investor_col, "asset_class", year_col, "agg_ccy_total"]]
        .rename(columns={"agg_ccy_total": "cap_othc"})
    )

    caps = cand.merge(
        agg_exact[[investor_col, "asset_class", year_col, "local_currency", "cap_exact"]],
        on=[investor_col, "asset_class", year_col, "local_currency"],
        how="left",
    )
    caps = caps.merge(agg_othc, on=[investor_col, "asset_class", year_col], how="left")

    is_core = caps["local_currency"].astype(str).isin(CORE_REPORTED_CURRENCIES)
    cap_exact = pd.to_numeric(caps["cap_exact"], errors="coerce")
    cap_othc = pd.to_numeric(caps["cap_othc"], errors="coerce")
    caps["cap_total"] = np.where(
        is_core,
        cap_exact.where(cap_exact.notna(), cap_othc),
        cap_exact.fillna(0.0) + cap_othc.fillna(0.0),
    )
    caps.loc[cap_exact.isna() & cap_othc.isna(), "cap_total"] = np.nan

    if disable_cap:
        caps["cap_ratio"] = 1.0
    else:
        relax_mult = max(0.0, float(cap_relax_mult))
        caps["cap_ratio"] = ((caps["cap_total"] * relax_mult) / caps["cand_local_sum"]).clip(lower=0.0, upper=1.0)

    df = df.merge(
        caps[[investor_col, "asset_class", year_col, "local_currency", "cap_ratio"]],
        on=[investor_col, "asset_class", year_col, "local_currency"],
        how="left",
    )
    # Appendix C missing-data discipline: if issuer local currency is known but
    # no cap bucket can be assigned/observed, keep allocation as missing.
    # Only truly unknown issuer local currency mappings stay at 0 local.
    has_local_map = df["local_currency"].notna()
    has_cap_obs = df["cap_ratio"].notna()
    df.loc[has_local_map & (~has_cap_obs), "cap_ratio"] = np.nan

    df["value_local"] = np.where(
        has_local_map,
        df["value_total"] * df["cap_ratio"],
        0.0,
    )
    df["value_foreign"] = np.where(
        has_local_map & df["value_local"].isna(),
        np.nan,
        df["value_total"] - df["value_local"],
    )

    keep = [
        investor_col, issuer_col, "asset_class", year_col,
        "local_currency", "value_total", "value_local", "value_foreign"
    ]
    return df[keep]


# ----------------------------
# Main runner
# ----------------------------
def main(out_dir: Optional[str] = None, end_year: int = 2024):
    # NOTE: output values are raw USD (value_usd column).
    # Downstream scripts must divide by 1e9 to convert to USD billions
    # to match the convention used in the Stata pipeline (IMF_CPIS.do).
    if out_dir is None:
        out_dir = str(_settings_config("DATA_DIR"))
    cl_ind = get_cl_pip_indicator()
    equity_pos_indicator, equity_dic_template = resolve_equity_indicators(cl_ind)
    print(f"Using equity indicators: position={equity_pos_indicator}, denomination_template={equity_dic_template}")

    reserve_codes_from_txt = load_reserve_sector_codes_from_txt()
    if reserve_codes_from_txt:
        reserve_sector_codes = reserve_codes_from_txt
        print(
            f"Using reserve-sector code(s) from txt ({RESERVE_SECTOR_CODE_FILE}): "
            f"{reserve_sector_codes}"
        )
    else:
        candidates = discover_reserve_sector_candidates()
        probe = probe_reserve_sector_codes(candidates, equity_indicator=equity_pos_indicator, end_year=end_year)
        nonempty = probe[probe["rows"] > 0].copy()
        if nonempty.empty:
            reserve_sector_codes = []
            print(
                "Reserve-sector auto-discovery found no non-empty codes; "
                "reserve-sector pull will be skipped."
            )
        else:
            if RESERVE_SECTOR_MODE == "strict":
                # Strict mode prioritizes central-bank-only convention.
                priority = ["S121", "S1XA", "S1X", "S1KK", "S122"]
                picked = None
                for c in priority:
                    if c in set(nonempty["sector_code"].tolist()):
                        picked = c
                        break
                if picked is None:
                    picked = str(nonempty.iloc[0]["sector_code"])
                reserve_sector_codes = [picked]
            else:
                # Broad mode keeps all non-empty reserve-like sectors to maximize completeness.
                reserve_sector_codes = nonempty["sector_code"].astype(str).tolist()

            print(
                "Reserve-sector auto-discovery diagnostics (top):\n"
                f"{probe.head(10).to_string(index=False)}"
            )
            print(
                f"Reserve-sector mode={RESERVE_SECTOR_MODE}; selected code(s): {reserve_sector_codes}"
            )

    # 1) Pull bilateral for the Table C1 issuers only (ALL investors)
    bilat_parts = []
    for issuer in ISSUERS:
        sy = ISSUER_START_YEAR[issuer]
        print(f"Pulling bilateral for issuer={issuer}, years={sy}-{end_year} ...")
        df_i = pull_bilateral_for_issuer(issuer, sy, end_year, equity_indicator=equity_pos_indicator)
        bilat_parts.append(df_i)

    df_bilat = pd.concat(bilat_parts, ignore_index=True)

    # Keep IMF confidentiality aggregate investor rows (non-ISO codes, e.g. TX093)
    # as a separate reserve aggregate source before ISO filtering.
    df_bilat_reserve_agg = pd.DataFrame()
    if "COUNTRY" in df_bilat.columns:
        ctry_all = df_bilat["COUNTRY"].astype(str).str.upper()
        noniso_mask = ~ctry_all.str.match(r"^[A-Z]{3}$", na=False)
        reserve_like_mask = ctry_all.str.match(r"^TX\d+$", na=False)
        keep_mask = noniso_mask & reserve_like_mask
        if keep_mask.any():
            df_bilat_reserve_agg = df_bilat.loc[keep_mask].copy()

    # 1b) Pull bilateral for reserve-sector investors only (if IMF codelist is resolvable)
    df_bilat_reserve = pd.DataFrame()
    if reserve_sector_codes:
        reserve_parts = []
        for sector_code in reserve_sector_codes:
            for issuer in ISSUERS:
                sy = ISSUER_START_YEAR[issuer]
                print(
                    f"Pulling reserve-sector bilateral for issuer={issuer}, years={sy}-{end_year}, "
                    f"sector={sector_code} ..."
                )
                df_i = pull_bilateral_for_issuer(
                    issuer,
                    sy,
                    end_year,
                    equity_indicator=equity_pos_indicator,
                    sector_code=sector_code,
                )
                if not df_i.empty:
                    df_i = df_i.copy()
                    df_i["reserve_sector_code"] = sector_code
                reserve_parts.append(df_i)
        if reserve_parts:
            df_bilat_reserve = pd.concat(reserve_parts, ignore_index=True)

    us_treasury = load_us_treasury_holdings(end_year=end_year)
    us_treasury_available = not us_treasury.empty
    if us_treasury_available:
        print(
            "US Treasury nationality data detected: "
            f"rows={len(us_treasury):,}, years={us_treasury['TIME_PERIOD'].min()}-{us_treasury['TIME_PERIOD'].max()}"
        )

    # Drop OFCs always; drop USA only if separate TIC data is present.
    tic_fallback_available = any(os.path.exists(p) for p in TIC_FALLBACK_PATHS)
    tic_available = bool(us_treasury_available or tic_fallback_available)
    exclude_investors = set(OFC_INVESTORS)
    if tic_available and not us_treasury_available:
        exclude_investors.add("USA")

    if "COUNTRY" in df_bilat.columns:
        before = len(df_bilat)
        df_bilat = df_bilat[~df_bilat["COUNTRY"].isin(exclude_investors)].copy()
        # Keep only country-like investor codes (ISO3 style). This removes aggregates such as TX*.
        ctry = df_bilat["COUNTRY"].astype(str).str.upper()
        valid_country_code = ctry.str.match(r"^[A-Z]{3}$", na=False)
        before_iso = len(df_bilat)
        df_bilat = df_bilat[valid_country_code].copy()
        print(
            f"Dropped investors: {sorted(exclude_investors)} | "
            f"TIC available={tic_available} | Treasury table available={us_treasury_available} | "
            f"rows removed(exclusions+nonISO)={(before - len(df_bilat)):,} | nonISO removed={(before_iso - len(df_bilat)):,}"
        )

    # Scale
    df_bilat = scale_to_usd(df_bilat)
    if not df_bilat_reserve_agg.empty:
        df_bilat_reserve_agg = scale_to_usd(df_bilat_reserve_agg)
    if not df_bilat_reserve.empty:
        df_bilat_reserve = scale_to_usd(df_bilat_reserve)

    # Restate issuer residency -> nationality before currency-cap allocation and before Treasury override.
    df_bilat = apply_gcap_restatement_matrices(df_bilat)
    df_bilat = round_up_holdings_to_reporting_minimum(df_bilat, value_col="value_usd", minimum_usd=1_000.0)
    if not df_bilat_reserve_agg.empty:
        # Keep same reporting-minimum handling for confidentiality aggregates.
        df_bilat_reserve_agg = round_up_holdings_to_reporting_minimum(
            df_bilat_reserve_agg,
            value_col="value_usd",
            minimum_usd=1_000.0,
        )
    if not df_bilat_reserve.empty:
        df_bilat_reserve = apply_gcap_restatement_matrices(df_bilat_reserve)
        df_bilat_reserve = round_up_holdings_to_reporting_minimum(df_bilat_reserve, value_col="value_usd", minimum_usd=1_000.0)
        df_bilat_reserve = consolidate_reserve_sector_rows(df_bilat_reserve)

    # If Treasury nationality tables exist, override USA LT_DEBT/EQUITY with Treasury values.
    if us_treasury_available:
        df_bilat = apply_us_treasury_override(df_bilat, us_treasury)

    # Save bilateral
    bilat_parquet = os.path.join(out_dir, "pip_bilateral_positions.parquet")
    bilat_csv = os.path.join(out_dir, "pip_bilateral_positions.csv")
    df_bilat.to_parquet(bilat_parquet, index=False)
    df_bilat.to_csv(bilat_csv, index=False)
    print(f"Saved bilateral to:\n  {bilat_parquet}\n  {bilat_csv}")

    # Save reserve-sector bilateral if available
    if not df_bilat_reserve.empty:
        if "COUNTRY" in df_bilat_reserve.columns:
            df_bilat_reserve = df_bilat_reserve[~df_bilat_reserve["COUNTRY"].isin(exclude_investors)].copy()
            # Do NOT force ISO3 on reserve pulls: confidentiality aggregates may use non-ISO investor codes.
            # Keep all remaining investor codes so reserve totals are not mechanically understated.

        reserve_parquet = os.path.join(out_dir, "pip_bilateral_positions_reserve.parquet")
        reserve_csv = os.path.join(out_dir, "pip_bilateral_positions_reserve.csv")
        df_bilat_reserve.to_parquet(reserve_parquet, index=False)
        df_bilat_reserve.to_csv(reserve_csv, index=False)
        print(f"Saved reserve-sector bilateral to:\n  {reserve_parquet}\n  {reserve_csv}")

    # Save confidentiality aggregate reserve investor rows from standard bilateral pulls.
    if not df_bilat_reserve_agg.empty:
        reserve_agg_parquet = os.path.join(out_dir, "pip_bilateral_positions_reserve_aggregate.parquet")
        reserve_agg_csv = os.path.join(out_dir, "pip_bilateral_positions_reserve_aggregate.csv")
        keep_cols = [
            c
            for c in ["issuer", "COUNTRY", "COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class", "value_usd"]
            if c in df_bilat_reserve_agg.columns
        ]
        df_bilat_reserve_agg = df_bilat_reserve_agg[keep_cols].copy()
        df_bilat_reserve_agg.to_parquet(reserve_agg_parquet, index=False)
        df_bilat_reserve_agg.to_csv(reserve_agg_csv, index=False)
        print(
            "Saved reserve aggregate (confidentiality investor codes) to:\n"
            f"  {reserve_agg_parquet}\n"
            f"  {reserve_agg_csv}"
        )

    # 2) Pull currency aggregates for all investors (COUNTERPART=G001)
    # Pull instruction core buckets plus issuer local currencies so non-core
    # currencies can use exact+OTHC combined caps in allocation.
    desired_currencies = sorted(set(BASE_CURRENCIES).union(set(ISSUER_TO_LOCAL_CCY.values())).union({"OTHC"}))

    # Keep only currencies that actually have any DIC indicator codes in CL_PIP_INDICATOR
    valid_rows = build_currency_indicator_list(desired_currencies, cl_ind, equity_dic_template)
    valid_currencies = sorted(set([cur for _, cur, _ in valid_rows]))

    print(f"Pulling currency aggregates for currencies={valid_currencies} ...")
    df_ccy = pull_currency_aggregates_all_investors(
        currencies=valid_currencies,
        start_year=2003,
        end_year=end_year,
        equity_dic_template=equity_dic_template,
    )
    df_ccy = scale_to_usd(df_ccy)

    # IMF-structure-consistent debt DIC completion:
    # where ST/LT currency indicators are missing but debt total DIC exists,
    # split total via same-year IMF ST/LT position shares.
    df_debt_maturity = pull_debt_maturity_totals_all_investors(start_year=2003, end_year=end_year)
    df_debt_maturity = scale_to_usd(df_debt_maturity)
    before_ccy_rows = len(df_ccy)
    df_ccy = apply_debt_dic_total_fallback(df_ccy, df_debt_maturity)
    print(f"Applied debt DIC total fallback split: rows {before_ccy_rows:,} -> {len(df_ccy):,}")

    ccy_parquet = os.path.join(out_dir, "pip_currency_aggregates.parquet")
    ccy_csv = os.path.join(out_dir, "pip_currency_aggregates.csv")
    df_ccy.to_parquet(ccy_parquet, index=False)
    df_ccy.to_csv(ccy_csv, index=False)
    print(f"Saved currency aggregates to:\n  {ccy_parquet}\n  {ccy_csv}")

    # 3) OPTIONAL: local-vs-foreign allocation with caps (paper logic)
    print(f"Using cap relaxation multiplier: {CAP_RELAX_MULT}")
    print(f"Disable cap mode: {DISABLE_CAP}")
    alloc = allocate_local_foreign_by_currency_caps(
        bilateral=df_bilat,
        ccy_agg=df_ccy,
        issuer_to_local_ccy=ISSUER_TO_LOCAL_CCY,
        cap_relax_mult=CAP_RELAX_MULT,
        disable_cap=DISABLE_CAP,
        value_col="value_usd",
    )
    alloc_parquet = os.path.join(out_dir, "pip_local_foreign_allocated.parquet")
    alloc_csv = os.path.join(out_dir, "pip_local_foreign_allocated.csv")
    alloc.to_parquet(alloc_parquet, index=False)
    alloc.to_csv(alloc_csv, index=False)
    print(f"Saved local/foreign allocation to:\n  {alloc_parquet}\n  {alloc_csv}")

    # Diagnostic extract: rows where issuer local currency is known but local allocation
    # could not be computed (typically missing cap bucket in currency aggregates).
    alloc_missing = alloc[
        alloc["local_currency"].notna() & pd.to_numeric(alloc["value_local"], errors="coerce").isna()
    ].copy()
    missing_parquet = os.path.join(out_dir, "pip_local_foreign_missing_alloc.parquet")
    missing_csv = os.path.join(out_dir, "pip_local_foreign_missing_alloc.csv")
    alloc_missing.to_parquet(missing_parquet, index=False)
    alloc_missing.to_csv(missing_csv, index=False)
    print(
        "Saved missing-local-allocation diagnostics to:\n"
        f"  {missing_parquet}\n"
        f"  {missing_csv}\n"
        f"  rows={len(alloc_missing):,}"
    )

    # Issuer-level audit: helps diagnose when domestic shares become too large
    # because local-currency foreign holdings are very small vs total foreign holdings.
    issuer_audit = (
        alloc.groupby(["COUNTERPART_COUNTRY", "TIME_PERIOD", "asset_class"], as_index=False)
        .agg(
            value_total=("value_total", "sum"),
            value_local=("value_local", "sum"),
            value_foreign=("value_foreign", "sum"),
            row_count=("COUNTRY", "size"),
        )
        .rename(columns={"COUNTERPART_COUNTRY": "issuer"})
    )
    issuer_audit["local_share_in_foreign_holdings"] = pd.to_numeric(
        issuer_audit["value_local"], errors="coerce"
    ) / pd.to_numeric(issuer_audit["value_total"], errors="coerce")
    issuer_audit_parquet = os.path.join(out_dir, "pip_local_foreign_issuer_audit.parquet")
    issuer_audit_csv = os.path.join(out_dir, "pip_local_foreign_issuer_audit.csv")
    issuer_audit.to_parquet(issuer_audit_parquet, index=False)
    issuer_audit.to_csv(issuer_audit_csv, index=False)
    print(
        "Saved issuer-level local allocation audit to:\n"
        f"  {issuer_audit_parquet}\n"
        f"  {issuer_audit_csv}\n"
        f"  rows={len(issuer_audit):,}"
    )


if __name__ == "__main__":
    # Uses DATA_DIR from settings.py (defaults to _data/).
    # Override via: python pull_imf.py --DATA_DIR=/path/to/data
    main(end_year=2024)