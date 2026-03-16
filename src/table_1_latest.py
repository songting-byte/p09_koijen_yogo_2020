"""table_1_latest.py
===================
Produces Table 1 (Market Values of Financial Assets) for the latest available
year using API-sourced parquets, and for 2020 as a cross-check against the
paper replication.

Data sources (all from pull_* scripts):
  Debt outstanding  : OECD T720 parquet (OECD members)
                      BIS DDS parquet   (non-OECD fallback)
  Equity outstanding: OECD T720 parquet (OECD members, F5 proxy)
                      World Bank WDI    (non-OECD fallback)
  Bilateral holdings: IMF PIP parquet
  Reserve holdings  : IMF PIP reserve parquet

Key differences from table_1.py (dataverse replication):
  - Reads API parquets, not .dta files
  - Nationality restatements not applied (not available from free sources)
  - US Treasury supplement not applied
  - Latest year auto-detected as min(max year across all sources)

Run:
    python table_1_latest.py           # outputs table_1_YYYY.txt
    python table_1_latest.py --year 2023
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "_data"
OUTPUT_DIR = Path(__file__).parent.parent / "_output"
SMALL      = 1e-6


# ── Type / instrument mappings ────────────────────────────────────────────────
OECD_INSTR_TO_TYPE: dict[str, int] = {
    "F3S":  1,   # short-term debt securities
    "F3L":  2,   # long-term debt securities
    "F5":   3,   # equity + investment fund shares (proxy; known 2-3x overcount)
    "F51":  3,   # equity (excluding fund shares)
}

BIS_MATURITY_TO_TYPE: dict[str, int] = {
    "C": 1, "S": 1,   # short-term (WS_DEBT_SEC2_PUB used C; WS_NA_SEC_DSS uses S)
    "K": 2, "L": 2,   # long-term  (WS_DEBT_SEC2_PUB used K; WS_NA_SEC_DSS uses L)
}

IMF_ASSET_CLASS_TO_TYPE: dict[str, int] = {
    "ST_DEBT": 1,
    "LT_DEBT": 2,
    "EQUITY":  3,
}

# ISO2 → ISO3 for BIS issuer codes (same mapping as pull_bis.py, reversed)
ISO2_TO_ISO3: dict[str, str] = {
    "AU": "AUS", "AT": "AUT", "BE": "BEL", "BR": "BRA", "CA": "CAN",
    "CH": "CHE", "CN": "CHN", "CO": "COL", "CZ": "CZE", "DE": "DEU",
    "DK": "DNK", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR",
    "GR": "GRC", "HK": "HKG", "HU": "HUN", "IL": "ISR", "IN": "IND",
    "IT": "ITA", "JP": "JPN", "KR": "KOR", "MX": "MEX", "MY": "MYS",
    "NL": "NLD", "NO": "NOR", "NZ": "NZL", "PH": "PHL", "PL": "POL",
    "PT": "PRT", "RU": "RUS", "SE": "SWE", "SG": "SGP", "TH": "THA",
    "US": "USA", "ZA": "ZAF",
}

# MSCI regional groupings (same as table_1.py)
MSCI_GROUP: dict[str, int] = {
    "DM: Americas":                         1,
    "DM: Europe & Middle East":             2,
    "DM: Pacific":                          3,
    "EM: Americas":                         4,
    "EM: Asia":                             4,
    "EM: Europe, Middle East & Africa":     4,
}
REGION_LABEL: dict[int, str] = {
    1: "Developed markets: North America",
    2: "Developed markets: Europe",
    3: "Developed markets: Pacific",
    4: "Emerging markets",
}

# OECD ISO3 members whose debt is covered by T720
OECD_MEMBERS_ISO3 = {
    "AUS", "AUT", "BEL", "CAN", "CHE", "CHL", "COL", "CZE", "DEU",
    "DNK", "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HUN", "IRL",
    "ISL", "ISR", "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX",
    "NLD", "NOR", "NZL", "POL", "PRT", "SVK", "SVN", "SWE", "TUR",
    "USA",
}

# ─────────────────────────────────────────────────────────────────────────────
# Country metadata
# ─────────────────────────────────────────────────────────────────────────────

def _load_countries() -> pd.DataFrame:
    """
    Return country metadata covering the 33 paper countries.
    """
    # (Counterpart, Name, MSCI_group)
    rows = [
        # DM: Americas (group=1)
        ("CAN", "Canada",          1), ("USA", "United States",   1),
        # DM: Europe (group=2)
        ("AUT", "Austria",         2), ("BEL", "Belgium",         2),
        ("CHE", "Switzerland",     2), ("DEU", "Germany",         2),
        ("DNK", "Denmark",         2), ("ESP", "Spain",           2),
        ("FIN", "Finland",         2), ("FRA", "France",          2),
        ("GBR", "United Kingdom",  2), ("GRC", "Greece",          2),
        ("IRL", "Ireland",         2), ("ISR", "Israel",          2),
        ("ITA", "Italy",           2), ("NLD", "Netherlands",     2),
        ("NOR", "Norway",          2), ("PRT", "Portugal",        2),
        ("SWE", "Sweden",          2),
        # DM: Pacific (group=3)
        ("AUS", "Australia",       3), ("HKG", "Hong Kong",       3),
        ("JPN", "Japan",           3), ("NZL", "New Zealand",     3),
        ("SGP", "Singapore",       3),
        # EM (group=4)
        ("BRA", "Brazil",          4), ("CHN", "China",           4),
        ("COL", "Colombia",        4), ("CZE", "Czech Republic",  4),
        ("HUN", "Hungary",         4), ("IND", "India",           4),
        ("KOR", "South Korea",     4), ("MEX", "Mexico",          4),
        ("MYS", "Malaysia",        4), ("PHL", "Philippines",     4),
        ("POL", "Poland",          4), ("RUS", "Russia",          4),
        ("THA", "Thailand",        4), ("ZAF", "South Africa",    4),
    ]
    ctry = pd.DataFrame(rows, columns=["Counterpart", "Name", "group"])
    return ctry


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_oecd_amounts(data_dir: Path) -> pd.DataFrame:
    """
    Load OECD T720 parquet → tidy (Counterpart_iso3, year, type, outstand_bn).

    Values are USD millions (UNIT_MULT=6, CURRENCY_DENOM=USD requested in pull).
    outstand_bn = value / 1000.

    Instrument priority:
      F3S → ST debt (type=1)
      F3L → LT debt (type=2)
      F5  → equity proxy (type=3) — 2-3x larger than market cap
    """
    path = data_dir / "oecd_t720.parquet"
    if not path.exists():
        print("  OECD parquet not found — skipping OECD amounts")
        return pd.DataFrame(columns=["Counterpart", "year", "type", "outstand"])

    df = pd.read_parquet(path)

    # Keep only instruments we need
    df = df[df["financial_instrument"].isin(["F3", "F3S", "F3L", "F5", "F51"])].copy()

    # Filter sector S1 (total economy), liabilities (accounting_entry=L or blank)
    if "sector" in df.columns:
        df = df[df["sector"].isin(["S1", ""])].copy()
    if "accounting_entry" in df.columns:
        df = df[df["accounting_entry"].isin(["L", ""])].copy()

    # Exclude MATURITY=T (total = L+S combined; would double-count)
    if "original_maturity" in df.columns:
        df = df[~df["original_maturity"].isin(["T"])].copy()

    # Assign type: F3 debt split by maturity S→1, L→2; F3S→1, F3L→2; F5/F51→3
    def _assign_type(row):
        instr = row["financial_instrument"]
        mat   = row.get("original_maturity", "_Z") if "original_maturity" in row.index else "_Z"
        if instr in ("F5", "F51"):
            return 3
        if instr == "F3S":
            return 1
        if instr == "F3L":
            return 2
        if instr == "F3":
            if mat == "S":
                return 1
            if mat == "L":
                return 2
        return None

    df["type"] = df.apply(_assign_type, axis=1)
    df["year"]         = pd.to_numeric(df["time_period"], errors="coerce").astype("Int64")
    df["Counterpart"]  = df["reference_area"].astype(str).str.upper()
    df["outstand"]     = pd.to_numeric(df["value"], errors="coerce") / 1000.0  # millions → billions

    df_clean = df.dropna(subset=["Counterpart", "year", "type", "outstand"]).query("outstand > 0")

    # For equity (type=3): prefer F51 (equity shares) over F5 (equity + fund shares)
    # per country-year to avoid including investment fund shares.
    eq_mask = df_clean["type"] == 3
    eq = df_clean[eq_mask].copy()
    non_eq = df_clean[~eq_mask].copy()

    if not eq.empty and "F51" in eq["financial_instrument"].values:
        ctry_year_f51 = set(
            eq[eq["financial_instrument"] == "F51"]
            .apply(lambda r: (r["Counterpart"], r["year"]), axis=1)
        )
        eq = eq[~(
            (eq["financial_instrument"] == "F5") &
            eq.apply(lambda r: (r["Counterpart"], r["year"]) in ctry_year_f51, axis=1)
        )].copy()

    df_clean = pd.concat([non_eq, eq], ignore_index=True)
    agg = (
        df_clean
        .groupby(["Counterpart", "year", "type"], as_index=False)["outstand"]
        .sum()
    )
    return agg


def _load_bis_dds_amounts(data_dir: Path) -> pd.DataFrame:
    """
    Load BIS domestic debt securities parquet → (Counterpart_iso3, year, type, outstand_bn).

    BIS reports in USD with UNIT_MULT (typically 6 = millions).
    Keeps only Q4 observations as the annual snapshot.
    Maps ISO2 issuer codes → ISO3.
    Uses ISSUE_OR_MAT: C=ST(1), K=LT(2). Drops All-debt (A) rows.
    """
    path = data_dir / "bis_dds_q.parquet"
    if not path.exists():
        print("  BIS DDS parquet not found — skipping BIS amounts")
        return pd.DataFrame(columns=["Counterpart", "year", "type", "outstand"])

    df = pd.read_parquet(path)

    # Keep Q4 observations only
    time_col = next((c for c in ["TIME_PERIOD", "time_period"] if c in df.columns), None)
    if time_col is None:
        return pd.DataFrame(columns=["Counterpart", "year", "type", "outstand"])
    df = df[df[time_col].astype(str).str.endswith("Q4")].copy()
    df["year"] = df[time_col].astype(str).str[:4].astype(int)

    # Map maturity → type (keep only ST and LT, drop All-debt)
    mat_col = next((c for c in ["ISSUE_OR_MAT", "issue_or_mat", "MATURITY", "maturity"] if c in df.columns), None)
    if mat_col is None:
        return pd.DataFrame(columns=["Counterpart", "year", "type", "outstand"])
    df["type"] = df[mat_col].map(BIS_MATURITY_TO_TYPE)
    df = df.dropna(subset=["type"]).copy()
    df["type"] = df["type"].astype(int)

    # Map ISO2 → ISO3 (WS_DEBT_SEC2_PUB used ISSUER_RES; WS_NA_SEC_DSS uses REF_AREA)
    res_col = next((c for c in ["ISSUER_RES", "issuer_res", "REF_AREA", "ref_area"] if c in df.columns), None)
    if res_col is None:
        return pd.DataFrame(columns=["Counterpart", "year", "type", "outstand"])
    df["Counterpart"] = df[res_col].astype(str).str.upper().map(ISO2_TO_ISO3)

    # Compute value in billions: OBS_VALUE * 10^(UNIT_MULT - 9)
    obs_col  = next((c for c in ["OBS_VALUE", "obs_value"] if c in df.columns), None)
    unit_col = next((c for c in ["UNIT_MULT", "unit_mult"] if c in df.columns), None)
    if obs_col is None:
        return pd.DataFrame(columns=["Counterpart", "year", "type", "outstand"])

    df["obs"] = pd.to_numeric(df[obs_col], errors="coerce")
    if unit_col is not None:
        df["umult"] = pd.to_numeric(df[unit_col], errors="coerce").fillna(6)
    else:
        df["umult"] = 6.0
    df["outstand"] = df["obs"] * (10.0 ** (df["umult"] - 9.0))

    agg = (
        df.dropna(subset=["Counterpart", "year", "type", "outstand"])
          .query("outstand > 0")
          .groupby(["Counterpart", "year", "type"], as_index=False)["outstand"]
          .sum()
    )
    return agg


def _load_ids_foreign_amounts(data_dir: Path) -> pd.DataFrame:
    """
    Load BIS IDS foreign-currency parquet → (Counterpart_iso3, year, type, ids_foreign_bn).

    Used to subtract from DDS / OECD T720 totals to get local-currency amounts.
    UNIT_MULT=6 → USD millions; convert to billions.
    Keeps only Q4 observations.
    Maps ISSUE_OR_MAT: C→ST(1), K→LT(2).
    """
    path = data_dir / "bis_ids_foreign_q.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Counterpart", "year", "type", "ids_foreign"])

    df = pd.read_parquet(path)

    time_col = next((c for c in ["TIME_PERIOD", "time_period"] if c in df.columns), None)
    res_col  = next((c for c in ["ISSUER_RES", "issuer_res"] if c in df.columns), None)
    mat_col  = next((c for c in ["ISSUE_OR_MAT", "issue_or_mat"] if c in df.columns), None)
    obs_col  = next((c for c in ["OBS_VALUE", "obs_value"] if c in df.columns), None)
    unit_col = next((c for c in ["UNIT_MULT", "unit_mult"] if c in df.columns), None)

    if any(c is None for c in [time_col, res_col, mat_col, obs_col]):
        return pd.DataFrame(columns=["Counterpart", "year", "type", "ids_foreign"])

    df = df[df[time_col].astype(str).str.endswith("Q4")].copy()
    df["year"] = df[time_col].astype(str).str[:4].astype(int)

    mat_map = {"C": 1, "K": 2}
    df["type"] = df[mat_col].map(mat_map)
    df = df.dropna(subset=["type"]).copy()
    df["type"] = df["type"].astype(int)

    df["Counterpart"] = df[res_col].astype(str).str.upper().map(ISO2_TO_ISO3)

    df["obs"] = pd.to_numeric(df[obs_col], errors="coerce")
    if unit_col is not None:
        df["umult"] = pd.to_numeric(df[unit_col], errors="coerce").fillna(6)
    else:
        df["umult"] = 6.0
    # UNIT_MULT=6 → millions → divide by 1000 for billions
    df["ids_foreign"] = df["obs"] * (10.0 ** (df["umult"] - 9.0))

    agg = (
        df.dropna(subset=["Counterpart", "year", "type", "ids_foreign"])
          .query("ids_foreign >= 0")
          .groupby(["Counterpart", "year", "type"], as_index=False)["ids_foreign"]
          .sum()
    )
    return agg


def _load_wb_equity(data_dir: Path) -> pd.DataFrame:
    """
    Load World Bank market-cap parquet → (Counterpart_iso3, year, outstand_bn).

    Indicator: CM.MKT.LCAP.CD  (market cap in current USD).
    outstand_bn = value / 1e9.
    """
    path = data_dir / "wb_data360_wdi_selected.parquet"
    if not path.exists():
        print("  WB parquet not found — skipping WB equity")
        return pd.DataFrame(columns=["Counterpart", "year", "outstand"])

    df = pd.read_parquet(path)

    # Accept various column name styles
    area_col  = next((c for c in ["ref_area", "REF_AREA", "reference_area"] if c in df.columns), None)
    time_col  = next((c for c in ["time_period", "TIME_PERIOD"] if c in df.columns), None)
    val_col   = next((c for c in ["value", "OBS_VALUE", "obs_value"] if c in df.columns), None)
    ind_col   = next((c for c in ["indicator", "INDICATOR"] if c in df.columns), None)

    if area_col is None or time_col is None or val_col is None:
        return pd.DataFrame(columns=["Counterpart", "year", "outstand"])

    # Filter for market cap indicator — try metric column first, then indicator column
    metric_col = next((c for c in ["metric", "METRIC"] if c in df.columns), None)
    mktcap_codes = {
        "CM.MKT.LCAP.CD", "MKTCAP", "market_cap",
        "WB_WDI_CM_MKT_LCAP_CD", "market_cap_listed_domestic_companies_current_usd",
    }
    filtered = False
    if metric_col is not None:
        mask = df[metric_col].astype(str).isin(mktcap_codes)
        if mask.any():
            df = df[mask].copy()
            filtered = True
    if not filtered and ind_col is not None:
        mask = df[ind_col].astype(str).isin(mktcap_codes)
        if mask.any():
            df = df[mask].copy()

    df["Counterpart"] = df[area_col].astype(str).str.upper()
    df["year"]        = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    df["outstand"]    = pd.to_numeric(df[val_col], errors="coerce") / 1e9  # USD → billions

    agg = (
        df.dropna(subset=["Counterpart", "year", "outstand"])
          .query("outstand > 0")
          .groupby(["Counterpart", "year"], as_index=False)["outstand"]
          .sum()
    )
    agg["type"] = 3
    return agg[["Counterpart", "year", "type", "outstand"]]


def _load_cpis_bilateral(data_dir: Path) -> pd.DataFrame:
    """
    Load IMF PIP bilateral parquet → (investor_iso3, issuer_iso3, type, year, value_bn).

    Parquet columns: COUNTRY (investor ISO3), COUNTERPART_COUNTRY (issuer ISO3),
                     TIME_PERIOD (year), asset_class (ST_DEBT/LT_DEBT/EQUITY), value_usd.
    value_bn = value_usd / 1000 (USD millions → billions).
    """
    path = data_dir / "pip_bilateral_positions.parquet"
    if not path.exists():
        print("  IMF bilateral parquet not found")
        return pd.DataFrame(columns=["investor", "Counterpart", "type", "year", "value"])

    df = pd.read_parquet(path)

    investor_col    = next((c for c in ["COUNTRY", "country", "investor"] if c in df.columns), None)
    counterpart_col = next((c for c in ["COUNTERPART_COUNTRY", "counterpart_country",
                                         "COUNTERPART", "counterpart", "issuer"] if c in df.columns), None)
    time_col        = next((c for c in ["TIME_PERIOD", "time_period", "year"] if c in df.columns), None)
    asset_col       = next((c for c in ["asset_class", "ASSET_CLASS"] if c in df.columns), None)
    val_col         = next((c for c in ["value_usd", "value", "VALUE_USD", "OBS_VALUE"] if c in df.columns), None)

    if any(c is None for c in [investor_col, counterpart_col, time_col, val_col]):
        print(f"  IMF bilateral: missing expected columns. Found: {list(df.columns)}")
        return pd.DataFrame(columns=["investor", "Counterpart", "type", "year", "value"])

    df["investor"]    = df[investor_col].astype(str).str.upper()
    df["Counterpart"] = df[counterpart_col].astype(str).str.upper()
    df["year"]        = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    df["value"]       = pd.to_numeric(df[val_col], errors="coerce") / 1_000_000_000.0  # raw USD → billions

    if asset_col is not None:
        df["type"] = df[asset_col].map(IMF_ASSET_CLASS_TO_TYPE)
    else:
        # If no asset_class column, try type column directly
        type_col = next((c for c in ["type", "TYPE"] if c in df.columns), None)
        if type_col:
            df["type"] = pd.to_numeric(df[type_col], errors="coerce")
        else:
            return pd.DataFrame(columns=["investor", "Counterpart", "type", "year", "value"])

    agg = (
        df.dropna(subset=["investor", "Counterpart", "year", "type", "value"])
          .query("value > 0")
          .groupby(["investor", "Counterpart", "year", "type"], as_index=False)["value"]
          .sum()
    )
    return agg


def _load_cpis_reserves(data_dir: Path) -> pd.DataFrame:
    """
    Load IMF PIP reserve bilateral parquet → (issuer_iso3, type, year, reserve_bn).

    Sums all central-bank investors per (issuer, type, year).
    """
    path = data_dir / "pip_bilateral_positions_reserve.parquet"
    if not path.exists():
        # Try the aggregate version
        path = data_dir / "pip_bilateral_positions_reserve_aggregate.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Counterpart", "type", "year", "reserve"])

    df = pd.read_parquet(path)

    counterpart_col = next((c for c in ["COUNTERPART_COUNTRY", "counterpart_country",
                                         "COUNTERPART", "issuer", "counterpart"] if c in df.columns), None)
    time_col        = next((c for c in ["TIME_PERIOD", "time_period", "year"] if c in df.columns), None)
    asset_col       = next((c for c in ["asset_class", "ASSET_CLASS"] if c in df.columns), None)
    val_col         = next((c for c in ["value_usd", "value", "VALUE_USD"] if c in df.columns), None)

    if counterpart_col is None or time_col is None or val_col is None:
        return pd.DataFrame(columns=["Counterpart", "type", "year", "reserve"])

    df["Counterpart"] = df[counterpart_col].astype(str).str.upper()
    df["year"]        = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    df["reserve"]     = pd.to_numeric(df[val_col], errors="coerce") / 1_000_000_000.0  # raw USD → billions

    if asset_col is not None:
        df["type"] = df[asset_col].map(IMF_ASSET_CLASS_TO_TYPE)
    else:
        type_col = next((c for c in ["type", "TYPE"] if c in df.columns), None)
        df["type"] = pd.to_numeric(df[type_col], errors="coerce") if type_col else np.nan

    agg = (
        df.dropna(subset=["Counterpart", "year", "type", "reserve"])
          .query("reserve > 0")
          .groupby(["Counterpart", "year", "type"], as_index=False)["reserve"]
          .sum()
    )
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Latest-year detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_latest_year(data_dir: Path = DATA_DIR) -> int:
    """
    Return the latest year that has data in ALL three core sources:
    OECD T720, BIS DDS, and IMF bilateral.
    Falls back to a single source if others are absent.
    """
    years: list[int] = []

    oecd_path = data_dir / "oecd_t720.parquet"
    if oecd_path.exists():
        df = pd.read_parquet(oecd_path, columns=["time_period"])
        y = pd.to_numeric(df["time_period"], errors="coerce").dropna().astype(int).max()
        years.append(int(y))
        print(f"  OECD T720 max year: {y}")

    bis_path = data_dir / "bis_dds_q.parquet"
    if bis_path.exists():
        df = pd.read_parquet(bis_path)
        time_col = next((c for c in ["TIME_PERIOD", "time_period"] if c in df.columns), None)
        if time_col:
            y = (df[time_col].astype(str).str[:4]
                 .pipe(lambda s: pd.to_numeric(s, errors="coerce"))
                 .dropna().astype(int).max())
            years.append(int(y))
            print(f"  BIS DDS max year: {y}")

    cpis_path = data_dir / "pip_bilateral_positions.parquet"
    if cpis_path.exists():
        df = pd.read_parquet(cpis_path)
        time_col = next((c for c in ["TIME_PERIOD", "time_period", "year"] if c in df.columns), None)
        if time_col:
            y = pd.to_numeric(df[time_col], errors="coerce").dropna().astype(int).max()
            years.append(int(y))
            print(f"  IMF bilateral max year: {y}")

    if not years:
        print("  No parquets found — defaulting to 2023")
        return 2023

    latest = min(years)  # min = latest year with COMPLETE data across all sources
    print(f"  → Latest common year: {latest}")
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# Build amounts outstanding
# ─────────────────────────────────────────────────────────────────────────────

def build_amounts_latest(
    countries: pd.DataFrame,
    year: int,
    data_dir: Path = DATA_DIR,
    local_currency_only: bool = True,
) -> pd.DataFrame:
    """
    Build amounts outstanding for a given year from API parquets.

    Priority:
      Debt  → OECD T720 for OECD members, BIS DDS for non-OECD
      Equity → OECD T720 F5 for OECD members, World Bank WDI for non-OECD

    Returns: (Counterpart, type, year, outstand)  [outstand in USD billions]
    """
    sample_countries = set(countries["Counterpart"].dropna().astype(str).str.upper())

    # ── OECD amounts ─────────────────────────────────────────────────────────
    oecd = _load_oecd_amounts(data_dir)
    oecd_y = oecd[oecd["year"] == year] if not oecd.empty else pd.DataFrame()
    oecd_countries = set(oecd_y["Counterpart"].unique()) if not oecd_y.empty else set()

    # ── BIS amounts (fallback for non-OECD, and check for OECD too) ──────────
    bis = _load_bis_dds_amounts(data_dir)
    bis_y = bis[bis["year"] == year] if not bis.empty else pd.DataFrame()

    # ── IDS foreign-currency correction (BIS WS_DEBT_SEC2_PUB) ──────────────
    # Subtract international bonds issued in foreign currency from total debt amounts
    # to get local-currency-only amounts outstanding (matching Table 1 methodology).
    # Skip this correction when local_currency_only=False (e.g. for Table 2 top-investor
    # calculation where total portfolio value is needed, not just domestic-currency bonds).
    ids_f_lookup: dict[tuple[str, int], float] = {}
    if local_currency_only:
        ids_f = _load_ids_foreign_amounts(data_dir)
        ids_f_y = ids_f[ids_f["year"] == year] if not ids_f.empty else pd.DataFrame()
        if not ids_f_y.empty:
            for _, r in ids_f_y.iterrows():
                ids_f_lookup[(str(r["Counterpart"]), int(r["type"]))] = float(r["ids_foreign"])

    # ── World Bank equity ────────────────────────────────────────────────────
    wb = _load_wb_equity(data_dir)
    wb_y = wb[wb["year"] == year] if not wb.empty else pd.DataFrame()

    parts: list[pd.DataFrame] = []

    # For each country in sample, pick best source
    for ctry in sample_countries:
        for tp in [1, 2, 3]:
            # Debt (type 1/2): prefer OECD, fallback BIS
            if tp in [1, 2]:
                src = oecd_y if ctry in oecd_countries else bis_y
            else:
                # Equity (type 3): prefer OECD F5, fallback World Bank
                src = oecd_y if ctry in oecd_countries else wb_y

            if src.empty:
                continue
            row = src[(src["Counterpart"] == ctry) & (src["type"] == tp)]
            if row.empty:
                continue

            # For debt types, subtract IDS foreign-currency to get local-currency amount
            if tp in [1, 2]:
                raw = row[["Counterpart", "year", "type", "outstand"]].copy()
                foreign_adj = ids_f_lookup.get((ctry, tp), 0.0)
                if foreign_adj > 0:
                    raw = raw.copy()
                    raw["outstand"] = (raw["outstand"] - foreign_adj).clip(lower=SMALL)
                parts.append(raw)
            else:
                parts.append(row[["Counterpart", "year", "type", "outstand"]])

    if not parts:
        return pd.DataFrame(columns=["Counterpart", "year", "type", "outstand"])

    out = pd.concat(parts, ignore_index=True)
    out["Idomestic"] = 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build bilateral holdings
# ─────────���───────────────────────────────────────────────────────────────────

def build_holdings_latest(
    amounts: pd.DataFrame,
    year: int,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Build (investor, issuer, type, year, amount) from IMF bilateral + own residual.

    Steps:
      1. Load IMF PIP bilateral → foreign holdings per issuer
      2. Own holdings = max(outstand − Σ foreign_domestic, ε)
      3. Load reserve bilateral → tag central bank investors (_CR)

    Returns DataFrame with (country, Counterpart, type, year, amount, Iown, Ireserve).
    """
    bilateral = _load_cpis_bilateral(data_dir)
    reserves  = _load_cpis_reserves(data_dir)

    if bilateral.empty:
        print("  WARNING: No bilateral data — domestic share will be 1.0 everywhere")
        # Fall back: everything is domestically held
        df = amounts[["Counterpart", "year", "type", "outstand"]].copy()
        df = df[df["year"] == year].copy()
        df = df.rename(columns={"outstand": "amount", "Counterpart": "country"})
        df["Counterpart"] = df["country"]
        df["Iown"] = 1
        df["Ireserve"] = 0
        return df

    bilat_y = bilateral[bilateral["year"] == year].copy()

    # Sum foreign holdings per issuer per type
    foreign_by_issuer = (
        bilat_y[bilat_y["investor"] != bilat_y["Counterpart"]]
        .groupby(["Counterpart", "type"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "foreign_total"})
    )

    # Merge with amounts outstanding
    amounts_y = amounts[amounts["year"] == year][["Counterpart", "type", "outstand"]].copy()
    merged = amounts_y.merge(foreign_by_issuer, on=["Counterpart", "type"], how="left")
    merged["foreign_total"] = merged["foreign_total"].fillna(0.0)
    merged["own"] = (merged["outstand"] - merged["foreign_total"]).clip(lower=SMALL)

    # Own holdings rows (investor = issuer)
    own_rows = merged[["Counterpart", "type", "own"]].copy()
    own_rows["investor"] = own_rows["Counterpart"]
    own_rows = own_rows.rename(columns={"own": "value"})
    own_rows["Iown"] = 1
    own_rows["Ireserve"] = 0
    own_rows["year"] = year

    # Foreign holdings rows
    foreign_rows = bilat_y[bilat_y["investor"] != bilat_y["Counterpart"]].copy()
    foreign_rows["Iown"] = 0
    foreign_rows["Ireserve"] = 0

    # Tag reserve investors
    res_y = reserves[reserves["year"] == year] if not reserves.empty else pd.DataFrame()
    if not res_y.empty:
        res_rows = res_y.copy()
        res_rows["investor"] = "_CR"
        res_rows["value"]    = res_rows["reserve"]
        res_rows["Iown"]     = 0
        res_rows["Ireserve"] = 1
        all_rows = pd.concat([own_rows, foreign_rows, res_rows], ignore_index=True)
    else:
        all_rows = pd.concat([own_rows, foreign_rows], ignore_index=True)

    all_rows["year"]    = year
    all_rows["country"] = all_rows["investor"]
    return all_rows[["country", "Counterpart", "type", "year", "value", "Iown", "Ireserve"]]


# ─────────────────────────────────────────────────────────────────────────────
# Compute Table 1
# ─────────────────────────────────────────────────────────────────────────────

def compute_table1_latest(
    holdings: pd.DataFrame,
    amounts: pd.DataFrame,
    countries: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Aggregate holdings → Table 1 metrics for a given year.

    Returns: (Counterpart, type, market, domestic, reserve, Name, group)
    """
    # Use amounts outstanding as market value (more reliable than summing holdings)
    amounts_y = (
        amounts[amounts["year"] == year][["Counterpart", "type", "outstand"]]
        .rename(columns={"outstand": "market"})
    )

    # Domestic share from holdings
    hold_y = holdings[holdings["year"] == year].copy()
    own_agg = (
        hold_y[hold_y["Iown"] == 1]
        .groupby(["Counterpart", "type"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "own_total"})
    )
    res_agg = (
        hold_y[hold_y["Ireserve"] == 1]
        .groupby(["Counterpart", "type"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "reserve_total"})
    )

    tbl = amounts_y.copy()
    tbl = tbl.merge(own_agg, on=["Counterpart", "type"], how="left")
    tbl = tbl.merge(res_agg, on=["Counterpart", "type"], how="left")

    tbl["own_total"]     = tbl["own_total"].fillna(0.0)
    tbl["reserve_total"] = tbl["reserve_total"].fillna(0.0)

    tbl["domestic"] = (tbl["own_total"] / tbl["market"].clip(lower=SMALL)).clip(upper=1.0)
    tbl["reserve"]  = (tbl["reserve_total"] / tbl["market"].clip(lower=SMALL)).clip(upper=1.0)

    # Merge country metadata
    ctry_meta = countries[["Counterpart", "Name", "group"]].copy() if "Name" in countries.columns \
        else countries[["Counterpart", "group"]].copy()
    if "Name" not in ctry_meta.columns:
        ctry_meta["Name"] = ctry_meta["Counterpart"]

    tbl = tbl.merge(ctry_meta, on="Counterpart", how="left")
    tbl = tbl[tbl["group"].notna()].copy()
    tbl["group"] = pd.to_numeric(tbl["group"], errors="coerce")
    tbl = tbl[tbl["type"].isin([1, 2, 3])].copy()

    # Ieuro flag (euro area countries joined EMU before 2002)
    EURO_COUNTRIES = {
        "AUT", "BEL", "DEU", "ESP", "FIN", "FRA", "GRC", "IRL",
        "ITA", "LUX", "NLD", "PRT",
    }
    tbl["Ieuro"] = tbl["Counterpart"].isin(EURO_COUNTRIES).astype(int)

    tbl = tbl.sort_values(
        ["type", "group", "Ieuro", "Name"],
        ascending=[True, True, False, True]
    ).reset_index(drop=True)

    return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_table1_latest(tbl: pd.DataFrame, year: int, path: str | None = None) -> None:
    """Print and save Table 1 for a given year in paper format."""
    wide = {}
    for tp in [1, 2, 3]:
        sub = tbl[tbl["type"] == tp][["Name", "group", "market", "domestic", "reserve"]].copy()
        sub = sub.rename(columns={"market": f"mkt{tp}",
                                  "domestic": f"dom{tp}",
                                  "reserve": f"res{tp}"})
        wide[tp] = sub.set_index("Name")

    names_order = tbl[tbl["type"] == 1][["Name", "group"]].copy()
    df = names_order.copy()
    for tp in [1, 2, 3]:
        df = df.merge(wide[tp].drop(columns=["group"], errors="ignore"),
                      left_on="Name", right_index=True, how="left")

    lines: list[str] = []
    SEP = "-" * 100

    lines += [
        f"Table 1 (Latest Data: {year})",
        "Market Values of Financial Assets",
        f"(Source: OECD T720 / BIS DDS / IMF PIP / World Bank WDI — pulled {year})",
        "",
        f"{'':25s}  {'Short-term debt':>26s}  {'Long-term debt':>26s}  {'Equity':>26s}",
        f"{'':25s}  {'Billion':>7s}  {'Domestic':>8s}  {'Share in':>8s}"
        f"  {'Billion':>7s}  {'Domestic':>8s}  {'Share in':>8s}"
        f"  {'Billion':>7s}  {'Domestic':>8s}  {'Share in':>8s}",
        f"{'Issuer':<25s}  {'US$':>7s}  {'share':>8s}  {'reserves':>8s}"
        f"  {'US$':>7s}  {'share':>8s}  {'reserves':>8s}"
        f"  {'US$':>7s}  {'share':>8s}  {'reserves':>8s}",
        SEP,
    ]

    current_group = None
    for _, row in df.iterrows():
        grp = int(row["group"]) if pd.notna(row["group"]) else None
        if grp != current_group:
            current_group = grp
            lines.append(REGION_LABEL.get(grp, ""))

        def fmt_mkt(v):
            if pd.isna(v):
                return f"{'':>7s}"
            return f"{round(v):>7,d}"

        def fmt_share(v):
            if pd.isna(v):
                return f"{'':>8s}"
            return f"{v:>8.2f}"

        name = str(row["Name"])
        line = (f"{name:<25s}"
                f"  {fmt_mkt(row.get('mkt1'))}  {fmt_share(row.get('dom1'))}  {fmt_share(row.get('res1'))}"
                f"  {fmt_mkt(row.get('mkt2'))}  {fmt_share(row.get('dom2'))}  {fmt_share(row.get('res2'))}"
                f"  {fmt_mkt(row.get('mkt3'))}  {fmt_share(row.get('dom3'))}  {fmt_share(row.get('res3'))}")
        lines.append(line)

    lines += [
        SEP,
        f"Note.—Produced from publicly available API data (OECD T720, BIS, IMF PIP, World Bank WDI).",
        "Nationality restatements not applied. Equity uses OECD F5 proxy (may overstate market cap).",
    ]

    output = "\n".join(lines)
    print("\n" + output)

    dest = Path(path) if path else OUTPUT_DIR / f"table_1_{year}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output)
    print(f"\n  [Saved to {dest}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(year: int | None = None) -> None:
    print("=" * 60)
    print("table_1_latest.py  —  Koijen & Yogo extended to latest data")
    print("=" * 60)

    print("\nLoading country metadata ...")
    countries = _load_countries()
    print(f"  {len(countries)} countries in sample")

    if year is None:
        print("\nDetecting latest year with complete data ...")
        year = detect_latest_year(DATA_DIR)

    print(f"\n[Amounts outstanding] Year = {year}")
    amounts = build_amounts_latest(countries, year, DATA_DIR)
    print(f"  {len(amounts)} rows, "
          f"{amounts['Counterpart'].nunique()} issuers, "
          f"types: {sorted(amounts['type'].unique().tolist())}")

    print(f"\n[Bilateral holdings] Year = {year}")
    holdings = build_holdings_latest(amounts, year, DATA_DIR)
    print(f"  {len(holdings)} rows")

    print(f"\n[Table 1] Year = {year}")
    tbl = compute_table1_latest(holdings, amounts, countries, year)

    export_table1_latest(tbl, year)

    # Also produce 2020 version for comparison if data covers it
    if year != 2020:
        oecd = _load_oecd_amounts(DATA_DIR)
        has_2020 = (not oecd.empty and 2020 in oecd["year"].values)
        if has_2020:
            print("\n[Table 1 — 2020 cross-check]")
            amounts_2020  = build_amounts_latest(countries, 2020, DATA_DIR)
            holdings_2020 = build_holdings_latest(amounts_2020, 2020, DATA_DIR)
            tbl_2020      = compute_table1_latest(holdings_2020, amounts_2020, countries, 2020)
            export_table1_latest(tbl_2020, 2020)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Table 1 from latest API data")
    parser.add_argument("--year", type=int, default=None,
                        help="Year to compute (default: auto-detect latest)")
    args = parser.parse_args()
    main(year=args.year)
