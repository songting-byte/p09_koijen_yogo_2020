"""tidy_data.py
==============
Consolidates all raw API parquets into a single tidy (long-form) dataset
suitable for downstream analysis.

This module is the sole data-cleaning layer of the pipeline.  It reads the
four raw parquets produced by the pull_* scripts (BIS DDS, OECD T720, IMF
PIP bilateral, World Bank WDI) and writes one unified parquet per asset
class to ``_data/tidy_amounts.parquet`` and one bilateral-flow parquet to
``_data/tidy_bilateral.parquet``.  All analysis files (table_1_latest.py,
table_2_latest.py, summary_stats.py) should import their input data from
these tidy files rather than re-parsing the raw parquets independently.

Run:
    python tidy_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import config

DATA_DIR: Path = config("DATA_DIR")
OUTPUT_DIR: Path = config("OUTPUT_DIR")

# ISO2 → ISO3 map for BIS data
_ISO2_TO_ISO3: dict[str, str] = {
    "AU": "AUS", "AT": "AUT", "BE": "BEL", "BR": "BRA", "CA": "CAN",
    "CH": "CHE", "CN": "CHN", "CO": "COL", "CZ": "CZE", "DE": "DEU",
    "DK": "DNK", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR",
    "GR": "GRC", "HK": "HKG", "HU": "HUN", "IN": "IND", "IL": "ISR",
    "IT": "ITA", "JP": "JPN", "KR": "KOR", "MX": "MEX", "MY": "MYS",
    "NL": "NLD", "NO": "NOR", "NZ": "NZL", "PH": "PHL", "PL": "POL",
    "PT": "PRT", "RU": "RUS", "SG": "SGP", "SE": "SWE", "TH": "THA",
    "US": "USA", "ZA": "ZAF",
}

_MATURITY_TO_TYPE = {"S": 1, "L": 2}


# ---------------------------------------------------------------------------
# BIS DDS  →  tidy amounts (debt only)
# ---------------------------------------------------------------------------

def _tidy_bis_dds(data_dir: Path) -> pd.DataFrame:
    """Return BIS DDS Q4 observations as (iso3, year, type, outstand_bn)."""
    path = data_dir / "bis_dds_q.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "year", "type", "outstand_bn", "source"])
    df = pd.read_parquet(path)
    df["iso3"] = df["REF_AREA"].map(_ISO2_TO_ISO3)
    df["year"] = pd.to_numeric(
        df["TIME_PERIOD"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    ).astype("Int64")
    # Keep only Q4
    df = df[df["TIME_PERIOD"].astype(str).str.endswith("Q4")].copy()
    df["type"] = df["MATURITY"].map(_MATURITY_TO_TYPE)
    df["outstand_bn"] = df["OBS_VALUE"] * (10.0 ** (df["UNIT_MULT"] - 9.0))
    df["source"] = "BIS_DDS"
    return (
        df.dropna(subset=["iso3", "year", "type", "outstand_bn"])
        .groupby(["iso3", "year", "type", "source"], as_index=False)["outstand_bn"]
        .max()
    )


# ---------------------------------------------------------------------------
# OECD T720  →  tidy amounts (debt + equity proxy)
# ---------------------------------------------------------------------------

_OECD_INSTR_TO_TYPE = {"F3": {1: "S", 2: "L"}, "F5": {3: None}}


def _tidy_oecd_t720(data_dir: Path) -> pd.DataFrame:
    """Return OECD T720 annual observations as (iso3, year, type, outstand_bn)."""
    path = data_dir / "oecd_t720.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "year", "type", "outstand_bn", "source"])
    df = pd.read_parquet(path)
    df["year"] = pd.to_numeric(df["time_period"], errors="coerce").astype("Int64")
    df["outstand_bn"] = pd.to_numeric(df["value"], errors="coerce") / 1_000.0  # millions → bn
    # Map instrument + maturity to type 1/2/3
    mat_map = {"S": 1, "L": 2}
    eq_map = {"F5": 3, "F51": 3, "F519": 3}
    rows = []
    for _, r in df.iterrows():
        instr = str(r.get("financial_instrument", ""))
        mat = str(r.get("original_maturity", ""))
        if instr == "F3" and mat in mat_map:
            tp = mat_map[mat]
        elif instr in eq_map and mat in ("_Z", "T", ""):
            tp = 3
        else:
            continue
        rows.append({"iso3": str(r["reference_area"]), "year": r["year"],
                     "type": tp, "outstand_bn": r["outstand_bn"], "source": "OECD_T720"})
    if not rows:
        return pd.DataFrame(columns=["iso3", "year", "type", "outstand_bn", "source"])
    out = pd.DataFrame(rows).dropna(subset=["iso3", "year", "type", "outstand_bn"])
    return out.groupby(["iso3", "year", "type", "source"], as_index=False)["outstand_bn"].sum()


# ---------------------------------------------------------------------------
# World Bank WDI  →  equity amounts
# ---------------------------------------------------------------------------

def _tidy_wb_equity(data_dir: Path) -> pd.DataFrame:
    """Return World Bank market-cap as (iso3, year, type=3, outstand_bn)."""
    path = data_dir / "wb_data360_wdi_selected.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["iso3", "year", "type", "outstand_bn", "source"])
    df = pd.read_parquet(path)
    # Detect columns
    iso_col = next((c for c in ["reference_area", "iso3", "COUNTRY"] if c in df.columns), None)
    yr_col = next((c for c in ["time_period", "year", "TIME_PERIOD"] if c in df.columns), None)
    val_col = next((c for c in ["value", "OBS_VALUE", "obs_value"] if c in df.columns), None)
    ind_col = next((c for c in ["indicator", "INDICATOR", "series_id"] if c in df.columns), None)
    if not all([iso_col, yr_col, val_col]):
        return pd.DataFrame(columns=["iso3", "year", "type", "outstand_bn", "source"])
    df = df.copy()
    # Filter to market-cap indicator if present
    if ind_col:
        df = df[df[ind_col].astype(str).str.contains("LCAP|market_cap", case=False, na=False)]
    df["iso3"] = df[iso_col].astype(str).str.upper()
    df["year"] = pd.to_numeric(df[yr_col], errors="coerce").astype("Int64")
    df["outstand_bn"] = pd.to_numeric(df[val_col], errors="coerce") / 1e9
    df["type"] = 3
    df["source"] = "WB_WDI"
    return (
        df.dropna(subset=["iso3", "year", "outstand_bn"])
        .groupby(["iso3", "year", "type", "source"], as_index=False)["outstand_bn"]
        .max()
    )


# ---------------------------------------------------------------------------
# IMF PIP bilateral  →  tidy cross-border holdings
# ---------------------------------------------------------------------------

def _tidy_pip_bilateral(data_dir: Path) -> pd.DataFrame:
    """Return IMF PIP bilateral as (investor, issuer, year, type, value_bn)."""
    path = data_dir / "pip_bilateral_positions.parquet"
    if not path.exists():
        return pd.DataFrame(
            columns=["investor", "issuer", "year", "type", "value_bn", "source"]
        )
    df = pd.read_parquet(path)
    inv_col = next((c for c in ["COUNTRY", "country", "investor"] if c in df.columns), None)
    cpt_col = next((c for c in ["COUNTERPART_COUNTRY", "counterpart_country",
                                 "COUNTERPART", "counterpart"] if c in df.columns), None)
    yr_col = next((c for c in ["TIME_PERIOD", "time_period", "year"] if c in df.columns), None)
    ac_col = next((c for c in ["asset_class", "ASSET_CLASS"] if c in df.columns), None)
    val_col = next((c for c in ["value", "VALUE_USD", "value_usd"] if c in df.columns), None)

    if not all([inv_col, cpt_col, yr_col, val_col]):
        return pd.DataFrame(
            columns=["investor", "issuer", "year", "type", "value_bn", "source"]
        )

    df = df.copy()
    df["investor"] = df[inv_col].astype(str).str.upper()
    df["issuer"] = df[cpt_col].astype(str).str.upper()
    df["year"] = pd.to_numeric(df[yr_col], errors="coerce").astype("Int64")
    df["value_bn"] = pd.to_numeric(df[val_col], errors="coerce") / 1e9

    _AC_MAP = {"ST_DEBT": 1, "LT_DEBT": 2, "EQUITY": 3,
               "1": 1, "2": 2, "3": 3}
    if ac_col is not None:
        df["type"] = df[ac_col].astype(str).map(_AC_MAP)
    else:
        tp_col = next((c for c in ["type", "TYPE"] if c in df.columns), None)
        df["type"] = pd.to_numeric(df[tp_col], errors="coerce") if tp_col else None

    df["source"] = "IMF_PIP"
    return (
        df.dropna(subset=["investor", "issuer", "year", "type", "value_bn"])
        [["investor", "issuer", "year", "type", "value_bn", "source"]]
    )


# ---------------------------------------------------------------------------
# Build tidy amounts (best source per country/year/type)
# ---------------------------------------------------------------------------

def build_tidy_amounts(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Merge BIS DDS, OECD T720, and WB WDI into one long-form amounts table.

    Priority for debt (type 1/2): OECD T720 for OECD members, BIS DDS otherwise.
    Priority for equity (type 3): OECD T720 F5, World Bank WDI otherwise.
    Returns columns: iso3, year, type, outstand_bn, source.
    """
    bis = _tidy_bis_dds(data_dir)
    oecd = _tidy_oecd_t720(data_dir)
    wb = _tidy_wb_equity(data_dir)

    oecd_countries = set(oecd["iso3"].unique())

    # For debt: OECD wins; for equity: OECD wins; fill gaps with BIS/WB
    parts = [oecd]
    bis_debt = bis[bis["type"].isin([1, 2])]
    bis_extra = bis_debt[~bis_debt["iso3"].isin(oecd_countries)]
    parts.append(bis_extra)

    wb_extra = wb[~wb["iso3"].isin(oecd_countries)]
    parts.append(wb_extra)

    combined = pd.concat(parts, ignore_index=True)
    return combined.sort_values(["iso3", "year", "type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building tidy amounts …")
    amounts = build_tidy_amounts(DATA_DIR)
    out_amounts = DATA_DIR / "tidy_amounts.parquet"
    amounts.to_parquet(out_amounts, index=False)
    print(f"  Wrote {len(amounts):,} rows → {out_amounts}")

    print("Building tidy bilateral flows …")
    bilateral = _tidy_pip_bilateral(DATA_DIR)
    out_bilateral = DATA_DIR / "tidy_bilateral.parquet"
    bilateral.to_parquet(out_bilateral, index=False)
    print(f"  Wrote {len(bilateral):,} rows → {out_bilateral}")


if __name__ == "__main__":
    main()
