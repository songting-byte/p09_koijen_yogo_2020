"""Pull BIS debt securities data.

Outputs
-------
1) Domestic debt securities statistics (WS_NA_SEC_DSS)
   - Written to: `_data/bis_debt_securities_cleaned.parquet`

2) International debt securities (IDS) *foreign-currency* slice (WS_DEBT_SEC2_PUB)
   - Quarterly, all target issuers
   - Written to: `_data/bis_ids_foreign_currency_q.parquet`
    - Includes normalized `million_usd = OBS_VALUE * 10**UNIT_MULT / 1e6`

3) International debt securities (IDS) *all-currency* slice (WS_DEBT_SEC2_PUB)
    - Quarterly, all target issuers
    - Written to: `_data/bis_ids_all_currency_q.parquet`
    - Includes normalized `million_usd = OBS_VALUE * 10**UNIT_MULT / 1e6`

4) Total debt securities (all currency, all markets) from NA_SEC flow
    - Source: `WS_NA_SEC_DSS`
    - Quarterly, all target issuers
    - Written to: `_data/bis_total_debt_all_currency_q.parquet`
    - Includes normalized `million_usd = OBS_VALUE * 10**UNIT_MULT / 1e6`

Country coverage
----------------
Targets default to the project's shared 37-country list (see `src/pull_WB.py`)
and are converted from ISO3 to BIS ISO2 codes.

Notes
-----
- We intentionally pull only the *foreign-currency* IDS slice needed for the
  Appendix-C adjustment used elsewhere in this repo.
- BIS SDMX v2 CSV returns `UNIT_MULT`; values are interpreted as
  `OBS_VALUE * 10**UNIT_MULT` in `UNIT_MEASURE`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import config


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

_data_dir_value = config("DATA_DIR")
DATA_DIR = _data_dir_value if isinstance(_data_dir_value, Path) else Path(str(_data_dir_value))

DOMESTIC_START = str(config("BIS_DOMESTIC_START", default="2003-12-25", cast=str))
DOMESTIC_END = str(config("BIS_DOMESTIC_END", default="2020-12-31", cast=str))

# IDS uses quarterly periods like YYYY-Q4
IDS_START = str(config("BIS_IDS_START", default="2003-Q1", cast=str))
IDS_END = str(config("BIS_IDS_END", default="2020-Q4", cast=str))


# --------------------------------------------------------------------------------------
# Country mapping (ISO3 -> ISO2)
# --------------------------------------------------------------------------------------

# Minimal mapping covering the countries used by this project (OECD pull list).
# If you expand the OECD country list, add mappings here.
ISO3_TO_ISO2: dict[str, str] = {
    "AUS": "AU",
    "AUT": "AT",
    "BEL": "BE",
    "BRA": "BR",
    "CAN": "CA",
    "CHE": "CH",
    "CHN": "CN",
    "COL": "CO",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "GRC": "GR",
    "HKG": "HK",
    "HUN": "HU",
    "IND": "IN",
    "ISR": "IL",
    "ITA": "IT",
    "JPN": "JP",
    "KOR": "KR",
    "MEX": "MX",
    "MYS": "MY",
    "NLD": "NL",
    "NOR": "NO",
    "NZL": "NZ",
    "PHL": "PH",
    "POL": "PL",
    "PRT": "PT",
    "RUS": "RU",
    "SGP": "SG",
    "SWE": "SE",
    "THA": "TH",
    "USA": "US",
    "ZAF": "ZA",
}


def _target_issuer_nat_iso2() -> list[str]:
    """Return BIS ISSUER_NAT ISO2 codes for the project's target country list.

    Primary source: `pull_WB.target_ref_areas_iso3()` (37-country list).
    Fallback: scan the OECD parquet if present.
    """

    # Preferred: use the same 37-country target list used across the project.
    try:
        from pull_WB import target_ref_areas_iso3  # type: ignore

        iso3 = [str(x).upper() for x in target_ref_areas_iso3()]
    except Exception:
        # Fallback: scan OECD parquet if available.
        oecd_output_file = config("OECD_OUTPUT_FILE", default="oecd_t720.parquet", cast=str)
        oecd_path = DATA_DIR / Path(str(oecd_output_file))
        if not oecd_path.exists():
            raise
        df = pd.read_parquet(oecd_path, columns=["reference_area"])
        iso3 = (
            df["reference_area"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

    iso3 = [c for c in iso3 if len(c) == 3 and c.isalpha()]
    iso3 = sorted(set(iso3))

    missing = [c for c in iso3 if c not in ISO3_TO_ISO2]
    if missing:
        raise KeyError(
            "Missing ISO3->ISO2 mappings in src/pull_bis.py for: "
            f"{missing}. Add them to ISO3_TO_ISO2."
        )

    iso2 = [ISO3_TO_ISO2[c] for c in iso3]
    return sorted(set(iso2))


# --------------------------------------------------------------------------------------
# BIS helpers
# --------------------------------------------------------------------------------------


def _build_sdmx_key(segments: list[str]) -> str:
    """Join SDMX key segments; empty string means wildcard for that dimension."""

    return ".".join(segments)


def _bis_v2_csv_url(dataset: str, key: str, *, start_period: str, end_period: str) -> str:
    return (
        "https://stats.bis.org/api/v2/data/dataflow/"
        f"BIS/{dataset}/1.0/"
        f"{key}?startPeriod={start_period}&endPeriod={end_period}&format=csv"
    )


# --------------------------------------------------------------------------------------
# Pulls
# --------------------------------------------------------------------------------------


def pull_domestic_debt_securities(*, start_period: str, end_period: str) -> pd.DataFrame:
    # Keep existing behavior: pull broad slice then filter.
    # Do not hard-code COUNTERPART_AREA=XW at pull stage; some BIS issuers
    # report additional sector coverage under other counterpart areas.
    # WS_NA_SEC_DSS has 18 series dimensions. Keep REF_SECTOR broad and include
    # CUST_BREAKDOWN _T/C01/C02 because some issuers publish only C01/C02.
    # We de-duplicate/standardize CUST in post-processing.
    key = _build_sdmx_key(
        [
            "",  # FREQ
            "",  # ADJUSTMENT
            "",  # REF_AREA
            "",  # COUNTERPART_AREA
            "",  # REF_SECTOR
            "",  # COUNTERPART_SECTOR
            "",  # CONSOLIDATION
            "L",  # ACCOUNTING_ENTRY
            "LE",  # STO
            "F3",  # INSTR_ASSET
            "T+L+LS+S+TT",  # MATURITY
            "",  # EXPENDITURE
            "USD",  # UNIT_MEASURE
            "X1+XDC+_T",  # CURRENCY_DENOM
            "",  # VALUATION
            "",  # PRICES
            "",  # TRANSFORMATION
            "_T+C01+C02",  # CUST_BREAKDOWN
        ]
    )
    url = _bis_v2_csv_url("WS_NA_SEC_DSS", key, start_period=start_period, end_period=end_period)
    return pd.read_csv(url)


def pull_ids_foreign_currency(
    *,
    issuer_res_iso2: list[str],
    start_period: str,
    end_period: str,
) -> pd.DataFrame:
    """Quarterly IDS positions for foreign-currency group (needed for Appendix C).

    Uses MEASURE='I' for amounts outstanding.
    """

    issuer_seg = "+".join(sorted(set(issuer_res_iso2)))

    # Dimension order inferred from BIS response columns:
    # FREQ.ISSUER_RES.ISSUER_NAT.ISSUER_BUS_IMM.ISSUER_BUS_ULT.MARKET.ISSUE_TYPE.
    # ISSUE_CUR_GROUP.ISSUE_CUR.ISSUE_OR_MAT.ISSUE_RE_MAT.ISSUE_RATE.ISSUE_RISK.
    # ISSUE_COL.MEASURE
    key = _build_sdmx_key(
        [
            "Q",  # FREQ
            issuer_seg,  # ISSUER_RES (multi: issuer residence)
            "",  # ISSUER_NAT (wildcard)
            "",  # ISSUER_BUS_IMM (all immediate sector classes)
            "",  # ISSUER_BUS_ULT (all ultimate sector classes)
            "C",  # MARKET (domestic + international markets: compiled)
            "A",  # ISSUE_TYPE (all)
            "F",  # ISSUE_CUR_GROUP (foreign currency)
            "TO1",  # ISSUE_CUR (amounts outstanding)
            "A+C+K",  # ISSUE_OR_MAT (all/short/long original maturity)
            "A",  # ISSUE_RE_MAT (all remaining maturity)
            "A",  # ISSUE_RATE
            "A",  # ISSUE_RISK
            "A",  # ISSUE_COL
            "I",  # MEASURE (amounts outstanding)
        ]
    )

    url = _bis_v2_csv_url(
        "WS_DEBT_SEC2_PUB",
        key,
        start_period=start_period,
        end_period=end_period,
    )
    return pd.read_csv(url)


def pull_ids_all_currency(
    *,
    issuer_res_iso2: list[str],
    start_period: str,
    end_period: str,
) -> pd.DataFrame:
    """Quarterly IDS positions across all currency groups (A/D/F).

    Uses MEASURE='I' for amounts outstanding.
    """

    issuer_seg = "+".join(sorted(set(issuer_res_iso2)))

    key = _build_sdmx_key(
        [
            "Q",  # FREQ
            issuer_seg,  # ISSUER_RES
            "",  # ISSUER_NAT
            "",  # ISSUER_BUS_IMM
            "",  # ISSUER_BUS_ULT
            "C",  # MARKET
            "A",  # ISSUE_TYPE
            "",  # ISSUE_CUR_GROUP (all)
            "TO1",  # ISSUE_CUR
            "A+C+K",  # ISSUE_OR_MAT
            "A",  # ISSUE_RE_MAT
            "A",  # ISSUE_RATE
            "A",  # ISSUE_RISK
            "A",  # ISSUE_COL
            "I",  # MEASURE (amounts outstanding)
        ]
    )

    url = _bis_v2_csv_url(
        "WS_DEBT_SEC2_PUB",
        key,
        start_period=start_period,
        end_period=end_period,
    )
    return pd.read_csv(url)


def pull_total_debt_all_currency(
    *,
    issuer_res_iso2: list[str],
    start_period: str,
    end_period: str,
) -> pd.DataFrame:
    """Quarterly total debt securities from WS_NA_SEC_DSS (NA_SEC structure).

    This keeps total-debt output independent from IDS flow.
    """

    issuer_seg = "+".join(sorted(set(issuer_res_iso2)))

    # WS_NA_SEC_DSS dimension order (NA_SEC):
    # FREQ.ADJUSTMENT.REF_AREA.COUNTERPART_AREA.REF_SECTOR.COUNTERPART_SECTOR.
    # CONSOLIDATION.ACCOUNTING_ENTRY.STO.INSTR_ASSET.MATURITY.EXPENDITURE.
    # UNIT_MEASURE.CURRENCY_DENOM.VALUATION.PRICES.TRANSFORMATION.CUST_BREAKDOWN
    key = _build_sdmx_key(
        [
            "Q",  # FREQ
            "",  # ADJUSTMENT
            issuer_seg,  # REF_AREA
            "",  # COUNTERPART_AREA (all; select preferred counterpart downstream)
            "",  # REF_SECTOR (wildcard; S1M is sparse for many BIS-source countries)
            "",  # COUNTERPART_SECTOR
            "",  # CONSOLIDATION
            "L",  # ACCOUNTING_ENTRY (liabilities)
            "LE",  # STO (end-of-period stock)
            "F3",  # INSTR_ASSET (debt securities)
            "T",  # MATURITY (total)
            "",  # EXPENDITURE
            "USD",  # UNIT_MEASURE
            "_T",  # CURRENCY_DENOM (all currencies)
            "",  # VALUATION
            "",  # PRICES
            "",  # TRANSFORMATION
            "_T+C01+C02",  # CUST_BREAKDOWN
        ]
    )

    url = _bis_v2_csv_url(
        "WS_NA_SEC_DSS",
        key,
        start_period=start_period,
        end_period=end_period,
    )
    return pd.read_csv(url)


def _drop_domestic_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "CUST_BREAKDOWN",
        "COMMENT_DSET",
        "REF_PERIOD_DETAIL",
        "REPYEARSTART",
        "REPYEAREND",
        "TIME_FORMAT",
        "TIME_PER_COLLECT",
        "CUST_BREAKDOWN_LB",
        "REF_YEAR_PRICE",
        "DECIMALS",
        "TABLE_IDENTIFIER",
        "TITLE",
        "TITLE_COMPL",
        "UNIT_MULT",
        "LAST_UPDATE",
        "COMPILING_ORG",
        "COLL_PERIOD",
        "COMMENT_TS",
        "GFS_ECOFUNC",
        "GFS_TAXCAT",
        "DATA_COMP",
        "CURRENCY",
        "DISS_ORG",
        "OBS_PRE_BREAK",
        "OBS_STATUS",
        "CONF_STATUS",
        "COMMENT_OBS",
        "EMBARGO_DATE",
        "OBS_EDP_WBB",
    ]
    existing = [c for c in columns_to_drop if c in df.columns]
    return df.drop(columns=existing)


def _normalize_ref_sector_parent(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse NA_SEC sub-sectors to S11/S12/S13 parents when possible.

    This preserves coverage for countries reporting only sub-sectors
    (e.g., S1311) while keeping downstream logic compatible.
    """

    if "REF_SECTOR" not in df.columns:
        return df

    out = df.copy()
    s = out["REF_SECTOR"].astype(str)
    out.loc[s.str.startswith("S11"), "REF_SECTOR"] = "S11"
    out.loc[s.str.startswith("S12"), "REF_SECTOR"] = "S12"
    out.loc[s.str.startswith("S13"), "REF_SECTOR"] = "S13"
    return out


def _prefer_cust_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Select preferred CUST_BREAKDOWN per NA_SEC key: _T > C01 > C02.

    Some countries publish only C01/C02 for debt securities. Enforcing _T
    globally can drop valid country data entirely.
    """

    if "CUST_BREAKDOWN" not in df.columns or df.empty:
        return df

    out = df.copy()
    priority = {"_T": 0, "C01": 1, "C02": 2}
    out["_cust_rank"] = out["CUST_BREAKDOWN"].astype(str).map(priority).fillna(9)

    key_cols = [
        c
        for c in [
            "FREQ",
            "ADJUSTMENT",
            "REF_AREA",
            "ISSUER_RES",
            "COUNTERPART_AREA",
            "REF_SECTOR",
            "COUNTERPART_SECTOR",
            "CONSOLIDATION",
            "ACCOUNTING_ENTRY",
            "STO",
            "INSTR_ASSET",
            "MATURITY",
            "EXPENDITURE",
            "UNIT_MEASURE",
            "CURRENCY_DENOM",
            "VALUATION",
            "PRICES",
            "TRANSFORMATION",
            "TIME_PERIOD",
        ]
        if c in out.columns
    ]

    if not key_cols:
        out = out.drop(columns=["_cust_rank"])
        return out

    out = out.sort_values(key_cols + ["_cust_rank"])
    out = out.drop_duplicates(subset=key_cols, keep="first").copy()
    out["CUST_BREAKDOWN"] = "_T"
    out = out.drop(columns=["_cust_rank"])
    return out


def _prefer_valuation(df: pd.DataFrame) -> pd.DataFrame:
    """Select preferred valuation per NA_SEC key: M > N > F."""

    if "VALUATION" not in df.columns or df.empty:
        return df

    out = df.copy()
    priority = {"M": 0, "N": 1, "F": 2}
    out["_val_rank"] = out["VALUATION"].astype(str).map(priority).fillna(9)

    key_cols = [
        c
        for c in [
            "FREQ",
            "ADJUSTMENT",
            "REF_AREA",
            "ISSUER_RES",
            "COUNTERPART_AREA",
            "REF_SECTOR",
            "COUNTERPART_SECTOR",
            "CONSOLIDATION",
            "ACCOUNTING_ENTRY",
            "STO",
            "INSTR_ASSET",
            "MATURITY",
            "EXPENDITURE",
            "UNIT_MEASURE",
            "CURRENCY_DENOM",
            "PRICES",
            "TRANSFORMATION",
            "TIME_PERIOD",
        ]
        if c in out.columns
    ]

    if not key_cols:
        out = out.drop(columns=["_val_rank"])
        return out

    out = out.sort_values(key_cols + ["_val_rank"])
    out = out.drop_duplicates(subset=key_cols, keep="first").copy()
    out = out.drop(columns=["_val_rank"])
    return out


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    issuer_nat_iso2 = _target_issuer_nat_iso2()
    print(f"Target issuer countries (ISO2): {len(issuer_nat_iso2)}")

    # 1) Domestic debt securities statistics (existing output)
    df_dom = pull_domestic_debt_securities(start_period=DOMESTIC_START, end_period=DOMESTIC_END)
    if "REF_AREA" in df_dom.columns:
        df_dom = df_dom[df_dom["REF_AREA"].isin(set(issuer_nat_iso2))].copy()

    # Keep country coverage when only C01/C02 are published and harmonize to
    # a preferred one-row-per-key output.
    df_dom = _prefer_cust_breakdown(df_dom)
    df_dom = _normalize_ref_sector_parent(df_dom)
    df_dom = _prefer_valuation(df_dom)

    df_dom = _drop_domestic_columns(df_dom)

    domestic_out = DATA_DIR / "bis_debt_securities_cleaned.parquet"
    df_dom.to_parquet(domestic_out, index=False)
    print("Wrote:", domestic_out)
    print("Domestic rows:", len(df_dom))

    # 2) IDS foreign-currency slice (new output)
    df_ids = pull_ids_foreign_currency(
        issuer_res_iso2=issuer_nat_iso2,
        start_period=IDS_START,
        end_period=IDS_END,
    )

    for col in ["OBS_VALUE", "UNIT_MULT"]:
        if col in df_ids.columns:
            df_ids[col] = pd.to_numeric(df_ids[col], errors="coerce")

    # Defensive post-filter in case BIS API expands wildcard behavior in the future.
    ids_filters = {
        "FREQ": "Q",
        "MARKET": "C",
        "ISSUE_TYPE": "A",
        "ISSUE_CUR_GROUP": "F",
        "ISSUE_CUR": "TO1",
        "ISSUE_RE_MAT": "A",
        "ISSUE_RATE": "A",
        "ISSUE_RISK": "A",
        "ISSUE_COL": "A",
        "MEASURE": "I",
    }
    for col, val in ids_filters.items():
        if col in df_ids.columns:
            df_ids = df_ids[df_ids[col].astype(str) == val].copy()

    if "ISSUER_RES" in df_ids.columns:
        df_ids = df_ids[df_ids["ISSUER_RES"].astype(str).isin(set(issuer_nat_iso2))].copy()

    if "ISSUE_OR_MAT" in df_ids.columns:
        df_ids = df_ids[df_ids["ISSUE_OR_MAT"].astype(str).isin({"A", "C", "K"})].copy()

    # When pulling by ISSUER_RES, ISSUER_NAT can contain various BIS classifications;
    # do not filter it further here.

    if {"OBS_VALUE", "UNIT_MULT"}.issubset(df_ids.columns):
        df_ids["million_usd"] = (df_ids["OBS_VALUE"] * (10 ** df_ids["UNIT_MULT"])) / 1e6

    ids_out = DATA_DIR / "bis_ids_foreign_currency_q.parquet"
    df_ids.to_parquet(ids_out, index=False)
    print("Wrote:", ids_out)
    print("IDS rows:", len(df_ids))
    print("IDS columns:", df_ids.columns.tolist())

    # 3) IDS all-currency slice (A/D/F groups)
    df_ids_all = pull_ids_all_currency(
        issuer_res_iso2=issuer_nat_iso2,
        start_period=IDS_START,
        end_period=IDS_END,
    )

    for col in ["OBS_VALUE", "UNIT_MULT"]:
        if col in df_ids_all.columns:
            df_ids_all[col] = pd.to_numeric(df_ids_all[col], errors="coerce")

    ids_all_filters = {
        "FREQ": "Q",
        "MARKET": "C",
        "ISSUE_TYPE": "A",
        "ISSUE_CUR": "TO1",
        "ISSUE_RE_MAT": "A",
        "ISSUE_RATE": "A",
        "ISSUE_RISK": "A",
        "ISSUE_COL": "A",
        "MEASURE": "I",
    }
    for col, val in ids_all_filters.items():
        if col in df_ids_all.columns:
            df_ids_all = df_ids_all[df_ids_all[col].astype(str) == val].copy()

    if "ISSUER_RES" in df_ids_all.columns:
        df_ids_all = df_ids_all[df_ids_all["ISSUER_RES"].astype(str).isin(set(issuer_nat_iso2))].copy()

    if "ISSUE_OR_MAT" in df_ids_all.columns:
        df_ids_all = df_ids_all[df_ids_all["ISSUE_OR_MAT"].astype(str).isin({"A", "C", "K"})].copy()

    if "ISSUE_CUR_GROUP" in df_ids_all.columns:
        df_ids_all = df_ids_all[df_ids_all["ISSUE_CUR_GROUP"].astype(str).isin({"A", "D", "F"})].copy()

    if {"OBS_VALUE", "UNIT_MULT"}.issubset(df_ids_all.columns):
        df_ids_all["million_usd"] = (df_ids_all["OBS_VALUE"] * (10 ** df_ids_all["UNIT_MULT"])) / 1e6

    ids_all_out = DATA_DIR / "bis_ids_all_currency_q.parquet"
    df_ids_all.to_parquet(ids_all_out, index=False)
    print("Wrote:", ids_all_out)
    print("IDS all-currency rows:", len(df_ids_all))

    # 4) Total debt securities all-currency slice (MEASURE=I)
    df_tot = pull_total_debt_all_currency(
        issuer_res_iso2=issuer_nat_iso2,
        start_period=IDS_START,
        end_period=IDS_END,
    )

    for col in ["OBS_VALUE", "UNIT_MULT"]:
        if col in df_tot.columns:
            df_tot[col] = pd.to_numeric(df_tot[col], errors="coerce")

    tot_filters = {
        "FREQ": "Q",
        "ACCOUNTING_ENTRY": "L",
        "STO": "LE",
        "INSTR_ASSET": "F3",
        "MATURITY": "T",
        "UNIT_MEASURE": "USD",
        "CURRENCY_DENOM": "_T",
    }
    for col, val in tot_filters.items():
        if col in df_tot.columns:
            df_tot = df_tot[df_tot[col].astype(str) == val].copy()

    if "REF_AREA" in df_tot.columns:
        df_tot = df_tot[df_tot["REF_AREA"].astype(str).isin(set(issuer_nat_iso2))].copy()
        # Normalize naming to preserve downstream compatibility.
        df_tot = df_tot.rename(columns={"REF_AREA": "ISSUER_RES"})

    # Keep country coverage when only C01/C02 are published and harmonize to
    # a preferred one-row-per-key output.
    df_tot = _prefer_cust_breakdown(df_tot)
    df_tot = _normalize_ref_sector_parent(df_tot)
    df_tot = _prefer_valuation(df_tot)

    if "ISSUER_RES" in df_tot.columns:
        df_tot = df_tot[df_tot["ISSUER_RES"].astype(str).isin(set(issuer_nat_iso2))].copy()

    if "UNIT_MULT" not in df_tot.columns:
        df_tot["UNIT_MULT"] = 0
    else:
        df_tot["UNIT_MULT"] = pd.to_numeric(df_tot["UNIT_MULT"], errors="coerce").fillna(0)

    if {"OBS_VALUE", "UNIT_MULT"}.issubset(df_tot.columns):
        df_tot["million_usd"] = (df_tot["OBS_VALUE"] * (10 ** df_tot["UNIT_MULT"])) / 1e6

    tot_out = DATA_DIR / "bis_total_debt_all_currency_q.parquet"
    df_tot.to_parquet(tot_out, index=False)
    print("Wrote:", tot_out)
    print("Total debt rows:", len(df_tot))


if __name__ == "__main__":
    main()
