"""Pull BIS debt securities data.

Outputs
-------
1) Domestic debt securities statistics (WS_NA_SEC_DSS)
   - Written to: `_data/bis_debt_securities_cleaned.parquet`

2) International debt securities (IDS) *foreign-currency* slice (WS_DEBT_SEC2_PUB)
   - Quarterly, all target issuers
   - Written to: `_data/bis_ids_foreign_currency_q.parquet`

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
    key = "...XW.S11+S12+S13...L.LE.F3.L+LS+S+TT..USD.X1+XDC+_T"
    url = _bis_v2_csv_url("WS_NA_SEC_DSS", key, start_period=start_period, end_period=end_period)
    return pd.read_csv(url)


def pull_ids_foreign_currency(
    *,
    issuer_nat_iso2: list[str],
    start_period: str,
    end_period: str,
) -> pd.DataFrame:
    """Quarterly IDS positions for foreign-currency group (needed for Appendix C)."""

    issuer_seg = "+".join(sorted(set(issuer_nat_iso2)))

    # Dimension order inferred from BIS response columns:
    # FREQ.ISSUER_RES.ISSUER_NAT.ISSUER_BUS_IMM.ISSUER_BUS_ULT.MARKET.ISSUE_TYPE.
    # ISSUE_CUR_GROUP.ISSUE_CUR.ISSUE_OR_MAT.ISSUE_RE_MAT.ISSUE_RATE.ISSUE_RISK.
    # ISSUE_COL.MEASURE
    key = _build_sdmx_key(
        [
            "Q",  # FREQ
            "3P",  # ISSUER_RES (aggregate)
            issuer_seg,  # ISSUER_NAT (multi)
            "",  # ISSUER_BUS_IMM
            "",  # ISSUER_BUS_ULT
            "C",  # MARKET (domestic + international markets: compiled)
            "A",  # ISSUE_TYPE (all)
            "F",  # ISSUE_CUR_GROUP (foreign currency)
            "TO1",  # ISSUE_CUR (amounts outstanding)
            "",  # ISSUE_OR_MAT (keep; we need short vs long)
            "A",  # ISSUE_RE_MAT (all remaining maturity)
            "A",  # ISSUE_RATE
            "A",  # ISSUE_RISK
            "A",  # ISSUE_COL
            "I",  # MEASURE (positions)
        ]
    )

    url = _bis_v2_csv_url(
        "WS_DEBT_SEC2_PUB",
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


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    issuer_nat_iso2 = _target_issuer_nat_iso2()
    print(f"Target issuer countries (ISO2): {len(issuer_nat_iso2)}")

    # 1) Domestic debt securities statistics (existing output)
    df_dom = pull_domestic_debt_securities(start_period=DOMESTIC_START, end_period=DOMESTIC_END)
    if "REF_AREA" in df_dom.columns:
        df_dom = df_dom[df_dom["REF_AREA"].isin(set(issuer_nat_iso2))].copy()

    df_dom = _drop_domestic_columns(df_dom)

    domestic_out = DATA_DIR / "bis_debt_securities_cleaned.parquet"
    df_dom.to_parquet(domestic_out, index=False)
    print("Wrote:", domestic_out)
    print("Domestic rows:", len(df_dom))

    # 2) IDS foreign-currency slice (new output)
    df_ids = pull_ids_foreign_currency(
        issuer_nat_iso2=issuer_nat_iso2,
        start_period=IDS_START,
        end_period=IDS_END,
    )

    for col in ["OBS_VALUE", "UNIT_MULT"]:
        if col in df_ids.columns:
            df_ids[col] = pd.to_numeric(df_ids[col], errors="coerce")

    ids_out = DATA_DIR / "bis_ids_foreign_currency_q.parquet"
    df_ids.to_parquet(ids_out, index=False)
    print("Wrote:", ids_out)
    print("IDS rows:", len(df_ids))
    print("IDS columns:", df_ids.columns.tolist())


if __name__ == "__main__":
    main()
