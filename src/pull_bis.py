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
DOMESTIC_END = str(config("BIS_DOMESTIC_END", default="2024-12-31", cast=str))

# IDS uses quarterly periods like YYYY-Q4
IDS_START = str(config("BIS_IDS_START", default="2003-Q1", cast=str))
IDS_END = str(config("BIS_IDS_END", default="2024-Q4", cast=str))


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


def pull_domestic_debt_securities(
    *,
    issuer_res_iso2: list[str],
    start_period: str,
    end_period: str,
) -> pd.DataFrame:
    """Domestic debt securities outstanding from WS_NA_SEC_DSS (replaces WS_DEBT_SEC2_PUB).

    WS_DEBT_SEC2_PUB was retired by BIS; WS_NA_SEC_DSS is the live replacement.

    Dimension order (WS_NA_SEC_DSS):
    FREQ.ADJUSTMENT.REF_AREA.COUNTERPART_AREA.REF_SECTOR.COUNTERPART_SECTOR.
    CONSOLIDATION.ACCOUNTING_ENTRY.STO.INSTR_ASSET.MATURITY.EXPENDITURE.
    UNIT_MEASURE.CURRENCY_DENOM.VALUATION.PRICES.TRANSFORMATION

    Strategy:
      Pass 1 — REF_SECTOR=S1 (total economy).  Covers ~28 of 37 target countries.
      Pass 2 — REF_SECTOR wildcard for the remaining countries; keeps S13+S11+S12
               aggregate as best available fallback.

    Output columns: REF_AREA, MATURITY (S/L), TIME_PERIOD, OBS_VALUE, UNIT_MULT.
    OBS_VALUE is already in USD billions (UNIT_MULT=9 in the API response).
    """
    iso2_set = set(issuer_res_iso2)

    # Convert start/end to quarterly period format expected by WS_NA_SEC_DSS
    # (e.g. "2003-12-25" → "2003-Q4",  "2024-12-31" → "2024-Q4")
    def _to_q_period(s: str) -> str:
        if "Q" in s.upper():
            return s
        year = s[:4]
        month = int(s[5:7]) if len(s) >= 7 else 12
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"

    q_start = _to_q_period(start_period)
    q_end   = _to_q_period(end_period)

    # Common dimensions:
    # FREQ=Q, ADJUSTMENT=N, COUNTERPART_AREA=XW (world),
    # CONSOLIDATION=N, ACCOUNTING_ENTRY=L (liabilities = issued),
    # STO=LE (amounts outstanding), INSTR_ASSET=F3 (debt securities),
    # MATURITY=S+L, EXPENDITURE=_Z, UNIT_MEASURE=USD,
    # CURRENCY_DENOM=_T (all currencies), VALUATION varies per pass,
    # PRICES=V, TRANSFORMATION=N
    def _make_key(sector_seg: str, valuation: str = "N") -> str:
        return _build_sdmx_key([
            "Q",        # FREQ
            "N",        # ADJUSTMENT
            "",         # REF_AREA — wildcard; filter after fetch
            "XW",       # COUNTERPART_AREA: world (all holders)
            sector_seg, # REF_SECTOR
            "S1",       # COUNTERPART_SECTOR
            "N",        # CONSOLIDATION
            "L",        # ACCOUNTING_ENTRY: liabilities (issued by country)
            "LE",       # STO: amounts outstanding
            "F3",       # INSTR_ASSET: debt securities
            "S+L",      # MATURITY: short-term + long-term
            "_Z",       # EXPENDITURE
            "USD",      # UNIT_MEASURE
            "_T",       # CURRENCY_DENOM: all currencies
            valuation,  # VALUATION: N=nominal, M=market, F=face value
            "V",        # PRICES: current values
            "N",        # TRANSFORMATION: none
        ])

    keep_cols = ["REF_AREA", "MATURITY", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT"]

    def _fetch(key: str) -> pd.DataFrame:
        url = _bis_v2_csv_url("WS_NA_SEC_DSS", key, start_period=q_start, end_period=q_end)
        df = pd.read_csv(url)
        df = df[df["REF_AREA"].isin(iso2_set)].copy()
        return df[[c for c in keep_cols if c in df.columns]]

    # Pass 1: VALUATION=N, S1 (total economy) — covers most OECD members
    df_s1 = _fetch(_make_key("S1", "N"))
    covered = set(df_s1["REF_AREA"].unique()) if not df_s1.empty else set()
    missing = iso2_set - covered

    parts = [df_s1]

    # Pass 2: VALUATION=N, wildcard sector — fallback for countries not reporting S1
    if missing:
        df_fb = _fetch(_make_key("", "N"))
        df_fb = df_fb[df_fb["REF_AREA"].isin(missing)].copy()
        if not df_fb.empty:
            # Sum over sectors to get country totals
            df_fb = (df_fb.groupby(["REF_AREA", "MATURITY", "TIME_PERIOD", "UNIT_MULT"],
                                    as_index=False)["OBS_VALUE"].sum())
            parts.append(df_fb)
            covered |= set(df_fb["REF_AREA"].unique())
            missing = iso2_set - covered

    # Pass 3: VALUATION=M (market value) — for countries only reporting market valuation (e.g. AUS)
    if missing:
        df_m = _fetch(_make_key("S1", "M"))
        df_m = df_m[df_m["REF_AREA"].isin(missing)].copy()
        if df_m.empty:
            df_m = _fetch(_make_key("", "M"))
            df_m = df_m[df_m["REF_AREA"].isin(missing)].copy()
            if not df_m.empty:
                df_m = (df_m.groupby(["REF_AREA", "MATURITY", "TIME_PERIOD", "UNIT_MULT"],
                                      as_index=False)["OBS_VALUE"].sum())
        if not df_m.empty:
            parts.append(df_m)
            covered |= set(df_m["REF_AREA"].unique())
            missing = iso2_set - covered

    # Pass 4: VALUATION=F (face value) — last resort for remaining countries (e.g. THA)
    if missing:
        df_f = _fetch(_make_key("S1", "F"))
        df_f = df_f[df_f["REF_AREA"].isin(missing)].copy()
        if df_f.empty:
            df_f = _fetch(_make_key("", "F"))
            df_f = df_f[df_f["REF_AREA"].isin(missing)].copy()
            if not df_f.empty:
                df_f = (df_f.groupby(["REF_AREA", "MATURITY", "TIME_PERIOD", "UNIT_MULT"],
                                      as_index=False)["OBS_VALUE"].sum())
        if not df_f.empty:
            parts.append(df_f)

    result = pd.concat(parts, ignore_index=True)

    # Deduplicate: BIS API sometimes returns sub-sector rows alongside the S1 aggregate.
    # Keep the maximum value per (REF_AREA, MATURITY, TIME_PERIOD) which equals the aggregate.
    result = (
        result.groupby(["REF_AREA", "MATURITY", "TIME_PERIOD", "UNIT_MULT"], as_index=False)["OBS_VALUE"]
              .max()
    )
    return result


def pull_total_debt_securities(
    *,
    issuer_res_iso2: list[str],
    start_period: str,
    end_period: str,
) -> pd.DataFrame:
    """Alias for pull_domestic_debt_securities — WS_DEBT_SEC2_PUB no longer exists.

    Previously returned total securities across all markets; now returns the same
    WS_NA_SEC_DSS S1 totals as pull_domestic_debt_securities.  Kept for API
    compatibility so callers don't break.
    """
    return pull_domestic_debt_securities(
        issuer_res_iso2=issuer_res_iso2,
        start_period=start_period,
        end_period=end_period,
    )


def pull_ids_foreign_currency(
    *,
    issuer_res_iso2: list[str],
    start_period: str,
    end_period: str,
) -> pd.DataFrame:
    """Quarterly IDS foreign-currency amounts from WS_DEBT_SEC2_PUB (v2 API).

    Returns the face value of international bonds issued in foreign currency by
    residents of each target country.  Used to correct BIS DDS / OECD T720 totals
    (which include all currencies) to obtain local-currency-only amounts outstanding.

    Key dimensions used:
      FREQ           = Q
      ISSUER_RES     = {iso2}   (issuing country — resident basis)
      ISSUER_NAT     = 3P       (aggregate nationality, only value available)
      ISSUER_BUS_IMM = 1        (all immediate parents)
      ISSUER_BUS_ULT = 1        (all ultimate parents)
      MARKET         = C        (international / cross-border market)
      ISSUE_TYPE     = A        (all types)
      ISSUE_CUR_GROUP= F        (foreign-currency bonds only)
      ISSUE_CUR      = TO1      (all non-domestic currencies aggregate)
      ISSUE_OR_MAT   = C+K      (short-term C, long-term K)
      ISSUE_RE_MAT   = A
      ISSUE_RATE     = A
      ISSUE_RISK     = A
      ISSUE_COL      = A
      MEASURE        = I        (amounts outstanding)

    UNIT_MULT=6 in the response → values are USD millions.

    Output columns: ISSUER_RES (ISO2), ISSUE_OR_MAT (C=ST, K=LT),
                    TIME_PERIOD, OBS_VALUE (USD millions).
    """
    rows: list[pd.DataFrame] = []
    for iso2 in issuer_res_iso2:
        key = f"Q.{iso2}.3P.1.1.C.A.F.TO1.C+K.A.A.A.A.I"
        url = (
            f"https://stats.bis.org/api/v2/data/dataflow/BIS/WS_DEBT_SEC2_PUB/1.0/{key}"
            f"?startPeriod={start_period}&endPeriod={end_period}&format=csv"
        )
        try:
            df = pd.read_csv(url)
            if df.empty:
                continue
            keep = ["ISSUER_RES", "ISSUE_OR_MAT", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT"]
            df = df[[c for c in keep if c in df.columns]].copy()
            if "UNIT_MULT" not in df.columns:
                df["UNIT_MULT"] = 6
            rows.append(df)
        except Exception:
            continue  # skip countries where IDS data is unavailable

    if not rows:
        return pd.DataFrame(columns=["ISSUER_RES", "ISSUE_OR_MAT", "TIME_PERIOD",
                                     "OBS_VALUE", "UNIT_MULT"])
    return pd.concat(rows, ignore_index=True)


def pull_ids(
    *,
    start_period: str = "",   # noqa: unused — kept for API compatibility
    end_period: str = "",     # noqa: unused — kept for API compatibility
) -> pd.DataFrame:
    """Deprecated stub — use pull_ids_foreign_currency instead.

    Returns an empty DataFrame so legacy callers don't crash.
    """
    return pd.DataFrame()



def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    issuer_res_iso2 = _target_issuer_nat_iso2()
    print(f"Target issuer countries (ISO2): {len(issuer_res_iso2)}")

    # Domestic debt securities via WS_NA_SEC_DSS (replaces retired WS_DEBT_SEC2_PUB)
    df_dom = pull_domestic_debt_securities(
        issuer_res_iso2=issuer_res_iso2,
        start_period=DOMESTIC_START,
        end_period=DOMESTIC_END,
    )
    domestic_out = DATA_DIR / "bis_dds_q.parquet"
    df_dom.to_parquet(domestic_out, index=False)
    print("Wrote:", domestic_out)
    print(f"DDS rows: {len(df_dom)}  countries: {df_dom['REF_AREA'].nunique() if not df_dom.empty else 0}")

    # IDS foreign-currency slice via WS_DEBT_SEC2_PUB (v2)
    # Used to correct DDS / OECD T720 totals to local-currency-only amounts.
    df_ids = pull_ids_foreign_currency(
        issuer_res_iso2=issuer_res_iso2,
        start_period=IDS_START,
        end_period=IDS_END,
    )
    ids_out = DATA_DIR / "bis_ids_foreign_q.parquet"
    df_ids.to_parquet(ids_out, index=False)
    print("Wrote:", ids_out)
    print(f"IDS rows: {len(df_ids)}  countries: {df_ids['ISSUER_RES'].nunique() if not df_ids.empty else 0}")


if __name__ == "__main__":
    main()
