"""process_script.py
====================
Replicates Data0 (partial) + Data2 + Data3 + Data4 + Summary0
from Koijen & Yogo (2020), restricted to Tables 1 and 2.

Data1 (Nelson-Siegel yield curves) is NOT needed for Tables 1 & 2 and is skipped.

Input priority:
  1. Python pull script outputs in DATA_DIR (_data/)
  2. Pre-processed Stata .dta files in dataverse Code/1 Data/  (automatic fallback)

Static lookup files (no Python pull equivalent — always loaded from dataverse):
  Countries.dta, Restatement_bilateral.dta, Restatement_issuance.dta

Outputs (in DATA_DIR):
  data3.parquet               — holdings by (year, country, counterpart, type)
  data4_for_tables.parquet    — merged final dataset
  table1_market_values.csv    — Table 1: Market Values of Financial Assets
  table2_top_investors.csv    — Table 2: Top Ten Investors by Asset Class
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── settings ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from settings import config

DATA_DIR = Path(str(config("DATA_DIR")))
DATAVERSE_DIR = (
    Path(__file__).parent.parent.parent
    / "dataverse_files (1)"
    / "Code"
    / "1 Data"
)

YEAR_MIN = 2003
YEAR_MAX = 2020
SMALL = 1e-6   # $global small from Stata global.do


# ============================================================================
# I/O helpers
# ============================================================================

def _dta(filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_stata(DATAVERSE_DIR / filename, **kwargs)


def _parquet_or_dta(parquet_name: str, dta_name: str, **dta_kw) -> pd.DataFrame:
    p = DATA_DIR / parquet_name
    if p.exists():
        return pd.read_parquet(p)
    print(f"  {parquet_name} not found — falling back to {dta_name}")
    return _dta(dta_name, **dta_kw)


# ============================================================================
# LOOKUP TABLES  (always from dataverse .dta)
# ============================================================================

def load_countries() -> pd.DataFrame:
    """Countries.dta — country-level metadata."""
    df = _dta("Countries.dta")
    keep = [c for c in ("country", "Name", "MSCI", "Yeuro", "Ynat") if c in df.columns]
    return df[keep].copy()


def load_restatement_bilateral() -> pd.DataFrame:
    """Restatement_bilateral.dta — (year, country, Counterpart, type) → (counterpart, Value)."""
    return _dta("Restatement_bilateral.dta")


def load_restatement_issuance() -> pd.DataFrame:
    """Restatement_issuance.dta — (year, Counterpart, type) → (counterpart, Value)."""
    return _dta("Restatement_issuance.dta")


# ============================================================================
# DATA0 (partial): equity outstanding + MSCI market value
# ============================================================================

def _load_wb_outstand() -> pd.DataFrame:
    """World Bank CM.MKT.LCAP.CD → outstand (USD billions) per (year, Counterpart)."""
    p = DATA_DIR / "wb_data360_wdi_selected.parquet"
    if not p.exists():
        p = DATA_DIR / "WDI_bundle.parquet"

    if p.exists():
        df = pd.read_parquet(p)
        # Tidy format: columns INDICATOR, REF_AREA, TIME_PERIOD, value (or OBS_VALUE)
        val_col = "value" if "value" in df.columns else "OBS_VALUE"
        ind_col = next((c for c in df.columns if "INDICATOR" in c.upper()), None)
        area_col = next((c for c in df.columns if "REF_AREA" in c.upper()), None)
        time_col = next((c for c in df.columns if "TIME_PERIOD" in c.upper()), None)

        if ind_col and area_col and time_col:
            mkt_mask = df[ind_col].astype(str).str.replace(".", "", regex=False).str.upper().str.contains("CMMKTLCAP")
            df = df[mkt_mask].copy()
            df = df.rename(columns={area_col: "Counterpart", time_col: "year", val_col: "outstand"})
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["outstand"] = pd.to_numeric(df["outstand"], errors="coerce") / 1e9  # USD → USD bn
            return df[["year", "Counterpart", "outstand"]].dropna()

    # Fallback: WorldBank.dta (already in USD billions from WorldBank.do)
    print("  WB parquet not found — using WorldBank.dta fallback")
    df = _dta("WorldBank.dta", columns=["year", "country", "outstand"])
    return df.rename(columns={"country": "Counterpart"})[["year", "Counterpart", "outstand"]].dropna()


def _load_msci_market() -> pd.DataFrame:
    """MSCI equity market value (USD billions) per (year, Counterpart)."""
    p = DATA_DIR / "data_msci_datastream.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # Python pull output: metric, entity_id (ISO3), obs_date, value
        mv = df[df["metric"].astype(str).str.lower() == "market equity value usd"].copy()
        mv["obs_date"] = pd.to_datetime(mv["obs_date"], errors="coerce")
        mv = mv[mv["obs_date"].dt.month == 12].copy()  # year-end (Stata: keep if _n==_N by year country)
        mv["year"] = mv["obs_date"].dt.year
        mv = mv.groupby(["year", "entity_id"])["value"].sum().reset_index()
        mv = mv.rename(columns={"entity_id": "Counterpart", "value": "market"})
        # WRDS MV in USD millions → USD billions
        mv["market"] = mv["market"] / 1e3
        return mv[["year", "Counterpart", "market"]]

    # Fallback: MSCI.dta (already in USD billions from MSCI.do)
    msci_dta = DATAVERSE_DIR / "MSCI.dta"
    if msci_dta.exists():
        df = pd.read_stata(msci_dta, columns=["year", "country", "market"])
        return df.rename(columns={"country": "Counterpart"})[["year", "Counterpart", "market"]]

    print("  WARNING: No MSCI data found — equity extrapolation will be impaired")
    return pd.DataFrame(columns=["year", "Counterpart", "market"])


def _extrapolate_with_growth(df: pd.DataFrame, value_col: str, growth_col: str) -> pd.DataFrame:
    """
    Forward-then-backward extrapolation:
      value[t] = value[last_observed] * growth_index[t] / growth_index[last_observed]

    Mirrors Stata:
      by country: replace outstand = outstand[_n-1]*market/market[_n-1] if missing(outstand)
    """
    df = df.copy()
    # Forward fill the "anchor" outstand and its corresponding market level
    df["_anchor"] = df[value_col].copy()
    df["_manchor"] = df[growth_col].where(df[value_col].notna())

    df["_anchor"] = df["_anchor"].ffill()
    df["_manchor"] = df["_manchor"].ffill()

    fwd_mask = df[value_col].isna() & df["_anchor"].notna() & df["_manchor"].notna() & (df["_manchor"] != 0)
    df.loc[fwd_mask, value_col] = df.loc[fwd_mask, "_anchor"] * df.loc[fwd_mask, growth_col] / df.loc[fwd_mask, "_manchor"]

    # Backward fill
    df["_anchor"] = df[value_col].bfill()
    df["_manchor"] = df[growth_col].where(df[value_col].notna()).bfill()

    bwd_mask = df[value_col].isna() & df["_anchor"].notna() & df["_manchor"].notna() & (df["_manchor"] != 0)
    df.loc[bwd_mask, value_col] = df.loc[bwd_mask, "_anchor"] * df.loc[bwd_mask, growth_col] / df.loc[bwd_mask, "_manchor"]

    return df.drop(columns=["_anchor", "_manchor"])


def build_data0_equity() -> pd.DataFrame:
    """
    Data0 + Data2 Step 1: stock market capitalisation extrapolated with MSCI market value.
    Returns (year, Counterpart, outstand) in USD billions.

    Stata: sort Counterpart year
           by Counterpart: replace outstand = outstand[_n-1]*market/market[_n-1] if missing(outstand)
           gsort Counterpart -year
           by Counterpart: replace outstand = outstand[_n-1]*market/market[_n-1] if missing(outstand)
    """
    wb = _load_wb_outstand()
    msci = _load_msci_market()

    df = wb.merge(msci, on=["year", "Counterpart"], how="outer")
    df = df.sort_values(["Counterpart", "year"])

    parts = []
    for _, g in df.groupby("Counterpart"):
        g = g.sort_values("year").reset_index(drop=True)
        g = _extrapolate_with_growth(g, "outstand", "market")
        parts.append(g)

    result = pd.concat(parts, ignore_index=True)
    result = result.dropna(subset=["outstand"])
    result["year"] = result["year"].astype(int)
    return result[["year", "Counterpart", "outstand"]].copy()


# ============================================================================
# DATA2: Amounts Outstanding
# ============================================================================

def _load_oecd() -> pd.DataFrame:
    """
    OECD financial balance sheets — amount outstanding (USD billions) per
    (year, Counterpart, type).  type: 0=all debt, 1=ST, 2=LT, 3=equity, 4=fund shares.

    Mirrors OECD.do output (OECD.dta).
    """
    p = DATA_DIR / "oecd_t720.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # Columns from pull_oecd.py: reference_area, time_period, financial_instrument, value
        instr_map = {"F3": 0, "F3S": 1, "F3L": 2, "F5": 5, "F51": 3, "F519": 99, "F52": 4}
        df = df.rename(columns={"reference_area": "Counterpart", "time_period": "year"})
        df["type"] = df["financial_instrument"].map(instr_map)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["outstand"] = pd.to_numeric(df["value"], errors="coerce") / 1e3  # USD millions → USD billions
        # Drop F5 aggregate and F519 other equity (reallocated in OECD.do)
        df = df[df["type"].isin([0, 1, 2, 3, 4])].copy()
        df["type"] = df["type"].astype(int)
        return df[["year", "Counterpart", "type", "outstand"]].dropna()

    # Fallback: OECD.dta (already in USD billions, already has correct types)
    print("  oecd_t720.parquet not found — using OECD.dta fallback")
    df = _dta("OECD.dta")
    return df[["year", "Counterpart", "type", "outstand"]].dropna()


def _load_bis_dds() -> pd.DataFrame:
    """
    BIS domestic debt securities — annual (Q4), USD billions, (year, Counterpart, type).
    type: 0=all, 1=ST (C), 2=LT (K).
    """
    p = DATA_DIR / "bis_dds_q.parquet"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_parquet(p)

    # Keep Q4 as annual representative
    df = df[df["TIME_PERIOD"].astype(str).str.endswith("Q4")].copy()
    df["year"] = df["TIME_PERIOD"].astype(str).str[:4].astype(int)

    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    if "UNIT_MULT" in df.columns:
        mult = pd.to_numeric(df["UNIT_MULT"], errors="coerce").fillna(0).astype(int)
        df["OBS_VALUE"] = df["OBS_VALUE"] * (10 ** mult)
    df["dds"] = df["OBS_VALUE"] / 1e9  # raw USD → USD billions

    mat_map = {"A": 0, "C": 1, "K": 2}
    df["type"] = df["ISSUE_OR_MAT"].map(mat_map)

    # ISO2 → ISO3
    from pull_bis import ISO3_TO_ISO2
    iso2_to_3 = {v: k for k, v in ISO3_TO_ISO2.items()}
    df["Counterpart"] = df["ISSUER_RES"].map(iso2_to_3)

    df = df.groupby(["year", "Counterpart", "type"])["dds"].sum().reset_index()
    return df.dropna(subset=["Counterpart", "type"])


def _load_bis_ids() -> pd.DataFrame:
    """
    BIS international debt securities — annual (Q4), all currencies, USD billions,
    (year, Counterpart, type).  Used to subtract from OECD domestic outstanding.
    """
    p = DATA_DIR / "bis_ids_q.parquet"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_parquet(p)

    df = df[df["TIME_PERIOD"].astype(str).str.endswith("Q4")].copy()
    df["year"] = df["TIME_PERIOD"].astype(str).str[:4].astype(int)

    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    if "UNIT_MULT" in df.columns:
        mult = pd.to_numeric(df["UNIT_MULT"], errors="coerce").fillna(0).astype(int)
        df["OBS_VALUE"] = df["OBS_VALUE"] * (10 ** mult)
    df["ids"] = df["OBS_VALUE"] / 1e9

    mat_map = {"A": 0, "C": 1, "K": 2}
    df["type"] = df["ISSUE_OR_MAT"].map(mat_map)

    from pull_bis import ISO3_TO_ISO2
    iso2_to_3 = {v: k for k, v in ISO3_TO_ISO2.items()}
    df["Counterpart"] = df["ISSUER_RES"].map(iso2_to_3)

    # All-currency total (ISSUE_CUR_GROUP == 'A') for the domestic IDS subtraction
    if "ISSUE_CUR_GROUP" in df.columns:
        df = df[df["ISSUE_CUR_GROUP"] == "A"].copy()

    df = df.groupby(["year", "Counterpart", "type"])["ids"].sum().reset_index()
    return df.dropna(subset=["Counterpart", "type"])


def build_data2(countries: pd.DataFrame) -> pd.DataFrame:
    """
    Data2.do equivalent — amounts outstanding (USD billions) by
    (year, counterpart, type, Idomestic).

    Steps mirror Data2.do:
      1. Stock market cap from WorldBank + MSCI extrapolation
      2. OECD debt + equity, extrapolated with MSCI
      3. Merge BIS; subtract IDS from OECD domestic; BIS as fallback
      4. Restate residency → nationality (Restatement_issuance)
    """
    print("Building Data2 (amounts outstanding)...")

    msci = _load_msci_market()
    restat = load_restatement_issuance()

    # -- Step 1: Stock market cap --------------------------------------------------
    stock = build_data0_equity()
    stock["type"] = 3

    # -- Step 2: OECD debt + equity extrapolated with MSCI ------------------------
    oecd = _load_oecd()
    oecd_m = oecd.merge(msci, on=["year", "Counterpart"], how="left")

    parts = []
    for (ctry, tp), g in oecd_m.groupby(["Counterpart", "type"]):
        g = g.sort_values("year").reset_index(drop=True)
        if tp in (3, 4) and "market" in g.columns:
            g = _extrapolate_with_growth(g, "outstand", "market")
        parts.append(g)

    oecd_extrap = pd.concat(parts, ignore_index=True) if parts else oecd_m.copy()
    oecd_extrap = oecd_extrap.dropna(subset=["outstand"])

    # -- Step 3: Merge BIS --------------------------------------------------------
    dds = _load_bis_dds()
    ids = _load_bis_ids()

    if not ids.empty:
        # Stata: replace outstand = max(outstand - ids, 0) if !missing(outstand - ids)
        oecd_extrap = oecd_extrap.merge(
            ids.rename(columns={"ids": "_ids"}),
            on=["year", "Counterpart", "type"],
            how="left",
        )
        oecd_extrap["outstand"] = np.where(
            oecd_extrap["_ids"].notna(),
            np.maximum(oecd_extrap["outstand"] - oecd_extrap["_ids"], 0),
            oecd_extrap["outstand"],
        )
        oecd_extrap = oecd_extrap.drop(columns=["_ids"])

    if not dds.empty:
        # Extrapolate OECD backward with BIS DDS growth rates where OECD is missing
        # Stata: gsort Counterpart type currency -year
        #        by ...: replace outstand = outstand[_n-1]*dds/dds[_n-1] if missing(outstand)
        oecd_extrap = oecd_extrap.merge(
            dds.rename(columns={"dds": "_dds"}),
            on=["year", "Counterpart", "type"],
            how="outer",
        )
        parts2 = []
        for (ctry, tp), g in oecd_extrap.groupby(["Counterpart", "type"]):
            g = g.sort_values("year").reset_index(drop=True)
            g = _extrapolate_with_growth(g, "outstand", "_dds")
            parts2.append(g)
        oecd_extrap = pd.concat(parts2, ignore_index=True) if parts2 else oecd_extrap
        # Use BIS DDS directly if OECD still missing
        oecd_extrap["outstand"] = oecd_extrap["outstand"].fillna(oecd_extrap["_dds"])
        oecd_extrap = oecd_extrap.drop(columns=["_dds"])

    # Keep only ST(1), LT(2), equity(3), fund(4); drop all-debt aggregate(0)
    oecd_extrap = oecd_extrap[oecd_extrap["type"].isin([1, 2, 3, 4])].copy()

    # Supplement equity with WB stock market cap for countries not in OECD
    oecd_equity_ctry = set(oecd_extrap.loc[oecd_extrap["type"] == 3, "Counterpart"].dropna())
    stock_extra = stock[~stock["Counterpart"].isin(oecd_equity_ctry)].copy()
    data2_raw = pd.concat([oecd_extrap, stock_extra], ignore_index=True)
    data2_raw = data2_raw.dropna(subset=["outstand"])
    data2_raw["year"] = data2_raw["year"].astype(int)

    # -- Step 4: Restate residency → nationality ----------------------------------
    # Stata: joinby year Counterpart type using Restatement_issuance, unmatched(master)
    #        replace counterpart = Counterpart if _merge==1
    #        replace Value = 1 if _merge==1
    data2_r = data2_raw.merge(
        restat[["year", "Counterpart", "type", "counterpart", "Value"]],
        on=["year", "Counterpart", "type"],
        how="left",
    )
    unmatched = data2_r["counterpart"].isna()
    data2_r.loc[unmatched, "counterpart"] = data2_r.loc[unmatched, "Counterpart"]
    data2_r.loc[unmatched, "Value"] = 1.0

    # Fund shares: nationality == residency always
    fs_mask = data2_r["type"] == 4
    data2_r.loc[fs_mask, "counterpart"] = data2_r.loc[fs_mask, "Counterpart"]
    data2_r.loc[fs_mask, "Value"] = 1.0

    data2_r["outstand"] = data2_r["outstand"] * data2_r["Value"]

    # Merge Yeuro and Ynat from Countries
    ctry_meta = countries[["country", "Yeuro", "Ynat"]].rename(columns={"country": "counterpart"}).drop_duplicates("counterpart")
    data2_r = data2_r.merge(ctry_meta, on="counterpart", how="left")

    # Idomestic: whether outstanding is in domestic currency
    # For issuance data: currency == counterpart or EUR after euro adoption
    # Using Counterpart-level currency:
    #   currency == counterpart → domestic  OR  (currency == "EUR" & year >= Yeuro)
    # For simplicity: all issuances by nationality are domestic (Stata approximation)
    data2_r["Idomestic"] = 1  # domestic currency by nationality

    # Keep sample years
    data2_r = data2_r[data2_r["year"] >= data2_r["Ynat"].fillna(0).astype(int)]
    data2_r = data2_r.drop(columns=["Value", "Counterpart", "market", "Ynat", "Yeuro"], errors="ignore")

    data2_out = (
        data2_r.groupby(["year", "counterpart", "type", "Idomestic"])["outstand"]
        .sum()
        .reset_index()
    )

    print(f"  Data2: {len(data2_out):,} rows, {data2_out['counterpart'].nunique()} issuers")
    return data2_out


# ============================================================================
# DATA3: Holdings
# ============================================================================

def _load_cpis() -> pd.DataFrame:
    """
    IMF CPIS bilateral holdings — (year, country, Counterpart, type, amount) USD billions.
    Excludes USA (handled by Treasury) and offshore financial centres.

    Python primary: pip_bilateral_positions.parquet (value_usd is raw USD → ÷1e9)
    Fallback: IMF_CPIS.dta (already USD billions, already excludes OFCs)
    """
    p = DATA_DIR / "pip_bilateral_positions.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df = df.rename(columns={
            "COUNTRY": "country",
            "COUNTERPART_COUNTRY": "Counterpart",
            "TIME_PERIOD": "year",
        })
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        ac_map = {"ST_DEBT": 1, "LT_DEBT": 2, "EQUITY": 3}
        df["type"] = df["asset_class"].map(ac_map)
        df["amount"] = pd.to_numeric(df["value_usd"], errors="coerce") / 1e9  # raw USD → USD bn
        df = df[df["country"] != "USA"].copy()
        return df[["year", "country", "Counterpart", "type", "amount"]].dropna()

    # Fallback: IMF_CPIS.dta (correct format from Stata pipeline)
    print("  pip_bilateral_positions.parquet not found — using IMF_CPIS.dta fallback")
    df = _dta("IMF_CPIS.dta")
    if "Iofc" in df.columns:
        df = df[~df["Iofc"].astype(bool)].copy()
    df = df[df["country"] != "USA"].copy()
    return df[["year", "country", "Counterpart", "type", "amount"]]


def _load_cpis_currency() -> pd.DataFrame:
    """
    IMF CPIS currency denomination — (year, country, type, currency, Tcurrency).
    Used to cap local-currency holdings in Data3.do.
    """
    dta_path = DATAVERSE_DIR / "IMF_CPIS_currency.dta"
    if dta_path.exists():
        return pd.read_stata(dta_path)
    return pd.DataFrame()


def _load_treasury() -> pd.DataFrame:
    """
    US Treasury SHC holdings — (year, country="USA", Counterpart, type, amount) USD billions.
    Aggregated across currency denomination dimension.

    Primary: Treasury.dta (from dataverse, already processed by Treasury.do)
    """
    dta_path = DATAVERSE_DIR / "Treasury.dta"
    if dta_path.exists():
        df = pd.read_stata(dta_path)
        agg_cols = ["year", "country", "Counterpart", "type"]
        df = df.groupby(agg_cols)["amount"].sum().reset_index()
        return df
    print("  WARNING: Treasury.dta not found — USA holdings will be from CPIS only")
    return pd.DataFrame()


def _apply_restatement(
    df: pd.DataFrame,
    restat: pd.DataFrame,
    join_keys: list[str],
) -> pd.DataFrame:
    """
    Stata `joinby ... unmatched(master)` + fill defaults.

    Matched rows are expanded (one row per restated nationality).
    Unmatched rows keep Counterpart as counterpart and Value=1.
    """
    merged = df.merge(
        restat[join_keys + ["counterpart", "Value"]],
        on=join_keys,
        how="left",
    )
    unmatched = merged["counterpart"].isna()
    merged.loc[unmatched, "counterpart"] = merged.loc[unmatched, "Counterpart"]
    merged.loc[unmatched, "Value"] = 1.0
    return merged


def build_data3(countries: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
    """
    Data3.do equivalent — holdings by (year, country, counterpart, type) USD billions.

    Steps:
      1. Restate CPIS bilateral by nationality (Restatement_bilateral)
      2. Restate US Treasury by nationality
      3. Impute own (domestic) holdings = max(outstand − foreign, small)
      4. Aggregate investor countries; winsorize outside-asset left tail
    """
    print("Building Data3 (holdings)...")

    restat_bil = load_restatement_bilateral()
    ctry_meta = (
        countries[["country", "Yeuro", "Ynat"]]
        .rename(columns={"country": "counterpart"})
        .drop_duplicates("counterpart")
    )

    # ── Step 1: CPIS ──────────────────────────────────────────────────────────
    cpis = _load_cpis()

    cpis_r = _apply_restatement(cpis, restat_bil, ["year", "country", "Counterpart", "type"])
    cpis_r = cpis_r.merge(ctry_meta, on="counterpart", how="left")
    cpis_r["amount"] = cpis_r["amount"] * cpis_r["Value"]

    # Currency of issuer (nationality): assume domestic unless euro area
    cpis_r["currency"] = cpis_r["counterpart"]
    euro_mask = cpis_r["year"] >= cpis_r["Yeuro"].fillna(9999).astype(int)
    cpis_r.loc[euro_mask, "currency"] = "EUR"

    # Merge currency cap (Tcurrency) — scales down domestic holdings if total > cap
    cpis_ccy = _load_cpis_currency()
    if not cpis_ccy.empty and "Tcurrency" in cpis_ccy.columns:
        cpis_r = cpis_r.merge(cpis_ccy, on=["year", "country", "type", "currency"], how="left")
        Tc = cpis_r.groupby(["year", "country", "type", "currency"])["amount"].transform("sum")
        cap_mask = cpis_r["Tcurrency"].notna() & (cpis_r["Tcurrency"] < Tc)
        cpis_r.loc[cap_mask, "amount"] = (
            cpis_r.loc[cap_mask, "amount"]
            * cpis_r.loc[cap_mask, "Tcurrency"]
            / Tc[cap_mask]
        )
        cpis_r = cpis_r.drop(columns=["Tcurrency"], errors="ignore")

    # Idomestic
    cpis_r["Idomestic"] = (
        (cpis_r["currency"] == cpis_r["counterpart"])
        | ((cpis_r["currency"] == "EUR") & euro_mask)
    ).astype(int)

    # Countries outside sample → _OC
    cpis_r.loc[cpis_r["year"] < cpis_r["Ynat"].fillna(0).astype(int), "counterpart"] = "_OC"

    cpis_r = cpis_r.drop(columns=["Value", "Counterpart", "currency", "Yeuro", "Ynat"], errors="ignore")
    cpis_agg = (
        cpis_r.groupby(["year", "country", "counterpart", "type", "Idomestic"])["amount"]
        .sum()
        .reset_index()
    )

    # ── Step 2: US Treasury ───────────────────────────────────────────────────
    treasury = _load_treasury()
    if not treasury.empty:
        treas_r = _apply_restatement(treasury, restat_bil, ["year", "country", "Counterpart", "type"])
        treas_r = treas_r.merge(ctry_meta, on="counterpart", how="left")
        treas_r["amount"] = treas_r["amount"] * treas_r["Value"]
        # For Treasury: Idomestic = 1 for equity/fund shares (type >= 3), else by currency
        treas_r["Idomestic"] = (treas_r["type"] >= 3).astype(int)
        treas_r.loc[treas_r["year"] < treas_r["Ynat"].fillna(0).astype(int), "counterpart"] = "_OC"
        treas_r = treas_r.drop(columns=["Value", "Counterpart", "Yeuro", "Ynat"], errors="ignore")
        treas_agg = (
            treas_r.groupby(["year", "country", "counterpart", "type", "Idomestic"])["amount"]
            .sum()
            .reset_index()
        )
        holdings_foreign = pd.concat([treas_agg, cpis_agg], ignore_index=True)
    else:
        holdings_foreign = cpis_agg.copy()

    # ── Step 3: Own holdings ──────────────────────────────────────────────────
    # Stata: total foreign holdings (other investors), then own = max(outstand - foreign, small)
    # We use domestic (Idomestic=1) foreign holdings as the subtraction base
    dom_foreign = (
        holdings_foreign[holdings_foreign["Idomestic"] == 1]
        .groupby(["year", "counterpart", "type"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "foreign_total"})
    )

    own = (
        data2[data2["Idomestic"] == 1]
        .merge(dom_foreign, on=["year", "counterpart", "type"], how="left")
    )
    own["foreign_total"] = own["foreign_total"].fillna(0)
    own["amount"] = np.maximum(
        own["outstand"] - own["foreign_total"],
        SMALL,  # Stata: max(amount, outstand-Tamount, $small*Idomestic)
    )
    own["country"] = own["counterpart"]
    own["Idomestic"] = 1
    own = own[["year", "country", "counterpart", "type", "Idomestic", "amount"]].copy()

    # Combine all holdings; drop fund shares (type 4)
    all_holdings = pd.concat([holdings_foreign, own], ignore_index=True)
    all_holdings = all_holdings[all_holdings["type"] != 4].copy()

    # Aggregate foreign-currency holdings into _OC counterpart
    all_holdings.loc[all_holdings["Idomestic"] == 0, "counterpart"] = "_OC"
    data3 = (
        all_holdings.groupby(["year", "country", "counterpart", "type"])["amount"]
        .sum()
        .reset_index()
    )

    # ── Step 4: Aggregate investor countries ──────────────────────────────────
    inv_ynat = (
        countries[["country", "Ynat"]]
        .drop_duplicates("country")
    )
    data3 = data3.merge(inv_ynat, on="country", how="left")
    outside_sample = (data3["year"] < data3["Ynat"].fillna(0).astype(int)) & (data3["country"] != "_CR")
    data3.loc[outside_sample, "country"] = "_OC"
    data3 = data3.drop(columns=["Ynat"])
    data3 = data3.groupby(["year", "country", "counterpart", "type"])["amount"].sum().reset_index()

    # Round up to minimum (CPIS reporting floor)
    non_oc = data3["counterpart"] != "_OC"
    data3.loc[non_oc, "amount"] = np.maximum(data3.loc[non_oc, "amount"], 1e-6)

    # Winsorise outside assets left tail
    # Stata: replace amount = max(amount, 1e-3*wealthA) if counterpart=="_OC"
    wA = (
        data3.groupby(["year", "country", "type"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "wealthA"})
    )
    data3 = data3.merge(wA, on=["year", "country", "type"], how="left")
    oc_mask = data3["counterpart"] == "_OC"
    data3.loc[oc_mask, "amount"] = np.maximum(
        data3.loc[oc_mask, "amount"], 1e-3 * data3.loc[oc_mask, "wealthA"]
    )
    data3 = data3.drop(columns=["wealthA"])

    print(f"  Data3: {len(data3):,} rows, {data3['country'].nunique()} investors")
    return data3.sort_values(["year", "country", "counterpart", "type"]).reset_index(drop=True)


# ============================================================================
# DATA4 (partial): add summary variables needed for Tables 1 & 2
# ============================================================================

def build_data4_for_tables(data3: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """
    Partial Data4.do — creates the variables Summary0.do reads:
      amount, Iown, I_CR, market, wealthA, Iycounterpart, Iycountry,
      Name, group, Ieuro, country_Name.

    Drops _OC counterparts (Data4.do does this before computing market totals).
    """
    print("Building Data4 (final merge for tables)...")

    df = data3[data3["counterpart"] != "_OC"].copy()

    # Dummy variables
    df["Iown"] = (df["country"] == df["counterpart"]).astype(int)
    df["I_CR"] = (df["country"] == "_CR").astype(int)

    # Issuer country metadata
    iss_meta = (
        countries[["country", "Name", "MSCI", "Yeuro"]]
        .rename(columns={"country": "counterpart", "MSCI": "MSCI_iss"})
        .drop_duplicates("counterpart")
    )
    df = df.merge(iss_meta, on="counterpart", how="left")
    df["Ieuro"] = (df["year"] >= df["Yeuro"].fillna(9999).astype(int)).astype(int)
    df = df.drop(columns=["Yeuro"])

    # Issuer group (Stata: MSCI 1-3 = DM, 4-6 = EM)
    df["group"] = np.where(
        df["MSCI_iss"].between(1, 3), df["MSCI_iss"],
        np.where(df["MSCI_iss"].between(4, 6), 4, np.nan),
    )

    # Market = total amount per (year, counterpart, type)
    # Stata: counter_euro groups euro-area ST debt; for Tables 1&2 use counterpart directly
    mkt = (
        df.groupby(["year", "counterpart", "type"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "market"})
    )
    df = df.merge(mkt, on=["year", "counterpart", "type"], how="left")

    # WealthA = total amount per (year, country, type)
    wA = (
        df.groupby(["year", "country", "type"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "wealthA"})
    )
    df = df.merge(wA, on=["year", "country", "type"], how="left")

    # Iycounterpart: flag the first (unique) observation per (year, counterpart, type)
    # Stata: gsort year counterpart type -Iown; by ...: gen Iycounterpart = _n==1
    df = df.sort_values(["year", "counterpart", "type", "Iown"], ascending=[True, True, True, False])
    df["Iycounterpart"] = (df.groupby(["year", "counterpart", "type"]).cumcount() == 0).astype(int)

    # Iycountry: first observation per (year, country, type)
    df = df.sort_values(["year", "country", "type", "Iown"], ascending=[True, True, True, False])
    df["Iycountry"] = (df.groupby(["year", "country", "type"]).cumcount() == 0).astype(int)

    # Investor country name
    inv_name = (
        countries[["country", "Name"]]
        .rename(columns={"Name": "country_Name"})
        .drop_duplicates("country")
    )
    df = df.merge(inv_name, on="country", how="left")
    df.loc[df["country"] == "_CR", "country_Name"] = "Reserves"
    df.loc[df["country"] == "_OC", "country_Name"] = "Other"

    print(f"  Data4: {len(df):,} rows")
    return df.reset_index(drop=True)


# ============================================================================
# TABLES 1 & 2  (Summary0.do)
# ============================================================================

def build_table1(data4: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: Market Values of Financial Assets (2020 cross-section).

    Stata Summary0.do Step 2:
      keep if year==2020 & Iycounterpart
      market = total(amount) by year Name type
      domestic = total(amount/market * Iown)
      reserve  = total(amount/market * I_CR)
    """
    print("Building Table 1...")
    df = data4[(data4["year"] == 2020)].copy()

    if df.empty:
        print("  WARNING: No 2020 data found for Table 1")
        return pd.DataFrame()

    # Compute per-row contributions
    df["dom_wt"] = df["amount"] / df["market"] * df["Iown"]
    df["res_wt"] = df["amount"] / df["market"] * df["I_CR"]

    agg = (
        df.groupby(["Name", "group", "Ieuro", "type"])
        .agg(
            market=("market", "first"),
            domestic=("dom_wt", "sum"),
            reserve=("res_wt", "sum"),
        )
        .reset_index()
    )

    # Stata: gsort type group -Ieuro Name
    agg = agg.sort_values(
        ["type", "group", "Ieuro", "Name"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    return agg.reset_index(drop=True)


def build_table2(data4: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: Top Ten Investors by Asset Class (2020 cross-section).

    Stata Summary0.do Step 3:
      keep if year==2020 & Iycountry
      gsort type -wealthA; by type: l if _n<=10
    """
    print("Building Table 2...")
    df = data4[(data4["year"] == 2020) & (data4["Iycountry"] == 1)].copy()

    rows = []
    type_labels = {1: "Short-term debt", 2: "Long-term debt", 3: "Equity"}
    for tp in [1, 2, 3]:
        sub = df[df["type"] == tp].sort_values("wealthA", ascending=False).head(10)
        sub = sub[["country_Name", "wealthA"]].copy()
        sub.insert(0, "type_label", type_labels[tp])
        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("Koijen & Yogo (2020) — process_script.py")
    print("Tables 1 & 2  |  year range:", YEAR_MIN, "–", YEAR_MAX)
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load lookup tables
    print("\nLoading Countries.dta ...")
    countries = load_countries()

    # Data2: amounts outstanding
    data2 = build_data2(countries)
    (DATA_DIR / "data2.parquet").unlink(missing_ok=True)
    data2.to_parquet(DATA_DIR / "data2.parquet", index=False)

    # Data3: holdings
    data3 = build_data3(countries, data2)
    data3.to_parquet(DATA_DIR / "data3.parquet", index=False)
    print(f"  Saved data3.parquet")

    # Data4 partial
    data4 = build_data4_for_tables(data3, countries)
    data4.to_parquet(DATA_DIR / "data4_for_tables.parquet", index=False)
    print(f"  Saved data4_for_tables.parquet")

    # Tables
    t1 = build_table1(data4)
    t2 = build_table2(data4)

    t1_path = DATA_DIR / "table1_market_values.csv"
    t2_path = DATA_DIR / "table2_top_investors.csv"
    t1.to_csv(t1_path, index=False)
    t2.to_csv(t2_path, index=False)

    print(f"\nTable 1 saved → {t1_path}")
    print(t1.to_string(index=False))

    print(f"\nTable 2 saved → {t2_path}")
    print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
