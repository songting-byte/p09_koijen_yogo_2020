"""table_1.py
============
Reproduces Table 1 (Market Values of Financial Assets, 2020) from
Koijen & Yogo (2020), using the processed .dta files from the dataverse.

Pipeline:  OECD.dta + WorldBank.dta  →  Data2 (amounts outstanding)
           IMF_CPIS.dta + Treasury.dta + Restatement_bilateral.dta
                                       →  Data3 (holdings)
           Data3 + Countries.dta      →  Table 1

Run:  python table_1.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
DATA = Path(__file__).parent.parent.parent / "dataverse_files (1)" / "Code" / "1 Data"
SMALL = 1e-6

# ── type label → numeric ─────────────────────────────────────────────────────
TYPE_MAP = {
    "Short-term debt": 1, "Long-term debt": 2,
    "Equity": 3, "Fund shares": 4, "All debt": 0,
}

# MSCI string → numeric group (matches Data4.do group_label)
MSCI_GROUP = {
    "DM: Americas": 1,
    "DM: Europe & Middle East": 2,
    "DM: Pacific": 3,
    "EM: Americas": 4,
    "EM: Asia": 4,
    "EM: Europe, Middle East & Africa": 4,
}
GROUP_LABEL = {1: "DM: Americas", 2: "DM: Europe & Middle East",
               3: "DM: Pacific", 4: "EM"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _map_type(s: pd.Series) -> pd.Series:
    return s.astype(str).map(TYPE_MAP)


def _read(fname, **kw):
    return pd.read_stata(DATA / fname, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Nelson-Siegel market-to-book  (Data1.do)
# ─────────────────────────────────────────────────────────────────────────────

def _ns_factor(t: float, tau: float = 0.01) -> float:
    """Nelson-Siegel loading: (1 - exp(-t/tau)) / (t/tau)."""
    return (1.0 - np.exp(-t / tau)) / (t / tau)


def _nelson_siegel_yield5(yieldST: float, yieldLT: float,
                          tau: float = 0.01, nstep: int = 1000,
                          mgap: float = 1e-9) -> float:
    """
    Port of Data1.do Newton iteration.
    Fits beta0/beta1 so that a 10-year par bond priced with Nelson-Siegel
    zero-coupon yields equals 1.  Returns the 5-year zero-coupon yield.
    """
    beta0 = 0.0
    beta1 = 0.0
    coupon = np.exp(yieldLT) - 1.0
    ts = np.arange(1, 11)
    factors = np.array([_ns_factor(t, tau) for t in ts])
    f5 = _ns_factor(5.0, tau)
    f025 = _ns_factor(0.25, tau)

    for _ in range(nstep):
        beta0 = yieldST - beta1 * f025
        yz = beta0 + beta1 * factors          # zero-coupon yields t=1..10
        price = coupon * np.sum(np.exp(-ts * yz)) + np.exp(-10.0 * yz[-1])
        error = np.log(price)
        if abs(error) < mgap:
            break
        beta1 -= error

    return beta0 + beta1 * f5


def _build_data1_mb(countries: pd.DataFrame) -> pd.DataFrame | None:
    """
    Load Datastream interest rates (Interest Rates.xlsm), compute Nelson-Siegel
    yield5, and return DataFrame with (year, Counterpart, mb_ST, mb_LT, Idds).

    mb_ST = exp(-0.25 * yieldST)
    mb_LT = exp(-5    * yield5)   where yield5 from Nelson-Siegel
    yieldST = ln(1+IR3M/100),  yieldLT = ln(1+IR10Y/100)

    Mirrors Data0.do + Data1.do.
    """
    RATES_PATH = (DATA.parent.parent / "Data" / "Datastream" / "Interest Rates.xlsm")
    if not RATES_PATH.exists():
        print("  WARNING: Interest Rates.xlsm not found — mb conversion skipped")
        return None

    # ── Legend: Name → Code3M, Code10Y ──────────────────────────────────
    legend = pd.read_excel(RATES_PATH, sheet_name="Legend")
    # Stata: replace Code3M = subinstr(Code3M,".","",.);
    legend["Code3M"]  = legend["Code3M"].astype(str).str.replace(".", "", regex=False).str.strip()
    legend["Code10Y"] = legend["Code10Y"].astype(str).str.strip()
    # Drop rows where code is "nan" (missing in Excel)
    leg3m  = legend[legend["Code3M"]  != "nan"][["Name", "Code3M"]].dropna()
    leg10y = legend[legend["Code10Y"] != "nan"][["Name", "Code10Y"]].dropna()

    # ── 3M Interbank ─────────────────────────────────────────────────────
    ir3m_raw = pd.read_excel(RATES_PATH, sheet_name="3M Interbank", header=1)
    ir3m_raw = ir3m_raw.rename(columns={"Code": "Date"})
    ir3m_raw["Date"] = pd.to_datetime(ir3m_raw["Date"], errors="coerce")
    ir3m_raw = ir3m_raw[ir3m_raw["Date"].dt.month == 12].copy()
    ir3m_raw["year"] = ir3m_raw["Date"].dt.year
    # Clean column names (remove dots so AGI60L.. → AGI60L)
    ir3m_raw = ir3m_raw.rename(columns={
        c: c.replace(".", "") for c in ir3m_raw.columns if c not in ("Date", "year")
    }).drop(columns=["Date"])
    ir3m_long = ir3m_raw.melt(id_vars=["year"], var_name="Code3M", value_name="IR3M")
    ir3m_long = ir3m_long.dropna(subset=["IR3M"])
    ir3m_long = ir3m_long.merge(leg3m, on="Code3M", how="inner").drop(columns=["Code3M"])

    # ── 10Y Government ───────────────────────────────────────────────────
    ir10y_raw = pd.read_excel(RATES_PATH, sheet_name="10Y Government", header=1)
    ir10y_raw = ir10y_raw.rename(columns={"Code": "Date"})
    ir10y_raw["Date"] = pd.to_datetime(ir10y_raw["Date"], errors="coerce")
    ir10y_raw = ir10y_raw[ir10y_raw["Date"].dt.month == 12].copy()
    ir10y_raw["year"] = ir10y_raw["Date"].dt.year
    ir10y_raw = ir10y_raw.drop(columns=["Date"])
    ir10y_long = ir10y_raw.melt(id_vars=["year"], var_name="Code10Y", value_name="IR10Y")
    ir10y_long = ir10y_long.dropna(subset=["IR10Y"])
    ir10y_long = ir10y_long.merge(leg10y, on="Code10Y", how="inner").drop(columns=["Code10Y"])

    # ── Merge and handle euro area / USD ─────────────────────────────────
    rates = ir3m_long.merge(ir10y_long, on=["year", "Name"], how="outer")

    # Euro 3M common rate (Euribor)
    euro_3m = (rates[rates["Name"] == "Euro"][["year", "IR3M"]]
               .rename(columns={"IR3M": "euro_3m"}))
    rates = rates.merge(euro_3m, on="year", how="left")

    # Merge country metadata
    ctry_meta = (countries[["country", "Name", "Yeuro"]]
                 .rename(columns={"country": "Counterpart", "Name": "CtryName"})
                 .drop_duplicates("Counterpart"))
    rates = rates.merge(ctry_meta, left_on="Name", right_on="CtryName", how="left")

    # Euro area countries → use common 3M rate
    is_euro = rates["Yeuro"].notna() & (rates["year"] >= rates["Yeuro"])
    rates.loc[is_euro, "IR3M"] = rates.loc[is_euro, "euro_3m"]
    rates = rates[rates["Name"] != "Euro"].copy()

    # HKG → use USA 3M rate  (Iusd logic in Datastream.do)
    usa_3m = (rates[rates["Counterpart"] == "USA"][["year", "IR3M"]]
              .rename(columns={"IR3M": "usa_3m"}))
    rates = rates.merge(usa_3m, on="year", how="left")
    rates.loc[rates["Counterpart"] == "HKG", "IR3M"] = (
        rates.loc[rates["Counterpart"] == "HKG", "usa_3m"]
    )

    # Convert percent → decimal
    rates["IR3M"]  = rates["IR3M"]  / 100.0
    rates["IR10Y"] = rates["IR10Y"] / 100.0

    # Log yields
    rates = rates.dropna(subset=["IR3M", "IR10Y", "Counterpart"]).copy()
    rates["yieldST"] = np.log1p(rates["IR3M"])
    rates["yieldLT"] = np.log1p(rates["IR10Y"])

    # Nelson-Siegel yield5 (vectorised row-wise)
    rates["yield5"] = [
        _nelson_siegel_yield5(float(r.yieldST), float(r.yieldLT))
        for r in rates.itertuples()
    ]

    # Market-to-book
    rates["mb_ST"] = np.exp(-0.25 * rates["yieldST"])
    rates["mb_LT"] = np.exp(-5.0  * rates["yield5"])

    # Attach Idds
    idds = (countries[["country", "Idds"]]
            .rename(columns={"country": "Counterpart"})
            .drop_duplicates("Counterpart"))
    rates = rates.merge(idds, on="Counterpart", how="left")

    out = (rates[["year", "Counterpart", "mb_ST", "mb_LT", "Idds"]]
           .dropna(subset=["Counterpart", "mb_ST", "mb_LT"])
           .drop_duplicates(["year", "Counterpart"]))
    print(f"  Data1 mb: {len(out):,} rows, {out['Counterpart'].nunique()} countries, "
          f"years {out['year'].min()}-{out['year'].max()}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BIS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _impute_st_lt_from_all(dds: pd.DataFrame,
                            ids_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    For countries that only have type=0 (All debt) but not type=1 or 2,
    split using IDS-based ST share (mirrors Data2.do's IDS-share imputation).
    Falls back to global DDS median if no IDS ratio available.
    Drop type=0 rows.
    """
    has_12 = set(dds.loc[dds["type"].isin([1, 2]), "Counterpart"].unique())
    keep = dds[dds["type"].isin([1, 2])].copy()

    to_split = dds[(dds["type"] == 0) & ~dds["Counterpart"].isin(has_12)].copy()
    if to_split.empty:
        return keep

    # Global median ST share from countries with both DDS ST and LT
    st_g = (keep[keep["type"] == 1]
            .groupby(["year", "Counterpart"])["dds"].sum().reset_index()
            .rename(columns={"dds": "st"}))
    lt_g = (keep[keep["type"] == 2]
            .groupby(["year", "Counterpart"])["dds"].sum().reset_index()
            .rename(columns={"dds": "lt"}))
    if not st_g.empty and not lt_g.empty:
        r = st_g.merge(lt_g, on=["year", "Counterpart"])
        global_st = float((r["st"] / (r["st"] + r["lt"]).clip(lower=1e-9)).median())
    else:
        global_st = 0.15

    # Per-country ST share from IDS breakdown (Data2.do: IDS-based split of All debt)
    ctry_shares: dict[str, float] = {}
    if ids_df is not None:
        ids_st = (ids_df[ids_df["type"] == 1][["year", "Counterpart", "ids"]]
                  .rename(columns={"ids": "ids_st"}))
        ids_lt = (ids_df[ids_df["type"] == 2][["year", "Counterpart", "ids"]]
                  .rename(columns={"ids": "ids_lt"}))
        ids_r = ids_st.merge(ids_lt, on=["year", "Counterpart"], how="inner")
        ids_r["st_share"] = ids_r["ids_st"] / (ids_r["ids_st"] + ids_r["ids_lt"]).clip(lower=1e-9)
        # Use most recent year per country
        for ctry, grp in ids_r.groupby("Counterpart"):
            ctry_shares[ctry] = float(grp.sort_values("year")["st_share"].iloc[-1])

    # Assign per-country share (IDS-based), fall back to global median
    shares = to_split["Counterpart"].map(ctry_shares).fillna(global_st)

    st_new = to_split.copy(); st_new["type"] = 1
    st_new["dds"] = st_new["dds"] * shares.values
    lt_new = to_split.copy(); lt_new["type"] = 2
    lt_new["dds"] = lt_new["dds"] * (1 - shares.values)

    cols = ["year", "Counterpart", "type", "dds"]
    return pd.concat([keep, st_new[cols], lt_new[cols]], ignore_index=True)


def _build_bis(countries: pd.DataFrame):
    """
    Load WS_DEBT_SEC2_PUB_csv_col.dta and return:
      dds_df : (year, Counterpart, type∈{1,2}) → domestic debt, USD billions
      ids_df : (year, Counterpart, type∈{0,1,2}) → intl debt ALL-ccy, USD billions
    Values in raw DTA are USD millions; divide by 1000 here.
    """
    BIS_PATH = (DATA.parent.parent / "Data" / "BIS"
                / "Debt securities statistics" / "WS_DEBT_SEC2_PUB_csv_col.dta")
    if not BIS_PATH.exists():
        print(f"  WARNING: BIS DTA not found — skipping BIS integration")
        return None, None

    print("  Loading BIS DTA ...")
    bis = pd.read_stata(BIS_PATH)

    iso2_map = (countries[["ISO2", "Counterpart"]]
                .dropna(subset=["ISO2"]).drop_duplicates("ISO2")
                .set_index("ISO2")["Counterpart"].to_dict())
    ORMAT = {"A": 0, "C": 1, "K": 2}
    q4_cols = [c for c in bis.columns if c.startswith("_") and c.endswith("_Q4")]

    def _melt_q4(df_sub, id_cols, val):
        melted = df_sub[id_cols + q4_cols].melt(
            id_vars=id_cols, value_vars=q4_cols, var_name="qc", value_name=val)
        melted["year"] = melted["qc"].str.extract(r"_(\d{4})_Q4").astype(int)
        return melted.drop(columns=["qc"])

    # ── IDS (international debt, all-currency total) — computed first ────
    ids_mask = (
        (bis["ISSUER_BUS_IMM"] == "1") & (bis["market"] == "C")
        & (bis["ISSUE_CUR_GROUP"] == "A") & (bis["ISSUE_CUR"] == "TO1")
    )
    ids_long = _melt_q4(bis[ids_mask], ["ISSUER_RES", "ISSUE_OR_MAT"], "ids")
    ids_long["Counterpart"] = ids_long["ISSUER_RES"].map(iso2_map)
    ids_long["type"] = ids_long["ISSUE_OR_MAT"].map(ORMAT)
    ids_long = ids_long.dropna(subset=["Counterpart", "type"]).copy()
    ids_long["ids"] = ids_long["ids"].fillna(0.0)
    ids_long["type"] = ids_long["type"].astype(int)

    ids_agg = (ids_long.groupby(["year", "Counterpart", "type"])["ids"]
               .sum().reset_index())
    ids_agg["ids"] /= 1000.0  # USD millions → billions

    # ── DDS (domestic debt securities) ──────────────────────────────────
    dds_mask = bis["ISSUER_BUS_IMM"].isin(["2", "B", "J"]) & (bis["market"] == "A")
    dds_long = _melt_q4(bis[dds_mask], ["ISSUER_RES", "ISSUER_BUS_IMM", "ISSUE_OR_MAT"], "dds")
    dds_long["Counterpart"] = dds_long["ISSUER_RES"].map(iso2_map)
    dds_long["type"] = dds_long["ISSUE_OR_MAT"].map(ORMAT)
    dds_long = (dds_long.dropna(subset=["Counterpart", "type", "dds"])
                .query("dds > 0").copy())
    dds_long["type"] = dds_long["type"].astype(int)

    dds_agg = (dds_long.groupby(["year", "Counterpart", "type"])["dds"]
               .sum().reset_index())
    # Use IDS-based ST share for countries with only All-debt DDS (e.g. China)
    dds_agg = _impute_st_lt_from_all(dds_agg, ids_agg)
    dds_agg["dds"] /= 1000.0  # USD millions → billions

    # ── Apply Nelson-Siegel market-to-book (Data1.do / BIS.do) ──────────
    # ids: always apply mb (nominal → market value)
    # dds: apply mb only where Idds==0 (book value)
    mb_df = _build_data1_mb(countries)
    if mb_df is not None:
        mb_st_lt = mb_df[["year", "Counterpart", "mb_ST", "mb_LT", "Idds"]]

        # IDS: multiply by mb for type=1,2
        ids_agg = ids_agg.merge(mb_st_lt, on=["year", "Counterpart"], how="left")
        ids_agg["mb"] = np.where(ids_agg["type"] == 1, ids_agg["mb_ST"],
                        np.where(ids_agg["type"] == 2, ids_agg["mb_LT"], 1.0))
        ids_agg["ids"] *= ids_agg["mb"].fillna(1.0)
        ids_agg = ids_agg.drop(columns=["mb", "mb_ST", "mb_LT", "Idds"])

        # DDS: multiply by mb only where Idds==0
        dds_agg = dds_agg.merge(mb_st_lt, on=["year", "Counterpart"], how="left")
        dds_agg["mb"] = np.where(dds_agg["type"] == 1, dds_agg["mb_ST"],
                        np.where(dds_agg["type"] == 2, dds_agg["mb_LT"], 1.0))
        needs_mb = dds_agg["Idds"] == 0
        dds_agg.loc[needs_mb, "dds"] *= dds_agg.loc[needs_mb, "mb"]
        dds_agg = dds_agg.drop(columns=["mb", "mb_ST", "mb_LT", "Idds"])

    print(f"  BIS loaded: DDS {len(dds_agg):,} rows / {dds_agg['Counterpart'].nunique()} issuers, "
          f"IDS {len(ids_agg):,} rows")
    return dds_agg, ids_agg


# ─────────────────────────────────────────────────────────────────────────────
# DATA2  –  Amounts outstanding
# ─────────────────────────────────────────────────────────────────────────────

def build_data2(countries: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors Data2.do:
      - OECD.dta as primary source for debt + equity outstanding
      - BIS DDS as fallback; IDS subtracted from OECD total to get domestic
      - WorldBank.dta equity for countries not covered by OECD
      - Restatement_issuance for nationality restatement
    Output: (year, counterpart, type, Idomestic=1, outstand)  USD billions
    """
    print("  Loading OECD.dta ...")
    oecd = _read("OECD.dta")
    oecd["type"] = _map_type(oecd["type"])
    oecd = oecd[oecd["type"].isin([1, 2, 3, 4])].dropna(subset=["type"]).copy()
    oecd["type"] = oecd["type"].astype(int)

    # WorldBank equity supplement for non-OECD countries
    print("  Loading WorldBank.dta ...")
    wb = _read("WorldBank.dta", columns=["year", "country", "outstand"])
    wb = wb.rename(columns={"country": "Counterpart"}).dropna(subset=["outstand"])
    wb["type"] = 3
    oecd_eq_ctry = set(oecd.loc[oecd["type"] == 3, "Counterpart"])
    wb_extra = wb[~wb["Counterpart"].isin(oecd_eq_ctry)].copy()

    # ── MSCI-style equity extrapolation (Data2.do lines 22-31) ──────────
    # For OECD equity countries whose data ends before 2020, extrapolate
    # forward (and backward) using WorldBank market-cap growth as a proxy
    # for MSCI returns.  Mirrors: replace outstand = outstand[t-1]*market[t]/market[t-1]
    oecd_eq = oecd[oecd["type"] == 3].copy()
    all_years = sorted(oecd_eq["year"].unique())
    wb_idx = wb.set_index(["Counterpart", "year"])["outstand"]

    extrap_rows = []
    for ctry, grp in oecd_eq.groupby("Counterpart"):
        last_yr = int(grp["year"].max())
        first_yr = int(grp["year"].min())
        last_val = float(grp.loc[grp["year"] == last_yr, "outstand"].iloc[0])
        first_val = float(grp.loc[grp["year"] == first_yr, "outstand"].iloc[0])

        # Forward extrapolation (e.g. Russia 2019, 2020)
        for yr in [y for y in all_years if y > last_yr]:
            prev_yr = yr - 1
            wb_curr = wb_idx.get((ctry, yr))
            wb_prev = wb_idx.get((ctry, prev_yr))
            if wb_curr is not None and wb_prev is not None and wb_prev > 0:
                last_val = last_val * (wb_curr / wb_prev)
            extrap_rows.append({"Counterpart": ctry, "year": yr,
                                 "type": 3, "outstand": last_val})

        # Backward extrapolation (rare — mirrors gsort -year logic)
        for yr in sorted([y for y in all_years if y < first_yr], reverse=True):
            next_yr = yr + 1
            wb_curr = wb_idx.get((ctry, yr))
            wb_next = wb_idx.get((ctry, next_yr))
            if wb_curr is not None and wb_next is not None and wb_next > 0:
                first_val = first_val * (wb_curr / wb_next)
            extrap_rows.append({"Counterpart": ctry, "year": yr,
                                 "type": 3, "outstand": first_val})

    if extrap_rows:
        extrap_df = pd.DataFrame(extrap_rows)
        # Only add rows for year/country combos not already in OECD equity
        existing_keys = set(zip(oecd_eq["Counterpart"], oecd_eq["year"]))
        extrap_df = extrap_df[
            ~extrap_df.apply(lambda r: (r["Counterpart"], r["year"]) in existing_keys, axis=1)
        ]
        oecd = pd.concat([oecd, extrap_df], ignore_index=True)

    # ── BIS integration (Data2.do Step 3) ───────────────────────────────
    dds_df, ids_df = _build_bis(countries)

    if dds_df is not None:
        oecd_debt = oecd[oecd["type"].isin([1, 2])].copy()
        oecd_eq   = oecd[oecd["type"].isin([3, 4])].copy()

        # Step A: subtract IDS from OECD total → domestic debt
        ids_12 = ids_df[ids_df["type"].isin([1, 2])].copy()
        oecd_debt = oecd_debt.merge(ids_12, on=["year", "Counterpart", "type"], how="left")
        oecd_debt["ids"] = oecd_debt["ids"].fillna(0.0)
        oecd_debt["outstand"] = np.maximum(oecd_debt["outstand"] - oecd_debt["ids"], 0.0)
        oecd_debt = oecd_debt.drop(columns=["ids"])

        # Step B: outer-merge with DDS; fallback to DDS for non-OECD countries
        # (backward extrapolation skipped — not needed for 2020 cross-section)
        dds_12 = dds_df[dds_df["type"].isin([1, 2])].copy()
        merged = oecd_debt.merge(dds_12, on=["year", "Counterpart", "type"], how="outer")

        # Step C: where OECD has no data, use BIS DDS directly
        merged["outstand"] = merged["outstand"].combine_first(merged["dds"])
        merged = merged.drop(columns=["dds"]).dropna(subset=["outstand"])

        oecd = pd.concat([merged, oecd_eq], ignore_index=True)

    raw = pd.concat([oecd, wb_extra], ignore_index=True)

    # Issuance restatement: (year, Counterpart, type) → (counterpart, Value)
    print("  Loading Restatement_issuance.dta ...")
    ri = _read("Restatement_issuance.dta")
    ri["type"] = _map_type(ri["type"])
    ri = ri.dropna(subset=["type"]).copy()
    ri["type"] = ri["type"].astype(int)

    data2 = raw.merge(
        ri[["year", "Counterpart", "type", "counterpart", "Value"]],
        on=["year", "Counterpart", "type"], how="left"
    )
    unmatched = data2["counterpart"].isna()
    data2.loc[unmatched, "counterpart"] = data2.loc[unmatched, "Counterpart"]
    data2.loc[unmatched, "Value"] = 1.0
    # Fund shares: nationality == residency always
    fs = data2["type"] == 4
    data2.loc[fs, "counterpart"] = data2.loc[fs, "Counterpart"]
    data2.loc[fs, "Value"] = 1.0

    data2["outstand"] = data2["outstand"] * data2["Value"]
    data2["Idomestic"] = 1  # OECD outstand = domestic currency by construction

    # Filter to sample (year >= Ynat)
    ynat = (countries[["country", "Ynat"]]
            .rename(columns={"country": "counterpart"})
            .drop_duplicates("counterpart"))
    data2 = data2.merge(ynat, on="counterpart", how="left")
    data2 = data2[data2["year"] >= data2["Ynat"].fillna(0)].copy()
    data2 = data2.drop(columns=["Value", "Counterpart", "Ynat"], errors="ignore")

    data2 = (data2
             .groupby(["year", "counterpart", "type", "Idomestic"])["outstand"]
             .sum().reset_index())
    print(f"  Data2: {len(data2):,} rows, {data2['counterpart'].nunique()} issuers")
    return data2


# ─────────────────────────────────────────────────────────────────────────────
# DATA3  –  Holdings
# ─────────────────────────────────────────────────────────────────────────────

def _apply_bil_restat(df: pd.DataFrame, rb: pd.DataFrame) -> pd.DataFrame:
    """Stata joinby year country Counterpart type, unmatched(master)."""
    merged = df.merge(
        rb[["year", "country", "Counterpart", "type", "counterpart", "Value"]],
        on=["year", "country", "Counterpart", "type"], how="left"
    )
    unmatched = merged["counterpart"].isna()
    merged.loc[unmatched, "counterpart"] = merged.loc[unmatched, "Counterpart"]
    merged.loc[unmatched, "Value"] = 1.0
    return merged


def build_data3(countries: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors Data3.do:
      Step 1: CPIS bilateral restated by nationality + currency cap
      Step 2: Treasury restated by nationality
      Step 3: Own holdings = max(outstand - foreign_domestic, SMALL)
      Step 4: Aggregate investor countries
    Output: (year, country, counterpart, type, amount)  USD billions
            counterpart != "_OC"  (foreign-currency bucket dropped for Table 1)
    """
    # ── Load Restatement_bilateral (filter to 2003-2020 first)
    print("  Loading Restatement_bilateral.dta (large file) ...")
    rb = _read("Restatement_bilateral.dta")
    rb["type"] = _map_type(rb["type"])
    rb = rb.dropna(subset=["type"]).copy()
    rb["type"] = rb["type"].astype(int)
    rb = rb[rb["year"].between(2003, 2020)].copy()

    ctry_meta = (countries[["country", "Yeuro", "Ynat"]]
                 .rename(columns={"country": "counterpart"})
                 .drop_duplicates("counterpart"))

    # ── Step 1: CPIS ──────────────────────────────────────────────────────
    print("  Loading IMF_CPIS.dta ...")
    cpis = _read("IMF_CPIS.dta")
    cpis["type"] = _map_type(cpis["type"])
    cpis = cpis.dropna(subset=["type"]).copy()
    cpis["type"] = cpis["type"].astype(int)
    cpis = cpis[(cpis["country"] != "USA") & (~cpis["Iofc"].astype(bool))].copy()

    # Pre-filter restatement to CPIS country-pairs (speeds up merge)
    cpis_keys = cpis[["year", "country", "Counterpart", "type"]].drop_duplicates()
    rb_cpis = rb.merge(cpis_keys, on=["year", "country", "Counterpart", "type"], how="inner")

    cpis_r = _apply_bil_restat(cpis, rb_cpis)
    cpis_r = cpis_r.merge(ctry_meta, on="counterpart", how="left")
    cpis_r["amount"] = cpis_r["amount"] * cpis_r["Value"]

    # Currency: counterpart's local currency; EUR for euro area
    cpis_r["currency"] = cpis_r["counterpart"].astype(str)
    euro_mask = cpis_r["Yeuro"].notna() & (cpis_r["year"] >= cpis_r["Yeuro"])
    cpis_r.loc[euro_mask, "currency"] = "EUR"

    # ── Currency cap (Data3.do logic) ────────────────────────────────────
    # IMF_CPIS_currency gives Tcurrency = total reported in each currency
    # If sum of restated holdings (Tcounterpart) > Tcurrency: scale down
    ccy = _read("IMF_CPIS_currency.dta")
    ccy["type"] = _map_type(ccy["type"])
    ccy = ccy.dropna(subset=["type"]).copy()
    ccy["type"] = ccy["type"].astype(int)

    cpis_r = cpis_r.merge(
        ccy.rename(columns={"currency": "currency_cap"}),
        left_on=["year", "country", "type", "currency"],
        right_on=["year", "country", "type", "currency_cap"],
        how="left"
    ).drop(columns=["currency_cap"], errors="ignore")

    # Total domestic-currency holdings per (year, investor, type, currency)
    Tc = cpis_r.groupby(["year", "country", "type", "currency"])["amount"].transform("sum")

    cap_mask = cpis_r["Tcurrency"].notna() & (cpis_r["Tcurrency"] < Tc)

    # Scale domestic portion down; remainder is foreign-currency
    cpis_r["amount_dom"] = np.where(
        cap_mask,
        cpis_r["amount"] * cpis_r["Tcurrency"] / Tc,
        cpis_r["amount"]
    )
    cpis_r["amount_fc"] = np.where(
        cap_mask,
        cpis_r["amount"] * (1 - cpis_r["Tcurrency"] / Tc),
        0.0
    )

    cpis_r["Idomestic"] = (
        (cpis_r["currency"] == cpis_r["counterpart"].astype(str))
        | euro_mask
    ).astype(int)

    # Collapse: domestic holdings (Idomestic=1) per counterpart
    cpis_r["amount"] = cpis_r["amount_dom"]
    cpis_r.loc[cpis_r["Idomestic"] == 0, "amount"] = 0.0  # FC goes to _OC (dropped later)

    # Out-of-sample counterparts → _OC
    cpis_r.loc[
        cpis_r["Ynat"].notna() & (cpis_r["year"] < cpis_r["Ynat"]),
        "counterpart"
    ] = "_OC"

    drop_cols = ["Value", "Counterpart", "Iofc", "Yeuro", "Ynat",
                 "currency", "Tcurrency", "amount_dom", "amount_fc", "Idomestic"]
    cpis_r = cpis_r.drop(columns=drop_cols, errors="ignore")
    cpis_agg = (cpis_r.groupby(["year", "country", "counterpart", "type"])["amount"]
                .sum().reset_index())

    # ── Step 2: Treasury ──────────────────────────────────────────────────
    print("  Loading Treasury.dta ...")
    treas = _read("Treasury.dta")
    treas["type"] = _map_type(treas["type"])
    treas = treas.dropna(subset=["type"]).copy()
    treas["type"] = treas["type"].astype(int)
    # Aggregate across currency dimension
    treas_raw = (treas.groupby(["year", "country", "Counterpart", "type"])["amount"]
                 .sum().reset_index())

    treas_keys = treas_raw[["year", "country", "Counterpart", "type"]].drop_duplicates()
    rb_treas = rb.merge(treas_keys, on=["year", "country", "Counterpart", "type"], how="inner")

    treas_r = _apply_bil_restat(treas_raw, rb_treas)
    treas_r = treas_r.merge(ctry_meta, on="counterpart", how="left")
    treas_r["amount"] = treas_r["amount"] * treas_r["Value"]
    # Treasury: equity/fund = domestic; debt follows nationality restatement
    # (simplified: treat all restated Treasury as domestic for Table 1)
    treas_r.loc[
        treas_r["Ynat"].notna() & (treas_r["year"] < treas_r["Ynat"]),
        "counterpart"
    ] = "_OC"
    treas_r = treas_r.drop(columns=["Value", "Counterpart", "Yeuro", "Ynat"], errors="ignore")
    treas_agg = (treas_r.groupby(["year", "country", "counterpart", "type"])["amount"]
                 .sum().reset_index())

    holdings = pd.concat([treas_agg, cpis_agg], ignore_index=True)

    # ── Step 3: Own holdings ──────────────────────────────────────────────
    # foreign_domestic = sum of non-own holders' domestic amounts
    foreign_dom = (
        holdings[holdings["counterpart"] != "_OC"]
        .groupby(["year", "counterpart", "type"])["amount"]
        .sum().reset_index()
        .rename(columns={"amount": "foreign_total"})
    )

    own = (data2[data2["Idomestic"] == 1]
           .merge(foreign_dom, on=["year", "counterpart", "type"], how="left"))
    own["foreign_total"] = own["foreign_total"].fillna(0.0)
    own["amount"] = np.maximum(own["outstand"] - own["foreign_total"], SMALL)
    own["country"] = own["counterpart"]
    own = own[["year", "country", "counterpart", "type", "amount"]].copy()

    # Drop fund shares (type=4)
    all_h = pd.concat([holdings, own], ignore_index=True)
    all_h = all_h[all_h["type"] != 4].copy()

    # Drop _OC counterpart (Table 1 uses domestic-currency markets only)
    all_h = all_h[all_h["counterpart"] != "_OC"].copy()

    data3 = (all_h.groupby(["year", "country", "counterpart", "type"])["amount"]
             .sum().reset_index())

    # ── Step 4: Aggregate investor countries ──────────────────────────────
    inv_ynat = countries[["country", "Ynat"]].drop_duplicates("country")
    data3 = data3.merge(inv_ynat, on="country", how="left")
    oos = (data3["Ynat"].notna() & (data3["year"] < data3["Ynat"])) & (data3["country"] != "_CR")
    data3.loc[oos, "country"] = "_OC"
    data3 = data3.drop(columns=["Ynat"])
    data3 = (data3.groupby(["year", "country", "counterpart", "type"])["amount"]
             .sum().reset_index())

    # Remove _OC investors (keep only identified countries + _CR)
    data3 = data3[data3["country"] != "_OC"].copy()

    print(f"  Data3: {len(data3):,} rows")
    return data3


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1
# ─────────────────────────────────────────────────────────────────────────────

def compute_table1(data3: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    """
    Summary0.do Step 2:
      market   = total(amount)       by year Name type
      domestic = total(amount/market*Iown)  by year Name type
      reserve  = total(amount/market*I_CR)  by year Name type
      keep if year==2020 & Iycounterpart
    """
    df = data3[data3["year"] == 2020].copy()

    df["Iown"] = (df["country"] == df["counterpart"]).astype(int)
    df["I_CR"] = (df["country"] == "_CR").astype(int)

    # Market per (counterpart, type)
    mkt = (df.groupby(["counterpart", "type"])["amount"]
           .sum().reset_index().rename(columns={"amount": "market"}))
    df = df.merge(mkt, on=["counterpart", "type"], how="left")

    df["dom_wt"] = df["amount"] / df["market"] * df["Iown"]
    df["res_wt"] = df["amount"] / df["market"] * df["I_CR"]

    tbl = (df.groupby(["counterpart", "type"])
           .agg(market=("market", "first"),
                domestic=("dom_wt", "sum"),
                reserve=("res_wt", "sum"))
           .reset_index())

    # Merge issuer name + group
    iss = (countries[["country", "Name", "MSCI"]]
           .rename(columns={"country": "counterpart"})
           .drop_duplicates("counterpart"))
    iss["group"] = iss["MSCI"].map(MSCI_GROUP)
    tbl = tbl.merge(iss[["counterpart", "Name", "group"]], on="counterpart", how="left")

    # Euro flag for sorting
    yeuro = (countries[["country", "Yeuro"]]
             .rename(columns={"country": "counterpart"})
             .drop_duplicates("counterpart"))
    tbl = tbl.merge(yeuro, on="counterpart", how="left")
    tbl["Ieuro"] = tbl["Yeuro"].notna().astype(int)

    # Keep only paper sample countries
    tbl = tbl[tbl["group"].notna()].copy()

    # Sort: type, group, -Ieuro, Name  (matches Stata gsort type group -Ieuro Name)
    tbl = tbl.sort_values(
        ["type", "group", "Ieuro", "Name"],
        ascending=[True, True, False, True]
    ).reset_index(drop=True)

    return tbl




# ─────────────────────────────────────────────────────────────────────────────
# PAPER-FORMAT TABLE EXPORT
# ─────────────────────────────────────────────────────────────────────────────

# Region label for each MSCI group (matches paper section headers)
REGION_LABEL = {
    1: "Developed markets: North America",
    2: "Developed markets: Europe",
    3: "Developed markets: Pacific",
    4: "Emerging markets",
}


OUTPUT_DIR = Path(__file__).parent.parent / "_output"


def export_table1(tbl: pd.DataFrame, path: str | None = None) -> None:
    """
    Print (and optionally save) Table 1 in the exact paper layout:

        Table 1
        Market Values of Financial Assets
                        Short-term debt          Long-term debt           Equity
        Issuer          Bil US$  Dom  Res  ...
        ---
        Developed markets: North America
        Canada          521      0.89 0.06  ...
        ...
    """
    # Pivot: one row per (Name, group), three column blocks for type 1/2/3
    wide = {}
    for tp in [1, 2, 3]:
        sub = tbl[tbl["type"] == tp][["Name", "group", "market", "domestic", "reserve"]].copy()
        sub = sub.rename(columns={"market": f"mkt{tp}",
                                  "domestic": f"dom{tp}",
                                  "reserve": f"res{tp}"})
        wide[tp] = sub.set_index("Name")

    # Merge into single wide DataFrame preserving sort order from tbl
    names_order = tbl[tbl["type"] == 1][["Name", "group"]].copy()
    df = names_order.copy()
    for tp in [1, 2, 3]:
        df = df.merge(wide[tp].drop(columns=["group"], errors="ignore"),
                      left_on="Name", right_index=True, how="left")

    lines = []
    SEP = "-" * 100

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        "Table 1",
        "Market Values of Financial Assets",
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
        "Note.—This table reports only local currency debt. "
        "All market values are in billion US dollars at year-end 2020.",
    ]

    output = "\n".join(lines)
    print("\n" + output)

    dest = Path(path) if path else OUTPUT_DIR / "table_1.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output)
    print(f"\n  [Saved to {dest}]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("table_1.py  —  Koijen & Yogo (2020)")
    print("=" * 60)

    print("\nLoading Countries.dta ...")
    ctry = _read("Countries.dta")
    ctry["group_num"] = ctry["MSCI"].astype(str).map(MSCI_GROUP)

    print("\n[Data2] Amounts outstanding")
    data2 = build_data2(ctry)

    print("\n[Data3] Holdings")
    data3 = build_data3(ctry, data2)

    print("\n[Table 1]")
    tbl = compute_table1(data3, ctry)

    export_table1(tbl)


if __name__ == "__main__":
    main()
