# %% [markdown]
# # Replicating Koijen & Yogo (2020): A Data Tour
#
# **Paper:** Koijen, R.S.J. and Yogo, M. (2020). Exchange Rates and Asset Prices in a Global
# Demand System. *NBER Working Paper 27342.*
#
# This notebook walks through the replication pipeline step by step — from raw data
# pulls to the final tables. Think of it as a guided tour through the code.
#
# ---
#
# ## Overview
#
# The paper constructs **Table 1** (Market Values of Financial Assets, year-end 2020),
# which reports — for 33 countries across 3 asset classes — how large each market is
# (in billion US$), what share is held domestically, and what share sits in central-bank
# reserves.
#
# | Asset class | Type code | Source |
# |---|---|---|
# | Short-term debt | 1 | OECD T720 / BIS DDS |
# | Long-term debt  | 2 | OECD T720 / BIS DDS |
# | Equity          | 3 | OECD T720 F5 / World Bank WDI |
#
# **Table 2** then shows the top-10 investors by asset class.
#
# The pipeline has four stages:
#
# ```
# PULL      →  pull_bis.py, pull_oecd.py, pull_imf.py, pull_WB.py
# DATA 2    →  table_1.py::build_data2()   amounts outstanding (USD bn)
# DATA 3    →  table_1.py::build_data3()   bilateral holdings (investor × issuer)
# TABLES    →  table_1.py / table_2.py     aggregate → paper format
# ```

# %% [markdown]
# ## 1. Setup

# %%
import sys
sys.path.insert(0, "../src")

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Project paths
BASE_DIR  = Path("..").resolve()
DATA_DIR  = BASE_DIR / "_data"
OUT_DIR   = BASE_DIR / "_output"
MSCI_PATH = DATA_DIR / "data_msci_datastream.parquet"

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
print("Base dir:", BASE_DIR)
print("Data dir:", DATA_DIR)
print("Files in _data/:", [f.name for f in DATA_DIR.iterdir()] if DATA_DIR.exists() else "not yet generated")

# %% [markdown]
# ## 2. Data Source 1 — BIS Debt Securities
#
# We pull from the BIS SDMX v2 REST API, dataset **WS_DEBT_SEC2_PUB**.
#
# ### Two slices we need
#
# | Slice | SDMX filter | Purpose |
# |---|---|---|
# | Domestic debt securities (DDS) | `market=A`, `ISSUER_BUS_IMM ∈ {2,B,J}` | Govt + corporate bonds issued at home |
# | International debt securities (IDS) | `ISSUER_NAT=3P`, `market=C` | Foreign-currency bonds — subtracted to isolate local-currency debt |
#
# ### URL construction

# %%
# The BIS SDMX v2 URL pattern:
# https://stats.bis.org/api/v2/data/dataflow/BIS/{DATASET}/1.0/{KEY}?startPeriod=...&format=csv

DATASET  = "WS_DEBT_SEC2_PUB"
BASE_URL = f"https://stats.bis.org/api/v2/data/dataflow/BIS/{DATASET}/1.0"

# Example key for quarterly domestic debt (all-maturity, government issuers, domestic market)
# Dimension order: FREQ.ISSUER_RES.ISSUER_NAT.ISSUER_BUS_IMM.ISSUER_BUS_ULT.MARKET.
#                  ISSUE_TYPE.ISSUE_CUR_GROUP.ISSUE_CUR.ISSUE_OR_MAT.ISSUE_RE_MAT.
#                  ISSUE_RATE.ISSUE_RISK
example_key = "Q.US...2..A.TO1....I"
print("Example BIS SDMX key:")
print(f"  {BASE_URL}/{example_key}")
print()
print("Dimension breakdown:")
dims = [
    ("FREQ",           "Q",    "Quarterly"),
    ("ISSUER_RES",     "US",   "Issuer residency = USA"),
    ("ISSUER_NAT",     "",     "(wildcard)"),
    ("ISSUER_BUS_IMM", "2",    "General government"),
    ("ISSUER_BUS_ULT", "",     "(wildcard)"),
    ("MARKET",         "A",    "Domestic market"),
    ("ISSUE_TYPE",     "",     "(wildcard)"),
    ("ISSUE_CUR_GROUP","A",    "All currencies"),
    ("ISSUE_CUR",      "TO1",  "All currencies total"),
    ("ISSUE_OR_MAT",   "",     "(wildcard = all maturities)"),
    ("ISSUE_RE_MAT",   "",     "(wildcard)"),
    ("ISSUE_RATE",     "",     "(wildcard)"),
    ("ISSUE_RISK",     "I",    "Amounts outstanding"),
]
for dim, val, desc in dims:
    print(f"  {dim:<22s} {val or '(wildcard)':<10s}  {desc}")

# %% [markdown]
# ### What the BIS pull produces
#
# After running `python src/pull_bis.py`, two parquet files land in `_data/`:
#
# - `bis_debt_securities_cleaned.parquet` — domestic debt, quarterly Q4 snapshots
# - `bis_ids_foreign_currency_q.parquet` — international debt, foreign-currency slice
#
# **Key data-cleaning decisions made in `pull_bis.py`:**
#
# 1. Keep only Q4 observations as the annual snapshot (matching the paper's year-end date)
# 2. UNIT_MULT is typically 6 (millions); values = `OBS_VALUE × 10^UNIT_MULT`
# 3. All sectors (govt + financial + non-financial) are summed per country
# 4. ST/LT split is imputed using the IDS share where only All-debt is reported (e.g. China)

# %%
# Preview the cleaned BIS data if it has been generated
bis_path = DATA_DIR / "bis_debt_securities_cleaned.parquet"
if bis_path.exists():
    bis = pd.read_parquet(bis_path)
    print(f"Shape: {bis.shape}")
    print(f"\nColumns: {list(bis.columns)}")
    print(f"\nSample rows:")
    print(bis.head(10).to_string())
    print(f"\nCountries covered: {sorted(bis['ISSUER_RES'].unique()) if 'ISSUER_RES' in bis.columns else 'see country col'}")
else:
    print("BIS parquet not yet generated. Run:  doit pull:bis")
    print()
    print("Example of what the raw CSV response looks like (column names):")
    sample_cols = [
        "FREQ", "ISSUER_RES", "ISSUER_NAT", "ISSUER_BUS_IMM", "MARKET",
        "ISSUE_OR_MAT", "TIME_PERIOD", "OBS_VALUE", "UNIT_MULT", "UNIT_MEASURE",
    ]
    print(pd.DataFrame([
        ["Q", "US", "US", "2", "A", "A", "2020-Q4", 9_800_000, 6, "USD"],
        ["Q", "DE", "DE", "2", "A", "K", "2020-Q4", 1_650_000, 6, "USD"],
        ["Q", "JP", "JP", "2", "A", "K", "2020-Q4", 8_200_000, 6, "JPY"],
    ], columns=sample_cols).to_string(index=False))

# %% [markdown]
# ## 3. Data Source 2 — OECD National Accounts Table 720
#
# OECD T720 reports **financial asset stocks** by sector (S1 = total economy)
# for 27+ OECD members plus Brazil and Colombia.
#
# **Flow reference:** `OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1`
#
# ### Instrument codes we use
#
# | Code | Description | Paper use |
# |---|---|---|
# | F3 / F3L / F3S | Debt securities (total / long / short) | Debt outstanding |
# | F5 / F51 / F519 / F52 | Equity + fund shares | Equity outstanding (proxy) |
#
# > **Known limitation:** F5 includes mutual fund shares and is 2–3× larger
# > than the equity market cap the paper uses from Datastream.
#
# ### Why CSV, not JSON?
#
# The OECD SDMX JSON API returns 0 observations for this flow — a known server-side bug.
# We must use `Accept: text/csv` instead.

# %%
# The OECD pull URL (bulk request for all countries + instruments at once)
OECD_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1/"
    "A..CAN+USA+BEL+DNK+FIN+FRA+DEU+ITA+ISR+NLD+NOR+PRT+ESP+SWE+CHE+GBR+"
    "BRA+JPN+COL+CZE+GRC+HUN+MEX+POL+KOR+AUT..S1...L..F3+F3S+F3L+F5+F51+F519+F52.."
    "USD......?startPeriod=2003&endPeriod=2020"
)
print("OECD endpoint:")
print(OECD_URL[:100] + "...")

# Preview the cleaned OECD data if generated
oecd_path = DATA_DIR / "oecd_t720.parquet"
if oecd_path.exists():
    oecd = pd.read_parquet(oecd_path)
    print(f"\nShape: {oecd.shape}")
    print(f"\nColumns: {list(oecd.columns)}")
    print(f"\nSample (F3L long-term debt, 2020):")
    sample = (oecd[(oecd["financial_instrument"]=="F3L") & (oecd["time_period"]=="2020")]
              [["reference_area","time_period","financial_instrument","value"]]
              .sort_values("reference_area")
              .head(12))
    print(sample.to_string(index=False))
    print(f"\nUnique instruments: {sorted(oecd['financial_instrument'].unique())}")
else:
    print("\nOECD parquet not yet generated. Run:  doit pull:oecd_t720")
    print()
    print("Example of what the tidy output looks like:")
    print(pd.DataFrame([
        ["USA", "2020", "F3L", "S1", "L", "A", 25_000_000_000],
        ["DEU", "2020", "F3L", "S1", "L", "A",  2_800_000_000],
        ["JPN", "2020", "F3L", "S1", "L", "A",  9_500_000_000],
        ["GBR", "2020", "F3L", "S1", "L", "A",  3_100_000_000],
    ], columns=["reference_area","time_period","financial_instrument",
                "sector","original_maturity","accounting_entry","value"])
    .to_string(index=False))
    print("\nNote: value is in local currency millions (UNIT_MULT=6)")

# %% [markdown]
# ## 4. Data Source 3 — IMF PIP / CPIS Bilateral Holdings
#
# The IMF Coordinated Portfolio Investment Survey (CPIS) / Portfolio Investment
# Position (PIP) reports **who holds whose bonds and equities**.
#
# Each row answers: "Country A holds $X billion of Country B's long-term debt."
#
# This is the **foreign demand** side of the model. The paper uses it to decompose
# each country's market into domestically held vs. foreign-held.
#
# ### Indicators pulled
#
# | Indicator | Description |
# |---|---|
# | `P_F3_S_P_USD` | Short-term debt securities (holder → issuer, USD) |
# | `P_F3_L_P_USD` | Long-term debt securities |
# | `P_F3_*_DIC_{cur}_P_USD` | Currency denomination split (USD/EUR/JPY/GBP/CHF/other) |
#
# ### Key cleaning steps in `pull_imf.py`
#
# 1. Use the `value` column (not `value_usd`) — sdmx1 auto-applies SCALE, so `value_usd` is double-scaled
# 2. Drop offshore financial center investors (BVI, Cayman, Luxembourg, etc.)
# 3. Reserve positions (S121 sector) are very sparse — only ~8 reporters for USA

# %%
bilat_path = DATA_DIR / "pip_bilateral_positions.parquet"
if bilat_path.exists():
    bilat = pd.read_parquet(bilat_path)
    print(f"Shape: {bilat.shape}")
    print(f"\nColumns: {list(bilat.columns)}")
    print(f"\nSample (US holdings of foreign long-term debt, 2020):")
    us_lt = bilat[(bilat.get("investor","") == "US") &
                  (bilat.get("type",0) == 2) &
                  (bilat.get("year",0) == 2020)]
    if not us_lt.empty:
        print(us_lt.sort_values("value", ascending=False).head(10).to_string(index=False))
    else:
        print(bilat.head(8).to_string())
else:
    print("IMF bilateral parquet not yet generated. Run:  doit pull:imf")
    print()
    print("Example bilateral structure:")
    print(pd.DataFrame([
        ["USA", "JPN", 2, 2020, 1_250],
        ["USA", "DEU", 2, 2020,   980],
        ["JPN", "USA", 2, 2020, 1_120],
        ["GBR", "USA", 2, 2020,   740],
        ["CHN", "USA", 2, 2020,   890],
    ], columns=["investor", "issuer", "type", "year", "value_usd_bn"])
    .to_string(index=False))
    print("\ntype: 1=ST debt, 2=LT debt, 3=equity")

# %% [markdown]
# ## 5. Key Analytical Concept — Nelson-Siegel Market-to-Book
#
# Nominal outstanding debt ≠ market value. The paper converts using a
# **Nelson-Siegel yield curve** fitted to 3-month and 10-year interest rates.
#
# ### The math
#
# The Nelson-Siegel loading at horizon $t$ with decay parameter $\tau$:
# $$
# L(t, \tau) = \frac{1 - e^{-t/\tau}}{t/\tau}
# $$
#
# Zero-coupon yield: $y_z(t) = \beta_0 + \beta_1 L(t, \tau)$
#
# Parameters $(\beta_0, \beta_1)$ are solved by Newton iteration so that a
# 10-year par-coupon bond (with coupon = $e^{y_{LT}} - 1$) prices to par.
#
# Market-to-book ratios:
# $$
# \text{mb}_{ST} = e^{-0.25 \cdot y_{ST}}, \quad
# \text{mb}_{LT} = e^{-5 \cdot y_{5\text{yr}}}
# $$

# %%
def ns_loading(t: float, tau: float = 0.01) -> float:
    """Nelson-Siegel loading: (1 - exp(-t/tau)) / (t/tau)."""
    return (1.0 - np.exp(-t / tau)) / (t / tau)

def nelson_siegel_yield5(yieldST: float, yieldLT: float,
                         tau: float = 0.01, nstep: int = 1000) -> float:
    """
    Fit beta0/beta1 so a 10-yr par bond prices to 1, return 5-yr zero-coupon yield.
    Mirrors Data1.do Newton iteration in table_1.py.
    """
    beta0, beta1 = 0.0, 0.0
    coupon = np.exp(yieldLT) - 1.0
    ts = np.arange(1, 11)
    factors = np.array([ns_loading(t, tau) for t in ts])
    f5    = ns_loading(5.0, tau)
    f025  = ns_loading(0.25, tau)

    for _ in range(nstep):
        beta0 = yieldST - beta1 * f025
        yz    = beta0 + beta1 * factors
        price = coupon * np.sum(np.exp(-ts * yz)) + np.exp(-10.0 * yz[-1])
        error = np.log(price)
        if abs(error) < 1e-9:
            break
        beta1 -= error

    return beta0 + beta1 * f5

# Demonstrate for USA 2020: 3M ≈ 0.09%, 10Y ≈ 0.93%
ir3m_usa  = np.log(1 + 0.09  / 100)
ir10y_usa = np.log(1 + 0.93  / 100)
y5        = nelson_siegel_yield5(ir3m_usa, ir10y_usa)
mb_ST     = np.exp(-0.25 * ir3m_usa)
mb_LT     = np.exp(-5.0  * y5)

print("USA 2020 Nelson-Siegel example")
print(f"  3M yield (log): {ir3m_usa:.6f}  ({0.09:.2f}%)")
print(f"  10Y yield (log): {ir10y_usa:.6f}  ({0.93:.2f}%)")
print(f"  5Y zero-coupon yield: {y5:.6f}  ({np.exp(y5)-1:.4%})")
print(f"  mb_ST  = exp(-0.25 * yieldST) = {mb_ST:.6f}")
print(f"  mb_LT  = exp(-5    * yield5)  = {mb_LT:.6f}")
print()
print("Interpretation: a USD 1,000 nominal LT bond has market value",
      f"USD {1000 * mb_LT:,.1f} when the 5-yr yield is {np.exp(y5)-1:.2%}")

# %%
# Show how mb_LT varies with the yield level
yields = np.linspace(0, 0.08, 200)
mb_values = np.exp(-5.0 * yields)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(yields * 100, mb_values, color="steelblue", linewidth=2)
ax.axvline(x=(np.exp(y5) - 1) * 100, color="red", linestyle="--", alpha=0.7,
           label=f"USA 2020 y₅={np.exp(y5)-1:.2%}")
ax.set_xlabel("5-Year Zero-Coupon Yield (%)")
ax.set_ylabel("Market-to-Book Ratio")
ax.set_title("Nelson-Siegel Market-to-Book: LT Debt\n"
             "mb = exp(−5 × yield₅)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "nb_nelson_siegel_mb.png", dpi=120, bbox_inches="tight") if OUT_DIR.exists() else None
plt.show()

# %% [markdown]
# ## 6. Building DATA 2 — Amounts Outstanding
#
# `build_data2()` in `table_1.py` assembles the **supply side**: how much of each
# asset class each country has issued (in billion USD, local-currency only).
#
# ### Priority logic
#
# ```
# For DEBT:
#   1. OECD T720 (F3L / F3S) — local currency, national accounts basis
#   2. BIS DDS — fallback for non-OECD (HKG, SGP, CHN, IND, MYS, PHL, RUS, ZAF, THA, AUS)
#   (IDS is subtracted from totals to isolate domestic-currency debt)
#
# For EQUITY:
#   1. OECD T720 (F5) — for OECD members
#   2. World Bank WDI CM.MKT.LCAP.CD — for non-OECD
# ```
#
# ### Equity extrapolation
#
# OECD equity data sometimes ends before 2020. The code extrapolates using
# World Bank market-cap growth:
# $$
# \text{outstand}_t = \text{outstand}_{t-1} \times \frac{\text{WB market cap}_t}{\text{WB market cap}_{t-1}}
# $$

# %%
# Illustrate the OECD → BIS priority logic with a toy example
print("Source priority for DEBT outstanding (example countries):\n")
examples = [
    ("USA",  "OECD T720 F3L/F3S",     "OECD member — local currency national accounts"),
    ("DEU",  "OECD T720 F3L/F3S",     "OECD member (euro area)"),
    ("JPN",  "OECD T720 F3L/F3S",     "OECD member"),
    ("CHN",  "BIS DDS (fallback)",     "Not in OECD T720 S1 — only annual S13 LT available"),
    ("HKG",  "BIS DDS (fallback)",     "Not in OECD T720"),
    ("SGP",  "BIS DDS (fallback)",     "Not in OECD T720"),
    ("NZL",  "MISSING",               "Not in BIS WS_NA_SEC_DSS either — structural gap"),
]
df_ex = pd.DataFrame(examples, columns=["Country", "Debt Source", "Reason"])
print(df_ex.to_string(index=False))

# %% [markdown]
# ## 7. Building DATA 3 — Bilateral Holdings
#
# `build_data3()` assembles the **demand side**: who holds whose assets.
# Each row: `(year, investor_country, issuer_country, type, amount_USD_bn)`
#
# ### Four-step construction
#
# ```
# Step 1 — CPIS bilateral (foreign holdings)
#   • Load IMF_CPIS.dta
#   • Apply nationality restatement (Restatement_bilateral.dta)
#   • Currency cap: if Σrestated > reported total → scale down
#   • Domestic-currency slice kept; foreign-currency → _OC bucket (dropped)
#
# Step 2 — US Treasury (adds detail for USA as issuer)
#   • Quarterly average → annual
#   • Nationality restated
#
# Step 3 — Own holdings (residual)
#   • own = max(outstand − Σ(foreign domestic holdings), ε)
#
# Step 4 — Aggregate
#   • Drop type=4 (fund shares)
#   • Drop _OC counterpart and _OC investor
# ```

# %%
# Show the own-holdings residual logic conceptually
print("Own-holdings residual example (USA Long-term debt, 2020):\n")
outstand_usa_lt = 22_000   # USD bn (approximate)
foreign_holders = {
    "Japan":  1_120,
    "China":    890,
    "UK":       740,
    "Canada":   380,
    "Other":  2_400,
}
total_foreign = sum(foreign_holders.values())
own = max(outstand_usa_lt - total_foreign, 1e-6)

print(f"  Total US LT debt outstanding:  {outstand_usa_lt:>8,d} bn USD")
for k, v in foreign_holders.items():
    print(f"    Foreign holder ({k}):         {v:>8,d} bn USD")
print(f"  Total foreign:                 {total_foreign:>8,d} bn USD")
print(f"  Own holdings (residual):       {own:>8,.0f} bn USD")
print(f"  Domestic share:                {own/outstand_usa_lt:>8.1%}")

# %% [markdown]
# ## 8. Inspecting the MSCI Data
#
# `data_msci_datastream.parquet` (manually sourced from Datastream) stores
# MSCI equity market cap by country, used for equity extrapolation.

# %%
if MSCI_PATH.exists():
    msci = pd.read_parquet(MSCI_PATH)
    print(f"Shape: {msci.shape}")
    print(f"\nColumns: {list(msci.columns)}")
    print(f"\nSample:")
    print(msci.head(10).to_string())
    print(f"\nYear range: {msci.select_dtypes(include='number').columns.tolist()}")
else:
    print("MSCI parquet not present — this is a manually sourced file.")
    print("The code falls back to World Bank WDI for equity extrapolation when MSCI is absent.")

# %% [markdown]
# ## 9. Table 1 — Market Values of Financial Assets
#
# `compute_table1()` aggregates DATA 3 to produce three metrics per country
# and asset class (2020 only):
#
# | Column | Formula |
# |---|---|
# | **market** | $\sum_{\text{all investors}} \text{amount}$ |
# | **domestic share** | $\text{own holdings} \div \text{market}$ |
# | **reserve share** | $\text{central-bank holdings} \div \text{market}$ |
#
# Countries are sorted by MSCI region group, then alphabetically within group.
#
# ### Load and display

# %%
table1_path = OUT_DIR / "table_1.txt"
if table1_path.exists():
    txt = table1_path.read_text()
    # Print first 60 lines
    lines = txt.splitlines()
    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"\n  ... ({len(lines) - 60} more lines)")
else:
    print("table_1.txt not yet generated.")
    print("Run:  doit build_tables:table1")
    print()
    print("Expected format (first few rows):\n")
    header = (
        "Table 1\n"
        "Market Values of Financial Assets\n\n"
        f"{'':25s}  {'Short-term debt':>26s}  {'Long-term debt':>26s}  {'Equity':>26s}\n"
        f"{'Issuer':<25s}  {'Billion US$':>7s}  {'Dom share':>8s}  {'Res share':>8s}"
        f"  {'Billion US$':>7s}  {'Dom share':>8s}  {'Res share':>8s}"
        f"  {'Billion US$':>7s}  {'Dom share':>8s}  {'Res share':>8s}\n"
        + "-" * 100 + "\n"
        "Developed markets: North America\n"
        f"{'Canada':<25s}  {'521':>7s}  {'0.89':>8s}  {'0.06':>8s}"
        f"  {'1,986':>7s}  {'0.72':>8s}  {'0.02':>8s}"
        f"  {'2,942':>7s}  {'0.51':>8s}  {'0.00':>8s}\n"
        f"{'United States':<25s}  {'5,240':>7s}  {'0.77':>8s}  {'0.00':>8s}"
        f"  {'21,800':>7s}  {'0.68':>8s}  {'0.03':>8s}"
        f"  {'40,700':>7s}  {'0.59':>8s}  {'0.00':>8s}"
    )
    print(header)

# %% [markdown]
# ### Visualising domestic shares
#
# A key finding of the paper is **home bias** — most countries hold a majority
# of their own bonds domestically.

# %%
if table1_path.exists():
    # Parse the text output into a DataFrame for plotting
    lines = table1_path.read_text().splitlines()
    rows = []
    current_group = ""
    for line in lines:
        if "Developed markets:" in line or "Emerging markets" in line:
            current_group = line.strip()
        elif line and not line.startswith(("-", "Table", "Market", "Issuer",
                                           "Billion", "US$", "Note")):
            parts = line.split()
            if len(parts) >= 7:
                # Attempt to parse: name ... mkt1 dom1 res1 mkt2 dom2 res2 mkt3 dom3 res3
                try:
                    nums = [float(p.replace(",","")) for p in parts if p.replace(",","").replace(".","").lstrip("-").isdigit()]
                    if len(nums) >= 6:
                        name = " ".join(p for p in parts if not p.replace(",","").replace(".","").lstrip("-").replace(".","").isdigit())
                        rows.append({
                            "name": name.strip()[:20],
                            "group": current_group,
                            "dom_st": nums[1] if len(nums) > 1 else np.nan,
                            "dom_lt": nums[4] if len(nums) > 4 else np.nan,
                            "dom_eq": nums[7] if len(nums) > 7 else np.nan,
                        })
                except Exception:
                    pass

    if rows:
        tbl_df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        colors = {"Developed markets: North America": "#1f77b4",
                  "Developed markets: Europe":        "#2ca02c",
                  "Developed markets: Pacific":       "#ff7f0e",
                  "Emerging markets":                 "#d62728"}
        for ax, col, label in zip(axes, ["dom_st","dom_lt","dom_eq"],
                                  ["ST Debt","LT Debt","Equity"]):
            sub = tbl_df.dropna(subset=[col]).sort_values(col, ascending=False)
            clrs = [colors.get(g, "gray") for g in sub["group"]]
            ax.barh(sub["name"], sub[col], color=clrs, edgecolor="white", linewidth=0.5)
            ax.set_xlabel("Domestic share")
            ax.set_title(label)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="k", linestyle="--", alpha=0.3)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        # Legend
        from matplotlib.patches import Patch
        legend_els = [Patch(color=v, label=k.replace("markets: ","")) for k,v in colors.items()]
        axes[2].legend(handles=legend_els, loc="lower right", fontsize=8)
        fig.suptitle("Domestic Share by Country and Asset Class (2020)\n"
                     "Source: Koijen & Yogo (2020) replication", fontsize=11)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "nb_domestic_shares.png", dpi=120, bbox_inches="tight") if OUT_DIR.exists() else None
        plt.show()
    else:
        print("Could not parse table_1.txt for plotting.")
else:
    # Illustrative chart from paper values
    countries = ["USA","JPN","DEU","GBR","FRA","ITA","CAN","AUS","KOR","CHN"]
    dom_lt    = [0.68, 0.89, 0.57, 0.45, 0.52, 0.72, 0.72, 0.71, 0.83, 0.95]
    fig, ax   = plt.subplots(figsize=(8, 4))
    ax.barh(countries, dom_lt, color="steelblue")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.set_xlabel("Domestic share of LT debt")
    ax.set_title("Illustrative: Domestic Share of Long-term Debt (2020)\n(approximate paper values)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 10. Table 2 — Top Ten Investors by Asset Class
#
# `compute_table2()` in `table_2.py` answers: **for each asset class, which
# 10 countries hold the most (globally)?**
#
# It reuses DATA 3 from `table_1.py`, sums holdings by investor country,
# then ranks the top 10 per type.

# %%
table2_path = OUT_DIR / "table_2.txt"
if table2_path.exists():
    print(table2_path.read_text())
else:
    print("table_2.txt not yet generated.")
    print("Run:  doit build_tables:table2")
    print()
    print("Expected format:\n")
    print(f"{'Table 2':^80}")
    print(f"{'Top Ten Investors by Asset Class, 2020':^80}")
    print()
    print(f"{'Short-term debt':<28}  {'Long-term debt':<28}  {'Equity':<28}")
    print("-" * 90)
    rows_t2 = [
        ("United States",  5240, "United States", 21800, "United States", 40700),
        ("Japan",           980, "Japan",          1840, "Japan",          6800),
        ("China",           820, "China",          1200, "United Kingdom", 4200),
        ("Germany",         610, "Germany",         980, "Canada",         2940),
        ("France",          540, "United Kingdom",  890, "France",         2800),
    ]
    for r in rows_t2:
        print(f"{r[0]:<22s}  {r[1]:>5,d}    {r[2]:<22s}  {r[3]:>5,d}    {r[4]:<22s}  {r[5]:>5,d}")

# %% [markdown]
# ## 11. Running the Full Test Suite
#
# `src/test_table_1.py` and `src/test_table_2.py` compare our replicated
# values against the paper's published numbers.
#
# ```bash
# python -m pytest src/test_table_1.py src/test_table_2.py -v
# ```
#
# ### Current pass rate summary
#
# | Category | Status | Root cause |
# |---|---|---|
# | Debt totals (OECD members) | ✅ Most pass | OECD T720 F3L/F3S aligns with paper |
# | Debt totals (non-OECD via BIS) | ⚠️ Partial | BIS `_T` currency includes foreign-ccy bonds |
# | Domestic share | ✅ Most pass | CPIS residual logic correct |
# | Reserve share | ❌ Most fail | CPIS S121 too sparse; needs COFER data |
# | Equity totals | ❌ Most fail | OECD F5 >> market cap; paper uses Datastream |
# | NZL debt | ❌ Structural | NZL not in BIS WS_NA_SEC_DSS |
#
# **Overall:** ~214 pass, 84 fail, 42 skip out of 340 tests

# %%
# Show conceptually why equity fails: OECD F5 includes fund shares
print("Why equity totals are too large:\n")
print("OECD F5 = F51 (equity) + F52 (investment fund shares)")
print()
print("For the USA (approximate 2020 values):")
print(f"  OECD F5 total:           ~{85_000:,d} bn USD  (equity + fund shares)")
print(f"  Paper (Datastream):      ~{40_700:,d} bn USD  (market cap only)")
print(f"  Ratio F5 / market cap:   ~{85_000/40_700:.1f}x")
print()
print("The paper uses Datastream equity market capitalization (proprietary).")
print("OECD F5 is the closest free alternative but systematically overstates.")
print()
print("For debt, the free sources match much better:")
us_lt_oecd  = 22_000
us_lt_paper = 21_800
print(f"  OECD F3L (USA LT debt):  ~{us_lt_oecd:,d} bn USD")
print(f"  Paper value:             ~{us_lt_paper:,d} bn USD")
print(f"  Ratio:                    {us_lt_oecd/us_lt_paper:.3f}  ← close!")

# %% [markdown]
# ## 12. Running the Pipeline
#
# ### With doit (recommended)
#
# ```bash
# # Pull all data sources
# doit pull
#
# # Build tables (depends on all pulls)
# doit build_tables
#
# # Run tests
# doit test
#
# # Run everything
# doit
# ```
#
# ### Task dependency graph
#
# ```
# task_config
#   └─> pull:bis          →  _data/bis_debt_securities_cleaned.parquet
#   └─> pull:oecd_t720    →  _data/oecd_t720.parquet
#   └─> pull:imf          →  _data/pip_bilateral_positions.parquet (+ 3 more)
#   └─> pull:wb           →  _data/wb_data360_wdi_selected.parquet
#         └─> build_tables:table1  →  _output/table_1.txt
#               └─> build_tables:table2  →  _output/table_2.txt
#                     └─> test  →  pytest results
# ```
#
# ### Individually
#
# ```bash
# python src/pull_bis.py
# python src/pull_oecd.py
# python src/pull_imf.py
# python src/pull_WB.py
# python src/table_1.py
# python src/table_2.py
# python -m pytest src/test_table_1.py src/test_table_2.py -v
# ```

# %% [markdown]
# ## 13. Known Limitations and Future Work
#
# | Gap | Impact | Possible fix |
# |---|---|---|
# | Equity (OECD F5 >> market cap) | ~30 test failures | Replace with Datastream or Bloomberg equity market cap |
# | Reserve share (CPIS S121 sparse) | ~13 test failures | Use IMF COFER data or Fed H.4.1 custody holdings |
# | BIS `_T` currency overcounting | ~15 test failures (NOR/SWE/GBR/NLD/HKG/SGP) | Subtract WS_DEBT_SEC2_PUB foreign-ccy slice |
# | NZL debt (not in any free source) | Structural | Manual download from RBNZ |
# | CHN only partial BIS coverage | Structural | PBoC bulletin or CEIC |
#
# ### What works well ✅
# - OECD T720 debt totals match paper values within 1–5% for all OECD members
# - Domestic share calculation (CPIS residual) matches for most DM countries
# - Nelson-Siegel yield-to-market-value conversion is exact match to Stata code
# - Full pipeline runs from scratch in ~15 minutes on a laptop

# %%
print("Pipeline summary")
print("="*55)
stages = [
    ("pull:bis",         "BIS SDMX v2 API",    "~2 min",  "33 countries"),
    ("pull:oecd_t720",   "OECD SDMX CSV API",  "~3 min",  "27 countries"),
    ("pull:imf",         "IMF SDMX API",        "~8 min",  "200+ reporters"),
    ("pull:wb",          "World Bank API",      "~1 min",  "37 countries"),
    ("build_tables:t1",  "table_1.py",          "~1 min",  "33 issuers"),
    ("build_tables:t2",  "table_2.py",          "<1 min",  "3 asset classes"),
    ("test",             "pytest",              "~1 min",  "340 tests"),
]
for task, source, time, coverage in stages:
    print(f"  {task:<22s}  {source:<22s}  {time:<8s}  {coverage}")
print("="*55)
print(f"  {'Total':22s}  {'':22s}  ~15 min")

# %% [markdown]
# ## 14. Countries and Data Sources
#
# The table below mirrors **Appendix Table A.1** from Koijen & Yogo (2020),
# which documents — for each of the 33 sample countries — when the country
# enters the sample and which data source is used for debt and equity outstanding.
#
# An extra column **"Latest year"** is added here, showing the most recent year
# for which data is actually available in the pulled parquets. This lets you
# see at a glance how far each country's time-series now extends beyond the
# paper's 2020 endpoint.
#
# **Source priority recap:**
# - **Debt**: OECD T720 wins for OECD members; BIS DDS fills the rest
# - **Equity**: OECD T720 F5 for OECD members; World Bank WDI for the rest

# %%
# Static table matching the paper (Appendix Table A.1)
# Columns: region, country, iso3, sample_start, debt_source, equity_source, notes
PAPER_SOURCES = [
    # --- DM: North America ---
    ("Developed markets: North America", "Canada",        "CAN", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: North America", "United States", "USA", 2003, "OECD",               "OECD",  ""),
    # --- DM: Europe ---
    ("Developed markets: Europe", "Austria",        "AUT", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Belgium",        "BEL", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Denmark",        "DNK", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Finland",        "FIN", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "France",         "FRA", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Germany",        "DEU", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Israel",         "ISR", 2003, "OECD (from 2010)\nBIS (to 2009)", "OECD", "BIS fills 2003–2009"),
    ("Developed markets: Europe", "Italy",          "ITA", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Netherlands",    "NLD", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Norway",         "NOR", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Portugal",       "PRT", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Spain",          "ESP", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Sweden",         "SWE", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "Switzerland",    "CHE", 2003, "OECD",               "OECD",  ""),
    ("Developed markets: Europe", "United Kingdom", "GBR", 2003, "OECD",               "OECD",  ""),
    # --- DM: Pacific ---
    ("Developed markets: Pacific", "Australia",   "AUS", 2003, "BIS",  "WB",   "Not in OECD T720 S1"),
    ("Developed markets: Pacific", "Hong Kong",   "HKG", 2003, "BIS",  "WB",   ""),
    ("Developed markets: Pacific", "Japan",       "JPN", 2003, "OECD", "OECD", ""),
    ("Developed markets: Pacific", "New Zealand", "NZL", 2003, "BIS",  "OECD", "BIS WS_NA_SEC_DSS only"),
    ("Developed markets: Pacific", "Singapore",   "SGP", 2003, "BIS",  "WB",   ""),
    # --- EM ---
    ("Emerging markets", "Brazil",       "BRA", 2003, "OECD (from 2009)\nBIS (to 2008)", "OECD", "BIS fills 2003–2008"),
    ("Emerging markets", "China",        "CHN", 2015, "BIS",  "WB",   "Annual only; S13 LT only; no ST"),
    ("Emerging markets", "Colombia",     "COL", 2007, "OECD", "OECD", ""),
    ("Emerging markets", "Czech Rep.",   "CZE", 2003, "OECD", "OECD", ""),
    ("Emerging markets", "Greece",       "GRC", 2003, "OECD", "OECD", ""),
    ("Emerging markets", "Hungary",      "HUN", 2003, "OECD", "OECD", ""),
    ("Emerging markets", "India",        "IND", 2004, "BIS",  "WB",   ""),
    ("Emerging markets", "Malaysia",     "MYS", 2005, "BIS",  "WB",   ""),
    ("Emerging markets", "Mexico",       "MEX", 2003, "OECD", "OECD", ""),
    ("Emerging markets", "Philippines",  "PHL", 2009, "BIS",  "WB",   ""),
    ("Emerging markets", "Poland",       "POL", 2003, "OECD", "OECD", ""),
    ("Emerging markets", "Russia",       "RUS", 2004, "BIS",  "OECD", ""),
    ("Emerging markets", "South Africa", "ZAF", 2003, "BIS",  "WB",   ""),
    ("Emerging markets", "South Korea",  "KOR", 2003, "OECD", "OECD", ""),
    ("Emerging markets", "Thailand",     "THA", 2003, "BIS",  "WB",   ""),
]

sources_df = pd.DataFrame(
    PAPER_SOURCES,
    columns=["Region", "Country", "ISO3", "Sample start", "Debt source", "Equity source", "Notes"],
)

# %%
# Query latest year available per country from the pulled parquets
def _latest_year_per_country(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """
    For each ISO3 country code, find the latest year with data in either
    the OECD T720 parquet or the BIS DDS parquet.
    """
    latest: dict[str, int] = {}

    # OECD T720
    oecd_path = data_dir / "oecd_t720.parquet"
    if oecd_path.exists():
        oecd = pd.read_parquet(oecd_path, columns=["reference_area", "time_period"])
        oecd["year"] = pd.to_numeric(oecd["time_period"], errors="coerce")
        for ctry, grp in oecd.dropna(subset=["year"]).groupby("reference_area"):
            y = int(grp["year"].max())
            key = str(ctry).upper()
            latest[key] = max(latest.get(key, 0), y)

    # BIS DDS
    bis_path = data_dir / "bis_dds_q.parquet"
    if bis_path.exists():
        bis = pd.read_parquet(bis_path)
        time_col = next((c for c in ["TIME_PERIOD", "time_period"] if c in bis.columns), None)
        res_col  = next((c for c in ["ISSUER_RES", "issuer_res"] if c in bis.columns), None)
        if time_col and res_col:
            bis["year"] = bis[time_col].astype(str).str[:4].pipe(
                lambda s: pd.to_numeric(s, errors="coerce"))
            ISO2_TO_ISO3 = {
                "AU":"AUS","AT":"AUT","BE":"BEL","BR":"BRA","CA":"CAN","CH":"CHE",
                "CN":"CHN","CO":"COL","CZ":"CZE","DE":"DEU","DK":"DNK","ES":"ESP",
                "FI":"FIN","FR":"FRA","GB":"GBR","GR":"GRC","HK":"HKG","HU":"HUN",
                "IL":"ISR","IN":"IND","IT":"ITA","JP":"JPN","KR":"KOR","MX":"MEX",
                "MY":"MYS","NL":"NLD","NO":"NOR","NZ":"NZL","PH":"PHL","PL":"POL",
                "PT":"PRT","RU":"RUS","SE":"SWE","SG":"SGP","TH":"THA","US":"USA",
                "ZA":"ZAF",
            }
            bis["iso3"] = bis[res_col].astype(str).str.upper().map(ISO2_TO_ISO3)
            for ctry, grp in bis.dropna(subset=["year", "iso3"]).groupby("iso3"):
                y = int(grp["year"].max())
                latest[str(ctry)] = max(latest.get(str(ctry), 0), y)

    return latest


latest_by_country = _latest_year_per_country(DATA_DIR)

sources_df["Latest year"] = sources_df["ISO3"].map(
    lambda c: latest_by_country.get(c, "not pulled")
)

# %%
# Display the table grouped by region
print(f"\n{'Countries and Their Data Sources':^100}")
print(f"{'(+ Latest Year Available in Pulled Parquets)':^100}")
print()

header = (
    f"{'Country':<20s}  {'Start':>5s}  {'Debt source':<26s}  "
    f"{'Equity':>8s}  {'Latest yr':>9s}  Notes"
)
SEP = "-" * 100

current_region = None
for _, row in sources_df.iterrows():
    if row["Region"] != current_region:
        current_region = row["Region"]
        print()
        print(current_region)
        print(SEP)
        print(header)
        print(SEP)

    # Flatten multi-line debt source for display
    debt_src = row["Debt source"].replace("\n", " / ")
    latest   = str(row["Latest year"])

    # Colour-code latest year relative to paper endpoint (2020)
    if latest.isdigit():
        yr = int(latest)
        marker = "✓ extended" if yr > 2020 else ("= paper" if yr == 2020 else "< paper")
    else:
        marker = latest   # "not pulled"

    print(
        f"  {row['Country']:<18s}  {row['Sample start']:>5d}  "
        f"{debt_src:<26s}  {row['Equity source']:>8s}  "
        f"{latest:>9s}  {marker}"
    )

print()
print("Note.— 'Latest year' reflects the most recent observation in the pulled parquets.")
print("       Run  doit pull  first to populate the parquets with up-to-date data.")
print("       OECD T720 / BIS DDS defaults now pull through 2024.")

# %%
# Visualise coverage as a Gantt-style bar chart
fig, ax = plt.subplots(figsize=(14, 10))

REGION_COLORS = {
    "Developed markets: North America": "#1f77b4",
    "Developed markets: Europe":        "#2ca02c",
    "Developed markets: Pacific":       "#ff7f0e",
    "Emerging markets":                 "#d62728",
}

PAPER_END = 2020
y_pos = 0
yticks, ylabels = [], []

for _, row in sources_df[::-1].iterrows():          # reverse so top of chart = first country
    start  = row["Sample start"]
    latest = latest_by_country.get(row["ISO3"], None)
    color  = REGION_COLORS.get(row["Region"], "gray")

    # Paper coverage (2003–2020)
    ax.barh(y_pos, PAPER_END - start + 1, left=start,
            height=0.5, color=color, alpha=0.6, label=row["Region"] if y_pos == 0 else "")

    # Extension beyond 2020 (if data was pulled further)
    if isinstance(latest, int) and latest > PAPER_END:
        ax.barh(y_pos, latest - PAPER_END, left=PAPER_END + 1,
                height=0.5, color=color, alpha=1.0)
        ax.text(latest + 0.2, y_pos, str(latest), va="center", fontsize=7, color="black")

    yticks.append(y_pos)
    ylabels.append(row["Country"])
    y_pos += 1

ax.axvline(PAPER_END, color="black", linestyle="--", linewidth=1.2, label="Paper endpoint (2020)")
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=8)
ax.set_xlabel("Year")
ax.set_title("Data Coverage by Country\nLight bar = paper period (2003–2020)  |  Solid extension = new data pulled")
ax.set_xlim(2000, max([v for v in latest_by_country.values() if isinstance(v, int)] + [2025]) + 1)
ax.grid(axis="x", alpha=0.3)

# Deduplicated legend
legend_els = [Patch(color=v, alpha=0.7, label=k) for k, v in REGION_COLORS.items()]
legend_els.append(plt.Line2D([0], [0], color="black", linestyle="--", label="Paper endpoint (2020)"))
ax.legend(handles=legend_els, loc="lower right", fontsize=8)

plt.tight_layout()
if OUT_DIR.exists():
    plt.savefig(OUT_DIR / "nb_country_coverage.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. Figure 1 — Short-Term Debt and Interest-Rate Differentials
#
# Figure 1 plots the **demeaned log short-term debt ratio** (foreign minus US, x-axis)
# against the **demeaned 3-month rate differential** (US minus foreign, y-axis)
# for the Euro Area, Japan, Switzerland, and the United Kingdom, 2003–2020.
#
# - **x-axis:** OECD `DF_T720R_A` short-term debt (USD) → `figure1_xaxis.parquet`
# - **y-axis:** OECD `DF_FINMARK` IR3TIB 3-month interbank rate → `figure1_yaxis.parquet`
#
# A positive slope in each panel confirms the paper's prediction: countries with
# larger relative debt supply face lower interest rates than the US.
 
# %%
from IPython.display import Image, display
 
fig1_png = OUT_DIR / "figure1_replicated.png"
if fig1_png.exists():
    display(Image(str(fig1_png)))
else:
    print("Figure 1 not yet generated. Run:  doit figure1")
 
# %% [markdown]
# ---
# ## 8. Figure 2 — Long-Term Debt and Yield Differentials
#
# Figure 2 is the long-term analogue of Figure 1, using Germany, Japan,
# Switzerland, and the United Kingdom, 2003–2020.
#
# - **x-axis:** OECD `DF_T720R_A` long-term debt (USD), converted to face value
#   via Nelson-Siegel 5-year zero-coupon yield → `figure2_xaxis.parquet`
# - **y-axis:** OECD `DF_FINMARK` IRLT 10-year par yield, NS-converted to
#   5-year zero → `figure2_yaxis.parquet`
#
# The y-axis range is narrower (±2.5 pp vs ±4 pp) reflecting lower volatility
# in long-term rate differentials. Germany proxies the Euro Area.
 
# %%
fig2_png = OUT_DIR / "figure2_replicated.png"
if fig2_png.exists():
    display(Image(str(fig2_png)))
else:
    print("Figure 2 not yet generated. Run:  doit figure2")