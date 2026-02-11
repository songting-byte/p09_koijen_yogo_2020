"""Pull BIS debt securities data for domestic and international markets.

Domestic debt securities come from WS_NA_SEC_DSS (NA debt securities statistics).
International debt securities come from WS_DEBT_SEC2_PUB (BIS-compiled IDS).
"""
import pandas as pd
from pathlib import Path

COUNTRY_NAME_TO_CODE = {
    "Australia": "AU",
    "Hong Kong": "HK",
    "Singapore": "SG",
    "New Zealand": "NZ",
    "China": "CN",
    "India": "IN",
    "Malaysia": "MY",
    "Philippines": "PH",
    "Russia": "RU",
    "South Africa": "ZA",
    "Israel": "IL",
    "Brazil": "BR",
}


# Querying the data

urls = ["https://stats.bis.org/api/v2/data/dataflow/BIS/WS_NA_SEC_DSS/1.0/...XW.S11+S12+S13...L.LE.F3.L+LS+S+TT..USD.X1+XDC+_T?startPeriod=2003-12-25&endPeriod=2020-12-31&format=csv"]

df = pd.concat([pd.read_csv(url) for url in urls])



# Processing the data 

TARGET_COUNTRY_CODES = set(COUNTRY_NAME_TO_CODE.values())

print("Filtering to country codes:", TARGET_COUNTRY_CODES)

df = df[df["REF_AREA"].isin(TARGET_COUNTRY_CODES)].copy()

print("Rows after country filter:", len(df))
print(df["REF_AREA"].value_counts())


COLUMNS_TO_DROP = [
    'CUST_BREAKDOWN', 'COMMENT_DSET', 'REF_PERIOD_DETAIL', 'REPYEARSTART',
    'REPYEAREND', 'TIME_FORMAT', 'TIME_PER_COLLECT', 'CUST_BREAKDOWN_LB',
    'REF_YEAR_PRICE', 'DECIMALS', 'TABLE_IDENTIFIER', 'TITLE',
    'TITLE_COMPL', 'UNIT_MULT', 'LAST_UPDATE', 'COMPILING_ORG',
    'COLL_PERIOD', 'COMMENT_TS', 'GFS_ECOFUNC', 'GFS_TAXCAT', 'DATA_COMP',
    'CURRENCY', 'DISS_ORG', 'OBS_PRE_BREAK',
    'OBS_STATUS', 'CONF_STATUS', 'COMMENT_OBS', 'EMBARGO_DATE',
    'OBS_EDP_WBB'
]

existing_cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]

print(f"Dropping {len(existing_cols_to_drop)} columns")

df = df.drop(columns=existing_cols_to_drop)

print("Remaining columns:")
print(df.columns.tolist())


# Paths 

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Path for cleaned data
CLEAN_PATH = OUTPUT_DIR / "bis_debt_securities_cleaned.parquet"

# Save cleaned df to parquet

df.to_parquet(CLEAN_PATH, index=False)

print("Cleaned data saved to:", CLEAN_PATH)
