"""Pull BIS debt securities data for domestic and international markets.

Domestic debt securities come from WS_NA_SEC_DSS (NA debt securities statistics).
International debt securities come from WS_DEBT_SEC2_PUB (BIS-compiled IDS).
"""

from datetime import date, datetime
from pathlib import Path
import io
import json
import random
import time

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")

_CODELIST_CACHE = {}

BIS_DATA_BASE_URL = config(
    "BIS_DATA_BASE_URL",
    default=(
        "https://stats.bis.org/api/v2/data/dataflow/BIS/"
        "{dataflow_id}/{version}/{key}"
    ),
    cast=str,
)
BIS_STRUCTURE_BASE_URL = config(
    "BIS_STRUCTURE_BASE_URL",
    default=(
        "https://stats.bis.org/api/v2/structure/datastructure/BIS/"
        "{dsd_id}/{version}"
    ),
    cast=str,
)
BIS_CODELIST_BASE_URL = config(
    "BIS_CODELIST_BASE_URL",
    default=(
        "https://stats.bis.org/api/v2/structure/codelist/"
        "{agency}/{codelist_id}/{version}"
    ),
    cast=str,
)

DOMESTIC_DATAFLOW_ID = config("DOMESTIC_DATAFLOW_ID", default="WS_NA_SEC_DSS", cast=str)
DOMESTIC_DATAFLOW_VERSION = config("DOMESTIC_DATAFLOW_VERSION", default="1.0", cast=str)
DOMESTIC_DSD_ID = config("DOMESTIC_DSD_ID", default="NA_SEC", cast=str)
DOMESTIC_DSD_VERSION = config("DOMESTIC_DSD_VERSION", default="1.0", cast=str)

INTERNATIONAL_DATAFLOW_ID = config(
    "INTERNATIONAL_DATAFLOW_ID", default="WS_DEBT_SEC2_PUB", cast=str
)
INTERNATIONAL_DATAFLOW_VERSION = config(
    "INTERNATIONAL_DATAFLOW_VERSION", default="1.0", cast=str
)
INTERNATIONAL_DSD_ID = config(
    "INTERNATIONAL_DSD_ID", default="BIS_DEBT_SEC2", cast=str
)
INTERNATIONAL_DSD_VERSION = config(
    "INTERNATIONAL_DSD_VERSION", default="1.0", cast=str
)

BIS_START_PERIOD = config("BIS_START_PERIOD", default="2003", cast=str)
BIS_END_PERIOD = config("BIS_END_PERIOD", default="2020", cast=str)
BIS_OUTPUT_FILE = config(
    "BIS_OUTPUT_FILE", default="bis_debt_securities.csv", cast=str
)

BIS_REQUEST_SLEEP_MIN_SECONDS = config(
    "BIS_REQUEST_SLEEP_MIN_SECONDS", default="2", cast=float
)
BIS_REQUEST_SLEEP_MAX_SECONDS = config(
    "BIS_REQUEST_SLEEP_MAX_SECONDS", default="4", cast=float
)
BIS_MAX_RETRIES = config("BIS_MAX_RETRIES", default="5", cast=int)
BIS_REF_AREA_BATCH_SIZE = config("BIS_REF_AREA_BATCH_SIZE", default="3", cast=int)

DOMESTIC_FREQUENCY = "A"
INTERNATIONAL_FREQUENCY = "Q"
DOMESTIC_ISSUER_SECTORS = ["S13"]
DOMESTIC_MATURITIES = ["S", "L"]
DOMESTIC_INSTRUMENT = "F3"
DOMESTIC_ACCOUNTING_ENTRY = "L"
DOMESTIC_STOCK_POSITION = "LE"
DOMESTIC_ADJUSTMENT = "N"
DOMESTIC_COUNTERPART_AREA = "XW"
DOMESTIC_COUNTERPART_SECTOR = "S1"
DOMESTIC_CONSOLIDATION = "N"
DOMESTIC_EXPENDITURE = "_Z"
DOMESTIC_UNIT_MEASURE = "USD"
DOMESTIC_CURRENCY_DENOM = "XDC"
DOMESTIC_VALUATION = "N"
DOMESTIC_PRICES = "V"
DOMESTIC_TRANSFORMATION = "N"
DOMESTIC_CUST_BREAKDOWN = "_T"

INTERNATIONAL_MATURITIES = ["C", "K"]
INTERNATIONAL_MEASURE = "I"
INTERNATIONAL_MARKET = "C"
INTERNATIONAL_ISSUER_ALL = "1"
INTERNATIONAL_ISSUER_NAT_TOTAL = "3P"
INTERNATIONAL_ISSUE_TYPE = "A"
INTERNATIONAL_ISSUE_CUR_GROUP = "A"
INTERNATIONAL_ISSUE_CUR = "TO1"
INTERNATIONAL_ISSUE_RE_MAT = "A"
INTERNATIONAL_ISSUE_RATE = "A"
INTERNATIONAL_ISSUE_RISK = "A"
INTERNATIONAL_ISSUE_COL = "A"

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

TARGET_COUNTRIES = [
    "Australia",
    "Hong Kong",
    "Singapore",
    "New Zealand",
    "China",
    "India",
    "Malaysia",
    "Philippines",
    "Russia",
    "South Africa",
    "Israel",
    "Brazil",
]


def _format_period(value):
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y")
    return str(value)


def _build_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "bis-sdmx-pull/1.0"})
    return session


def _sleep_random():
    time.sleep(random.uniform(BIS_REQUEST_SLEEP_MIN_SECONDS, BIS_REQUEST_SLEEP_MAX_SECONDS))


def _get_text_with_retry(session, url, params=None, max_retries=None):
    retries = max_retries if max_retries is not None else BIS_MAX_RETRIES
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=(10, 120))
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            if attempt >= retries - 1:
                raise
            _sleep_random()


def _parse_codelist_urn(urn):
    if not urn or not isinstance(urn, str):
        return None
    if "Codelist=" not in urn:
        return None
    payload = urn.split("Codelist=")[-1]
    left, right = payload.split("(", 1)
    agency, codelist_id = left.split(":", 1)
    version = right.split(")", 1)[0]
    return agency, codelist_id, version


def _fetch_datastructure(session, dsd_id, version):
    url = BIS_STRUCTURE_BASE_URL.format(dsd_id=dsd_id, version=version)
    payload = json.loads(_get_text_with_retry(session, url))
    data_structures = payload.get("data", {}).get("dataStructures", [])
    dsd = next((d for d in data_structures if d.get("id") == dsd_id), None)
    if not dsd:
        raise ValueError("BIS datastructure not found in response")
    dims = dsd.get("dataStructureComponents", {}).get("dimensionList", {}).get(
        "dimensions", []
    )
    return [d.get("id") for d in dims], dims


def _fetch_codelist(session, agency, codelist_id, version):
    cache_key = (agency, codelist_id, version)
    if cache_key in _CODELIST_CACHE:
        return _CODELIST_CACHE[cache_key]
    url = BIS_CODELIST_BASE_URL.format(
        agency=agency, codelist_id=codelist_id, version=version
    )
    payload = json.loads(_get_text_with_retry(session, url))
    codelists = payload.get("data", {}).get("codelists", [])
    cl = next((c for c in codelists if c.get("id") == codelist_id), None)
    if not cl:
        raise ValueError(f"Codelist {agency}:{codelist_id}({version}) not found")
    _CODELIST_CACHE[cache_key] = cl
    return cl


def _resolve_country_codes(session, names):
    cl = _fetch_codelist(session, "BIS", "CL_BIS_IF_REF_AREA", "1.0")
    codes = {c.get("id"): c.get("name") for c in cl.get("codes", [])}

    resolved = {}
    for name in names:
        if name in codes:
            resolved[name] = name
            continue
        if name in COUNTRY_NAME_TO_CODE:
            resolved[name] = COUNTRY_NAME_TO_CODE[name]
            continue
        matches = [k for k, v in codes.items() if v and name.lower() in v.lower()]
        if len(matches) == 1:
            resolved[name] = matches[0]
        else:
            raise ValueError(f"Unable to resolve country code for '{name}'")
    return resolved


def _select_default_code(codelist):
    if not codelist:
        return ""
    codes = codelist.get("codes", [])
    if not codes:
        return ""

    for code in codes:
        name = (code.get("name") or "").lower()
        if "total" in name or "all" in name:
            return code.get("id")

    for fallback in ["_Z", "_X"]:
        if any(code.get("id") == fallback for code in codes):
            return fallback

    return codes[0].get("id", "")


def _build_empty_defaults(dim_order):
    return {dim: "" for dim in dim_order}


def _build_key(dim_order, overrides, defaults):
    parts = []
    for dim in dim_order:
        value = overrides.get(dim)
        if value is None:
            value = defaults.get(dim, "")
        if isinstance(value, (list, tuple, set)):
            parts.append("+".join(value))
        else:
            parts.append(str(value))
    return ".".join(parts)


def _fetch_csv(session, dataflow_id, version, key, start_period, end_period):
    data_url = BIS_DATA_BASE_URL.format(
        dataflow_id=dataflow_id, version=version, key=key
    )
    params = {
        "startPeriod": _format_period(start_period),
        "endPeriod": _format_period(end_period),
        "format": "csv",
    }
    payload = _get_text_with_retry(session, data_url, params=params)
    if payload.lstrip().startswith("<") or payload.lstrip().startswith("{"):
        raise ValueError("BIS data endpoint returned non-CSV response")
    return pd.read_csv(io.StringIO(payload))


def _normalize_columns(df, mappings):
    df = df.rename(columns=mappings)
    if "TIME_PERIOD" in df.columns:
        df = df.rename(columns={"TIME_PERIOD": "year"})
    if "OBS_VALUE" in df.columns:
        df = df.rename(columns={"OBS_VALUE": "value"})
    if "value" not in df.columns and "obs_value" in df.columns:
        df = df.rename(columns={"obs_value": "value"})
    return df


def _pull_domestic(session, start_period, end_period, country_codes):
    dim_order, _ = _fetch_datastructure(session, DOMESTIC_DSD_ID, DOMESTIC_DSD_VERSION)
    defaults = _build_empty_defaults(dim_order)

    ref_areas = list(country_codes.values())
    if BIS_REF_AREA_BATCH_SIZE > 0:
        batches = [
            ref_areas[i : i + BIS_REF_AREA_BATCH_SIZE]
            for i in range(0, len(ref_areas), BIS_REF_AREA_BATCH_SIZE)
        ]
    else:
        batches = [ref_areas]

    frames = []
    for batch in batches:
        overrides = {
            "FREQ": DOMESTIC_FREQUENCY,
            "REF_AREA": batch,
            "REF_SECTOR": DOMESTIC_ISSUER_SECTORS,
            "ADJUSTMENT": DOMESTIC_ADJUSTMENT,
            "COUNTERPART_AREA": DOMESTIC_COUNTERPART_AREA,
            "COUNTERPART_SECTOR": DOMESTIC_COUNTERPART_SECTOR,
            "CONSOLIDATION": DOMESTIC_CONSOLIDATION,
            "MATURITY": DOMESTIC_MATURITIES,
            "INSTR_ASSET": DOMESTIC_INSTRUMENT,
            "ACCOUNTING_ENTRY": DOMESTIC_ACCOUNTING_ENTRY,
            "STO": DOMESTIC_STOCK_POSITION,
            "EXPENDITURE": DOMESTIC_EXPENDITURE,
            "UNIT_MEASURE": DOMESTIC_UNIT_MEASURE,
            "CURRENCY_DENOM": DOMESTIC_CURRENCY_DENOM,
            "VALUATION": DOMESTIC_VALUATION,
            "PRICES": DOMESTIC_PRICES,
            "TRANSFORMATION": DOMESTIC_TRANSFORMATION,
            "CUST_BREAKDOWN": DOMESTIC_CUST_BREAKDOWN,
        }
        key = _build_key(dim_order, overrides, defaults)
        df = _fetch_csv(
            session,
            DOMESTIC_DATAFLOW_ID,
            DOMESTIC_DATAFLOW_VERSION,
            key,
            start_period,
            end_period,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    normalized = _normalize_columns(
        raw,
        {
            "REF_AREA": "country",
            "REF_SECTOR": "issuer_sector",
            "MATURITY": "maturity",
        },
    )
    normalized["market"] = "domestic"
    normalized["source"] = "WS_NA_SEC_DSS"
    return normalized


def _pull_international(session, start_period, end_period, country_codes):
    dim_order, _ = _fetch_datastructure(
        session, INTERNATIONAL_DSD_ID, INTERNATIONAL_DSD_VERSION
    )
    defaults = _build_empty_defaults(dim_order)

    ref_areas = list(country_codes.values())
    if BIS_REF_AREA_BATCH_SIZE > 0:
        batches = [
            ref_areas[i : i + BIS_REF_AREA_BATCH_SIZE]
            for i in range(0, len(ref_areas), BIS_REF_AREA_BATCH_SIZE)
        ]
    else:
        batches = [ref_areas]

    frames = []
    for batch in batches:
        overrides = {
            "FREQ": INTERNATIONAL_FREQUENCY,
            "ISSUER_RES": batch,
            "ISSUER_NAT": INTERNATIONAL_ISSUER_NAT_TOTAL,
            "ISSUER_BUS_IMM": INTERNATIONAL_ISSUER_ALL,
            "ISSUER_BUS_ULT": INTERNATIONAL_ISSUER_ALL,
            "MARKET": INTERNATIONAL_MARKET,
            "ISSUE_OR_MAT": INTERNATIONAL_MATURITIES,
            "MEASURE": INTERNATIONAL_MEASURE,
            "ISSUE_TYPE": INTERNATIONAL_ISSUE_TYPE,
            "ISSUE_CUR_GROUP": INTERNATIONAL_ISSUE_CUR_GROUP,
            "ISSUE_CUR": INTERNATIONAL_ISSUE_CUR,
            "ISSUE_RE_MAT": INTERNATIONAL_ISSUE_RE_MAT,
            "ISSUE_RATE": INTERNATIONAL_ISSUE_RATE,
            "ISSUE_RISK": INTERNATIONAL_ISSUE_RISK,
            "ISSUE_COL": INTERNATIONAL_ISSUE_COL,
        }
        key = _build_key(dim_order, overrides, defaults)
        df = _fetch_csv(
            session,
            INTERNATIONAL_DATAFLOW_ID,
            INTERNATIONAL_DATAFLOW_VERSION,
            key,
            start_period,
            end_period,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    normalized = _normalize_columns(
        raw,
        {
            "ISSUER_RES": "country",
            "ISSUE_OR_MAT": "maturity",
        },
    )
    normalized["issuer_sector"] = "ALL"
    normalized["market"] = "international"
    normalized["source"] = "WS_DEBT_SEC2_PUB"
    return normalized


def pull_bis_debt_securities(
    start_period=BIS_START_PERIOD,
    end_period=BIS_END_PERIOD,
    countries=None,
):
    session = _build_session()

    if countries is not None:
        country_names = countries
        country_codes = _resolve_country_codes(session, country_names)
    else:
        country_codes = _resolve_country_codes(session, TARGET_COUNTRIES)

    domestic = _pull_domestic(session, start_period, end_period, country_codes)
    international = _pull_international(session, start_period, end_period, country_codes)

    frames = [df for df in [domestic, international] if not df.empty]
    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[[
        "country",
        "year",
        "issuer_sector",
        "market",
        "maturity",
        "value",
        "source",
    ]]
    return panel


def main():
    df = pull_bis_debt_securities()
    output_path = Path(DATA_DIR) / BIS_OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
