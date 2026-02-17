"""Pull selected World Bank Data360 (WDI) indicators.

This module is intentionally a *framework* / scaffold:
- It wires up the Data360 REST endpoints using the published OpenAPI spec.
- It provides pagination, basic normalization to a tidy pandas DataFrame,
  and a country list consistent with the MSCI/OECD pull scripts.
- The exact indicator IDs and disaggregation filters should be validated
  via `/data360/indicators` or `/data360/searchv2`.

OpenAPI spec (upstream):
https://raw.githubusercontent.com/worldbank/open-api-specs/refs/heads/main/Data360%20Open_API.json

Primary data endpoint:
GET https://data360api.worldbank.org/data360/data

Notes
-----
Data360 returns up to 1000 rows per call; use `skip` to paginate.

The query parameters for `/data360/data` are *mostly* simple query-string
fields (e.g., DATABASE_ID, INDICATOR, REF_AREA, TIME_PERIOD, etc.).
For many indicators, leaving disaggregation dimensions as defaults (e.g. _T/_Z)
works, but this needs confirmation per-indicator.

Target series (requested):
- GDP
- GDP per capita at PPP (current international $)
- CPI
- PPP conversion factor for GDP (LCU per international $)
- Market capitalization of listed domestic companies (only for countries
  missing from OECD)

"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable, Iterator

import pandas as pd
import requests

from settings import config


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

_data_dir_value = config("DATA_DIR")
DATA_DIR = _data_dir_value if isinstance(_data_dir_value, Path) else Path(str(_data_dir_value))

DATA360_OPENAPI_SPEC_URL = str(
    config(
        "DATA360_OPENAPI_SPEC_URL",
        default=(
            "https://raw.githubusercontent.com/worldbank/open-api-specs/refs/heads/main/"
            "Data360%20Open_API.json"
        ),
        cast=str,
    )
)

DATA360_BASE_URL = str(
    config("DATA360_BASE_URL", default="https://data360api.worldbank.org", cast=str)
).rstrip("/")

WB_DATA360_OUTPUT_FILE = str(
    config(
        "WB_DATA360_OUTPUT_FILE",
        default="wb_data360_wdi_selected.parquet",
        cast=str,
    )
)

WB_START_PERIOD = str(config("WB_START_PERIOD", default="2003", cast=str))
WB_END_PERIOD = str(config("WB_END_PERIOD", default="2020", cast=str))

WB_TIMEOUT_SECONDS = int(str(config("WB_TIMEOUT_SECONDS", default="120", cast=str)))
WB_PAGE_SIZE = int(str(config("WB_PAGE_SIZE", default="1000", cast=str)))

WB_HTTP_RETRIES = int(str(config("WB_HTTP_RETRIES", default="5", cast=str)))
WB_HTTP_RETRY_BACKOFF_SECONDS = float(
    str(config("WB_HTTP_RETRY_BACKOFF_SECONDS", default="1.5", cast=str))
)


# --------------------------------------------------------------------------------------
# Target countries (match MSCI TARGET_COUNTRIES) + ISO codes
# --------------------------------------------------------------------------------------

TARGET_COUNTRIES = [
    "CANADA",
    "UNITED STATES",
    "AUSTRIA",
    "BELGIUM",
    "DENMARK",
    "FINLAND",
    "FRANCE",
    "GERMANY",
    "ISRAEL",
    "ITALY",
    "NETHERLANDS",
    "NORWAY",
    "PORTUGAL",
    "SPAIN",
    "SWEDEN",
    "SWITZERLAND",
    "UNITED KINGDOM",
    "AUSTRALIA",
    "HONG KONG",
    "JAPAN",
    "NEW ZEALAND",
    "SINGAPORE",
    "BRAZIL",
    "CHINA",
    "COLOMBIA",
    "CZECH REPUBLIC",
    "GREECE",
    "HUNGARY",
    "INDIA",
    "MALAYSIA",
    "MEXICO",
    "PHILIPPINES",
    "POLAND",
    "RUSSIA",
    "SOUTH AFRICA",
    "SOUTH KOREA",
    "THAILAND",
]

# Minimal mapping to ISO-3166 alpha-3 (commonly used by WDI / WB datasets).
# If Data360 uses a different REF_AREA scheme for a given database, adjust here.
COUNTRY_NAME_TO_ISO3: dict[str, str] = {
    "CANADA": "CAN",
    "UNITED STATES": "USA",
    "AUSTRIA": "AUT",
    "BELGIUM": "BEL",
    "DENMARK": "DNK",
    "FINLAND": "FIN",
    "FRANCE": "FRA",
    "GERMANY": "DEU",
    "ISRAEL": "ISR",
    "ITALY": "ITA",
    "NETHERLANDS": "NLD",
    "NORWAY": "NOR",
    "PORTUGAL": "PRT",
    "SPAIN": "ESP",
    "SWEDEN": "SWE",
    "SWITZERLAND": "CHE",
    "UNITED KINGDOM": "GBR",
    "AUSTRALIA": "AUS",
    "HONG KONG": "HKG",
    "JAPAN": "JPN",
    "NEW ZEALAND": "NZL",
    "SINGAPORE": "SGP",
    "BRAZIL": "BRA",
    "CHINA": "CHN",
    "COLOMBIA": "COL",
    "CZECH REPUBLIC": "CZE",
    "GREECE": "GRC",
    "HUNGARY": "HUN",
    "INDIA": "IND",
    "MALAYSIA": "MYS",
    "MEXICO": "MEX",
    "PHILIPPINES": "PHL",
    "POLAND": "POL",
    "RUSSIA": "RUS",
    "SOUTH AFRICA": "ZAF",
    "SOUTH KOREA": "KOR",
    "THAILAND": "THA",
}


def target_ref_areas_iso3(country_names: Iterable[str] = TARGET_COUNTRIES) -> list[str]:
    missing = [c for c in country_names if c not in COUNTRY_NAME_TO_ISO3]
    if missing:
        raise KeyError(f"Missing ISO3 mapping for: {missing}")
    return [COUNTRY_NAME_TO_ISO3[c] for c in country_names]


# --------------------------------------------------------------------------------------
# Indicator placeholders (WDI via Data360)
# --------------------------------------------------------------------------------------

# These IDs follow the common WDI code convention as exposed by Data360.
# Verified locally via `trying_WB.py` using `/data360/searchv2` within `WB_WDI`.
WDI_INDICATORS: dict[str, str] = {
    # GDP, current US$: NY.GDP.MKTP.CD
    "gdp_current_usd": "WB_WDI_NY_GDP_MKTP_CD",
    # GDP, PPP (current international $): NY.GDP.MKTP.PP.CD
    "gdp_ppp_current_intl_usd": "WB_WDI_NY_GDP_MKTP_PP_CD",
    # GDP per capita, PPP (current international $): NY.GDP.PCAP.PP.CD
    "gdp_per_capita_ppp_current_intl_usd": "WB_WDI_NY_GDP_PCAP_PP_CD",
    # CPI: Consumer price index (2010 = 100): FP.CPI.TOTL
    "cpi": "WB_WDI_FP_CPI_TOTL",
    # PPP conversion factor, GDP (LCU per international $): PA.NUS.PPP
    "ppp_conversion_factor_gdp_lcu_per_intl_usd": "WB_WDI_PA_NUS_PPP",
}

WDI_MARKET_CAP_INDICATOR = "WB_WDI_CM_MKT_LCAP_CD"  # verified via trying_WB/searchv2

# Countries missing OECD T720 equity outstanding in this workspace's extract.
#
# Definition used (from `_data/oecd_t720.parquet`):
# - `financial_instrument == 'F5'` (Equity and investment fund shares)
# - `transaction == 'LE'` (Levels / outstanding)
# - `accounting_entry == 'L'` (Liabilities)
#
# For these countries, we pull the WB WDI market cap series as a fallback.
OECD_T720_EQUITY_OUTSTANDING_MISSING_COUNTRIES = [
    "AUSTRALIA",
    "HONG KONG",
    "NEW ZEALAND",
    "SINGAPORE",
    "CHINA",
    "INDIA",
    "MALAYSIA",
    "PHILIPPINES",
    "RUSSIA",
    "SOUTH AFRICA",
    "THAILAND",
]


def oecd_missing_equity_outstanding_ref_areas_iso3() -> list[str]:
    return target_ref_areas_iso3(OECD_T720_EQUITY_OUTSTANDING_MISSING_COUNTRIES)


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Data360Client:
    base_url: str = DATA360_BASE_URL
    timeout_seconds: int = WB_TIMEOUT_SECONDS
    page_size: int = WB_PAGE_SIZE

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(int(WB_HTTP_RETRIES)):
            try:
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= int(WB_HTTP_RETRIES) - 1:
                    raise
                time.sleep(float(WB_HTTP_RETRY_BACKOFF_SECONDS) * (attempt + 1))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Data360 GET failed")

    def iter_data(self, params: dict[str, Any], *, paginate: bool = True) -> Iterator[dict[str, Any]]:
        """Yield records from `/data360/data`.

        Parameters
        ----------
        params:
            Query-string params for the Data360 endpoint.
        paginate:
            If true, uses `skip` to fetch all pages.
        """

        skip = int(params.get("skip", 0) or 0)
        while True:
            page_params = {**params, "skip": skip}
            payload = self._get("/data360/data", page_params)

            records = payload.get("value") or []
            for rec in records:
                yield rec

            if (not paginate) or (not records):
                return

            total = payload.get("count")
            if total is None:
                # Defensive: if no total count, stop after one page.
                return

            skip += len(records)
            if skip >= int(total):
                return

    def get_data_df(self, params: dict[str, Any], *, paginate: bool = True) -> pd.DataFrame:
        records = list(self.iter_data(params, paginate=paginate))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame.from_records(records)
        if "OBS_VALUE" in df.columns:
            df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        return df


# --------------------------------------------------------------------------------------
# OpenAPI spec helpers (optional; useful for discovery/debug)
# --------------------------------------------------------------------------------------


def download_openapi_spec(
    *,
    url: str = DATA360_OPENAPI_SPEC_URL,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Download (and optionally cache) the Data360 OpenAPI spec JSON."""

    if cache_path is None:
        cache_path = Path(DATA_DIR) / "data360_openapi.json"

    if cache_path.exists() and (not refresh):
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    spec = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec


def spec_base_url(spec: dict[str, Any], default: str = DATA360_BASE_URL) -> str:
    servers = spec.get("servers") or []
    if not servers:
        return default
    url = servers[0].get("url")
    if not url:
        return default
    return str(url).rstrip("/")


# --------------------------------------------------------------------------------------
# Indicator discovery helpers
# --------------------------------------------------------------------------------------


def search_indicators(
    search: str,
    *,
    database_id: str = "WB_WDI",
    top: int = 25,
    skip: int = 0,
    base_url: str = DATA360_BASE_URL,
    timeout_seconds: int = WB_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Search Data360 metadata to help find indicator IDs.

    Uses POST `/data360/searchv2` (vector/keyword search).

    Returns a DataFrame containing at least:
    - series_description/idno
    - series_description/name
    - series_description/database_id

    This is intended for manual discovery during setup.
    """

    url = f"{base_url.rstrip('/')}/data360/searchv2"
    body = {
        "count": True,
        "select": "series_description/idno,series_description/name,series_description/database_id,type",
        "search": search,
        "top": int(top),
        "skip": int(skip),
        "filter": f"series_description/database_id eq '{database_id}' and type eq 'indicator'",
    }

    response = requests.post(url, json=body, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()

    values = payload.get("value") or payload.get("values") or []
    return pd.DataFrame.from_records(values)


# --------------------------------------------------------------------------------------
# Pull routines (scaffold)
# --------------------------------------------------------------------------------------


def _default_wdi_params(
    *,
    database_id: str,
    indicator_id: str,
    ref_area: str,
    start_period: str,
    end_period: str,
    freq: str = "A",
) -> dict[str, Any]:
    """Baseline parameters for most annual WDI series."""

    return {
        "format": "json",
        "DATABASE_ID": database_id,
        "INDICATOR": indicator_id,
        "REF_AREA": ref_area,
        "FREQ": freq,
        "timePeriodFrom": str(start_period),
        "timePeriodTo": str(end_period),
        # Defaults commonly used by many indicators. These may need adjustment.
        "SEX": "_T",
        "AGE": "_T",
        "URBANISATION": "_T",
        "COMP_BREAKDOWN_1": "_Z",
        "COMP_BREAKDOWN_2": "_Z",
        "COMP_BREAKDOWN_3": "_Z",
        "skip": 0,
    }


def pull_wdi_indicator(
    indicator_id: str,
    *,
    countries_ref_area: Iterable[str] | None = None,
    start_period: str = WB_START_PERIOD,
    end_period: str = WB_END_PERIOD,
    database_id: str = "WB_WDI",
    client: Data360Client | None = None,
    paginate: bool = True,
) -> pd.DataFrame:
    """Pull a single indicator for a set of countries.

    Implementation choice: one API call per country to avoid guessing whether
    REF_AREA supports multi-values.
    """

    if client is None:
        client = Data360Client()

    if countries_ref_area is None:
        countries_ref_area = target_ref_areas_iso3()

    frames: list[pd.DataFrame] = []
    for ref_area in countries_ref_area:
        params = _default_wdi_params(
            database_id=database_id,
            indicator_id=indicator_id,
            ref_area=str(ref_area),
            start_period=start_period,
            end_period=end_period,
        )
        df = client.get_data_df(params, paginate=paginate)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["indicator_id"] = indicator_id
    out["database_id"] = database_id
    return out


def pull_wdi_bundle(
    *,
    indicators: dict[str, str] | None = None,
    include_market_cap: bool = True,
    market_cap_ref_areas: Iterable[str] | None = None,
    countries_ref_area: Iterable[str] | None = None,
    start_period: str = WB_START_PERIOD,
    end_period: str = WB_END_PERIOD,
    client: Data360Client | None = None,
) -> pd.DataFrame:
    """Pull the requested set of WDI indicators.

    Parameters
    ----------
    indicators:
        Mapping {metric_name: indicator_id}.
    include_market_cap:
        If true, additionally pulls the market cap series.
    market_cap_ref_areas:
                Country list for the market cap pull only.

                - If provided: used as-is for the market cap series.
                - If None (default): uses the OECD-missing list derived from this workspace's
                    T720 extract (see `OECD_T720_EQUITY_OUTSTANDING_MISSING_COUNTRIES`).

                Note: This does *not* affect the other indicators in `indicators`, which are
                always pulled for `countries_ref_area` (default: all 37 target countries).
    """

    if client is None:
        client = Data360Client()

    if indicators is None:
        indicators = dict(WDI_INDICATORS)

    if countries_ref_area is None:
        countries_ref_area = target_ref_areas_iso3()

    frames: list[pd.DataFrame] = []
    for metric_name, indicator_id in indicators.items():
        df = pull_wdi_indicator(
            indicator_id,
            countries_ref_area=countries_ref_area,
            start_period=start_period,
            end_period=end_period,
            client=client,
        )
        if df.empty:
            continue
        df["metric"] = metric_name
        frames.append(df)

    if include_market_cap:
        if market_cap_ref_areas is None:
            market_cap_countries = oecd_missing_equity_outstanding_ref_areas_iso3()
        else:
            market_cap_countries = list(market_cap_ref_areas)
        df_mc = pull_wdi_indicator(
            WDI_MARKET_CAP_INDICATOR,
            countries_ref_area=market_cap_countries,
            start_period=start_period,
            end_period=end_period,
            client=client,
        )
        if not df_mc.empty:
            df_mc["metric"] = "market_cap_listed_domestic_companies_current_usd"
            frames.append(df_mc)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Standardize a few key columns when present.
    rename = {
        "REF_AREA": "ref_area",
        "TIME_PERIOD": "time_period",
        "OBS_VALUE": "value",
        "INDICATOR": "indicator",
        "DATABASE_ID": "database_id_api",
        "FREQ": "freq",
    }
    existing = {k: v for k, v in rename.items() if k in out.columns}
    if existing:
        out = out.rename(columns=existing)

    return out


if __name__ == "__main__":
    df = pull_wdi_bundle()

    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / WB_DATA360_OUTPUT_FILE

    df.to_parquet(output_path, index=False)
    print(f"Wrote: {output_path}")
