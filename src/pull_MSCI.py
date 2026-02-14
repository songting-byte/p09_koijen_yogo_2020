"""Pull only confirmed required MSCI-related datasets from WRDS Datastream.

Target metrics (from validated dictionary mappings):
1) 3M interbank rate              -> trdstrm.ecodata.series_value + trdstrm.ecoinfo
2) 10Y government yield            -> trdstrm.ecodata.series_value + trdstrm.ecoinfo
3) market equity outstanding       -> trdstrm.ds2numshares.numshrs
4) market-to-book equity           -> trdstrm.ds2indexaddldata.datatypevalue (BP/MSPB)
5) monthly equity return (USD proxy) -> trdstrm.ds2primqtri.ri + trdstrm.ds2scdqtri.ri
"""

from pathlib import Path
import time
from typing import Any

import pandas as pd

from settings import config

try:
    import wrds
except ImportError as exc:  # pragma: no cover
    raise ImportError("The 'wrds' package is required. Install with: pip install wrds") from exc


_data_dir_value = config("DATA_DIR")
DATA_DIR = _data_dir_value if isinstance(_data_dir_value, Path) else Path(str(_data_dir_value))
WRDS_USERNAME = config("WRDS_USERNAME", default=None, cast=str)
WRDS_PASSWORD = config("WRDS_PASSWORD", default=None, cast=str)

OUTPUT_COLUMNS = [
    "metric",
    "source_table",
    "entity_id",
    "obs_date",
    "value",
    "series_code",
    "description",
    "unitcode",
    "freqcode",
    "currcode",
    "value_column",
    "key_columns",
]


def _config_int(name: str, default: int) -> int:
    value: Any = config(name, default=str(default), cast=str)
    return int(str(value))


WRDS_CONNECT_RETRIES = _config_int("WRDS_CONNECT_RETRIES", 8)
WRDS_CONNECT_RETRY_SECONDS = _config_int("WRDS_CONNECT_RETRY_SECONDS", 20)

MSCI_START_PERIOD = "2003"
MSCI_END_PERIOD = "2020"
MSCI_NEEDED_OUTPUT_FILE = str(
    config("MSCI_NEEDED_OUTPUT_FILE", default="data_msci_datastream.parquet", cast=str)
)
MSCI_INCLUDE_SECONDARY_RETURNS = config(
    "MSCI_INCLUDE_SECONDARY_RETURNS", default=False, cast=bool
)
MSCI_MARKET_EQUITY_LEVEL = str(
    config("MSCI_MARKET_EQUITY_LEVEL", default="country", cast=str)
).strip().lower()
MSCI_MARKET_TO_BOOK_LEVEL = str(
    config("MSCI_MARKET_TO_BOOK_LEVEL", default="country", cast=str)
).strip().lower()
MSCI_EQUITY_RETURN_LEVEL = str(
    config("MSCI_EQUITY_RETURN_LEVEL", default="country", cast=str)
).strip().lower()
MSCI_SORT_OUTPUT = config("MSCI_SORT_OUTPUT", default=False, cast=bool)


def _config_csv_upper(name: str, default: str) -> tuple[str, ...]:
    raw = str(config(name, default=default, cast=str))
    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return tuple(values) if values else tuple(item.strip() for item in default.split(",") if item.strip())


MSCI_MARKET_TO_BOOK_SERIES = _config_csv_upper("MSCI_MARKET_TO_BOOK_SERIES", "BP")
if not MSCI_MARKET_TO_BOOK_SERIES:
    MSCI_MARKET_TO_BOOK_SERIES = ("BP",)

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

TARGET_REGION_CODES = [
    "CA", "US", "AT", "BE", "DK", "FI", "FR", "DE", "IL", "IT", "NL", "NO", "PT", "ES", "SE", "CH", "GB",
    "AU", "HK", "JP", "NZ", "SG", "BR", "CN", "CO", "CZ", "GR", "HU", "IN", "MY", "MX", "PH", "PL", "RU", "ZA", "KR", "TH",
]


def _period_bounds(start_period: str, end_period: str) -> tuple[str, str]:
    return f"{start_period}-01-01", f"{end_period}-12-31"


def _region_period_params(start_period: str, end_period: str) -> dict[str, Any]:
    start, end = _period_bounds(start_period, end_period)
    return {
        "start": start,
        "end": end,
        "region_codes": tuple(TARGET_REGION_CODES),
    }


def _country_period_params(start_period: str, end_period: str) -> dict[str, Any]:
    start, end = _period_bounds(start_period, end_period)
    return {
        "start": start,
        "end": end,
        "country_names": tuple(TARGET_COUNTRIES),
    }


def _connect_wrds():
    if not WRDS_USERNAME:
        raise ValueError("WRDS_USERNAME is required in .env")

    last_error = None
    for attempt in range(WRDS_CONNECT_RETRIES):
        try:
            if WRDS_PASSWORD:
                return wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)
            return wrds.Connection(wrds_username=WRDS_USERNAME)
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            retryable = "too many connections" in message
            if (not retryable) or attempt == WRDS_CONNECT_RETRIES - 1:
                raise
            wait_seconds = WRDS_CONNECT_RETRY_SECONDS * (attempt + 1)
            print(
                "WRDS connection limit reached, retrying in "
                f"{wait_seconds}s ({attempt + 1}/{WRDS_CONNECT_RETRIES})..."
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to connect to WRDS")


def _fetch_ecodata_metric(
    db,
    start_period: str,
    end_period: str,
    metric_name: str,
    metric_filter_sql: str,
):
    # Use a window function to pick the last observation in each month without a second scan + self-join.
    query = """
        WITH target_ecoseries AS (
            SELECT DISTINCT
                i.ecoseriesid,
                i.dsmnemonic,
                i.desc_english,
                i.unitcode,
                i.freqcode,
                i.currcode
            FROM trdstrm.ecoinfo i
            JOIN trdstrm.ecocode c
              ON c.code = i.mktcode
            WHERE c.series_type = 5
              AND upper(c.description) IN %(country_names)s
              AND {metric_filter}
        ),
        ranked AS (
            SELECT
                t.ecoseriesid,
                t.dsmnemonic,
                t.desc_english,
                t.unitcode,
                t.freqcode,
                t.currcode,
                d.perioddate,
                date_trunc('month', d.perioddate)::date AS month_key,
                d.series_value,
                MAX(d.perioddate) OVER (
                    PARTITION BY t.ecoseriesid, date_trunc('month', d.perioddate)::date
                ) AS max_perioddate
            FROM trdstrm.ecodata d
            JOIN target_ecoseries t
              ON t.ecoseriesid = d.ecoseriesid
            WHERE d.perioddate BETWEEN %(start)s AND %(end)s
        ),
        picked AS (
            SELECT
                ecoseriesid,
                dsmnemonic,
                desc_english,
                unitcode,
                freqcode,
                currcode,
                perioddate,
                AVG(series_value) AS series_value
            FROM ranked
            WHERE perioddate = max_perioddate
            GROUP BY ecoseriesid, dsmnemonic, desc_english, unitcode, freqcode, currcode, perioddate
        )
        SELECT
            %(metric_name)s::text AS metric,
            'trdstrm.ecodata'::text AS source_table,
            p.ecoseriesid::text AS entity_id,
            p.perioddate AS obs_date,
            p.series_value AS value,
            p.dsmnemonic AS series_code,
            p.desc_english AS description,
            p.unitcode,
            p.freqcode,
            p.currcode,
            'series_value'::text AS value_column,
            'ecoseriesid,perioddate'::text AS key_columns
        FROM picked p
    """.format(metric_filter=metric_filter_sql)
    return db.raw_sql(
        query,
        params={
            **_country_period_params(start_period, end_period),
            "metric_name": metric_name,
        },
    )


def _fetch_3m_interbank(db, start_period: str, end_period: str):
    return _fetch_ecodata_metric(
        db,
        start_period,
        end_period,
        metric_name="3M interbank rate",
        metric_filter_sql=(
            "(lower(i.desc_english) LIKE '%%interbank%%' OR lower(i.dsmnemonic) LIKE '%%inter%%') "
            "AND (lower(i.desc_english) LIKE '%%3 month%%' OR lower(i.dsmnemonic) LIKE '%%3m%%')"
        ),
    )


def _fetch_10y_gov_yield(db, start_period: str, end_period: str):
    return _fetch_ecodata_metric(
        db,
        start_period,
        end_period,
        metric_name="10Y government yield",
        metric_filter_sql=(
            "(lower(i.desc_english) LIKE '%%government%%' OR lower(i.dsmnemonic) LIKE '%%gov%%') "
            "AND lower(i.desc_english) LIKE '%%yield%%' "
            "AND ("
            "lower(i.desc_english) LIKE '%%10 year%%' "
            "OR lower(i.desc_english) LIKE '%%10 years%%' "
            "OR lower(i.dsmnemonic) LIKE '%%10%%'"
            ")"
        ),
    )


def _fetch_market_equity_outstanding(db, start_period: str, end_period: str):
    if MSCI_MARKET_EQUITY_LEVEL == "infocode":
        query = """
            WITH target_infocodes AS (
                SELECT DISTINCT cq.infocode
                FROM trdstrm.ds2ctryqtinfo cq
                WHERE cq.region IN %(region_codes)s
            ),
            ranked AS (
                SELECT
                    n.infocode,
                    date_trunc('month', n.eventdate)::date AS month_key,
                    n.eventdate,
                    n.numshrs,
                    MAX(n.eventdate) OVER (
                        PARTITION BY n.infocode, date_trunc('month', n.eventdate)::date
                    ) AS max_eventdate
                FROM trdstrm.ds2numshares n
                JOIN target_infocodes t
                  ON t.infocode = n.infocode
                WHERE n.eventdate BETWEEN %(start)s AND %(end)s
                  AND n.numshrs IS NOT NULL
            ),
            picked AS (
                SELECT
                    infocode,
                    eventdate,
                    AVG(numshrs) AS numshrs
                FROM ranked
                WHERE eventdate = max_eventdate
                GROUP BY infocode, eventdate
            )
            SELECT
                'market equity outstanding'::text AS metric,
                'trdstrm.ds2numshares'::text AS source_table,
                p.infocode::text AS entity_id,
                p.eventdate AS obs_date,
                p.numshrs AS value,
                NULL::text AS series_code,
                'Number of Shares Outstanding'::text AS description,
                NULL::text AS unitcode,
                NULL::text AS freqcode,
                NULL::text AS currcode,
                'numshrs'::text AS value_column,
                'infocode,eventdate'::text AS key_columns
            FROM picked p
        """
        return db.raw_sql(
            query,
            params=_region_period_params(start_period, end_period),
        )

    query = """
        WITH target_infocodes AS (
            SELECT DISTINCT cq.infocode, cq.region
            FROM trdstrm.ds2ctryqtinfo cq
            WHERE cq.region IN %(region_codes)s
        ),
        ranked AS (
            SELECT
                t.region,
                n.infocode,
                date_trunc('month', n.eventdate)::date AS month_key,
                n.eventdate,
                n.numshrs,
                MAX(n.eventdate) OVER (
                    PARTITION BY n.infocode, date_trunc('month', n.eventdate)::date
                ) AS max_eventdate
            FROM trdstrm.ds2numshares n
            JOIN target_infocodes t
              ON t.infocode = n.infocode
            WHERE n.eventdate BETWEEN %(start)s AND %(end)s
              AND n.numshrs IS NOT NULL
        ),
        picked_infocode AS (
            SELECT
                region,
                infocode,
                month_key,
                AVG(numshrs) AS numshrs
            FROM ranked
            WHERE eventdate = max_eventdate
            GROUP BY region, infocode, month_key
        )
        SELECT
            'market equity outstanding'::text AS metric,
            'trdstrm.ds2numshares'::text AS source_table,
            p.region::text AS entity_id,
            p.month_key AS obs_date,
            SUM(p.numshrs) AS value,
            NULL::text AS series_code,
            'Number of Shares Outstanding (sum over securities in country)'::text AS description,
            NULL::text AS unitcode,
            NULL::text AS freqcode,
            NULL::text AS currcode,
            'numshrs'::text AS value_column,
            'region,month'::text AS key_columns
        FROM picked_infocode p
        GROUP BY p.region, p.month_key
    """
    return db.raw_sql(
        query,
        params=_region_period_params(start_period, end_period),
    )


def _fetch_market_to_book_equity(db, start_period: str, end_period: str):
    if MSCI_MARKET_TO_BOOK_LEVEL == "index":
        query = """
            WITH target_index AS (
                SELECT DISTINCT ei.dsindexcode
                FROM trdstrm.ds2equityindex ei
                WHERE ei.region IN %(region_codes)s
            ),
            ranked AS (
                SELECT
                    a.dsindexcode,
                    a.datatypemnem,
                    date_trunc('month', a.valuedate)::date AS month_key,
                    a.valuedate,
                    a.datatypevalue,
                    MAX(a.valuedate) OVER (
                        PARTITION BY a.dsindexcode, a.datatypemnem, date_trunc('month', a.valuedate)::date
                    ) AS max_valuedate
                FROM trdstrm.ds2indexaddldata a
                JOIN target_index t
                  ON t.dsindexcode = a.dsindexcode
                WHERE a.datatypemnem IN %(mtb_series)s
                  AND a.valuedate BETWEEN %(start)s AND %(end)s
                  AND a.datatypevalue IS NOT NULL
            ),
            picked AS (
                SELECT
                    dsindexcode,
                    datatypemnem,
                    valuedate,
                    AVG(datatypevalue) AS datatypevalue
                FROM ranked
                WHERE valuedate = max_valuedate
                GROUP BY dsindexcode, datatypemnem, valuedate
            )
            SELECT
                'market-to-book equity'::text AS metric,
                'trdstrm.ds2indexaddldata'::text AS source_table,
                p.dsindexcode::text AS entity_id,
                p.valuedate AS obs_date,
                p.datatypevalue AS value,
                p.datatypemnem AS series_code,
                CASE
                    WHEN p.datatypemnem = 'BP' THEN 'Price to Book Value'
                    WHEN p.datatypemnem = 'MSPB' THEN 'MSCI Price to book value'
                    ELSE 'n/a'
                END AS description,
                NULL::text AS unitcode,
                NULL::text AS freqcode,
                NULL::text AS currcode,
                'datatypevalue'::text AS value_column,
                'dsindexcode,datatypemnem,valuedate'::text AS key_columns
            FROM picked p
        """
        return db.raw_sql(
            query,
            params={
                **_region_period_params(start_period, end_period),
                "mtb_series": tuple(MSCI_MARKET_TO_BOOK_SERIES),
            },
        )

    query = """
        WITH target_index AS (
            SELECT DISTINCT ei.dsindexcode, ei.region
            FROM trdstrm.ds2equityindex ei
            WHERE ei.region IN %(region_codes)s
        ),
        ranked AS (
            SELECT
                t.region,
                a.dsindexcode,
                a.datatypemnem,
                date_trunc('month', a.valuedate)::date AS month_key,
                a.valuedate,
                a.datatypevalue,
                MAX(a.valuedate) OVER (
                    PARTITION BY a.dsindexcode, a.datatypemnem, date_trunc('month', a.valuedate)::date
                ) AS max_valuedate
            FROM trdstrm.ds2indexaddldata a
            JOIN target_index t
              ON t.dsindexcode = a.dsindexcode
            WHERE a.datatypemnem IN %(mtb_series)s
              AND a.valuedate BETWEEN %(start)s AND %(end)s
              AND a.datatypevalue IS NOT NULL
        ),
        picked_infocode AS (
            SELECT
                region,
                dsindexcode,
                datatypemnem,
                month_key,
                AVG(datatypevalue) AS datatypevalue
            FROM ranked
            WHERE valuedate = max_valuedate
            GROUP BY region, dsindexcode, datatypemnem, month_key
        ),
        aggregated AS (
            SELECT
                p.region,
                p.datatypemnem,
                p.month_key,
                AVG(p.datatypevalue) AS datatypevalue
            FROM picked_infocode p
            GROUP BY p.region, p.datatypemnem, p.month_key
        )
        SELECT
            'market-to-book equity'::text AS metric,
            'trdstrm.ds2indexaddldata'::text AS source_table,
            a.region::text AS entity_id,
            a.month_key AS obs_date,
            a.datatypevalue AS value,
            a.datatypemnem AS series_code,
            CASE
                WHEN a.datatypemnem = 'BP' THEN 'Price to Book Value'
                WHEN a.datatypemnem = 'MSPB' THEN 'MSCI Price to book value'
                ELSE 'n/a'
            END AS description,
            NULL::text AS unitcode,
            NULL::text AS freqcode,
            NULL::text AS currcode,
            'datatypevalue'::text AS value_column,
            'region,datatypemnem,month'::text AS key_columns
        FROM aggregated a
    """
    return db.raw_sql(
        query,
        params={
            **_region_period_params(start_period, end_period),
            "mtb_series": tuple(MSCI_MARKET_TO_BOOK_SERIES),
        },
    )


def _fetch_monthly_equity_return_proxy(db, start_period: str, end_period: str):
    # Fast path (default): use index-level return index from ds2indexdata joined via ds2equityindex.
    # Slow legacy path: security-level ds2primqtri/ds2scdqtri when MSCI_EQUITY_RETURN_LEVEL=infocode.
    if MSCI_EQUITY_RETURN_LEVEL != "infocode":
        if MSCI_EQUITY_RETURN_LEVEL not in {"country", "index"}:
            raise ValueError(
                "MSCI_EQUITY_RETURN_LEVEL must be one of: country, index, infocode"
            )

        query = """
            WITH target_index AS (
                SELECT DISTINCT ei.dsindexcode, ei.region
                FROM trdstrm.ds2equityindex ei
                WHERE ei.region IN %(region_codes)s
                  AND ei.sourcecode = '870'
                  AND ei.ldb = 'SIF'
                  AND upper(ei.indexdesc) LIKE '%%MSCI%%'
            ),
            ranked AS (
                SELECT
                    t.region,
                    d.dsindexcode,
                    date_trunc('month', d.valuedate)::date AS month_key,
                    d.valuedate,
                    d.ri,
                    MAX(d.valuedate) OVER (
                        PARTITION BY d.dsindexcode, date_trunc('month', d.valuedate)::date
                    ) AS max_valuedate
                FROM trdstrm.ds2indexdata d
                JOIN target_index t
                  ON t.dsindexcode = d.dsindexcode
                WHERE d.valuedate BETWEEN %(start)s AND %(end)s
                  AND d.ri IS NOT NULL
            ),
            picked_index AS (
                SELECT
                    region,
                    dsindexcode,
                    month_key,
                    AVG(ri) AS ri
                FROM ranked
                WHERE valuedate = max_valuedate
                GROUP BY region, dsindexcode, month_key
            ),
            aggregated AS (
                SELECT
                    region,
                    month_key,
                    AVG(ri) AS ri
                FROM picked_index
                GROUP BY region, month_key
            )
            SELECT
                'monthly equity return (USD proxy)'::text AS metric,
                'trdstrm.ds2indexdata'::text AS source_table,
                a.region::text AS entity_id,
                a.month_key AS obs_date,
                a.ri AS value,
                NULL::text AS series_code,
                'Equity index return index (ri) from ds2indexdata (region average if multiple indices)'::text AS description,
                NULL::text AS unitcode,
                NULL::text AS freqcode,
                NULL::text AS currcode,
                'ri'::text AS value_column,
                'region,month'::text AS key_columns
            FROM aggregated a
        """

        if MSCI_EQUITY_RETURN_LEVEL == "index":
            query = """
                WITH target_index AS (
                    SELECT DISTINCT ei.dsindexcode, ei.region
                    FROM trdstrm.ds2equityindex ei
                    WHERE ei.region IN %(region_codes)s
                      AND ei.sourcecode = '870'
                      AND ei.ldb = 'SIF'
                      AND upper(ei.indexdesc) LIKE '%%MSCI%%'
                ),
                ranked AS (
                    SELECT
                        t.region,
                        d.dsindexcode,
                        date_trunc('month', d.valuedate)::date AS month_key,
                        d.valuedate,
                        d.ri,
                        MAX(d.valuedate) OVER (
                            PARTITION BY d.dsindexcode, date_trunc('month', d.valuedate)::date
                        ) AS max_valuedate
                    FROM trdstrm.ds2indexdata d
                    JOIN target_index t
                      ON t.dsindexcode = d.dsindexcode
                    WHERE d.valuedate BETWEEN %(start)s AND %(end)s
                      AND d.ri IS NOT NULL
                ),
                picked AS (
                    SELECT
                        region,
                        dsindexcode,
                        month_key,
                        AVG(ri) AS ri
                    FROM ranked
                    WHERE valuedate = max_valuedate
                    GROUP BY region, dsindexcode, month_key
                )
                SELECT
                    'monthly equity return (USD proxy)'::text AS metric,
                    'trdstrm.ds2indexdata'::text AS source_table,
                    p.dsindexcode::text AS entity_id,
                    p.month_key AS obs_date,
                    p.ri AS value,
                    NULL::text AS series_code,
                    'Equity index return index (ri) from ds2indexdata (per dsindexcode)'::text AS description,
                    NULL::text AS unitcode,
                    NULL::text AS freqcode,
                    NULL::text AS currcode,
                    'ri'::text AS value_column,
                    'dsindexcode,month'::text AS key_columns
                FROM picked p
            """

        return db.raw_sql(query, params=_region_period_params(start_period, end_period))

    # Legacy slow path (security-level)
    def _fetch_from_table(table_name: str, source_table: str, description: str) -> pd.DataFrame:
        query = f"""
            WITH target_infocodes AS (
                SELECT DISTINCT cq.infocode, cq.region
                FROM trdstrm.ds2ctryqtinfo cq
                WHERE cq.region IN %(region_codes)s
            ),
            ranked AS (
                SELECT
                    t.region,
                    r.infocode,
                    date_trunc('month', r.marketdate)::date AS month_key,
                    r.marketdate,
                    r.ri,
                    MAX(r.marketdate) OVER (
                        PARTITION BY r.infocode, date_trunc('month', r.marketdate)::date
                    ) AS max_marketdate
                FROM trdstrm.{table_name} r
                JOIN target_infocodes t
                  ON t.infocode = r.infocode
                WHERE r.marketdate BETWEEN %(start)s AND %(end)s
                  AND r.ri IS NOT NULL
            ),
            picked_infocode AS (
                SELECT
                    region,
                    infocode,
                    month_key,
                    AVG(ri) AS ri
                FROM ranked
                WHERE marketdate = max_marketdate
                GROUP BY region, infocode, month_key
            ),
            aggregated AS (
                SELECT
                    p.region,
                    p.month_key,
                    AVG(p.ri) AS ri
                FROM picked_infocode p
                GROUP BY p.region, p.month_key
            )
            SELECT
                'monthly equity return (USD proxy)'::text AS metric,
                %(source_table)s::text AS source_table,
                a.region::text AS entity_id,
                a.month_key AS obs_date,
                a.ri AS value,
                NULL::text AS series_code,
                %(description)s::text AS description,
                NULL::text AS unitcode,
                NULL::text AS freqcode,
                NULL::text AS currcode,
                'ri'::text AS value_column,
                'region,month'::text AS key_columns
            FROM aggregated a
        """
        return db.raw_sql(
            query,
            params={
                **_region_period_params(start_period, end_period),
                "source_table": source_table,
                "description": description,
            },
        )

    frames = [
        _fetch_from_table(
            table_name="ds2primqtri",
            source_table="trdstrm.ds2primqtri",
            description="Primary quote return index (ri)",
        )
    ]

    if MSCI_INCLUDE_SECONDARY_RETURNS:
        frames.append(
            _fetch_from_table(
                table_name="ds2scdqtri",
                source_table="trdstrm.ds2scdqtri",
                description="Secondary quote return index (ri)",
            )
        )

    return pd.concat(frames, ignore_index=True, copy=False)


def pull_needed_msci_data(
    start_period=MSCI_START_PERIOD,
    end_period=MSCI_END_PERIOD,
):
    def _timed_fetch(label: str, fetch_fn):
        start_ts = time.perf_counter()
        frame = fetch_fn()
        elapsed = time.perf_counter() - start_ts
        print(f"{label}: {len(frame):,} rows in {elapsed:.2f}s")
        return frame

    pull_steps = [
        ("3M interbank rate", _fetch_3m_interbank),
        ("10Y government yield", _fetch_10y_gov_yield),
        ("market equity outstanding", _fetch_market_equity_outstanding),
        ("market-to-book equity", _fetch_market_to_book_equity),
        ("monthly equity return proxy", _fetch_monthly_equity_return_proxy),
    ]

    with _connect_wrds() as db:
        frames = []
        for label, fetch_fn in pull_steps:
            print(f"Pulling {label}...")
            frames.append(
                _timed_fetch(
                    label,
                    lambda fn=fetch_fn: fn(db, start_period, end_period),
                )
            )

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.concat(frames, ignore_index=True, copy=False)
    result["obs_date"] = pd.to_datetime(result["obs_date"], errors="coerce")
    if MSCI_SORT_OUTPUT:
        result = result.sort_values(["metric", "obs_date"], ascending=[True, False]).reset_index(drop=True)
    return result


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / MSCI_NEEDED_OUTPUT_FILE

    df = pull_needed_msci_data(
        start_period=MSCI_START_PERIOD,
        end_period=MSCI_END_PERIOD,
    )

    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
