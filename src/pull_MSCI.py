"""Pull only confirmed required MSCI-related datasets from WRDS Datastream.

Target metrics (from validated dictionary mappings):
1) 3M interbank rate              -> trdstrm.ecodata.series_value + trdstrm.ecoinfo
2) 10Y government yield            -> trdstrm.ecodata.series_value + trdstrm.ecoinfo
3) market equity outstanding       -> trdstrm.ds2numshares.numshrs
4) market-to-book equity           -> trdstrm.ds2indexaddldata.datatypevalue (BP/MSPB)
5) monthly equity return (USD proxy) -> trdstrm.ds2primqtri.ri + trdstrm.ds2scdqtri.ri
"""

from pathlib import Path

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

MSCI_START_PERIOD = "2003"
MSCI_END_PERIOD = "2020"
MSCI_NEEDED_OUTPUT_FILE = str(
    config("MSCI_NEEDED_OUTPUT_FILE", default="data_msci_datastream.parquet", cast=str)
)
MSCI_INCLUDE_SECONDARY_RETURNS = config(
    "MSCI_INCLUDE_SECONDARY_RETURNS", default=False, cast=bool
)

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


def _connect_wrds():
    if not WRDS_USERNAME:
        raise ValueError("WRDS_USERNAME is required in .env")
    if WRDS_PASSWORD:
        return wrds.Connection(wrds_username=WRDS_USERNAME, wrds_password=WRDS_PASSWORD)
    return wrds.Connection(wrds_username=WRDS_USERNAME)


def _fetch_3m_interbank(db, start_period: str, end_period: str):
    query = """
        WITH max_dates AS (
                        SELECT
                                i.ecoseriesid,
                date_trunc('month', d.perioddate) AS month_key,
                MAX(d.perioddate) AS perioddate
                        FROM trdstrm.ecoinfo i
                        JOIN trdstrm.ecodata d
                            ON d.ecoseriesid = i.ecoseriesid
                        WHERE (lower(i.desc_english) LIKE '%%interbank%%' OR lower(i.dsmnemonic) LIKE '%%inter%%')
                            AND (lower(i.desc_english) LIKE '%%3 month%%' OR lower(i.dsmnemonic) LIKE '%%3m%%')
                            AND d.perioddate BETWEEN %(start)s AND %(end)s
                            AND EXISTS (
                                SELECT 1
                                FROM trdstrm.ecocode c
                                WHERE c.series_type = 5
                                  AND c.code = i.mktcode
                                  AND upper(c.description) IN %(country_names)s
                            )
                        GROUP BY i.ecoseriesid, date_trunc('month', d.perioddate)
                )
                SELECT
                        '3M interbank rate'::text AS metric,
                        'trdstrm.ecodata'::text AS source_table,
                        m.ecoseriesid::text AS entity_id,
                        d.perioddate AS obs_date,
                        d.series_value AS value,
                        i.dsmnemonic AS series_code,
                        i.desc_english AS description,
                        i.unitcode,
                        i.freqcode,
                        i.currcode,
                        'series_value'::text AS value_column,
                        'ecoseriesid,perioddate'::text AS key_columns
                FROM max_dates m
                JOIN trdstrm.ecodata d
                    ON d.ecoseriesid = m.ecoseriesid
                 AND d.perioddate = m.perioddate
                JOIN trdstrm.ecoinfo i
                    ON i.ecoseriesid = m.ecoseriesid
                ORDER BY d.perioddate DESC, m.ecoseriesid
    """
    return db.raw_sql(
        query,
        params={
            "start": f"{start_period}-01-01",
            "end": f"{end_period}-12-31",
            "country_names": tuple(TARGET_COUNTRIES),
        },
    )


def _fetch_10y_gov_yield(db, start_period: str, end_period: str):
    query = """
        WITH max_dates AS (
                        SELECT
                                i.ecoseriesid,
                date_trunc('month', d.perioddate) AS month_key,
                MAX(d.perioddate) AS perioddate
                        FROM trdstrm.ecoinfo i
                        JOIN trdstrm.ecodata d
                            ON d.ecoseriesid = i.ecoseriesid
                        WHERE (lower(i.desc_english) LIKE '%%government%%' OR lower(i.dsmnemonic) LIKE '%%gov%%')
                            AND lower(i.desc_english) LIKE '%%yield%%'
                            AND (
                                lower(i.desc_english) LIKE '%%10 year%%'
                                OR lower(i.desc_english) LIKE '%%10 years%%'
                                OR lower(i.dsmnemonic) LIKE '%%10%%'
                            )
                            AND d.perioddate BETWEEN %(start)s AND %(end)s
                            AND EXISTS (
                                SELECT 1
                                FROM trdstrm.ecocode c
                                WHERE c.series_type = 5
                                  AND c.code = i.mktcode
                                  AND upper(c.description) IN %(country_names)s
                            )
                        GROUP BY i.ecoseriesid, date_trunc('month', d.perioddate)
                )
                SELECT
                        '10Y government yield'::text AS metric,
                        'trdstrm.ecodata'::text AS source_table,
                        m.ecoseriesid::text AS entity_id,
                        d.perioddate AS obs_date,
                        d.series_value AS value,
                        i.dsmnemonic AS series_code,
                        i.desc_english AS description,
                        i.unitcode,
                        i.freqcode,
                        i.currcode,
                        'series_value'::text AS value_column,
                        'ecoseriesid,perioddate'::text AS key_columns
                FROM max_dates m
                JOIN trdstrm.ecodata d
                    ON d.ecoseriesid = m.ecoseriesid
                 AND d.perioddate = m.perioddate
                JOIN trdstrm.ecoinfo i
                    ON i.ecoseriesid = m.ecoseriesid
                ORDER BY d.perioddate DESC, m.ecoseriesid
    """
    return db.raw_sql(
        query,
        params={
            "start": f"{start_period}-01-01",
            "end": f"{end_period}-12-31",
            "country_names": tuple(TARGET_COUNTRIES),
        },
    )


def _fetch_market_equity_outstanding(db, start_period: str, end_period: str):
    query = """
        WITH max_dates AS (
            SELECT
                n.infocode,
                date_trunc('month', n.eventdate) AS month_key,
                MAX(n.eventdate) AS eventdate
            FROM trdstrm.ds2numshares n
            WHERE n.eventdate BETWEEN %(start)s AND %(end)s
              AND n.numshrs IS NOT NULL
                            AND EXISTS (
                                    SELECT 1
                                    FROM trdstrm.ds2ctryqtinfo cq
                                    WHERE cq.infocode = n.infocode
                                        AND upper(cq.region) IN %(region_codes)s
                            )
            GROUP BY n.infocode, date_trunc('month', n.eventdate)
        )
        SELECT
            'market equity outstanding'::text AS metric,
            'trdstrm.ds2numshares'::text AS source_table,
            m.infocode::text AS entity_id,
            n.eventdate AS obs_date,
            n.numshrs AS value,
            NULL::text AS series_code,
            'Number of Shares Outstanding'::text AS description,
            NULL::text AS unitcode,
            NULL::text AS freqcode,
            NULL::text AS currcode,
            'numshrs'::text AS value_column,
            'infocode,eventdate'::text AS key_columns
        FROM max_dates m
        JOIN trdstrm.ds2numshares n
          ON n.infocode = m.infocode
         AND n.eventdate = m.eventdate
        ORDER BY n.eventdate DESC, m.infocode
    """
    return db.raw_sql(
        query,
        params={
            "start": f"{start_period}-01-01",
            "end": f"{end_period}-12-31",
            "region_codes": tuple(TARGET_REGION_CODES),
        },
    )


def _fetch_market_to_book_equity(db, start_period: str, end_period: str):
    query = """
        WITH target_index AS (
            SELECT DISTINCT ei.dsindexcode
            FROM trdstrm.ds2equityindex ei
            WHERE upper(ei.region) IN %(region_codes)s
        ),
        picked AS (
            SELECT DISTINCT ON (
                a.dsindexcode,
                upper(a.datatypemnem),
                date_trunc('month', a.valuedate)
            )
                a.dsindexcode,
                a.valuedate,
                a.datatypevalue,
                upper(a.datatypemnem) AS datatypemnem
            FROM trdstrm.ds2indexaddldata a
            JOIN target_index t
              ON t.dsindexcode = a.dsindexcode
            WHERE upper(a.datatypemnem) IN ('BP', 'MSPB')
              AND a.valuedate BETWEEN %(start)s AND %(end)s
              AND a.datatypevalue IS NOT NULL
            ORDER BY
                a.dsindexcode,
                upper(a.datatypemnem),
                date_trunc('month', a.valuedate),
                a.valuedate DESC
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
        ORDER BY p.valuedate DESC, p.dsindexcode
    """
    return db.raw_sql(
        query,
        params={
            "start": f"{start_period}-01-01",
            "end": f"{end_period}-12-31",
            "region_codes": tuple(TARGET_REGION_CODES),
        },
    )


def _fetch_monthly_equity_return_proxy(db, start_period: str, end_period: str):
    query = """
        WITH prim AS (
            SELECT
                r.infocode,
                date_trunc('month', r.marketdate) AS month_key,
                MAX(r.marketdate) AS marketdate
            FROM trdstrm.ds2primqtri r
            WHERE r.marketdate BETWEEN %(start)s AND %(end)s
              AND r.ri IS NOT NULL
                            AND EXISTS (
                                    SELECT 1
                                    FROM trdstrm.ds2ctryqtinfo cq
                                    WHERE cq.infocode = r.infocode
                                        AND upper(cq.region) IN %(region_codes)s
                            )
            GROUP BY r.infocode, date_trunc('month', r.marketdate)
        ),
        scd AS (
            SELECT
                r.infocode,
                date_trunc('month', r.marketdate) AS month_key,
                MAX(r.marketdate) AS marketdate
            FROM trdstrm.ds2scdqtri r
            WHERE r.marketdate BETWEEN %(start)s AND %(end)s
              AND r.ri IS NOT NULL
                            AND EXISTS (
                                    SELECT 1
                                    FROM trdstrm.ds2ctryqtinfo cq
                                    WHERE cq.infocode = r.infocode
                                        AND upper(cq.region) IN %(region_codes)s
                            )
            GROUP BY r.infocode, date_trunc('month', r.marketdate)
        )

        SELECT
            'monthly equity return (USD proxy)'::text AS metric,
            'trdstrm.ds2primqtri'::text AS source_table,
            m.infocode::text AS entity_id,
            r.marketdate AS obs_date,
            r.ri AS value,
            NULL::text AS series_code,
            'Primary quote return index (ri)'::text AS description,
            NULL::text AS unitcode,
            NULL::text AS freqcode,
            NULL::text AS currcode,
            'ri'::text AS value_column,
            'infocode,marketdate'::text AS key_columns
        FROM prim m
        JOIN trdstrm.ds2primqtri r
          ON r.infocode = m.infocode AND r.marketdate = m.marketdate

        UNION ALL

        SELECT
            'monthly equity return (USD proxy)'::text AS metric,
            'trdstrm.ds2scdqtri'::text AS source_table,
            m.infocode::text AS entity_id,
            r.marketdate AS obs_date,
            r.ri AS value,
            NULL::text AS series_code,
            'Secondary quote return index (ri)'::text AS description,
            NULL::text AS unitcode,
            NULL::text AS freqcode,
            NULL::text AS currcode,
            'ri'::text AS value_column,
            'infocode,marketdate'::text AS key_columns
        FROM scd m
        JOIN trdstrm.ds2scdqtri r
          ON r.infocode = m.infocode AND r.marketdate = m.marketdate
    """
    df = db.raw_sql(
        query,
        params={
            "start": f"{start_period}-01-01",
            "end": f"{end_period}-12-31",
            "region_codes": tuple(TARGET_REGION_CODES),
        },
    )
    if not MSCI_INCLUDE_SECONDARY_RETURNS:
        return df[df["source_table"] == "trdstrm.ds2primqtri"].copy()
    return df


def pull_needed_msci_data(
    start_period=MSCI_START_PERIOD,
    end_period=MSCI_END_PERIOD,
    persist_path=None,
):
    collected_frames = []

    def _persist_if_needed():
        if persist_path is None or not collected_frames:
            return
        combined = pd.concat(collected_frames, ignore_index=True)
        combined["obs_date"] = pd.to_datetime(combined["obs_date"], errors="coerce")
        combined = combined.sort_values(["metric", "obs_date"], ascending=[True, False]).reset_index(drop=True)
        combined.to_parquet(persist_path, index=False)

    with _connect_wrds() as db:
        print("Pulling 3M interbank rate...")
        df_3m = _fetch_3m_interbank(db, start_period, end_period)
        if not df_3m.empty:
            collected_frames.append(df_3m)
            _persist_if_needed()

        print("Pulling 10Y government yield...")
        df_10y = _fetch_10y_gov_yield(db, start_period, end_period)
        if not df_10y.empty:
            collected_frames.append(df_10y)
            _persist_if_needed()

        print("Pulling market equity outstanding...")
        df_mkt_out = _fetch_market_equity_outstanding(db, start_period, end_period)
        if not df_mkt_out.empty:
            collected_frames.append(df_mkt_out)
            _persist_if_needed()

        print("Pulling market-to-book equity...")
        df_mtb = _fetch_market_to_book_equity(db, start_period, end_period)
        if not df_mtb.empty:
            collected_frames.append(df_mtb)
            _persist_if_needed()

        print("Pulling monthly equity return proxy...")
        df_ret = _fetch_monthly_equity_return_proxy(db, start_period, end_period)
        if not df_ret.empty:
            collected_frames.append(df_ret)
            _persist_if_needed()

        frames = [
            df_3m,
            df_10y,
            df_mkt_out,
            df_mtb,
            df_ret,
        ]

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
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
        )

    result = pd.concat(frames, ignore_index=True)
    result["obs_date"] = pd.to_datetime(result["obs_date"], errors="coerce")
    result = result.sort_values(["metric", "obs_date"], ascending=[True, False]).reset_index(drop=True)
    return result


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / MSCI_NEEDED_OUTPUT_FILE

    df = pull_needed_msci_data(
        start_period=MSCI_START_PERIOD,
        end_period=MSCI_END_PERIOD,
        persist_path=output_path,
    )

    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
