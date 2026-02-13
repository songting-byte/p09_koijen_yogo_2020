"""Pull OECD Table T720R_A raw observations only.

This script follows OECD SDMX REST endpoint pattern:
https://sdmx.oecd.org/public/rest/data/{flow_ref}/{key}

Flow used:
OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1

Output:
- one raw parquet file only (no derived aggregation)
"""

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from settings import config


DATA_DIR = config("DATA_DIR")
OECD_OUTPUT_FILE = config("OECD_OUTPUT_FILE", default="oecd_t720.parquet", cast=str)
OECD_START_PERIOD = config("OECD_START_PERIOD", default="2003", cast=str)
OECD_END_PERIOD = config("OECD_END_PERIOD", default="2020", cast=str)

OECD_BASE_DATA_URL = config(
    "OECD_BASE_DATA_URL",
    default=(
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1/"
        "A..CAN+USA+BEL+DNK+FIN+FRA+DEU+ITA+ISR+NLD+NOR+PRT+ESP+SWE+CHE+GBR+"
        "BRA+JPN+COL+CZE+GRC+HUN+MEX+POL+KOR+AUT..S1...L+A..F3+F5+F2+F4.."
        "XDC......?startPeriod=2003&endPeriod=2020"
    ),
    cast=str,
)

def _parse_base_url(base_url):
    parsed = urlparse(base_url)
    path_parts = parsed.path.split("/data/")
    if len(path_parts) != 2:
        raise ValueError("Base URL must contain '/data/' segment")
    flow_and_key = path_parts[1]
    flow_ref, key = flow_and_key.split("/", 1)
    key_template = key.split("?")[0]
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    return flow_ref, key_template, params


def _data_url(flow_ref, key):
    return f"https://sdmx.oecd.org/public/rest/data/{flow_ref}/{key}"


def _get_json(session, url, params):
    headers = {"Accept": "application/vnd.sdmx.data+json;version=2.0"}
    response = session.get(url, params=params, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def _index_to_value(dim_values, dim_id, index):
    values = dim_values.get(dim_id, [])
    if index < 0 or index >= len(values):
        return None, None
    value = values[index]
    return value.get("id"), value.get("name")


def _sdmx_json_to_tidy(payload):
    data_block = payload.get("data", payload)
    structures = data_block.get("structures", [])
    datasets = data_block.get("dataSets", [])

    if not structures or not datasets:
        return pd.DataFrame()

    observations = datasets[0].get("observations", {})
    obs_dims = structures[0].get("dimensions", {}).get("observation", [])

    dim_ids = [dim["id"] for dim in obs_dims]
    dim_values = {
        dim["id"]: [
            {"id": v.get("id"), "name": v.get("name")} for v in dim.get("values", [])
        ]
        for dim in obs_dims
    }

    rename_map = {
        "REF_AREA": "reference_area",
        "TIME_PERIOD": "time_period",
        "INSTR_ASSET": "financial_instrument",
        "MATURITY": "original_maturity",
        "ACCOUNTING_ENTRY": "accounting_entry",
        "TRANSACTION": "transaction",
        "SECTOR": "sector",
        "FREQ": "frequency",
        "UNIT_MEASURE": "unit_measure",
        "CURRENCY_DENOM": "currency_denom",
    }

    rows = []
    for key, obs in observations.items():
        indices = [int(i) for i in key.split(":")]
        row = {}
        for dim_id, idx in zip(dim_ids, indices):
            code, label = _index_to_value(dim_values, dim_id, idx)
            col = rename_map.get(dim_id, dim_id.lower())
            row[col] = code
            row[f"{col}_label"] = label
        row["value"] = obs[0] if isinstance(obs, list) and obs else obs
        rows.append(row)

    return pd.DataFrame(rows)


def pull_oecd_table_0720(
    base_url=OECD_BASE_DATA_URL,
    start_period=OECD_START_PERIOD,
    end_period=OECD_END_PERIOD,
):
    flow_ref, key_template, base_params = _parse_base_url(base_url)
    key = key_template

    params = {
        **base_params,
        "startPeriod": str(start_period),
        "endPeriod": str(end_period),
        "dimensionAtObservation": "AllDimensions",
        "format": "jsondata",
    }

    with requests.Session() as session:
        payload = _get_json(session, _data_url(flow_ref, key), params=params)

    raw_df = _sdmx_json_to_tidy(payload)
    return raw_df


if __name__ == "__main__":
    raw = pull_oecd_table_0720()

    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OECD_OUTPUT_FILE
    raw.to_parquet(output_path, index=False)
    print(f"Wrote: {output_path}")