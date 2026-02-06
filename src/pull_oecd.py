"""Pull OECD SDMX Table 0720 data and save as a tidy CSV.

This script uses the OECD SDMX REST API with the Data Explorer “Developer API
query builder” URL as a template. It first queries the SDMX structure to learn
dimension order and code lists, then batches data requests (by reference area
and, if needed, by instrument/maturity) to avoid oversized responses.

To reproduce or change the year range, update `OECD_START_PERIOD` and
`OECD_END_PERIOD` in your `.env` file (or edit the defaults below).
"""

from datetime import date, datetime
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
OECD_USERNAME = config("OECD_USERNAME", default=None, cast=str)
OECD_PASSWORD = config("OECD_PASSWORD", default=None, cast=str)

# Base URL copied from OECD Data Explorer “Developer API query builder”
OECD_BASE_DATA_URL = config(
    "OECD_BASE_DATA_URL",
    default=(
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.1/"
        "A..USA..S1......T+S+L.USD......?startPeriod=2003&endPeriod=2020"
        "&dimensionAtObservation=AllDimensions"
    ),
    cast=str,
)

OECD_START_PERIOD = config("OECD_START_PERIOD", default="2003", cast=str)
OECD_END_PERIOD = config("OECD_END_PERIOD", default="2020", cast=str)
OECD_OUTPUT_FILE = config("OECD_OUTPUT_FILE", default="oecd_t720.csv", cast=str)
OECD_STRUCTURE_CACHE = config(
    "OECD_STRUCTURE_CACHE",
    default=str(Path(DATA_DIR) / "oecd_t720_structure.xml"),
    cast=str,
)
OECD_STRUCTURE_REFRESH = config("OECD_STRUCTURE_REFRESH", default="0", cast=str)

# Target instruments and maturity splits
TARGET_INSTRUMENTS = ["F3", "F51", "F5", "F519"]
TARGET_MATURITY_CODES = ["T", "S", "L"]


def _format_period(value):
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y")
    return str(value)


def _build_session(username=None, password=None):
    session = requests.Session()
    if username and password:
        session.auth = (username, password)
    return session


def _get_json_with_retry(session, url, params=None, accept=None, max_retries=10):
    headers = None
    if accept:
        headers = {"Accept": accept}
    for attempt in range(max_retries):
        response = session.get(url, params=params, headers=headers)
        if response.status_code == 401 and getattr(session, "auth", None):
            fallback = requests.Session()
            response = fallback.get(url, params=params, headers=headers)
        if response.status_code == 429 and attempt < max_retries - 1:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (2 ** attempt)
            wait_seconds = max(wait_seconds, 10)
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return response.json()


def _get_text_with_retry(session, url, params=None, accept=None, max_retries=10):
    headers = None
    if accept:
        headers = {"Accept": accept}
    for attempt in range(max_retries):
        response = session.get(url, params=params, headers=headers)
        if response.status_code == 401 and getattr(session, "auth", None):
            fallback = requests.Session()
            response = fallback.get(url, params=params, headers=headers)
        if response.status_code == 429 and attempt < max_retries - 1:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (2 ** attempt)
            wait_seconds = max(wait_seconds, 10)
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        return response.text
    response.raise_for_status()
    return response.text


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


def _structure_url(flow_ref):
    return "https://sdmx.oecd.org/public/rest/datastructure/" f"{flow_ref}"


def _dataflow_url(flow_ref):
    return "https://sdmx.oecd.org/public/rest/dataflow/" f"{flow_ref}"


def _dataflow_list_url():
    return "https://sdmx.oecd.org/public/rest/dataflow"


def _datastructure_url_from_ref(agency_id, dsd_id, version):
    return (
        "https://sdmx.oecd.org/public/rest/datastructure/"
        f"{agency_id},{dsd_id},{version}"
    )


def _datastructure_url_from_ref_slash(agency_id, dsd_id, version=None):
    base = "https://sdmx.oecd.org/public/rest/datastructure/"
    if version:
        return f"{base}{agency_id}/{dsd_id}/{version}"
    return f"{base}{agency_id}/{dsd_id}"


def _parse_flow_ref(flow_ref):
    parts = flow_ref.split(",")
    if len(parts) != 3:
        return None
    agency_id, middle, version = parts
    dsd_id = None
    df_id = None
    if "@" in middle:
        dsd_id, df_id = middle.split("@", 1)
    return {
        "agency_id": agency_id,
        "version": version,
        "dsd_id": dsd_id,
        "dataflow_id": df_id,
    }


def _data_url(flow_ref, key):
    return "https://sdmx.oecd.org/public/rest/data/" f"{flow_ref}/{key}"


def _parse_structure_xml(xml_text):
    root = ET.fromstring(xml_text)

    ns = {
        "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
        "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
        "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    }

    codelists = {}
    for codelist in root.findall(".//str:Codelist", ns):
        cl_id = codelist.get("id")
        codes = []
        for code in codelist.findall("str:Code", ns):
            code_id = code.get("id")
            name_el = code.find("com:Name", ns)
            label = name_el.text if name_el is not None else None
            codes.append({"code": code_id, "label": label})
        codelists[cl_id] = codes

    series_dims = []
    dim_values = {}
    dim_list = root.findall(".//str:DimensionList/str:Dimension", ns)
    time_dim = root.find(".//str:DimensionList/str:TimeDimension", ns)
    for dim in dim_list:
        dim_id = dim.get("id")
        ref = dim.find(".//{*}Enumeration/{*}Ref")
        cl_id = ref.get("id") if ref is not None else None
        values = codelists.get(cl_id, [])
        series_dims.append({"id": dim_id, "values": values})
        dim_values[dim_id] = values

    obs_dims = []
    if time_dim is not None:
        obs_dims.append({"id": time_dim.get("id", "TIME_PERIOD"), "values": []})

    return series_dims, obs_dims, dim_values


def _is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _extract_dsd_ref(xml_text):
    root = ET.fromstring(xml_text)
    ns = {
        "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
        "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
        "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    }
    ref = root.find(".//str:Dataflow/str:Structure/com:Ref", ns)
    if ref is None:
        return None
    return {
        "agency_id": ref.get("agencyID"),
        "dsd_id": ref.get("id"),
        "version": ref.get("version"),
    }


def _extract_dsd_ref_for_dataflow(xml_text, dataflow_id):
    root = ET.fromstring(xml_text)
    ns = {
        "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
        "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
        "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    }
    for df in root.findall(".//str:Dataflow", ns):
        if df.get("id") != dataflow_id:
            continue
        ref = df.find("str:Structure/com:Ref", ns)
        if ref is None:
            return None
        return {
            "agency_id": ref.get("agencyID"),
            "dsd_id": ref.get("id"),
            "version": ref.get("version"),
        }
    return None


def fetch_structure(flow_ref, session):
    cache_path = Path(OECD_STRUCTURE_CACHE)
    if cache_path.exists() and not _is_truthy(OECD_STRUCTURE_REFRESH):
        xml_text = cache_path.read_text(encoding="utf-8")
        return _parse_structure_xml(xml_text)

    flow_parts = _parse_flow_ref(flow_ref)
    if flow_parts and flow_parts.get("dsd_id"):
        versions = [flow_parts["version"], "latest"]
        agencies = [flow_parts["agency_id"], "OECD"]
        last_error = None
        for agency in agencies:
            for version in versions:
                urls = [
                    _datastructure_url_from_ref(agency, flow_parts["dsd_id"], version),
                    _datastructure_url_from_ref_slash(agency, flow_parts["dsd_id"], version),
                    _datastructure_url_from_ref_slash(agency, flow_parts["dsd_id"]),
                ]
                for url in urls:
                    try:
                        xml_text = _get_text_with_retry(
                            session,
                            url,
                            params={"references": "all"},
                            accept="application/vnd.sdmx.structure+xml;version=2.1",
                        )
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(xml_text, encoding="utf-8")
                        return _parse_structure_xml(xml_text)
                    except requests.HTTPError as exc:
                        last_error = exc
                        if exc.response is None or exc.response.status_code != 404:
                            raise
        if last_error is not None:
            raise last_error

    try:
        xml_text = _get_text_with_retry(
            session,
            _structure_url(flow_ref),
            accept="application/vnd.sdmx.structure+xml;version=2.1",
        )
        return _parse_structure_xml(xml_text)
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise

    dataflow_xml = _get_text_with_retry(
        session,
        _dataflow_list_url(),
        accept="application/vnd.sdmx.structure+xml;version=2.1",
    )
    dsd_ref = None
    if flow_parts and flow_parts.get("dataflow_id"):
        dsd_ref = _extract_dsd_ref_for_dataflow(dataflow_xml, flow_parts["dataflow_id"])
    if dsd_ref is None:
        dsd_ref = _extract_dsd_ref(dataflow_xml)
    if not dsd_ref or not all(dsd_ref.values()):
        raise ValueError("Unable to resolve DSD reference from dataflow response")

    urls = [
        _datastructure_url_from_ref(
            dsd_ref["agency_id"],
            dsd_ref["dsd_id"],
            dsd_ref["version"],
        ),
        _datastructure_url_from_ref_slash(
            dsd_ref["agency_id"],
            dsd_ref["dsd_id"],
            dsd_ref["version"],
        ),
        _datastructure_url_from_ref_slash(
            dsd_ref["agency_id"],
            dsd_ref["dsd_id"],
        ),
    ]
    last_error = None
    for url in urls:
        try:
            xml_text = _get_text_with_retry(
                session,
                url,
                params={"references": "all"},
                accept="application/vnd.sdmx.structure+xml;version=2.1",
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(xml_text, encoding="utf-8")
            return _parse_structure_xml(xml_text)
        except requests.HTTPError as exc:
            last_error = exc
            if exc.response is None or exc.response.status_code != 404:
                raise
    if last_error is not None:
        raise last_error
    raise ValueError("Unable to retrieve datastructure")


def _index_to_code(dim_values, dim_id, index):
    values = dim_values.get(dim_id, [])
    if index < 0 or index >= len(values):
        return None
    return values[index]["code"], values[index]["label"]


def _sdmx_json_to_tidy(payload, dim_rename):
    structure = payload.get("structure", {})
    data = payload.get("dataSets", [{}])[0]
    observations = data.get("observations", {})
    obs_dims = structure.get("dimensions", {}).get("observation", [])

    dim_values = {
        dim["id"]: [
            {"code": v.get("id"), "label": v.get("name")} for v in dim.get("values", [])
        ]
        for dim in obs_dims
    }

    dim_ids = [dim["id"] for dim in obs_dims]

    rows = []
    for key, obs in observations.items():
        indices = [int(i) for i in key.split(":")]
        row = {}
        for dim_id, idx in zip(dim_ids, indices):
            code, label = _index_to_code(dim_values, dim_id, idx)
            output_name = dim_rename.get(dim_id, dim_id.lower())
            row[output_name] = code
            row[f"{output_name}_label"] = label
        row["value"] = obs[0] if isinstance(obs, list) and obs else obs
        rows.append(row)

    return pd.DataFrame(rows)


def _build_key(series_dims, key_template, overrides):
    template_parts = key_template.split(".")
    if len(template_parts) != len(series_dims):
        raise ValueError("Key template length does not match series dimensions")

    key_parts = []
    for dim, template_value in zip(series_dims, template_parts):
        dim_id = dim["id"]
        override_value = overrides.get(dim_id)
        if override_value is None:
            key_parts.append(template_value)
        elif isinstance(override_value, (list, tuple, set)):
            key_parts.append("+".join(override_value))
        else:
            key_parts.append(str(override_value))
    return ".".join(key_parts)


def _resolve_dim_id(dim_values, candidates):
    for candidate in candidates:
        if candidate in dim_values:
            return candidate
    return None


def pull_oecd_table_0720(
    base_url=OECD_BASE_DATA_URL,
    start_period=OECD_START_PERIOD,
    end_period=OECD_END_PERIOD,
    username=OECD_USERNAME,
    password=OECD_PASSWORD,
):
    session = _build_session(username=username, password=password)
    flow_ref, key_template, base_params = _parse_base_url(base_url)
    series_dims, obs_dims, dim_values = fetch_structure(flow_ref, session)

    ref_area_dim = _resolve_dim_id(dim_values, ["REF_AREA", "REFERENCE_AREA", "LOCATION"])
    instrument_dim = _resolve_dim_id(
        dim_values, ["INSTR_ASSET", "FINANCIAL_INSTRUMENT", "INSTRUMENT"]
    )
    maturity_dim = _resolve_dim_id(dim_values, ["MATURITY", "ORIGINAL_MATURITY"])

    if not ref_area_dim or not instrument_dim:
        raise ValueError("Required dimensions (reference area, financial instrument) not found")

    ref_areas = [v["code"] for v in dim_values[ref_area_dim]]

    instrument_codes = {v["code"] for v in dim_values[instrument_dim]}
    instruments = [code for code in TARGET_INSTRUMENTS if code in instrument_codes]

    maturity_codes = []
    if maturity_dim:
        available_maturity = {v["code"] for v in dim_values[maturity_dim]}
        maturity_codes = [m for m in TARGET_MATURITY_CODES if m in available_maturity]

    params = {
        **base_params,
        "startPeriod": _format_period(start_period),
        "endPeriod": _format_period(end_period),
        "dimensionAtObservation": "AllDimensions",
    }

    dim_rename = {
        ref_area_dim: "reference_area",
        "TIME_PERIOD": "time_period",
        instrument_dim: "financial_instrument",
    }
    if maturity_dim:
        dim_rename[maturity_dim] = "original_maturity"
    dim_rename.update(
        {
            "ACCOUNTING_ENTRY": "accounting_entry",
            "UNIT_MEASURE": "unit_of_measure",
            "CURRENCY": "currency_of_denomination",
            "CURRENCY_DENOM": "currency_of_denomination",
            "INSTITUTIONAL_SECTOR": "institutional_sector",
            "SECTOR": "institutional_sector",
        }
    )

    frames = []
    for ref_area in ref_areas:
        time.sleep(0.2)
        for instrument in instruments:
            if instrument == "F3" and maturity_codes:
                instrument_maturities = maturity_codes
            elif maturity_codes:
                instrument_maturities = [maturity_codes[0]] if "T" in maturity_codes else [maturity_codes[0]]
            else:
                instrument_maturities = None

            overrides = {
                ref_area_dim: ref_area,
                instrument_dim: instrument,
            }
            if maturity_dim and instrument_maturities:
                overrides[maturity_dim] = instrument_maturities

            key = _build_key(series_dims, key_template, overrides)
            payload = _get_json_with_retry(
                session,
                _data_url(flow_ref, key),
                params=params,
                accept="application/vnd.sdmx.data+json;version=2.0",
            )
            time.sleep(0.5)

            if not payload.get("dataSets"):
                continue
            df = _sdmx_json_to_tidy(payload, dim_rename)
            if not df.empty:
                frames.append(df)

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "reference_area",
                "time_period",
                "financial_instrument",
                "original_maturity",
                "accounting_entry",
                "unit_of_measure",
                "currency_of_denomination",
                "institutional_sector",
                "value",
            ]
        )

    return result
if __name__ == "__main__":
    df = pull_oecd_table_0720()
    path = Path(DATA_DIR) / OECD_OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
