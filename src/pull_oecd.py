"""Pull OECD SDMX Table 0720 balance sheet data and save as a tidy Parquet.

This script pulls Table 720 (Non-Consolidated Financial Balance Sheets,
SNA 2008) for general government. It targets annual, end-of-period stocks in
national currency (XDC) for 2003-2020 and a fixed list of instruments and
countries.
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
        "A..CAN+USA+BEL+DNK+FIN+FRA+DEU+ITA+ISR+NLD+NOR+PRT+ESP+SWE+CHE+GBR+"
        "BRA+JPN+COL+CZE+GRC+HUN+MEX+POL+KOR+AUT..S13....LE.F2+F3+F4+F5.."
        "XDC._T.S.V.N.T0720._Z?startPeriod=2003&endPeriod=2020"
        "&dimensionAtObservation=AllDimensions"
    ),
    cast=str,
)

OECD_START_PERIOD = config("OECD_START_PERIOD", default="2003", cast=str)
OECD_END_PERIOD = config("OECD_END_PERIOD", default="2020", cast=str)
OECD_OUTPUT_FILE = config("OECD_OUTPUT_FILE", default="oecd_t720.parquet", cast=str)
OECD_STRUCTURE_CACHE = config(
    "OECD_STRUCTURE_CACHE",
    default=str(Path(DATA_DIR) / "oecd_t720_structure.xml"),
    cast=str,
)
OECD_STRUCTURE_REFRESH = config("OECD_STRUCTURE_REFRESH", default="0", cast=str)

# Target OECD instrument codes for DF_T720R_A
TARGET_INSTRUMENTS = [
    "F2",
    "F3",
    "F4",
    "F5",
]

# Optional aliasing from legacy LF* codes to DF_T720R_A F* codes
INSTRUMENT_ALIASES = {
    "LF3SLINK": "F3",
    "LF3LLINK": "F3",
    "LF51LINK": "F5",
    "LF3LINC": "F3",
    "LF519LINC": "F5",
    "LF5LINC": "F5",
}

# OECD reference areas (ISO3)
TARGET_REFERENCE_AREAS = [
    "CAN",
    "USA",
    "BEL",
    "DNK",
    "FIN",
    "FRA",
    "DEU",
    "ITA",
    "ISR",
    "NLD",
    "NOR",
    "PRT",
    "ESP",
    "SWE",
    "CHE",
    "GBR",
    "BRA",
    "JPN",
    "COL",
    "CZE",
    "GRC",
    "HUN",
    "MEX",
    "POL",
    "KOR",
    "AUT",
]


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


def _normalize_label(text):
    if not text:
        return ""
    return " ".join(str(text).lower().replace("-", " ").split())


def _find_code_by_label(
    dim_values,
    dim_id,
    patterns,
    preferred_codes=None,
    allow_missing=False,
):
    values = dim_values.get(dim_id, [])
    if not values:
        if allow_missing:
            return None
        raise ValueError(f"Dimension '{dim_id}' not found or empty")

    if preferred_codes:
        preferred_set = {code.upper() for code in preferred_codes}
        for value in values:
            if value.get("code", "").upper() in preferred_set:
                return value["code"]

    normalized_patterns = [_normalize_label(pat) for pat in patterns or []]
    for value in values:
        code = value.get("code")
        label = _normalize_label(value.get("label"))
        for pattern in normalized_patterns:
            if pattern and (pattern in label or pattern == str(code).lower()):
                return code

    if allow_missing:
        return None

    sample = ", ".join(
        f"{value.get('code')}:{value.get('label')}" for value in values[:5]
    )
    raise ValueError(
        f"Unable to match '{dim_id}' using patterns {patterns}. Sample: {sample}"
    )


def pull_oecd_table_0720(
    base_url=OECD_BASE_DATA_URL,
    start_period=OECD_START_PERIOD,
    end_period=OECD_END_PERIOD,
    username=OECD_USERNAME,
    password=OECD_PASSWORD,
):
    session = _build_session(username=username, password=password)
    flow_ref, key_template, base_params = _parse_base_url(base_url)
    series_dims, _, dim_values = fetch_structure(flow_ref, session)

    ref_area_dim = _resolve_dim_id(dim_values, ["REF_AREA", "REFERENCE_AREA", "LOCATION"])
    instrument_dim = _resolve_dim_id(
        dim_values, ["INSTR_ASSET", "FINANCIAL_INSTRUMENT", "INSTRUMENT"]
    )
    maturity_dim = _resolve_dim_id(dim_values, ["MATURITY", "ORIGINAL_MATURITY"])
    freq_dim = _resolve_dim_id(dim_values, ["FREQ", "FREQUENCY"])
    adjustment_dim = _resolve_dim_id(dim_values, ["ADJUSTMENT"])
    sector_dim = _resolve_dim_id(dim_values, ["SECTOR", "INSTITUTIONAL_SECTOR"])
    counterpart_area_dim = _resolve_dim_id(dim_values, ["COUNTERPART_AREA"])
    counterpart_sector_dim = _resolve_dim_id(dim_values, ["COUNTERPART_SECTOR"])
    consolidation_dim = _resolve_dim_id(dim_values, ["CONSOLIDATION"])
    transaction_dim = _resolve_dim_id(dim_values, ["TRANSACTION"])
    unit_measure_dim = _resolve_dim_id(dim_values, ["UNIT_MEASURE"])
    currency_denom_dim = _resolve_dim_id(dim_values, ["CURRENCY_DENOM", "CURRENCY"])
    valuation_dim = _resolve_dim_id(dim_values, ["VALUATION"])
    price_base_dim = _resolve_dim_id(dim_values, ["PRICE_BASE"])
    accounting_entry_dim = _resolve_dim_id(dim_values, ["ACCOUNTING_ENTRY"])
    transformation_dim = _resolve_dim_id(dim_values, ["TRANSFORMATION"])
    table_identifier_dim = _resolve_dim_id(dim_values, ["TABLE_IDENTIFIER"])
    debt_breakdown_dim = _resolve_dim_id(dim_values, ["DEBT_BREAKDOWN"])

    if not ref_area_dim or not instrument_dim:
        raise ValueError(
            "Required dimensions (reference area, financial instrument) not found"
        )

    available_ref_areas = {v["code"] for v in dim_values[ref_area_dim]}
    ref_areas = [code for code in TARGET_REFERENCE_AREAS if code in available_ref_areas]
    missing_areas = [code for code in TARGET_REFERENCE_AREAS if code not in available_ref_areas]
    if not ref_areas:
        raise ValueError("None of the requested reference areas were found")
    if missing_areas:
        print(f"Warning: missing reference areas: {', '.join(missing_areas)}")

    available_instruments = {v["code"] for v in dim_values[instrument_dim]}
    instruments = []
    missing_instruments = []
    for code in TARGET_INSTRUMENTS:
        if code in available_instruments:
            instruments.append(code)
            continue
        alias = INSTRUMENT_ALIASES.get(code)
        if alias and alias in available_instruments:
            instruments.append(alias)
            continue
        missing_instruments.append(code)
    if missing_instruments:
        missing = ", ".join(missing_instruments)
        raise ValueError(
            "Requested INSTR_ASSET codes not found in dataflow: "
            f"{missing}"
        )

    freq_code = (
        _find_code_by_label(
            dim_values,
            freq_dim,
            patterns=["annual"],
            preferred_codes=["A"],
        )
        if freq_dim
        else None
    )
    adjustment_code = (
        _find_code_by_label(
            dim_values,
            adjustment_dim,
            patterns=["neither seasonally adjusted", "not seasonally"],
            preferred_codes=["N"],
        )
        if adjustment_dim
        else None
    )
    sector_code = (
        _find_code_by_label(
            dim_values,
            sector_dim,
            patterns=["general government"],
            preferred_codes=["S13"],
        )
        if sector_dim
        else None
    )
    counterpart_area_code = (
        _find_code_by_label(
            dim_values,
            counterpart_area_dim,
            patterns=["world"],
            preferred_codes=["W"],
        )
        if counterpart_area_dim
        else None
    )
    counterpart_sector_code = (
        _find_code_by_label(
            dim_values,
            counterpart_sector_dim,
            patterns=["total economy"],
            preferred_codes=["S1"],
        )
        if counterpart_sector_dim
        else None
    )
    consolidation_code = (
        _find_code_by_label(
            dim_values,
            consolidation_dim,
            patterns=["non consolidated", "non-consolidated"],
            preferred_codes=["N", "NC"],
        )
        if consolidation_dim
        else None
    )
    transaction_code = (
        _find_code_by_label(
            dim_values,
            transaction_dim,
            patterns=["closing balance", "positions", "stocks"],
            preferred_codes=["LE"],
        )
        if transaction_dim
        else None
    )
    unit_measure_code = (
        _find_code_by_label(
            dim_values,
            unit_measure_dim,
            patterns=["national currency"],
            preferred_codes=["XDC"],
        )
        if unit_measure_dim
        else None
    )
    currency_denom_code = (
        _find_code_by_label(
            dim_values,
            currency_denom_dim,
            patterns=["all currencies"],
            preferred_codes=["_T"],
            allow_missing=True,
        )
        if currency_denom_dim
        else None
    )
    valuation_code = (
        _find_code_by_label(
            dim_values,
            valuation_dim,
            patterns=["standard valuation"],
            preferred_codes=["S"],
            allow_missing=True,
        )
        if valuation_dim
        else None
    )
    price_base_code = (
        _find_code_by_label(
            dim_values,
            price_base_dim,
            patterns=["current prices"],
            preferred_codes=["V"],
            allow_missing=True,
        )
        if price_base_dim
        else None
    )
    transformation_code = (
        _find_code_by_label(
            dim_values,
            transformation_dim,
            patterns=["non transformed"],
            preferred_codes=["N"],
            allow_missing=True,
        )
        if transformation_dim
        else None
    )
    table_identifier_code = (
        _find_code_by_label(
            dim_values,
            table_identifier_dim,
            patterns=["table 0720"],
            preferred_codes=["T0720"],
            allow_missing=True,
        )
        if table_identifier_dim
        else None
    )
    maturity_code = (
        _find_code_by_label(
            dim_values,
            maturity_dim,
            patterns=["not applicable"],
            preferred_codes=["_Z"],
            allow_missing=True,
        )
        if maturity_dim
        else None
    )
    debt_breakdown_code = (
        _find_code_by_label(
            dim_values,
            debt_breakdown_dim,
            patterns=["not applicable"],
            preferred_codes=["_Z"],
            allow_missing=True,
        )
        if debt_breakdown_dim
        else None
    )

    params = {
        **base_params,
        "startPeriod": _format_period(start_period),
        "endPeriod": _format_period(end_period),
        "dimensionAtObservation": "AllDimensions",
        "format": "jsondata",
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
            "FREQ": "frequency",
            "ADJUSTMENT": "adjustment",
            "COUNTERPART_AREA": "counterpart_area",
            "COUNTERPART_SECTOR": "counterpart_sector",
            "ACCOUNTING_ENTRY": "accounting_entry",
            "TRANSACTION": "transaction",
            "CONSOLIDATION": "consolidation",
            "UNIT_MEASURE": "unit_of_measure",
            "CURRENCY": "currency_of_denomination",
            "CURRENCY_DENOM": "currency_of_denomination",
            "INSTITUTIONAL_SECTOR": "institutional_sector",
            "SECTOR": "institutional_sector",
            "VALUATION": "valuation",
            "PRICE_BASE": "price_base",
            "TRANSFORMATION": "transformation",
            "TABLE_IDENTIFIER": "table_identifier",
            "DEBT_BREAKDOWN": "debt_breakdown",
        }
    )

    frames = []
    for ref_area in ref_areas:
        time.sleep(0.2)
        for instrument in instruments:
            overrides = {
                ref_area_dim: ref_area,
                instrument_dim: instrument,
            }
            if freq_dim and freq_code:
                overrides[freq_dim] = freq_code
            if adjustment_dim and adjustment_code:
                overrides[adjustment_dim] = adjustment_code
            if sector_dim and sector_code:
                overrides[sector_dim] = sector_code
            if counterpart_area_dim and counterpart_area_code:
                overrides[counterpart_area_dim] = counterpart_area_code
            if counterpart_sector_dim and counterpart_sector_code:
                overrides[counterpart_sector_dim] = counterpart_sector_code
            if consolidation_dim and consolidation_code:
                overrides[consolidation_dim] = consolidation_code
            if transaction_dim and transaction_code:
                overrides[transaction_dim] = transaction_code
            if unit_measure_dim and unit_measure_code:
                overrides[unit_measure_dim] = unit_measure_code
            if currency_denom_dim and currency_denom_code:
                overrides[currency_denom_dim] = currency_denom_code
            if valuation_dim and valuation_code:
                overrides[valuation_dim] = valuation_code
            if price_base_dim and price_base_code:
                overrides[price_base_dim] = price_base_code
            if transformation_dim and transformation_code:
                overrides[transformation_dim] = transformation_code
            if table_identifier_dim and table_identifier_code:
                overrides[table_identifier_dim] = table_identifier_code
            if maturity_dim and maturity_code:
                overrides[maturity_dim] = maturity_code
            if debt_breakdown_dim and debt_breakdown_code:
                overrides[debt_breakdown_dim] = debt_breakdown_code

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
                "accounting_entry",
                "transaction",
                "consolidation",
                "institutional_sector",
                "unit_of_measure",
                "currency_of_denomination",
                "transformation",
                "value",
            ]
        )

    return result


if __name__ == "__main__":
    df = pull_oecd_table_0720()
    path = Path(DATA_DIR) / OECD_OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
