"""test_table_1.py
=================
Pytest suite that runs the full Table 1 replication pipeline and checks that
every market-value figure converges with the Stata ground truth within 10%.

Run:
    pytest tests/test_table_1.py -v
    pytest tests/test_table_1.py -v --tb=short   # compact tracebacks
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ── Make src importable ──────────────────────────────────────────────────────
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from table_1 import _read, build_data2, build_data3, compute_table1  # noqa: E402

# ── Ground truth from Summary0.smcl (Stata log, run Oct 31 2025) ─────────────
# Format: {(name, type): (market, domestic, reserve)}
GROUND_TRUTH = {
    # ── Short-term debt (type=1) ──────────────────────────────────────────
    ("Canada",           1): (520.86499,   0.88971031, 0.06309296),
    ("United States",    1): (5488.576,    0.91697675, 0.04477147),
    ("Austria",          1): (17.544564,   0.1518878,  0.64997096),
    ("Belgium",          1): (121.13743,   0.68868078, 0.11101773),
    ("Finland",          1): (69.073631,   0.59656739, 0.1593831),
    ("France",           1): (627.91601,   0.79377561, 0.11717916),
    ("Germany",          1): (321.26564,   0.42637457, 0.33901101),
    ("Italy",            1): (167.53023,   0.61501126, 0.01001989),
    ("Netherlands",      1): (67.325883,   0.40079716, 0.24551599),
    ("Portugal",         1): (25.702473,   0.59607449, 0.01072783),
    ("Spain",            1): (194.96783,   0.64538563, 0.12182161),
    ("Denmark",          1): (48.440207,   0.58074125, 0.28585838),
    ("Israel",           1): (28.502921,   0.90366263, 0.00993516),
    ("Norway",           1): (29.819894,   0.59959782, 0.07529094),
    ("Sweden",           1): (122.96256,   0.54933686, 0.07447747),
    ("Switzerland",      1): (115.13287,   0.75566907, 0.16463776),
    ("United Kingdom",   1): (340.77378,   0.8406607,  0.03906768),
    ("Australia",        1): (202.25916,   0.89287634, 0.04331996),
    ("Hong Kong",        1): (12.93199,    0.40092487, 0.05704731),
    ("Japan",            1): (1889.2016,   0.74255649, 0.11545027),
    ("New Zealand",      1): (8.661976,    0.71427537, 0.09703344),
    ("Singapore",        1): (41.439288,   0.03678556, 0.39021365),
    ("Greece",           1): (15.951343,   0.70515634, 0.00448934),
    ("Brazil",           1): (399.11267,   0.97703605, 0.00236549),
    ("China",            1): (455.24004,   0.80073967, 0.00839347),
    ("Colombia",         1): (26.728046,   0.99244318, 0.00009351),
    ("Czech Republic",   1): (28.294775,   0.00001767, 0.00081345),
    ("Hungary",          1): (6.8522408,   0.99057422, 0.00142995),
    ("India",            1): (326.2799,    0.97702513, 0.0022583),
    ("Malaysia",         1): (18.030125,   0.82993045, 0.01272398),
    ("Mexico",           1): (94.151187,   0.95831811, 0.00363375),
    ("Philippines",      1): (16.94112,    0.98561119, 0.00169884),
    ("Poland",           1): (42.368184,   0.98968159, 0.00266285),
    ("Russia",           1): (9.3944541,   0.42428848, 0.10007722),
    ("South Africa",     1): (45.297286,   0.9754736,  0.00652409),
    ("South Korea",      1): (308.94048,   0.92401666, 0.03285618),
    ("Thailand",         1): (75.166021,   0.9762702,  0.00292503),
    # ── Long-term debt (type=2) ───────────────────────────────────────────
    ("Canada",           2): (2521.9382,   0.77252854, 0.05397445),
    ("United States",    2): (41070.4,     0.83966507, 0.05430811),
    ("Austria",          2): (558.52926,   0.49030556, 0.08469685),
    ("Belgium",          2): (1151.8208,   0.62128865, 0.06262311),
    ("Finland",          2): (383.56635,   0.428355,   0.09292829),
    ("France",           2): (5064.7096,   0.64522756, 0.07063768),
    ("Germany",          2): (4341.6881,   0.57786282, 0.13194582),
    ("Italy",            2): (3748.9734,   0.78094067, 0.00216319),
    ("Netherlands",      2): (1234.8898,   0.44700371, 0.08091598),
    ("Portugal",         2): (336.35861,   0.69307795, 0.00460898),
    ("Spain",            2): (2620.7663,   0.66107208, 0.04051824),
    ("Denmark",          2): (670.49948,   0.74799987, 0.02061245),
    ("Israel",           2): (508.91266,   0.90074554, 0.0036399),
    ("Norway",           2): (224.27662,   0.19468895, 0.10703137),
    ("Sweden",           2): (428.45743,   0.46089121, 0.05328417),
    ("Switzerland",      2): (907.79166,   0.87506293, 0.03243117),
    ("United Kingdom",   2): (4163.0644,   0.84431259, 0.04113897),
    ("Australia",        2): (1823.9787,   0.71311734, 0.04284855),
    ("Hong Kong",        2): (55.072719,   0.11411331, 0.01869706),
    ("Japan",            2): (13467.385,   0.96351455, 0.01209776),
    ("New Zealand",      2): (124.91184,   0.75661392, 0.02369751),
    ("Singapore",        2): (225.16348,   0.65753157, 0.03179867),
    ("Greece",           2): (133.53581,   0.86131058, 0.00625637),
    ("Brazil",           2): (1433.7464,   0.89786994, 0.00386728),
    ("China",            2): (17358.558,   0.97088496, 0.00029349),
    ("Colombia",         2): (185.19169,   0.8111774,  0.00023663),
    ("Czech Republic",   2): (94.961778,   0.54497106, 0.00177146),
    ("Hungary",          2): (106.87329,   0.78644676, 0.00047104),
    ("India",            2): (2026.7676,   0.96551549, 0.0011527),
    ("Malaysia",         2): (341.513,     0.84395009, 0.00054405),
    ("Mexico",           2): (591.4654,    0.77812939, 0.00340948),
    ("Philippines",      2): (94.359038,   0.7772634,  0.00168488),
    ("Poland",           2): (300.95892,   0.76824334, 0.001106),
    ("Russia",           2): (297.90611,   0.68691503, 0.01205555),
    ("South Africa",     2): (195.57893,   0.7489279,  0.00844827),
    ("South Korea",      2): (2103.0053,   0.91847415, 0.01811158),
    ("Thailand",         2): (338.06352,   0.91646076, 0.00083613),
    # ── Equity (type=3) ──────────────────────────────────────────────────
    ("Canada",           3): (6514.5493,   0.86834543, 0.0015724),
    ("United States",    3): (55622.725,   0.86986009, 0.00405284),
    ("Austria",          3): (282.38465,   0.78586112, 0.00107572),
    ("Belgium",          3): (1738.5613,   0.92878767, 0.00069101),
    ("Finland",          3): (678.38938,   0.78477782, 0.00265598),
    ("France",           3): (10614.035,   0.89116126, 0.00182004),
    ("Germany",          3): (3671.7601,   0.57161293, 0.00436416),
    ("Italy",            3): (2384.1603,   0.88798197, 0.00131274),
    ("Netherlands",      3): (6056.5851,   0.80382937, 0.00126258),
    ("Portugal",         3): (385.90393,   0.91927858, 0.00058559),
    ("Spain",            3): (2229.4918,   0.87511489, 0.00138971),
    ("Denmark",          3): (1705.2843,   0.84648929, 0.00164444),
    ("Israel",           3): (660.40132,   0.85657595, 0.00003),
    ("Norway",           3): (1037.6571,   0.90506672, 0.00067613),
    ("Sweden",           3): (3019.0553,   0.88266907, 0.00113817),
    ("Switzerland",      3): (3746.8108,   0.75412413, 0.0014168),
    ("United Kingdom",   3): (6666.0902,   0.76281618, 0.00274702),
    ("Australia",        3): (1763.2644,   0.73487427, 0.00371392),
    ("Hong Kong",        3): (2454.2083,   0.81310095, 0.00051195),
    ("Japan",            3): (12171.7,     0.86895971, 0.00283606),
    ("New Zealand",      3): (1029.9664,   0.96100333, 0.00039862),
    ("Singapore",        3): (887.09656,   0.82142176, 0.00079769),
    ("Greece",           3): (182.36593,   0.91387333, 3.132e-07),
    ("Brazil",           3): (2398.8035,   0.91165953, 0.00003101),
    ("China",            3): (15002.008,   0.77260056, 0.00012969),
    ("Colombia",         3): (483.26817,   0.98845539, 2.185e-07),
    ("Czech Republic",   3): (106.14188,   0.97831386, 0.00019703),
    ("Hungary",          3): (104.08491,   0.88892267, 0.0),
    ("India",            3): (2354.165,    0.80160129, 9.978e-06),
    ("Malaysia",         3): (447.393,     0.89321075, 0.00020049),
    ("Mexico",           3): (1313.2745,   0.93721348, 0.00037548),
    ("Philippines",      3): (260.52174,   0.87838224, 9.087e-07),
    ("Poland",           3): (254.73986,   0.91286605, 0.00005375),
    ("Russia",           3): (3039.2663,   0.95950118, 0.00002475),
    ("South Africa",     3): (1105.3874,   0.90548921, 0.00015604),
    ("South Korea",      3): (3146.7453,   0.84828777, 0.00013215),
    ("Thailand",         3): (538.39964,   0.86409095, 0.00003966),
}

# ── Tolerance ────────────────────────────────────────────────────────────────
MARKET_THRESHOLD_PCT = 10.0   # |%error| must be below this


# ── Session-scoped fixture: build the table once for all tests ───────────────
@pytest.fixture(scope="session")
def table1() -> pd.DataFrame:
    """Run the full pipeline and return the computed Table 1 DataFrame."""
    ctry = _read("Countries.dta")
    data2 = build_data2(ctry)
    data3 = build_data3(ctry, data2)
    return compute_table1(data3, ctry)


@pytest.fixture(scope="session")
def lookup(table1: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """Build a {(Name, type): row_dict} lookup from the computed table."""
    result = {}
    for _, row in table1.iterrows():
        result[(row["Name"], int(row["type"]))] = row.to_dict()
    return result


# ── Parametrised tests ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, asset_type, gt_market, gt_domestic, gt_reserve",
    [
        (name, asset_type, gt[0], gt[1], gt[2])
        for (name, asset_type), gt in sorted(GROUND_TRUTH.items())
    ],
    ids=[f"{name}_type{t}" for (name, t) in sorted(GROUND_TRUTH.keys())],
)
def test_market_value_within_10pct(
    lookup: dict,
    name: str,
    asset_type: int,
    gt_market: float,
    gt_domestic: float,
    gt_reserve: float,
) -> None:
    """Market value must be within ±10 % of the Stata ground truth."""
    key = (name, asset_type)
    assert key in lookup, f"Country/type not found in computed table: {key}"

    py_market = lookup[key]["market"]
    pct_err = 100.0 * abs(py_market - gt_market) / gt_market

    assert pct_err <= MARKET_THRESHOLD_PCT, (
        f"{name} type={asset_type}: "
        f"computed={py_market:.1f}  gt={gt_market:.1f}  "
        f"|err|={pct_err:.1f}% > {MARKET_THRESHOLD_PCT}%"
    )
