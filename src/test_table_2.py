"""test_table_2.py
=================
Pytest suite that runs the full Table 2 replication pipeline and checks that
every top-10 investor amount converges with the paper's ground truth within 10%.

Run:
    pytest tests/test_table_2.py -v
    pytest tests/test_table_2.py -v --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ── Make src importable ──────────────────────────────────────────────────────
SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

from table_1_latest import _load_countries, DATA_DIR  # noqa: E402
from table_2_latest import compute_table2_latest       # noqa: E402

# ── Ground truth from Koijen & Yogo (2020), Table 2 ─────────────────────────
# Format: {(investor_name, asset_type): amount_billion_usd}
#   type 1 = Short-term debt
#   type 2 = Long-term debt
#   type 3 = Equity
GROUND_TRUTH: dict[tuple[str, int], float] = {
    # Short-term debt
    ("United States",   1): 5423.0,
    ("Japan",           1): 1444.0,
    ("Reserves",        1): 1025.0,
    ("France",          1):  827.0,
    ("United Kingdom",  1):  496.0,
    ("Canada",          1):  471.0,
    ("China",           1):  440.0,
    ("Brazil",          1):  395.0,
    ("India",           1):  325.0,
    ("South Korea",     1):  301.0,
    # Long-term debt
    ("United States",   2): 38283.0,
    ("China",           2): 17331.0,
    ("Japan",           2): 16206.0,
    ("United Kingdom",  2):  5752.0,
    ("Germany",         2):  5513.0,
    ("France",          2):  5490.0,
    ("Reserves",        2):  4952.0,
    ("Italy",           2):  3721.0,
    ("Canada",          2):  2979.0,
    ("South Korea",     2):  2350.0,
    # Equity
    ("United States",   3): 56324.0,
    ("Japan",           3): 12424.0,
    ("China",           3): 11952.0,
    ("France",          3): 10376.0,
    ("Canada",          3):  7361.0,
    ("United Kingdom",  3):  6800.0,
    ("Netherlands",     3):  5971.0,
    ("Germany",         3):  3393.0,
    ("Switzerland",     3):  3390.0,
    ("Hong Kong",       3):  3240.0,
}

# ── Tolerance ────────────────────────────────────────────────────────────────
AMOUNT_THRESHOLD_PCT = 10.0


# ── Session-scoped fixture: build the table once for all tests ───────────────
@pytest.fixture(scope="session")
def table2() -> dict[int, pd.DataFrame]:
    """Run the full pipeline (year=2020) and return the computed Table 2 dict."""
    countries = _load_countries()
    return compute_table2_latest(2020, countries, DATA_DIR)


@pytest.fixture(scope="session")
def lookup(table2: dict[int, pd.DataFrame]) -> dict[tuple[str, int], float]:
    """Build a {(investor_name, type): amount} lookup from the computed table."""
    result: dict[tuple[str, int], float] = {}
    for tp, df in table2.items():
        for _, row in df.iterrows():
            result[(row["Investor"], int(tp))] = float(row["value_bn"])
    return result


# ── Parametrised tests ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "investor, asset_type, gt_amount",
    [
        (investor, asset_type, gt_amount)
        for (investor, asset_type), gt_amount in sorted(GROUND_TRUTH.items())
    ],
    ids=[f"{inv}_type{t}" for (inv, t) in sorted(GROUND_TRUTH.keys())],
)
def test_amount_within_10pct(
    lookup: dict,
    investor: str,
    asset_type: int,
    gt_amount: float,
) -> None:
    """Investor amount must be within ±10 % of the paper's ground truth."""
    key = (investor, asset_type)
    assert key in lookup, (
        f"Investor/type not found in computed top-10: {key}. "
        f"Available type-{asset_type} entries: "
        f"{[k for k in lookup if k[1] == asset_type]}"
    )

    py_amount = lookup[key]
    pct_err = 100.0 * abs(py_amount - gt_amount) / gt_amount

    assert pct_err <= AMOUNT_THRESHOLD_PCT, (
        f"{investor} type={asset_type}: "
        f"computed={py_amount:.0f}  gt={gt_amount:.0f}  "
        f"|err|={pct_err:.1f}% > {AMOUNT_THRESHOLD_PCT}%"
    )
