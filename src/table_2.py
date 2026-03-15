"""table_2.py
============
Reproduces Table 2 (Top Ten Investors by Asset Class, 2020) from
Koijen & Yogo (2020).

Pipeline:  reuses build_data2 / build_data3 from table_1.py
           → aggregate holdings by investor country
           → top 10 per asset class

Run:  python table_2.py
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from table_1 import _read, build_data2, build_data3

OUTPUT_DIR = Path(__file__).parent.parent / "_output"


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

def compute_table2(
    data3: pd.DataFrame,
    countries: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """
    Returns the top-10 investors per asset class for year=2020.

    Output: {type: DataFrame(rank, investor, amount)}
      type 1 = Short-term debt
      type 2 = Long-term debt
      type 3 = Equity
    """
    df = data3[data3["year"] == 2020].copy()

    # Name map: ISO3 → full name; _CR → "Reserves"
    name_map: dict[str, str] = (
        countries[["country", "Name"]]
        .drop_duplicates("country")
        .set_index("country")["Name"]
        .to_dict()
    )
    name_map["_CR"] = "Reserves"

    # Keep only recognised single-country investors + reserves
    known = set(name_map)
    df = df[df["country"].isin(known)].copy()

    inv = (
        df.groupby(["country", "type"])["amount"]
        .sum()
        .reset_index()
    )
    inv["investor"] = inv["country"].map(name_map)

    result: dict[int, pd.DataFrame] = {}
    for tp in [1, 2, 3]:
        top = (
            inv[inv["type"] == tp]
            .sort_values("amount", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        top["rank"] = top.index + 1
        result[tp] = top[["rank", "investor", "amount"]].copy()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_table2(
    top10: dict[int, pd.DataFrame],
    path: str | None = None,
) -> None:
    """Print and save Table 2 in the paper layout."""
    st = top10[1].reset_index(drop=True)
    lt = top10[2].reset_index(drop=True)
    eq = top10[3].reset_index(drop=True)

    SEP = "-" * 90

    lines = [
        "Table 2",
        "Top Ten Investors by Asset Class",
        "",
        f"{'Short-term debt':<30s}  {'Long-term debt':<30s}  {'Equity':<25s}",
        f"{'Investor':<22s}  {'Billion US$':>7s}  "
        f"{'Investor':<22s}  {'Billion US$':>7s}  "
        f"{'Investor':<22s}  {'Billion US$':>7s}",
        SEP,
    ]

    for i in range(10):
        def row(df: pd.DataFrame, i: int) -> str:
            name = str(df.at[i, "investor"]) if i < len(df) else ""
            amt  = df.at[i, "amount"] if i < len(df) else float("nan")
            amt_s = f"{round(amt):>7,d}" if amt == amt else f"{'':>7s}"
            return f"{name:<22s}  {amt_s:>7s}"

        lines.append(f"  {row(st, i)}    {row(lt, i)}    {row(eq, i)}")

    lines += [
        SEP,
        "Note.—The International Monetary Fund (2003–2020a) aggregates foreign exchange",
        "reserves across all foreign central banks for confidentiality. All market values are in",
        "billion US dollars at year-end 2020.",
    ]

    output = "\n".join(lines)
    print("\n" + output)

    dest = Path(path) if path else OUTPUT_DIR / "table_2.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output)
    print(f"\n  [Saved to {dest}]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("table_2.py  —  Koijen & Yogo (2020)")
    print("=" * 60)

    print("\nLoading Countries.dta ...")
    ctry = _read("Countries.dta")

    print("\n[Data2] Amounts outstanding")
    data2 = build_data2(ctry)

    print("\n[Data3] Holdings")
    data3 = build_data3(ctry, data2)

    print("\n[Table 2]")
    top10 = compute_table2(data3, ctry)

    export_table2(top10)


if __name__ == "__main__":
    main()
