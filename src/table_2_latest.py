"""table_2_latest.py
===================
Produces Table 2 (Top Ten Investors by Asset Class) for the latest available
year using the IMF PIP bilateral parquet.

Reuses compute_table2 / export_table2 logic from table_2.py, but reads from
the API parquet instead of the dataverse .dta files.

Run:
    python table_2_latest.py
    python table_2_latest.py --year 2023
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from table_1_latest import (
    DATA_DIR,
    OUTPUT_DIR,
    IMF_ASSET_CLASS_TO_TYPE,
    _load_countries,
    detect_latest_year,
    build_amounts_latest,
)

# ─────────────────────────────────────────────────────────────────────────────
# Country name lookup
# ─────────────────────────────────────────────────────────────────────────────

def _country_names(countries: pd.DataFrame) -> dict[str, str]:
    """Return {ISO3_code: full_name} mapping from countries DataFrame."""
    if "Name" in countries.columns and "Counterpart" in countries.columns:
        return dict(zip(
            countries["Counterpart"].astype(str).str.upper(),
            countries["Name"].astype(str),
        ))
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────────────────────────────────────

def compute_table2_latest(
    year: int,
    countries: pd.DataFrame,
    data_dir: Path = DATA_DIR,
) -> dict[int, pd.DataFrame]:
    """
    Build top-10 investors per asset class for a given year.

    Total holdings = domestic (own) holdings + cross-border (CPIS) holdings.
    Domestic holdings are imputed as: amounts_outstanding − Σ foreign_held.

    Returns: {type: DataFrame(rank, investor_name, value_bn)}
    """
    path = data_dir / "pip_bilateral_positions.parquet"
    if not path.exists():
        print(f"  IMF bilateral parquet not found at {path}")
        return {}

    df = pd.read_parquet(path)

    investor_col    = next((c for c in ["COUNTRY", "country", "investor"] if c in df.columns), None)
    counterpart_col = next((c for c in ["COUNTERPART_COUNTRY", "counterpart_country",
                                         "COUNTERPART", "counterpart"] if c in df.columns), None)
    time_col        = next((c for c in ["TIME_PERIOD", "time_period", "year"] if c in df.columns), None)
    asset_col       = next((c for c in ["asset_class", "ASSET_CLASS"] if c in df.columns), None)
    val_col         = next((c for c in ["value", "VALUE_USD", "value_usd"] if c in df.columns), None)

    if any(c is None for c in [investor_col, time_col, val_col]):
        print(f"  Missing columns in bilateral parquet. Found: {list(df.columns)}")
        return {}

    df["investor"]    = df[investor_col].astype(str).str.upper()
    df["counterpart"] = df[counterpart_col].astype(str).str.upper() if counterpart_col else df["investor"]
    df["year"]        = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    df["value"]       = pd.to_numeric(df[val_col], errors="coerce") / 1_000_000_000.0  # raw USD → billions

    if asset_col is not None:
        df["type"] = df[asset_col].map(IMF_ASSET_CLASS_TO_TYPE)
    else:
        type_col = next((c for c in ["type", "TYPE"] if c in df.columns), None)
        df["type"] = pd.to_numeric(df[type_col], errors="coerce") if type_col else None

    df = df[df["year"] == year].dropna(subset=["investor", "type", "value"]).copy()

    # ── Cross-border: investor's foreign holdings (sum over all issuers) ──────
    foreign_invested = (
        df.groupby(["investor", "type"])["value"]
          .sum()
          .reset_index()
          .rename(columns={"value": "foreign_out"})
    )

    # ── Own holdings: imputed from amounts outstanding ────────────────────────
    # For each issuer country: own = max(amounts - Σ foreign_inflows, ε)
    amounts = build_amounts_latest(countries, year, data_dir)
    amounts_y = amounts[amounts["year"] == year][["Counterpart", "type", "outstand"]].copy()

    foreign_in = (
        df[df["investor"] != df["counterpart"]]
        .groupby(["counterpart", "type"])["value"]
        .sum()
        .reset_index()
        .rename(columns={"counterpart": "Counterpart", "value": "foreign_in"})
    )

    own = amounts_y.merge(foreign_in, on=["Counterpart", "type"], how="left")
    own["foreign_in"] = own["foreign_in"].fillna(0.0)
    own["own"] = (own["outstand"] - own["foreign_in"]).clip(lower=0.0)
    own = own.rename(columns={"Counterpart": "investor"})

    # ── Combine: total = own + foreign invested abroad ────────────────────────
    total = foreign_invested.merge(
        own[["investor", "type", "own"]],
        on=["investor", "type"],
        how="outer",
    )
    total["foreign_out"] = total["foreign_out"].fillna(0.0)
    total["own"]         = total["own"].fillna(0.0)
    total["value"]       = total["foreign_out"] + total["own"]

    name_map = _country_names(countries)
    total["investor_name"] = total["investor"].map(name_map).fillna(total["investor"])

    result: dict[int, pd.DataFrame] = {}
    for tp in [1, 2, 3]:
        sub = (
            total[total["type"] == tp]
            .sort_values("value", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        sub["rank"] = sub.index + 1
        result[tp] = sub[["rank", "investor_name", "value"]].rename(
            columns={"investor_name": "Investor", "value": "value_bn"}
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_table2_latest(top10: dict[int, pd.DataFrame], year: int,
                         path: str | None = None) -> None:
    """Print and save Table 2 in paper format."""
    TYPE_LABEL = {1: "Short-term debt", 2: "Long-term debt", 3: "Equity"}

    lines: list[str] = [
        f"Table 2 (Latest Data: {year})",
        f"Top Ten Investors by Asset Class, {year}",
        f"(Source: IMF PIP bilateral parquet)",
        "",
        f"{'Short-term debt':<30s}  {'Long-term debt':<30s}  {'Equity':<30s}",
        "-" * 95,
    ]

    max_rows = max((len(v) for v in top10.values()), default=0)
    for i in range(max_rows):
        row_parts: list[str] = []
        for tp in [1, 2, 3]:
            df = top10.get(tp, pd.DataFrame())
            if i < len(df):
                name = str(df.iloc[i]["Investor"])[:22]
                val  = df.iloc[i]["value_bn"]
                row_parts.append(f"{name:<22s}  {val:>6,.0f}")
            else:
                row_parts.append(f"{'':22s}  {'':>6s}")
        lines.append("  ".join(row_parts))

    lines += [
        "-" * 95,
        "Note.— Values in billion US dollars. Source: IMF PIP bilateral parquet.",
        "Own holdings not included (foreign holdings only from CPIS reporters).",
    ]

    output = "\n".join(lines)
    print("\n" + output)

    dest = Path(path) if path else OUTPUT_DIR / f"table_2_{year}.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output)
    print(f"\n  [Saved to {dest}]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(year: int | None = None) -> None:
    print("=" * 60)
    print("table_2_latest.py  —  Top 10 investors, latest data")
    print("=" * 60)

    print("\nLoading country metadata ...")
    countries = _load_countries()

    if year is None:
        print("\nDetecting latest year with complete data ...")
        year = detect_latest_year(DATA_DIR)

    print(f"\n[Table 2] Year = {year}")
    top10 = compute_table2_latest(year, countries, DATA_DIR)

    if not top10:
        print("No data available — run  doit pull  first.")
        return

    export_table2_latest(top10, year)

    # Cross-check 2020 if data exists and differs from latest
    if year != 2020:
        bilat_path = DATA_DIR / "pip_bilateral_positions.parquet"
        if bilat_path.exists():
            df_check = pd.read_parquet(bilat_path)
            time_col = next((c for c in ["TIME_PERIOD", "time_period", "year"]
                             if c in df_check.columns), None)
            if time_col:
                has_2020 = 2020 in pd.to_numeric(
                    df_check[time_col], errors="coerce"
                ).dropna().astype(int).values
                if has_2020:
                    print("\n[Table 2 — 2020 cross-check]")
                    top10_2020 = compute_table2_latest(2020, countries, DATA_DIR)
                    export_table2_latest(top10_2020, 2020)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Table 2 from latest API data")
    parser.add_argument("--year", type=int, default=None,
                        help="Year to compute (default: auto-detect latest)")
    args = parser.parse_args()
    main(year=args.year)
