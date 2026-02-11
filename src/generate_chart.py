from pathlib import Path
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Paths (repo-safe)
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "_data" / "bis_debt_securities_cleaned.parquet"
OUTPUT_DIR = REPO_ROOT / "_output"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading data from:", DATA_PATH)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)

print("Raw dataframe shape:", df.shape)
print(df.head())


# -------------------------------
# Extract calendar year
# -------------------------------
df["year_annual"] = df["TIME_PERIOD"].astype(str).str.extract(r"(\d{4})")[0].astype(int)

# Drop rows missing OBS_VALUE or REF_SECTOR
df = df.dropna(subset=["OBS_VALUE", "REF_SECTOR", "year_annual"])

# -------------------------------
# Aggregate average by year × sector
# -------------------------------
agg_df = (
    df
    .groupby(["year_annual", "REF_SECTOR"], as_index=False)
    .agg(avg_value=("OBS_VALUE", "mean"))
)

print("Aggregated data preview:")
print(agg_df.head(20))

# -------------------------------
# Plot: line chart
# -------------------------------
fig = px.line(
    agg_df,
    x="year_annual",
    y="avg_value",
    color="REF_SECTOR",
    markers=True,
    title="Average OBS_VALUE by Sector (Annualized)",
    labels={
        "year_annual": "Year",
        "avg_value": "Average OBS_VALUE",
        "REF_SECTOR": "Sector"
    }
)

# -------------------------------
# Save interactive HTML
# -------------------------------
output_path = OUTPUT_DIR / "obs_value_by_sector.html"
fig.write_html(output_path)

print("Saved plot to:", output_path)
