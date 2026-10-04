from scripts.ingestUnifiedMatrix import parse_cash_and_shorts
from datetime import datetime
import polars as pl

file = r"offline_data_cache/master_archives/nse_cash_18092026.csv"

df = parse_cash_and_shorts(file)

print("Rows:", df.height)
print("Columns:", df.columns)

print(
    df.filter(
        (pl.col("Ticker") == "RELIANCE")
        & (pl.col("ReportDate") == datetime(2026, 9, 18).date())
    )
)
