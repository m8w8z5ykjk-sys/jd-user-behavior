import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "results"
df = pd.read_parquet(DATA_DIR / "user_behavior_cleaned.parquet")
df["datetime"] = pd.to_datetime(df["time"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
print(df.head())

print(df["date"].min())
print(df["date"].max())
