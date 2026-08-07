import os
import re
import pandas as pd
from datetime import datetime, timezone

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE_PQ = os.path.join(BASE_DIR, "output", "orm_long.parquet")

SENSOR_DEPTH_MAP = {
    "A1": 0.5,
    "A2": 1.0,
    "A3": 2.0,
    "A4": 3.0,
    "A5": 4.0,
}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE_PQ), exist_ok=True)


def get_data_files():
    return sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])


def parse_file(filepath):
    records = []

    with open(filepath, "r") as f:
        for line in f:

            # Only process O3 records
            if ",O3," not in line:
                continue

            # Extract timestamp at start of line
            match_time = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
                line
            )
            if not match_time:
                continue

            timestamp = datetime.strptime(
                match_time.group(1),
                "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)

            # Extract temperatures A1_TEMP_C ... A5_TEMP_C
            for sensor, depth in SENSOR_DEPTH_MAP.items():

                pattern = rf"{sensor}_TEMP_C=(-?\d+(?:\.\d+)?)"
                match_val = re.search(pattern, line)

                if match_val:
                    records.append({
                        "datetime": timestamp,
                        "depth": depth,
                        "observation": float(match_val.group(1)),
                    })

    return records



def process_files(files):
    all_records = []

    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"Parsing {f}")
        all_records.extend(parse_file(path))

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.drop_duplicates(subset=["datetime", "depth", "observation"])
    return df


def build_daily_12utc(df):
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    daily_frames = []
    for depth, group in df.groupby("depth", sort=False):
        series = group.set_index("datetime")["observation"].sort_index()
        if series.index.duplicated().any():
            series = series.groupby(level=0).mean()

        start = series.index.min().floor("D") #+ pd.Timedelta(hours=12)
        end = series.index.max().floor("D")   #+ pd.Timedelta(hours=12)
        target_index = pd.date_range(start=start, end=end, freq="D", tz="UTC")

        interp = series.reindex(series.index.union(target_index)).interpolate(method="time").reindex(target_index)
        interp = interp.dropna()
        if interp.empty:
            continue

        frame = interp.reset_index().rename(columns={"index": "datetime"})
        frame["site_id"] = "ORMS"
        frame["depth"] = depth
        frame["variable"] = "temperature"
        daily_frames.append(frame)

    if not daily_frames:
        return pd.DataFrame(columns=["datetime", "site_id", "depth", "observation", "variable"])

    result = pd.concat(daily_frames, ignore_index=True, sort=False)
    result = result.sort_values(["datetime", "depth"]).reset_index(drop=True)
    result["datetime"] = pd.to_datetime(result["datetime"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    return result[["datetime", "site_id", "depth", "observation", "variable"]]


if __name__ == "__main__":
    local_files = get_data_files()

    if not local_files:
        print("No local data files found in", DATA_DIR)
    else:
        df_raw = process_files(local_files)
        df_daily = build_daily_12utc(df_raw)
        df_daily.to_parquet(OUTPUT_FILE_PQ, index=False)
        print(f"Saved {len(df_daily)} rows to {OUTPUT_FILE_PQ}")
