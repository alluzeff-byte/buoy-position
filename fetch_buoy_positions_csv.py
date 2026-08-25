"""
Download buoy Latitude/Longitude from the Vaisala Elements Online API and
keep a running CSV per location, one row every 30 minutes.

- If a location's CSV does not exist yet, this fetches the latest month of
  data and creates it.
- If the file already exists, this reads the last recorded timestamp and
  fetches only newer samples, appending them to the file.

Buoys actually report every ~3-4 minutes, but the API has no documented
resolution/aggregation parameter, so each 30-minute UTC grid mark (:00/:30)
is filled with the nearest actual reading instead (no interpolation - every
stored lat/lon is a genuine reading, just filed under the closest grid mark,
which can be up to ~15 minutes off from when it was actually recorded).
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from elements_api import login, fetch_channel

USERNAME = os.environ["ELEMENTS_USERNAME"]
PASSWORD = os.environ["ELEMENTS_PASSWORD"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKFILL_WINDOW = timedelta(days=30)
SAMPLE_INTERVAL = timedelta(minutes=30)
GRID_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

LOCATIONS = [
    {
        "name": "Pierce",
        "lat_channel": "2bbd6a30-887f-11eb-9d91-cf7c0f6efbea",
        "lon_channel": "2b6d9c80-887f-11eb-9d91-cf7c0f6efbea",
        "csv_path": os.path.join(SCRIPT_DIR, "Pierce_buoy_position.csv"),
    },
    {
        "name": "Penguins",
        "lat_channel": "921287c0-9c75-11f0-8f61-ffb240084fa4",
        "lon_channel": "91ae20a0-9c75-11f0-8f61-ffb240084fa4",
        "csv_path": os.path.join(SCRIPT_DIR, "Penguins_buoy_position.csv"),
    },
]


def get_start_time(csv_path: str) -> datetime:
    """Latest timestamp already in the CSV (exclusive), or one month ago if
    the CSV doesn't exist yet."""
    now = datetime.now(timezone.utc)
    if not os.path.exists(csv_path):
        return now - BACKFILL_WINDOW

    existing = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if existing.empty:
        return now - BACKFILL_WINDOW

    last_timestamp = existing["timestamp"].max()
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize("UTC")
    return last_timestamp


def fetch_new_positions(
    session: requests.Session, lat_channel: str, lon_channel: str, start: datetime, end: datetime
) -> pd.DataFrame:
    ts_1 = int(start.timestamp() * 1000)
    ts_2 = int(end.timestamp() * 1000)

    lat_df = fetch_channel(lat_channel, session, ts_1, ts_2).rename(columns={"value": "lat"})
    lon_df = fetch_channel(lon_channel, session, ts_1, ts_2).rename(columns={"value": "lon"})

    positions = pd.merge(lat_df, lon_df, on="timestamp", how="inner").sort_values("timestamp")
    return positions[["timestamp", "lat", "lon"]]


def floor_to_grid(dt: datetime) -> datetime:
    """Round dt down to the nearest 30-minute UTC mark (:00/:30)."""
    step = SAMPLE_INTERVAL.total_seconds()
    seconds = (dt - GRID_EPOCH).total_seconds()
    return GRID_EPOCH + timedelta(seconds=seconds - (seconds % step))


def build_grid(after: datetime, through: datetime) -> pd.DatetimeIndex:
    """30-minute UTC marks strictly after `after`, up to and including the
    grid mark at or before `through`."""
    first = floor_to_grid(after) + SAMPLE_INTERVAL
    if first > through:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(first, through, freq=SAMPLE_INTERVAL, tz="UTC")


def align_to_grid(positions: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Snap each grid mark to the nearest actual reading (within half the
    grid spacing); grid marks with no reading nearby are dropped."""
    if len(grid) == 0 or positions.empty:
        return positions.iloc[0:0]
    grid_df = pd.DataFrame({"timestamp": grid})
    aligned = pd.merge_asof(
        grid_df,
        positions.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=SAMPLE_INTERVAL / 2,
    )
    return aligned.dropna(subset=["lat", "lon"])


def update_location(session: requests.Session, location: dict, now: datetime) -> None:
    name = location["name"]
    csv_path = location["csv_path"]
    start = get_start_time(csv_path)

    grid = build_grid(start, now)
    if len(grid) == 0:
        print(f"{name}: already up to date.")
        return

    raw_positions = fetch_new_positions(session, location["lat_channel"], location["lon_channel"], start, now)
    new_positions = align_to_grid(raw_positions, grid)

    if new_positions.empty:
        print(f"{name}: no new data since the last recorded position.")
        return

    file_exists = os.path.exists(csv_path)
    new_positions.to_csv(
        csv_path,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )
    print(f"{name}: added {len(new_positions)} new position(s) to {csv_path}")


def update_all() -> None:
    now = datetime.now(timezone.utc)
    session = login(USERNAME, PASSWORD)
    for location in LOCATIONS:
        update_location(session, location, now)


if __name__ == "__main__":
    update_all()
