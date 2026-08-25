"""Shared grid-alignment logic and per-location update, backed by Azure Blob
Storage instead of the local filesystem.

Same 30-minute-grid approach as the local script: buoys report every ~3-4
minutes, so each 30-minute UTC mark (:00/:30) is filled with the nearest
actual reading (no interpolation).
"""

import io
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from azure.storage.blob import ContainerClient, ContentSettings

from elements_api import fetch_channel, login

BACKFILL_WINDOW = timedelta(days=30)
SAMPLE_INTERVAL = timedelta(minutes=30)
GRID_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

BLOB_CONTAINER = "buoy-data"

LOCATIONS = [
    {
        "name": "Pierce",
        "lat_channel": "2bbd6a30-887f-11eb-9d91-cf7c0f6efbea",
        "lon_channel": "2b6d9c80-887f-11eb-9d91-cf7c0f6efbea",
        "blob_name": "Pierce_buoy_position.csv",
    },
    {
        "name": "Penguins",
        "lat_channel": "921287c0-9c75-11f0-8f61-ffb240084fa4",
        "lon_channel": "91ae20a0-9c75-11f0-8f61-ffb240084fa4",
        "blob_name": "Penguins_buoy_position.csv",
    },
    {
        "name": "Penguins(sat)",
        "lat_channel": "3b9d1460-f7a5-11f0-bd98-21743c6b84e7",
        "lon_channel": "3baca4c0-f7a5-11f0-bd98-21743c6b84e7",
        "blob_name": "Penguins(sat)_buoy_position.csv",
    },
]


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


def fetch_new_positions(
    session: requests.Session, lat_channel: str, lon_channel: str, start: datetime, end: datetime
) -> pd.DataFrame:
    ts_1 = int(start.timestamp() * 1000)
    ts_2 = int(end.timestamp() * 1000)

    lat_df = fetch_channel(lat_channel, session, ts_1, ts_2).rename(columns={"value": "lat"})
    lon_df = fetch_channel(lon_channel, session, ts_1, ts_2).rename(columns={"value": "lon"})

    positions = pd.merge(lat_df, lon_df, on="timestamp", how="inner").sort_values("timestamp")
    return positions[["timestamp", "lat", "lon"]]


def get_container_client() -> ContainerClient:
    conn_str = os.environ["AzureWebJobsStorage"]
    container = ContainerClient.from_connection_string(conn_str, container_name=BLOB_CONTAINER)
    if not container.exists():
        container.create_container()
    return container


def read_existing(container: ContainerClient, blob_name: str) -> pd.DataFrame:
    blob = container.get_blob_client(blob_name)
    if not blob.exists():
        return pd.DataFrame(columns=["timestamp", "lat", "lon"])
    data = blob.download_blob().readall()
    df = pd.read_csv(io.BytesIO(data), parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def write_csv(container: ContainerClient, blob_name: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    container.upload_blob(
        blob_name,
        buf.getvalue(),
        overwrite=True,
        content_settings=ContentSettings(content_type="text/csv"),
    )


def get_start_time(existing: pd.DataFrame) -> datetime:
    """Latest timestamp already stored (exclusive), or one month ago if
    nothing is stored yet."""
    now = datetime.now(timezone.utc)
    if existing.empty:
        return now - BACKFILL_WINDOW
    last_timestamp = existing["timestamp"].max()
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize("UTC")
    return last_timestamp


def update_location(
    session: requests.Session, container: ContainerClient, location: dict, now: datetime
) -> str:
    name = location["name"]
    existing = read_existing(container, location["blob_name"])
    start = get_start_time(existing)

    grid = build_grid(start, now)
    if len(grid) == 0:
        return f"{name}: already up to date."

    raw_positions = fetch_new_positions(session, location["lat_channel"], location["lon_channel"], start, now)
    new_positions = align_to_grid(raw_positions, grid)

    if new_positions.empty:
        return f"{name}: no new data since the last recorded position."

    combined = pd.concat([existing, new_positions], ignore_index=True)
    write_csv(container, location["blob_name"], combined)
    return f"{name}: added {len(new_positions)} new position(s) to blob {location['blob_name']}"


def update_all() -> None:
    now = datetime.now(timezone.utc)
    username = os.environ["ELEMENTS_USERNAME"]
    password = os.environ["ELEMENTS_PASSWORD"]

    session = login(username, password)
    container = get_container_client()

    for location in LOCATIONS:
        logging.info(update_location(session, container, location, now))
