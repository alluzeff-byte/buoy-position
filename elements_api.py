"""Minimal client for the Vaisala Elements Online API - just login + CSV
channel download, with none of the plotting-side dependencies."""

import io

import pandas as pd
import requests

LOGIN_URL = "https://api.elements.vaisala.com/base/auth/login"
DATA_URL = (
    "https://api.elements.vaisala.com/base/data/csv/v2"
    "?l&meta=1&channelIds={channel}&from={ts_1}&to={ts_2}"
)


def login(username: str, password: str) -> requests.Session:
    """Log in to Elements Online; returns a Session holding the auth cookie."""
    session = requests.Session()
    resp = session.post(
        LOGIN_URL,
        json={"email": username, "password": password},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Login failed ({resp.status_code}): {resp.text}")
    return session


def fetch_channel(channel_id: str, session: requests.Session, ts_1: int, ts_2: int) -> pd.DataFrame:
    """Download one channel's CSV export and return it as a DataFrame."""
    url = DATA_URL.format(channel=channel_id, ts_1=ts_1, ts_2=ts_2)
    resp = session.get(url, timeout=60)
    if not resp.ok:
        raise RuntimeError(
            f"Data request failed for channel {channel_id} "
            f"({resp.status_code}): {resp.text}"
        )

    # meta=1 prepends a metadata block; the real header starts at the
    # "Timestamp (UTC),<channel-id>" line.
    lines = resp.text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Timestamp"))
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))

    time_col, value_col = df.columns[0], df.columns[1]
    df = df[[time_col, value_col]].rename(columns={time_col: "timestamp", value_col: "value"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
