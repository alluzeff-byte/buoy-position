# Buoy position

Tracks the GPS position history of three moored buoys (Pierce, Penguins,
Penguins(sat)) and renders it as an interactive Plotly page.

**Live page:** published via GitHub Pages from [`docs/index.html`](docs/index.html).

## How it fits together

- **`azure-function/`** - an Azure Function (timer trigger, every 30 min) that
  logs into the Vaisala Elements Online API, downloads each buoy's lat/lon,
  and appends new 30-minute-grid-aligned rows to a CSV per buoy in Azure Blob
  Storage (container `buoy-data`).
- **`fetch_buoy_position.py`** - reads those CSVs from Blob Storage (via a
  read-only SAS token) and renders the tabbed Plotly page, writing it to both
  `buoy_position.html` (local preview) and `docs/index.html` (published via
  GitHub Pages). Run it, then `git add docs && git commit && git push` to
  update the live page.
- **`fetch_buoy_positions_csv.py`** - a local equivalent of the Azure
  Function, for downloading straight to local CSV files instead of Blob
  Storage.
- **`elements_api.py`** - shared Elements Online API client (login + CSV
  channel download) used by the two scripts above.

## Setup

```
pip install -r requirements.txt
```

Required environment variables:

| Variable | Used by | Purpose |
|---|---|---|
| `BUOY_BLOB_SAS_TOKEN` | `fetch_buoy_position.py` | Read-only, container-scoped SAS token for the `buoy-data` Blob container. |
| `ELEMENTS_USERNAME` / `ELEMENTS_PASSWORD` | `fetch_buoy_positions_csv.py` | Vaisala Elements Online login. |

`azure-function/` deploys separately (`func azure functionapp publish`); its
credentials live in the Function App's application settings in Azure, not in
this repo (`azure-function/local.settings.json` is local-only and gitignored).
