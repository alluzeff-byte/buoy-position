"""
Serve a live-updating, tabbed plot of buoy position history, one tab per
location (Pierce, Penguins). Position data is read from the per-location CSV
files in Azure Blob Storage (container "buoy-data" on storage account
"stbuoypos062535"), which are kept fresh by a separate Azure Function running
every 30 minutes - this script does not call the Elements Online API itself.
"""

import io
import json
import math
import os
import urllib.parse
import webbrowser
from datetime import datetime, timedelta, timezone

import truststore

truststore.inject_into_ssl()  # trust the Windows/OS cert store (needed behind corporate TLS-inspecting proxies)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

from elements_api import fetch_channel, login

# --- Elements Online (wave conditions only - position data comes from Blob
# Storage, see below) ------------------------------------------------------
ELEMENTS_USERNAME = os.environ.get("ELEMENTS_USERNAME")
ELEMENTS_PASSWORD = os.environ.get("ELEMENTS_PASSWORD")

# Display spec for wave-condition fields: (key in "wave_channels", label,
# unit suffix, decimal places, row). Row 1 is the top row, row 2 sits
# directly above "Data as of" - a location only shows the rows/fields it
# has channel IDs for.
WAVE_FIELD_SPECS = [
    ("cs", "CS", "m/s", 2, 1),
    ("cd", "CD", "°", 0, 1),
    ("hs", "Hs", "m", 2, 2),
    ("tp", "Tp", "s", 1, 2),
    ("t2", "T2", "s", 1, 2),
    ("mdir", "Mdir", "°", 0, 2),
    ("pdir", "Pdir", "°", 0, 2),
]

# --- Azure Blob Storage -------------------------------------------------
# Read-only, container-scoped SAS token (expires 2027-08-25). Least-privilege
# on purpose: it can only read blobs in the "buoy-data" container, nothing
# else on the storage account. Set via env var, never hardcoded here, so this
# file is safe to publish (e.g. to a public GitHub repo).
STORAGE_ACCOUNT = "stbuoypos062535"
BLOB_CONTAINER = "buoy-data"
BLOB_SAS_TOKEN = os.environ["BUOY_BLOB_SAS_TOKEN"]


def blob_url(blob_name: str) -> str:
    return f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/{BLOB_CONTAINER}/{blob_name}?{BLOB_SAS_TOKEN}"


# --- Locations -----------------------------------------------------------
# "anchor" is the mooring's fixed anchor position + excursion radius, drawn
# as a marker/circle for reference - purely visual, it does not filter what
# gets plotted, so a buoy currently outside its circle still shows up (worth
# noticing, not worth hiding). Set to None for locations with no known
# anchor geometry - the plot then shows just the trajectory and
# latest-position marker.
# "wave_channels" (optional) are Elements Online channel IDs for live wave
# conditions, fetched fresh at page-generation time (unlike position, these
# aren't piped through Blob Storage) and shown as Hs/Tp fields above the
# "Data as of" caption. Omit or set to None where not available.
LOCATIONS = [
    {
        "key": "pierce",
        "label": "Pierce",
        "blob_name": "Pierce_buoy_position.csv",
        "anchor": {"lat": 57.15614, "lon": 2.26915, "radius_m": 105.0},
        "wave_channels": {
            "hs": "2bdb5270-887f-11eb-9d91-cf7c0f6efbea",
            "tp": "2baca150-887f-11eb-9d91-cf7c0f6efbea",
            "t2": "2be2cc80-887f-11eb-9d91-cf7c0f6efbea",
            "mdir": "2b57ca90-887f-11eb-9d91-cf7c0f6efbea",
            "pdir": "2b7e1740-887f-11eb-9d91-cf7c0f6efbea",
        },
    },
    {
        "key": "penguins",
        "label": "Penguins",
        "blob_name": "Penguins_buoy_position.csv",
        "anchor": {"lat": 61.5769, "lon": 1.5259, "radius_m": 330.0},
        "wave_channels": {
            "hs": "91fb2f30-9c75-11f0-8f61-ffb240084fa4",
            "tp": "909adbe0-9c75-11f0-8f61-ffb240084fa4",
            "t2": "90e15ac0-9c75-11f0-8f61-ffb240084fa4",
            "mdir": "646344f0-a458-11f0-89c3-2ba086547dfc",
            "pdir": "644b5020-a458-11f0-89c3-2ba086547dfc",
            "cs": "91428d90-9c75-11f0-8f61-ffb240084fa4",
            "cd": "92031e70-9c75-11f0-8f61-ffb240084fa4",
        },
    },
    {
        "key": "penguins-sat",
        "label": "Penguins(sat)",
        "blob_name": "Penguins(sat)_buoy_position.csv",
        "wave_channels": {
            "hs": "3b260fa0-f7a5-11f0-bd98-21743c6b84e7",
            "tp": "3b383810-f7a5-11f0-bd98-21743c6b84e7",
            "t2": "3b4afcc0-f7a5-11f0-bd98-21743c6b84e7",
            "pdir": "3b7f2c20-f7a5-11f0-bd98-21743c6b84e7",
            "cs": "3b6c8e80-f7a5-11f0-bd98-21743c6b84e7",
            "cd": "3b60f5c0-f7a5-11f0-bd98-21743c6b84e7",
        },
        "anchor": {"lat": 61.5769, "lon": 1.5259, "radius_m": 330.0},
    },
]

# WGS84 ellipsoid parameters (used by the Vincenty geodesic formulas below).
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_B = (1 - WGS84_F) * WGS84_A

# --- Page styling --------------------------------------------------------
ANCHOR_COLOR = "#7b7fe6"
LATEST_COLOR = "#1f9d55"
POINTS_COLOR = "#d92b2b"
CIRCLE_COLOR = "#7b7fe6"
TRAJECTORY_MARKER_SIZE = 5  # same size in the plot and in the legend swatch
LATEST_MARKER_SIZE = 19

# --- Trajectory window dropdown -------------------------------------------
WINDOW_OPTIONS = [
    ("30 days", timedelta(days=30)),
    ("14 days", timedelta(days=14)),
    ("7 days", timedelta(days=7)),
    ("3 days", timedelta(days=3)),
    ("2 days", timedelta(days=2)),
    ("1 day", timedelta(days=1)),
    ("12 hours", timedelta(hours=12)),
    ("6 hours", timedelta(hours=6)),
    ("3 hours", timedelta(hours=3)),
]
DEFAULT_WINDOW_LABEL = "3 days"
WINDOW_LABELS = [label for label, _ in WINDOW_OPTIONS]

# Trace index of the trajectory scatter in each location's figure, used by
# the window dropdown's restyle buttons.
TRAJECTORY_TRACE_INDEX = 0

# --- Favicon: stylized oceanographic buoy (yellow sphere, black instrument
# band, whip antenna) - a real yellow spherical mooring buoy like the ones
# this page tracks. Inline SVG data URI so no separate asset file is needed.
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <radialGradient id="g" cx="35%" cy="30%" r="75%">
      <stop offset="0%" stop-color="#fff67a"/>
      <stop offset="55%" stop-color="#ffd400"/>
      <stop offset="100%" stop-color="#e2a600"/>
    </radialGradient>
    <clipPath id="c"><circle cx="32" cy="36" r="24"/></clipPath>
  </defs>
  <circle cx="32" cy="36" r="24" fill="url(#g)"/>
  <g clip-path="url(#c)">
    <ellipse cx="32" cy="20" rx="26" ry="9" fill="#1a1a1a"/>
  </g>
  <circle cx="32" cy="36" r="24" fill="none" stroke="#c98f00" stroke-width="1"/>
  <line x1="32" y1="16" x2="32" y2="3" stroke="#cfd8dc" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="32" cy="3" r="2.2" fill="#9aa5ab"/>
</svg>"""
FAVICON_DATA_URI = "data:image/svg+xml," + urllib.parse.quote(FAVICON_SVG)

# --- Figure sizing -----------------------------------------------------
FIGURE_WIDTH = 1075
FIGURE_HEIGHT = 750

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="icon" type="image/svg+xml" href="{favicon_data_uri}">
<style>
  :root {{
    --bg: #eef1ea;
    --card-bg: #ffffff;
    --border: #d8ddd3;
    --text: #1f2320;
    --muted: #6b716a;
    --accent: #7b7fe6;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    padding: 0;
    background-color: var(--bg);
    background-image:
      linear-gradient(rgba(0, 0, 0, 0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 0, 0, 0.045) 1px, transparent 1px);
    background-size: 28px 28px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--text);
    min-height: 100vh;
  }}
  .wrap {{
    max-width: 1165px;
    margin: 0 auto;
    padding: 8px 24px 32px;
  }}
  .header-row {{
    position: relative;
    display: flex;
    align-items: center;
    min-height: 40px;
    margin: 8px 0 10px;
  }}
  h1 {{
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 700;
    font-size: 24px;
    margin: 0;
    white-space: nowrap;
  }}
  .tabs {{
    display: flex;
    justify-content: flex-start;
    gap: 8px;
  }}
  .tab-btn {{
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 22px;
    font-size: 14px;
    font-weight: 600;
    color: var(--muted);
    cursor: pointer;
  }}
  .tab-btn.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
  }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 12px 32px rgba(0, 0, 0, 0.08);
    padding: 12px;
  }}
  .chart {{
    position: relative;
    width: {figure_width}px;
    margin: 0 auto;
  }}
  .side-controls {{
    position: absolute;
    right: 8px;
    bottom: 8px;
    text-align: right;
  }}
  .last-updated {{
    font-size: 12px;
    color: var(--muted);
  }}
  .wave-conditions {{
    display: flex;
    justify-content: flex-end;
    gap: 14px;
    margin-bottom: 4px;
  }}
  .wave-field {{
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
  }}
  .wave-field .wave-label {{
    color: var(--muted);
  }}
  .window-controls {{
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    align-items: center;
    gap: 24px;
  }}
  .pca-row {{
    position: absolute;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .pca-stats {{
    font-size: 12px;
    color: var(--muted);
    white-space: nowrap;
  }}
  .custom-range {{
    position: absolute;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }}
  .custom-range:not([hidden]) {{
    display: flex;
  }}
  .custom-range-field {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
  }}
  .custom-range-field input[type="datetime-local"] {{
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 6px;
    font-size: 12px;
    color: var(--text);
    background: #ffffff;
  }}
  .toggle-label {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  .toggle-label input {{
    accent-color: var(--accent);
    cursor: pointer;
    flex-shrink: 0;
    margin: 0;
  }}
  .window-select {{
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 13px;
    color: var(--text);
    background: #ffffff;
  }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="header-row">
      <div class="tabs">
        {tab_buttons}
      </div>
      <h1 id="page-title">{initial_title}</h1>
    </div>
    {tab_panels}
  </div>
  <script>
    window.__windowData = {window_data_json};
    window.__latestData = {latest_data_json};
    window.__customRanges = {{}};

    // Render each "Data as of" timestamp in the viewer's own local time zone
    // (the underlying value is UTC, embedded in data-timestamp).
    document.querySelectorAll(".last-updated").forEach(function (el) {{
      var iso = el.dataset.timestamp;
      if (!iso) return;
      var formatted = new Date(iso).toLocaleString(undefined, {{
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        timeZoneName: "short",
      }});
      el.textContent = "Data as of: " + formatted;
    }});

    // Park the window-controls box just below the Plotly legend, whatever
    // height that legend actually renders at (varies with trace names/count) -
    // avoids a fixed pixel guess that can overlap it.
    function positionWindowControls(chartEl) {{
      var gd = chartEl.querySelector(".plotly-graph-div");
      var controls = chartEl.querySelector(".window-controls");
      if (!gd || !controls) return;
      var legend = gd.querySelector(".legend");
      if (!legend) return;
      var chartRect = chartEl.getBoundingClientRect();
      var legendRect = legend.getBoundingClientRect();
      if (legendRect.height === 0) return; // chart currently hidden (inactive tab)
      controls.style.top = (legendRect.bottom - chartRect.top + 28) + "px";
    }}

    // Pin the PCA row's left edge to the Track checkbox's own left edge (not
    // the row container's edge, which sits a few pixels further left than
    // the checkbox itself due to the browser's default input margin) and
    // its top edge just below the controls row - flex alignment can't
    // express "match another element's edge", so this is computed in JS.
    function positionPcaRow(chartEl) {{
      var controls = chartEl.querySelector(".window-controls");
      var trackCheckbox = chartEl.querySelector('input[id^="track-"]');
      var pcaRow = chartEl.querySelector(".pca-row");
      var pcaCheckbox = chartEl.querySelector('input[id^="pca-"]');
      if (!controls || !trackCheckbox || !pcaRow || !pcaCheckbox) return null;
      var chartRect = chartEl.getBoundingClientRect();
      var controlsRect = controls.getBoundingClientRect();
      var trackRect = trackCheckbox.getBoundingClientRect();
      if (controlsRect.height === 0) return null; // chart currently hidden (inactive tab)
      var left = trackRect.left - chartRect.left;
      var top = controlsRect.bottom - chartRect.top + 8;
      pcaRow.style.left = left + "px";
      pcaRow.style.top = top + "px";
      // Same default-input-margin issue as Track itself: the PCA checkbox
      // (first child of its own row, like Track is of its row) sits a few
      // pixels right of the row's own edge. One correction pass fixes it.
      var residual = trackRect.left - pcaCheckbox.getBoundingClientRect().left;
      if (residual !== 0) {{
        pcaRow.style.left = (left + residual) + "px";
      }}
      return pcaRow.getBoundingClientRect();
    }}

    // Custom date row goes directly below the PCA row (same left edge),
    // so it shifts down automatically when the PCA row's own height changes.
    function positionCustomRange(chartEl) {{
      var trackCheckbox = chartEl.querySelector('input[id^="track-"]');
      var pcaRow = chartEl.querySelector(".pca-row");
      var customRange = chartEl.querySelector(".custom-range");
      if (!trackCheckbox || !pcaRow || !customRange) return;
      var chartRect = chartEl.getBoundingClientRect();
      var trackRect = trackCheckbox.getBoundingClientRect();
      var pcaRect = pcaRow.getBoundingClientRect();
      if (pcaRect.height === 0) return; // chart currently hidden (inactive tab)
      customRange.style.left = (trackRect.left - chartRect.left) + "px";
      customRange.style.top = (pcaRect.bottom - chartRect.top + 8) + "px";
    }}

    function positionControls(chartEl) {{
      positionWindowControls(chartEl);
      positionPcaRow(chartEl);
      positionCustomRange(chartEl);
    }}

    document.querySelectorAll(".chart").forEach(function (chartEl) {{
      positionControls(chartEl);
    }});

    document.querySelectorAll(".tab-btn").forEach(function (btn) {{
      btn.addEventListener("click", function () {{
        var key = btn.dataset.tab;
        document.querySelectorAll(".tab-btn").forEach(function (b) {{
          b.classList.toggle("active", b === btn);
        }});
        document.querySelectorAll(".tab-panel").forEach(function (panel) {{
          panel.classList.toggle("active", panel.id === "tab-" + key);
        }});
        document.getElementById("page-title").textContent = btn.dataset.title;

        var activePanel = document.getElementById("tab-" + key);
        var chartEl = activePanel && activePanel.querySelector(".chart");
        if (chartEl) positionControls(chartEl);
      }});
    }});

    // Current window's points plus the latest buoy position appended, so the
    // Track line (and the Direction arrow's tail) always reach the current
    // window, not just the trajectory markers.
    function currentWindow(key) {{
      var select = document.getElementById("window-select-" + key);
      if (select.value === "Custom") {{
        return window.__customRanges[key] || {{ x: [], y: [], customdata: [] }};
      }}
      return window.__windowData[key][select.value];
    }}

    function trackXY(key) {{
      var w = currentWindow(key);
      var latest = window.__latestData[key];
      return {{ x: w.x.concat([latest.lon]), y: w.y.concat([latest.lat]) }};
    }}

    function updateDirectionArrow(gd, key, annIdx) {{
      var w = currentWindow(key);
      var latest = window.__latestData[key];
      var earliestLon = w.x.length ? w.x[0] : latest.lon;
      var earliestLat = w.y.length ? w.y[0] : latest.lat;
      var update = {{}};
      update["annotations[" + annIdx + "].ax"] = earliestLon;
      update["annotations[" + annIdx + "].ay"] = earliestLat;
      update["annotations[" + annIdx + "].x"] = latest.lon;
      update["annotations[" + annIdx + "].y"] = latest.lat;
      Plotly.relayout(gd, update);
    }}

    // Dominant axis of spatial spread (first principal component) within
    // the current window - mirrors compute_pca_axis() in Python, used for
    // the initial render; this is the client-side version used whenever the
    // window selection changes. Longitude is corrected for cos(latitude)
    // before computing covariance, so the axis direction reflects true
    // physical distance, not raw degrees.
    function computePcaAxis(key) {{
      var w = currentWindow(key);
      var n = w.x.length;
      if (n < 2) {{
        return {{
          x: [], y: [], centroidX: null, centroidY: null,
          bearing1: null, bearing2: null, variancePct: null,
        }};
      }}

      var lat0 = 0, lon0 = 0;
      for (var i = 0; i < n; i++) {{ lon0 += w.x[i]; lat0 += w.y[i]; }}
      lon0 /= n;
      lat0 /= n;

      var mPerDegLat = 111320;
      var mPerDegLon = 111320 * Math.cos(lat0 * Math.PI / 180);

      var sxx = 0, syy = 0, sxy = 0;
      for (var j = 0; j < n; j++) {{
        var px = (w.x[j] - lon0) * mPerDegLon;
        var py = (w.y[j] - lat0) * mPerDegLat;
        sxx += px * px;
        syy += py * py;
        sxy += px * py;
      }}
      sxx /= n;
      syy /= n;
      sxy /= n;

      var trace = sxx + syy;
      var det = sxx * syy - sxy * sxy;
      var disc = Math.sqrt(Math.max(trace * trace / 4 - det, 0));
      var lambda1 = trace / 2 + disc;

      var vx, vy;
      if (sxy !== 0) {{
        vx = lambda1 - syy;
        vy = sxy;
      }} else {{
        vx = sxx >= syy ? 1 : 0;
        vy = sxx >= syy ? 0 : 1;
      }}
      var norm = Math.sqrt(vx * vx + vy * vy) || 1;
      vx /= norm;
      vy /= norm;

      var std1 = Math.sqrt(Math.max(lambda1, 0));
      var halfLen = 2.2 * std1;
      var dLon = (vx * halfLen) / mPerDegLon;
      var dLat = (vy * halfLen) / mPerDegLat;

      var bearing1 = ((Math.atan2(vx, vy) * 180 / Math.PI) % 180 + 180) % 180;
      var bearing2 = (bearing1 + 180) % 360;
      var variancePct = trace > 1e-9 ? (lambda1 / trace * 100) : 0;

      return {{
        x: [lon0 - dLon, lon0 + dLon],
        y: [lat0 - dLat, lat0 + dLat],
        centroidX: lon0,
        centroidY: lat0,
        bearing1: bearing1,
        bearing2: bearing2,
        variancePct: variancePct,
      }};
    }}

    function formatPcaStats(axis) {{
      if (axis.bearing1 === null) return "";
      return Math.round(axis.bearing1) + "°/" + Math.round(axis.bearing2) + "°, " + Math.round(axis.variancePct) + "%";
    }}

    function applyPcaDisplay(key) {{
      var gd = document.getElementById("plot-" + key);
      var checkbox = document.getElementById("pca-" + key);
      var statsEl = document.getElementById("pca-stats-" + key);
      var pcaIdx = parseInt(checkbox.dataset.pcaIndex, 10);
      var centroidIdx = parseInt(checkbox.dataset.pcaCentroidIndex, 10);
      var axis = computePcaAxis(key);
      Plotly.restyle(gd, {{ x: [axis.x], y: [axis.y] }}, [pcaIdx]);
      Plotly.restyle(gd, {{ x: [[axis.centroidX]], y: [[axis.centroidY]] }}, [centroidIdx]);
      if (statsEl) statsEl.textContent = formatPcaStats(axis);
    }}

    // "Custom" has no precomputed window - it's filtered client-side from
    // the longest available precomputed range ("30 days"), which is as far
    // back as the page has data for.
    function parseCustomTimestamp(s) {{
      return new Date(s.replace(" UTC", "Z").replace(" ", "T"));
    }}

    function computeCustomRange(key) {{
      var startInput = document.getElementById("custom-start-" + key);
      var endInput = document.getElementById("custom-end-" + key);
      if (!startInput.value || !endInput.value) return;
      var start = new Date(startInput.value + "Z");
      var end = new Date(endInput.value + "Z");
      var full = window.__windowData[key]["30 days"];
      var x = [], y = [], customdata = [];
      for (var i = 0; i < full.customdata.length; i++) {{
        var t = parseCustomTimestamp(full.customdata[i]);
        if (t >= start && t <= end) {{
          x.push(full.x[i]);
          y.push(full.y[i]);
          customdata.push(full.customdata[i]);
        }}
      }}
      window.__customRanges[key] = {{ x: x, y: y, customdata: customdata }};
    }}

    function applyCurrentWindow(key) {{
      var gd = document.getElementById("plot-" + key);
      var select = document.getElementById("window-select-" + key);
      var trajIdx = parseInt(select.dataset.trajectoryIndex, 10);
      var w = currentWindow(key);
      Plotly.restyle(gd, {{ x: [w.x], y: [w.y], customdata: [w.customdata] }}, [trajIdx]);

      var trackCheckbox = document.getElementById("track-" + key);
      if (trackCheckbox && trackCheckbox.checked) {{
        var trackIdx = parseInt(trackCheckbox.dataset.trackIndex, 10);
        var t = trackXY(key);
        Plotly.restyle(gd, {{ x: [t.x], y: [t.y] }}, [trackIdx]);
      }}

      var directionCheckbox = document.getElementById("direction-" + key);
      if (directionCheckbox && directionCheckbox.checked) {{
        updateDirectionArrow(gd, key, parseInt(directionCheckbox.dataset.annotationIndex, 10));
      }}

      var pcaCheckbox = document.getElementById("pca-" + key);
      if (pcaCheckbox && pcaCheckbox.checked) {{
        applyPcaDisplay(key);
      }}
    }}

    document.querySelectorAll(".window-select").forEach(function (select) {{
      select.addEventListener("change", function () {{
        var key = select.dataset.key;
        var rangeRow = document.getElementById("custom-range-" + key);

        if (select.value === "Custom") {{
          rangeRow.hidden = false;
          var startInput = document.getElementById("custom-start-" + key);
          var endInput = document.getElementById("custom-end-" + key);
          if (!startInput.value || !endInput.value) {{
            var full = window.__windowData[key]["30 days"];
            var endDate = full.customdata.length
              ? parseCustomTimestamp(full.customdata[full.customdata.length - 1])
              : new Date();
            var startDate = new Date(endDate.getTime() - 3 * 24 * 60 * 60 * 1000);
            endInput.value = endDate.toISOString().slice(0, 16);
            startInput.value = startDate.toISOString().slice(0, 16);
          }}
          computeCustomRange(key);
        }} else {{
          rangeRow.hidden = true;
        }}

        applyCurrentWindow(key);
      }});
    }});

    function applyCustomRangeFromInput(input) {{
      var key = input.dataset.key;
      computeCustomRange(key);
      applyCurrentWindow(key);
    }}

    document.querySelectorAll('.custom-range-field input[type="datetime-local"]').forEach(function (input) {{
      input.addEventListener("change", function () {{
        applyCustomRangeFromInput(input);
      }});
      // "change" alone only fires once the field loses focus - Enter should
      // redraw immediately without requiring the user to click away first.
      input.addEventListener("keydown", function (event) {{
        if (event.key === "Enter") {{
          applyCustomRangeFromInput(input);
        }}
      }});
    }});

    document.querySelectorAll('input[id^="track-"]').forEach(function (checkbox) {{
      checkbox.addEventListener("change", function () {{
        var key = checkbox.dataset.key;
        var gd = document.getElementById("plot-" + key);
        var trackIdx = parseInt(checkbox.dataset.trackIndex, 10);
        if (checkbox.checked) {{
          var t = trackXY(key);
          Plotly.restyle(gd, {{ x: [t.x], y: [t.y], visible: true }}, [trackIdx]);
        }} else {{
          Plotly.restyle(gd, {{ visible: false }}, [trackIdx]);
        }}
      }});
    }});

    document.querySelectorAll('input[id^="direction-"]').forEach(function (checkbox) {{
      checkbox.addEventListener("change", function () {{
        var key = checkbox.dataset.key;
        var gd = document.getElementById("plot-" + key);
        var annIdx = parseInt(checkbox.dataset.annotationIndex, 10);
        if (checkbox.checked) {{
          updateDirectionArrow(gd, key, annIdx);
          var showUpdate = {{}};
          showUpdate["annotations[" + annIdx + "].visible"] = true;
          Plotly.relayout(gd, showUpdate);
        }} else {{
          var hideUpdate = {{}};
          hideUpdate["annotations[" + annIdx + "].visible"] = false;
          Plotly.relayout(gd, hideUpdate);
        }}
      }});
    }});

    document.querySelectorAll('input[id^="pca-"]').forEach(function (checkbox) {{
      checkbox.addEventListener("change", function () {{
        var key = checkbox.dataset.key;
        var gd = document.getElementById("plot-" + key);
        var pcaIdx = parseInt(checkbox.dataset.pcaIndex, 10);
        var centroidIdx = parseInt(checkbox.dataset.pcaCentroidIndex, 10);
        if (checkbox.checked) {{
          applyPcaDisplay(key);
          Plotly.restyle(gd, {{ visible: true }}, [pcaIdx]);
          Plotly.restyle(gd, {{ visible: true }}, [centroidIdx]);
        }} else {{
          Plotly.restyle(gd, {{ visible: false }}, [pcaIdx]);
          Plotly.restyle(gd, {{ visible: false }}, [centroidIdx]);
          var statsEl = document.getElementById("pca-stats-" + key);
          if (statsEl) statsEl.textContent = "";
        }}
      }});
    }});
  </script>
</body>
</html>
"""


def vincenty_direct(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Destination point at distance_m/bearing_deg from (lat_deg, lon_deg) on the WGS84
    ellipsoid (Vincenty direct formula)."""
    a, f, b = WGS84_A, WGS84_F, WGS84_B
    lat1 = math.radians(lat_deg)
    alpha1 = math.radians(bearing_deg)
    sin_alpha1, cos_alpha1 = math.sin(alpha1), math.cos(alpha1)

    tan_u1 = (1 - f) * math.tan(lat1)
    cos_u1 = 1 / math.sqrt(1 + tan_u1**2)
    sin_u1 = tan_u1 * cos_u1

    sigma1 = math.atan2(tan_u1, cos_alpha1)
    sin_alpha = cos_u1 * sin_alpha1
    cos_sq_alpha = 1 - sin_alpha**2
    u_sq = cos_sq_alpha * (a**2 - b**2) / b**2
    cap_a = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    cap_b = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    sigma = distance_m / (b * cap_a)
    for _ in range(20):
        cos2_sigma_m = math.cos(2 * sigma1 + sigma)
        sin_sigma, cos_sigma = math.sin(sigma), math.cos(sigma)
        delta_sigma = cap_b * sin_sigma * (
            cos2_sigma_m
            + cap_b
            / 4
            * (
                cos_sigma * (-1 + 2 * cos2_sigma_m**2)
                - cap_b / 6 * cos2_sigma_m * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos2_sigma_m**2)
            )
        )
        sigma_new = distance_m / (b * cap_a) + delta_sigma
        if abs(sigma_new - sigma) < 1e-12:
            sigma = sigma_new
            break
        sigma = sigma_new

    tmp = sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_alpha1
    lat2 = math.atan2(
        sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_alpha1,
        (1 - f) * math.sqrt(sin_alpha**2 + tmp**2),
    )
    lam = math.atan2(sin_sigma * sin_alpha1, cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_alpha1)
    cap_c = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
    big_l = lam - (1 - cap_c) * f * sin_alpha * (
        sigma + cap_c * sin_sigma * (cos2_sigma_m + cap_c * cos_sigma * (-1 + 2 * cos2_sigma_m**2))
    )
    lon2 = math.radians(lon_deg) + big_l
    return math.degrees(lat2), math.degrees(lon2)


def format_anchor_position(lat_deg: float, lon_deg: float) -> str:
    lat_dir = "N" if lat_deg >= 0 else "S"
    lon_dir = "E" if lon_deg >= 0 else "W"
    return f"{abs(lat_deg):.4f}{lat_dir}, {abs(lon_deg):.4f}{lon_dir}"


def compute_pca_axis(lons: list, lats: list) -> dict:
    """Dominant axis of spatial spread (first principal component) for a set
    of lon/lat points: a 2-point line through the centroid spanning the
    data's spread (+-2.2 std along the axis), the centroid itself, the
    axis's bearing (as a 0-180 range line, e.g. 15/195 - a line has no
    "direction", only an orientation), and the percentage of total variance
    it explains. Longitude is corrected for the cos(latitude) distortion
    before computing covariance, so the axis reflects true physical
    distance, not raw degrees. Mirrors the client-side JS version used when
    the window selection changes."""
    empty = {
        "line_x": [], "line_y": [], "centroid_lon": None, "centroid_lat": None,
        "bearing1": None, "bearing2": None, "variance_pct": None,
    }
    if len(lons) < 2:
        return empty
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    xs = [(lon - lon0) * m_per_deg_lon for lon in lons]
    ys = [(lat - lat0) * m_per_deg_lat for lat in lats]
    n = len(xs)
    sxx = sum(x * x for x in xs) / n
    syy = sum(y * y for y in ys) / n
    sxy = sum(x * y for x, y in zip(xs, ys)) / n
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(trace * trace / 4 - det, 0))
    lambda1 = trace / 2 + disc
    if sxy != 0:
        vx, vy = lambda1 - syy, sxy
    else:
        vx, vy = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
    norm = math.hypot(vx, vy) or 1
    vx, vy = vx / norm, vy / norm
    std1 = math.sqrt(max(lambda1, 0))
    half_len = 2.2 * std1
    d_lon = (vx * half_len) / m_per_deg_lon
    d_lat = (vy * half_len) / m_per_deg_lat
    bearing1 = math.degrees(math.atan2(vx, vy)) % 180
    bearing2 = (bearing1 + 180) % 360
    variance_pct = (lambda1 / trace * 100) if trace > 1e-9 else 0.0
    return {
        "line_x": [lon0 - d_lon, lon0 + d_lon],
        "line_y": [lat0 - d_lat, lat0 + d_lat],
        "centroid_lon": lon0,
        "centroid_lat": lat0,
        "bearing1": bearing1,
        "bearing2": bearing2,
        "variance_pct": variance_pct,
    }


def fetch_positions(location: dict) -> tuple[pd.DataFrame, datetime]:
    """Download the location's CSV from Azure Blob Storage and return its
    full lat/lon history - including points outside the mooring's excursion
    circle, since a real reading out there (dragging anchor, mismeasured
    tether length) is exactly the kind of thing this page should surface,
    not silently hide."""
    now = datetime.now(timezone.utc)
    resp = requests.get(blob_url(location["blob_name"]), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"Blob download failed for {location['blob_name']} ({resp.status_code}): {resp.text}")

    positions = pd.read_csv(io.StringIO(resp.text))
    positions["timestamp"] = pd.to_datetime(positions["timestamp"], utc=True)
    positions = positions.sort_values("timestamp")
    if positions.empty:
        raise RuntimeError(f"No position data found in {location['blob_name']}.")

    return positions, now


def fetch_latest_waves(session: requests.Session, wave_channels: dict) -> dict:
    """Fetch the most recent Hs/Tp reading directly from Elements Online -
    unlike position data, this isn't piped through Blob Storage, so it's
    fetched fresh each time the page is generated."""
    now = datetime.now(timezone.utc)
    ts_1 = int((now - timedelta(hours=6)).timestamp() * 1000)
    ts_2 = int(now.timestamp() * 1000)

    values = {}
    for field, channel_id in wave_channels.items():
        df = fetch_channel(channel_id, session, ts_1, ts_2)
        values[field] = float(df.iloc[-1]["value"]) if not df.empty else None
    return values


def build_dataset(positions: pd.DataFrame, now: datetime) -> dict:
    """Everything a location's page section needs that can change on an
    "Update" refresh: the trajectory for each dropdown window, and the
    latest-position marker."""
    latest = positions.iloc[-1]
    other_points = positions.iloc[:-1]

    windows = {}
    for label, delta in WINDOW_OPTIONS:
        cutoff = now - delta
        subset = other_points[other_points["timestamp"] >= cutoff]
        windows[label] = {
            "x": subset["lon"].tolist(),
            "y": subset["lat"].tolist(),
            "customdata": subset["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC").tolist(),
        }

    return {
        "windows": windows,
        "latest": {
            "lon": float(latest["lon"]),
            "lat": float(latest["lat"]),
            "name": f"Latest buoy position ({format_anchor_position(latest['lat'], latest['lon'])})",
            "time_label": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
        # ISO timestamp of the newest data point (not "when this page was
        # generated") - the browser converts it to the viewer's local time.
        "latest_timestamp": latest["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def render_location_section(location: dict, dataset: dict, include_plotlyjs: bool) -> str:
    """The chart + controls for one location's tab panel."""
    anchor = location["anchor"]
    default_window = dataset["windows"][DEFAULT_WINDOW_LABEL]
    latest = dataset["latest"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=default_window["x"],
            y=default_window["y"],
            mode="markers",
            # A hairline stroke in the same fill color doesn't change the visible
            # color, but gives the SVG renderer a defined edge to anti-alias
            # against - small unstroked circles otherwise look uneven/jagged
            # depending on where their center lands relative to the pixel grid.
            marker=dict(
                color=POINTS_COLOR,
                size=TRAJECTORY_MARKER_SIZE,
                line=dict(width=0.5, color=POINTS_COLOR),
            ),
            name="Buoy trajectory",
            legendgroup="trajectory",
            customdata=default_window["customdata"],
            hovertemplate=(
                "Time: %{customdata}<br>"
                "Latitude: %{y:.6f}<br>"
                "Longitude: %{x:.6f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[latest["lon"]],
            y=[latest["lat"]],
            mode="markers",
            marker=dict(symbol="triangle-up", size=LATEST_MARKER_SIZE, color=LATEST_COLOR),
            name=latest["name"],
            legendgroup="latest",
            customdata=[latest["time_label"]],
            hovertemplate=(
                "Time: %{customdata}<br>"
                "Latitude: %{y:.6f}<br>"
                "Longitude: %{x:.6f}<extra></extra>"
            ),
        )
    )

    if anchor is not None:
        fig.add_trace(
            go.Scatter(
                x=[anchor["lon"]],
                y=[anchor["lat"]],
                mode="markers",
                marker=dict(symbol="circle-open", size=14, color=ANCHOR_COLOR, line=dict(width=2)),
                name=f"Anchor position ({format_anchor_position(anchor['lat'], anchor['lon'])})",
                legendgroup="anchor",
                hoverinfo="skip",
            )
        )
        # Excursion circle: a cloud of points, each exactly radius_m from the
        # anchor on the WGS84 ellipsoid (Vincenty direct formula), one per
        # bearing swept around the full 360 degrees.
        bearings = np.linspace(0, 360, 361)
        circle_points = [vincenty_direct(anchor["lat"], anchor["lon"], b, anchor["radius_m"]) for b in bearings]
        circle_lat = np.array([p[0] for p in circle_points])
        circle_lon = np.array([p[1] for p in circle_points])
        fig.add_trace(
            go.Scatter(
                x=circle_lon,
                y=circle_lat,
                mode="lines",
                line=dict(color=CIRCLE_COLOR, width=1.5),
                name=f"Excursion radius ({anchor['radius_m']:.1f} m)",
                legendgroup="excursion",
                hoverinfo="skip",
            )
        )
        # Frame the excursion circle plus whatever's actually plotted - a
        # buoy currently outside its circle should stay visible, not get
        # cropped out of view. scaleratio comes from the circle's own true
        # geodesic proportions (the meridian-convergence correction for this
        # latitude), not the combined extent, so the circle still renders as
        # a true circle; Plotly only ever widens a range to satisfy it, so
        # this can't clip data even if the union box is lopsided.
        circle_lon_span = circle_lon.max() - circle_lon.min()
        circle_lat_span = circle_lat.max() - circle_lat.min()
        scaleratio = circle_lon_span / circle_lat_span

        all_lon = default_window["x"] + [latest["lon"], anchor["lon"]]
        all_lat = default_window["y"] + [latest["lat"], anchor["lat"]]
        lon_min = min(circle_lon.min(), min(all_lon))
        lon_max = max(circle_lon.max(), max(all_lon))
        lat_min = min(circle_lat.min(), min(all_lat))
        lat_max = max(circle_lat.max(), max(all_lat))
        pad_lon = 0.05 * circle_lon_span
        pad_lat = 0.05 * circle_lat_span
        lon_range = [lon_min - pad_lon, lon_max + pad_lon]
        lat_range = [lat_min - pad_lat, lat_max + pad_lat]
    else:
        # No anchor geometry known - just frame the trajectory + latest point.
        all_lon = default_window["x"] + [latest["lon"]]
        all_lat = default_window["y"] + [latest["lat"]]
        lon_span = max(all_lon) - min(all_lon) or 0.001
        lat_span = max(all_lat) - min(all_lat) or 0.001
        pad = 0.15
        lon_range = [min(all_lon) - pad * lon_span, max(all_lon) + pad * lon_span]
        lat_range = [min(all_lat) - pad * lat_span, max(all_lat) + pad * lat_span]
        scaleratio = lon_span / lat_span

    # Track line: connects the trajectory points in chronological order,
    # ending at the latest buoy position. Off by default - toggled by the
    # "Track" checkbox in the HTML controls.
    track_trace_index = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=default_window["x"] + [latest["lon"]],
            y=default_window["y"] + [latest["lat"]],
            mode="lines",
            line=dict(color="#000000", width=1),
            name="Track",
            showlegend=False,
            hoverinfo="skip",
            visible=False,
        )
    )

    # PCA axis: dominant direction of spatial spread within the selected
    # window (first principal component, through the centroid). Off by
    # default - toggled by the "PCA" checkbox.
    pca_trace_index = len(fig.data)
    pca = compute_pca_axis(default_window["x"], default_window["y"])
    fig.add_trace(
        go.Scatter(
            x=pca["line_x"],
            y=pca["line_y"],
            mode="lines",
            line=dict(color="#e67e22", width=2, dash="dot"),
            name="PCA axis",
            showlegend=False,
            hoverinfo="skip",
            visible=False,
        )
    )
    pca_centroid_trace_index = len(fig.data)
    centroid_xy = (
        ([pca["centroid_lon"]], [pca["centroid_lat"]]) if pca["centroid_lon"] is not None else ([], [])
    )
    fig.add_trace(
        go.Scatter(
            x=centroid_xy[0],
            y=centroid_xy[1],
            mode="markers",
            marker=dict(symbol="x", size=10, color="#e67e22", line=dict(width=2)),
            name="PCA centroid",
            showlegend=False,
            hoverinfo="skip",
            visible=False,
        )
    )

    # Direction arrow: earliest trajectory point in the selected window to
    # the latest buoy position. Off by default - toggled by the "Direction"
    # checkbox. A layout annotation (not a trace), since Plotly's arrowheads
    # are an annotation feature.
    direction_annotation_index = 0
    earliest_lon = default_window["x"][0] if default_window["x"] else latest["lon"]
    earliest_lat = default_window["y"][0] if default_window["y"] else latest["lat"]
    fig.add_annotation(
        x=latest["lon"],
        y=latest["lat"],
        ax=earliest_lon,
        ay=earliest_lat,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="#1a73e8",
        visible=False,
    )

    fig.update_layout(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_white",
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        margin=dict(t=30, l=70, r=260, b=60),
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top", tracegroupgap=10),
    )
    # Show full coordinate values (no offset/exponent notation) and correct the
    # y/x aspect ratio for the convergence of meridians at this latitude, so the
    # excursion circle (or trajectory bounds) renders with true proportions.
    fig.update_xaxes(tickformat=".4f", exponentformat="none", range=lon_range)
    fig.update_yaxes(
        tickformat=".4f",
        exponentformat="none",
        range=lat_range,
        scaleanchor="x",
        scaleratio=scaleratio,
    )

    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={"displayModeBar": False},
        div_id=f"plot-{location['key']}",
    )

    key = location["key"]
    options_html = "\n          ".join(
        f'<option value="{label}"{" selected" if label == DEFAULT_WINDOW_LABEL else ""}>{label}</option>'
        for label in WINDOW_LABELS
    )
    options_html += '\n          <option value="Custom">Custom</option>'

    wave_html = ""
    waves = dataset.get("waves")
    if waves:
        def fmt_wave(value, suffix, decimals):
            return f"{value:.{decimals}f} {suffix}" if value is not None else "N/A"

        rows_html = []
        for row_num in sorted({row for *_, row in WAVE_FIELD_SPECS}):
            fields_html = "\n          ".join(
                f'<div class="wave-field"><span class="wave-label">{label}:</span> {fmt_wave(waves.get(field_key), suffix, decimals)}</div>'
                for field_key, label, suffix, decimals, row in WAVE_FIELD_SPECS
                if row == row_num and field_key in waves
            )
            if fields_html:
                rows_html.append(f'<div class="wave-conditions">\n          {fields_html}\n        </div>')
        wave_html = "\n        " + "\n        ".join(rows_html)

    return f"""
    <div class="card">
      <div class="chart">
        {plot_div}
        <div class="window-controls">
          <label class="toggle-label">
            <input type="checkbox" id="track-{key}" data-key="{key}" data-track-index="{track_trace_index}">
            Track
          </label>
          <label class="toggle-label">
            <input type="checkbox" id="direction-{key}" data-key="{key}" data-annotation-index="{direction_annotation_index}">
            Direction
          </label>
          <select id="window-select-{key}" class="window-select" data-key="{key}" data-trajectory-index="{TRAJECTORY_TRACE_INDEX}">
            {options_html}
          </select>
        </div>
        <div class="pca-row">
          <label class="toggle-label">
            <input type="checkbox" id="pca-{key}" data-key="{key}" data-pca-index="{pca_trace_index}" data-pca-centroid-index="{pca_centroid_trace_index}">
            PCA
          </label>
          <span class="pca-stats" id="pca-stats-{key}"></span>
        </div>
        <div class="custom-range" id="custom-range-{key}" hidden>
          <label class="custom-range-field">Start <input type="datetime-local" id="custom-start-{key}" data-key="{key}"></label>
          <label class="custom-range-field">End <input type="datetime-local" id="custom-end-{key}" data-key="{key}"></label>
        </div>
        <div class="side-controls">{wave_html}
          <div class="last-updated" data-timestamp="{dataset['latest_timestamp']}">Data as of: {dataset['latest_timestamp']}</div>
        </div>
      </div>
    </div>
    """


def render_page(datasets: dict) -> str:
    tab_buttons = []
    tab_panels = []
    window_data = {}
    latest_data = {}
    for i, location in enumerate(LOCATIONS):
        key = location["key"]
        active = " active" if i == 0 else ""
        title = f'{location["label"]} buoy position'
        tab_buttons.append(
            f'<button class="tab-btn{active}" data-tab="{key}" data-title="{title}">{location["label"]}</button>'
        )
        section_html = render_location_section(location, datasets[key], include_plotlyjs=(i == 0))
        tab_panels.append(f'<div id="tab-{key}" class="tab-panel{active}">{section_html}</div>')
        window_data[key] = datasets[key]["windows"]
        latest_data[key] = datasets[key]["latest"]

    return PAGE_TEMPLATE.format(
        title="Buoy position",
        favicon_data_uri=FAVICON_DATA_URI,
        initial_title=f'{LOCATIONS[0]["label"]} buoy position',
        tab_buttons="\n      ".join(tab_buttons),
        tab_panels="\n    ".join(tab_panels),
        figure_width=FIGURE_WIDTH,
        window_data_json=json.dumps(window_data),
        latest_data_json=json.dumps(latest_data),
    )


def main() -> None:
    elements_session = None
    if any(location.get("wave_channels") for location in LOCATIONS):
        if not ELEMENTS_USERNAME or not ELEMENTS_PASSWORD:
            raise RuntimeError(
                "ELEMENTS_USERNAME/ELEMENTS_PASSWORD must be set to fetch wave conditions (Hs/Tp)."
            )
        elements_session = login(ELEMENTS_USERNAME, ELEMENTS_PASSWORD)

    datasets = {}
    for location in LOCATIONS:
        positions, now = fetch_positions(location)
        dataset = build_dataset(positions, now)
        if location.get("wave_channels"):
            dataset["waves"] = fetch_latest_waves(elements_session, location["wave_channels"])
        datasets[location["key"]] = dataset
    html = render_page(datasets)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "buoy_position.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Also publish into docs/ - GitHub Pages serves this repo's site from
    # there, so the page stays reachable online after a
    # `git add docs && git commit && git push`. Two copies: index.html for
    # the bare repo URL, buoys.html for the explicit link people share.
    docs_dir = os.path.join(script_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    for pages_name in ("index.html", "buoys.html"):
        pages_path = os.path.join(docs_dir, pages_name)
        with open(pages_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Wrote {pages_path} (for GitHub Pages)")

    print(f"Wrote {output_path}")
    if not os.environ.get("CI"):
        webbrowser.open(f"file://{output_path}")


if __name__ == "__main__":
    main()
