# -*- coding: utf-8 -*-
import os, time, json
from pathlib import Path
from typing import List, Dict, Any
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
from flask import Flask
import requests

# -------------------- Config --------------------
# If DATA_SOURCE (local file path) is set, it overrides DATA_URL (remote).
DATA_SOURCE = os.environ.get("DATA_SOURCE", "").strip()
DATA_URL    = os.environ.get(
    "DATA_URL",
    "https://raw.githubusercontent.com/AlejandroCastelloDiez/BESS-OMIE-Index/main/BESS%20OMIE%20Results.json"
).strip()

REFRESH_MS  = int(os.environ.get("REFRESH_MS", "60000"))  # 60s
WP_ORIGIN   = os.environ.get("WP_ORIGIN", "https://yourwordpressdomain.com").strip()

SERIES_ORDER = ["DA_Earnings", "IDA1_Delta", "IDA2_Delta", "IDA3_Delta", "IDC_Delta"]
INTRADAY_SERIES = [s for s in SERIES_ORDER if s != "DA_Earnings"]

# -------------------- Flask base (iframe headers) --------------------
server = Flask(__name__)

@server.after_request
def add_frame_headers(resp):
    # Allow embedding from your WP site (adjust WP_ORIGIN as needed)
    resp.headers["Content-Security-Policy"] = f"frame-ancestors 'self' {WP_ORIGIN}"
    resp.headers["X-Frame-Options"] = f"ALLOW-FROM {WP_ORIGIN}"
    return resp

@server.get("/health")
def _health():
    return {"ok": True}

# -------------------- Data helpers --------------------
def _load_json() -> List[Dict[str, Any]]:
    """
    Load list[dict] from local DATA_SOURCE or remote DATA_URL.
    Remote loads include a 5-min cache-busting query param.
    """
    src = None
    try:
        if DATA_SOURCE:
            p = Path(DATA_SOURCE)
            if not p.exists():
                return []
            src = f"file://{p}"
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            # Remote
            bust = int(time.time() // 300)  # 5-minute cache buster
            url = f"{DATA_URL}?t={bust}"
            src = url
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
        if isinstance(data, list):
            return data
        return []
    except Exception:
        # Keep it quiet for the UI; status label will show "No data"
        return []

def _to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of dicts -> DataFrame, coerce numerics, parse/sort dates, sum duplicates.
    Ensures 'date' is a real column used for groupby to avoid FutureWarning/KeyError.
    """
    if not rows:
        cols = ["date"] + SERIES_ORDER + ["Total"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)

    # Ensure numeric columns and coerce
    for col in SERIES_ORDER:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Parse dates -> datetime64[ns], normalize to midnight, keep as COLUMN
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])

    # Aggregate by 'date' (sum duplicates)
    agg = {c: "sum" for c in SERIES_ORDER}
    df = (
        df.groupby("date", as_index=False)
          .agg(agg)
          .sort_values("date")
          .reset_index(drop=True)
    )

    # Total per day
    df["Total"] = df[SERIES_ORDER].sum(axis=1)
    return df

def _fmt_euro(v: float) -> str:
    try:
        return f"€{v:,.2f}"
    except Exception:
        return "€0.00"

def make_stacked_area(df: pd.DataFrame, include_intraday) -> go.Figure:
    df = df.sort_values("date").copy()
    fig = go.Figure()

    # Base: DA
    if "DA_Earnings" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["DA_Earnings"],
            mode="lines", name="DA Earnings",
            stackgroup="one",
            hovertemplate="DA: €%{y:.2f}<extra></extra>",
        ))

    # Intraday deltas
    for col in INTRADAY_SERIES:
        if col not in include_intraday:
            continue
        pretty = col.replace("_", " ").replace("Delta", "Δ")
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col],
            mode="lines", name=pretty,
            stackgroup="one",
            hovertemplate=f"{pretty}: €%{{y:.2f}}<extra></extra>",
        ))

    # Total overlay (thin, not stacked)
    if "Total" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["Total"],
            mode="lines", name="Total (overlay)",
            line=dict(width=1),
            hovertemplate="Total: €%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="Perfect Foresight - 1MW 2h 1c/d BESS",
        xaxis_title="Date",
        yaxis_title="€/MW",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(tickprefix="€")
    return fig

# -------------------- Dash app --------------------
app = Dash(__name__, server=server, title="BESS OMIE Index")

# Preload for sensible date picker defaults
_initial_rows = _load_json()
_initial_df = _to_df(_initial_rows)
if _initial_df.empty:
    default_start = date.today() - timedelta(days=365)
    default_end = date.today()
else:
    max_day = _initial_df["date"].max().date()
    min_day = _initial_df["date"].min().date()
    default_end = max_day
    default_start = max(min_day, max_day - timedelta(days=365))

DATA_LABEL = DATA_SOURCE if DATA_SOURCE else DATA_URL

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "1rem"},
    children=[
        html.H2("Stacked Earnings by Date"),
        html.Div(
            style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Div([html.Label("Data source:"), html.Code(DATA_LABEL, style={"fontSize": "0.9rem"})]),
                html.Div(
                    [
                        html.Label("Show intraday layers:"),
                        dcc.Checklist(
                            id="layers",
                            options=[{"label": s.replace("_", " ").replace("Delta", "Δ"), "value": s} for s in INTRADAY_SERIES],
                            value=INTRADAY_SERIES.copy(),
                            inline=True,
                        ),
                    ],
                    style={"fontSize": "0.95rem"},
                ),
                html.Div(
                    [
                        html.Label("Date range:"),
                        dcc.DatePickerRange(
                            id="range",
                            start_date=default_start,
                            end_date=default_end,
                            display_format="YYYY-MM-DD",
                        ),
                    ]
                ),
            ],
        ),

        # KPI row (3 cards)
        html.Div(
            id="kpi-row",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(180px, 1fr))",
                "gap": "12px",
                "marginTop": "8px",
                "marginBottom": "8px",
            },
            children=[
                html.Div(
                    style={"padding": "12px 14px", "border": "1px solid #e5e7eb", "borderRadius": "12px",
                           "boxShadow": "0 1px 2px rgba(0,0,0,0.04)"},
                    children=[html.Div("YTD Total (calendar year)", style={"fontSize": "0.85rem", "color": "#555"}),
                              html.Div(id="kpi-ytd-value", style={"fontSize": "1.2rem", "fontWeight": "600"})],
                ),
                html.Div(
                    style={"padding": "12px 14px", "border": "1px solid #e5e7eb", "borderRadius": "12px",
                           "boxShadow": "0 1px 2px rgba(0,0,0,0.04)"},
                    children=[html.Div("Avg Daily Total (selected range)", style={"fontSize": "0.85rem", "color": "#555"}),
                              html.Div(id="kpi-avg-value", style={"fontSize": "1.2rem", "fontWeight": "600"})],
                ),
                html.Div(
                    style={"padding": "12px 14px", "border": "1px solid #e5e7eb", "borderRadius": "12px",
                           "boxShadow": "0 1px 2px rgba(0,0,0,0.04)"},
                    children=[html.Div("Expected Yearly Total (avg × 365)", style={"fontSize": "0.85rem", "color": "#555"}),
                              html.Div(id="kpi-exp-value", style={"fontSize": "1.2rem", "fontWeight": "600"})],
                ),
            ],
        ),

        dcc.Graph(id="stacked-graph", config={"displaylogo": False}),
        dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),
        html.Div(id="data-status", style={"fontSize": "0.85rem", "color": "#555"}),
    ],
)

@callback(
    Output("stacked-graph", "figure"),
    Output("data-status", "children"),
    Output("kpi-ytd-value", "children"),
    Output("kpi-avg-value", "children"),
    Output("kpi-exp-value", "children"),
    Input("layers", "value"),
    Input("range", "start_date"),
    Input("range", "end_date"),
    Input("refresh", "n_intervals"),
)
def update_chart(selected_layers, start_date, end_date, _n):
    rows = _load_json()
    df = _to_df(rows)

    if df.empty:
        empty = go.Figure()
        empty.update_layout(
            title="No data available",
            xaxis_title="Date",
            yaxis_title="€",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return empty, "No data found or failed to load. Check DATA_URL/DATA_SOURCE.", "€0.00", "€0.00", "€0.00"

    # Filter by date range
    df_range = df.copy()
    if start_date:
        start_date = pd.to_datetime(start_date).normalize()
        df_range = df_range[df_range["date"] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date).normalize()
        df_range = df_range[df_range["date"] <= end_date]
    if df_range.empty:
        df_range = df.copy()  # fallback to avoid blank chart

    # KPIs
    today = pd.Timestamp.today().normalize()
    ytd_mask = (df["date"] >= pd.Timestamp(today.year, 1, 1)) & (df["date"] <= today)
    ytd_total = float(df.loc[ytd_mask, "Total"].sum()) if "Total" in df.columns else 0.0

    avg_daily = float(df_range["Total"].mean()) if not df_range.empty else 0.0
    expected_yearly = avg_daily * 365.0

    fig = make_stacked_area(df_range, include_intraday=(selected_layers or []))
    status = f"Loaded {len(df_range)} days · First: {df_range['date'].min().date()} · Last: {df_range['date'].max().date()}"

    return fig, status, _fmt_euro(ytd_total), _fmt_euro(avg_daily), _fmt_euro(expected_yearly)

# Local debugging
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=True)

