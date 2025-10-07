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

# Allow multiple origins (comma-separated). Falls back to WP_ORIGIN for backward-compat.
ALLOWED_EMBED_ORIGINS = os.environ.get(
    "ALLOWED_EMBED_ORIGINS",
    os.environ.get("WP_ORIGIN", "")
).strip()

def _csp_frame_ancestors_value() -> str:
    parts = ["'self'"]
    if ALLOWED_EMBED_ORIGINS:
        parts += [t.strip() for t in ALLOWED_EMBED_ORIGINS.split(",") if t.strip()]
    # de-duplicate while preserving order
    seen = set(); out = []
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return " ".join(out)

@server.after_request
def add_frame_headers(resp):
    # Do NOT send X-Frame-Options (ALLOW-FROM is deprecated/ignored by modern browsers)
    resp.headers.pop("X-Frame-Options", None)
    # The header that actually controls iframe embedding:
    resp.headers["Content-Security-Policy"] = f"frame-ancestors {_csp_frame_ancestors_value()};"
    # Optional nicety
    resp.headers.setdefault("Referrer-Policy", "no-referrer-when-downgrade")
    return resp

@server.get("/health")
def _health():
    return {"ok": True}

# -------------------- Data helpers --------------------
def _load_json() -> List[Dict[str, Any]]:
    """
    Load list[dict] from local DATA_SOURCE (if it exists) or remote DATA_URL.
    Accepts top-level list, or dict wrapping the list under common keys.
    """
    try:
        use_local = False
        if DATA_SOURCE:
            p = Path(DATA_SOURCE)
            use_local = p.exists() and p.is_file()

        if use_local:
            raw = json.loads(Path(DATA_SOURCE).read_text(encoding="utf-8"))
        else:
            bust = int(time.time() // 300)  # 5-min cache buster
            url = f"{DATA_URL}?t={bust}"
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            raw = r.json()

        # Normalize shapes
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for k in ("rows", "data", "items", "results"):
                if k in raw and isinstance(raw[k], list):
                    return raw[k]
        # Fallback: single dict -> wrap
        if isinstance(raw, dict):
            return [raw]
        return []
    except Exception:
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

    # Total per day (all layers)
    df["Total"] = df[SERIES_ORDER].sum(axis=1)
    return df

def _fmt_euro(v: float) -> str:
    try:
        return f"€{v:,.2f}"
    except Exception:
        return "€0.00"

# --- Compute Visible Total (DA + selected intraday layers) ---
def _compute_visible_total(df: pd.DataFrame, selected_layers: list) -> pd.Series:
    """Return a Series = DA_Earnings + sum(selected intraday layers)."""
    selected_layers = list(selected_layers or [])
    cols = ["DA_Earnings"] + [c for c in selected_layers if c in INTRADAY_SERIES]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)

def make_stacked_area(df: pd.DataFrame, include_intraday, visible_total: pd.Series) -> go.Figure:
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

    # Visible Total overlay (thin, not stacked)
    fig.add_trace(go.Scatter(
        x=df["date"], y=visible_total,
        mode="lines", name="Total (overlay)",
        line=dict(width=1),
        hovertemplate="Total: €%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="€/MW",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="ui-sans-serif")
    )
    fig.update_yaxes(tickprefix="€")
    return fig

# -------------------- Dash app --------------------
app = Dash(
    __name__,
    server=server,
    title="BESS OMIE Index",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

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

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "1rem", "fontFamily": "ui-sans-serif"},
    children=[
        # Top controls: only the intraday layer toggles
        html.Div(
            style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                html.Div(
                    [
                        html.Label("Show Intraday Layers:", style={"fontWeight": 600}),
                        dcc.Checklist(
                            id="layers",
                            options=[{"label": s.replace("_", " ").replace("Delta", "Δ"), "value": s} for s in INTRADAY_SERIES],
                            value=INTRADAY_SERIES.copy(),
                            inline=True,
                        ),
                    ],
                    style={"fontSize": "0.95rem"},
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
                    children=[html.Div("YTD Total (calendar year, visible layers)", style={"fontSize": "0.85rem", "color": "#555"}),
                              html.Div(id="kpi-ytd-value", style={"fontSize": "1.2rem", "fontWeight": "600"})],
                ),
                html.Div(
                    style={"padding": "12px 14px", "border": "1px solid #e5e7eb", "borderRadius": "12px",
                           "boxShadow": "0 1px 2px rgba(0,0,0,0.04)"},
                    children=[html.Div("Avg Daily (selected range, visible layers)", style={"fontSize": "0.85rem", "color": "#555"}),
                              html.Div(id="kpi-avg-value", style={"fontSize": "1.2rem", "fontWeight": "600"})],
                ),
                html.Div(
                    style={"padding": "12px 14px", "border": "1px solid #e5e7eb", "borderRadius": "12px",
                           "boxShadow": "0 1px 2px rgba(0,0,0,0.04)"},
                    children=[html.Div("Expected Yearly (avg × 365, visible layers)", style={"fontSize": "0.85rem", "color": "#555"}),
                              html.Div(id="kpi-exp-value", style={"fontSize": "1.2rem", "fontWeight": "600"})],
                ),
            ],
        ),

        # Date range selector under the KPIs (card style)
        html.Div(
            style={
                "padding": "12px 14px",
                "border": "1px solid #e5e7eb",
                "borderRadius": "12px",
                "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
                "marginBottom": "8px",
                "display": "inline-block"
            },
            children=[
                html.Label("Date range", style={"fontWeight": 600, "display": "block", "marginBottom": "6px"}),
                dcc.DatePickerRange(
                    id="range",
                    start_date=default_start,
                    end_date=default_end,
                    display_format="YYYY-MM-DD",
                ),
            ],
        ),

        dcc.Graph(id="stacked-graph", config={"displaylogo": False}, style={"width": "100%"}),
        dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),
        # removed data-status line
    ],
)

@callback(
    Output("stacked-graph", "figure"),
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
            font=dict(family="ui-sans-serif"),
        )
        return empty, "€0.00", "€0.00", "€0.00"

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

    # Visible totals (for KPIs & overlay)
    visible_total_full  = _compute_visible_total(df, selected_layers)
    visible_total_range = _compute_visible_total(df_range, selected_layers)

    # KPIs (based on visible layers)
    today = pd.Timestamp.today().normalize()
    ytd_mask = (df["date"] >= pd.Timestamp(today.year, 1, 1)) & (df["date"] <= today)
    ytd_total = float(visible_total_full[ytd_mask].sum())

    avg_daily = float(visible_total_range.mean()) if not df_range.empty else 0.0
    expected_yearly = avg_daily * 365.0

    fig = make_stacked_area(
        df_range,
        include_intraday=(selected_layers or []),
        visible_total=visible_total_range,
    )
    fig.update_layout(font=dict(family="ui-sans-serif"))

    return fig, _fmt_euro(ytd_total), _fmt_euro(avg_daily), _fmt_euro(expected_yearly)

# Local debugging
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=True)


