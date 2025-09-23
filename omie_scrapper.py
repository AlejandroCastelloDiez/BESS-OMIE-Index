import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
from matplotlib.lines import Line2D
import pulp
import numpy as np
import matplotlib.pyplot as plt
import os, json

#@title OMIE Data Functions
# ---------------- URLs ----------------
URL_DA   = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{date}.1"
URL_IDA  = "https://www.omie.es/es/file-download?parents=marginalpibc&filename=marginalpibc_{date}{sess}.1"
URL_IDC  = "https://www.omie.es/es/file-download?parents=precios_pibcic&filename=precios_pibcic_{date}.1"

# --------------- Helpers ---------------
def _download_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def _clean_numeric(s: str) -> float:
    """Tolerant numeric parse: '72,20' -> 72.20, '.52' -> 0.52, else NaN on failure."""
    if s is None:
        return float("nan")
    s = s.strip().replace(",", ".")
    if s.startswith("."):
        s = "0" + s
    try:
        return float(s)
    except Exception:
        return float("nan")

def _strip_header_footer_lines(lines):
    """Drop optional first 'MARGINAL...' line and trailing '*' line."""
    if lines and lines[0].strip().upper().startswith("MARGINAL"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("*"):
        lines = lines[:-1]
    return lines

def _upsample_hourly_block_to_qh(df: pd.DataFrame, period_col: str, value_col: str, date_col: str = "date") -> pd.DataFrame:
    """
    Generic upsampler: if max(period) <= 30 treat as hourly and repeat each hour 4x to QH indices.
    Returns ['date','qh',value_col]. Keeps partial-day blocks (e.g., IDA3 subset).
    """
    if df[period_col].max() <= 30:
        rows = []
        for _, r in df.sort_values(period_col).iterrows():
            hour = int(r[period_col])
            start_qh = 4 * (hour - 1) + 1
            for qh_new in range(start_qh, start_qh + 4):
                rows.append({date_col: r[date_col], "qh": qh_new, value_col: r[value_col]})
        out = pd.DataFrame(rows)
    else:
        out = df.rename(columns={period_col: "qh"})[[date_col, "qh", value_col]]
    return out

# --------- Common parser (DA/IDA generic) → ES long with 'period' ----------
def _parse_da_ida_raw_to_period_es(raw_text: str) -> pd.DataFrame:
    """
    Parse DA/IDA text into a long df (Spain only) with a 'period' column (hour or QH as-is).
    Records: YYYY;MM;DD;PERIOD;PT;ES; (with possible header & trailing '*')
    Returns: ['date','period','es_price'] where 'period' is hour (<=30) or QH (>30).
    """
    lines = raw_text.strip().splitlines()
    lines = _strip_header_footer_lines(lines)

    buf = StringIO("\n".join(lines))
    df = pd.read_csv(buf, sep=";", header=None, engine="python", dtype=str)

    # Keep first 6 fields: YYYY,MM,DD,PERIOD,PT,ES (ignore trailing empty column)
    df = df.iloc[:, :6]
    df.columns = ["year", "month", "day", "period", "pt_price", "es_price"]

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"]   = df["day"].astype(int)
    df["period"]    = df["period"].astype(int)
    df["es_price"]  = df["es_price"].apply(_clean_numeric)
    df["date"]  = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))

    return df[["date","period","es_price"]].sort_values(["date","period"]).reset_index(drop=True)

def _to_qh_from_period(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample if hourly; else keep QH. Returns ['date','qh','es_price']."""
    return _upsample_hourly_block_to_qh(df.copy(), period_col="period", value_col="es_price", date_col="date")

# --------------- DA (Day-Ahead, ES only) ---------------
def download_da_day_es_qh(date_str: str) -> pd.DataFrame:
    """
    Download & parse DA (marginalpdbc) for YYYYMMDD.
    If file is hourly, upsample x4 to QH (23->92, 24->96, 25->100).
    Returns: ['date','qh','es_price','market','session']
    """
    raw = _download_text(URL_DA.format(date=date_str))
    df_period = _parse_da_ida_raw_to_period_es(raw)
    df = _to_qh_from_period(df_period)  # already QH (upsampled if hourly)
    df["market"] = "DA"
    df["session"] = pd.NA
    return df[["date","qh","es_price","market","session"]]

# --------------- IDA (Intraday Auctions, ES only) ---------------
def _normalize_ida3_periods_to_absolute(df_period: pd.DataFrame) -> pd.DataFrame:
    """
    For IDA3, some files use relative time starting at 13:00:
      - If hourly and periods start at 1..N: hour_abs = hour + 12
      - If QH and periods start at 1..M:    qh_abs = qh_rel + 48
    If periods already absolute (hours >=13 or QH >=49), leave unchanged.
    Input: ['date','period','es_price'] where 'period' is hour or QH as-is.
    Output: same schema with 'period' adjusted to absolute scale when needed.
    """
    if df_period.empty:
        return df_period

    pmin, pmax = df_period["period"].min(), df_period["period"].max()

    # Hourly relative -> absolute hours (add 12)
    if pmax <= 30 and pmin == 1:
        df = df_period.copy()
        df["period"] = df["period"] + 12  # 1->13, 12->24 (DST handled implicitly)
        return df

    # QH relative -> absolute QH (add 48)
    if pmax <= 60 and pmin == 1:
        df = df_period.copy()
        df["period"] = df["period"] + 48  # 1.. -> 49..
        return df

    # Already absolute (hours ≥13 or QH ≥49)
    return df_period

def download_ida_session_es(date_str: str, session: int) -> pd.DataFrame:
    """
    Download & parse a single IDA session (1,2,3) for YYYYMMDD. ES only.
    - Detect hourly vs QH and upsample hourly to QH.
    - For IDA3, also normalize relative periods to absolute starting at 13:00.
    Returns ['date','qh','es_price','market','session']
    """
    if session not in (1, 2, 3):
        raise ValueError("session must be 1, 2, or 3")
    ss = f"{session:02d}"
    raw = _download_text(URL_IDA.format(date=date_str, sess=ss))
    df_period = _parse_da_ida_raw_to_period_es(raw)

    # Special handling for IDA3 relative periods
    if session == 3:
        df_period = _normalize_ida3_periods_to_absolute(df_period)

    df = _to_qh_from_period(df_period)  # QH (upsampled if hourly)
    df["market"] = "IDA"
    df["session"] = session
    return df[["date","qh","es_price","market","session"]]

def download_ida_all_sessions_es(date_str: str, sessions=(1,2,3), silent=False) -> pd.DataFrame:
    """Fetch multiple IDA sessions for the given date (skip missing ones)."""
    out = []
    for s in sessions:
        try:
            if not silent:
                print(f"  IDA{s:02d}...", end=" ")
            out.append(download_ida_session_es(date_str, s))
            if not silent:
                print("OK")
        except Exception as e:
            if not silent:
                print(f"skip ({e})")
    valid = [p for p in out if p is not None and not p.empty]
    if not valid:
        return pd.DataFrame(columns=["date","qh","es_price","market","session"])
    return pd.concat(valid, ignore_index=True, sort=False).sort_values(["date","market","session","qh"])

# --------------- IDC (Intraday Continuous, ES only: MedioES) ---------------
def _parse_idc_es(raw_text: str) -> pd.DataFrame:
    """
    Parse IDC (precios_pibcic) files. Handles both hourly ('Hora') and QH ('Periodo').
    Upsamples hourly to QH. Keeps only MedioES.
    Returns: ['date','qh','IDC_MedioES'].
    """
    rows = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        parts = s.split(";")
        if len(parts) < 13:
            continue
        y, m, d = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not (y.isdigit() and m.isdigit() and d.isdigit()):
            continue  # header/title
        period_str   = parts[3].strip()     # Hora or Periodo
        medio_es_str = parts[10].strip()    # MedioES
        try:
            period = int(period_str)
        except Exception:
            continue
        rows.append({
            "year": int(y),
            "month": int(m),
            "day": int(d),
            "period": period,
            "IDC_MedioES": _clean_numeric(medio_es_str),
        })
    if not rows:
        raise ValueError("No IDC data rows parsed")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))

    # Upsample hourly IDC to QH; keep QH as-is
    idc_qh = _upsample_hourly_block_to_qh(df[[ "date","period","IDC_MedioES" ]].copy(),
                                          period_col="period", value_col="IDC_MedioES", date_col="date")
    return idc_qh.sort_values(["date","qh"]).reset_index(drop=True)

def download_idc_es(date_str: str) -> pd.DataFrame:
    """Download & parse IDC (precios_pibcic) for YYYYMMDD. -> ['date','qh','IDC_MedioES']"""
    raw = _download_text(URL_IDC.format(date=date_str))
    return _parse_idc_es(raw)

# --------------- Merge & Pivot (ES + IDC MedioES) ---------------
def download_day_all_markets_es_wide(date_str: str, ida_sessions=(1,2,3)) -> pd.DataFrame:
    """
    Fetch DA (QH), IDA sessions (QH), and IDC (QH, MedioES), then return wide:
      date | qh | DA_ES_PRICE | IDA1_ES_PRICE | IDA2_ES_PRICE | IDA3_ES_PRICE | IDC_MedioES
    No forced number of periods. Avoids pandas FutureWarnings.
    """
    # ---- DA (price block) ----
    price_parts = []
    try:
        print(f"DA {date_str} ...", end=" ")
        price_parts.append(download_da_day_es_qh(date_str))
        print("OK")
    except Exception as e:
        print(f"DA skip ({e})")

    # ---- IDAs (price block) ----
    print(f"IDAs {date_str}:")
    price_parts.append(download_ida_all_sessions_es(date_str, sessions=ida_sessions, silent=False))
    price_parts = [p for p in price_parts if p is not None and not p.empty]

    # If DA/IDA missing, return IDC-only if present
    if not price_parts:
        base_cols = ["date","qh","DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"]
        base = pd.DataFrame(columns=base_cols)
        idc_df = None
        try:
            print(f"IDC {date_str} ...", end=" ")
            idc_df = download_idc_es(date_str)
            print("OK")
        except Exception as e:
            print(f"IDC skip ({e})")
        if idc_df is None or idc_df.empty:
            return base
        return idc_df.sort_values(["date","qh"]).reset_index(drop=True)

    # Concat only DA/IDA (real es_price)
    price_long = pd.concat(price_parts, ignore_index=True, sort=False)

    # Pivot ES auction prices (DA/IDAs) -> wide
    def _label(row):
        if row["market"] == "DA":
            return "DA_ES_PRICE"
        if row["market"] == "IDA":
            return f"IDA{int(row['session'])}_ES_PRICE"
        return None

    price_long = price_long.copy()
    price_long["label"] = price_long.apply(_label, axis=1)
    price_long = price_long.dropna(subset=["label"])

    wide_prices = price_long.pivot_table(
        index=["date","qh"], columns="label", values="es_price"
    ).reset_index()
    wide_prices.columns.name = None

    # ---- IDC (kept separate; no concat with price blocks) ----
    idc_df = None
    try:
        print(f"IDC {date_str} ...", end=" ")
        idc_df = download_idc_es(date_str)  # ['date','qh','IDC_MedioES']
        print("OK")
    except Exception as e:
        print(f"IDC skip ({e})")

    # Merge IDC (if present)
    if idc_df is not None and not idc_df.empty:
        wide = pd.merge(wide_prices, idc_df, on=["date","qh"], how="outer")
    else:
        wide = wide_prices

    # Order columns
    desired = ["date","qh","DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"]
    existing = [c for c in desired if c in wide.columns]
    remaining = [c for c in wide.columns if c not in existing]
    wide = wide[existing + remaining].sort_values(["date","qh"]).reset_index(drop=True)
    return wide

# --------------- Range helper (wide) ---------------
def download_range_es_wide(start_date: str, end_date: str, ida_sessions=(1,2,3)) -> pd.DataFrame:
    """
    Fetch DA + IDAs + IDC (ES only) for a date range [start_date, end_date] (YYYYMMDD).
    Returns one DataFrame with all days stacked (wide format).
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    end   = datetime.strptime(end_date,   "%Y%m%d")

    dfs = []
    cur = start
    while cur <= end:
        ds = cur.strftime("%Y%m%d")
        try:
            print(f"Processing {ds} ...")
            df = download_day_all_markets_es_wide(ds, ida_sessions=ida_sessions)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"skip {ds} ({e})")
        cur += timedelta(days=1)

    if not dfs:
        return pd.DataFrame(columns=["date","qh","DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"])

    out = pd.concat(dfs, ignore_index=True, sort=False)
    return out.sort_values(["date","qh"]).reset_index(drop=True)

#@title BESS Optimization

# ==== BESS sequential optimizer with explicit order tracking ====
# Inputs (per-day df):
#   date | qh | DA_ES_PRICE | IDA1_ES_PRICE | IDA2_ES_PRICE | IDA3_ES_PRICE | IDC_MedioES
#
# Model:
#   - P_MAX = 1 MW, E_MAX = 2 MWh, Δt = 0.25 h
#   - Exactly 8 charge QHs and 8 discharge QHs (full 1 MW) in the final schedule
#   - DA picks the initial 8+8; later markets can only change tradable QHs
#   - Changing a QH is modeled as: cancel previous order + place new order
#   - SOC(0)=SOC(T)=0, 0 ≤ SOC ≤ E_MAX, average SOC ≤ 50%
#   - Round-trip efficiency 85% applied as revenue loss on discharge (η on sells)
#
# Requires PuLP
# !pip install pulp

import numpy as np
import pandas as pd
import pulp

# Battery & market parameters
P_MAX = 1.0      # MW
E_MAX = 2.0      # MWh
DT    = 0.25     # h per QH
N_SIDE = int(round(E_MAX / (P_MAX * DT)))   # 8 QHs per side
ETA_DIS_REV = 0.85                          # all losses on discharge, applied to revenue
AVG_SOC_CAP = 0.5 * E_MAX                   # average SOC ≤ 50%

def _stage_optimize_discrete(df_day: pd.DataFrame,
                             price_col: str,
                             dis_prev: np.ndarray,
                             ch_prev:  np.ndarray) -> dict:
    """
    One market stage optimization with explicit 0/1 MW orders and previous schedule lock.
    - dis_prev, ch_prev are 0/1 MW from the previous stage.
    - Non-tradable QHs (NaN price) are fully locked: dis == dis_prev, ch == ch_prev.
    - dis, ch ∈ {0, P_MAX}; at most one active per QH; exactly N_SIDE per side total.
    - Objective is incremental revenue: sum price * (ETA*(dis-dis_prev) - (ch-ch_prev)) * DT

    Robustness:
      If the stage is infeasible or not Optimal, we carry forward the previous schedule,
      set objective=0.0, and return an empty orders list.
    """
    df = df_day.copy().sort_values("qh").reset_index(drop=True)
    prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy()
    T = len(prices)
    tradable = ~np.isnan(prices)

    m = pulp.LpProblem(f"Stage_{price_col}", pulp.LpMaximize)

    # Binary activity (1 -> place full 1 MW)
    b_dis = pulp.LpVariable.dicts("b_dis", range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)
    b_ch  = pulp.LpVariable.dicts("b_ch",  range(T), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Continuous convenience vars forced to 0 or P_MAX
    dis = pulp.LpVariable.dicts("dis", range(T), lowBound=0, upBound=P_MAX)
    ch  = pulp.LpVariable.dicts("ch",  range(T), lowBound=0, upBound=P_MAX)

    # State of charge
    soc = pulp.LpVariable.dicts("soc", range(T+1), lowBound=0, upBound=E_MAX)

    # Link binaries to power (exact 0 or P_MAX) + no simultaneous ch/dis
    for t in range(T):
        m += dis[t] == P_MAX * b_dis[t]
        m += ch[t]  == P_MAX * b_ch[t]
        m += b_dis[t] + b_ch[t] <= 1

    # Lock non-tradable quarters to previous schedule
    for t in range(T):
        if not tradable[t]:
            m += b_dis[t] == int(dis_prev[t] > 0.5 * P_MAX)
            m += b_ch[t]  == int(ch_prev[t]  > 0.5 * P_MAX)

    # Exactly N_SIDE per side (final valid orders remain 8+8)
    m += pulp.lpSum([b_dis[t] for t in range(T)]) == N_SIDE
    m += pulp.lpSum([b_ch[t]  for t in range(T)]) == N_SIDE

    # SOC dynamics (no energy efficiency; losses modeled in revenue)
    for t in range(T):
        m += soc[t+1] == soc[t] + DT * ch[t] - DT * dis[t]
    m += soc[0] == 0.0
    m += soc[T] == 0.0

    # Average SOC ≤ 50%
    m += (pulp.lpSum([soc[t] for t in range(T)]) / T) <= AVG_SOC_CAP

    # Incremental revenue objective (η on discharge deltas only)
    obj_terms = []
    for t in range(T):
        delta_dis = dis[t] - dis_prev[t]
        delta_ch  = ch[t]  - ch_prev[t]
        price     = 0.0 if np.isnan(prices[t]) else prices[t]
        obj_terms.append(DT * (ETA_DIS_REV * delta_dis - delta_ch) * price)
    m += pulp.lpSum(obj_terms)

    # Solve
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[m.status]

    def _pack(dis_v, ch_v, soc_v, Pg_v, obj, orders, status):
        # ensure numeric objective
        try:
            obj = float(obj)
            if pd.isna(obj):
                obj = 0.0
        except Exception:
            obj = 0.0
        return {
            "status": status,
            "objective": obj,
            "dis": dis_v, "ch": ch_v, "soc": soc_v, "Pg": Pg_v,
            "orders": orders
        }

    # Fallback if not optimal: keep previous schedule, zero increment, no orders
    if status != "Optimal":
        dis_v = dis_prev.copy()
        ch_v  = ch_prev.copy()
        # compute a consistent SOC trace from the carried schedule
        soc_v = np.zeros(T+1)
        for t in range(T):
            soc_v[t+1] = soc_v[t] + DT * ch_v[t] - DT * dis_v[t]
        Pg_v  = dis_v - ch_v
        return _pack(dis_v, ch_v, soc_v, Pg_v, 0.0, [], status)

    # Extract optimal solution
    dis_v = np.array([pulp.value(dis[t]) for t in range(T)])
    ch_v  = np.array([pulp.value(ch[t])  for t in range(T)])
    soc_v = np.array([pulp.value(soc[t]) for t in range(T+1)])
    Pg_v  = dis_v - ch_v
    obj   = pulp.value(m.objective)

    # Build explicit order list for this stage:
    #  - If dis increases: SELL new
    #  - If dis decreases: BUY  (cancel previous sell)
    #  - If ch  increases: BUY  new
    #  - If ch  decreases: SELL (cancel previous buy)
    orders = []
    for t in range(T):
        if dis_v[t] - dis_prev[t] > 0.5 * P_MAX:
            orders.append({"qh": int(df.loc[t, "qh"]), "side": "sell", "mw": P_MAX})
        if dis_prev[t] - dis_v[t] > 0.5 * P_MAX:
            orders.append({"qh": int(df.loc[t, "qh"]), "side": "buy",  "mw": P_MAX})
        if ch_v[t] - ch_prev[t] > 0.5 * P_MAX:
            orders.append({"qh": int(df.loc[t, "qh"]), "side": "buy",  "mw": P_MAX})
        if ch_prev[t] - ch_v[t] > 0.5 * P_MAX:
            orders.append({"qh": int(df.loc[t, "qh"]), "side": "sell", "mw": P_MAX})

    return _pack(dis_v, ch_v, soc_v, Pg_v, obj, orders, status)


def optimize_bess_day_sequential_orders(df_day: pd.DataFrame) -> dict:
    """
    Sequential run: DA -> IDA1 -> IDA2 -> IDA3 -> IDC
    Returns dict with schedules, earnings (DA + stage deltas), and per-stage orders.
    """
    cols = ["DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"]
    df = df_day.sort_values(["date","qh"]).reset_index(drop=True).copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    T = len(df)
    zeros = np.zeros(T)

    out = {}

    # ---- DA (prev = zeros) ----
    res_DA = _stage_optimize_discrete(df, "DA_ES_PRICE", zeros, zeros)
    out["DA_status"]   = res_DA["status"]
    out["DA_earnings"] = res_DA["objective"]
    out["DA_dis"]      = res_DA["dis"]; out["DA_ch"] = res_DA["ch"]; out["DA_Pg"] = res_DA["Pg"]
    out["DA_orders"]   = [{"market":"DA", **o} for o in res_DA["orders"]]

    # ---- IDA1 ----
    res_IDA1 = _stage_optimize_discrete(df, "IDA1_ES_PRICE", out["DA_dis"], out["DA_ch"])
    out["IDA1_status"]    = res_IDA1["status"]
    out["IDA1_increment"] = res_IDA1["objective"]
    out["IDA1_dis"]       = res_IDA1["dis"]; out["IDA1_ch"] = res_IDA1["ch"]; out["IDA1_Pg"] = res_IDA1["Pg"]
    out["IDA1_orders"]    = [{"market":"IDA1", **o} for o in res_IDA1["orders"]]

    # ---- IDA2 ----
    res_IDA2 = _stage_optimize_discrete(df, "IDA2_ES_PRICE", out["IDA1_dis"], out["IDA1_ch"])
    out["IDA2_status"]    = res_IDA2["status"]
    out["IDA2_increment"] = res_IDA2["objective"]
    out["IDA2_dis"]       = res_IDA2["dis"]; out["IDA2_ch"] = res_IDA2["ch"]; out["IDA2_Pg"] = res_IDA2["Pg"]
    out["IDA2_orders"]    = [{"market":"IDA2", **o} for o in res_IDA2["orders"]]

    # ---- IDA3 ----
    res_IDA3 = _stage_optimize_discrete(df, "IDA3_ES_PRICE", out["IDA2_dis"], out["IDA2_ch"])
    out["IDA3_status"]    = res_IDA3["status"]
    out["IDA3_increment"] = res_IDA3["objective"]
    out["IDA3_dis"]       = res_IDA3["dis"]; out["IDA3_ch"] = res_IDA3["ch"]; out["IDA3_Pg"] = res_IDA3["Pg"]
    out["IDA3_orders"]    = [{"market":"IDA3", **o} for o in res_IDA3["orders"]]

    # ---- IDC ----
    res_IDC = _stage_optimize_discrete(df, "IDC_MedioES", out["IDA3_dis"], out["IDA3_ch"])
    out["IDC_status"]     = res_IDC["status"]
    out["IDC_increment"]  = res_IDC["objective"]
    out["Final_dis"]      = res_IDC["dis"]; out["Final_ch"] = res_IDC["ch"]; out["Final_Pg"] = res_IDC["Pg"]
    out["IDC_orders"]     = [{"market":"IDC", **o} for o in res_IDC["orders"]]

    # Totals
    out["Total_earnings"] = out["DA_earnings"] + out["IDA1_increment"] + out["IDA2_increment"] + out["IDA3_increment"] + out["IDC_increment"]
    out["orders_all"] = out["DA_orders"] + out["IDA1_orders"] + out["IDA2_orders"] + out["IDA3_orders"] + out["IDC_orders"]
    return out

def optimize_bess_day_summary(df_day: pd.DataFrame) -> pd.DataFrame:
    """Return one-row summary: date | DA_Earnings | IDA1_Delta | IDA2_Delta | IDA3_Delta | IDC_Delta"""
    res = optimize_bess_day_sequential_orders(df_day)
    day = df_day["date"].iloc[0]
    if isinstance(day, pd.Timestamp): day = day.date()
    return pd.DataFrame([{
        "date": day,
        "DA_Earnings":  res["DA_earnings"],
        "IDA1_Delta":   res["IDA1_increment"],
        "IDA2_Delta":   res["IDA2_increment"],
        "IDA3_Delta":   res["IDA3_increment"],
        "IDC_Delta":    res["IDC_increment"],
    }])

def optimize_bess_range(df: pd.DataFrame) -> pd.DataFrame:
    """Run per-day and stack summaries."""
    outs = []
    for d, dd in df.groupby("date", sort=True):
        outs.append(optimize_bess_day_summary(dd.sort_values("qh").reset_index(drop=True)))
    return pd.concat(outs, ignore_index=True)

#@title Plotter

def plot_prices_net_and_trades_total_cancels(df_day, result,
                                             marker_size=40, eps=1e-6,
                                             step_where="post", cancel_pad=0.1,
                                             rte=0.85):
    """
    1) Stepwise prices
    2) Final net position bars (green/red). Discharge bars are scaled by RTE (default 0.85).
    3) Orders per market in rows + single dashed line from placement → cancel/flip (total position).
    """
    df = df_day.sort_values("qh").reset_index(drop=True)
    qh = df["qh"].to_numpy()

    # Prices
    p = {
        "DA":   df["DA_ES_PRICE"].to_numpy(),
        "IDA1": df["IDA1_ES_PRICE"].to_numpy(),
        "IDA2": df["IDA2_ES_PRICE"].to_numpy(),
        "IDA3": df["IDA3_ES_PRICE"].to_numpy(),
        "IDC":  df["IDC_MedioES"].to_numpy(),
    }

    # Discrete schedules (0/1 MW), net Pg, and sides
    DA_dis, DA_ch     = result["DA_dis"],   result["DA_ch"]
    IDA1_dis, IDA1_ch = result["IDA1_dis"], result["IDA1_ch"]
    IDA2_dis, IDA2_ch = result["IDA2_dis"], result["IDA2_ch"]
    IDA3_dis, IDA3_ch = result["IDA3_dis"], result["IDA3_ch"]
    IDC_dis, IDC_ch   = result["Final_dis"], result["Final_ch"]
    Pg_final          = np.asarray(result["Final_Pg"])

    stages = ["DA","IDA1","IDA2","IDA3","IDC"]
    row_y  = {"DA":4, "IDA1":3, "IDA2":2, "IDA3":1, "IDC":0}
    shapes = {"DA":"s", "IDA1":"^", "IDA2":"v", "IDA3":"D", "IDC":"o"}

    def side(dis, ch): return (dis > 0.5).astype(int) - (ch > 0.5).astype(int)
    S = {
        "DA":   side(DA_dis,   DA_ch),
        "IDA1": side(IDA1_dis, IDA1_ch),
        "IDA2": side(IDA2_dis, IDA2_ch),
        "IDA3": side(IDA3_dis, IDA3_ch),
        "IDC":  side(IDC_dis,  IDC_ch),
    }

    orders_by_stage = {
        "DA":   result.get("DA_orders", []),
        "IDA1": result.get("IDA1_orders", []),
        "IDA2": result.get("IDA2_orders", []),
        "IDA3": result.get("IDA3_orders", []),
        "IDC":  result.get("IDC_orders", []),
    }

    fig, (axP, axN, axO) = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios":[2.0, 1.1, 1.2]}
    )
    fig.subplots_adjust(right=0.80)

    # -- prices
    axP.step(qh, p["DA"],   where=step_where, label="DA",   linestyle="--", color="black")
    axP.step(qh, p["IDA1"], where=step_where, label="IDA1", color="royalblue")
    axP.step(qh, p["IDA2"], where=step_where, label="IDA2", color="seagreen")
    axP.step(qh, p["IDA3"], where=step_where, label="IDA3", color="darkorange")
    axP.step(qh, p["IDC"],  where=step_where, label="IDC",  color="crimson")
    axP.set_ylabel("Price [€/MWh]")
    axP.set_title(f"OMIE Market Prices ({df['date'].iloc[0].strftime('%Y-%m-%d')})")
    axP.grid(True, alpha=0.3)
    axP.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    # -- final net bars (apply RTE on discharge only)
    Pg_plot = np.where(Pg_final > 0, Pg_final * rte, Pg_final)
    axN.axhline(0, color="k", lw=0.8)
    axN.bar(qh, Pg_plot, width=0.9, color=np.where(Pg_plot>=0,"green","red"), alpha=0.6)
    axN.set_ylabel("BESS Net final position MW")
    axN.set_ylim(-1.2, 1.2)
    axN.grid(True, alpha=0.3)

    # -- orders panel
    for y in [row_y[s] for s in stages]:
        axO.axhline(y, color="lightgray", linewidth=1)

    for stage in stages:
        orders = orders_by_stage.get(stage, [])
        if not orders: continue
        qh_buy  = [o["qh"] for o in orders if o["side"] == "buy"]
        qh_sell = [o["qh"] for o in orders if o["side"] == "sell"]
        if qh_buy:
            axO.scatter(qh_buy,  [row_y[stage]]*len(qh_buy),
                        marker=shapes[stage], s=marker_size, color="red", edgecolor="none", zorder=5)
        if qh_sell:
            axO.scatter(qh_sell, [row_y[stage]]*len(qh_sell),
                        marker=shapes[stage], s=marker_size, color="green", edgecolor="none", zorder=5)

    # -- cancel/flip lines vs total position (from first placement to end of run)
    for t in range(len(qh)):
        seq = [S[s][t] for s in stages]  # -1,0,+1 across stages
        k = 0
        while k < len(stages):
            sgn = seq[k]
            if sgn == 0:
                k += 1
                continue
            if k == 0 or seq[k-1] != sgn:
                j = k + 1
                while j < len(stages) and seq[j] == sgn:
                    j += 1
                if j < len(stages):
                    yk, yj = row_y[stages[k]], row_y[stages[j]]
                    y1, y2 = (yk + cancel_pad, yj - cancel_pad) if yj > yk else (yk - cancel_pad, yj + cancel_pad)
                    axO.plot([qh[t], qh[t]], [y1, y2], ls="--", color="gray", alpha=0.5, lw=1.8,
                             solid_capstyle="round", zorder=10)
                k = j
            else:
                k += 1

    axO.set_yticks([row_y[s] for s in stages])
    axO.set_yticklabels(stages)
    axO.set_xlabel("Quarter-hour (qh)")
    axO.set_ylabel("Orders by Market")
    axO.set_xlim(qh.min(), qh.max())
    axO.set_ylim(-0.8, 4.2)
    axO.grid(True, axis="x", alpha=0.2)


    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None', color='w',
              markerfacecolor='green', markersize=10, label='Sell order'),
        Line2D([0],[0], marker='o', linestyle='None', color='w',
              markerfacecolor='red',   markersize=10, label='Buy order'),
        Line2D([0],[0], linestyle='--', color='black', label='Cancelled orders'),
    ]

    axO.legend(handles=legend_handles,
              loc="center left", bbox_to_anchor=(1.01, 0.5),
              title=None, ncols=1)

    plt.tight_layout()
    plt.show()

#@title JSON

# ===== Append/Update optimization summaries to a JSON file (records array) =====
# Stores rows with columns: date, DA_Earnings, IDA1_Delta, IDA2_Delta, IDA3_Delta, IDC_Delta


def _ensure_date_str(x):
    # return YYYY-MM-DD
    return pd.to_datetime(x).strftime("%Y-%m-%d")

def summary_df_to_records(df: pd.DataFrame):
    """
    Convert summary DF to list of dicts, rounding numeric values to 2 decimals.
    Keeps date as YYYY-MM-DD.
    """
    NUM_COLS = ["DA_Earnings","IDA1_Delta","IDA2_Delta","IDA3_Delta","IDC_Delta"]
    out = []
    for _, r in df.iterrows():
        rec = {"date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d")}
        for c in NUM_COLS:
            val = pd.to_numeric(r[c], errors="coerce")
            if pd.isna(val):
                rec[c] = None
            else:
                rec[c] = round(float(val), 2)   # <- 2 decimals
        out.append(rec)
    return out

def load_json_records(path: str):
    """Load existing JSON (array of objects). If not found/empty, return empty list."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # if someone stored a dict with 'records'
            if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
                return data["records"]
            return []
        except json.JSONDecodeError:
            return []

def save_json_records(path: str, records: list):
    """Write the full list back to disk (pretty printed)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def append_or_update_json(path: str, new_records: list):
    """
    Append/update by date key:
    - If a date already exists in the file, overwrite that row with the new one.
    - Otherwise, append as a new day.
    """
    existing = load_json_records(path)
    by_date = {rec["date"]: rec for rec in existing}
    for rec in new_records:
        by_date[rec["date"]] = rec   # overwrite or add
    # keep sorted by date
    all_recs = [by_date[d] for d in sorted(by_date.keys())]
    save_json_records(path, all_recs)
    return all_recs  # in case you want to inspect

def json_to_df(path: str) -> pd.DataFrame:
    """Read the JSON back into a DataFrame (sorted by date)."""
    recs = load_json_records(path)
    if not recs:
        return pd.DataFrame(columns=["date","DA_Earnings","IDA1_Delta","IDA2_Delta","IDA3_Delta","IDC_Delta"])
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df

# --- First day bootstrap: backfill a range and create the JSON ---
# (Pick your own dates)
#backfill_df = download_range_es_wide("20250101", "20250920", ida_sessions=(1,2,3))
#backfill_summary = optimize_bess_range(backfill_df)
#append_or_update_json(JSON_PATH, summary_df_to_records(backfill_summary))
#print("Backfill saved:", JSON_PATH)
#display(json_to_df(JSON_PATH))

if __name__ == "__main__":
    # Path to JSON file in same folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    JSON_PATH = os.path.join(BASE_DIR, "BESS OMIE Results.json")

    # Use yesterday’s date
    day = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
    print("Processing day:", day)

    # Download, optimize, update JSON
    df_day = download_day_all_markets_es_wide(day, ida_sessions=(1,2,3))
    if df_day.empty:
        print("⚠️ No data for", day)
    else:
        daily_summary = optimize_bess_day_summary(df_day)
        append_or_update_json(JSON_PATH, summary_df_to_records(daily_summary))
        print("✅ Updated JSON:", JSON_PATH)

        # Optional: show the last few records
        print(json_to_df(JSON_PATH).tail())
