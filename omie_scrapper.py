import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

# OMIE Data Functions

# ---------------- URLs ----------------
URL_DA   = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{date}.1"
URL_IDA  = "https://www.omie.es/es/file-download?parents=marginalpibc&filename=marginalpibc_{date}{sess}.1"
URL_IDC  = "https://www.omie.es/es/file-download?parents=precios_pibcic&filename=precios_pibcic_{date}.1"

def _download_text(url: str):
    """Return text on success; None if 404/timeout/any error."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except Exception:
        return None

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
    If max(period) <= 30 treat as hourly and repeat each hour 4x to QH indices.
    Returns ['date','qh',value_col]. Keeps partial-day blocks (e.g., IDA3 subset).
    """
    if df.empty:
        return pd.DataFrame(columns=[date_col, "qh", value_col])
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

def _parse_da_ida_raw_to_period_es(raw_text: str) -> pd.DataFrame:
    """
    Parse DA/IDA text into long df (Spain only) with 'period' (hour or QH as-is).
    Records: YYYY;MM;DD;PERIOD;PT;ES; (header & trailing '*'
    may be present and are stripped).
    Returns: ['date','period','es_price'].
    """
    if raw_text is None or raw_text.strip() == "":
        return pd.DataFrame(columns=["date","period","es_price"])

    lines = raw_text.strip().splitlines()
    lines = _strip_header_footer_lines(lines)

    if not lines:
        return pd.DataFrame(columns=["date","period","es_price"])

    buf = StringIO("\n".join(lines))
    df = pd.read_csv(buf, sep=";", header=None, engine="python", dtype=str)

    if df.shape[1] < 6:
        return pd.DataFrame(columns=["date","period","es_price"])

    df = df.iloc[:, :6]
    df.columns = ["year", "month", "day", "period", "pt_price", "es_price"]

    # Validate numeric rows; drop non-numeric
    mask_num = df["year"].str.isdigit() & df["month"].str.isdigit() & df["day"].str.isdigit() & df["period"].str.isdigit()
    df = df[mask_num].copy()
    if df.empty:
        return pd.DataFrame(columns=["date","period","es_price"])

    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["day"]   = df["day"].astype(int)
    df["period"]    = df["period"].astype(int)
    df["es_price"]  = df["es_price"].apply(_clean_numeric)
    df["date"]  = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))

    return df[["date","period","es_price"]].sort_values(["date","period"]).reset_index(drop=True)

def _to_qh_from_period(df: pd.DataFrame) -> pd.DataFrame:
    """Upsample if hourly; else keep QH. Returns ['date','qh','es_price']."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","qh","es_price"])
    return _upsample_hourly_block_to_qh(df.copy(), period_col="period", value_col="es_price", date_col="date")

# --------------- DA (Day-Ahead, ES only) ---------------
def download_da_day_es_qh(date_str: str) -> pd.DataFrame:
    """
    Download & parse DA (marginalpdbc) for YYYYMMDD.
    If file is missing, return empty df (filled with NaN later).
    """
    raw = _download_text(URL_DA.format(date=date_str))
    if raw is None:
        return pd.DataFrame(columns=["date","qh","es_price","market","session"])
    df_period = _parse_da_ida_raw_to_period_es(raw)
    df = _to_qh_from_period(df_period)
    df["market"] = "DA"
    df["session"] = pd.NA
    return df[["date","qh","es_price","market","session"]]

# --------------- IDA (Intraday Auctions, ES only) ---------------
def _normalize_ida3_periods_to_absolute(df_period: pd.DataFrame) -> pd.DataFrame:
    """
    For IDA3, some files use relative time starting at 13:00:
      - If hourly and periods start at 1..N: hour_abs = hour + 12
      - If QH and periods start at 1..M:    qh_abs = qh_rel + 48
    If already absolute (hours ≥13 or QH ≥49), leave unchanged.
    """
    if df_period is None or df_period.empty:
        return df_period

    pmin, pmax = df_period["period"].min(), df_period["period"].max()

    # Hourly relative -> absolute hours (add 12)
    if pmax <= 30 and pmin == 1:
        df = df_period.copy()
        df["period"] = df["period"] + 12  # 1->13, 12->24
        return df

    # QH relative -> absolute QH (add 48)
    if pmax <= 60 and pmin == 1:
        df = df_period.copy()
        df["period"] = df["period"] + 48  # 1.. -> 49..
        return df

    # Already absolute
    return df_period

def download_ida_session_es(date_str: str, session: int) -> pd.DataFrame:
    """Download & parse a single IDA session (1,2,3). If missing, empty df."""
    if session not in (1, 2, 3):
        raise ValueError("session must be 1, 2, or 3")
    ss = f"{session:02d}"
    raw = _download_text(URL_IDA.format(date=date_str, sess=ss))
    if raw is None:
        return pd.DataFrame(columns=["date","qh","es_price","market","session"])

    df_period = _parse_da_ida_raw_to_period_es(raw)
    if session == 3:
        df_period = _normalize_ida3_periods_to_absolute(df_period)

    df = _to_qh_from_period(df_period)
    df["market"] = "IDA"
    df["session"] = session
    return df[["date","qh","es_price","market","session"]]

def download_ida_all_sessions_es(date_str: str, sessions=(1,2,3), silent=False) -> pd.DataFrame:
    """Fetch multiple IDA sessions (skip missing); may return empty df."""
    outs = []
    for s in sessions:
        try:
            if not silent:
                print(f"  IDA{s:02d}...", end=" ")
            d = download_ida_session_es(date_str, s)
            outs.append(d)
            if not silent:
                print("OK" if not d.empty else "missing → NaN later")
        except Exception as e:
            if not silent:
                print(f"skip ({e})")
    valid = [p for p in outs if p is not None and not p.empty]
    if not valid:
        return pd.DataFrame(columns=["date","qh","es_price","market","session"])
    return pd.concat(valid, ignore_index=True, sort=False).sort_values(["date","market","session","qh"])

# --------------- IDC (Intraday Continuous, ES only: MedioES) ---------------
def _parse_idc_es(raw_text: str) -> pd.DataFrame:
    """Parse IDC (precios_pibcic); if nothing parsed, empty df."""
    if raw_text is None or raw_text.strip() == "":
        return pd.DataFrame(columns=["date","qh","IDC_MedioES"])

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
        return pd.DataFrame(columns=["date","qh","IDC_MedioES"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    idc_qh = _upsample_hourly_block_to_qh(df[["date","period","IDC_MedioES"]].copy(),
                                          period_col="period", value_col="IDC_MedioES", date_col="date")
    return idc_qh.sort_values(["date","qh"]).reset_index(drop=True)

def download_idc_es(date_str: str) -> pd.DataFrame:
    """Download & parse IDC (precios_pibcic) for YYYYMMDD. -> ['date','qh','IDC_MedioES']"""
    raw = _download_text(URL_IDC.format(date=date_str))
    if raw is None:
        return pd.DataFrame(columns=["date","qh","IDC_MedioES"])
    return _parse_idc_es(raw)

# --------------- Merge & Pivot (ES + IDC MedioES) ---------------
def download_day_all_markets_es_wide(date_str: str, ida_sessions=(1,2,3)) -> pd.DataFrame:
    """
    Always returns wide:
      date | qh | DA_ES_PRICE | IDA1_ES_PRICE | IDA2_ES_PRICE | IDA3_ES_PRICE | IDC_MedioES
    Missing source files => NaNs in their columns.
    QH skeleton is inferred from any available source; if none, default 1..96.
    """
    # ---- Fetch parts ----
    print(f"DA {date_str} ...", end=" ")
    da_df = download_da_day_es_qh(date_str)
    print("OK" if not da_df.empty else "missing → NaN later")

    print(f"IDAs {date_str}:")
    ida_df = download_ida_all_sessions_es(date_str, sessions=ida_sessions, silent=False)

    print(f"IDC {date_str} ...", end=" ")
    idc_df = download_idc_es(date_str)
    print("OK" if not idc_df.empty else "missing → NaN later")

    # ---- QH skeleton ----
    qh_sets = []
    if not da_df.empty:  qh_sets.append(set(da_df["qh"].unique()))
    if not ida_df.empty: qh_sets.append(set(ida_df["qh"].unique()))
    if not idc_df.empty: qh_sets.append(set(idc_df["qh"].unique()))
    if qh_sets:
        qhs = sorted(set.union(*qh_sets))
    else:
        qhs = list(range(1, 96 + 1))  # default skeleton when nothing available

    date_obj = pd.to_datetime(date_str, format="%Y%m%d")
    base = pd.DataFrame({"date": [date_obj]*len(qhs), "qh": qhs})

    # ---- Auction prices (DA + IDAs) -> wide ----
    price_parts = []
    if not da_df.empty:  price_parts.append(da_df)
    if not ida_df.empty: price_parts.append(ida_df)

    if price_parts:
        price_long = pd.concat(price_parts, ignore_index=True, sort=False)
        def _label(row):
            if row["market"] == "DA":
                return "DA_ES_PRICE"
            if row["market"] == "IDA":
                return f"IDA{int(row['session'])}_ES_PRICE"
            return None
        price_long = price_long.copy()
        price_long["label"] = price_long.apply(_label, axis=1)
        price_long = price_long.dropna(subset=["label"])
        wide_prices = price_long.pivot_table(index=["date","qh"], columns="label", values="es_price").reset_index()
        wide_prices.columns.name = None
        wide = base.merge(wide_prices, on=["date","qh"], how="left")
    else:
        wide = base.copy()

    # ---- IDC merge ----
    if not idc_df.empty:
        wide = wide.merge(idc_df, on=["date","qh"], how="left")
    else:
        wide["IDC_MedioES"] = np.nan

    # ---- Ensure columns exist & order
    desired = ["date","qh","DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"]
    for c in desired:
        if c not in wide.columns:
            wide[c] = np.nan
    remaining = [c for c in wide.columns if c not in desired]
    wide = wide[desired + remaining].sort_values(["date","qh"]).reset_index(drop=True)
    return wide

def download_range_es_wide(start_date: str, end_date: str, ida_sessions=(1,2,3)) -> pd.DataFrame:
    """
    Fetch DA + IDAs + IDC (ES only) for [start_date, end_date] (YYYYMMDD).
    Always returns full schema; NaNs filled where files are missing.
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
            dfs.append(df)
        except Exception as e:
            # Fallback skeleton: 96 QHs, all NaN prices
            date_obj = pd.to_datetime(ds, format="%Y%m%d")
            base = pd.DataFrame({"date": [date_obj]*96, "qh": list(range(1, 97))})
            for c in ["DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"]:
                base[c] = np.nan
            dfs.append(base)
            print(f"fallback skeleton for {ds} ({e})")
        cur += timedelta(days=1)

    out = pd.concat(dfs, ignore_index=True, sort=False)
    desired = ["date","qh","DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"]
    for c in desired:
        if c not in out.columns:
            out[c] = np.nan
    out = out[desired + [c for c in out.columns if c not in desired]]
    return out.sort_values(["date","qh"]).reset_index(drop=True)


P_MAX = 1.0
E_MAX = 2.0
DT    = 0.25
RTE   = 0.85
MARKET_EPS = 1e-3
MIN_LOT = 0.0  # honored when emitting orders only

# Split efficiency (exact RTE with symmetric split by default)
ETA_C = float(np.sqrt(RTE))
ETA_D = float(np.sqrt(RTE))

# -------------- Internal helpers --------------
def _sanitize_prev(dis_prev, ch_prev):
    dis0 = np.asarray(dis_prev, dtype=float).copy()
    ch0  = np.asarray(ch_prev,  dtype=float).copy()
    dis0[~np.isfinite(dis0)] = 0.0
    ch0[~np.isfinite(ch0)]   = 0.0
    dis0 = np.clip(dis0, 0.0, P_MAX)
    ch0  = np.clip(ch0,  0.0, P_MAX)
    tiny = 1e-12
    dis0[np.abs(dis0) < tiny] = 0.0
    ch0[np.abs(ch0)   < tiny] = 0.0
    return dis0, ch0

def _soc_from(dis_v, ch_v, eta_c=ETA_C, eta_d=ETA_D):
    T = len(dis_v)
    soc = np.zeros(T+1, dtype=float)
    for t in range(T):
        soc[t+1] = soc[t] + eta_c * ch_v[t] * DT - (dis_v[t] * DT) / eta_d
    return soc

def _delta_revenue(prices, dis_new, ch_new, dis_prev, ch_prev, tradable_mask):
    """
    Incremental revenue ONLY over tradable (finite-price) periods:
      sum_{t in trad} price[t] * ((dis_new - ch_new) - (dis_prev - ch_prev)) * DT
    Ignores NaN/±inf entirely (no 'price = 0' shortcuts).
    """
    prices = np.asarray(prices, dtype=float)
    trad   = np.asarray(tradable_mask, dtype=bool)
    dn = (np.asarray(dis_new, dtype=float) - np.asarray(ch_new, dtype=float))
    dp = (np.asarray(dis_prev, dtype=float) - np.asarray(ch_prev, dtype=float))

    idx = trad & np.isfinite(prices)
    if not np.any(idx):
        return 0.0

    delta_pg = dn[idx] - dp[idx]
    return float(DT * np.dot(prices[idx], delta_pg))

def _nn(x):  # clamp negatives to 0 for reporting
    try:
        v = float(x)
        return v if np.isfinite(v) and v > 0 else 0.0
    except Exception:
        return 0.0

# -------------- Core: one-cycle-max relaxed (non-consecutive) --------------
def _best_relaxed_one_cycle(prices: np.ndarray, tradable: np.ndarray) -> dict:
    """
    Find best plan with total SOC throughput <= E_MAX (i.e., up to one daily cycle),
    allowing scattered charge/discharge QHs, but enforcing:
      - all chosen CH QHs occur before split τ,
      - all chosen DIS QHs occur after τ,
      - SOC in [0, E_MAX], SOC(0)=SOC(T)=0.
    Returns dict: dis (MW), ch (MW), soc, revenue.
    """
    T = len(prices)
    prices = np.asarray(prices, dtype=float)
    trad   = np.asarray(tradable, dtype=bool)
    E_qh   = P_MAX * DT  # per-QH energy (MWh) at full power

    best_rev  = 0.0
    best_plan = None

    idx_all = np.arange(T)
    for tau in range(T-1):
        left  = idx_all[(idx_all <= tau) & trad]
        right = idx_all[(idx_all >  tau) & trad]
        if left.size == 0 or right.size == 0:
            continue

        left_sorted  = left[np.argsort(prices[left], kind="stable")]
        right_sorted = right[np.argsort(-prices[right], kind="stable")]
        kmax = min(left_sorted.size, right_sorted.size)

        spreads = []
        for k in range(kmax):
            s = float(prices[right_sorted[k]] - prices[left_sorted[k]] / (ETA_C * ETA_D))
            spreads.append(s)

        if not spreads or max(spreads) <= 0:
            continue

        ch = np.zeros(T, dtype=float)
        dis = np.zeros(T, dtype=float)
        soc = np.zeros(T+1, dtype=float)
        revenue = 0.0
        used_throughput_soc = 0.0
        cap_soc = E_MAX

        for k, s in enumerate(spreads):
            if s <= 0.0:
                break
            add_dis = E_qh
            possible_soc_after = used_throughput_soc + (add_dis / ETA_D)
            if possible_soc_after > cap_soc + 1e-12:
                add_dis = max(0.0, (cap_soc - used_throughput_soc) * ETA_D)
            if add_dis <= 1e-12:
                break

            i = left_sorted[k]
            j = right_sorted[k]
            dis[j] += add_dis / DT
            ch[i]  += (add_dis / (ETA_C * ETA_D)) / DT

            revenue += add_dis * prices[j] - (add_dis / (ETA_C * ETA_D)) * prices[i]
            used_throughput_soc += add_dis / ETA_D
            if used_throughput_soc >= cap_soc - 1e-12:
                break

        if revenue > best_rev + 1e-9:
            soc = _soc_from(dis, ch, ETA_C, ETA_D)
            if np.all(soc >= -1e-7) and np.all(soc <= E_MAX + 1e-7) and abs(soc[-1]) <= 1e-6:
                best_rev = revenue
                best_plan = (dis, ch, soc)

    if best_plan is None or best_rev <= 0.0:
        zeros = np.zeros(T)
        return {"dis": zeros, "ch": zeros, "soc": np.zeros(T+1), "revenue": 0.0}

    dis, ch, soc = best_plan

    if MIN_LOT > 0.0:
        dis[np.where(dis < MIN_LOT - 1e-12)] = 0.0
        ch[np.where(ch   < MIN_LOT - 1e-12)] = 0.0

    return {"dis": dis, "ch": ch, "soc": soc, "revenue": float(best_rev)}

# -------------- Compatibility wrapper: old stage API --------------
def _stage_optimize_discrete(df_day: pd.DataFrame,
                             price_col: str,
                             dis_prev: np.ndarray,
                             ch_prev:  np.ndarray,
                             solver_time_limit: float = 10.0,
                             solver_gap: float | None = 0.01,
                             solver_threads: int | None = None) -> dict:
    """
    Compatibility shim: computes a fresh best plan for price_col with relaxed one-cycle
    (throughput <= E_MAX), non-negative guard, and returns delta orders vs previous.
    """
    df = df_day.copy().sort_values("qh").reset_index(drop=True)
    prices = pd.to_numeric(df[price_col], errors="coerce").to_numpy()
    tradable = np.isfinite(prices)
    T = len(prices)

    # previous schedule
    dis0, ch0 = _sanitize_prev(dis_prev, ch_prev)

    # FIX: If the entire stage is non-tradable, carry forward previous schedule and emit NO orders
    if not np.any(tradable):  # all NaN/±inf
        return {
            "status": "NoMarket",
            "objective": 0.0,
            "dis": dis0.copy(),
            "ch": ch0.copy(),
            "soc": _soc_from(dis0, ch0, ETA_C, ETA_D),
            "Pg":  dis0 - ch0,
            "orders": []
        }

    # compute best plan from scratch (only uses tradable QHs)
    res = _best_relaxed_one_cycle(prices, tradable)

    # FIX: keep previous schedule on NON-tradable QHs (don't zero them → no spurious cancellations)
    dis_v = res["dis"].copy().astype(float)
    ch_v  = res["ch"].copy().astype(float)
    dis_v[~tradable] = dis0[~tradable]      # FIX
    ch_v[~tradable]  = ch0[~tradable]       # FIX

    soc_v = _soc_from(dis_v, ch_v, ETA_C, ETA_D)
    Pg_v  = dis_v - ch_v

    # non-negative incremental guard (masked)
    rev_delta = _delta_revenue(prices, dis_v, ch_v, dis0, ch0, tradable)
    if rev_delta < -1e-6:
        return {
            "status": "NoChange-Guard",
            "objective": 0.0,
            "dis": dis0.copy(),
            "ch": ch0.copy(),
            "soc": _soc_from(dis0, ch0, ETA_C, ETA_D),
            "Pg":  dis0 - ch0,
            "orders": []
        }
    rev_delta = max(0.0, rev_delta)

    # Build delta orders vs previous
    orders = []
    for t in range(T):
        qh_val = int(df.loc[t, "qh"])
        trad_t = bool(tradable[t])  # finite price?

        d_new, d_old = float(dis_v[t]), float(dis0[t])
        c_new, c_old = float(ch_v[t]),  float(ch0[t])

        # Only create NEW volume on tradable QHs; cancellations allowed only if tradable (or forbid? choose strict)
        # We choose: new only if tradable; cancellations also only if tradable to avoid any emission on missing markets.
        # FIX: require trad_t for both directions so missing markets produce zero orders.
        if trad_t and (d_new - d_old) > MARKET_EPS:
            mw = float(d_new - d_old)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "sell", "mw": mw})
        if trad_t and (d_old - d_new) > MARKET_EPS:
            mw = float(d_old - d_new)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "buy",  "mw": mw})

        if trad_t and (c_new - c_old) > MARKET_EPS:
            mw = float(c_new - c_old)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "buy",  "mw": mw})
        if trad_t and (c_old - c_new) > MARKET_EPS:
            mw = float(c_old - c_new)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "sell", "mw": mw})

    return {
        "status": "OK" if rev_delta > 0 else "DoNothing",
        "objective": rev_delta,
        "dis": dis_v, "ch": ch_v, "soc": soc_v, "Pg": Pg_v,
        "orders": orders
    }

def optimize_bess_day_sequential_orders(df_day: pd.DataFrame) -> dict:
    cols = ["DA_ES_PRICE","IDA1_ES_PRICE","IDA2_ES_PRICE","IDA3_ES_PRICE","IDC_MedioES"]
    df = df_day.sort_values(["date","qh"]).reset_index(drop=True).copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    T = len(df)
    zeros = np.zeros(T)
    out = {}

    res_DA = _stage_optimize_discrete(df, "DA_ES_PRICE", zeros, zeros)
    out["DA_status"]   = res_DA["status"]
    out["DA_earnings"] = _nn(res_DA["objective"])
    out["DA_dis"]      = res_DA["dis"]; out["DA_ch"] = res_DA["ch"]; out["DA_Pg"] = res_DA["Pg"]
    out["DA_orders"]   = [{"market":"DA", **o} for o in res_DA["orders"]]

    res_IDA1 = _stage_optimize_discrete(df, "IDA1_ES_PRICE", out["DA_dis"], out["DA_ch"])
    out["IDA1_status"]    = res_IDA1["status"]
    out["IDA1_increment"] = _nn(res_IDA1["objective"])
    out["IDA1_dis"]       = res_IDA1["dis"]; out["IDA1_ch"] = res_IDA1["ch"]; out["IDA1_Pg"] = res_IDA1["Pg"]
    out["IDA1_orders"]    = [{"market":"IDA1", **o} for o in res_IDA1["orders"]]

    res_IDA2 = _stage_optimize_discrete(df, "IDA2_ES_PRICE", out["IDA1_dis"], out["IDA1_ch"])
    out["IDA2_status"]    = res_IDA2["status"]
    out["IDA2_increment"] = _nn(res_IDA2["objective"])
    out["IDA2_dis"]       = res_IDA2["dis"]; out["IDA2_ch"] = res_IDA2["ch"]; out["IDA2_Pg"] = res_IDA2["Pg"]
    out["IDA2_orders"]    = [{"market":"IDA2", **o} for o in res_IDA2["orders"]]

    res_IDA3 = _stage_optimize_discrete(df, "IDA3_ES_PRICE", out["IDA2_dis"], out["IDA2_ch"])
    out["IDA3_status"]    = res_IDA3["status"]
    out["IDA3_increment"] = _nn(res_IDA3["objective"])
    out["IDA3_dis"]       = res_IDA3["dis"]; out["IDA3_ch"] = res_IDA3["ch"]; out["IDA3_Pg"] = res_IDA3["Pg"]
    out["IDA3_orders"]    = [{"market":"IDA3", **o} for o in res_IDA3["orders"]]

    res_IDC = _stage_optimize_discrete(df, "IDC_MedioES", out["IDA3_dis"], out["IDA3_ch"])
    out["IDC_status"]     = res_IDC["status"]
    out["IDC_increment"]  = _nn(res_IDC["objective"])
    out["Final_dis"]      = res_IDC["dis"]; out["Final_ch"] = res_IDC["ch"]; out["Final_Pg"] = res_IDC["Pg"]
    out["IDC_orders"]     = [{"market":"IDC", **o} for o in res_IDC["orders"]]

    out["Total_earnings"] = (out["DA_earnings"] + out["IDA1_increment"] +
                             out["IDA2_increment"] + out["IDA3_increment"] +
                             out["IDC_increment"])
    out["orders_all"] = (out["DA_orders"] + out["IDA1_orders"] +
                         out["IDA2_orders"] + out["IDA3_orders"] +
                         out["IDC_orders"])
    return out

def optimize_bess_day_summary(df_day: pd.DataFrame,
                              solver_time_limit: float | None = None,
                              solver_gap: float | None = None,
                              solver_threads: int | None = None) -> pd.DataFrame:
    """Compatibility one-row summary."""
    res = optimize_bess_day_sequential_orders(df_day)
    day = df_day["date"].iloc[0]
    if isinstance(day, pd.Timestamp): day = day.date()
    return pd.DataFrame([{
        "date": day,
        "DA_Earnings":  _nn(res.get("DA_earnings", 0.0)),
        "IDA1_Delta":   _nn(res.get("IDA1_increment", 0.0)),
        "IDA2_Delta":   _nn(res.get("IDA2_increment", 0.0)),
        "IDA3_Delta":   _nn(res.get("IDA3_increment", 0.0)),
        "IDC_Delta":    _nn(res.get("IDC_increment", 0.0)),
    }])

def optimize_bess_range(df: pd.DataFrame,
                        solver_time_limit: float = 2.0,
                        solver_gap: float | None = 0.005,
                        solver_threads: int | None = None,
                        progress: bool = True) -> pd.DataFrame:
    outs = []
    all_days = pd.to_datetime(df["date"]).dt.date.unique()
    all_days = sorted(all_days)
    n = len(all_days)
    for i, d in enumerate(all_days, start=1):
        dd = df[df["date"].astype(str).str[:10].isin([str(d)])].sort_values("qh").reset_index(drop=True)
        if progress:
            print(f"[{i}/{n}] {d} … running", flush=True)
        day_df = optimize_bess_day_summary(dd, solver_time_limit, solver_gap, solver_threads)
        outs.append(day_df)
        if progress:
            tot = float(day_df.iloc[0][["DA_Earnings","IDA1_Delta","IDA2_Delta","IDA3_Delta","IDC_Delta"]].sum())
            print(f"[{i}/{n}] {d} ✓ total={tot:.2f}", flush=True)
    if not outs:
        return pd.DataFrame(columns=["date","DA_Earnings","IDA1_Delta","IDA2_Delta","IDA3_Delta","IDC_Delta"])
    return pd.concat(outs, ignore_index=True)

#@title plotter
def plot_prices_net_and_trades_total_cancels(df_day, result,
                                             marker_size=40, eps=1e-6,
                                             step_where="post", cancel_pad=0.1,
                                             rte=0.85,
                                             bar_width=0.9,
                                             save_png=True,
                                             save_svg=True,
                                             save_dir=None,
                                             filename=None,
                                             filename_svg=None,
                                             dpi=220):
    """
    1) Stepwise prices
    2) Final net position bars (green/red). Discharge bars are scaled by RTE (default 0.85).
       + SoC line (0–100%) on right axis, aligned to bar *edges* (period edges).
       + Left & right axes share the same visual headroom; top/bottom ticks shown.
    3) Orders per market in rows + dashed lines for cancel/flip (now piecewise:
       every alternation across stages is connected).

    Uses MARKET_EPS (if defined) as the single tolerance for activity/cancellation;
    otherwise falls back to `eps`.
    """
    # ---- unified tolerance (same as optimizer)
    try:
        mw_eps = float(MARKET_EPS)
    except NameError:
        mw_eps = float(eps)

    df = df_day.sort_values("qh").reset_index(drop=True)
    qh = df["qh"].to_numpy()
    T  = len(qh)

    # Prices (KEEPING YOUR COLORS & TITLES BELOW)
    p = {
        "DA":   df["DA_ES_PRICE"].to_numpy(),
        "IDA1": df["IDA1_ES_PRICE"].to_numpy(),
        "IDA2": df["IDA2_ES_PRICE"].to_numpy(),
        "IDA3": df["IDA3_ES_PRICE"].to_numpy(),
        "IDC":  df["IDC_MedioES"].to_numpy(),
    }

    # Final schedules and net Pg
    DA_dis, DA_ch     = result["DA_dis"],   result["DA_ch"]
    IDA1_dis, IDA1_ch = result["IDA1_dis"], result["IDA1_ch"]
    IDA2_dis, IDA2_ch = result["IDA2_dis"], result["IDA2_ch"]
    IDA3_dis, IDA3_ch = result["IDA3_dis"], result["IDA3_ch"]
    IDC_dis, IDC_ch   = result["Final_dis"], result["Final_ch"]
    Pg_final          = np.asarray(result["Final_Pg"], dtype=float)

    stages = ["DA","IDA1","IDA2","IDA3","IDC"]
    row_y  = {"DA":4, "IDA1":3, "IDA2":2, "IDA3":1, "IDC":0}
    shapes = {"DA":"s", "IDA1":"^", "IDA2":"v", "IDA3":"D", "IDC":"o"}

    # ---------- volume-aware net + signs with unified tolerance ----------
    def stage_net(dis, ch):
        return np.asarray(dis, dtype=float) - np.asarray(ch, dtype=float)

    NET = {
        "DA":   stage_net(DA_dis,   DA_ch),
        "IDA1": stage_net(IDA1_dis, IDA1_ch),
        "IDA2": stage_net(IDA2_dis, IDA2_ch),
        "IDA3": stage_net(IDA3_dis, IDA3_ch),
        "IDC":  stage_net(IDC_dis,  IDC_ch),
    }
    def sign_from_net(x, thr):
        s = np.zeros_like(x, dtype=int)
        s[x >  thr] = +1
        s[x < -thr] = -1
        return s
    S = {k: sign_from_net(v, mw_eps) for k, v in NET.items()}

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

    # -- prices (KEEPING YOUR COLORS)
    axP.step(qh, p["DA"],   where=step_where, label="DA",   linestyle="--", color="black")
    axP.step(qh, p["IDA1"], where=step_where, label="IDA1", color="#263cc8")
    axP.step(qh, p["IDA2"], where=step_where, label="IDA2", color="#7c9599")
    axP.step(qh, p["IDA3"], where=step_where, label="IDA3", color="#6d32ff")
    axP.step(qh, p["IDC"],  where=step_where, label="IDC",  color="#28ff52")
    axP.set_ylabel("Price [€/MWh]")
    title_date = df['date'].iloc[0]
    if hasattr(title_date, "strftime"):
        title_date = title_date.strftime("%Y-%m-%d")
    axP.set_title(f"OMIE Market Prices ({title_date})")
    axP.grid(True, alpha=0.3)
    axP.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    # --- period edges to align SoC with bar edges
    half_w = bar_width / 2.0
    edges = np.concatenate(([qh[0] - half_w], qh + half_w))  # length T+1

    # -- final net bars (apply RTE on discharge only; zero tiny residuals first)
    Pg_disp = Pg_final.copy().astype(float)
    Pg_disp[np.abs(Pg_disp) <= mw_eps] = 0.0
    Pg_plot = np.where(Pg_disp > 0, Pg_disp * rte, Pg_disp)

    axN.axhline(0, color="k", lw=0.8)
    axN.bar(qh, Pg_plot, width=bar_width, color=np.where(Pg_plot>=0,"green","red"), alpha=0.6)
    axN.set_ylabel("Final Market Orders [p.u.]")
    axN.set_ylim(-1.2, 1.2)
    axN.grid(True, alpha=0.3)

    # -- SoC % line on right axis (√RTE physics), aligned to edges
    SQRT_RTE = float(np.sqrt(rte))
    soc = np.zeros(T + 1, dtype=float)   # soc[k] at edges[k]
    for t in range(T):
        soc[t+1] = soc[t] + (DT * SQRT_RTE) * IDC_ch[t] - (DT / SQRT_RTE) * IDC_dis[t]
    soc_pct_edges = 100.0 * (soc / float(E_MAX))
    soc_pct_edges = np.clip(soc_pct_edges, 0.0, 100.0)

    axN2 = axN.twinx()
    axN2.plot(edges, soc_pct_edges, linewidth=2.0, linestyle="-", color="black", label="BESS SOC (%)")
    axN2.set_ylabel("SOC [%]")

    # === Headroom sync
    left_top, left_bot = axN.get_ylim()[1], axN.get_ylim()[0]
    pad_frac = (left_top - 1.0) / 1.0 if left_top > 0 else 0.0
    soc_lo, soc_hi = 0.0, 100.0
    axN2.set_ylim(soc_lo - pad_frac * (soc_hi - soc_lo),
                  soc_hi + pad_frac * (soc_hi - soc_lo))

    axN.tick_params(axis='y', direction='inout', length=6)
    axN2.tick_params(axis='y', direction='inout', length=6)

    # -- orders panel
    for y in [row_y[s] for s in stages]:
        axO.axhline(y, color="lightgray", linewidth=1)

    for stage in stages:
        orders = orders_by_stage.get(stage, [])
        if not orders:
            continue
        qh_buy  = [o["qh"] for o in orders if o["side"] == "buy"]
        qh_sell = [o["qh"] for o in orders if o["side"] == "sell"]
        if qh_buy:
            axO.scatter(qh_buy,  [row_y[stage]]*len(qh_buy),
                        marker=shapes[stage], s=marker_size, color="red", edgecolor="none", zorder=5)
        if qh_sell:
            axO.scatter(qh_sell, [row_y[stage]]*len(qh_sell),
                        marker=shapes[stage], s=marker_size, color="green", edgecolor="none", zorder=5)

    # -- PIECEWISE cancel/flip connectors (draw every alternation)
    for t in range(T):
        s_prev = 0
        k_start = None
        for j, st in enumerate(stages):
            s_now = S[st][t]  # -1,0,+1 with tolerance
            # start of an active segment
            if s_prev == 0 and s_now != 0:
                k_start = j
                s_prev  = s_now
                continue
            # end of segment: turned off or flipped sign
            if s_prev != 0 and (s_now == 0 or s_now != s_prev):
                yk, yj = row_y[stages[k_start]], row_y[st]
                y1, y2 = (yk + cancel_pad, yj - cancel_pad) if yj > yk else (yk - cancel_pad, yj + cancel_pad)
                axO.plot([qh[t], qh[t]], [y1, y2], ls="--", color="gray", alpha=0.5, lw=1.8,
                         solid_capstyle="round", zorder=10)
                # if flipped, start a new segment here
                if s_now != 0:
                    k_start = j
                    s_prev  = s_now
                else:
                    k_start = None
                    s_prev  = 0

    axO.set_yticks([row_y[s] for s in stages])
    axO.set_yticklabels(stages)
    axO.set_xlabel("Quarter-hour (qh)")
    axO.set_ylabel("Orders by Market")
    axO.set_xlim(qh.min(), qh.max())
    axO.set_ylim(-0.8, 4.2)
    axO.grid(True, axis="x", alpha=0.2)

    # --- LEGEND FOR THE ORDERS PANEL ---
    legend_handles = [
        Line2D([0],[0], marker='o', linestyle='None', color='w',
               markerfacecolor='green', markersize=10, label='Sell order'),
        Line2D([0],[0], marker='o', linestyle='None', color='w',
               markerfacecolor='red',   markersize=10, label='Buy order'),
        Line2D([0],[0], linestyle='--', color='gray', label='Cancelled orders'),
    ]
    axO.legend(handles=legend_handles,
               loc="center left", bbox_to_anchor=(1.01, 0.5),
               title=None, ncols=1)

    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    # Check if we have any data, otherwise put a message in the figure
    try:
        has_any_price = np.isfinite(df_day.get("DA_ES_PRICE", np.nan)).any() \
                        or np.isfinite(df_day.get("IDA1_ES_PRICE", np.nan)).any() \
                        or np.isfinite(df_day.get("IDA2_ES_PRICE", np.nan)).any() \
                        or np.isfinite(df_day.get("IDA3_ES_PRICE", np.nan)).any() \
                        or np.isfinite(df_day.get("IDC_MedioES", np.nan)).any()
    except Exception:
        has_any_price = True

    if not has_any_price:
        fig.text(0.5, 0.5, "No price data available",
                 ha="center", va="center", fontsize=16)

    # Force render to avoid blank SVGs in headless runners
    fig.canvas.draw()

    if save_png:
        if filename is None:
            filename = f"OMIE_BESS_{title_date}.png"
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, dpi=dpi,
                    bbox_inches="tight", facecolor="white")
        print(f"Saved PNG to: {out_path}")

    if save_svg:
        if filename_svg is None:
            if filename and filename.lower().endswith(".png"):
                filename_svg = filename[:-4] + ".svg"
            else:
                filename_svg = f"OMIE_BESS_{title_date}.svg"
        out_svg = os.path.join(save_dir, filename_svg)
        fig.savefig(out_svg, format="svg",
                    bbox_inches="tight", facecolor="white")
        print(f"Saved SVG to: {out_svg}")

    plt.close(fig)

    plt.close(fig)  # IMPORTANT: avoid leaking a blank figure into the next save
    # Keep x-limits consistent with edges so the SoC line spans exactly the bars
    axN.set_xlim(edges[0], edges[-1])
    axN2.set_xlim(edges[0], edges[-1])

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

if __name__ == "__main__":
    # Path to JSON & SVG in the repo root (same folder as this file)
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    JSON_PATH = os.path.join(BASE_DIR, "BESS OMIE Results.json")
    PLOT_PATH = os.path.join(BASE_DIR, "OMIE_BESS.svg")

    # Yesterday’s date
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
        print(json_to_df(JSON_PATH).tail())

        # Save SVG right next to JSON (directory = BASE_DIR, filename = OMIE_BESS.svg)
        try:
            plot_prices_net_and_trades_total_cancels(
                df_day,
                result=daily_summary,
                save_png=False,
                save_svg=True,
                save_dir=BASE_DIR,         
                filename_svg="OMIE_BESS.svg",
                dpi=220
            )
            print(f"✅ Saved SVG to: {PLOT_PATH}")
        except Exception as e:
            print(f"⚠️ Could not save SVG via plotter: {e}")
            # Fallback: try saving whatever current figure exists
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.savefig(PLOT_PATH, format="svg", bbox_inches="tight", facecolor="white")
                print(f"✅ Fallback saved SVG to: {PLOT_PATH}")
            except Exception as ee:
                print(f"❌ Fallback save failed: {ee}")
