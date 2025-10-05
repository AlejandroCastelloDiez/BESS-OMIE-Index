import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from datetime import datetime, timedelta
from io import StringIO
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import os, json
from typing import Dict, Any

#@title OMIE Data Functions
# ---------------- URLs ----------------
URL_DA   = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{date}.1"
URL_IDA  = "https://www.omie.es/es/file-download?parents=marginalpibc&filename=marginalpibc_{date}{sess}.1"
URL_IDC  = "https://www.omie.es/es/file-download?parents=precios_pibcic&filename=precios_pibcic_{date}.1"

def _download_text(url: str):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def _clean_numeric(s: str) -> float:
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


# ==================== OPTIMIZATION (τ-greedy with POI power caps) ====================

P_MAX = 1.0         # MW
E_MAX = 2.0         # MWh usable energy
DT    = 0.25        # h per interval (quarter-hour)
RTE   = 0.85        # round-trip efficiency
MARKET_EPS = 1e-3   # MW threshold when emitting orders
MIN_LOT    = 0.0    # MW minimum tradable lot

# Per-QH power caps at point of interconnection (POI)
P_CH_MAX  = P_MAX           # MW cap for CHARGE orders at POI
P_DIS_MAX = P_MAX * RTE     # MW cap for DISCHARGE orders at POI (export already includes losses)

ETA_C = float(np.sqrt(RTE))
ETA_D = float(np.sqrt(RTE))

# -------------------- Internal helpers --------------------
def _sanitize_prev(dis_prev, ch_prev):
    dis0 = np.asarray(dis_prev, dtype=float).copy()
    ch0  = np.asarray(ch_prev,  dtype=float).copy()
    dis0[~np.isfinite(dis0)] = 0.0
    ch0[~np.isfinite(ch0)]   = 0.0
    # Respect POI caps
    dis0 = np.clip(dis0, 0.0, P_DIS_MAX)
    ch0  = np.clip(ch0, 0.0, P_CH_MAX)
    tiny = 1e-12
    dis0[np.abs(dis0) < tiny] = 0.0
    ch0[np.abs(ch0)   < tiny] = 0.0
    return dis0, ch0

def _soc_from(dis_v, ch_v, eta_c=ETA_C, eta_d=ETA_D):
    T = len(dis_v)
    soc = np.zeros(T + 1, dtype=float)
    for t in range(T):
        # SOC measured in DC MWh inside the battery:
        #   + ηc * ch_v[t] * DT    (charging raises SOC)
        #   - (dis_v[t] * DT)/ηd   (discharging lowers SOC)
        soc[t+1] = soc[t] + eta_c * ch_v[t] * DT - (dis_v[t] * DT) / eta_d
    return soc

def _delta_revenue(prices, dis_new, ch_new, dis_prev, ch_prev, tradable_mask):
    prices = np.asarray(prices, dtype=float)
    trad   = np.asarray(tradable_mask, dtype=bool)
    dn = (np.asarray(dis_new, dtype=float) - np.asarray(ch_new, dtype=float))
    dp = (np.asarray(dis_prev, dtype=float) - np.asarray(ch_prev, dtype=float))
    idx = trad & np.isfinite(prices)
    if not np.any(idx):
        return 0.0
    delta_pg = dn[idx] - dp[idx]
    return float(DT * np.dot(prices[idx], delta_pg))

def _nn(x):
    try:
        v = float(x)
        return v if np.isfinite(v) and v > 0 else 0.0
    except Exception:
        return 0.0

# -------------------- Core: τ-greedy relaxed one-cycle --------------------
def _best_relaxed_one_cycle(prices: np.ndarray, tradable: np.ndarray) -> Dict[str, Any]:
    """
    SOC throughput ≤ E_MAX, charging before τ, discharging after τ.
    Honors per-QH power caps: P_CH_MAX on charge, P_DIS_MAX on discharge.

    Charge-centric pairing:
      Spread per MWh bought = p_sell * RTE - p_buy
      We select charge MWh first; discharged MWh = charge_mwh * RTE.
    """
    T = len(prices)
    prices = np.asarray(prices, dtype=float)
    trad   = np.asarray(tradable, dtype=bool)

    best_rev  = 0.0
    best_plan = None
    idx_all   = np.arange(T)

    rte = ETA_C * ETA_D               # round-trip efficiency
    E_QH_CH_MAX  = P_CH_MAX  * DT     # max MWh that can be charged in one QH at POI
    E_QH_DIS_MAX = P_DIS_MAX * DT     # max MWh that can be discharged in one QH at POI

    for tau in range(T - 1):
        left  = idx_all[(idx_all <= tau) & trad]   # charge candidates
        right = idx_all[(idx_all >  tau) & trad]   # discharge candidates
        if left.size == 0 or right.size == 0:
            continue

        # Cheapest buys left, richest sells right
        left_sorted  = left[np.argsort(prices[left],  kind="stable")]
        right_sorted = right[np.argsort(-prices[right], kind="stable")]
        kmax = min(left_sorted.size, right_sorted.size)

        # Pairwise spreads per 1 MWh bought
        spreads = []
        for k in range(kmax):
            i = left_sorted[k]
            j = right_sorted[k]
            spreads.append(float(prices[j] * rte - prices[i]))
        if not spreads or max(spreads) <= 0.0:
            continue

        ch  = np.zeros(T, dtype=float)  # MW at POI
        dis = np.zeros(T, dtype=float)  # MW at POI
        soc = np.zeros(T + 1, dtype=float)
        revenue        = 0.0
        used_soc_peak  = 0.0
        cap_soc        = E_MAX

        for k, s in enumerate(spreads):
            if s <= 0.0:
                break

            i = left_sorted[k]
            j = right_sorted[k]

            # Remaining SOC headroom in DC MWh
            headroom_soc = max(0.0, cap_soc - used_soc_peak)

            # Max charge MWh we can buy at i (limited by: POI charge cap, POI discharge cap after losses, SOC headroom)
            ch_cap_mwh              = E_QH_CH_MAX
            ch_cap_by_dis_power_mwh = E_QH_DIS_MAX / max(rte, 1e-12)
            ch_cap_by_soc_mwh       = headroom_soc / max(ETA_C, 1e-12)
            add_ch_mwh = min(ch_cap_mwh, ch_cap_by_dis_power_mwh, ch_cap_by_soc_mwh)

            if add_ch_mwh <= 1e-12:
                continue

            add_dis_mwh = add_ch_mwh * rte  # MWh exported at POI in QH j

            # Write MW, staying within POI caps
            ch[i]  += min(P_CH_MAX,  add_ch_mwh / DT)
            dis[j] += min(P_DIS_MAX, add_dis_mwh / DT)

            # Revenue at POI
            revenue += add_dis_mwh * prices[j] - add_ch_mwh * prices[i]

            # SOC increases by ηc * ch_mwh
            used_soc_peak += ETA_C * add_ch_mwh
            if used_soc_peak >= cap_soc - 1e-12:
                break

        if revenue > best_rev + 1e-9:
            soc = _soc_from(dis, ch, ETA_C, ETA_D)
            # Feasibility checks
            if np.all(soc >= -1e-7) and np.all(soc <= E_MAX + 1e-7) and abs(soc[-1]) <= 1e-6:
                best_rev  = revenue
                best_plan = (dis, ch, soc)

    if best_plan is None or best_rev <= 0.0:
        zeros = np.zeros(T)
        return {"dis": zeros, "ch": zeros, "soc": np.zeros(T + 1), "revenue": 0.0}

    dis, ch, soc = best_plan
    if MIN_LOT > 0.0:
        dis[np.where(dis < MIN_LOT - 1e-12)] = 0.0
        ch[np.where(ch   < MIN_LOT - 1e-12)] = 0.0
    return {"dis": dis, "ch": ch, "soc": soc, "revenue": float(best_rev)}

# -------------------- Stage optimizer (per market) --------------------
def _stage_optimize_discrete(df_day: pd.DataFrame,
                             price_col: str,
                             dis_prev: np.ndarray,
                             ch_prev:  np.ndarray) -> Dict[str, Any]:
    """
    Compute a fresh τ-greedy plan for 'price_col' and return delta orders vs previous.
    Preserves previous schedule on non-tradable QHs (NaN/±inf prices).
    Enforces POI caps on both charge and discharge.
    """
    df = df_day.copy().sort_values("qh").reset_index(drop=True)
    prices   = pd.to_numeric(df[price_col], errors="coerce").to_numpy()
    tradable = np.isfinite(prices)
    T = len(prices)

    dis0, ch0 = _sanitize_prev(dis_prev, ch_prev)

    if not np.any(tradable):
        soc0 = _soc_from(dis0, ch0, ETA_C, ETA_D)
        return {
            "status": "NoMarket",
            "objective": 0.0,
            "dis": dis0.copy(),
            "ch": ch0.copy(),
            "soc": soc0,
            "Pg":  dis0 - ch0,
            "orders": []
        }

    res = _best_relaxed_one_cycle(prices, tradable)

    # Keep previous schedule where market is missing
    dis_v = res["dis"].astype(float)
    ch_v  = res["ch"].astype(float)
    dis_v[~tradable] = dis0[~tradable]
    ch_v[~tradable]  = ch0[~tradable]

    # Enforce per-interval POI caps (safety)
    dis_v = np.clip(dis_v, 0.0, P_DIS_MAX)
    ch_v  = np.clip(ch_v,  0.0, P_CH_MAX)

    soc_v = _soc_from(dis_v, ch_v, ETA_C, ETA_D)
    Pg_v  = dis_v - ch_v

    # Non-negative incremental revenue guard
    rev_delta = _delta_revenue(prices, dis_v, ch_v, dis0, ch0, tradable)
    if rev_delta < -1e-6:
        soc0 = _soc_from(dis0, ch0, ETA_C, ETA_D)
        return {
            "status": "NoChange-Guard",
            "objective": 0.0,
            "dis": dis0.copy(),
            "ch": ch0.copy(),
            "soc": soc0,
            "Pg":  dis0 - ch0,
            "orders": []
        }
    rev_delta = max(0.0, rev_delta)

    # Build incremental orders (only on tradable QHs)
    orders = []
    for t in range(T):
        if not tradable[t]:
            continue
        qh_val = int(df.loc[t, "qh"])
        d_new, d_old = float(dis_v[t]), float(dis0[t])
        c_new, c_old = float(ch_v[t]),  float(ch0[t])

        if (d_new - d_old) > MARKET_EPS:
            mw = float(d_new - d_old)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "sell", "mw": mw})
        if (d_old - d_new) > MARKET_EPS:
            mw = float(d_old - d_new)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "buy",  "mw": mw})

        if (c_new - c_old) > MARKET_EPS:
            mw = float(c_new - c_old)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "buy",  "mw": mw})
        if (c_old - c_new) > MARKET_EPS:
            mw = float(c_old - c_new)
            if mw >= max(MIN_LOT - 1e-12, 0.0):
                orders.append({"qh": qh_val, "side": "sell", "mw": mw})

    return {
        "status": "OK" if rev_delta > 0 else "DoNothing",
        "objective": rev_delta,
        "dis": dis_v, "ch": ch_v, "soc": soc_v, "Pg": Pg_v,
        "orders": orders
    }

# -------------------- Public API: day & range --------------------
def optimize_bess_day_sequential_orders(df_day: pd.DataFrame) -> Dict[str, Any]:
    cols = ["DA_ES_PRICE", "IDA1_ES_PRICE", "IDA2_ES_PRICE", "IDA3_ES_PRICE", "IDC_MedioES"]
    df = df_day.sort_values(["date", "qh"]).reset_index(drop=True).copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    T = len(df)
    zeros = np.zeros(T)
    out: Dict[str, Any] = {}

    res_DA = _stage_optimize_discrete(df, "DA_ES_PRICE", zeros, zeros)
    out["DA_status"]   = res_DA["status"]
    out["DA_earnings"] = _nn(res_DA["objective"])
    out["DA_dis"]      = res_DA["dis"]; out["DA_ch"] = res_DA["ch"]; out["DA_Pg"] = res_DA["Pg"]
    out["DA_orders"]   = [{"market": "DA", **o} for o in res_DA["orders"]]

    res_IDA1 = _stage_optimize_discrete(df, "IDA1_ES_PRICE", out["DA_dis"], out["DA_ch"])
    out["IDA1_status"]    = res_IDA1["status"]
    out["IDA1_increment"] = _nn(res_IDA1["objective"])
    out["IDA1_dis"]       = res_IDA1["dis"]; out["IDA1_ch"] = res_IDA1["ch"]; out["IDA1_Pg"] = res_IDA1["Pg"]
    out["IDA1_orders"]    = [{"market": "IDA1", **o} for o in res_IDA1["orders"]]

    res_IDA2 = _stage_optimize_discrete(df, "IDA2_ES_PRICE", out["IDA1_dis"], out["IDA1_ch"])
    out["IDA2_status"]    = res_IDA2["status"]
    out["IDA2_increment"] = _nn(res_IDA2["objective"])
    out["IDA2_dis"]       = res_IDA2["dis"]; out["IDA2_ch"] = res_IDA2["ch"]; out["IDA2_Pg"] = res_IDA2["Pg"]
    out["IDA2_orders"]    = [{"market": "IDA2", **o} for o in res_IDA2["orders"]]

    res_IDA3 = _stage_optimize_discrete(df, "IDA3_ES_PRICE", out["IDA2_dis"], out["IDA2_ch"])
    out["IDA3_status"]    = res_IDA3["status"]
    out["IDA3_increment"] = _nn(res_IDA3["objective"])
    out["IDA3_dis"]       = res_IDA3["dis"]; out["IDA3_ch"] = res_IDA3["ch"]; out["IDA3_Pg"] = res_IDA3["Pg"]
    out["IDA3_orders"]    = [{"market": "IDA3", **o} for o in res_IDA3["orders"]]

    res_IDC = _stage_optimize_discrete(df, "IDC_MedioES", out["IDA3_dis"], out["IDA3_ch"])
    out["IDC_status"]     = res_IDC["status"]
    out["IDC_increment"]  = _nn(res_IDC["objective"])
    out["Final_dis"]      = res_IDC["dis"]; out["Final_ch"] = res_IDC["ch"]; out["Final_Pg"] = res_IDC["Pg"]
    out["IDC_orders"]     = [{"market": "IDC", **o} for o in res_IDC["orders"]]

    out["Total_earnings"] = (
        out["DA_earnings"] + out["IDA1_increment"] +
        out["IDA2_increment"] + out["IDA3_increment"] + out["IDC_increment"]
    )
    out["orders_all"] = (
        out["DA_orders"] + out["IDA1_orders"] +
        out["IDA2_orders"] + out["IDA3_orders"] + out["IDC_orders"]
    )
    return out

def optimize_bess_day_summary(df_day: pd.DataFrame) -> pd.DataFrame:
    res = optimize_bess_day_sequential_orders(df_day)
    day = df_day["date"].iloc[0]
    if isinstance(day, pd.Timestamp):
        day = day.date()
    return pd.DataFrame([{
        "date": day,
        "DA_Earnings":  _nn(res.get("DA_earnings", 0.0)),
        "IDA1_Delta":   _nn(res.get("IDA1_increment", 0.0)),
        "IDA2_Delta":   _nn(res.get("IDA2_increment", 0.0)),
        "IDA3_Delta":   _nn(res.get("IDA3_increment", 0.0)),
        "IDC_Delta":    _nn(res.get("IDC_increment", 0.0)),
    }])

def optimize_bess_range(df: pd.DataFrame, progress: bool = True) -> pd.DataFrame:
    outs = []
    all_days = pd.to_datetime(df["date"]).dt.date.unique()
    all_days = sorted(all_days)
    n = len(all_days)
    for i, d in enumerate(all_days, start=1):
        dd = df[df["date"].astype(str).str[:10].isin([str(d)])].sort_values("qh").reset_index(drop=True)
        if progress:
            print(f"[{i}/{n}] {d} … running", flush=True)
        day_df = optimize_bess_day_summary(dd)
        outs.append(day_df)
        if progress:
            tot = float(day_df.iloc[0][["DA_Earnings","IDA1_Delta","IDA2_Delta","IDA3_Delta","IDC_Delta"]].sum())
            print(f"[{i}/{n}] {d} ✓ total={tot:.2f}", flush=True)
    if not outs:
        return pd.DataFrame(columns=["date","DA_Earnings","IDA1_Delta","IDA2_Delta","IDA3_Delta","IDC_Delta"])
    return pd.concat(outs, ignore_index=True)

# ==================== NEW PLOTTER ====================
#@title New Plotter
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_bess_markets_dashboard(df_day, result,
                                rte=0.85,
                                scale_discharge=True,
                                bar_width=0.9,
                                step_where="post",
                                eps=1e-6,
                                save_png=True,
                                save_dir="./",
                                filename=None,
                                dpi=220):

    # ----- unified tolerance
    try:
        mw_eps = float(MARKET_EPS)
    except NameError:
        mw_eps = float(eps)

    # ----- data prep
    df = df_day.sort_values("qh").reset_index(drop=True)
    qh = df["qh"].to_numpy()
    T  = len(qh)

    prices = {
        "DA":   df["DA_ES_PRICE"].to_numpy(),
        "IDA1": df["IDA1_ES_PRICE"].to_numpy(),
        "IDA2": df["IDA2_ES_PRICE"].to_numpy(),
        "IDA3": df["IDA3_ES_PRICE"].to_numpy(),
        "IDC":  df["IDC_MedioES"].to_numpy(),
    }

    # Final schedules
    DA_dis, DA_ch     = np.asarray(result["DA_dis"],   float), np.asarray(result["DA_ch"],   float)
    IDA1_dis, IDA1_ch = np.asarray(result["IDA1_dis"], float), np.asarray(result["IDA1_ch"], float)
    IDA2_dis, IDA2_ch = np.asarray(result["IDA2_dis"], float), np.asarray(result["IDA2_ch"], float)
    IDA3_dis, IDA3_ch = np.asarray(result["IDA3_dis"], float), np.asarray(result["IDA3_ch"], float)
    IDC_dis, IDC_ch   = np.asarray(result["Final_dis"],float), np.asarray(result["Final_ch"],float)
    Pg_final          = np.asarray(result["Final_Pg"], float)

    stages = ["DA","IDA1","IDA2","IDA3","IDC"]

    # nets per stage
    nets = {
        "DA":   DA_dis - DA_ch,
        "IDA1": IDA1_dis - IDA1_ch,
        "IDA2": IDA2_dis - IDA2_ch,
        "IDA3": IDA3_dis - IDA3_ch,
        "IDC":  IDC_dis - IDC_ch,
    }

    def display_net(v):
        x = np.asarray(v, float).copy()
        x[np.abs(x) <= mw_eps] = 0.0
        if scale_discharge:
            x = np.where(x > 0, x * rte, x)
        return x

    qmin, qmax = float(np.min(qh)), float(np.max(qh))
    candidates = [qmin, 20, 40, 60, 80, qmax]
    xticks = [val for val in candidates if qmin - 1e-9 <= val <= qmax + 1e-9]

    # ----- figure layout -----
    fig = plt.figure(figsize=(14, 18))

    height_prices     = 1.9
    height_spacer_sm  = 0.12
    height_title_row  = 0.34
    height_spacer_tiny = 0.20   # smaller gap (new tweak)
    height_spacer_lg  = 0.40
    height_trade      = 1.3

    heights = [
        height_prices,      # 0 prices
        height_spacer_sm,   # 1
        height_title_row,   # 2 trades done text
        height_spacer_tiny, # 3 smaller than before → decreased spacing before first chart
        height_trade,       # 4 final net
        height_spacer_lg,   # 5
        height_trade,       # 6 DA
        height_spacer_lg,   # 7
        height_trade,       # 8 IDA1
        height_spacer_lg,   # 9
        height_trade,       # 10 IDA2
        height_spacer_lg,   # 11
        height_trade,       # 12 IDA3
        height_spacer_lg,   # 13
        height_trade,       # 14 IDC
    ]

    gs = fig.add_gridspec(nrows=len(heights), ncols=1, height_ratios=heights, hspace=0.08)

    # ========= (0) PRICES =========
    axP = fig.add_subplot(gs[0, 0])
    axP.step(qh, prices["DA"],   where=step_where, label="DA",   linestyle="--", color="black")
    axP.step(qh, prices["IDA1"], where=step_where, label="IDA1", color="#263cc8")
    axP.step(qh, prices["IDA2"], where=step_where, label="IDA2", color="#7c9599")
    axP.step(qh, prices["IDA3"], where=step_where, label="IDA3", color="#6d32ff")
    axP.step(qh, prices["IDC"],  where=step_where, label="IDC",  color="#28ff52")
    axP.set_ylabel("Price [€/MWh]")
    axP.grid(True, alpha=0.3)
    axP.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    axP.set_xlim(qmin, qmax)

    title_date = df['date'].iloc[0]
    if hasattr(title_date, "strftime"):
        title_date = title_date.strftime("%Y-%m-%d")
    axP.set_title(f"OMIE Prices - {title_date}", loc="left")
    axP.set_xticks(xticks)

    # ========= (1) SMALL SPACER =========
    axSpacerSm = fig.add_subplot(gs[1, 0]); axSpacerSm.axis("off")

    # ========= (2) TITLE ROW "Trades done on (date)" =========
    axTitle = fig.add_subplot(gs[2, 0])
    axTitle.axis("off")
    axTitle.text(0.0, 0.5, f"Trades done on {title_date}", transform=axTitle.transAxes,
                 ha="left", va="center", fontsize=13, fontweight="bold")

    # ========= (3) SMALLER SPACER (less space before first chart) =========
    axSpacerTiny = fig.add_subplot(gs[3, 0]); axSpacerTiny.axis("off")

    # ========= (4) FINAL NET + SOC =========
    axN = fig.add_subplot(gs[4, 0])
    Pg_disp = display_net(Pg_final)
    axN.axhline(0, color="k", lw=0.8)
    COL_POS = "#00C853"; COL_NEG = "#E53935"
    colors_final = np.where(Pg_disp >= 0, COL_POS, COL_NEG)
    axN.bar(qh, Pg_disp, width=bar_width, color=colors_final, alpha=0.95, zorder=2)
    axN.set_ylabel("Final Net [MW]")
    axN.grid(True, alpha=0.3)
    axN.set_xlim(qmin, qmax); axN.set_xticks(xticks)
    TR_YMIN, TR_YMAX = -1.2, 1.2
    TR_YTICKS = [-1, -0.5, 0, 0.5, 1]
    axN.set_ylim(TR_YMIN, TR_YMAX); axN.set_yticks(TR_YTICKS)

    # SoC alignment
    half_w = bar_width / 2.0
    edges = np.concatenate(([qh[0] - half_w], qh + half_w))
    SQRT_RTE = float(np.sqrt(rte))
    soc = np.zeros(T + 1, dtype=float)
    for t in range(T):
        soc[t+1] = soc[t] + (SQRT_RTE * IDC_ch[t] * DT) - (IDC_dis[t] * DT) / SQRT_RTE
    try:
        Emax = float(E_MAX)
        if not np.isfinite(Emax) or Emax <= 0:
            raise ValueError
    except Exception:
        Emax = max(1.0, np.nanmax(np.abs(soc)) * 2.0)
    soc_pct_edges = 100.0 * np.clip(soc / Emax, 0.0, 1.0)
    axN2 = axN.twinx()
    axN2.plot(edges, soc_pct_edges, linewidth=2.0, linestyle="-", color="black")
    axN2.set_ylabel("SOC [%]")
    left_ymin, left_ymax = axN.get_ylim()
    axN2.set_ylim(50.0 * (left_ymin + 1.0), 50.0 * (left_ymax + 1.0))
    axN2.set_yticks([0, 20, 40, 60, 80, 100])

    total_rev = (
        result.get("DA_earnings", 0.0)
        + result.get("IDA1_increment", 0.0)
        + result.get("IDA2_increment", 0.0)
        + result.get("IDA3_increment", 0.0)
        + result.get("IDC_increment", 0.0)
    )
    axN.set_title(f"Total Revenues: € {total_rev:,.2f}", loc="left")

    # ========= Lower trades (same as before) =========
    COL_PREV_POS = "#A5D6A7"
    COL_PREV_NEG = "#EF9A9A"
    revenues = {
        "DA":   float(result.get("DA_earnings", 0.0)),
        "IDA1": float(result.get("IDA1_increment", 0.0)),
        "IDA2": float(result.get("IDA2_increment", 0.0)),
        "IDA3": float(result.get("IDA3_increment", 0.0)),
        "IDC":  float(result.get("IDC_increment", 0.0)),
    }

    def plot_stage_row(ax, net_prev, net_curr, label):
        prev, curr = display_net(net_prev), display_net(net_curr)
        ax.axhline(0, color="k", lw=0.8)
        colors_prev = np.where(prev >= 0, COL_PREV_POS, COL_PREV_NEG)
        ax.bar(qh, prev, width=bar_width, color=colors_prev, alpha=0.55, zorder=1)
        delta = curr - prev
        colors_delta = np.where(delta >= 0, COL_POS, COL_NEG)
        to_draw = np.abs(delta) > mw_eps
        if np.any(to_draw):
            ax.bar(qh[to_draw], delta[to_draw], width=bar_width,
                   color=colors_delta[to_draw], alpha=0.95, zorder=2)
        ax.set_ylim(TR_YMIN, TR_YMAX); ax.set_yticks(TR_YTICKS)
        ax.set_xlim(qmin, qmax); ax.set_xticks(xticks)
        ax.set_ylabel(f"{label} [MW]")
        inc_rev = revenues.get(label, 0.0)
        ax.set_title(f"Incremental Revenues: € {inc_rev:,.2f}", loc="left")
        ax.grid(True, alpha=0.3)

    # Define stages sequence
    seq = [
        (np.zeros_like(nets["DA"]), nets["DA"], "DA"),
        (nets["DA"], nets["IDA1"], "IDA1"),
        (nets["IDA1"], nets["IDA2"], "IDA2"),
        (nets["IDA2"], nets["IDA3"], "IDA3"),
        (nets["IDA3"], nets["IDC"], "IDC"),
    ]

    row_indices = [6, 8, 10, 12, 14]
    for (row, (prev_net, curr_net, label)) in zip(row_indices, seq):
        ax_stage = fig.add_subplot(gs[row, 0])
        plot_stage_row(ax_stage, prev_net, curr_net, label)

    for row in [5, 7, 9, 11, 13]:
        ax_sp = fig.add_subplot(gs[row, 0]); ax_sp.axis("off")

    fig.axes[-2].set_xlabel("Quarter-hour (qh)")

    first_stage_ax = fig.axes[-4]
    legend_handles = [
        Line2D([0],[0], marker='s', linestyle='None', color=COL_PREV_POS, label='Prev. Sell Orders'),
        Line2D([0],[0], marker='s', linestyle='None', color=COL_PREV_NEG, label='Prev. Buy Orders'),
        Line2D([0],[0], marker='s', linestyle='None', color=COL_POS, label='New Sell Orders'),
        Line2D([0],[0], marker='s', linestyle='None', color=COL_NEG, label='New Buy Orders'),
    ]
    first_stage_ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    # Save
    if save_png:
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            title_date_str = df['date'].iloc[0]
            if hasattr(title_date_str, "strftime"):
                title_date_str = title_date_str.strftime("%Y-%m-%d")
            filename = f"OMIE_BESS_Dashboard_{title_date_str}.png"
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to: {out_path}")

    plt.show()

# ==================== JSON HELPERS (deduplicated) ====================
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
                rec[c] = round(float(val), 2)
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

def save_daily_svg_prices(df_day: pd.DataFrame, out_path: str, dpi: int = 220):
    # Ensure folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # X axis
    qh = df_day["qh"].to_numpy() if "qh" in df_day.columns else np.arange(len(df_day)) + 1

    # Map available columns to readable labels
    series = [
        ("DA_ES_PRICE",   "DA"),
        ("IDA1_ES_PRICE", "IDA1"),
        ("IDA2_ES_PRICE", "IDA2"),
        ("IDA3_ES_PRICE", "IDA3"),
        ("IDC_MedioES",   "IDC (MedioES)")
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    has_any = False
    for col, label in series:
        if col in df_day.columns and df_day[col].notna().any():
            ax.step(qh, df_day[col].to_numpy(dtype=float), where="post", linewidth=1.4, label=label)
            has_any = True

    if not has_any:
        fig.text(0.5, 0.5, "No price data available", ha="center", va="center", fontsize=16)

    ax.set_xlabel("Quarter-hour (qh)")
    ax.set_ylabel("€/MWh")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()

    # Force render before save; prevents blank SVGs in headless runners
    fig.canvas.draw()
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ==================== MAIN ====================
if __name__ == "__main__":
    # Path to JSON & PNG in the repo root (same folder as this file)
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    JSON_PATH = os.path.join(BASE_DIR, "BESS OMIE Results.json")
    PLOT_PATH = os.path.join(BASE_DIR, "OMIE_BESS.png")

    # Use yesterday’s date
    day = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
    print("Processing day:", day)

    # Download all prices for the day
    df_day = download_day_all_markets_es_wide(day, ida_sessions=(1,2,3))
    if df_day.empty:
        print("No data for", day)
    else:
        # 1) Per-QH result for the plotter (arrays + orders by stage)
        full_result = optimize_bess_day_sequential_orders(df_day)

        # 2) One-row daily summary for JSON (keeps your existing pipeline)
        daily_summary = optimize_bess_day_summary(df_day)
        append_or_update_json(JSON_PATH, summary_df_to_records(daily_summary))
        print("Updated JSON:", JSON_PATH)
        print(json_to_df(JSON_PATH).tail())

        # 3) Plot the new dashboard to PNG in repo root
        try:
            plot_bess_markets_dashboard(
                df_day,
                result=full_result,
                save_png=True,
                save_dir=BASE_DIR,
                filename="OMIE_BESS.png",
                dpi=220
            )
            print(f"Saved PNG to: {PLOT_PATH}")
            try:
                print("PNG size:", os.path.getsize(PLOT_PATH), "bytes")
            except Exception:
                pass
        except Exception as e:
            print(f"Could not save PNG: {e}")
