import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

TRADING_SECONDS_PER_DAY_CN = 4 * 60 * 60  # 9:30-11:30, 13:00-15:00 共 14400 秒

# ---------- 時間/對齊 ----------
def _to_dt(s):
    s = pd.to_datetime(s)
    try:
        return s.tz_convert(None)
    except Exception:
        return s.tz_localize(None)

def china_continuous_session_mask(index: pd.DatetimeIndex) -> pd.Series:
    # 過濾 A 股連續競價時段
    t = index.tz_localize(None)
    hms = t.strftime("%H:%M:%S")
    am = (hms >= "09:30:00") & (hms <= "11:30:00")
    pm = (hms >= "13:00:00") & (hms <= "15:00:00")
    return pd.Series(am | pm, index=index)

# ---------- 逐筆成交打標籤（Lee-Ready 簡化版） ----------
def classify_trade_sides(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """
    trades: time, px, vol
    quotes: time, mid
    返回 trades 增加 side(+1/-1)、signed_vol
    """
    t = trades.copy()
    q = quotes[["mid"]].copy()
    t["time"] = _to_dt(t["time"])
    q["time"] = _to_dt(q.index if q.index.name is not None else q["time"])
    t = t.sort_values("time")
    q = q.sort_values("time")
    merged = pd.merge_asof(t, q.reset_index().rename(columns={"index":"time"}), on="time", direction="backward")
    mid = merged["mid"].astype(float)
    side = np.sign(merged["px"].astype(float) - mid)
    # 平價用上一筆 side（或 0）
    same = (side == 0)
    side = side.replace(0, np.nan).fillna(method="ffill").fillna(0.0)
    merged["side"] = side
    merged["signed_vol"] = merged["vol"].astype(float) * merged["side"].astype(float)
    return merged[["time","px","vol","side","signed_vol"]]

# ---------- 1秒網格 ----------
def make_1s_grid(quotes_df: pd.DataFrame,
                 trades_df: pd.DataFrame,
                 tick_size: float = 0.01) -> pd.DataFrame:
    # 报价
    q = quotes_df.copy()
    q["time"] = _to_dt(q["time"])
    q = q.sort_values("time").set_index("time")
    q = q[["bid","ask","bid_size","ask_size"]].astype(float)
    # 1s 重取樣（last），再向前填充
    q1 = q.resample("1S").last().ffill()
    q1 = q1[china_continuous_session_mask(q1.index)]
    q1["mid"] = (q1["bid"] + q1["ask"]) / 2.0
    q1["spread"] = (q1["ask"] - q1["bid"]).clip(lower=0)
    q1["rel_spread"] = q1["spread"] / (q1["mid"] + 1e-12)
    q1["tick_size"] = float(tick_size)
    q1["rel_tick"] = q1["tick_size"] / (q1["mid"] + 1e-12)

    # microprice 與 QI
    denom = (q1["bid_size"] + q1["ask_size"]).replace(0, np.nan)
    micro = (q1["ask"] * q1["bid_size"] + q1["bid"] * q1["ask_size"]) / denom
    q1["microprice"] = micro.fillna(q1["mid"])
    q1["qi"] = (q1["bid_size"] - q1["ask_size"]) / (q1["bid_size"] + q1["ask_size"] + 1e-12)
    q1["micro_prem"] = (q1["microprice"] - q1["mid"]) / (q1["mid"] + 1e-12)  # 相對中位價
    # 以「檔」為尺度（低價股友好）
    q1["micro_prem_ticks"] = (q1["microprice"] - q1["mid"]) / (q1["tick_size"] + 1e-12)

    # 逐筆成交聚合到 1s
    if trades_df is not None and len(trades_df) > 0:
        t = trades_df.copy()
        t["time"] = _to_dt(t["time"])
        t = t.sort_values("time")
        # 打標籤
        mid_series = q1[["mid"]].copy()
        mid_series["time"] = mid_series.index
        lab = classify_trade_sides(t, mid_series)
        lab = lab.set_index("time")
        agg = pd.DataFrame({
            "vwap": (lab["px"] * lab["vol"]).resample("1S").sum() / (lab["vol"].resample("1S").sum() + 1e-12),
            "trd_vol": lab["vol"].resample("1S").sum(),
            "trd_cnt": lab["vol"].resample("1S").count(),
            "buy_vol": lab.loc[lab["side"] > 0, "vol"].resample("1S").sum(),
            "sell_vol": lab.loc[lab["side"] < 0, "vol"].resample("1S").sum(),
            "signed_vol": lab["signed_vol"].resample("1S").sum(),
        })
        agg = agg.fillna(0.0)
        agg = agg[china_continuous_session_mask(agg.index)]
        df = q1.join(agg, how="left").fillna({"trd_vol":0.0,"trd_cnt":0.0,"buy_vol":0.0,"sell_vol":0.0,"signed_vol":0.0})
    else:
        df = q1.copy()
        for c in ["vwap","trd_vol","trd_cnt","buy_vol","sell_vol","signed_vol"]:
            df[c] = 0.0

    # 成交不平衡與強度
    df["ti"] = (df["buy_vol"] - df["sell_vol"]) / (df["trd_vol"] + 1e-12)
    df["trades_per_s"] = df["trd_cnt"]
    df["vol_per_s"] = df["trd_vol"]

    return df

# ---------- 特徵（以 1s 網格為基礎） ----------
def add_features(df: pd.DataFrame, windows: List[int] = [30, 60, 120]) -> pd.DataFrame:
    out = df.copy()
    out["ret_1s"] = out["mid"].pct_change()

    for w in windows:
        # 價格動量（w秒）
        out[f"ret_{w}s"] = out["mid"].pct_change(w)
        # 波動（1秒報酬的滾動標準差）
        out[f"rv_{w}s"] = out["ret_1s"].rolling(w).std().fillna(0.0)
        # 標準化動量（近似 t-統計）
        denom = (out[f"rv_{w}s"] * np.sqrt(max(w,1)) + 1e-12)
        out[f"z_slope_{w}s"] = out[f"ret_{w}s"] / denom
        # 成交/量強度（標準化）
        out[f"lambda_{w}s"] = (out["trades_per_s"].rolling(w).mean() /
                               (out["trades_per_s"].rolling(w).std() + 1e-12))
        out[f"vint_{w}s"] = (out["vol_per_s"].rolling(w).mean() /
                             (out["vol_per_s"].rolling(w).std() + 1e-12))
        # 平盤/噪音 regime
        out[f"flat_{w}s"] = ((out[f"rv_{w}s"] * 1e4) < (2.0 * out["rel_spread"] * 1e4)).astype(int)

        # 平滑後的不平衡
        out[f"qi_ma_{w}s"] = out["qi"].rolling(w).mean()
        out[f"ti_ma_{w}s"] = out["ti"].rolling(w).mean()
        out[f"micro_prem_ticks_ma_{w}s"] = out["micro_prem_ticks"].rolling(w).mean()

        # 薄量旗標（相對自身歷史分位）
        out[f"ask_thin_{w}s"] = (out["ask_size"] <= out["ask_size"].rolling(600).quantile(0.2)).astype(int)
        out[f"bid_thin_{w}s"] = (out["bid_size"] <= out["bid_size"].rolling(600).quantile(0.2)).astype(int)

    # 便於觀察的 bp 度量
    out["rel_spread_bp"] = out["rel_spread"] * 1e4
    out["rel_tick_bp"] = out["rel_tick"] * 1e4
    return out

# ---------- 前瞻收益（秒） ----------
def make_forward_returns_seconds(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        out[f"fret_{h}s"] = out["mid"].pct_change(h).shift(-h)
    return out

# ---------- 相關/Sharpe/勝率 ----------
def spearman_ts_corr(feature: pd.Series, target: pd.Series) -> float:
    d = pd.concat([feature, target], axis=1).dropna()
    if len(d) < 20:
        return np.nan
    return d.iloc[:,0].rank().corr(d.iloc[:,1].rank())

def evaluate_feature_time_series(df: pd.DataFrame,
                                 feature: str,
                                 horizons: List[int],
                                 q: float = 0.3,
                                 cost_mode: str = "none"  # "none" 或 "spread"
                                 ) -> pd.DataFrame:
    """
    單股票時間序列評估：
    - 相關（Spearman）：feature vs fret_h
    - 方向化策略：長 top q、短 bottom q，持有 h 秒
    - Sharpe：按 h 秒持有折算到日頻；勝率：PnL>0 比例
    - 成本：cost_mode="spread" 時，每筆扣 1×rel_spread（進出各半）
    """
    res = []
    z = (df[feature] - df[feature].rolling(600).mean()) / (df[feature].rolling(600).std() + 1e-12)
    lo = z.quantile(q)
    hi = z.quantile(1 - q)
    pos = pd.Series(0.0, index=df.index)
    pos[z <= lo] = -1.0
    pos[z >= hi] = +1.0

    for h in horizons:
        fret = df[f"fret_{h}s"]
        # 相關
        ic = spearman_ts_corr(df[feature], fret)

        # 收益序列（重疊，近似）
        pnl = pos * fret
        if cost_mode == "spread":
            # 進出合計 1×rel_spread，近似攤到當刻
            pnl = pnl - (pos.abs() * df["rel_spread"]).fillna(0.0)

        pnl = pnl.dropna()
        if len(pnl) < 50:
            res.append({"feature": feature, "h": h, "ic": ic, "n": len(pnl),
                        "mean": np.nan, "std": np.nan, "ann_sharpe": np.nan, "winrate": np.nan})
            continue

        mean = pnl.mean()
        std = pnl.std(ddof=0)
        # 年化：每筆是 h 秒持有，日內約 TRADING_SECONDS_PER_DAY_CN/h 次非重疊機會
        ann_sharpe = np.sqrt(252 * max(TRADING_SECONDS_PER_DAY_CN / max(h,1), 1)) * (mean / (std + 1e-12))
        winrate = (pnl > 0).mean()

        res.append({
            "feature": feature, "h": h, "ic": ic, "n": int(len(pnl)),
            "mean": float(mean), "std": float(std),
            "ann_sharpe": float(ann_sharpe), "winrate": float(winrate)
        })
    return pd.DataFrame(res)

# ---------- 一站式：從原始逐筆到評估 ----------
def run_l1_pipeline(quotes_df: pd.DataFrame,
                    trades_df: pd.DataFrame,
                    tick_size: float = 0.01,
                    feat_windows: List[int] = [30,60,120],
                    horizons: List[int] = [10,30,60,180],
                    cost_mode: str = "none") -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid = make_1s_grid(quotes_df, trades_df, tick_size=tick_size)
    feats = add_features(grid, windows=feat_windows)
    feats = make_forward_returns_seconds(feats, horizons=horizons)

    # 選幾個常用單因子做評估（你可自行擴展）
    feature_list = []
    for w in feat_windows:
        feature_list += [f"z_slope_{w}s", f"qi_ma_{w}s", f"ti_ma_{w}s", f"micro_prem_ticks_ma_{w}s"]

    reports = []
    for f in feature_list:
        if f in feats.columns:
            rep = evaluate_feature_time_series(feats, f, horizons=horizons, cost_mode=cost_mode)
            reports.append(rep)
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    return feats, report