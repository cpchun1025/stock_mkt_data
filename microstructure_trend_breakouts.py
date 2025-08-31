import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import BDay

EPS = 1e-9

class MicrostructureTrend:
    def __init__(self,
                 lookback_seconds=60,
                 slope_window=10,
                 breakout_alpha=0.001,          # 突破幅度 0.1%
                 stay_confirm_M=3,              # 連續停留確認bar數
                 max_rel_spread=0.0012,         # 相對點差上限 12bp
                 slip_k=2000,                   # 可達滑點代理係數（按你size量級調）
                 cooldown_seconds=30,           # 連續訊號冷卻期
                 quant_win=300):                # 分位數窗口（用於薄量判定）
        self.lookback_seconds = lookback_seconds
        self.slope_window = slope_window
        self.breakout_alpha = breakout_alpha
        self.stay_confirm_M = stay_confirm_M
        self.max_rel_spread = max_rel_spread
        self.slip_k = slip_k
        self.cooldown_seconds = cooldown_seconds
        self.quant_win = quant_win

    # ---------- 工具 ----------
    @staticmethod
    def _rolling_slope(series, window):
        # 對最近 window 筆做最小二乘斜率
        x = series.values
        n = len(x)
        if n < 2:
            return 0.0
        t = np.arange(n)
        t_mean = t.mean()
        x_mean = x.mean()
        denom = np.sum((t - t_mean) ** 2) + EPS
        slope = np.sum((t - t_mean) * (x - x_mean)) / denom
        return slope

    @staticmethod
    def _zscore(s, win=300):
        m = s.rolling(win, min_periods=max(5, win // 3)).mean()
        v = s.rolling(win, min_periods=max(5, win // 3)).std()
        return ((s - m) / (v + EPS)).fillna(0)

    @staticmethod
    def _confirm_series(flag_bool_arr, M):
        c = np.zeros(len(flag_bool_arr), dtype=bool)
        run = 0
        for i, f in enumerate(flag_bool_arr):
            run = run + 1 if f else 0
            c[i] = (run >= M)
        return c

    # ---------- 主計算：微結構趨勢與盒子 ----------
    def calculate_micro_trends(self, df):
        """
        期望輸入欄位:
          time (datetime64), bidprice, bidsize, askprice, asksize
        回傳含以下欄位:
          mid, spread, rel_spread, depth_imb, micro_high, micro_low,
          range_width, position_in_range, slope_short, z_slope, z_spread,
          trend_strength, micro_high_prev, micro_low_prev, breakout_up_raw/down_raw,
          breakout_up/down
        """
        df = df.copy().sort_values("time").reset_index(drop=True)

        # 基礎欄位
        df["mid"] = (df["askprice"] + df["bidprice"]) / 2.0
        df["spread"] = (df["askprice"] - df["bidprice"]).clip(lower=0)
        df["rel_spread"] = (df["spread"] / df["mid"]).fillna(0)
        df["depth_imb"] = (df["bidsize"] - df["asksize"]) / (df["bidsize"] + df["asksize"] + EPS)

        # 時間秒
        df["time_seconds"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

        # micro_high / micro_low（時間窗口）
        micro_high = np.full(len(df), np.nan)
        micro_low = np.full(len(df), np.nan)
        j0 = 0
        ts = df["time_seconds"].values
        for i in range(len(df)):
            t_now = ts[i]
            while ts[j0] < t_now - self.lookback_seconds:
                j0 += 1
            window_mid = df["mid"].iloc[j0:i+1]
            if len(window_mid) > 0:
                micro_high[i] = window_mid.max()
                micro_low[i] = window_mid.min()
        df["micro_high"] = micro_high
        df["micro_low"] = micro_low

        # 區間寬度與位置
        df["range_width"] = (df["micro_high"] - df["micro_low"]).fillna(0)
        df["position_in_range"] = ((df["mid"] - df["micro_low"]) / (df["range_width"] + EPS)).clip(0, 1)

        # 短窗斜率 + 標準化
        df["slope_short"] = df["mid"].rolling(self.slope_window, min_periods=2)\
            .apply(lambda x: MicrostructureTrend._rolling_slope(pd.Series(x), self.slope_window), raw=False)\
            .fillna(0)
        df["z_slope"] = MicrostructureTrend._zscore(df["slope_short"], win=300)
        df["z_spread"] = MicrostructureTrend._zscore(df["rel_spread"], win=300)

        # 趨勢強度：斜率 - 成本
        df["trend_strength"] = df["z_slope"] - 0.5 * df["z_spread"]

        # 上一個完整窗口的高低
        micro_high_prev = np.full(len(df), np.nan)
        micro_low_prev = np.full(len(df), np.nan)
        j0 = 0
        for i in range(len(df)):
            t_now = ts[i]
            while ts[j0] < t_now - self.lookback_seconds:
                j0 += 1
            # 上一窗口用 (j0-1) 到 (i-1)，避免與當前重疊
            L = max(0, j0 - 1)
            R = max(0, i - 1)
            prev_slice = df["mid"].iloc[L:R+1]
            if len(prev_slice) > 1:
                micro_high_prev[i] = prev_slice.max()
                micro_low_prev[i] = prev_slice.min()
        df["micro_high_prev"] = micro_high_prev
        df["micro_low_prev"] = micro_low_prev

        # 初步突破（加入幅度 alpha）
        alpha = self.breakout_alpha
        df["breakout_up_raw"] = (df["mid"] > (df["micro_high_prev"] * (1 + alpha)))
        df["breakout_down_raw"] = (df["mid"] < (df["micro_low_prev"] * (1 - alpha)))

        # 連續停留確認
        df["breakout_up"] = MicrostructureTrend._confirm_series(df["breakout_up_raw"].fillna(False).values,
                                                                self.stay_confirm_M)
        df["breakout_down"] = MicrostructureTrend._confirm_series(df["breakout_down_raw"].fillna(False).values,
                                                                  self.stay_confirm_M)

        # 對向變薄（用滾動低分位）
        Nq = self.quant_win
        ask_q = df["asksize"].rolling(Nq, min_periods=max(5, Nq//3)).quantile(0.2)\
                 .fillna(method="bfill").fillna(df["asksize"].median())
        bid_q = df["bidsize"].rolling(Nq, min_periods=max(5, Nq//3)).quantile(0.2)\
                 .fillna(method="bfill").fillna(df["bidsize"].median())
        df["ask_thin"] = (df["asksize"] <= ask_q)
        df["bid_thin"] = (df["bidsize"] <= bid_q)

        # 可達滑點代理
        df["slip_proxy_up"] = df["rel_spread"] + (self.slip_k / (df["asksize"] + 1.0))
        df["slip_proxy_down"] = df["rel_spread"] + (self.slip_k / (df["bidsize"] + 1.0))

        return df[[
            "time","time_seconds","mid","bidprice","askprice","bidsize","asksize",
            "spread","rel_spread","depth_imb",
            "micro_high","micro_low","range_width","position_in_range",
            "slope_short","z_slope","z_spread","trend_strength",
            "micro_high_prev","micro_low_prev",
            "breakout_up_raw","breakout_down_raw","breakout_up","breakout_down",
            "ask_thin","bid_thin","slip_proxy_up","slip_proxy_down"
        ]]

    # ---------- 統計摘要 ----------
    def breakout_analysis(self, df):
        results = self.calculate_micro_trends(df)

        breakout_up_events = results[results["breakout_up"] == True]
        breakout_down_events = results[results["breakout_down"] == True]

        analysis = {
            "total_breakouts_up": len(breakout_up_events),
            "total_breakouts_down": len(breakout_down_events),
            "breakout_rate": (len(breakout_up_events) + len(breakout_down_events)) / max(1, len(results)),
            "avg_range_width": results["range_width"].mean(),
            "avg_trend_strength": results["trend_strength"].mean(),
            "consolidation_periods": int((results["trend_strength"].abs() < 0.1).sum()),
            "trending_periods": int((results["trend_strength"].abs() > 0.3).sum())
        }
        return analysis, breakout_up_events, breakout_down_events

    # ---------- 交易訊號 ----------
    def generate_trend_signals(self, df):
        results = self.calculate_micro_trends(df).copy()

        # 趨勢方向（沿用你原本分檔）
        results["trend_signal"] = "Neutral"
        pir = results["position_in_range"]
        results.loc[pir >= 0.6, "trend_signal"] = "Strong_Up"
        results.loc[(pir < 0.6) & (pir >= 0.5), "trend_signal"] = "Weak_Up"
        results.loc[pir <= 0.4, "trend_signal"] = "Strong_Down"
        results.loc[(pir > 0.4) & (pir < 0.5), "trend_signal"] = "Weak_Down"

        # 動能/突破訊號（加入成本與薄量過濾、趨勢一致性）
        results["momentum_signal"] = "Neutral"
        up_ok = (results["breakout_up"] &
                 (results["trend_strength"] > 0) &
                 (results["rel_spread"] <= self.max_rel_spread) &
                 (results["ask_thin"]) &
                 (results["slip_proxy_up"] <= (self.max_rel_spread * 2.0)))
        down_ok = (results["breakout_down"] &
                   (results["trend_strength"] < 0) &
                   (results["rel_spread"] <= self.max_rel_spread) &
                   (results["bid_thin"]) &
                   (results["slip_proxy_down"] <= (self.max_rel_spread * 2.0)))

        results.loc[up_ok, "momentum_signal"] = "Bullish_Breakout"
        results.loc[down_ok, "momentum_signal"] = "Bearish_Breakout"

        # 冷卻期 – 只允許隔 cooldown_seconds 觸發一次 entry
        results["entry_signal"] = False
        last_entry_t = -1e18
        timesec = results["time_seconds"].values
        for i in range(len(results)):
            if results["momentum_signal"].iat[i] != "Neutral":
                if timesec[i] - last_entry_t >= self.cooldown_seconds:
                    results["entry_signal"].iat[i] = True
                    last_entry_t = timesec[i]

        return results[[
            "time","mid","trend_signal","momentum_signal","entry_signal",
            "breakout_up","breakout_down","position_in_range",
            "rel_spread","asksize","bidsize","ask_thin","bid_thin",
            "trend_strength","range_width","slip_proxy_up","slip_proxy_down"
        ]]

# ---------------- 使用範例（示意） ----------------
if __name__ == "__main__":
    # 你自己的取數邏輯：確保 df 包含 time, bidprice, bidsize, askprice, asksize
    # 這裡僅示意：請替換為你實際的 intraday_tick 取得方式
    # df = your_loader(...)
    # 範例假造（請換掉）
    times = pd.date_range("2025-08-29 09:30:00", periods=2000, freq="S", tz="Asia/Hong_Kong")
    rng = np.sin(np.linspace(0, 40, len(times))) * 0.01 + 11.0
    df = pd.DataFrame({
        "time": times,
        "bidprice": rng - 0.005,
        "askprice": rng + 0.005,
        "bidsize": np.random.randint(2000, 12000, size=len(times)),
        "asksize": np.random.randint(2000, 12000, size=len(times)),
    })

    analyzer = MicrostructureTrend(
        lookback_seconds=60,
        slope_window=10,
        breakout_alpha=0.001,
        stay_confirm_M=3,
        max_rel_spread=0.0012,
        slip_k=2000,
        cooldown_seconds=30,
        quant_win=300
    )

    results = analyzer.calculate_micro_trends(df)
    analysis, up_evts, dn_evts = analyzer.breakout_analysis(df)
    signals = analyzer.generate_trend_signals(df)

    print("Microstructure Trend Analysis:")
    print(results.tail(10)[[
        "time","mid","micro_high","micro_low","trend_strength",
        "position_in_range","range_width"
    ]])

    print("\nBreakout Analysis Summary:")
    for k, v in analysis.items():
        print(f"{k}: {v}")

    print("\nRecent Trading Signals:")
    recent = signals[signals["entry_signal"]].tail()
    if len(recent) > 0:
        print(recent[["time","mid","trend_signal","momentum_signal",
                      "rel_spread","asksize","bidsize","trend_strength"]])
    else:
        print("No recent entry signals detected")

    # 當前市場狀態
    latest = signals.iloc[-1]
    print("\nCurrent Market State:")
    print(f"Current trend: {latest['trend_signal']}")
    print(f"Current momentum: {latest['momentum_signal']}")
    print(f"Position in range: {results['position_in_range'].iloc[-1]:.2f}")

import numpy as np
import pandas as pd

def _merge_time_to_index(df):
    """確保 time 單調，並建立秒級索引以加速查找。"""
    df = df.sort_values("time").reset_index(drop=True).copy()
    base = df["time"].iloc[0]
    df["_tsec"] = (df["time"] - base).dt.total_seconds().astype(float)
    return df, base

def _find_index_at_time(tsecs, target):
    """在 tsecs 單調數組中找到 <= target 的最後索引。若全都大於 target，回傳 -1。"""
    # 用二分搜索
    import bisect
    i = bisect.bisect_right(tsecs, target) - 1
    return i

def backtest_signals(
    df_ticks,
    signals,
    horizons=[30, 60, 120],
    cost_model="baseline",     # "baseline" 或 "slip_proxy"
    slip_coeff=0.6,            # slip_proxy 乘數（越大越保守）
    side_map={"Bullish_Breakout": 1, "Bearish_Breakout": -1},
    require_same_side=True,    # 要求 trend_signal 與 momentum 同向才入場
    max_holding_seconds=None   # 若設置，超過此時間強制平倉
):
    """
    df_ticks: 原始 L1 資料（需含 time, mid, rel_spread, asksize, bidsize）
    signals: generate_trend_signals 的輸出（需含 time, entry_signal, momentum_signal, rel_spread, slip_proxy_up/down, trend_strength 等）
    horizons: 評估視窗（秒）
    cost_model:
        - baseline: 進出場各扣 0.5 * rel_spread（雙邊合計 1 * rel_spread）
        - slip_proxy: 進場扣 slip_coeff * slip_proxy_{up/down}，出場同理用當下對應 slip_proxy
    side_map: 定義多空方向
    require_same_side: 若 True，只保留 trend_signal 與 momentum 同向的入場
    max_holding_seconds: 若不為 None，超過該秒數即在該時刻平倉
    回傳：
        - perf_df: 每筆交易在各 horizon 的績效與路徑統計
        - summary: 匯總指標
        - by_bins: 按 trend_strength/rel_spread 分層的摘要
    """

    # 準備基礎序列
    ticks, base_time = _merge_time_to_index(df_ticks)
    sigs, _ = _merge_time_to_index(signals)

    # 僅保留 entry_signal
    sigs = sigs[sigs["entry_signal"]].copy()
    if require_same_side and "trend_signal" in sigs:
        # 同向：上破且 Strong_Up；下破且 Strong_Down（弱信號可自定）
        up_mask = (sigs["momentum_signal"] == "Bullish_Breakout") & (sigs["trend_signal"].str.contains("Up"))
        dn_mask = (sigs["momentum_signal"] == "Bearish_Breakout") & (sigs["trend_signal"].str.contains("Down"))
        sigs = sigs[up_mask | dn_mask].copy()

    if len(sigs) == 0:
        return pd.DataFrame(), {}, {}

    tsec_ticks = ticks["_tsec"].values
    perf_rows = []

    for i, row in sigs.iterrows():
        t0 = row["_tsec"]
        side = side_map.get(row["momentum_signal"], 0)
        if side == 0:
            continue

        # 進場價格（用 mid）
        idx0 = _find_index_at_time(tsec_ticks, t0)
        if idx0 < 0:
            continue
        px_in = ticks["mid"].iat[idx0]

        # 進場成本
        if cost_model == "baseline":
            cost_in = 0.5 * row.get("rel_spread", 0.0)
        else:  # slip_proxy
            if side > 0:
                cost_in = slip_coeff * row.get("slip_proxy_up", row.get("rel_spread", 0.0))
            else:
                cost_in = slip_coeff * row.get("slip_proxy_down", row.get("rel_spread", 0.0))

        # 可選：強制最長持倉
        horizons_use = list(horizons)
        if max_holding_seconds is not None and max_holding_seconds not in horizons_use:
            horizons_use = sorted(set(horizons_use + [max_holding_seconds]))

        # 路徑極值（MFE/MAE）以秒為粒度
        # 我們掃描到最長 horizon 的末端一次，沿途記錄高低
        max_h = max(horizons_use)
        t_end = t0 + max_h
        idx_end = _find_index_at_time(tsec_ticks, t_end)
        if idx_end <= idx0:
            # 沒有足夠未來資料
            continue

        future_mid = ticks["mid"].iloc[idx0:idx_end+1].values
        future_t = tsec_ticks[idx0:idx_end+1]

        # 逐秒路徑的持倉盈虧（未扣出場成本）
        ret_path = side * (future_mid - px_in) / px_in

        # MFE/MAE（期間內的最大/最小浮動盈虧）
        mfe = np.max(ret_path) if len(ret_path) else 0.0
        mae = np.min(ret_path) if len(ret_path) else 0.0

        # 各 horizon 的出場與成本
        metrics = {
            "time": row["time"],
            "side": side,
            "px_in": px_in,
            "strength": row.get("trend_strength", np.nan),
            "rel_spread_in": row.get("rel_spread", np.nan),
            "asksize_in": row.get("asksize", np.nan),
            "bidsize_in": row.get("bidsize", np.nan),
            "mfe": mfe,
            "mae": mae
        }

        for H in horizons:
            tH = t0 + H
            idxH = _find_index_at_time(tsec_ticks, tH)
            if idxH <= idx0:
                metrics[f"ret_{H}s_net"] = np.nan
                continue

            px_out = ticks["mid"].iat[idxH]

            # 出場成本：取出場時刻對應的 rel_spread/ slip_proxy
            # 找 signals 中最接近 tH 的行以取 slip_proxy（或直接在 ticks 估算 rel_spread）
            # 為簡便，用 ticks 的 rel_spread 近似（若你把 slip_proxy 欄位合併到 ticks，可用 slip_proxy）
            rel_spread_out = ticks["rel_spread"].iat[idxH] if "rel_spread" in ticks.columns else row.get("rel_spread", 0.0)

            if cost_model == "baseline":
                cost_out = 0.5 * rel_spread_out
            else:
                # 沒有 slip_proxy 欄位在 ticks 時，用 rel_spread 近似
                if side > 0:
                    slip_out = ticks["rel_spread"].iat[idxH] if "rel_spread" in ticks.columns else rel_spread_out
                else:
                    slip_out = ticks["rel_spread"].iat[idxH] if "rel_spread" in ticks.columns else rel_spread_out
                cost_out = slip_coeff * slip_out

            gross = side * (px_out - px_in) / px_in
            net = gross - cost_in - cost_out
            metrics[f"ret_{H}s_net"] = net

        # 若有最長持倉限制，報告其淨損益
        if max_holding_seconds is not None:
            idxM = _find_index_at_time(tsec_ticks, t0 + max_holding_seconds)
            if idxM > idx0:
                px_outM = ticks["mid"].iat[idxM]
                rel_spread_outM = ticks["rel_spread"].iat[idxM]
                if cost_model == "baseline":
                    cost_outM = 0.5 * rel_spread_outM
                else:
                    cost_outM = slip_coeff * rel_spread_outM
                grossM = side * (px_outM - px_in) / px_in
                netM = grossM - cost_in - cost_outM
                metrics[f"ret_{max_holding_seconds}s_net"] = netM

        perf_rows.append(metrics)

    if len(perf_rows) == 0:
        return pd.DataFrame(), {}, {}

    perf_df = pd.DataFrame(perf_rows).sort_values("time").reset_index(drop=True)

    # 匯總
    def _summary_of(col):
        x = perf_df[col].dropna()
        if len(x) == 0:
            return {"mean": np.nan, "median": np.nan, "winrate": np.nan, "sharpe": np.nan, "n": 0}
        winrate = (x > 0).mean()
        sharpe = x.mean() / (x.std() + 1e-12)
        return {"mean": x.mean(), "median": x.median(), "winrate": winrate, "sharpe": sharpe, "n": len(x)}

    summary = {"trades": len(perf_df), "mfe_mean": perf_df["mfe"].mean(), "mae_mean": perf_df["mae"].mean()}
    for H in horizons:
        summary[f"ret_{H}s"] = _summary_of(f"ret_{H}s_net")

    # 分層（示例：按趨勢強度與相對點差）
    q_strength = perf_df["strength"].quantile([0.33, 0.66]).values if perf_df["strength"].notna().any() else [0, 0]
    q_spread = perf_df["rel_spread_in"].quantile([0.33, 0.66]).values if perf_df["rel_spread_in"].notna().any() else [0, 0]

    def _bin_strength(s):
        if pd.isna(s): return "NA"
        if s <= q_strength[0]: return "lowS"
        if s <= q_strength[1]: return "midS"
        return "highS"

    def _bin_spread(s):
        if pd.isna(s): return "NA"
        if s <= q_spread[0]: return "lowC"
        if s <= q_spread[1]: return "midC"
        return "highC"

    perf_df["bin_strength"] = perf_df["strength"].apply(_bin_strength)
    perf_df["bin_cost"] = perf_df["rel_spread_in"].apply(_bin_spread)

    by_bins = {}
    for H in horizons:
        col = f"ret_{H}s_net"
        grp = perf_df.groupby(["bin_strength", "bin_cost"])[col].agg(
            mean="mean", median="median", winrate=lambda s: (s > 0).mean(), n="count"
        ).reset_index()
        by_bins[f"H{H}"] = grp

    return perf_df, summary, by_bins    


# 1) 先產生 signals
analyzer = MicrostructureTrend(
    lookback_seconds=60,
    slope_window=10,
    breakout_alpha=0.001,
    stay_confirm_M=3,
    max_rel_spread=0.0012,
    slip_k=2000,
    cooldown_seconds=30,
    quant_win=300
)
signals = analyzer.generate_trend_signals(df_ticks)

# 2) 回測（以 30/60/120 秒視窗）
perf_df, summary, by_bins = backtest_signals(
    df_ticks=df_ticks,
    signals=signals,
    horizons=[30, 60, 120],
    cost_model="baseline",     # 初期建議 baseline，比較穩健
    slip_coeff=0.6,
    require_same_side=True,
    max_holding_seconds=None   # 可先不限制，之後再試 180 或 300
)

print("Backtest summary:")
print(summary)
print("\nPerf head:")
print(perf_df.head())

print("\nBy bins (H60 example):")
print(by_bins["H60"])