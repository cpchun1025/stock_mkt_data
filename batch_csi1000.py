import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import bisect
from pathlib import Path

# ===================== 核心模組（與前文一致，略微濃縮） =====================

EPS = 1e-12

def _merge_time_to_index(df: pd.DataFrame):
    df = df.sort_values("time").reset_index(drop=True).copy()
    base = df["time"].iloc[0]
    df["_tsec"] = (df["time"] - base).dt.total_seconds().astype(float)
    return df, base

def _find_index_at_time(tsecs: np.ndarray, target: float) -> int:
    return bisect.bisect_right(tsecs, target) - 1

def _zscore(s: pd.Series, win: int = 300) -> pd.Series:
    m = s.rolling(win, min_periods=max(5, win // 3)).mean()
    v = s.rolling(win, min_periods=max(5, win // 3)).std()
    return ((s - m) / (v + EPS)).fillna(0.0)

def _rolling_slope(series: pd.Series, window: int) -> float:
    x = series.values
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n)
    t_mean = t.mean()
    x_mean = x.mean()
    denom = np.sum((t - t_mean) ** 2) + EPS
    return float(np.sum((t - t_mean) * (x - x_mean)) / denom)

def _confirm_series(flag_bool_arr: np.ndarray, M: int) -> np.ndarray:
    c = np.zeros(len(flag_bool_arr), dtype=bool)
    run = 0
    for i, f in enumerate(flag_bool_arr):
        run = run + 1 if f else 0
        c[i] = (run >= M)
    return c

@dataclass
class MicrostructureTrend:
    lookback_seconds: int = 120
    slope_window: int = 10
    breakout_alpha: float = 0.001
    stay_confirm_M: int = 3
    max_rel_spread: float = 0.0012
    slip_k: float = 2000.0
    cooldown_seconds: int = 30
    quant_win: int = 300
    window_mode: str = "time"   # "time" or "count"
    count_n: int = 120
    use_flat_filter: bool = True

    def calculate_micro_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values("time").reset_index(drop=True)
        df["mid"] = (df["askprice"] + df["bidprice"]) / 2.0
        df["spread"] = (df["askprice"] - df["bidprice"]).clip(lower=0)
        df["rel_spread"] = (df["spread"] / (df["mid"] + EPS)).fillna(0.0)
        df["depth_imb"] = (df["bidsize"] - df["asksize"]) / (df["bidsize"] + df["asksize"] + EPS)
        df["time_seconds"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

        roll_std = df["mid"].rolling(120, min_periods=30).std().fillna(0.0)
        spread_med = df["spread"].rolling(600, min_periods=100).median().fillna(df["spread"].median())

        micro_high = np.full(len(df), np.nan)
        micro_low = np.full(len(df), np.nan)
        micro_high_prev = np.full(len(df), np.nan)
        micro_low_prev = np.full(len(df), np.nan)
        ts = df["time_seconds"].values
        j0 = 0

        if self.window_mode == "time":
            for i in range(len(df)):
                t_now = ts[i]
                while ts[j0] < t_now - self.lookback_seconds:
                    j0 += 1
                wa = df["askprice"].iloc[j0:i+1]
                wb = df["bidprice"].iloc[j0:i+1]
                if len(wa) > 0:
                    micro_high[i] = wa.max()
                    micro_low[i] = wb.min()
                L = max(0, j0 - 1)
                R = max(0, i - 1)
                if R >= L:
                    micro_high_prev[i] = df["askprice"].iloc[L:R+1].max()
                    micro_low_prev[i] = df["bidprice"].iloc[L:R+1].min()
        else:
            N = int(self.count_n)
            df["ask_high_N"] = df["askprice"].rolling(N, min_periods=max(5, N//3)).max()
            df["bid_low_N"] = df["bidprice"].rolling(N, min_periods=max(5, N//3)).min()
            df["ask_high_prevN"] = df["askprice"].shift(1).rolling(N, min_periods=max(5, N//3)).max()
            df["bid_low_prevN"] = df["bidprice"].shift(1).rolling(N, min_periods=max(5, N//3)).min()
            micro_high = df["ask_high_N"].values
            micro_low = df["bid_low_N"].values
            micro_high_prev = df["ask_high_prevN"].values
            micro_low_prev = df["bid_low_prevN"].values

        df["micro_high"] = micro_high
        df["micro_low"] = micro_low
        df["micro_high_prev"] = micro_high_prev
        df["micro_low_prev"] = micro_low_prev

        df["raw_range_width"] = (df["micro_high"] - df["micro_low"]).clip(lower=0).fillna(0.0)
        floor = np.maximum(0.5 * roll_std, 1.0 * spread_med)
        df["range_width"] = np.maximum(df["raw_range_width"], floor)
        df["position_in_range"] = ((df["mid"] - df["micro_low"]) / (df["range_width"] + EPS)).clip(0.0, 1.0)

        df["slope_short"] = df["mid"].rolling(self.slope_window, min_periods=2).apply(
            lambda x: _rolling_slope(pd.Series(x), self.slope_window), raw=False
        ).fillna(0.0)
        df["z_slope"] = _zscore(df["slope_short"], win=300)
        df["z_spread"] = _zscore(df["rel_spread"], win=300)
        df["trend_strength"] = df["z_slope"] - 0.5 * df["z_spread"]

        alpha = self.breakout_alpha
        df["breakout_up_raw"] = (df["mid"] > (df["micro_high_prev"] * (1.0 + alpha)))
        df["breakout_down_raw"] = (df["mid"] < (df["micro_low_prev"] * (1.0 - alpha)))
        df["breakout_up"] = _confirm_series(df["breakout_up_raw"].fillna(False).values, self.stay_confirm_M)
        df["breakout_down"] = _confirm_series(df["breakout_down_raw"].fillna(False).values, self.stay_confirm_M)

        Nq = self.quant_win
        ask_q = df["asksize"].rolling(Nq, min_periods=max(5, Nq//3)).quantile(0.2)\
                  .fillna(method="bfill").fillna(df["asksize"].median())
        bid_q = df["bidsize"].rolling(Nq, min_periods=max(5, Nq//3)).quantile(0.2)\
                  .fillna(method="bfill").fillna(df["bidsize"].median())
        df["ask_thin"] = (df["asksize"] <= ask_q)
        df["bid_thin"] = (df["bidsize"] <= bid_q)

        df["slip_proxy_up"] = df["rel_spread"] + (self.slip_k / (df["asksize"] + 1.0))
        df["slip_proxy_down"] = df["rel_spread"] + (self.slip_k / (df["bidsize"] + 1.0))

        if self.use_flat_filter:
            roll_std = roll_std.reindex(df.index, fill_value=0)
            spread_med = spread_med.reindex(df.index, fill_value=df["spread"].median())
            df["flat_regime"] = (df["raw_range_width"] <= 1.5 * spread_med) & (roll_std <= spread_med * 0.5)
        else:
            df["flat_regime"] = False

        cols = [
            "time","time_seconds","mid","bidprice","askprice","bidsize","asksize",
            "spread","rel_spread","depth_imb",
            "micro_high","micro_low","raw_range_width","range_width","position_in_range",
            "slope_short","z_slope","z_spread","trend_strength",
            "micro_high_prev","micro_low_prev",
            "breakout_up_raw","breakout_down_raw","breakout_up","breakout_down",
            "ask_thin","bid_thin","slip_proxy_up","slip_proxy_down",
            "flat_regime"
        ]
        if "volume" in df.columns and "volume" not in cols:
            cols.append("volume")
        return df[cols]

    def generate_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        res = self.calculate_micro_trends(df).copy()

        pir = res["position_in_range"]
        res["trend_signal"] = "Neutral"
        res.loc[pir >= 0.6, "trend_signal"] = "Strong_Up"
        res.loc[(pir < 0.6) & (pir >= 0.5), "trend_signal"] = "Weak_Up"
        res.loc[pir <= 0.4, "trend_signal"] = "Strong_Down"
        res.loc[(pir > 0.4) & (pir < 0.5), "trend_signal"] = "Weak_Down"

        up_ok = (res["breakout_up"] &
                 (res["trend_strength"] > 0) &
                 (res["rel_spread"] <= self.max_rel_spread) &
                 (res["ask_thin"]) &
                 (res["slip_proxy_up"] <= (self.max_rel_spread * 2.0)))
        down_ok = (res["breakout_down"] &
                   (res["trend_strength"] < 0) &
                   (res["rel_spread"] <= self.max_rel_spread) &
                   (res["bid_thin"]) &
                   (res["slip_proxy_down"] <= (self.max_rel_spread * 2.0)))

        if self.use_flat_filter and "flat_regime" in res.columns:
            flat_block = res["flat_regime"].fillna(False)
            up_ok = up_ok & (~flat_block)
            down_ok = down_ok & (~flat_block)

        res["momentum_signal"] = "Neutral"
        res.loc[up_ok, "momentum_signal"] = "Bullish_Breakout"
        res.loc[down_ok, "momentum_signal"] = "Bearish_Breakout"

        res["entry_signal"] = False
        last_entry_t = -1e18
        timesec = res["time_seconds"].values
        for i in range(len(res)):
            if res["momentum_signal"].iat[i] != "Neutral":
                if timesec[i] - last_entry_t >= self.cooldown_seconds:
                    res["entry_signal"].iat[i] = True
                    last_entry_t = timesec[i]

        return res[[
            "time","mid","trend_signal","momentum_signal","entry_signal",
            "breakout_up","breakout_down","position_in_range",
            "rel_spread","asksize","bidsize","ask_thin","bid_thin",
            "trend_strength","range_width","slip_proxy_up","slip_proxy_down",
            "flat_regime"
        ]]

def backtest_signals(
    df_ticks: pd.DataFrame,
    signals: pd.DataFrame,
    horizons: List[int] = [30, 60, 120],
    cost_model: str = "baseline",
    slip_coeff: float = 0.6,
    side_map: Dict[str, int] = {"Bullish_Breakout": 1, "Bearish_Breakout": -1},
    require_same_side: bool = True,
    max_holding_seconds: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    ticks, _ = _merge_time_to_index(df_ticks)
    sigs, _ = _merge_time_to_index(signals)
    sigs = sigs[sigs["entry_signal"]].copy()

    if require_same_side and "trend_signal" in sigs:
        up_mask = (sigs["momentum_signal"] == "Bullish_Breakout") & (sigs["trend_signal"].str.contains("Up"))
        dn_mask = (sigs["momentum_signal"] == "Bearish_Breakout") & (sigs["trend_signal"].str.contains("Down"))
        sigs = sigs[up_mask | dn_mask].copy()

    if len(sigs) == 0:
        return pd.DataFrame(), {}, {}

    tsec_ticks = ticks["_tsec"].values
    perf_rows = []

    for _, row in sigs.iterrows():
        t0 = row["_tsec"]
        side = side_map.get(row["momentum_signal"], 0)
        if side == 0:
            continue

        idx0 = _find_index_at_time(tsec_ticks, t0)
        if idx0 < 0:
            continue

        px_in = ticks["mid"].iat[idx0]
        if cost_model == "baseline":
            cost_in = 0.5 * row.get("rel_spread", 0.0)
        else:
            cost_in = slip_coeff * (row.get("slip_proxy_up", row.get("rel_spread", 0.0)) if side > 0
                                    else row.get("slip_proxy_down", row.get("rel_spread", 0.0)))

        horizons_use = list(horizons)
        if max_holding_seconds is not None and max_holding_seconds not in horizons_use:
            horizons_use = sorted(set(horizons_use + [max_holding_seconds]))
        max_h = max(horizons_use)

        t_end = t0 + max_h
        idx_end = _find_index_at_time(tsec_ticks, t_end)
        if idx_end <= idx0:
            continue

        future_mid = ticks["mid"].iloc[idx0:idx_end+1].values
        ret_path = side * (future_mid - px_in) / (px_in + EPS)
        mfe = float(np.max(ret_path)) if len(ret_path) else 0.0
        mae = float(np.min(ret_path)) if len(ret_path) else 0.0

        metrics = {
            "time": row["time"],
            "ric": row.get("ric", None),
            "side": side,
            "px_in": float(px_in),
            "strength": float(row.get("trend_strength", np.nan)),
            "rel_spread_in": float(row.get("rel_spread", np.nan)),
            "asksize_in": float(row.get("asksize", np.nan)),
            "bidsize_in": float(row.get("bidsize", np.nan)),
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
            rel_spread_out = ticks["rel_spread"].iat[idxH] if "rel_spread" in ticks.columns else row.get("rel_spread", 0.0)
            if cost_model == "baseline":
                cost_out = 0.5 * rel_spread_out
            else:
                cost_out = slip_coeff * rel_spread_out
            gross = side * (px_out - px_in) / (px_in + EPS)
            net = gross - cost_in - cost_out
            metrics[f"ret_{H}s_net"] = float(net)

        if max_holding_seconds is not None:
            idxM = _find_index_at_time(tsec_ticks, t0 + max_holding_seconds)
            if idxM > idx0:
                px_outM = ticks["mid"].iat[idxM]
                rel_spread_outM = ticks["rel_spread"].iat[idxM] if "rel_spread" in ticks.columns else row.get("rel_spread", 0.0)
                cost_outM = 0.5 * rel_spread_outM if cost_model == "baseline" else slip_coeff * rel_spread_outM
                grossM = side * (px_outM - px_in) / (px_in + EPS)
                metrics[f"ret_{max_holding_seconds}s_net"] = float(grossM - cost_in - cost_outM)

        perf_rows.append(metrics)

    if len(perf_rows) == 0:
        return pd.DataFrame(), {}, {}

    perf_df = pd.DataFrame(perf_rows).sort_values("time").reset_index(drop=True)

    def _summary_of(col: str):
        x = perf_df[col].dropna().values
        if len(x) == 0:
            return {"mean": np.nan, "median": np.nan, "winrate": np.nan, "sharpe": np.nan, "n": 0}
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "winrate": float((x > 0).mean()),
            "sharpe": float(np.mean(x) / (np.std(x, ddof=0) + EPS)),
            "n": int(len(x))
        }

    summary = {"trades": int(len(perf_df)), "mfe_mean": float(perf_df["mfe"].mean()), "mae_mean": float(perf_df["mae"].mean())}
    for H in horizons:
        summary[f"ret_{H}s"] = _summary_of(f"ret_{H}s_net")

    q_strength = perf_df["strength"].quantile([0.33, 0.66]).values if perf_df["strength"].notna().any() else [0, 0]
    q_spread = perf_df["rel_spread_in"].quantile([0.33, 0.66]).values if perf_df["rel_spread_in"].notna().any() else [0, 0]

    def _bin_strength(s):
        if pd.isna(s): return "NA"
        if s <= q_strength[0]: return "lowS"
        if s <= q_strength[1]: return "midS"
        return "highS"

    def _bin_cost(s):
        if pd.isna(s): return "NA"
        if s <= q_spread[0]: return "lowC"
        if s <= q_spread[1]: return "midC"
        return "highC"

    perf_df["bin_strength"] = perf_df["strength"].apply(_bin_strength)
    perf_df["bin_cost"] = perf_df["rel_spread_in"].apply(_bin_cost)

    by_bins = {}
    for H in horizons:
        col = f"ret_{H}s_net"
        grp = perf_df.groupby(["bin_strength", "bin_cost"], dropna=False)[col].agg(
            mean="mean", median="median", winrate=lambda s: (s > 0).mean(), n="count"
        ).reset_index()
        by_bins[f"H{H}"] = grp

    return perf_df, summary, by_bins

# ===================== 批量處理：載入 -> 信號 -> 回測 -> 匯出 =====================

def load_ticks_for_ric(path: str, time_col: str = "time") -> pd.DataFrame:
    """
    載入單一標的 L1 資料，並標準化欄位名稱。
    需要欄位（可同名或映射）：time, bidprice, askprice, bidsize, asksize（可選 volume）。
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # 欄位名映射：如你的實際欄位不同，請在此調整
    colmap = {
        time_col: "time",
        "bid": "bidprice",
        "ask": "askprice",
        "bidsize": "bidsize",
        "asksize": "asksize",
        "bidprice": "bidprice",
        "askprice": "askprice",
        "volume": "volume"
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    # 檢查必需列
    required = ["time", "bidprice", "askprice", "bidsize", "asksize"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    # 轉時間
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # 可選去掉非交易時段/開收盤極端（視交易所而定），此處預留鉤子
    # 例如：A股 09:30-11:30, 13:00-15:00，可自行加過濾。

    return df

def process_one_ric(path: str,
                    ric: str,
                    horizons: List[int],
                    analyzer_kwargs: Dict,
                    backtest_kwargs: Dict,
                    save_perf_csv: bool,
                    perf_dir: str) -> Tuple[str, Dict, Dict[str, pd.DataFrame]]:
    """處理單一 RIC：載入 -> 產生信號 -> 回測 -> 匯出 perf（可選） -> 回傳摘要與分層"""
    try:
        df_ticks = load_ticks_for_ric(path)
        analyzer = MicrostructureTrend(**analyzer_kwargs)
        signals = analyzer.generate_trend_signals(df_ticks)
        # 把 ric 帶進 signals，方便回測保存
        signals["ric"] = ric

        perf_df, summary, by_bins = backtest_signals(
            df_ticks=df_ticks,
            signals=signals,
            horizons=horizons,
            **backtest_kwargs
        )
        # 附帶 ric 字段
        if len(perf_df) > 0:
            perf_df["ric"] = ric

        # 匯出明細（可選）
        if save_perf_csv and len(perf_df) > 0:
            Path(perf_dir).mkdir(parents=True, exist_ok=True)
            perf_path = os.path.join(perf_dir, f"{ric}_perf.csv")
            perf_df.to_csv(perf_path, index=False)

        # 序列化 summary：攤平成單層 dict
        flat_summary = {"ric": ric, "trades": summary.get("trades", 0),
                        "mfe_mean": summary.get("mfe_mean", np.nan),
                        "mae_mean": summary.get("mae_mean", np.nan)}
        for H in horizons:
            sH = summary.get(f"ret_{H}s", {})
            flat_summary.update({
                f"ret{H}_mean": sH.get("mean", np.nan),
                f"ret{H}_median": sH.get("median", np.nan),
                f"ret{H}_winrate": sH.get("winrate", np.nan),
                f"ret{H}_sharpe": sH.get("sharpe", np.nan),
                f"ret{H}_n": sH.get("n", 0)
            })

        return ric, flat_summary, {f"H{H}": df.assign(ric=ric) for H, df in by_bins.items()}
    except Exception as e:
        # 失敗也回報，便於追蹤
        return ric, {"ric": ric, "error": str(e)}, {}

def batch_run_csi1000(
    data_dir: str,
    pattern: str = "*.parquet",        # 或 "*.csv"
    horizons: List[int] = [30, 60, 120],
    analyzer_kwargs: Dict = None,
    backtest_kwargs: Dict = None,
    workers: int = 8,
    save_perf_csv: bool = False,
    out_dir: str = "out"
):
    """
    批量執行流程：
    - data_dir：資料目錄，每檔一標的（檔名可視為 RIC）
    - pattern：檔案匹配模式
    - horizons：回測視窗
    - analyzer_kwargs：傳給 MicrostructureTrend 的參數字典
    - backtest_kwargs：傳給 backtest_signals 的參數字典（除 df_ticks/signals/horizons）
    - workers：並行進程數
    - save_perf_csv：是否匯出每標的交易明細 CSV
    - out_dir：輸出目錄
    產出：
    - out/summary.csv：總表
    - out/by_bins_H{H}.csv：每個 H 的分層合併表
    - out/perf/{RIC}_perf.csv：每標的交易明細（可選）
    """
    analyzer_kwargs = analyzer_kwargs or dict(
        lookback_seconds=120,
        slope_window=10,
        breakout_alpha=0.001,
        stay_confirm_M=3,
        max_rel_spread=0.0012,
        slip_k=2000.0,
        cooldown_seconds=30,
        quant_win=300,
        window_mode="time",
        count_n=120,
        use_flat_filter=True
    )
    backtest_kwargs = backtest_kwargs or dict(
        cost_model="baseline",
        slip_coeff=0.6,
        require_same_side=True,
        max_holding_seconds=None
    )

    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} with pattern {pattern}")

    # RIC 從檔名提取（去副檔名）
    rics = [Path(f).stem for f in files]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    perf_dir = out_dir / "perf"
    bybins_collect: Dict[str, List[pd.DataFrame]] = {f"H{H}": [] for H in horizons}

    summaries: List[Dict] = []
    tasks = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, ric in zip(files, rics):
            tasks.append(ex.submit(
                process_one_ric, path, ric, horizons, analyzer_kwargs, backtest_kwargs, save_perf_csv, str(perf_dir)
            ))

        for fut in as_completed(tasks):
            ric, flat_summary, by_bins = fut.result()
            if "error" in flat_summary:
                print(f"[ERROR] {ric}: {flat_summary['error']}")
            else:
                summaries.append(flat_summary)
                for H, dfH in by_bins.items():
                    bybins_collect[H].append(dfH)

    # 合併輸出
    summary_df = pd.DataFrame(summaries).sort_values("ric").reset_index(drop=True)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    for H, lst in bybins_collect.items():
        if len(lst) == 0:
            continue
        merged = pd.concat(lst, ignore_index=True)
        outp = out_dir / f"by_bins_{H}.csv"
        merged.to_csv(outp, index=False)
        print(f"By-bins ({H}) saved to: {outp}")

    if save_perf_csv:
        print(f"Per-RIC trade details saved under: {perf_dir}")

    return summary_df

# ===================== 直接執行示例 =====================

if __name__ == "__main__":
    """
    使用方法：
    1) 準備資料：每標的一檔（CSV 或 Parquet），欄位含 time, bid/ask price & size。
       例如：data/CSI1000/000001.SZ.parquet, 000002.SZ.parquet, ...
    2) 設定資料目錄與副檔名 pattern。
    3) 選擇是否匯出每筆交易明細（save_perf_csv）。
    4) 跑起來，輸出 summary.csv、by_bins_Hxx.csv、perf/*.csv。
    """
    data_dir = "data/CSI1000"      # TODO: 換成你的資料目錄
    pattern = "*.parquet"          # 或 "*.csv"
    horizons = [30, 60, 120]

    analyzer_kwargs = dict(
        lookback_seconds=120,
        slope_window=10,
        breakout_alpha=0.001,
        stay_confirm_M=3,
        max_rel_spread=0.0012,
        slip_k=2000.0,
        cooldown_seconds=30,
        quant_win=300,
        window_mode="time",   # 也可試 "count"
        count_n=120,
        use_flat_filter=True
    )

    backtest_kwargs = dict(
        cost_model="baseline",
        slip_coeff=0.6,
        require_same_side=True,
        max_holding_seconds=None
    )

    summary = batch_run_csi1000(
        data_dir=data_dir,
        pattern=pattern,
        horizons=horizons,
        analyzer_kwargs=analyzer_kwargs,
        backtest_kwargs=backtest_kwargs,
        workers=8,
        save_perf_csv=True,     # 若只要總表，改為 False
        out_dir="out"
    )

    print("Done. Summary head:")
    print(summary.head())