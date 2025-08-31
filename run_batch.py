symbols = load_csi1000_list()          # 你的CSI1000 RIC清單
start_date, end_date = "2025-08-01", "2025-08-31"  # 先用7天再擴
analyzer = MicrostructureTrend(...你的參數...)

all_trades = []
all_summary_rows = []

for ric in symbols:
    df_ticks = load_l1_ticks(ric, start_date, end_date)  # 需含 time, bidprice, bidsize, askprice, asksize
    df_ticks = clean_cut_sessions(df_ticks)              # 去開/收盤邊界、午休斷開

    signals = analyzer.generate_trend_signals(df_ticks)
    perf_df, summary, by_bins = backtest_signals(
        df_ticks=df_ticks,
        signals=signals,
        horizons=[30,60,120],
        cost_model="baseline",
        slip_coeff=0.6,
        require_same_side=True
    )
    if len(perf_df) == 0:
        continue

    perf_df["ric"] = ric
    all_trades.append(perf_df)

    row = {"ric": ric, "trades": summary.get("trades", 0)}
    for H in [30,60,120]:
        s = summary.get(f"ret_{H}s", {})
        row.update({f"H{H}_mean": s.get("mean", np.nan),
                    f"H{H}_win": s.get("winrate", np.nan),
                    f"H{H}_sharpe": s.get("sharpe", np.nan)})
    all_summary_rows.append(row)

all_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
all_summary = pd.DataFrame(all_summary_rows).sort_values("H60_mean", ascending=False)

# 保存
all_trades.to_csv("csi1000_7d_trades.csv", index=False)
all_summary.to_csv("csi1000_7d_summary.csv", index=False)

print("Top 20 by H60_mean:")
print(all_summary.head(20))