#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一目均衡表（9, 26, 52）のマルチタイムフレーム（MTF）シグナル通知ボット。

D1（日足）で相場認識、M15（15分足）でトリガーを判定し、両方が揃ったら
アラート（コンソール + ログ + 任意の webhook）を出します。

目視での判断を前提とするため、**このボットは注文を出しません**。シグナルを見て
人間がチャートを確認し、状況に応じて売買します。

判定ロジック:
- 【環境認識】D1 の終値が D1 の「雲（先行スパン1と2の上限/下限）」の上/下にあるか。
   - 雲の上 → 買い目線（LONG_ONLY）、雲の下 → 売り目線（SHORT_ONLY）、雲の中 → 静観。
- 【トリガー】M15 の（完成済み）終値が M15 の「基準線(26)」を
   - 下→上にクロスアップ、または基準線付近で反発して陽線確定 → 買い。
   - 上→下にクロスダウン、または基準線付近で反落して陰線確定 → 売り。

一目均衡表の対応（9, 26, 52）:
  転換線(9) = (9本の高値+安値)/2
  基準線(26) = (26本の高値+安値)/2   ← トリガーで使う線
  先行スパン1 = (転換線+基準線)/2
  先行スパン2 = (52本の高値+安値)/2
  雲 = 先行スパン1 と 2 の間。先行スパンは26本先へ描画されるため、
       「現在の足の位置にある雲」は 26 本前に算出された値を使う（見通しバイアス無し）。

使い方:
  1. 起動中の MT5 端末にログイン済み（または config_login.json で口座指定）。
  2. D1 と M15 の履歴が端末に同期されていること（無ければ該当チャートを一度開く）。
  3. python ichimoku_alert.py                 # 5分ごとに巡回（既定）
     python ichimoku_alert.py --once          # 1回だけ判定して終了
     python ichimoku_alert.py --symbols EURUSD,USDJPY --interval 60
     python ichimoku_alert.py --selftest      # MT5 無しでロジック自己診断

銘柄は config.json の symbols（テストで使っていたものと同じ）を既定で使う。
"""
import argparse
import json
import logging
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import load_config

# ---------- 一目均衡表パラメータ（9, 26, 52） ----------
TENKAN_N = 9      # 転換線
KIJUN_N = 26      # 基準線（トリガーの線）
SENKOU_B_N = 52   # 先行スパン2
KUMO_SHIFT = 26   # 先行スパン（雲）の描画シフト本数

# ---------- 判定パラメータ ----------
ENV_TF = "D1"            # 環境認識の時間足
TRIGGER_TF = "M15"       # トリガーの時間足
D1_BARS = 200            # 環境認識に使う履歴本数（雲 shift=26 を余裕で賄う）
M15_BARS = 100           # トリガーに使う履歴本数（基準線 26 + クロス検出分）
MIN_D1_BARS = SENKOU_B_N + KUMO_SHIFT + 1   # 雲計算に最低必要な D1 本数（52+26+α）
MIN_M15_BARS = KIJUN_N + 2                  # 基準線 + クロス検出に最低必要な M15 本数
ATR_N = 14               # 「基準線付近」判定に使う ATR の期間
TOUCH_ATR_MULT = 0.5     # 高値/安値が基準線からこの ATR 倍数以内なら「付近で反発」とみなす

POLL_INTERVAL_SEC = 300  # 巡回間隔（秒）。要件どおり既定 5 分
LOG_FILE = "ichimoku_alert.log"

# 任意: 汎用 webhook（Slack / Discord の Incoming Webhook 等）。空文字なら送信しない。
WEBHOOK_URL = ""

log = logging.getLogger("ichimoku.alert")


@dataclass
class Signal:
    symbol: str
    side: str            # "LONG" / "SHORT"
    trigger: str         # "cross_up" / "cross_down" / "bounce_long" / "bounce_short"
    bar_time: pd.Timestamp
    d1_close: float
    cloud_top: float
    cloud_bottom: float
    m15_close: float
    kijun: float         # 基準線(26)
    tenkan: float        # 転換線(9)


# ---------------------------------------------------------------------------
# 一目均衡表の計算（純粋関数・MT5 に依存しない）
# ---------------------------------------------------------------------------
def ichimoku_series(df: pd.DataFrame, tenkan_n: int = TENKAN_N,
                    kijun_n: int = KIJUN_N, senkou_b_n: int = SENKOU_B_N,
                    shift: int = KUMO_SHIFT):
    """DataFrame（open/high/low/close 列必須）から一目均衡表の系列を返す。

    戻り値: (転換線, 基準線, 雲上限, 雲下限)
    雲は 26 本先へ描画されるため、shift 分ずらして「現在の足の位置の雲」に合わせる。
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tenkan = (high.rolling(tenkan_n).max() + low.rolling(tenkan_n).min()) / 2.0
    kijun = (high.rolling(kijun_n).max() + low.rolling(kijun_n).min()) / 2.0
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = (high.rolling(senkou_b_n).max() + low.rolling(senkou_b_n).min()) / 2.0
    cloud_top = pd.Series(np.maximum(senkou_a, senkou_b), index=df.index).shift(shift)
    cloud_bottom = pd.Series(np.minimum(senkou_a, senkou_b), index=df.index).shift(shift)
    return tenkan, kijun, cloud_top, cloud_bottom


def atr_series(df: pd.DataFrame, period: int = ATR_N) -> pd.Series:
    """Average True Range（基準線付近の判定に使う）。"""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def drop_forming(df: pd.DataFrame) -> pd.DataFrame:
    """copy_rates_from_pos の末尾にある「未確定の形成中足」を除く。

    シグナルは完成済みの足でのみ判定する（形成中の足を使うと途中で値が変わって
    判定が揺れるため）。
    """
    if df is None or len(df) == 0:
        return df
    return df.iloc[:-1]


def environment(close: float, cloud_top: float, cloud_bottom: float) -> str:
    """D1 終値の雲との位置から相場認識を返す。

    戻り値: "LONG_ONLY" / "SHORT_ONLY" / "CLOUD"（雲の中は静観）
    """
    if pd.isna(close) or pd.isna(cloud_top) or pd.isna(cloud_bottom):
        return "CLOUD"
    if close > cloud_top:
        return "LONG_ONLY"
    if close < cloud_bottom:
        return "SHORT_ONLY"
    return "CLOUD"


def detect_trigger(m15: pd.DataFrame, tenkan: pd.Series, kijun: pd.Series,
                   mode: str, touch_atr_mult: float = TOUCH_ATR_MULT) -> Optional[str]:
    """完成済み M15 の直近2本からトリガーを判定する。

    mode は environment() の結果（"LONG_ONLY"/"SHORT_ONLY"）。トリガー名か None を返す。
    """
    if len(m15) < 2:
        return None
    prev, cur = m15.iloc[-2], m15.iloc[-1]
    prev_close, cur_close = float(prev["close"]), float(cur["close"])
    prev_k, cur_k = kijun.iloc[-2], kijun.iloc[-1]
    if pd.isna(prev_k) or pd.isna(cur_k):
        return None

    touch = float(touch_atr_mult * atr_series(m15).iloc[-1])

    if mode == "LONG_ONLY":
        cross_up = prev_close <= prev_k < cur_close
        bounce = (cur_close > cur["open"]            # 陽線確定
                  and cur_close > cur_k              # 基準線より上で引けた
                  and cur["low"] <= cur_k + touch)   # 安値が基準線付近まで寄った（反発）
        if cross_up:
            return "cross_up"
        if bounce:
            return "bounce_long"
    elif mode == "SHORT_ONLY":
        cross_down = prev_close >= prev_k > cur_close
        bounce = (cur_close < cur["open"]            # 陰線確定
                  and cur_close < cur_k              # 基準線より下で引けた
                  and cur["high"] >= cur_k - touch)  # 高値が基準線付近まで寄った（反落）
        if cross_down:
            return "cross_down"
        if bounce:
            return "bounce_short"
    return None


def evaluate(symbol: str, d1: pd.DataFrame, m15: pd.DataFrame) -> Optional[Signal]:
    """1 銘柄分の判定。シグナルがあれば Signal、無ければ None。"""
    if d1 is None or m15 is None or len(d1) < 3 or len(m15) < 3:
        return None
    _, _, cloud_top, cloud_bottom = ichimoku_series(d1)
    mode = environment(float(d1["close"].iloc[-1]),
                       float(cloud_top.iloc[-1]),
                       float(cloud_bottom.iloc[-1]))
    if mode == "CLOUD":
        return None

    tenkan, kijun, _, _ = ichimoku_series(m15)
    trigger = detect_trigger(m15, tenkan, kijun, mode)
    if trigger is None:
        return None

    side = "LONG" if mode == "LONG_ONLY" else "SHORT"
    return Signal(
        symbol=symbol,
        side=side,
        trigger=trigger,
        bar_time=m15["time"].iloc[-1] if "time" in m15.columns else pd.NaT,
        d1_close=float(d1["close"].iloc[-1]),
        cloud_top=float(cloud_top.iloc[-1]),
        cloud_bottom=float(cloud_bottom.iloc[-1]),
        m15_close=float(m15["close"].iloc[-1]),
        kijun=float(kijun.iloc[-1]),
        tenkan=float(tenkan.iloc[-1]),
    )


def format_signal(sig: Signal) -> str:
    """アラート本文（人が見て判断しやすいように要点を並べる）。"""
    trig_label = {
        "cross_up": "基準線を下→上にクロスアップ",
        "cross_down": "基準線を上→下にクロスダウン",
        "bounce_long": "基準線付近で反発し陽線確定",
        "bounce_short": "基準線付近で反落し陰線確定",
    }[sig.trigger]
    return (
        f"[一目均衡 MTF シグナル] {sig.side} {sig.symbol}  "
        f"{pd.to_datetime(sig.bar_time)} ({TRIGGER_TF})\n"
        f"環境認識 ({ENV_TF}): 終値 {sig.d1_close:.5f} は雲"
        f"{'上限' if sig.side == 'LONG' else '下限'} {sig.cloud_top if sig.side == 'LONG' else sig.cloud_bottom:.5f}"
        f" の{'上' if sig.side == 'LONG' else '下'} → "
        f"{'買い目線' if sig.side == 'LONG' else '売り目線'}\n"
        f"トリガー ({TRIGGER_TF}): {trig_label}（終値 {sig.m15_close:.5f}）\n"
        f"基準線(26)={sig.kijun:.5f} / 転換線(9)={sig.tenkan:.5f} / "
        f"雲(先行) 上限={sig.cloud_top:.5f} 下限={sig.cloud_bottom:.5f}\n"
        f"→ チャートでご確認のうえ、ご自身の判断で売買してください。"
    )


def notify(msg: str, webhook_url: str = WEBHOOK_URL) -> None:
    """アラート送信（コンソール + ログ + 任意の webhook）。"""
    log.info("シグナル検知:\n%s", msg)
    print("\n" + msg + "\n", flush=True)
    print("\a", end="", flush=True)  # 端末ベル（鳴る環境のみ）
    if webhook_url:
        try:
            _post_webhook(webhook_url, msg)
        except Exception as exc:
            log.warning("webhook 送信失敗: %s", exc)


def _post_webhook(url: str, text: str) -> int:
    """JSON を POST する汎用 webhook（Slack/Discord の Incoming Webhook 形式）。"""
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


# ---------------------------------------------------------------------------
# メインループ（MT5 接続はここで遅延 import する）
# ---------------------------------------------------------------------------
def check_once(broker, symbols: list, env_tf: str, trigger_tf: str) -> dict:
    """全銘柄を 1 回判定し、シグナルが出た銘柄を dict で返す。"""
    new = {}
    for sym in symbols:
        d1 = drop_forming(broker.get_rates(sym, count=D1_BARS, timeframe=env_tf))
        m15 = drop_forming(broker.get_rates(sym, count=M15_BARS, timeframe=trigger_tf))
        if d1 is None or len(d1) < MIN_D1_BARS:
            log.warning("%s の %s 履歴が不足/取得不可（雲計算に約%d本必要。チャートを開いて同期してください）",
                        sym, env_tf, MIN_D1_BARS)
            continue
        if m15 is None or len(m15) < MIN_M15_BARS:
            log.warning("%s の %s 履歴が不足/取得不可", sym, trigger_tf)
            continue
        sig = evaluate(sym, d1, m15)
        if sig is not None:
            new[sym] = sig
    return new


def run(args) -> None:
    cfg = load_config()
    symbols = args.symbols if args.symbols else cfg.symbols
    if not symbols:
        sys.exit("銘柄リストが空です。config.json の symbols か --symbols で指定してください。")

    from broker import MT5Broker
    broker = MT5Broker(cfg)
    broker.connect()
    log.info("一目均衡 MTF シグナル通知ボットを開始します（銘柄 %d / 巡回 %d 秒）",
             len(symbols), args.interval)

    # 同じ M15 足での重複通知を防ぐ（(銘柄, 方向) → 直近で通知済みの足時刻）
    last_fired = {}
    try:
        while True:
            for sym, sig in check_once(broker, symbols, args.tf_env, args.tf_trigger).items():
                key = (sym, sig.side)
                if last_fired.get(key) == sig.bar_time:
                    continue
                last_fired[key] = sig.bar_time
                notify(format_signal(sig), webhook_url=WEBHOOK_URL)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Ctrl+C で停止します")
    finally:
        broker.disconnect()


# ---------------------------------------------------------------------------
# 自己診断（MT5 無しで純粋ロジックを確認）
# ---------------------------------------------------------------------------
def selftest() -> int:
    """合成データで一目均衡表とトリガー判定の基本動作を確認する。"""
    n = 120
    t = pd.date_range("2026-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(0)
    # D1: 単調上昇 → 雲も上に持ち上がり、終値が雲上限を上回るはず
    d1 = pd.DataFrame({
        "time": t, "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
    })
    close = 100.0 + np.cumsum(rng.normal(0.5, 0.2, n))
    d1["close"] = close
    d1["open"] = close - rng.normal(0, 0.1, n)
    d1["high"] = np.maximum(d1["open"], d1["close"]) + 0.3
    d1["low"] = np.minimum(d1["open"], d1["close"]) - 0.3
    _, _, ctop, cbot = ichimoku_series(d1)
    assert float(d1["close"].iloc[-1]) > float(ctop.iloc[-1]), "上昇トレンドで終値が雲上限より上になるはず"
    assert float(cbot.iloc[-1]) < float(ctop.iloc[-1]), "雲下限は雲上限より下"

    # M15: 基準線を下→上にクロスするデータを作り、cross_up を検出できることを確認
    m15 = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
    })
    base = np.linspace(99.0, 101.0, n)
    m15["close"] = base
    m15["open"] = base - 0.2
    m15["high"] = base + 0.5
    m15["low"] = base - 0.5
    tenkan, kijun, _, _ = ichimoku_series(m15)
    # 終値を基準線を跨ぐように直近を強く上へ持ち上げる
    k_last = float(kijun.iloc[-2])
    m15.loc[m15.index[-2], "close"] = k_last - 0.5
    m15.loc[m15.index[-1], "close"] = k_last + 0.5
    m15.loc[m15.index[-1], "open"] = k_last - 0.1   # 陽線
    trig = detect_trigger(m15, tenkan, kijun, "LONG_ONLY")
    assert trig == "cross_up", f"cross_up になるはずが {trig}"

    sig = evaluate("EURUSD", d1, m15)
    assert sig is not None and sig.side == "LONG", "evaluate で LONG シグナルが出るはず"
    print(format_signal(sig))
    print("selftest OK: 雲判定・クロス検出・evaluate が正常に動作しています。")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    p = argparse.ArgumentParser(
        description="一目均衡表 MTF シグナル通知ボット（注文は出さず通知のみ）",
    )
    p.add_argument("--symbols", help="判定する銘柄（カンマ区切り）。省略時は config.json の symbols")
    p.add_argument("--interval", type=int, default=POLL_INTERVAL_SEC,
                   help=f"巡回間隔（秒、既定 {POLL_INTERVAL_SEC}）")
    p.add_argument("--once", action="store_true",
                   help="1回だけ判定して終了（cron / テスト用）")
    p.add_argument("--tf-env", default=ENV_TF, help=f"環境認識の時間足（既定 {ENV_TF}）")
    p.add_argument("--tf-trigger", default=TRIGGER_TF, help=f"トリガーの時間足（既定 {TRIGGER_TF}）")
    p.add_argument("--selftest", action="store_true", help="MT5 無しでロジック自己診断")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.symbols:
        args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())