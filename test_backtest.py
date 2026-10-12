"""バックテスターエンジン（backtest.py）のユニットテスト。

MT5 への実接続なしで、合成データを使い仮想売買の再現ロジックを検証する。
実行: python -m unittest test_backtest -v
"""
import unittest

import numpy as np
import pandas as pd

from strategy import Strategy, Signal, EMACrossStrategy
from range_strategy import RSIRangeStrategy
from regime import Regime
from backtest import (
    run, summarize, Trade, _Pos, _check_stops,
    _Indicators, _ema_signal, _rsi_range_signal,
)


def make_df(ohlc) -> pd.DataFrame:
    """(open, high, low, close) の列から backtest.run が期待する DataFrame を作る。"""
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
    df["time"] = pd.date_range("2026-01-01", periods=len(df), freq="5min")
    return df


def flat(n: int = 40) -> list:
    """ATR(14)=1.0 になる横ばい系列（close=100, high=100.5, low=99.5）。"""
    return [(100.0, 100.5, 99.5, 100.0)] * n


class HoldStrategy(Strategy):
    name = "hold"
    sl_atr_mult = 1.0
    tp_atr_mult = 1.0

    def on_bar(self, df, pos_dir: int = 0) -> Signal:
        return Signal.HOLD


class FireOnce(Strategy):
    """最初の1回だけ sig を返し、以降は HOLD（1トレードを確実に作るためのフェイク）。"""
    name = "fire_once"
    sl_atr_mult = 1.0
    tp_atr_mult = 1.0

    def __init__(self, sig: Signal = Signal.BUY):
        self.sig = sig
        self._fired = False

    def on_bar(self, df, pos_dir: int = 0) -> Signal:
        if not self._fired:
            self._fired = True
            return self.sig
        return Signal.HOLD


class Doten(Strategy):
    """フラットで BUY、ロング保有で SELL（ドテン）、ショートで HOLD。"""
    name = "doten"
    sl_atr_mult = 1.0
    tp_atr_mult = 1.0

    def on_bar(self, df, pos_dir: int = 0) -> Signal:
        if pos_dir == 0:
            return Signal.BUY
        if pos_dir == 1:
            return Signal.SELL
        return Signal.HOLD


class RunTest(unittest.TestCase):
    def _run(self, df, trend, range_, **kw):
        return run(df, "TEST", trend, range_, 14, 25.0, 14, **kw)

    def test_no_trades_when_hold(self):
        trades = self._run(make_df(flat()), HoldStrategy(), HoldStrategy())
        self.assertEqual(trades, [])

    def test_take_profit(self):
        # 横ばいでエントリー(SL=99,TP=101)した直後に上昇して TP 到達 → +1.0R
        ohlc = flat()
        ohlc.append((100.5, 102.0, 100.0, 100.5))
        trades = self._run(make_df(ohlc), HoldStrategy(), FireOnce(Signal.BUY))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.exit_reason, "TP")
        self.assertAlmostEqual(t.r, 1.0, places=3)

    def test_stop_loss(self):
        # TP に届かず SL へ到達 → -1.0R
        ohlc = flat()
        ohlc.append((100.0, 100.0, 98.0, 99.0))
        trades = self._run(make_df(ohlc), HoldStrategy(), FireOnce(Signal.BUY))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.exit_reason, "SL")
        self.assertAlmostEqual(t.r, -1.0, places=3)

    def test_reversal_doten(self):
        # フラット→BUY、ロング保有で逆向きシグナル→ドテン（BUYで決済、SELLで建て直し）
        trades = self._run(make_df(flat()), Doten(), Doten())
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].direction, "buy")
        self.assertEqual(trades[0].exit_reason, "REVERSAL")
        self.assertEqual(trades[1].direction, "sell")
        self.assertEqual(trades[1].exit_reason, "EOD")

    def test_spread_cost_deducted(self):
        # 素の +1.0R（TP）も、往復スプレッド 0.1（denom=1.0 なので 0.1R）を引くと +0.9R になる
        ohlc = flat()
        ohlc.append((100.5, 102.0, 100.0, 100.5))
        trades = self._run(make_df(ohlc), HoldStrategy(), FireOnce(Signal.BUY),
                           spread_price=0.1)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertAlmostEqual(t.cost_r, 0.1, places=3)
        self.assertAlmostEqual(t.r, 0.9, places=3)

    def test_force_regime(self):
        # 横ばい系列（ADX≈0 → 通常は RANGE 判定）でも、force_regime="trend" なら
        # トレンド戦略（FireOnce BUY）で建て、エントリー時のレジームが trend になる
        ohlc = flat()
        ohlc.append((100.5, 102.0, 100.0, 100.5))
        trades = self._run(make_df(ohlc), FireOnce(Signal.BUY), HoldStrategy(),
                           force_regime="trend")
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t.regime, "trend")
        self.assertEqual(t.strategy, "fire_once")
        self.assertEqual(t.exit_reason, "TP")


class StopsTest(unittest.TestCase):
    def test_same_bar_sl_priority(self):
        # 1本の足で SL と TP が同時に到達した場合は保守的に SL を優先する
        pos = _Pos(
            direction=1, entry=100.0, sl=99.0, tp=101.0, denom=1.0,
            entry_time=pd.Timestamp("2026-01-01"), entry_index=0,
            regime=Regime.RANGE, strategy_name="test",
        )
        bar = pd.Series({"open": 100.0, "high": 101.5, "low": 98.5, "close": 100.0})
        px, reason = _check_stops(pos, bar)
        self.assertEqual(reason, "SL")
        self.assertEqual(px, 99.0)

    def test_gap_fills_at_open(self):
        # SL を窓開けで飛ばした場合は open で約定扱い
        pos = _Pos(
            direction=1, entry=100.0, sl=99.0, tp=101.0, denom=1.0,
            entry_time=pd.Timestamp("2026-01-01"), entry_index=0,
            regime=Regime.RANGE, strategy_name="test",
        )
        bar = pd.Series({"open": 98.0, "high": 99.5, "low": 97.0, "close": 98.5})
        px, reason = _check_stops(pos, bar)
        self.assertEqual(reason, "SL")
        self.assertEqual(px, 98.0)


class VectorizedTest(unittest.TestCase):
    """ベクトル計算パス（高速化）が戦略本体の on_bar と完全一致することを担保する。

    _ema_signal / _rsi_range_signal は on_bar の判定を前計算済み指標で再現したもの。
    将来 strategy.py / range_strategy.py の判定条件を変えたとき、こちらを追随し忘れると
    バックテストだけ現実とズレるため、同一データで両者を突き合わせる。
    """

    def _strategies(self):
        trend = EMACrossStrategy(fast=20, mid=75, slow=200,
                                 tp_atr_mult=3.0, sl_atr_mult=2.0, trend_filter=True)
        range_ = RSIRangeStrategy(period=14, oversold=30.0, overbought=70.0,
                                  adx_period=14, adx_threshold=25.0,
                                  tp_atr_mult=2.0, sl_atr_mult=2.0)
        return trend, range_

    def test_signals_match_on_bar(self):
        np.random.seed(7)
        n = 1200
        close = 100.0 + np.cumsum(np.random.normal(0, 0.3, n))
        high = close + np.abs(np.random.normal(0, 0.2, n))
        low = close - np.abs(np.random.normal(0, 0.2, n))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=n, freq="5min"),
            "open": open_, "high": high, "low": low, "close": close,
        })
        trend, range_ = self._strategies()
        ind = _Indicators(df, trend, range_, 14, 14)
        warmup = max(trend.min_history, 14, 14) + 2
        for i in range(warmup, n):
            ref = df.iloc[: i + 1]
            self.assertEqual(_ema_signal(ind, i, trend), trend.on_bar(ref, 0), f"EMA @{i}")
            self.assertEqual(_rsi_range_signal(ind, i, range_), range_.on_bar(ref, 0), f"RSI @{i}")


class SummarizeTest(unittest.TestCase):
    def _trade(self, r: float, reason: str = "TP", regime: str = "trend") -> Trade:
        return Trade(
            symbol="X", direction="buy", strategy="s", regime=regime,
            entry_time=pd.Timestamp("2026-01-01"), exit_time=pd.Timestamp("2026-01-02"),
            entry=0.0, exit=0.0, sl_distance=1.0, r=r,
            exit_reason=reason, bars=1,
        )

    def test_summary_metrics(self):
        s = summarize([self._trade(1.0), self._trade(-1.0, "SL"), self._trade(1.5)])
        self.assertEqual(s.n, 3)
        self.assertAlmostEqual(s.win_rate, 2 / 3)
        self.assertAlmostEqual(s.total_r, 1.5)
        self.assertAlmostEqual(s.avg_r, 0.5)
        self.assertAlmostEqual(s.pf, 2.5)
        self.assertEqual(s.by_reason, {"TP": 2, "SL": 1})

    def test_by_regime_breakdown(self):
        s = summarize([
            self._trade(1.0, "TP", "trend"),
            self._trade(-1.0, "SL", "trend"),
            self._trade(1.0, "TP", "range"),
        ])
        self.assertEqual(set(s.by_regime_stat), {"trend", "range"})
        trend = s.by_regime_stat["trend"]
        self.assertEqual(trend["n"], 2)
        self.assertAlmostEqual(trend["win_rate"], 0.5)
        self.assertAlmostEqual(trend["avg_r"], 0.0)
        self.assertAlmostEqual(trend["pf"], 1.0)
        range_ = s.by_regime_stat["range"]
        self.assertEqual(range_["n"], 1)
        self.assertAlmostEqual(range_["win_rate"], 1.0)
        self.assertAlmostEqual(range_["avg_r"], 1.0)


if __name__ == "__main__":
    unittest.main()