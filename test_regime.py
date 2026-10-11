"""レジーム判定（ADX）とレンジ戦略（RSIRangeStrategy）のユニットテスト。

MT5 への実接続なしで、合成データからロジックを検証する。
実行: python -m unittest test_regime -v
"""
import unittest

import numpy as np
import pandas as pd

from strategy import Signal, compute_adx
from regime import Regime, detect_regime
from range_strategy import RSIRangeStrategy


def market(close, spread: float = 0.2) -> pd.DataFrame:
    """close の配列から high / low / close の DataFrame を作る。"""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"high": close + spread, "low": close - spread, "close": close})


def first_non_hold(strategy, close):
    """最初に HOLD 以外を返す (シグナル, バー番号) を返す。無ければ (HOLD, None)。"""
    df = market(close)
    for i in range(strategy.period + 2, len(close)):
        sig = strategy.on_bar(df.iloc[: i + 1], 0)
        if sig != Signal.HOLD:
            return sig, i
    return Signal.HOLD, None


class AdxTest(unittest.TestCase):
    def test_adx_high_in_trend_low_in_flat(self):
        # 単調トレンドでは ADX が高く、横ばいではほぼ 0
        trend = compute_adx(market(np.linspace(100, 130, 300)), 14).iloc[-1]
        flat = compute_adx(market(np.full(300, 100.0)), 14).iloc[-1]
        self.assertGreater(trend, 50)
        self.assertLess(flat, 1)

    def test_adx_within_bounds(self):
        adx = compute_adx(market(np.linspace(100, 130, 300)), 14)
        self.assertTrue(((adx >= 0) & (adx <= 100)).all())


class RegimeTest(unittest.TestCase):
    def test_trend_regime(self):
        self.assertEqual(detect_regime(market(np.linspace(100, 130, 300))), Regime.TREND)
        self.assertEqual(detect_regime(market(np.linspace(130, 100, 300))), Regime.TREND)

    def test_range_regime(self):
        # 横ばい（ADX=0）はレンジ
        self.assertEqual(detect_regime(market(np.full(300, 100.0))), Regime.RANGE)

    def test_insufficient_history_defaults_to_trend(self):
        # 履歴不足時は安全側の TREND
        self.assertEqual(detect_regime(market(np.linspace(100, 105, 10))), Regime.TREND)


class RSIRangeStrategyTest(unittest.TestCase):
    def test_insufficient_history_returns_hold(self):
        s = RSIRangeStrategy()
        self.assertEqual(s.on_bar(market(np.linspace(100, 90, 10))), Signal.HOLD)

    def test_rsi_reversal_buys_in_range(self):
        # ADX フィルタを緩め、RSI の逆張りロジックのみ検証する
        s = RSIRangeStrategy(adx_threshold=100.0)
        # 下落 → 戻し: RSI が 30 を下から上に抜けて BUY
        close = np.concatenate([np.linspace(100, 90, 30), np.linspace(90, 100, 30)])
        sig, _ = first_non_hold(s, close)
        self.assertEqual(sig, Signal.BUY)

    def test_rsi_reversal_sells_in_range(self):
        s = RSIRangeStrategy(adx_threshold=100.0)
        # 上昇 → 押し: RSI が 70 を上から下に抜けて SELL
        close = np.concatenate([np.linspace(100, 110, 30), np.linspace(110, 100, 30)])
        sig, _ = first_non_hold(s, close)
        self.assertEqual(sig, Signal.SELL)

    def test_adx_filter_blocks_signal_in_trend(self):
        # 強い上昇→下降では RSI が 70 を割り SELL 相当になるが、ADX が高くフィルタで HOLD
        close = np.concatenate([np.linspace(100, 130, 250), np.linspace(130, 115, 60)])
        df = market(close)

        loose = RSIRangeStrategy(adx_threshold=100.0)  # フィルタ無効相当
        sig, bar = first_non_hold(loose, close)
        self.assertEqual(sig, Signal.SELL, "フィルタ無効時に SELL が出るはず")

        strict = RSIRangeStrategy()
        self.assertEqual(strict.on_bar(df.iloc[: bar + 1], 0), Signal.HOLD)


if __name__ == "__main__":
    unittest.main()