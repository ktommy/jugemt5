"""BoxRangeStrategy（ボックスレンジ逆張り）のユニットテスト。

MT5 への実接続なしで、合成データからシグナル判定と決済価格を検証する。
実行: python -m unittest test_box_strategy -v
"""
import unittest

import numpy as np
import pandas as pd

from strategy import Signal, compute_atr
from box_strategy import BoxRangeStrategy, box_levels, bb_width


def make_df(ohlc) -> pd.DataFrame:
    """(open, high, low, close) の列から on_bar が期待する DataFrame を作る。"""
    df = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"])
    df["time"] = pd.date_range("2026-01-01", periods=len(df), freq="5min")
    return df


def flat(n: int = 300) -> list:
    """ATR(14)=1.0・ADX≈0・BB幅≈0 になる横ばい系列。box_low=99.5 / box_high=100.5。"""
    return [(100.0, 100.5, 99.5, 100.0)] * n


class BoxLevelsTest(unittest.TestCase):
    def test_shift_excludes_current_bar(self):
        # box_high/box_low は「1本前まで」の window 本から計算され、現在足を含まない
        df = make_df(flat(25))
        df.loc[24, ["high", "low"]] = [200.0, 50.0]  # 現在足を極端にしても…
        high, low, mid = box_levels(df, 20)
        self.assertEqual(high.iloc[-1], 100.5)   # 過去20本の最高値のまま
        self.assertEqual(low.iloc[-1], 99.5)     # 現在足の安値 50 を含まない
        self.assertEqual(mid.iloc[-1], 100.0)


class BoxRangeStrategyTest(unittest.TestCase):
    def setUp(self):
        self.strat = BoxRangeStrategy()

    def test_insufficient_history_returns_hold(self):
        self.assertEqual(self.strat.on_bar(make_df(flat(50)), 0), Signal.HOLD)

    def test_buy_on_low_touch_in_range(self):
        # 横ばい（レンジ）で最後の1本の安値が box_low(=99.5) を下抜け → BUY
        ohlc = flat() + [(99.8, 99.8, 99.0, 99.5)]
        self.assertEqual(self.strat.on_bar(make_df(ohlc), 0), Signal.BUY)

    def test_sell_on_high_touch_in_range(self):
        # 横ばい（レンジ）で最後の1本の高値が box_high(=100.5) を上抜け → SELL
        ohlc = flat() + [(100.2, 101.0, 100.2, 100.5)]
        self.assertEqual(self.strat.on_bar(make_df(ohlc), 0), Signal.SELL)

    def test_hold_in_strong_trend(self):
        # 強い上昇トレンド（ADX 高）では安値を下抜けてもトレンド回避フィルタで HOLD
        close = np.linspace(100.0, 130.0, 300)
        ohlc = [(c, c + 0.1, c - 0.1, c) for c in close]
        ohlc.append((125.0, 125.0, 124.0, 125.0))  # box_low を下抜ける1本
        self.assertEqual(self.strat.on_bar(make_df(ohlc), 0), Signal.HOLD)

    def test_bb_filter_blocks_when_wide(self):
        # 方向性なしだが振幅が大きい（BB 幅が広い）系列。ADX は無効化（閾値 1000）して
        # BB フィルタだけを検証する。狭い閾値なら HOLD、広い閾値なら BUY になる。
        base = np.tile([97.0, 103.0], 200)
        close = np.concatenate([base, [96.5]])  # 最後の1本が box_low を下抜け
        df = make_df([(c, c + 0.1, c - 0.1, c) for c in close])
        blocked = BoxRangeStrategy(adx_threshold=1000.0, bb_width_threshold=0.0)
        allow = BoxRangeStrategy(adx_threshold=1000.0, bb_width_threshold=100.0)
        self.assertEqual(blocked.on_bar(df, 0), Signal.HOLD)
        self.assertEqual(allow.on_bar(df, 0), Signal.BUY)

    def test_exit_plan_values(self):
        # 買い: entry=box_low / sl=box_low−buffer×ATR / tp=box_mid / denom=buffer×ATR
        df = make_df(flat())
        atr = float(compute_atr(df, 14).iloc[-1])  # 横ばいで ATR=1.0
        entry, sl, tp, denom = self.strat.exit_plan(df, 1, atr)
        self.assertAlmostEqual(entry, 99.5)
        self.assertAlmostEqual(sl, 99.0)
        self.assertAlmostEqual(tp, 100.0)
        self.assertAlmostEqual(denom, 0.5)
        # 売り: entry=box_high / sl=box_high+buffer×ATR / tp=box_mid
        entry, sl, tp, denom = self.strat.exit_plan(df, -1, atr)
        self.assertAlmostEqual(entry, 100.5)
        self.assertAlmostEqual(sl, 101.0)
        self.assertAlmostEqual(tp, 100.0)


class BbWidthTest(unittest.TestCase):
    def test_bb_width_narrow_for_flat_wide_for_volatile(self):
        self.assertAlmostEqual(bb_width(make_df(flat(200)), 20).iloc[-1], 0.0, places=3)
        volatile = make_df([(c, c + 0.1, c - 0.1, c)
                            for c in np.tile([97.0, 103.0], 100)])
        self.assertGreater(bb_width(volatile, 20).iloc[-1], 0.05)


if __name__ == "__main__":
    unittest.main()
