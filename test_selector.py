"""銘柄選別（selector.py）のユニットテスト。

MT5 への実接続なしで、合成データから「シグナル発火ベースの選別」を検証する。
実行: python -m unittest test_selector -v
"""
import unittest

import numpy as np
import pandas as pd

from strategy import Strategy, Signal, EMACrossStrategy
from range_strategy import RSIRangeStrategy
from regime import Regime
from selector import scan_signals


def market(close, spread: float = 0.2) -> pd.DataFrame:
    """close の配列から high / low / close を持つ DataFrame を作る。"""
    close = np.asarray(close, dtype=float)
    n = len(close)
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "open": close,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "tick_volume": 1,
        "spread": spread,
        "real_volume": 1,
    })


class FakeStrategy(Strategy):
    """on_bar が常に同じシグナルを返すフェイク。

    実戦略のクロス判定は test_strategy.py / test_regime.py が検証済みなので、
    ここでは「選別（レジーム振り分け・HOLD除去・スコア順）」に絞って検証する。
    """

    def __init__(self, sig: Signal):
        self._sig = sig

    def on_bar(self, df, pos_dir: int = 0) -> Signal:
        return self._sig


class ScanSignalsTest(unittest.TestCase):
    def setUp(self):
        self.trend = EMACrossStrategy()
        self.range = RSIRangeStrategy()

    def scan(self, rate_map, trend, range_):
        return scan_signals(
            list(rate_map.keys()), trend, range_,
            lambda s: rate_map.get(s), 14, 25.0,
        )

    def test_picks_firing_symbol_and_skips_others(self):
        # トレンド戦略が BUY、レンジ戦略が HOLD を返す世界を想定
        rate_map = {
            "flat": market(np.full(300, 100.0)),          # RANGE → HOLD
            "rise": market(np.linspace(100.0, 130.0, 300)),  # TREND → BUY
        }
        results = self.scan(rate_map, FakeStrategy(Signal.BUY), FakeStrategy(Signal.HOLD))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "rise")
        self.assertEqual(results[0].signal, Signal.BUY)
        self.assertEqual(results[0].regime, Regime.TREND)

    def test_routes_range_signal_to_range_strategy(self):
        rate_map = {"flat": market(np.full(300, 100.0))}
        results = self.scan(rate_map, FakeStrategy(Signal.HOLD), FakeStrategy(Signal.SELL))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].signal, Signal.SELL)
        self.assertEqual(results[0].regime, Regime.RANGE)

    def test_empty_when_none_fire(self):
        rate_map = {
            "A": market(np.full(300, 100.0)),
            "B": market(np.full(300, 100.0)),
        }
        results = self.scan(rate_map, FakeStrategy(Signal.HOLD), FakeStrategy(Signal.HOLD))
        self.assertEqual(results, [])

    def test_trend_setup_ranks_above_range(self):
        # トレンド候補（score=ADX 高）とレンジ候補（score=閾値-ADX）が並んだ場合、
        # スコア降順で並ぶこと（＝選別対象が複数いたとき上位を取る検証）。
        rate_map = {
            "range_flat": market(np.full(300, 100.0)),                # RANGE → SELL
            "trend_rise": market(np.linspace(100.0, 130.0, 300)),     # TREND → BUY
        }
        results = self.scan(rate_map, FakeStrategy(Signal.BUY), FakeStrategy(Signal.SELL))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].symbol, "trend_rise")
        self.assertEqual(results[0].regime, Regime.TREND)
        self.assertEqual(results[1].symbol, "range_flat")
        self.assertEqual(results[1].regime, Regime.RANGE)
        self.assertGreater(results[0].score, results[1].score)

    def test_real_strategies_fire_through_scan(self):
        # 実戦略（EMA クロス）で、末尾確定足にゴールデンクロスする系列が選ばれること
        rate_map = {
            "flat": market(np.full(400, 100.0)),
            "buy": market(_trend_reentry_close()),
        }
        results = scan_signals(
            list(rate_map.keys()), self.trend, self.range,
            lambda s: rate_map.get(s), 14, 25.0,
        )
        self.assertTrue(any(
            r.symbol == "buy" and r.signal == Signal.BUY and r.regime == Regime.TREND
            for r in results
        ))


def _trend_reentry_close() -> np.ndarray:
    """強い上昇 → 押し目 → 戻しで、最後の確定足に「EMA200 越えのゴールデンクロス」が
    出る close 系列を返す。末尾に形成中足 1 本分を足してある（scan が iloc[:-1] で
    落とす前提）。
    """
    close = np.concatenate([
        np.linspace(100.0, 130.0, 300),   # 上昇（トレンド確立: ADX 高）
        np.linspace(130.0, 118.0, 120),   # 押し目（EMA20 が EMA75 を下抜け）
        np.linspace(118.0, 126.0, 60),    # 戻し（再度ゴールデンクロス）
    ])
    s = pd.Series(close, dtype=float)
    e20 = s.ewm(span=20, adjust=False).mean()
    e75 = s.ewm(span=75, adjust=False).mean()
    e200 = s.ewm(span=200, adjust=False).mean()
    diff = (e20 - e75).values
    # 末尾から逆順に「EMA200 越えのゴールデンクロス」を探す。
    # 上昇初期にもクロスがあるため、後ろから探して押し目→戻しの再クロスを拾う。
    for i in range(len(close) - 1, 1, -1):
        if diff[i - 1] <= 0 < diff[i] and s.iloc[i] > e200.iloc[i]:
            return np.append(close[: i + 1], close[i])  # 末尾に形成中足 1 本
    raise AssertionError("トレンド継続の BUY クロスを作れませんでした")


if __name__ == "__main__":
    unittest.main()