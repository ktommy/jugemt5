"""EMACrossStrategy のユニットテスト。

MT5 / Python 実行環境が無くても、合成データだけでシグナル判定ロジックを検証する。
実行: python -m unittest test_strategy -v
"""
import unittest

import numpy as np
import pandas as pd

from strategy import EMACrossStrategy, Signal


def make_df(close) -> pd.DataFrame:
    """close の配列から strategy.on_bar が期待する形の DataFrame を作る。"""
    return pd.DataFrame({"close": close})


def _ema(series: pd.Series, span: int) -> pd.Series:
    """本番 (strategy.py) と同じ指数平滑移動平均。"""
    return series.ewm(span=span, adjust=False).mean()


def first_counter_trend_cross(close, direction: str) -> int:
    """EMA20×EMA75 のクロスのうち、長期トレンドに逆らう最初の位置を返す。

    direction="buy" : ゴールデンクロス かつ 価格 < EMA200（下降中の買い戻り）
    direction="sell": デッドクロス かつ 価格 > EMA200（上昇中の押し目）
    """
    s = pd.Series(close, dtype=float)
    diff = (_ema(s, 20) - _ema(s, 75)).values
    slow = _ema(s, 200).values
    for i in range(1, len(close)):
        prev, cur = diff[i - 1], diff[i]
        if direction == "buy" and prev <= 0 < cur and s.iloc[i] < slow[i]:
            return i
        if direction == "sell" and prev >= 0 > cur and s.iloc[i] > slow[i]:
            return i
    raise AssertionError(f"反トレンドの{direction}クロスが見つかりません")


class EMACrossStrategyTest(unittest.TestCase):
    def setUp(self):
        # 既定値: fast=20 mid=75 slow=200 / トレンドフィルタ有効
        self.strat = EMACrossStrategy()

    def test_insufficient_history_returns_hold(self):
        # slow(=200)本では EMA200 算出に足りない → HOLD
        df = make_df(np.linspace(100, 90, self.strat.slow))
        self.assertEqual(self.strat.on_bar(df, 0), Signal.HOLD)

    def test_flat_series_returns_hold(self):
        # 横ばいではクロスが起きない → HOLD
        df = make_df(np.full(300, 100.0))
        self.assertEqual(self.strat.on_bar(df, 0), Signal.HOLD)

    def test_golden_cross_buys_in_uptrend(self):
        # 長く横ばい → 最後の1本で急騰: EMA20 が EMA75 を上抜け、価格は EMA200 超え
        close = np.full(500, 100.0)
        close[-1] = 120.0
        self.assertEqual(self.strat.on_bar(make_df(close), 0), Signal.BUY)

    def test_dead_cross_sells_in_downtrend(self):
        # 長く横ばい → 最後の1本で急落: EMA20 が EMA75 を下抜け、価格は EMA200 割れ
        close = np.full(500, 100.0)
        close[-1] = 80.0
        self.assertEqual(self.strat.on_bar(make_df(close), 0), Signal.SELL)

    def test_trend_filter_blocks_counter_trend_buy(self):
        # 強い下降から反転: ゴールデンクロスが出ても価格が EMA200 未満なら見送る
        close = np.concatenate([
            np.linspace(130, 100, 300),   # 下降
            np.linspace(100, 108, 150),   # 反転
        ])
        i = first_counter_trend_cross(close, "buy")
        df = make_df(close[: i + 1])

        # フィルタ有効: 逆トレンドの買いクロスは無視して HOLD
        self.assertEqual(self.strat.on_bar(df, 0), Signal.HOLD)
        # フィルタ無効: 素直に BUY
        self.assertEqual(
            EMACrossStrategy(trend_filter=False).on_bar(df, 0), Signal.BUY
        )

    def test_trend_filter_blocks_counter_trend_sell(self):
        # 強い上昇から反転: デッドクロスが出ても価格が EMA200 超えなら見送る
        close = np.concatenate([
            np.linspace(100, 130, 300),   # 上昇
            np.linspace(130, 112, 150),   # 反転
        ])
        i = first_counter_trend_cross(close, "sell")
        df = make_df(close[: i + 1])

        self.assertEqual(self.strat.on_bar(df, 0), Signal.HOLD)
        self.assertEqual(
            EMACrossStrategy(trend_filter=False).on_bar(df, 0), Signal.SELL
        )

    def test_custom_periods(self):
        # 期間を変えてもクロス判定が正しく機能する
        s = EMACrossStrategy(fast=10, mid=40, slow=100)
        close = np.full(200, 100.0)
        close[-1] = 115.0
        self.assertEqual(s.on_bar(make_df(close), 0), Signal.BUY)


if __name__ == "__main__":
    unittest.main()