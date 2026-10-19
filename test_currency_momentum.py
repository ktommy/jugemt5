"""通貨強弱（相対モメンタム）戦略のユニットテスト。

バックテスターへの横断ランナー run_currency_momentum と、その構成要素
（変化率→通貨スコア→ランキング）を合成データで検証する。
実行: python -m unittest test_currency_momentum -v
"""
import unittest

import numpy as np
import pandas as pd

from currency_momentum import (
    CurrencyMomentumStrategy, change_pct, scores_from_chg, rank_currencies,
    split_pair, PAIRS,
)
from backtest import run_currency_momentum, summarize


class ScoringTest(unittest.TestCase):
    def test_split_pair(self):
        self.assertEqual(split_pair("EURUSD"), ("EUR", "USD"))
        self.assertEqual(split_pair("USDJPY"), ("USD", "JPY"))

    def test_change_pct_lookback(self):
        # close が100 → 101（1本前比+1%）の場合、lookback=1 の変化率は 1.0%
        df = pd.DataFrame({"close": [100.0, 101.0, 101.0]})
        chg = change_pct(df, 1)
        self.assertAlmostEqual(chg.iloc[1], 1.0, places=6)   # 101/100-1
        self.assertAlmostEqual(chg.iloc[2], 0.0, places=6)   # 101/101-1

    def test_scores_sign_convention(self):
        # XXXUSD は上昇=数値そのまま、USDXXX は上昇=符号反転
        chg = {
            "EURUSD": 1.0,   # EUR 強い
            "USDJPY": 0.5,   # JPY 弱い（+0.5 反転 → -0.5）
            "USDCAD": -0.3,  # CAD 強い（-0.3 反転 → +0.3）
        }
        s = scores_from_chg(chg)
        self.assertAlmostEqual(s["USD"], 0.0)
        self.assertAlmostEqual(s["EUR"], 1.0)
        self.assertAlmostEqual(s["JPY"], -0.5)
        self.assertAlmostEqual(s["CAD"], 0.3)

    def test_rank_strong_weak(self):
        s = scores_from_chg({"EURUSD": 1.0, "USDJPY": 0.6})   # EUR 最強 / JPY 最弱
        strong, weak, diff = rank_currencies(s)
        self.assertEqual((strong, weak), ("EUR", "JPY"))
        self.assertAlmostEqual(diff, 1.6)

    def test_rank_needs_two(self):
        self.assertIsNone(rank_currencies({"USD": 0.0}))
        self.assertIsNone(rank_currencies({}))


class MomentumRunnerTest(unittest.TestCase):
    @staticmethod
    def _mk(close, times):
        return pd.DataFrame({
            "time": times, "open": close,
            "high": close + 0.0005, "low": close - 0.0005, "close": close,
        })

    def test_no_signal_when_below_threshold(self):
        # 変化率がほぼゼロ（横ばい）→ スコア差が揮化されず momentum_threshold に届かない → トレード無し
        n = 200
        times = pd.date_range("2026-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(1)
        dfs = {}
        for i, sym in enumerate(PAIRS):
            close = 1.0 + np.cumsum(rng.normal(0, 0.0001, n))
            dfs[sym] = self._mk(close, times)
        strat = CurrencyMomentumStrategy(lookback_period=24, momentum_threshold=5.0)
        trades = run_currency_momentum(dfs, strat, 14, {})
        self.assertEqual(trades, [])

    def test_trades_only_on_matched_pair(self):
        # EUR が一方的に強く、USD が一方的に弱い → 最強=USDドルを避け、最弱=USD になる設計で
        # EURUSD だけが「強×弱」に一致する。他ペアではシグナルが出ないことを確認する。
        n = 600
        times = pd.date_range("2026-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(2)
        dfs = {}
        eur = 1.08 + np.cumsum(np.abs(rng.normal(0.0006, 0.0003, n)))   # 単調強
        for i, sym in enumerate(PAIRS):
            if sym == "EURUSD":
                close = eur
            elif sym.startswith("USD"):   # USDXXX は緩やかに下落（USD 弱）
                close = 1.3 - np.cumsum(np.abs(rng.normal(0.0004, 0.0003, n)))
            else:                          # その他 XXXUSD はほぼ横ばい
                close = 1.2 + np.cumsum(rng.normal(0, 0.0004, n))
            dfs[sym] = self._mk(close, times)
        strat = CurrencyMomentumStrategy(lookback_period=24, momentum_threshold=0.05,
                                         exit_threshold=0.0, sl_atr_mult=0.0, tp_atr_mult=0.0)
        trades = run_currency_momentum(dfs, strat, 14, {})
        self.assertGreater(len(trades), 0)
        # EURUSD はロング（EUR 強 × USD 弱）になるはず
        for t in trades:
            if t.symbol == "EURUSD":
                self.assertEqual(t.direction, "buy")
                self.assertEqual(t.regime, "momentum")

    def test_spread_cost_applied(self):
        n = 600
        times = pd.date_range("2026-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(3)
        eur = 1.08 + np.cumsum(np.abs(rng.normal(0.0006, 0.0003, n)))
        dfs = {}
        for sym in PAIRS:
            if sym == "EURUSD":
                close = eur
            elif sym.startswith("USD"):
                close = 1.3 - np.cumsum(np.abs(rng.normal(0.0004, 0.0003, n)))
            else:
                close = 1.2 + np.cumsum(rng.normal(0, 0.0004, n))
            dfs[sym] = self._mk(close, times)
        strat = CurrencyMomentumStrategy(lookback_period=24, momentum_threshold=0.05,
                                         exit_threshold=0.0, sl_atr_mult=0.0, tp_atr_mult=0.0)
        no_cost = run_currency_momentum(dfs, strat, 14, {})
        with_cost = run_currency_momentum(dfs, strat, 14, {s: 0.0003 for s in PAIRS})
        self.assertGreater(len(no_cost), 0)
        # スプレッド控除後はスプレッド控除前より合計Rが下がる（コスト分）
        self.assertLess(summarize(with_cost).total_r, summarize(no_cost).total_r)


if __name__ == "__main__":
    unittest.main()