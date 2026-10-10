"""戦略インターフェースとサンプル戦略。

新しい戦略を追加する場合は Strategy を継承し on_bar を実装するだけでよい。
main.py で strategy = MyStrategy() のように差し替える。
"""
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(Enum):
    HOLD = "hold"    # 何もしない
    BUY = "buy"      # 買い
    SELL = "sell"    # 売り
    CLOSE = "close"  # 保有ポジションを決済


class Strategy(ABC):
    """全戦略が実装すべきインターフェース。

    on_bar には「最新確定足まで」のローソク足 DataFrame が渡される。
    列: time(秒エポック), open, high, low, close, tick_volume, spread, real_volume
    """

    name: str = "base"

    @abstractmethod
    def on_bar(self, df: pd.DataFrame) -> Signal:
        """最新の確定足データから次のアクションを返す。"""
        raise NotImplementedError


class MACrossStrategy(Strategy):
    """移動平均クロス戦略（サンプル実装）。

    短期MA が長期MA を下→上に抜けたら BUY、上→下に抜けたら SELL。
    参考実装なので、実際の戦略はここを別途実装して差し替えること。
    """

    name = "ma_cross"

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    def on_bar(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.slow + 2:
            return Signal.HOLD

        close = df["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()

        # 最新の2本の確定足でクロスを判定
        prev_diff = fast_ma.iloc[-2] - slow_ma.iloc[-2]
        cur_diff = fast_ma.iloc[-1] - slow_ma.iloc[-1]

        if prev_diff <= 0 < cur_diff:
            return Signal.BUY
        if prev_diff >= 0 > cur_diff:
            return Signal.SELL
        return Signal.HOLD