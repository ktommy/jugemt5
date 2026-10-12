"""レンジ相場向け戦略（RSI 逆張り + ADX フィルタ）。

トレンド相場用の EMA クロス戦略（strategy.py）とは別ファイルで、
同じ Strategy インターフェースを実装している。
"""
import pandas as pd

from strategy import Strategy, Signal, compute_rsi, compute_adx


class RSIRangeStrategy(Strategy):
    """レンジ相場用の RSI 逆張り戦略。

    - ADX が閾値未満（レンジ）のときだけ発火する
    - RSI が売られすぎ(oversold)から反転上昇 → BUY
    - RSI が買われすぎ(overbought)から反転下降 → SELL
    - 決済は TP/SL と逆向きシグナルでの反転（main.py 側で処理）

    tp_atr_mult / sl_atr_mult は利食い・損切り幅（ATR 倍数、0 で無効）。
    """

    name = "range_rsi"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 2.0,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult

    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
        if len(df) < self.period + 2:
            return Signal.HOLD

        # ADX フィルタ: レンジ（ADX < 閾値）でなければ発火させない
        if compute_adx(df, self.adx_period).iloc[-1] >= self.adx_threshold:
            return Signal.HOLD

        rsi = compute_rsi(df["close"], self.period)
        prev, cur = rsi.iloc[-2], rsi.iloc[-1]

        if prev < self.oversold <= cur:
            return Signal.BUY
        if prev > self.overbought >= cur:
            return Signal.SELL
        return Signal.HOLD