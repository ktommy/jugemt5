"""戦略インターフェースとサンプル戦略。

新しい戦略を追加する場合は Strategy を継承し on_bar を実装するだけでよい。
main.py で strategy = MyStrategy() のように差し替える。

on_bar の第2引数 pos_dir は現在のポジション方向（+1=買い / -1=売り / 0=無し）。
方向を必要とする戦略（逆張りなど）はこれで判断できる。
"""
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(Enum):
    HOLD = "hold"    # 何もしない
    BUY = "buy"      # 買い
    SELL = "sell"    # 売り
    CLOSE = "close"  # 保有ポジションを決済


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder の RSI を計算する（平均は指数平滑で近似、期初期は 50 で補完）。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


class Strategy(ABC):
    """全戦略が実装すべきインターフェース。

    on_bar には「最新確定足まで」のローソク足 DataFrame と、
    現在のポジション方向 pos_dir が渡される。
    df の列: time(秒エポック), open, high, low, close, tick_volume, spread, real_volume
    pos_dir: +1=買い / -1=売り / 0=ポジション無し。不要なら無視してよい。
    """

    name: str = "base"

    @abstractmethod
    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
        """最新の確定足データと現在ポジションから次のアクションを返す。"""
        raise NotImplementedError


class MACrossStrategy(Strategy):
    """移動平均クロス戦略（サンプル実装）。

    短期MA が長期MA を下→上に抜けたら BUY、上→下に抜けたら SELL。
    """

    name = "ma_cross"

    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow

    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
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


class RSIReversalStrategy(Strategy):
    """RSI 逆張り戦略（TP/SL 付き）。

    - RSI が売られすぎ（oversold）から反転上昇 → BUY
    - RSI が買われすぎ（overbought）から反転下降 → SELL
    - 保有中に RSI が中立水準（exit_level）へ戻る → CLOSE

    tp_pct / sl_pct はエントリー価格に対する利食い・損切り幅（比率）。
    0 を指定するとその側の TP/SL を無効にする（main.py が参照する）。
    """

    name = "rsi_reversal"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
        tp_pct: float = 0.005,   # +0.5% で利食い
        sl_pct: float = 0.003,   # -0.3% で損切り
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct

    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
        if len(df) < self.period + 2:
            return Signal.HOLD

        rsi = compute_rsi(df["close"], self.period)
        prev, cur = rsi.iloc[-2], rsi.iloc[-1]

        # 決済: ポジション方向に応じて中立水準への回帰でクローズ
        if pos_dir == 1 and prev <= self.exit_level < cur:
            return Signal.CLOSE
        if pos_dir == -1 and prev >= self.exit_level > cur:
            return Signal.CLOSE

        # エントリー: 売られすぎ/買われすぎからの反転
        if prev < self.oversold <= cur:
            return Signal.BUY
        if prev > self.overbought >= cur:
            return Signal.SELL

        return Signal.HOLD


class EMACrossStrategy(Strategy):
    """EMA 20/75/200 トレンドフォロー戦略。

    - EMA20 が EMA75 を下→上に抜けたら BUY、上→下に抜けたら SELL
    - EMA200（価格との位置）で長期トレンドを判定し、トレンドに逆らうクロスを無視
    - 決済は逆シグナルでの反転（main.py 側で処理）＋ TP/SL 併用

    tp_pct / sl_pct はエントリー価格に対する利食い・損切り幅（比率）。
    0 を指定するとその側の TP/SL を無効にする（main.py が参照する）。
    trend_filter=False にすると EMA200 フィルタなしの単純クロスになる。
    """

    name = "ema_cross"

    def __init__(
        self,
        fast: int = 20,
        mid: int = 75,
        slow: int = 200,
        tp_pct: float = 0.005,   # +0.5% で利食い
        sl_pct: float = 0.003,   # -0.3% で損切り
        trend_filter: bool = True,
    ):
        self.fast = fast
        self.mid = mid
        self.slow = slow
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trend_filter = trend_filter
        # EMA(slow) の収束に十分な履歴本数。main.py が参照して取得本数を決める。
        self.min_history = slow + 100

    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
        if len(df) < self.slow + 2:
            return Signal.HOLD

        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_mid = close.ewm(span=self.mid, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        # 最新の2本の確定足でクロスを判定
        prev_diff = ema_fast.iloc[-2] - ema_mid.iloc[-2]
        cur_diff = ema_fast.iloc[-1] - ema_mid.iloc[-1]

        # 長期トレンド: 価格が EMA200 の上か下か
        price = close.iloc[-1]
        uptrend = price > ema_slow.iloc[-1]
        downtrend = price < ema_slow.iloc[-1]

        if prev_diff <= 0 < cur_diff:                    # ゴールデンクロス
            if not self.trend_filter or uptrend:
                return Signal.BUY
        if prev_diff >= 0 > cur_diff:                    # デッドクロス
            if not self.trend_filter or downtrend:
                return Signal.SELL

        return Signal.HOLD