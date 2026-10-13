"""ボックスレンジ逆張り戦略（ドンチャンチャネルベース）。

相場がレンジ（ボックス）のとき、その上限/下限へ指値を置いて逆張りする。
トレンド回避は「ADX 低 かつ ボリンジャーバンド幅が狭い」の AND で行う。
TP=中央線、SL=バンド端±バッファ（ATR 倍数）で、エントリー時に固定する。

仕様は SPEC_BOX_STRATEGY.md を参照。
"""
import pandas as pd

from strategy import Strategy, Signal, compute_adx


def box_levels(df: pd.DataFrame, window: int):
    """1本前までの足によるドンチャンチャネル (box_high, box_low, box_mid)。

    見通しバイアスを排除するため .shift(1) を使う。戻り値は df と同じ長さの Series。
    """
    box_high = df["high"].rolling(window).max().shift(1)
    box_low = df["low"].rolling(window).min().shift(1)
    box_mid = (box_high + box_low) / 2.0
    return box_high, box_low, box_mid


def bb_width(df: pd.DataFrame, period: int, num_std: float = 2.0) -> pd.Series:
    """ボリンジャーバンド幅 (Upper - Lower) / SMA。レンジ判定（狭いか）の補助指標。"""
    close = df["close"]
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return (2.0 * num_std * std) / sma


class BoxRangeStrategy(Strategy):
    """ドンチャンチャネル逆張り。

    - レンジ判定（AND）: ADX < adx_threshold かつ BB 幅 < bb_width_threshold
    - 最新確定足の安値が box_low をタッチ → BUY、高値が box_high をタッチ → SELL
    - 決済価格は exit_plan（entry=バンド / tp=mid / sl=バンド±buffer×ATR）で固定
    """

    name = "box_range"

    def __init__(
        self,
        window: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        use_bb_filter: bool = True,
        bb_period: int = 20,
        bb_width_threshold: float = 0.05,
        sl_buffer_atr: float = 0.5,
        min_mid_atr: float = 0.0,
    ):
        self.window = window
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_bb_filter = use_bb_filter
        self.bb_period = bb_period
        self.bb_width_threshold = bb_width_threshold
        self.sl_buffer_atr = sl_buffer_atr
        self.min_mid_atr = min_mid_atr
        # 各指標の収束に十分な履歴本数。main.py / backtest.py が参照する。
        self.min_history = max(window, bb_period, adx_period) + 100

    def _is_range(self, df: pd.DataFrame) -> bool:
        """AND フィルタ: ADX 低 かつ（BB フィルタ無効 or BB 幅狭）ならレンジとみなす。"""
        if compute_adx(df, self.adx_period).iloc[-1] >= self.adx_threshold:
            return False
        if self.use_bb_filter:
            width = bb_width(df, self.bb_period).iloc[-1]
            if pd.isna(width) or width >= self.bb_width_threshold:
                return False
        return True

    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
        if len(df) < self.min_history:
            return Signal.HOLD
        if not self._is_range(df):
            return Signal.HOLD

        box_high, box_low, _ = box_levels(df, self.window)
        high = box_high.iloc[-1]
        low = box_low.iloc[-1]
        if pd.isna(high) or pd.isna(low):
            return Signal.HOLD

        if df["low"].iloc[-1] <= low:
            return Signal.BUY
        if df["high"].iloc[-1] >= high:
            return Signal.SELL
        return Signal.HOLD

    def exit_plan(self, df: pd.DataFrame, direction: int, atr: float):
        """新規建て時の (entry, sl, tp, denom)。バンド逆張りの価格構造。"""
        box_high, box_low, box_mid = box_levels(df, self.window)
        high = box_high.iloc[-1]
        low = box_low.iloc[-1]
        mid = box_mid.iloc[-1]
        if pd.isna(high) or pd.isna(low) or pd.isna(mid):
            return None

        buffer = self.sl_buffer_atr * atr
        if direction == 1:
            entry = low
            sl = low - buffer if buffer > 0 else None
            denom = buffer if buffer > 0 else atr
        else:
            entry = high
            sl = high + buffer if buffer > 0 else None
            denom = buffer if buffer > 0 else atr

        # 狭いボックス（TP 距離がスプレッド以下）は構造的不採算なので見送る
        if self.min_mid_atr > 0 and abs(mid - entry) / atr < self.min_mid_atr:
            return None

        return entry, sl, mid, denom
