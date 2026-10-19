"""相場レジーム（トレンド / レンジ）の判定。

ADX（平均方向性指数）の大きさで「いまがトレンド相場か、横ばい（レンジ）相場か」を判別する。
"""
from enum import Enum

import pandas as pd

from strategy import compute_adx


class Regime(Enum):
    TREND = "trend"
    RANGE = "range"
    MOMENTUM = "momentum"   # 通貨強弱戦略（レジーム概念なし）のトレードを区別するための値


def detect_regime(df: pd.DataFrame, adx_period: int = 14,
                  adx_threshold: float = 25.0) -> Regime:
    """最新確定足の ADX から相場を判定する。

    ADX が adx_threshold 未満なら RANGE、以上なら TREND。
    履歴が不足する場合は安全側の TREND を返す。
    """
    if len(df) < adx_period + 2:
        return Regime.TREND
    adx = compute_adx(df, adx_period).iloc[-1]
    return Regime.RANGE if adx < adx_threshold else Regime.TREND