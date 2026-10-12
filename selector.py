"""候補銘柄を走査し、「稼げそうな」銘柄を選ぶ選別ロジック。

採用方式は案B（シグナル発火ベース）: 直近の確定足で戦略が実際に BUY/SELL を
出した銘柄だけを候補とし、同点時のタイブレークのみ補助指標で並べる。

戦略は銘柄に依存しない（on_bar は DataFrame だけを受け取る）ため、
このモジュールは MT5 接続とは独立しており、単体でテストできる。
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

from regime import Regime, detect_regime
from strategy import Strategy, Signal, compute_adx


@dataclass
class ScanResult:
    """1 銘柄の走査結果。signal が BUY/SELL のものだけが選別対象になる。"""

    symbol: str
    regime: Regime
    signal: Signal
    score: float
    strategy: Strategy


def scan_signals(
    symbols: List[str],
    trend_strategy: Strategy,
    range_strategy: Strategy,
    get_rates: Callable[[str], Optional[pd.DataFrame]],
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    get_regime_rates: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
) -> List[ScanResult]:
    """各候補銘柄を走査し、シグナルを出した銘柄をスコア降順で返す。

    - get_rates: 銘柄名を受け取りレートの DataFrame を返す関数。
      main.py では broker.get_rates のラッパ、テストではフェイクを渡す。
    - 末尾は形成中の未確定足なので、ここで除去して確定足のみで判定する。
    - レジーム判定は通常 get_rates のデータ（＝シグナルと同じ時間足）で行うが、
      get_regime_rates を渡すと上位時間足の判定になり、スコアもその ADX で付ける
      （「M5 のシグナル × M15 の相場判定」）。
    - スコアはタイブレーク用の補助指標: トレンドは ADX が高いほど、
      レンジは ADX が低いほど「そのレジームらしい」ため上位にする（発火が主条件）。
    """
    if get_regime_rates is None:
        get_regime_rates = get_rates

    results = []
    for symbol in symbols:
        df = get_rates(symbol)
        if df is None or len(df) < 2:
            continue
        completed = df.iloc[:-1]

        regime_df = get_regime_rates(symbol)
        if regime_df is None or len(regime_df) < 2:
            continue
        regime_completed = regime_df.iloc[:-1]

        regime = detect_regime(regime_completed, adx_period, adx_threshold)
        active = trend_strategy if regime == Regime.TREND else range_strategy
        signal = active.on_bar(completed, 0)  # フラット前提で走査

        if signal in (Signal.BUY, Signal.SELL):
            adx = float(compute_adx(regime_completed, adx_period).iloc[-1])
            score = adx if regime == Regime.TREND else adx_threshold - adx
            results.append(ScanResult(symbol, regime, signal, score, active))

    results.sort(key=lambda r: r.score, reverse=True)
    return results