"""通貨強弱（相対モメンタム）戦略。

単一銘柄のテクニカル指標ではなく、主要7通貨ペア（対USD）の直近の変化率を
突き合わせて「どの通貨が強く、どの通貨が弱いか」をランキングし、
最強通貨を買って最弱通貨を売る（最強/最弱ペアの合成）ロジックを提供する。

シグナルは「注目中の銘柄の構成通貨ペア（例 EURUSD なら EUR×USD）が、
全体ランキングの最強×最弱の組み合わせに一致したとき」だけ出る。
そのため単一銘柄の on_bar ループでは成立せず、バックテストは
backtest.run_currency_momentum（全銘柄を時間軸で突き合わせる専用ランナー）
から使われる。on_bar はインターフェース整合のためのプレースホルダで常に HOLD。

見通しバイアスは、変化率を .pct_change（= close を lookback 本だけ過去へ shift して
比較）で求め、さらに他銘柄の変化率を「その時点で完成済みの足」だけに
フォワードフィルで対応付けることで排除する。
"""
from typing import Dict, Optional

import pandas as pd

from strategy import Strategy, Signal


# 主要7ペア（すべて対USD）。前半3文字が base、後半3文字が quote。
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

# ランキング対象の8通貨（USD 含む）。スコア行列の列順として固定して使う。
CURRENCY_ORDER = ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF", "JPY"]

# 各ペアが「どの通貨のスコア」を生み、変化率を符号そのまま使うのか反転するのか。
# - XXXUSD（base=XXX, quote=USD）: ペア上昇 = XXX 強い → 変化率そのまま
# - USDXXX（base=USD, quote=XXX）: ペア上昇 = XXX 弱い → 変化率の符号を反転
PAIR_META = {
    "EURUSD": {"currency": "EUR", "sign": +1.0},
    "GBPUSD": {"currency": "GBP", "sign": +1.0},
    "AUDUSD": {"currency": "AUD", "sign": +1.0},
    "NZDUSD": {"currency": "NZD", "sign": +1.0},
    "USDCAD": {"currency": "CAD", "sign": -1.0},
    "USDCHF": {"currency": "CHF", "sign": -1.0},
    "USDJPY": {"currency": "JPY", "sign": -1.0},
}


def split_pair(symbol: str):
    """'EURUSD' → ('EUR', 'USD')。3+3 の構成を仮定。"""
    s = symbol.upper()
    return s[:3], s[3:]


def change_pct(df: pd.DataFrame, lookback: int) -> pd.Series:
    """lookback 本前の終値に対する現在終値の変化率（%）。

    pct_change は close を lookback 本だけ過去へ shift して比較するため、
    現在足から見て「見通し」を含まない（未来を覗かない）。
    """
    return df["close"].pct_change(lookback) * 100.0


def scores_from_chg(symbol_chg: Dict[str, float]) -> Dict[str, float]:
    """{銘柄: 変化率%} → {通貨: スコア}。USD は基準の 0。

    変化率が NaN の銘柄はスコア化しない（まだ履歴が足りない銘柄の扱い）。
    """
    scores = {"USD": 0.0}
    for sym, chg in symbol_chg.items():
        if chg is None or pd.isna(chg):
            continue
        base, quote = split_pair(sym)
        if quote == "USD":
            scores[base] = float(chg)
        elif base == "USD":
            scores[quote] = -float(chg)
        else:
            raise ValueError(f"USD を含まないペア {sym} は対象外")
    return scores


def rank_currencies(scores: Dict[str, float]):
    """スコアから (最強通貨, 最弱通貨, スコア差) を返す。値が足りなければ None。"""
    finite = {c: v for c, v in scores.items() if v is not None and not pd.isna(v)}
    if len(finite) < 2:
        return None
    strong = max(finite, key=finite.get)
    weak = min(finite, key=finite.get)
    return strong, weak, finite[strong] - finite[weak]


class CurrencyMomentumStrategy(Strategy):
    """通貨強弱（相対モメンタム）戦略のパラメータ集合。

    on_bar は単一銘柄の流し込みでは成立しないため常に HOLD を返す
    （実際のシグナル生成は backtest.run_currency_momentum が行う）。

    - lookback_period: 変化率の算出期間（H1 足なら 24 = 24時間を想定）
    - momentum_threshold: 最強/最弱ペアを取引する最低スコア差（%）
    - exit_threshold:    保有中にスコア差がこれを下回ったらクローズ（None なら momentum_threshold）
    - use_momentum_exit: ランキング入れ替わり / スコア差縮小でのクローズを使うか
    - sl_atr_mult / tp_atr_mult: 既存の ATR ベース TP/SL（0 で無効）
    """

    name = "currency_momentum"

    def __init__(
        self,
        lookback_period: int = 24,
        momentum_threshold: float = 0.5,
        exit_threshold: Optional[float] = None,
        use_momentum_exit: bool = True,
        sl_atr_mult: float = 2.0,
        tp_atr_mult: float = 3.0,
    ):
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.exit_threshold = exit_threshold
        self.use_momentum_exit = use_momentum_exit
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        # pct_change(lookback) が有効になるまで + バッファ
        self.min_history = lookback_period + 20

    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
        return Signal.HOLD