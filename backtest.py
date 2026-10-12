"""バックテスター（R ベースの成績計測）。

過去のローソク足（M5 など）を古い順に1本ずつ既存の戦略へ流し込み、
本番 main.py と同じ意思決定（レジーム判定 → シグナル → エントリー/ドテン）を再現して、
成績を R（リスク倍数）で集計する。

戦略ロジック（strategy.py / range_strategy.py / regime.py）は書き換えず import して再利用し、
ライブのフローを忠実になぞる:

- フラット時: シグナル（BUY/SELL）でエントリー。entry=シグナル足 close、SL/TP=ATR 倍数。
- 保有時: サーバー側 SL/TP を足の high/low で判定（ギャップ時は open で約定扱い、
  同足で SL/TP 両到達は保守的に SL 優先）。未到達ならレジーム判定し、
  同レジームで逆向きシグナルならドテン（決済＋反対建て）、レジームが変わったら維持。
- データ末尾で未決済なら最終 close で時価評価決済（exit_reason="EOD"）。

R は「SL 距離の何倍動いたか」（= 損益価格幅 ÷ SL距離）で、ロットや口座通貨に依存しない。
リスク正規化済みのため、FX と BTC を同じ物差しで比較できる。

実行例:
    python backtest.py                       # config.json の全銘柄を MT5 から
    python backtest.py --symbol EURUSD --source csv --csv-dir backtest_data
"""
import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import load_config
from strategy import Signal, EMACrossStrategy, compute_atr, compute_adx, compute_rsi
from range_strategy import RSIRangeStrategy
from regime import Regime, detect_regime

log = logging.getLogger("backtest")


# ---------- データ読込 ----------

def load_csv_history(symbol: str, csv_dir: str = "backtest_data") -> Optional[pd.DataFrame]:
    """MT5 エクスポート形式の CSV を読み、broker.get_rates と同じ列に正規化する。

    想定ヘッダ（MT5 標準出力）: Date,Time,Open,High,Low,Close,TickVol,Vol,Spread
    Date は yyyy.mm.dd。列名は小文字化して吸収し、time/open/high/low/close のみ返す。
    """
    path = os.path.join(csv_dir, f"{symbol}.csv")
    if not os.path.isfile(path):
        return None

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns or "time" not in df.columns:
        raise ValueError(f"{path}: Date/Time 列が見つかりません")

    date = df["date"].astype(str).str.replace(".", "-", regex=False)
    df["time"] = pd.to_datetime(date + " " + df["time"].astype(str), errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"{path}: {col} 列が見つかりません")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["time", "open", "high", "low", "close"]
    df = df.dropna(subset=keep).sort_values("time").reset_index(drop=True)
    return df[keep]


# ---------- トレード/成績のデータ構造 ----------

@dataclass
class Trade:
    symbol: str
    direction: str          # "buy" / "sell"
    strategy: str
    regime: str             # エントリー時のレジーム（trend/range）
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    sl_distance: float      # R の分母（SL 距離、SL 無効時は ATR）
    r: float                # 損益を R（リスク倍数）で表した値（スプレッド控除後）
    exit_reason: str        # "TP" / "SL" / "REVERSAL" / "EOD"
    bars: int               # 保有足数
    cost_r: float = 0.0     # 控除したスプレッドコスト（R 換算）


@dataclass
class Summary:
    n: int = 0
    win_rate: float = 0.0
    total_r: float = 0.0
    avg_r: float = 0.0
    pf: float = 0.0
    max_dd: float = 0.0
    avg_bars: float = 0.0
    total_cost: float = 0.0   # 控除したスプレッドコストの合計（R 換算）
    by_reason: dict = field(default_factory=dict)
    by_regime: dict = field(default_factory=dict)
    by_regime_stat: dict = field(default_factory=dict)  # レジーム別の成績内訳（trend/range）


def _sub_stats(trades: List[Trade]) -> dict:
    """トレード部分集合の簡易成績（件数・勝率・平均R・PF）。"""
    n = len(trades)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "pf": 0.0}
    wins = [t.r for t in trades if t.r > 0]
    losses = [t.r for t in trades if t.r <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "avg_r": sum(t.r for t in trades) / n,
        "pf": pf,
    }


def summarize(trades: List[Trade]) -> Summary:
    if not trades:
        return Summary()
    wins = [t.r for t in trades if t.r > 0]
    losses = [t.r for t in trades if t.r <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(wins) / len(trades)
    total_r = sum(t.r for t in trades)
    avg_r = total_r / len(trades)
    avg_bars = sum(t.bars for t in trades) / len(trades)

    # 最大ドローダウン（R 累計の過去最高からの下落幅）
    max_dd = 0.0
    cum = 0.0
    peak = 0.0
    for t in trades:
        cum += t.r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    by_reason = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
    by_regime = {}
    for t in trades:
        by_regime[t.regime] = by_regime.get(t.regime, 0) + 1
    by_regime_stat = {
        reg: _sub_stats([t for t in trades if t.regime == reg])
        for reg in sorted({t.regime for t in trades})
    }

    return Summary(
        n=len(trades), win_rate=win_rate, total_r=total_r, avg_r=avg_r,
        pf=pf, max_dd=max_dd, avg_bars=avg_bars,
        total_cost=sum(t.cost_r for t in trades),
        by_reason=by_reason, by_regime=by_regime,
        by_regime_stat=by_regime_stat,
    )


# ---------- バックテストエンジン ----------

@dataclass
class _Pos:
    direction: int          # +1=買い / -1=売り
    entry: float
    sl: Optional[float]
    tp: Optional[float]
    denom: float            # R の分母（SL 距離、SL 無効時は ATR）
    entry_time: pd.Timestamp
    entry_index: int
    regime: Regime
    strategy_name: str


def _atr(completed: pd.DataFrame, period: int) -> Optional[float]:
    val = float(compute_atr(completed, period).iloc[-1])
    return val if val > 0 else None


_TF_STEPS = {
    "M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
    "H1": "1h", "H4": "4h",
    "D1": "1D", "W1": "7D", "MN1": "30D",   # D1 以上は暦日近似（週末は滑らかでない点は割り切る）
}


def _tf_step(tf: str) -> pd.Timedelta:
    """時間足名をバーの長さ（Timedelta）に変換する。未知は H1 相当（timeframe_to_int と同じ既定）。"""
    return pd.to_timedelta(_TF_STEPS.get(str(tf).upper(), "1h"))


def _align_adx(trade_df: pd.DataFrame, regime_df: pd.DataFrame,
               adx_period: int, trade_tf: str, regime_tf: str) -> np.ndarray:
    """上位時間足の ADX を、取引足の各足 i に「その足の確定時点で完成済みの上位足」で対応付ける。

    取引足 i（開時刻 t_i）は確定時点が t_i + trade_step。上位足（開時刻 T、長さ regime_step）は
    T + regime_step <= t_i + trade_step を満たすとき完成済みなので、
    T <= t_i - (regime_step - trade_step) を満たす上位足まで使える（見通しバイアス回避）。
    十分な上位足履歴が無い足は安全側に TREND 相当の高 ADX（=100）を置く（detect_regime の
    「履歴不足 → TREND」既定と整合させる）。
    """
    n = len(trade_df)
    out = np.full(n, 100.0)
    reg_adx = compute_adx(regime_df, adx_period).to_numpy()
    reg_times = pd.DatetimeIndex(regime_df["time"])
    trade_times = pd.DatetimeIndex(trade_df["time"])
    lag = _tf_step(regime_tf) - _tf_step(trade_tf)
    idx = reg_times.searchsorted(trade_times - lag, side="right") - 1
    valid = idx >= adx_period   # ADX が安定する最低本数を待ってから採用
    out[valid] = reg_adx[idx[valid]]
    return out


class _Indicators:
    """全履歴の指標を一度だけベクトル計算し、各足では O(1) で参照するためのキャッシュ。

    各足で `compute_atr` / `compute_adx` / `ewm` を最初からやり直すと、1本当たり
    数ミリ秒 × 数万本で数分かかってしまう。EMA/RSI/ADX は再帰的に過去へ依存する
    ため「その足までの全履歴で計算した値」を配列として前計算しておけば、
    ループ内は配列の添字参照だけで済む（合計 O(n)）。
    """

    def __init__(self, df: pd.DataFrame, trend, range_, adx_period: int, atr_period: int,
                 regime_df: Optional[pd.DataFrame] = None,
                 trade_tf: Optional[str] = None, regime_tf: Optional[str] = None):
        close = df["close"]
        self.close = close.to_numpy()
        self.ema_fast = close.ewm(span=getattr(trend, "fast", 20), adjust=False).mean().to_numpy()
        self.ema_mid = close.ewm(span=getattr(trend, "mid", 75), adjust=False).mean().to_numpy()
        self.ema_slow = close.ewm(span=getattr(trend, "slow", 200), adjust=False).mean().to_numpy()
        self.atr = compute_atr(df, atr_period).to_numpy()
        self.rsi = compute_rsi(close, getattr(range_, "period", 14)).to_numpy()
        # ADX はレジーム判定とレンジ戦略の ADX フィルタの両方で使う。regime_df を渡すと
        # 上位時間足の ADX を取引足に時間対応付けたものになり、二つの判定が同じ尺度になる。
        if regime_df is None:
            self.adx = compute_adx(df, adx_period).to_numpy()
        else:
            self.adx = _align_adx(df, regime_df, adx_period, trade_tf, regime_tf)


def _ema_signal(ind: _Indicators, i: int, strategy: EMACrossStrategy) -> Signal:
    """EMACrossStrategy.on_bar と同じ判定を、前計算済み指標から再現する。"""
    prev = ind.ema_fast[i - 1] - ind.ema_mid[i - 1]
    cur = ind.ema_fast[i] - ind.ema_mid[i]
    price = ind.close[i]
    if prev <= 0 < cur:
        if not strategy.trend_filter or price > ind.ema_slow[i]:
            return Signal.BUY
    elif prev >= 0 > cur:
        if not strategy.trend_filter or price < ind.ema_slow[i]:
            return Signal.SELL
    return Signal.HOLD


def _rsi_range_signal(ind: _Indicators, i: int, strategy: RSIRangeStrategy) -> Signal:
    """RSIRangeStrategy.on_bar と同じ判定を、前計算済み指標から再現する。"""
    if ind.adx[i] >= strategy.adx_threshold:
        return Signal.HOLD
    prev, cur = ind.rsi[i - 1], ind.rsi[i]
    if prev < strategy.oversold <= cur:
        return Signal.BUY
    if prev > strategy.overbought >= cur:
        return Signal.SELL
    return Signal.HOLD


def _check_stops(pos: _Pos, bar) -> tuple:
    """サーバー側 SL/TP をその足の high/low で判定する。

    戻り値は (約定価格, 理由)。約定しない場合は (None, None)。
    同足で SL/TP 両方到達した場合は保守的に SL を優先し、ギャップ時は open で約定扱い。
    """
    o, h, l = bar["open"], bar["high"], bar["low"]
    if pos.direction == 1:
        sl_hit = pos.sl is not None and l <= pos.sl
        tp_hit = pos.tp is not None and h >= pos.tp
        if sl_hit:
            return (o if o <= pos.sl else pos.sl), "SL"
        if tp_hit:
            return (o if o >= pos.tp else pos.tp), "TP"
    else:
        sl_hit = pos.sl is not None and h >= pos.sl
        tp_hit = pos.tp is not None and l <= pos.tp
        if sl_hit:
            return (o if o >= pos.sl else pos.sl), "SL"
        if tp_hit:
            return (o if o <= pos.tp else pos.tp), "TP"
    return None, None


def _open(bar, direction: int, atr: float, active, regime: Regime, index: int) -> _Pos:
    entry = float(bar["close"])
    sl_mult = getattr(active, "sl_atr_mult", 0.0) or 0.0
    tp_mult = getattr(active, "tp_atr_mult", 0.0) or 0.0

    sl_distance = sl_mult * atr
    if sl_distance <= 0:
        sl = None               # SL 無効
        denom = atr             # R の分母は ATR にフォールバック
    else:
        sl = entry - sl_distance if direction == 1 else entry + sl_distance
        denom = sl_distance

    tp = None
    if tp_mult > 0:
        tp = entry + tp_mult * atr if direction == 1 else entry - tp_mult * atr

    return _Pos(direction, entry, sl, tp, denom, bar["time"], index, regime,
                getattr(active, "name", "strategy"))


def _close(pos: _Pos, bar, exit_px: float, reason: str, index: int,
           spread_price: float = 0.0) -> Trade:
    dist = (exit_px - pos.entry) * pos.direction
    gross_r = dist / pos.denom
    # 往復で 1 スプレッド分のコストを引く（約定は不利側＝買いは ask、売りは bid）。
    cost_r = spread_price / pos.denom if spread_price and pos.denom > 0 else 0.0
    r = gross_r - cost_r
    direction = "buy" if pos.direction == 1 else "sell"
    return Trade(
        symbol="",  # 呼び出し側で埋める
        direction=direction, strategy=pos.strategy_name, regime=pos.regime.value,
        entry_time=pos.entry_time, exit_time=bar["time"],
        entry=pos.entry, exit=float(exit_px),
        sl_distance=pos.denom, r=r, exit_reason=reason,
        bars=index - pos.entry_index, cost_r=cost_r,
    )


def run(df: pd.DataFrame, symbol: str, trend_strategy, range_strategy,
        adx_period: int = 14, adx_threshold: float = 25.0,
        atr_period: int = 14, spread_price: float = 0.0,
        force_regime: Optional[str] = None,
        regime_df: Optional[pd.DataFrame] = None,
        trade_tf: Optional[str] = None, regime_tf: Optional[str] = None) -> List[Trade]:
    """単一銘柄の履歴を1本ずつ流して仮想売買を再現し、トレード列を返す。

    組み込み戦略（EMAクロス / RSIレンジ）は指標を全履歴で一度だけ前計算し、
    ループ内では配列添字で参照する高速パスを使う。独自戦略（Strategy 継承で
    on_bar のみ実装したもの）はスライディングウィンドウで毎足 on_bar を呼ぶ
    フォールバックで動く（本数の多い履歴では遅くなるが、正しく動く）。

    spread_price は往復 1 トレードのスプレッドコスト（価格単位）。0 なら控除しない。
    force_regime に "trend" / "range" を渡すと ADX 判定を無視してその戦略に固定する
    （単一戦略の単独検証やパラメータスイープ用）。None なら通常の自動切替。
    regime_df に上位時間足の履歴を渡すと、レジーム判定（＝戦略切替とレンジ戦略の ADX
    フィルタ）をその上位足の ADX で行う（見通しバイアスは時間対応付けで回避）。None なら
    シグナルと同じ時間足（従来挙動）。上位足は組み込み戦略のベクトル計算パスのみ対応。
    """
    n = len(df)
    warmup = max(
        getattr(trend_strategy, "min_history", 0),
        getattr(range_strategy, "min_history", 0),
        atr_period, adx_period,
    ) + 2
    if n <= warmup:
        return []

    # 組み込み戦略ならベクトル計算パス、そうでなければ on_bar をそのまま呼ぶ
    vectorized = (isinstance(trend_strategy, EMACrossStrategy)
                  and isinstance(range_strategy, RSIRangeStrategy))
    ind = _Indicators(df, trend_strategy, range_strategy, adx_period, atr_period,
                      regime_df if vectorized else None, trade_tf, regime_tf) \
        if vectorized else None

    def regime_for(i: int, completed) -> Regime:
        if force_regime == "trend":
            return Regime.TREND
        if force_regime == "range":
            return Regime.RANGE
        if vectorized:
            return Regime.TREND if ind.adx[i] >= adx_threshold else Regime.RANGE
        return detect_regime(completed, adx_period, adx_threshold)

    def atr_at(i: int) -> Optional[float]:
        val = float(ind.atr[i])
        return val if val > 0 else None

    def signal_at(i: int, active, pos_dir: int, completed) -> Signal:
        if vectorized:
            if isinstance(active, EMACrossStrategy):
                return _ema_signal(ind, i, active)
            return _rsi_range_signal(ind, i, active)
        return active.on_bar(completed, pos_dir)

    pos: Optional[_Pos] = None
    trades: List[Trade] = []

    # 戦略・指標が必要とする直近本数だけを渡す（フォールバック用）。
    # 全履歴を毎回渡すと O(n²) になり実用に耐えないため、本番と同じスライディングウィンドウにする。
    window = warmup

    for i in range(warmup, n):
        bar = df.iloc[i]
        completed = None if vectorized else df.iloc[i - window + 1: i + 1]

        if pos is None:
            # --- フラット: シグナルでエントリー ---
            if vectorized:
                atr = atr_at(i)
                if atr is None:
                    continue
            else:
                atr = _atr(completed, atr_period)
                if atr is None:
                    continue
            regime = regime_for(i, completed)
            active = trend_strategy if regime == Regime.TREND else range_strategy
            signal = signal_at(i, active, 0, completed)
            if signal == Signal.BUY:
                pos = _open(bar, 1, atr, active, regime, i)
            elif signal == Signal.SELL:
                pos = _open(bar, -1, atr, active, regime, i)
        else:
            # --- 保有中: まずサーバー側 SL/TP を判定 ---
            exit_px, reason = _check_stops(pos, bar)
            if reason is not None:
                trades.append(_close(pos, bar, exit_px, reason, i, spread_price))
                pos = None
                continue

            # SL/TP 未到達: レジーム判定し、同レジームで逆向きシグナルならドテン
            regime = regime_for(i, completed)
            if pos.regime != regime:
                continue        # レジームが変わったら TP/SL に委ねる（本番と同じ）
            active = trend_strategy if regime == Regime.TREND else range_strategy
            signal = signal_at(i, active, pos.direction, completed)
            reverse = (
                (pos.direction == 1 and signal == Signal.SELL)
                or (pos.direction == -1 and signal == Signal.BUY)
            )
            if reverse:
                trades.append(_close(pos, bar, bar["close"], "REVERSAL", i, spread_price))
                atr = atr_at(i) if vectorized else _atr(completed, atr_period)
                pos = _open(bar, -pos.direction, atr, active, regime, i) if atr else None

    # データ末尾で未決済ポジションを時価評価決済
    if pos is not None:
        last = df.iloc[-1]
        trades.append(_close(pos, last, last["close"], "EOD", n - 1, spread_price))

    for t in trades:
        t.symbol = symbol
    return trades


# ---------- 出力 ----------

def print_summary(symbol: str, s: Summary, span: str = None) -> None:
    if span:
        print(f"=== {symbol}（実データ {span}） ===")
    else:
        print(f"=== {symbol} ===")
    print(f"  トレード数: {s.n}   勝率: {s.win_rate:.1%}   合計R: {s.total_r:+.2f}")
    print(f"  平均R(期待値): {s.avg_r:+.3f}   PF: {s.pf:.2f}   最大DD: {s.max_dd:.2f}R   平均保有: {s.avg_bars:.1f}足")
    if s.total_cost:
        print(f"  スプレッド控除: 合計 {s.total_cost:.2f}R（1トレード平均 {s.total_cost / s.n:.3f}R）")
    for reg in sorted(s.by_regime_stat):
        st = s.by_regime_stat[reg]
        pf = f"{st['pf']:.2f}" if st["pf"] != float("inf") else "∞"
        print(f"  [{reg}] 件数 {st['n']}  勝率 {st['win_rate']:.1%}  平均R {st['avg_r']:+.3f}  PF {pf}")
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(s.by_reason.items()))
    print(f"  決済理由: {reasons or '-'}")


def save_trades(trades: List[Trade], out_path: str) -> None:
    if not trades:
        return
    df = pd.DataFrame([t.__dict__ for t in trades])
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info("トレード履歴を %s に保存しました（%d 件）", out_path, len(trades))


# ---------- CLI ----------

def build_strategies(cfg, trend_tp=None, trend_sl=None,
                     range_tp=None, range_sl=None):
    """config から2戦略を生成。TP/SL 倍数は引数で上書きできる（スイープ用）。"""
    trend = EMACrossStrategy(
        fast=cfg.ema_fast, mid=cfg.ema_mid, slow=cfg.ema_slow,
        tp_atr_mult=cfg.ema_tp_atr if trend_tp is None else trend_tp,
        sl_atr_mult=cfg.ema_sl_atr if trend_sl is None else trend_sl,
        trend_filter=cfg.ema_trend_filter,
    )
    range_ = RSIRangeStrategy(
        period=cfg.rsi_period, oversold=cfg.rsi_oversold,
        overbought=cfg.rsi_overbought,
        adx_period=cfg.adx_period, adx_threshold=cfg.adx_threshold,
        tp_atr_mult=cfg.range_tp_atr if range_tp is None else range_tp,
        sl_atr_mult=cfg.range_sl_atr if range_sl is None else range_sl,
    )
    return trend, range_


def _parse_csv_floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _span_of(df) -> str:
    """データの実期間（time 列の最小〜最大日）を 'YYYY-MM-DD〜YYYY-MM-DD' で返す。"""
    t = df.get("time") if df is not None else None
    if t is None or len(df) == 0:
        return "-"
    try:
        lo = pd.Timestamp(t.min()).strftime("%Y-%m-%d")
        hi = pd.Timestamp(t.max()).strftime("%Y-%m-%d")
    except Exception:
        return "-"
    return f"{lo}〜{hi}"


def _slice_time(df, start, end):
    """--start/--end の日付範囲 [start, end]（境界含む）に絞る。両方 None ならそのまま。"""
    if start is None and end is None:
        return df
    t = df.get("time")
    if t is None or not pd.api.types.is_datetime64_any_dtype(t):
        return df
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end
    return df[mask].reset_index(drop=True)


def _fetch_df(args, broker, sym, start=None, end=None, timeframe=None):
    if args.source == "mt5":
        df = broker.get_history(sym, max_bars=args.max_bars, timeframe=timeframe)
    else:
        df = load_csv_history(sym, args.csv_dir)
    return _slice_time(df, start, end) if df is not None else None


def _spread_for(args, broker, sym, cfg) -> float:
    if args.spread is not None:
        return args.spread
    if args.source == "mt5":
        return broker.symbol_spread_price(sym)
    return cfg.spread_price


def _run_sweep(args, broker, cfg, symbols, start=None, end=None) -> List[Trade]:
    """TP/SL 倍数の総当たり。単一戦略（force_regime）で銘柄横断に集計する。"""
    tps = _parse_csv_floats(args.sweep_tp)
    sls = _parse_csv_floats(args.sweep_sl)
    strat = args.sweep_strategy

    # データは1回ずつ取得してキャッシュ（スイープ中に何度も取得しないため）
    dfs, spreads = {}, {}
    for sym in symbols:
        df = _fetch_df(args, broker, sym, start, end)
        if df is None or len(df) < 60:
            log.warning("履歴なし/不足: %s（%s 本）", sym, 0 if df is None else len(df))
            continue
        dfs[sym] = df
        spreads[sym] = _spread_for(args, broker, sym, cfg)
    if not dfs:
        return []

    rng = ""
    if start is not None or end is not None:
        a = start.strftime("%Y-%m-%d") if start is not None else "…"
        b = end.strftime("%Y-%m-%d") if end is not None else "…"
        rng = f" / 指定 {a}〜{b}"
    lo = min((df["time"].min() for df in dfs.values()), default=None)
    hi = max((df["time"].max() for df in dfs.values()), default=None)
    span = ""
    if lo is not None:
        span = f" / 実データ {pd.Timestamp(lo).strftime('%Y-%m-%d')}〜{pd.Timestamp(hi).strftime('%Y-%m-%d')}"
    header = f"=== sweep [{strat}]（{cfg.timeframe} / {len(dfs)}銘柄合算{span}{rng}） ==="
    if args.source == "mt5" and args.spread is None:
        header += " ※実スプレッド込み"
    print(header)
    for tp in tps:
        for sl in sls:
            if strat == "trend":
                trend, range_ = build_strategies(cfg, trend_tp=tp, trend_sl=sl)
            else:
                trend, range_ = build_strategies(cfg, range_tp=tp, range_sl=sl)
            combo: List[Trade] = []
            per_sym = {}
            for sym, df in dfs.items():
                tt = run(df, sym, trend, range_,
                         cfg.adx_period, cfg.adx_threshold, cfg.atr_period,
                         spread_price=spreads[sym], force_regime=strat)
                combo.extend(tt)
                per_sym[sym] = tt
            s = summarize(combo)
            pf = f"{s.pf:.2f}" if s.pf != float("inf") else "∞"
            print(f"  TP={tp} SL={sl}  件数 {s.n:4d}  勝率 {s.win_rate:5.1%}  "
                  f"平均R {s.avg_r:+6.3f}  PF {pf}  合計R {s.total_r:+8.1f}")
            if len(dfs) > 1:
                # 銘柄別の内訳（合計R 降順）。エッジが特定銘柄（例 BTC）による偏りかを見る
                for sym, tt in sorted(per_sym.items(),
                                      key=lambda kv: summarize(kv[1]).total_r, reverse=True):
                    ss = summarize(tt)
                    print(f"      {sym:<8} 件数 {ss.n:4d}  平均R {ss.avg_r:+6.3f}  "
                          f"合計R {ss.total_r:+8.1f}")
    return []  # スイープはトレード履歴の保存対象ではない


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="MT5 bot のバックテスター")
    parser.add_argument("--symbol", default=None, help="対象銘柄（省略時は config の全銘柄）")
    parser.add_argument("--source", choices=["mt5", "csv"], default="mt5",
                        help="データ源: mt5（端末から） / csv（backtest_data/<symbol>.csv）")
    parser.add_argument("--csv-dir", default="backtest_data", help="CSV の配置先")
    parser.add_argument("--timeframe", default=None,
                        help="時間足（M5/H1/H4 など）。省略時は config.json の timeframe。MT5 取得専用で config は変更しない")
    parser.add_argument("--regime-timeframe", default=None,
                        help="レジーム判定（ADX）に使う上位時間足（例 M15）。省略時は config.json の regime_timeframe。取引時間足と同じなら単一時間足")
    parser.add_argument("--max-bars", type=int, default=50000, help="MT5 から取得する最大本数")
    parser.add_argument("--spread", type=float, default=None,
                        help="スプレッドを価格単位で固定（省略時は mt5 で実値を取得 / csv で config.spread_price を使用）")
    parser.add_argument("--sweep", action="store_true",
                        help="TP/SL 倍数を総当たりでバックテスト（force_regime で単一戦略を固定）")
    parser.add_argument("--sweep-strategy", choices=["trend", "range"], default="trend",
                        help="スイープ対象の戦略（trend=EMAクロス / range=RSI逆張り）")
    parser.add_argument("--sweep-tp", default="2.0,3.0,4.0,5.0",
                        help="スイープする TP（ATR倍数）のカンマ区切りリスト")
    parser.add_argument("--sweep-sl", default="2.0",
                        help="スイープする SL（ATR倍数）のカンマ区切りリスト")
    parser.add_argument("--start", default=None,
                        help="開始日（YYYY-MM-DD）。この日以降のデータだけで検証（期間分割の再現性確認用）")
    parser.add_argument("--end", default=None,
                        help="終了日（YYYY-MM-DD）。この日以前のデータだけで検証")
    parser.add_argument("--out", default="backtest_trades.csv", help="トレード履歴の出力先")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.timeframe:
        cfg.timeframe = args.timeframe
        log.info("時間足を上書き: %s（config.json は変更しません）", args.timeframe)
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    if start is not None or end is not None:
        log.info("期間を絞る: %s 〜 %s", args.start or "…", args.end or "…")
    symbols = [args.symbol] if args.symbol else list(cfg.symbols)
    trend, range_ = build_strategies(cfg)

    # レジーム判定の上位時間足（シグナルは cfg.timeframe のまま）
    trade_tf = cfg.timeframe
    regime_tf_raw = args.regime_timeframe if args.regime_timeframe is not None else cfg.regime_timeframe
    regime_tf = (regime_tf_raw or "").strip() or trade_tf
    multi_tf = regime_tf.upper() != trade_tf.upper()
    if multi_tf and args.source != "mt5":
        log.warning("CSV は単一時間足のみ。上位TFレジーム（%s）は使わず %s で判定します", regime_tf, trade_tf)
        regime_tf = trade_tf
        multi_tf = False
    if multi_tf:
        log.info("上位TFレジーム: %s（シグナル %s）", regime_tf, trade_tf)

    broker = None
    if args.source == "mt5":
        import MetaTrader5 as mt5
        from broker import MT5Broker
        broker = MT5Broker(cfg)
        broker.connect()
        for s in symbols:
            mt5.symbol_select(s, True)
    elif args.source == "csv":
        pass

    if args.sweep:
        try:
            _run_sweep(args, broker, cfg, symbols, start, end)
        finally:
            if broker is not None:
                broker.disconnect()
        return

    all_trades: List[Trade] = []
    try:
        for sym in symbols:
            log.info("バックテスト対象: %s", sym)
            df = _fetch_df(args, broker, sym, start, end, timeframe=trade_tf)
            if df is None or len(df) < 60:
                log.warning("履歴なし/不足: %s（%s 本）", sym, 0 if df is None else len(df))
                continue
            spread_price = _spread_for(args, broker, sym, cfg)
            if spread_price:
                log.info("スプレッド想定: %s = %s（価格単位）", sym, spread_price)

            regime_df = None
            if multi_tf:
                regime_df = _fetch_df(args, broker, sym, start, end, timeframe=regime_tf)
                if regime_df is None or len(regime_df) < 60:
                    log.warning("上位TF履歴なし/不足: %s（%s）→ 単一時間足へフォールバック", sym, regime_tf)
                    regime_df = None

            trades = run(df, sym, trend, range_,
                         cfg.adx_period, cfg.adx_threshold, cfg.atr_period,
                         spread_price=spread_price,
                         regime_df=regime_df,
                         trade_tf=trade_tf, regime_tf=regime_tf)
            print_summary(sym, summarize(trades), _span_of(df))
            all_trades.extend(trades)
    finally:
        if broker is not None:
            broker.disconnect()

    save_trades(all_trades, args.out)


if __name__ == "__main__":
    main()