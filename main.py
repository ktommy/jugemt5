"""エントリポイント。

候補銘柄（config の symbols）を一定間隔で走査し、新しい確定足のタイミングで
「戦略がシグナルを出した銘柄」を選び、その銘柄だけを取引する（単一銘柄ローテーション）。
"""
import logging
import time

import MetaTrader5 as mt5

from config import load_config
from strategy import Signal, EMACrossStrategy, compute_atr
from range_strategy import RSIRangeStrategy
from regime import Regime, detect_regime
from selector import scan_signals
from broker import MT5Broker


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _exit_prices(tick, signal: Signal, atr, sl_atr_mult, tp_atr_mult):
    """新規注文の TP / SL 価格を ATR 倍数で計算する（BUY/SELL で上下が異なる）。

    atr は最新確定足時点の平均真のレンジ。倍数が None/0 ならその側は無効。
    リスク正規化のため、価格は FX と BTC で共通の「ATR 倍数」で決める。
    """
    if signal == Signal.BUY:
        entry = tick.ask
        tp = entry + tp_atr_mult * atr if tp_atr_mult else None
        sl = entry - sl_atr_mult * atr if sl_atr_mult else None
    else:
        entry = tick.bid
        tp = entry - tp_atr_mult * atr if tp_atr_mult else None
        sl = entry + sl_atr_mult * atr if sl_atr_mult else None
    return entry, tp, sl


def act_on_signal(broker: MT5Broker, symbol: str, signal: Signal, position,
                  atr, sl_atr_mult=None, tp_atr_mult=None):
    """シグナルと現在のポジションから実際の売買を実行する。

    ロットは「1 トレードのリスク（risk_percent × 証拠金）」を SL 距離で割って
    自動計算する（compute_lot）。これで FX と BTC が同じリスク額で発注される。
    """
    if signal == Signal.HOLD:
        return position

    # 決済指示: 保有中ならクローズ
    if signal == Signal.CLOSE:
        if position is not None:
            logging.info("決済: %s", symbol)
            broker.close_position(position)
        return broker.get_position(symbol)

    # BUY / SELL: 逆向きポジション保有中なら一旦決済してから新規
    if position is not None:
        want_buy = signal == Signal.BUY
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        if want_buy != is_buy:
            logging.info("反転: %s のポジションをクローズ", symbol)
            broker.close_position(position)
            position = None

    if position is None:
        tick = broker.get_tick(symbol)
        entry, tp, sl = _exit_prices(tick, signal, atr, sl_atr_mult, tp_atr_mult)
        risk_amount = broker.account_equity() * broker.config.risk_percent
        lot = broker.compute_lot(symbol, entry, sl, risk_amount)
        if signal == Signal.BUY:
            broker.market_buy(symbol, lot, sl=sl, tp=tp)
        elif signal == Signal.SELL:
            broker.market_sell(symbol, lot, sl=sl, tp=tp)

    return broker.get_position(symbol)


def _pos_dir(position) -> int:
    """ポジションから方向を返す（+1=買い / -1=売り / 0=無し）。"""
    if position is None:
        return 0
    return 1 if position.type == mt5.POSITION_TYPE_BUY else -1


def _latest_atr(df, period: int):
    """レート DataFrame（末尾に形成中足を含む）から最新確定足の ATR を返す。

    履歴不足や NaN・非正値の場合は None を返す（呼び出し側で発注を見送る）。
    """
    if df is None or len(df) < period + 3:
        return None
    atr = compute_atr(df.iloc[:-1], period)
    val = float(atr.iloc[-1])
    return val if val > 0 else None


def main() -> None:
    config = load_config()
    setup_logging(config.log_file)

    # トレンド相場用とレンジ相場用の戦略を用意し、レジーム判定で切り替える
    trend_strategy = EMACrossStrategy(
        fast=config.ema_fast, mid=config.ema_mid, slow=config.ema_slow,
        tp_atr_mult=config.ema_tp_atr, sl_atr_mult=config.ema_sl_atr,
        trend_filter=config.ema_trend_filter,
    )
    range_strategy = RSIRangeStrategy(
        period=config.rsi_period, oversold=config.rsi_oversold,
        overbought=config.rsi_overbought,
        adx_period=config.adx_period, adx_threshold=config.adx_threshold,
        tp_atr_mult=config.range_tp_atr, sl_atr_mult=config.range_sl_atr,
    )

    broker = MT5Broker(config)
    broker.connect()

    # 候補銘柄をすべてマーケットウォッチに追加（履歴同期・価格取得に必要）
    for symbol in config.symbols:
        mt5.symbol_select(symbol, True)

    # レジーム判定の時間足（上位足）。未設定/同一なら単一時間足（従来挙動）
    regime_tf = (config.regime_timeframe or "").strip() or config.timeframe
    multi_tf = regime_tf.upper() != config.timeframe.upper()

    # 起動時自己診断: 履歴が取得できる銘柄だけを候補に残す（上位足もチェック）
    symbols = []
    for symbol in config.symbols:
        try:
            broker.ensure_history(symbol)
            if multi_tf:
                broker.ensure_history(symbol, timeframe=regime_tf)
            symbols.append(symbol)
        except RuntimeError as exc:
            logging.error(exc)
    if not symbols:
        logging.error(
            "候補銘柄の履歴が1件も取得できませんでした。"
            "MT5 端末で該当チャートを開いてください。"
        )
        broker.disconnect()
        return

    # 候補全銘柄で統一して取得する履歴本数（戦略が必要とする本数を満たす）
    count = max(
        200,
        getattr(trend_strategy, "min_history", 200) + 2,
        getattr(range_strategy, "min_history", 200) + 2,
    )

    def get_rates(symbol):
        return broker.get_rates(symbol, count)

    def get_regime_rates(symbol):
        return broker.get_rates(symbol, count, timeframe=regime_tf)

    # 起動時に保有中のポジションがあれば引き継ぐ（レジーム不明なので直近barで採用）
    existing = broker.get_any_position()
    pos_symbol = existing.symbol if existing is not None else None
    pos_regime = None           # 現在のポジションを建てたレジーム（未知/無しなら None）
    holding_regime = None       # 保有中の銘柄で最後に観測したレジーム（ログ用）

    last_bar_time = None

    logging.info(
        "監視開始: symbols=%s timeframe=%s regime=%s trend=%s range=%s risk=%s%% atr=%s",
        symbols, config.timeframe, regime_tf,
        trend_strategy.name, range_strategy.name,
        config.risk_percent * 100, config.atr_period,
    )

    try:
        while True:
            # 参照銘柄（保有中はその銘柄、無ければ先頭）の直近足で確定バーを検知
            ref = pos_symbol if pos_symbol is not None else symbols[0]
            ref_df = broker.get_rates(ref, 3)
            if ref_df is None or len(ref_df) < 2:
                time.sleep(config.poll_interval)
                continue
            completed_ref = ref_df.iloc[:-1]
            bar_time = int(completed_ref["time"].iloc[-1].timestamp())
            if bar_time == last_bar_time:
                time.sleep(config.poll_interval)
                continue
            last_bar_time = bar_time

            if pos_symbol is None:
                # --- フラット: 全候補を走査し、最良のシグナルで新規エントリー ---
                candidates = scan_signals(
                    symbols, trend_strategy, range_strategy, get_rates,
                    config.adx_period, config.adx_threshold,
                    get_regime_rates=get_regime_rates if multi_tf else None,
                )
                for r in candidates:
                    logging.info(
                        "候補: %s %s score=%.1f（%s）",
                        r.symbol, r.signal.value, r.score, r.regime.value,
                    )
                if not candidates:
                    time.sleep(config.poll_interval)
                    continue

                best = candidates[0]
                logging.info(
                    "選択: %s %s（regime=%s）",
                    best.symbol, best.signal.value, best.regime.value,
                )
                atr = _latest_atr(get_rates(best.symbol), config.atr_period)
                if atr is None:
                    logging.warning("ATR を算出できず見送り: %s", best.symbol)
                    time.sleep(config.poll_interval)
                    continue
                act_on_signal(
                    broker, best.symbol, best.signal, None, atr,
                    sl_atr_mult=getattr(best.strategy, "sl_atr_mult", None),
                    tp_atr_mult=getattr(best.strategy, "tp_atr_mult", None),
                )

                new_position = broker.get_position(best.symbol)
                if new_position is not None:
                    pos_symbol = best.symbol
                    pos_regime = best.regime
                    holding_regime = best.regime
            else:
                # --- 保有中: その銘柄だけ評価（他銘柄へは乗り換えない） ---
                df = get_rates(pos_symbol)
                if df is None or len(df) < 2:
                    time.sleep(config.poll_interval)
                    continue
                completed = df.iloc[:-1]

                position = broker.get_position(pos_symbol)
                if position is None:
                    logging.info("ポジション消滅: %s（TP/SL 到達など）", pos_symbol)
                    pos_symbol = None
                    pos_regime = None
                    holding_regime = None
                    time.sleep(config.poll_interval)
                    continue

                if multi_tf:
                    regime_df = get_regime_rates(pos_symbol)
                    if regime_df is None or len(regime_df) < 2:
                        time.sleep(config.poll_interval)
                        continue
                    regime_completed = regime_df.iloc[:-1]
                else:
                    regime_completed = completed

                regime = detect_regime(
                    regime_completed, config.adx_period, config.adx_threshold
                )
                if regime != holding_regime:
                    holding_regime = regime
                    logging.info("相場判定: %s -> %s", pos_symbol, regime.value)
                active = trend_strategy if regime == Regime.TREND else range_strategy

                # 旧レジーム保有継続: レジームが変わったら TP/SL に委ね、反転しない
                if pos_regime is not None and pos_regime != regime:
                    signal = Signal.HOLD
                else:
                    signal = active.on_bar(completed, _pos_dir(position))

                if signal != Signal.HOLD:
                    logging.info(
                        "シグナル=%s（%s）戦略=%s（%s）",
                        signal.value, pos_symbol, active.name,
                        completed["time"].iloc[-1],
                    )
                atr = _latest_atr(df, config.atr_period)
                if atr is None:
                    logging.warning("ATR を算出できず見送り: %s", pos_symbol)
                    time.sleep(config.poll_interval)
                    continue
                act_on_signal(
                    broker, pos_symbol, signal, position, atr,
                    sl_atr_mult=getattr(active, "sl_atr_mult", None),
                    tp_atr_mult=getattr(active, "tp_atr_mult", None),
                )

                new_position = broker.get_position(pos_symbol)
                if new_position is None:
                    pos_symbol = None
                    pos_regime = None
                    holding_regime = None
                elif pos_regime is None:
                    pos_regime = regime  # 起動時の既存ポジションを採用

            time.sleep(config.poll_interval)
    except KeyboardInterrupt:
        logging.info("Ctrl+C で停止しました")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()