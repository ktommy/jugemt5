"""エントリポイント。

候補銘柄（config の symbols）を一定間隔で走査し、新しい確定足のタイミングで
「戦略がシグナルを出した銘柄」を選び、その銘柄だけを取引する（単一銘柄ローテーション）。
"""
import logging
import time

import MetaTrader5 as mt5

from config import load_config
from strategy import Signal, EMACrossStrategy
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


def _order_prices(tick, signal: Signal, tp_pct, sl_pct):
    """新規注文の TP / SL 価格を計算する（BUY/SELL で上下が異なる）。"""
    if signal == Signal.BUY:
        entry = tick.ask
        tp = entry * (1 + tp_pct) if tp_pct else None
        sl = entry * (1 - sl_pct) if sl_pct else None
    else:
        entry = tick.bid
        tp = entry * (1 - tp_pct) if tp_pct else None
        sl = entry * (1 + sl_pct) if sl_pct else None
    return entry, tp, sl


def act_on_signal(broker: MT5Broker, symbol: str, lot: float,
                  signal: Signal, position, sl_pct=None, tp_pct=None):
    """シグナルと現在のポジションから実際の売買を実行する。"""
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
        if signal == Signal.BUY:
            _, tp, sl = _order_prices(tick, signal, tp_pct, sl_pct)
            broker.market_buy(symbol, lot, sl=sl, tp=tp)
        elif signal == Signal.SELL:
            _, tp, sl = _order_prices(tick, signal, tp_pct, sl_pct)
            broker.market_sell(symbol, lot, sl=sl, tp=tp)

    return broker.get_position(symbol)


def _pos_dir(position) -> int:
    """ポジションから方向を返す（+1=買い / -1=売り / 0=無し）。"""
    if position is None:
        return 0
    return 1 if position.type == mt5.POSITION_TYPE_BUY else -1


def main() -> None:
    config = load_config()
    setup_logging(config.log_file)

    # トレンド相場用とレンジ相場用の戦略を用意し、レジーム判定で切り替える
    trend_strategy = EMACrossStrategy(
        fast=config.ema_fast, mid=config.ema_mid, slow=config.ema_slow,
        tp_pct=config.ema_tp_pct, sl_pct=config.ema_sl_pct,
        trend_filter=config.ema_trend_filter,
    )
    range_strategy = RSIRangeStrategy(
        period=config.rsi_period, oversold=config.rsi_oversold,
        overbought=config.rsi_overbought,
        adx_period=config.adx_period, adx_threshold=config.adx_threshold,
        tp_pct=config.range_tp_pct, sl_pct=config.range_sl_pct,
    )

    broker = MT5Broker(config)
    broker.connect()

    # 候補銘柄をすべてマーケットウォッチに追加（履歴同期・価格取得に必要）
    for symbol in config.symbols:
        mt5.symbol_select(symbol, True)

    # 起動時自己診断: 履歴が取得できる銘柄だけを候補に残す
    symbols = []
    for symbol in config.symbols:
        try:
            broker.ensure_history(symbol)
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

    # 起動時に保有中のポジションがあれば引き継ぐ（レジーム不明なので直近barで採用）
    existing = broker.get_any_position()
    pos_symbol = existing.symbol if existing is not None else None
    pos_regime = None           # 現在のポジションを建てたレジーム（未知/無しなら None）
    holding_regime = None       # 保有中の銘柄で最後に観測したレジーム（ログ用）

    last_bar_time = None

    logging.info(
        "監視開始: symbols=%s timeframe=%s trend=%s range=%s lot=%s",
        symbols, config.timeframe,
        trend_strategy.name, range_strategy.name, config.lot,
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
                act_on_signal(
                    broker, best.symbol, config.lot, best.signal, None,
                    sl_pct=getattr(best.strategy, "sl_pct", None),
                    tp_pct=getattr(best.strategy, "tp_pct", None),
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

                regime = detect_regime(
                    completed, config.adx_period, config.adx_threshold
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
                act_on_signal(
                    broker, pos_symbol, config.lot, signal, position,
                    sl_pct=getattr(active, "sl_pct", None),
                    tp_pct=getattr(active, "tp_pct", None),
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