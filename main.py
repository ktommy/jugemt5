"""エントリポイント。

一定間隔でローソク足を取得し、新しい確定足が発生したタイミングで
戦略 on_bar を評価して売買を実行する。
"""
import logging
import time

import MetaTrader5 as mt5

from config import Config
from strategy import Signal, EMACrossStrategy
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


def main() -> None:
    config = Config()
    setup_logging(config.log_file)

    # 戦略を差し替える場合はここを変更する
    strategy = EMACrossStrategy(
        fast=config.ema_fast, mid=config.ema_mid, slow=config.ema_slow,
        tp_pct=config.ema_tp_pct, sl_pct=config.ema_sl_pct,
        trend_filter=config.ema_trend_filter,
    )

    broker = MT5Broker(config)
    broker.connect()

    # 対象銘柄をマーケットウォッチに追加（価格流動・発注に必要）
    mt5.symbol_select(config.symbol, True)

    # 起動時自己診断: 履歴データが無ければ案内して終了する
    try:
        broker.ensure_history(config.symbol)
    except RuntimeError as exc:
        logging.error(exc)
        broker.disconnect()
        return

    last_bar_time = None
    logging.info(
        "監視開始: symbol=%s timeframe=%s strategy=%s lot=%s",
        config.symbol, config.timeframe, strategy.name, config.lot,
    )

    try:
        while True:
            # EMA(slow) の収束に必要な本数を戦略から取得（無ければ従来どおり 200）
            df = broker.get_rates(
                config.symbol,
                count=max(200, getattr(strategy, "min_history", 200) + 2),
            )
            if df is None:
                time.sleep(config.poll_interval)
                continue

            # 末尾は形成中の未確定足なので除外し、確定足のみで判定する
            if len(df) < 2:
                time.sleep(config.poll_interval)
                continue
            completed = df.iloc[:-1]
            bar_time = int(completed["time"].iloc[-1].timestamp())

            # 新しい確定足が発生したときだけ戦略を評価
            if bar_time != last_bar_time:
                last_bar_time = bar_time

                position = broker.get_position(config.symbol)
                pos_dir = 0 if position is None else (
                    1 if position.type == mt5.POSITION_TYPE_BUY else -1
                )
                signal = strategy.on_bar(completed, pos_dir)
                if signal != Signal.HOLD:
                    logging.info(
                        "シグナル=%s (%s) 保有=%s",
                        signal.value, completed["time"].iloc[-1],
                        "あり" if position else "なし",
                    )
                act_on_signal(
                    broker, config.symbol, config.lot, signal, position,
                    sl_pct=getattr(strategy, "sl_pct", None),
                    tp_pct=getattr(strategy, "tp_pct", None),
                )

            time.sleep(config.poll_interval)
    except KeyboardInterrupt:
        logging.info("Ctrl+C で停止しました")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()