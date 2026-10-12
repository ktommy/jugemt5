"""MetaTrader5 の接続・データ取得・注文・ポジション管理を束ねるラッパー。

MT5 公式 Python API (MetaTrader5 パッケージ) を利用する。
"""
import logging
import math
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import Config, timeframe_to_int

log = logging.getLogger("mt5.broker")


class MT5Broker:
    def __init__(self, config: Config):
        self.config = config
        self.timeframe = timeframe_to_int(config.timeframe)

    # ---------- 接続 ----------
    def connect(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"MT5 初期化失敗: {mt5.last_error()}")

        if self.config.login:
            ok = mt5.login(
                self.config.login,
                password=self.config.password,
                server=self.config.server,
            )
            if not ok:
                raise RuntimeError(f"MT5 ログイン失敗: {mt5.last_error()}")

        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"口座情報を取得できません: {mt5.last_error()}")

        log.info(
            "接続成功: login=%s name=%s server=%s balance=%s equity=%s",
            info.login, info.name, info.server, info.balance, info.equity,
        )

    def disconnect(self) -> None:
        mt5.shutdown()
        log.info("MT5 切断しました")

    # ---------- データ取得 ----------
    def _tf(self, timeframe: Optional[str] = None) -> int:
        """timeframe 指定を MT5 の定数に変換する。None なら既定（config）を使う。"""
        return self.timeframe if timeframe is None else timeframe_to_int(timeframe)

    def get_rates(self, symbol: str, count: int = 200,
                  timeframe: Optional[str] = None) -> Optional[pd.DataFrame]:
        """直近 count 本のローソク足を取得（末尾は未確定の形成中足を含む）。

        timeframe を省略すると config.timeframe。上位時間足のレジーム判定など、
        別の時間足を明示的に取得したい場合に指定する。
        """
        rates = mt5.copy_rates_from_pos(symbol, self._tf(timeframe), 0, count)
        if rates is None or len(rates) == 0:
            log.warning("レート取得失敗: %s (%s)", symbol, mt5.last_error())
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_history(self, symbol: str, max_bars: int = 50000,
                    timeframe: Optional[str] = None) -> Optional[pd.DataFrame]:
        """指定銘柄の履歴を可能な限り（最大 max_bars 本）ページング取得する。

        copy_rates_from_pos は端末が同期済みの本数しか返さないため、count 刻みで
        遡りながら取得し、時間昇順・重複除去して返す。バックテストのデータ源に使う。
        """
        step = 20000
        chunks = []
        start = 0
        total = 0
        tf = self._tf(timeframe)
        log.info("履歴取得開始: %s（最大 %d 本）", symbol, max_bars)
        while total < max_bars:
            chunk = mt5.copy_rates_from_pos(symbol, tf, start, step)
            if chunk is None or len(chunk) == 0:
                break
            chunks.append(chunk)
            total += len(chunk)
            start += len(chunk)
            log.info("履歴取得中: %s（%d 本取得）", symbol, total)
            if len(chunk) < step:
                break
        if not chunks:
            log.warning("履歴取得失敗: %s (%s)", symbol, mt5.last_error())
            return None
        rates = np.concatenate(chunks)
        rates = rates[np.argsort(rates["time"])]
        df = pd.DataFrame(rates).drop_duplicates(subset="time")
        df["time"] = pd.to_datetime(df["time"], unit="s")
        log.info("履歴取得完了: %s（%d 本）", symbol, len(df))
        return df.reset_index(drop=True)

    def get_tick(self, symbol: str):
        return mt5.symbol_info_tick(symbol)

    def symbol_spread_price(self, symbol: str) -> float:
        """現在のスプレッドを価格単位で返す（往復コスト想定 = 1 スプレッド）。"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.0
        return float(info.spread * info.point)

    def ensure_history(self, symbol: str, count: int = 3,
                       timeframe: Optional[str] = None) -> None:
        """起動時の自己診断: 履歴データが取得できるか確認する。

        取得できない場合（端末が該当シンボル×時間足の履歴を未同期）、
        ユーザーに具体的な対処（チャートを開く）を示す RuntimeError を投げる。
        """
        tf_name = self.config.timeframe if timeframe is None else timeframe
        df = self.get_rates(symbol, count=count, timeframe=timeframe)
        if df is None:
            raise RuntimeError(
                f"{symbol} の {tf_name} の履歴データを取得できません"
                f"（{mt5.last_error()}）。\n"
                f"MT5 端末で {symbol} の {tf_name} チャートを開いてください。"
            )
        log.info(
            "履歴データ確認OK: %s %s（取得 %d 本）",
            symbol, tf_name, len(df),
        )

    # ---------- ポジション ----------
    def get_position(self, symbol: str):
        """このボット（magic 一致）の保有ポジションを 1 件返す。無ければ None。"""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return None
        for p in positions:
            if p.magic == self.config.magic:
                return p
        return None

    def get_any_position(self):
        """magic 一致の保有ポジションを全銘柄から 1 件返す。無ければ None。

        単一銘柄ローテーションで、起動時に「どの銘柄を保有しているか」を
        引き継ぐために使う。
        """
        positions = mt5.positions_get()
        if positions is None:
            return None
        for p in positions:
            if p.magic == self.config.magic:
                return p
        return None

    # ---------- 注文 ----------
    def account_equity(self) -> float:
        """口座の現在の証拠金（equity）を返す。取得できなければ 0。"""
        info = mt5.account_info()
        return float(info.equity) if info is not None else 0.0

    def compute_lot(self, symbol: str, entry_price: float, sl_price: float,
                    risk_amount: float) -> float:
        """リスク額からロットを逆算する（リスク正規化）。

        - SL までの距離（価格幅）を 1 ティックの価値・サイズで換算し、
          「1 ロットあたりの最大損失」を求める。
        - risk_amount（口座通貨）をその損失で割ってロットを決める。
        - ティック情報が無効な銘柄（trade_tick_size/value が 0）では
          config.lot をフォールバックとして返す。
        """
        info = mt5.symbol_info(symbol)
        tick_size = getattr(info, "trade_tick_size", 0.0) if info else 0.0
        tick_value = getattr(info, "trade_tick_value", 0.0) if info else 0.0
        distance = abs(entry_price - sl_price) if sl_price is not None else 0.0

        if not tick_size or not tick_value or not distance or not risk_amount:
            return self.normalize_lot(symbol, self.config.lot)

        loss_per_lot = (distance / tick_size) * tick_value
        lot = risk_amount / loss_per_lot
        return self.normalize_lot(symbol, lot)

    def normalize_lot(self, symbol: str, lot: float) -> float:
        """lot を銘柄の volume_step の倍数・min/max 範囲に丸める。"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return lot
        step = info.volume_step or 0.01
        lot = math.floor(lot / step) * step
        lot = max(info.volume_min, min(info.volume_max, lot))
        return round(lot, 8)

    def _filling(self, symbol: str) -> int:
        """銘柄が許容する約定方式（FOK 優先 → IOC）を返す。"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        for mode in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC):
            if info.filling_mode & mode:
                return mode
        return mt5.ORDER_FILLING_IOC

    def _round_price(self, symbol: str, price: float) -> float:
        """SL/TP などの価格を銘柄の桁数 (digits) に丸める。"""
        info = mt5.symbol_info(symbol)
        digits = info.digits if info else 2
        return round(price, digits)

    def _send(self, request: dict):
        result = mt5.order_send(request)
        if result is None:
            log.error("order_send が None: %s", mt5.last_error())
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("注文失敗 retcode=%s comment=%s", result.retcode, result.comment)
            return None
        log.info(
            "注文成功: %s %s vol=%s price=%s order=%s deal=%s",
            "BUY" if request["type"] == mt5.ORDER_TYPE_BUY else "SELL",
            request["symbol"], request["volume"], request.get("price"),
            result.order, result.deal,
        )
        return result

    def market_buy(self, symbol: str, lot: float, sl: float = None, tp: float = None):
        tick = self.get_tick(symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.normalize_lot(symbol, lot),
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "deviation": self.config.deviation,
            "magic": self.config.magic,
            "comment": "pytrader",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(symbol),
        }
        if sl is not None:
            request["sl"] = self._round_price(symbol, sl)
        if tp is not None:
            request["tp"] = self._round_price(symbol, tp)
        return self._send(request)

    def market_sell(self, symbol: str, lot: float, sl: float = None, tp: float = None):
        tick = self.get_tick(symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.normalize_lot(symbol, lot),
            "type": mt5.ORDER_TYPE_SELL,
            "price": tick.bid,
            "deviation": self.config.deviation,
            "magic": self.config.magic,
            "comment": "pytrader",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(symbol),
        }
        if sl is not None:
            request["sl"] = self._round_price(symbol, sl)
        if tp is not None:
            request["tp"] = self._round_price(symbol, tp)
        return self._send(request)

    def close_position(self, position):
        """保有ポジションを反対売買でクローズする。"""
        tick = self.get_tick(position.symbol)
        if position.type == mt5.POSITION_TYPE_BUY:
            type_, price = mt5.ORDER_TYPE_SELL, tick.bid
        else:
            type_, price = mt5.ORDER_TYPE_BUY, tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": type_,
            "position": position.ticket,
            "price": price,
            "deviation": self.config.deviation,
            "magic": self.config.magic,
            "comment": "pytrader close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(position.symbol),
        }
        return self._send(request)