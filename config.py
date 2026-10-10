"""設定ファイル。

接続情報・取引対象・注文パラメータ・ループ間隔を一元管理する。
本番運用では Config を直接書き換えるか、main.py 側で上書きして使う。
"""
from dataclasses import dataclass


def timeframe_to_int(name: str) -> int:
    """'M15' などの文字列を mt5.TIMEFRAME_* の数値定数に変換する。"""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return 0  # MT5 が無い環境では使わない
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    return mapping.get(name.upper(), mt5.TIMEFRAME_H1)


@dataclass
class Config:
    # --- MT5 接続 ---
    # login=0 の場合、既に起動・ログイン済みの MT5 端末をそのまま使う。
    login: int = 0
    password: str = ""
    server: str = ""

    # --- 取引対象 ---
    symbol: str = "BTCUSD"
    timeframe: str = "M5"

    # --- 注文 ---
    lot: float = 0.01            # 固定ロット
    magic: int = 20261010        # このボットの注文を識別するマジックナンバー
    deviation: int = 20          # 許容スリッページ（ポイント）

    # --- メインループ ---
    poll_interval: float = 5.0   # 価格確認間隔（秒）

    # --- ログ ---
    log_file: str = "mt5_autotrade.log"