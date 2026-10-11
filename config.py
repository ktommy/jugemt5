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

    # --- 戦略（EMA 20/75/200 クロス） ---
    ema_fast: int = 20            # 短期 EMA
    ema_mid: int = 75             # 中期 EMA
    ema_slow: int = 200           # 長期 EMA（トレンドフィルタ）
    ema_trend_filter: bool = True  # EMA200 で逆トレンドのクロスを抑止
    ema_tp_pct: float = 0.005     # +0.5% で利食い
    ema_sl_pct: float = 0.003     # -0.3% で損切り

    # --- レジーム判定（トレンド / レンジ） ---
    adx_period: int = 14           # ADX の期間
    adx_threshold: float = 25.0    # これ未満をレンジ、以上をトレンドとみなす

    # --- レンジ相場戦略（RSI 逆張り + ADX フィルタ） ---
    rsi_period: int = 14           # RSI の期間
    rsi_oversold: float = 30.0     # 売られすぎ（これ未満から反転で BUY）
    rsi_overbought: float = 70.0   # 買われすぎ（これ超えから反転で SELL）
    range_tp_pct: float = 0.005    # +0.5% で利食い
    range_sl_pct: float = 0.003    # -0.3% で損切り

    # --- メインループ ---
    poll_interval: float = 5.0   # 価格確認間隔（秒）

    # --- ログ ---
    log_file: str = "mt5_autotrade.log"