"""設定ファイル。

接続情報・取引対象・注文パラメータ・ループ間隔を一元管理する。
本番運用では Config を直接書き換えるか、main.py 側で上書きして使う。
"""
import json
from dataclasses import dataclass, field, fields


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

    # --- 取引対象（候補リスト） ---
    # ホワイトリスト方式: この中からシグナルを出した銘柄を選び、常に 1 銘柄だけ取引する。
    # 銘柄名はブローカーにより異なる。存在しない銘柄は起動時にスキップされる。
    symbols: list = field(default_factory=lambda: [
        "BTCUSD", "EURUSD", "USDJPY", "GBPUSD",
        "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
    ])
    timeframe: str = "M5"

    # --- レジーム判定の上位時間足 ---
    # トレンド/レンジの判定（ADX）をどの時間足で行うか。シグナルは timeframe のまま。
    # "" または timeframe と同じなら単一時間足（従来挙動）。上位足（例 "M15"）を
    # 指定すると「M5 のシグナル × M15 の相場判定」になる。
    regime_timeframe: str = "M15"

    # --- 注文 ---
    lot: float = 0.01            # フォールバックロット（リスク算出不能時のみ使用）
    magic: int = 20261010        # このボットの注文を識別するマジックナンバー
    deviation: int = 20          # 許容スリッページ（ポイント）

    # --- リスク管理（リスク正規化） ---
    risk_percent: float = 0.01   # 1 トレードの最大損失を口座証拠金の何%にするか
    atr_period: int = 14         # ATR の期間（ボラティリティ計測）

    # --- 取引コスト（バックテスト用） ---
    # 往復スプレッド想定を価格単位で指定（例: EURUSD なら 0.00012）。
    # MT5 から取得する場合は銘柄の実スプレッドを自動で使うので、この値は
    # CSV 検証や「スプレッドを固定して比較したい」ときのフォールバック/上書きに使う。
    spread_price: float = 0.0

    # --- 戦略（EMA 20/75/200 クロス） ---
    ema_fast: int = 20            # 短期 EMA
    ema_mid: int = 75             # 中期 EMA
    ema_slow: int = 200           # 長期 EMA（トレンドフィルタ）
    ema_trend_filter: bool = True  # EMA200 で逆トレンドのクロスを抑止
    ema_sl_atr: float = 2.0       # 損切り幅（ATR 倍数）
    ema_tp_atr: float = 3.0       # 利食い幅（ATR 倍数）

    # --- レジーム判定（トレンド / レンジ） ---
    adx_period: int = 14           # ADX の期間
    adx_threshold: float = 25.0    # これ未満をレンジ、以上をトレンドとみなす

    # --- レンジ相場戦略（RSI 逆張り + ADX フィルタ） ---
    rsi_period: int = 14           # RSI の期間
    rsi_oversold: float = 30.0     # 売られすぎ（これ未満から反転で BUY）
    rsi_overbought: float = 70.0   # 買われすぎ（これ超えから反転で SELL）
    range_sl_atr: float = 2.0      # 損切り幅（ATR 倍数）
    range_tp_atr: float = 2.0      # 利食い幅（ATR 倍数）

    # --- ボックスレンジ逆張り戦略（ドンチャンチャネル） ---
    box_window: int = 20            # ドンチャンチャネルの期間（ボックス幅）
    box_sl_buffer_atr: float = 0.5  # 損切りバッファ（バンド端±この ATR 倍数）
    box_bb_filter: bool = True      # ボリンジャーバンド幅フィルタを使うか
    box_bb_period: int = 20         # BB の期間
    box_bb_width: float = 0.05      # これ未満のバンド幅をレンジとみなす
    box_min_mid_atr: float = 0.0    # TP(中央線)距離がこの ATR 倍数未満なら見送り（0=無効）

    # --- 通貨強弱（相対モメンタム）戦略 ---
    momentum_lookback: int = 24        # 変化率の算出期間（本数。H1 足なら 24=24時間）
    momentum_threshold: float = 0.5    # 最強/最弱ペアを取引する最低スコア差（%）
    momentum_exit_threshold: float = 0.3  # 保有中にスコア差がこれを下回ったらクローズ（%）
    momentum_exit: bool = True         # ランキング崩れ/スコア差縮小でのクローズを使うか
    momentum_sl_atr: float = 2.0       # 損切り幅（ATR 倍数、0 で無効）
    momentum_tp_atr: float = 3.0       # 利食い幅（ATR 倍数、0 で無効）

    # --- メインループ ---
    poll_interval: float = 5.0   # 価格確認間隔（秒）

    # --- ログ ---
    log_file: str = "mt5_autotrade.log"


def _load_json(path: str) -> dict:
    """JSON ファイルを dict として読み込む。無い、あるいは壊れていたら空 dict。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_config(base_path: str = "config.json",
                override_path: str = "config_local.json") -> Config:
    """JSON から設定を読み込み、Config を組み立てる。

    - ``config.json`` が無い場合は dataclass の既定値をそのまま使う。
    - ``config_local.json``（任意・Git 管理外）を ``config.json`` の上に重ねることで、
      口座情報などローカル固有の設定を分離できる。
    - JSON に未知のキーがあっても無視し、値が無いフィールドは既定値のまま。
    """
    data = _load_json(base_path)
    data.update(_load_json(override_path))
    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in data.items() if k in valid})