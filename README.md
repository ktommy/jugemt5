# MT5 自動売買スケルトン（Python）

MetaTrader5 公式 Python API を使った、差し替え可能な自動売買の骨組み。
現在は**デモ口座での動作確認**を想定しています。戦略ロジックは `strategy.py` に実装するだけで差し替えられます。

## 概要

- 一定間隔でローソク足を取得し、**新しい確定足が発生したときだけ**戦略を評価して発注します。
- 発注は `magic` 番号で管理し、**このボットが発注したポジションのみ**を操作します（手動・他EAのポジションには触れません）。
- 現在は「**EMA 20/75/200 クロス**」戦略が組み込まれています。

## 前提

- **Windows**（MetaTrader5 パッケージは Windows のみ対応）
- MetaTrader 5 端末がインストール済み
- 取引口座（**デモ推奨**）へログイン済み

## ファイル構成

| ファイル | 役割 |
|---|---|
| `config.py` | 設定（接続・銘柄・ロット・時間足・戦略パラメータなど） |
| `strategy.py` | 戦略インターフェース + 戦略（EMA20/75/200クロス・RSI逆張り・MAクロス） |
| `broker.py` | MT5 接続・レート取得・発注・決済・ポジション管理・起動時自己診断 |
| `main.py` | 監視ループ（確定足ごとに戦略評価 → 発注） |
| `test_strategy.py` | 戦略のユニットテスト（合成データで検証、MT5 不要） |
| `requirements.txt` | 依存関係 |
| `README.md` | 本ドキュメント |

## セットアップ

```powershell
# 依存関係をインストール（既存 venv を使用）
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## MT5 側の設定

1. MT5 端末を起動し、取引口座（デモ推奨）にログインする。
2. `ツール` → `オプション` → `エキスパートアドバイザー` タブを開く。
3. **「アルゴリズム取引を許可する」** にチェックを入れて `OK`。

### ⚠ 重要: 対象シンボル × 時間足のチャートを開いておく

MT5 の Python API は、**端末がローカルに保持している履歴データ**からレートを取得します。
端末は「あるシンボルのチャートを開いたとき」に、そのシンボル × 時間足のヒストリカルデータをサーバーから同期します。

そのため、**使用する銘柄・時間足のチャートを MT5 上で一度開いて履歴を同期**しておかないと、
`copy_rates_from_pos()` がデータを返さず、ボットが取得に失敗します。

- 例: `config.py` で `symbol = "BTCUSD"`, `timeframe = "M5"` なら → MT5 で **BTCUSD の 5分足チャートを開く**。

履歴が無い場合、後述の「起動時の自己診断」が検知して案内を出します。

## 設定（config.py）

| 項目 | 初期値 | 説明 |
|---|---|---|
| `login` | `0` | デモ口座のログイン番号。`0` のままなら現在ログイン中の端末をそのまま使う |
| `password` | `""` | `login` を指定する場合に必要 |
| `server` | `""` | `login` を指定する場合に必要 |
| `symbol` | `"BTCUSD"` | 取引対象（銘柄コードはブローカーにより異なる） |
| `timeframe` | `"M5"` | 時間足（`M1` / `M5` / `M15` / `H1` など） |
| `lot` | `0.01` | 固定ロット |
| `magic` | `20261010` | このボットの注文を識別する番号（他EAと重複しない値にする） |
| `deviation` | `20` | 許容スリッページ（ポイント） |
| `ema_fast` | `20` | 短期 EMA の期間 |
| `ema_mid` | `75` | 中期 EMA の期間 |
| `ema_slow` | `200` | 長期 EMA の期間（トレンドフィルタ） |
| `ema_trend_filter` | `True` | EMA200 で逆トレンドのクロスを抑止するか |
| `ema_tp_pct` | `0.005` | 利食い幅（エントリー価格比、`0` で無効） |
| `ema_sl_pct` | `0.003` | 損切り幅（エントリー価格比、`0` で無効） |
| `poll_interval` | `5.0` | 価格確認間隔（秒） |
| `log_file` | `"mt5_autotrade.log"` | ログ出力先ファイル |

## 現在の戦略（EMA 20/75/200 クロス + TP/SL）

`strategy.py` の `EMACrossStrategy` が有効です。確定足から EMA20・EMA75・EMA200 を計算し、
**EMA20 と EMA75 のクロス**を売買シグナルに、**EMA200 を長期トレンドフィルタ**に使うトレンドフォローです。
TP/SL（利食い・損切り）を注文時に設定します。

| 条件 | シグナル | 動作 |
|---|---|---|
| EMA20 が EMA75 を下→上に抜ける（かつ価格 > EMA200） | `BUY` | 買い |
| EMA20 が EMA75 を上→下に抜ける（かつ価格 < EMA200） | `SELL` | 売り |
| トレンドに逆らうクロス（価格が EMA200 の反対側） | `HOLD` | 見送り |
| それ以外 | `HOLD` | 何もしない |

クロス判定は直近2本の確定足で行います。EMA200 フィルタは長期トレンドに逆らうクロスを無視します
（`config.py` の `ema_trend_filter = False` で無効化できます）。

### 発注時の挙動

- 新規注文時に **テイクプロフィット（+0.5%）** と **ストップロス（-0.3%）** を同時に設定します（サーバー側で管理）。
- 逆向きポジションを保有中に逆向きシグナルが出た場合は、決済してから新規（＝ドテン）。
- 同じ向きのポジションを保有中は追加発注しません（増玉なし）。

> EMA の期間・トレンドフィルタ・TP/SL 幅は `config.py` の `ema_*` 項目で変更できます。
> TP/SL はブローカーが定める最小ストップ距離（`stops_level`）より十分離す必要があります（近すぎるとリジェクトされます）。

## 実行

```powershell
.\venv\Scripts\python.exe main.py
```

停止は `Ctrl+C`。ログはコンソールと `mt5_autotrade.log` の両方に出力されます。

### 起動時の自己診断

起動直後に履歴データが取得できるかを確認します。

- **成功時**: `履歴データ確認OK: BTCUSD M5（取得 3 本）` と表示して監視を開始。
- **失敗時**: 以下のような案内を出して終了します。

```
[ERROR] mt5.broker: BTCUSD の M5 の履歴データを取得できません（(...)）。
MT5 端末で BTCUSD の M5 チャートを開いてください。
```

## テスト

戦略のシグナル判定ロジックは、MT5 への実接続なしで合成データから検証できます（`test_strategy.py`）。

```powershell
# 詳細表示（各テスト名と OK/FAIL が出る）
.\venv\Scripts\python.exe -m unittest test_strategy -v

# 結果だけ表示
.\venv\Scripts\python.exe -m unittest test_strategy
```

`unittest` は標準ライブラリなので追加インストールは不要です（実行には `pandas` / `numpy` が必要で、`pip install -r requirements.txt` で揃います）。

検証内容: EMA20×EMA75 のクロスで BUY/SELL が出ること、EMA200 フィルタが逆トレンドのクロスを抑止すること、履歴不足時や横ばい時に HOLD になること、など。

## 戦略の差し替え方

`strategy.py` に `Strategy` を継承したクラスを作り、`on_bar(df, pos_dir)` で
`Signal.BUY` / `Signal.SELL` / `Signal.CLOSE` / `Signal.HOLD` を返します。
`main.py` の `strategy = ...` の行を変更するだけです。

```python
from strategy import Strategy, Signal
import pandas as pd

class MyStrategy(Strategy):
    name = "my_strategy"

    # 任意: 新規注文に TP/SL を付けたい場合は比率を設定する（0 なら無効）
    tp_pct = 0.0
    sl_pct = 0.0

    def on_bar(self, df: pd.DataFrame, pos_dir: int = 0) -> Signal:
        # df には「最新の確定足まで」が渡される（形成中の足は除外済み）
        # pos_dir: +1=買い / -1=売り / 0=ポジション無し
        ...
        return Signal.HOLD
```

`df` の列: `time`, `open`, `high`, `low`, `close`, `tick_volume`, `spread`, `real_volume`。

## 注意事項

- これは**学習・検証用の骨組み**です。実運用前に十分なバックテストとデモ検証を行ってください。
- 自動売買は相場変動や約定スリッページにより損失を被る可能性があります。投資判断は自己責任でお願いします。
- ブローカーによって約定方式（FOK/IOC）や銘柄コード、提供する時間足が異なります。エラー時はログの `retcode` / `last_error` を確認してください。