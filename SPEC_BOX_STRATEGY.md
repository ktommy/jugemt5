# ボックスレンジ逆張り戦略（ドンチャンチャネルベース）仕様設計

> 確定日: 2026-10-13
> 決定事項: 指値約定 / 単一戦略化 / フィルタは AND
> 前段の経緯は `BACKTEST_FINDINGS.md`（EMAクロス／RSI逆張りは近現代でエッジ無し）を参照。

---

## 0. 全体方針

- `regime.py` の trend/range ルーティングは**使わなくする**（削除はしない。過去検証の再現性のため残す）。
  ボックス戦略 1 本に、自前の「ADX 低 **かつ** BB 幅狭」フィルタでトレンド時は休止（現金）。
- エントリーは**指値**（バンドに注文を置く）。バックテストでは「バンド到達で約定、窓開けは open で約定」を再現。
- まず **Phase 1 でバックテスト検証**し、エッジの有無を見極めてからライブ接続（Phase 2）に進む。

## 1. 新モジュール `box_strategy.py`

```python
class BoxRangeStrategy(Strategy):
    name = "box_range"

    def __init__(self, window=20,
                 adx_period=14, adx_threshold=25.0,
                 use_bb_filter=True, bb_period=20, bb_width_threshold=0.05,
                 sl_buffer_atr=0.5, min_mid_atr=0.0):
        ...
        self.min_history = max(window, bb_period, adx_period) + 100
```

- **指標（見通し排除）**: `box_high = high.rolling(window).max().shift(1)`、
  `box_low = low.rolling(window).min().shift(1)`、`box_mid = (box_high + box_low) / 2`。
  shift(1) で「1 本前までの足」のみ参照。
- **フィルタ（AND）**: `ADX < adx_threshold` **かつ**
  （BB フィルタ無効 or `(BB_upper − BB_lower) / BB_SMA < bb_width_threshold`）。
  両方満たして初めてエントリー許可。
- **シグナル**: フィルタ通過時に、最新確定足で `low ≤ box_low` → `BUY`、`high ≥ box_high` → `SELL`。
- **`exit_plan(df, direction, atr) -> (entry_px, sl_px, tp_px, denom)`**（新設の任意メソッド）:
  - 買い: `entry = box_low`、`sl = box_low − sl_buffer_atr × atr`、`tp = box_mid`、`denom = sl_buffer_atr × atr`
  - 売り: `entry = box_high`、`sl = box_high + sl_buffer_atr × atr`、`tp = box_mid`、`denom = sl_buffer_atr × atr`
  - `min_mid_atr > 0` かつ `|box_mid − entry| / atr < min_mid_atr` のときは entry 自体を見送り。

`Strategy` に `exit_plan` を追加（既定 `return None` → 既存戦略は ATR 倍数フォールバックで無変更）。
TP/SL は on_bar では返さず、エントリー時に一度だけ固定（SL の追従はしない）。

## 2. `backtest.py` の拡張（指値約定の再現）

- `_Indicators.__init__` に `box_high / box_low / box_mid / bb_width` を前計算追加。
- `_box_range_signal(ind, i, strategy)` を追加し、`vectorized` 判定を
  `isinstance(range_strategy, BoxRangeStrategy)` に拡張。
- `_open` を一般化（ボックス戦略用に `_open_box`）: `entry = バンド価格`、SL/TP/denom は `exit_plan` 相当。
  **ストップ判定 `_check_stops` は無変更で流用**（絶対価格 SL/TP を既に受け取れるため）。
- **指値約定**: フラット時の足 i で
  - 買い: `low[i] ≤ box_low[i]` なら `entry = min(open[i], box_low[i])` で建て
  - 売り: `high[i] ≥ box_high[i]` なら `entry = max(open[i], box_high[i])`
  - `_check_stops` の既存ギャップ処理と同型。
- **エントリー足はストップ判定しない**（既存エンジン踏襲）。※同足 SL 到達を見逃す楽観寄りは注記として残す。
- 単一戦略化は**最小変更で** `run(df, sym, HoldStrategy(), BoxRangeStrategy(), ..., force_regime="range")` とする
  （`run()` のシグネチャ変更なし、既存テスト温存）。`build_strategies` も box 対応に。

## 3. `config.py` / `config.json` 追加

`box_window`(20), `box_sl_buffer_atr`(0.5), `box_bb_filter`(true), `box_bb_period`(20),
`box_bb_width`(0.05), `box_min_mid_atr`(0.0)。`adx_period` / `adx_threshold` は流用。
`rsi_*` は削除せず残す（過去検証の再現用）。

## 4. テスト

- `test_box_strategy.py`: (a) 平坦レンジで安値タッチ→BUY／高値タッチ→SELL、
  (b) shift(1) で見通し排除、(c) ADX 高 or BB 幅広で HOLD（AND の両条件を個別に検証）、
  (d) `exit_plan` の価格一致。
- `test_backtest.py` の `VectorizedTest` に `_box_range_signal` vs `on_bar` のパリティテストを追加。

## 5. ライブ接続（Phase 2・スコープのみ）

- `broker.py` に `limit_buy/limit_sell`（`ORDER_ACTION_PENDING` + `BUY_LIMIT/SELL_LIMIT` + `ORDER_TIME_GTC`、
  SL/TP 付き）と `get_pending(symbol)`（magic 一致ペンディング検索）を追加。
- `main.py` は単一戦略化＋「フラット時にバンドへ指値配置→約定待ち→約定後は既存のポジション管理」に組み替え。
  **stops_level の考慮**（バンドが現在価格から近すぎるとリジェクト）が必須。
- 指値は「確定足ごとに張り替え」（約定しなければ毎足最新の box_low/box_high に更新）。

## 6. スイープ（Phase 3）

既存 `--sweep` は TP/SL の ATR 倍数軸で、box では無意味。
`window` / `sl_buffer_atr` / `bb_width_threshold` を振る専用スイープを追加。

- 実装: `backtest.py --box-sweep`（`--box-windows` × `--box-buffers` の総当たり。BB 幅フィルタは
  config 値のまま、`--box-no-bb` で無効化可）。`--spread 0` 併用で「グロス（控除前）期待値」を分離できる。
- 読み方: buffer を広げて平均R が上向くのはスプレッド換算が減るだけ。グロスが全 combo ±0 なら
  「勝率が損益分岐勝率ちょうど＝方向予測力ゼロ」で、パラメータ調整では勝てない（EMA/RSI と同じ結論）。

## 実装の進め方

| フェーズ | 内容 | ゴール |
|---|---|---|
| Phase 1 | `box_strategy.py` + `backtest.py` 拡張 + config + テスト | 既存計測装置で「この戦略にエッジがあるか」を検証。エッジ無しならここで止める |
| Phase 2 | `broker.py` 指値注文 + `main.py` 単一戦略化 | デモ口座で実接続検証 |
| Phase 3 | パラメータスイープ | 過学習に注意しつつ候補絞り込み |
