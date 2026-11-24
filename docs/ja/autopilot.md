# 実ゲーム画面で AI に打たせる（観戦）— 設計と現状

イシュー #5（完全自動プレイ MOD・実ゲーム画面での AI 観戦）の実装。**現状は「制御系と推論の往復は完成・実フレームで通し試験済み、ゲーム側 MOD は状態送信まで」**です。行動の実行と観測フレームの生成が残っています。詳しくは [5. 実装状況](#5-実装状況と残り) と [7. 実機での確認](#7-実機での確認2026-11-24) を読んでください。

```bash
# ゲーム不要の通し試験（推論と制御の往復を実フレームで確認）
./play.sh autopilot --simulate --character DEFECT --seed ABC123

# ゲーム側 MOD を入れて、推論サーバを立てる
./play.sh mod install
./play.sh autopilot --character DEFECT          # 127.0.0.1:18765 で待つ
```

---

## 1. 全体の形

```text
Windows                                    WSL2 (Ubuntu)
┌──────────────────────────┐   TCP 127.0.0.1:18765   ┌──────────────────────────────┐
│ Slay the Spire 2         │  1 行 1 JSON           │ local/bridge                 │
│  └ MOD (game-mod/)       │ ─────────────────────► │  protocol  メッセージ定義     │
│     Harmony で NRun を    │   hello / probe /      │  autopilot 状態機械・ログ     │
│     監視し、状態を送る     │   state / done         │  agent     観測 → 推論        │
│                          │ ◄───────────────────── │  server    TCP 受け口・操作    │
│                          │   welcome / action /   │  sim       実フレームでの試験  │
│                          │   control / error      │  ↑ local/infer.Policy（#1）  │
└──────────────────────────┘                        └──────────────────────────────┘
```

- 推論は **#1 の共通推論 API（`local/infer.Policy`）をそのまま共有**します。ゲーム内でも、
  ヘッドレスでも、同じネットワーク・同じ前処理・同じ観測形式です。
- 接続は **MOD 側から**張ります。ゲームは自分の都合でいつ起動してもよく、推論側は待つだけです。
- 状態は **公開情報だけ**（幕・層・HP・ゴールド・戦闘中か・デッキ枚数、フレームを送る段階では
  観測フレームそのもの）。山札の中身や乱数は送りません（論文と同じ情報境界）。

## 2. 使い方

### 2.1 MOD を入れる

```bash
./play.sh mod install      # ビルドして <ゲーム>/mods/AlphaDefectAutopilot/ に置く
./play.sh mod status       # 入っているか、どのゲーム版向けにビルドしたか
./play.sh mod remove
```

ゲームを起動し、初回だけ出る MOD の同意画面で同意します（**設定 → Modding** で有効になって
いるか確認）。ゲームの起動引数 `--nomods` が付いていると MOD は読み込まれません。

### 2.2 推論サーバを立てる

```bash
./play.sh autopilot --character DEFECT
```

```text
[autopilot] キャラ DEFECT / ゲーム v0.111.1 / 形式 19 / ログ ~/.local/share/sls2/logs/autopilot-….jsonl
[autopilot] 127.0.0.1:18765 でゲーム側の接続を待っています（終了は Ctrl-C / quit）
操作:
  pause      一時停止（MOD は次の行動を待つ）
  resume     再開
  stop       即時停止
  speed N    速度倍率 0.1〜20（MOD に渡す待ち時間が 1/N になる）
  status     いまの状態と手数
```

MOD がつながると `connected` → `hello` → `probe` が流れます。

### 2.3 操作（観戦中の操作）

| 操作 | 効き方 |
|---|---|
| `pause` | 以後の状態に行動を返さない。MOD はその場で待つ |
| `resume` | 再開。MOD が同じ状態を送り直せば、そこから続く |
| `stop` | 即時停止。MOD は `control: stopped` を受けて以後何もしない |
| `speed 2` | 待ち時間（`hold_ms`）を 1/2 に。ゲーム側のアニメーション速度は設定 → Fast mode |
| `status` | 状態・行動数・重複数・経過秒 |

`Ctrl-C` でも止まります。

## 3. ゲームを使わない通し試験

ゲーム側の MOD が無くても（あるいは実機を汚したくないとき）、**無頭エンジンの実フレーム**を
プロトコルに流して、往復が正しいことを確かめられます。

```bash
./play.sh autopilot --simulate --character DEFECT --seed ABC123
```

実測（このリポジトリ、CPU）:

```text
[autopilot] action 第1幕 2層 -> [5] / 候補 用 BLESSING_OF_THE_FORGE → 自己 p=0.5749 (6.9ms)
...
[autopilot] done {"win": true, "floor": 49, "act": 2, "steps": 935, ...}

=== 通し試験の結果（13.2 秒 / 行動 935 回）===
シード ABC123 / DEFECT / A10: 勝利（全 3 幕クリア）
```

935 回の「状態 → 推論 → 行動 → エンジンが実行」が通り、1 ラン完走しました。フレームは
MOD が送るものと同じバイト列なので、MOD 側の実装が正しければ実機でも同じ経路を通ります。

## 4. プロトコル（版 1）

TCP 上を 1 行 1 つの JSON が流れます。定義は `local/bridge/protocol.py`。

| 向き | 種別 | 内容 |
|---|---|---|
| MOD → 推論 | `hello` | `v`（プロトコル版）、`client`、`game.version`、`character`、`mod.{id,version}` |
| MOD → 推論 | `probe` | `data`: 公開状態の要約（幕・層・HP・ゴールド・デッキ枚数など） |
| MOD → 推論 | `state` | `seq`、`frame`（観測フレームの base64）、`meta`（表示用の幕・層） |
| MOD → 推論 | `done` | `win`、`floor`、`act` — 1 ラン終了 |
| MOD → 推論 | `log` / `bye` | ログ・切断の理由 |
| 推論 → MOD | `welcome` | `run`、`obs_format`（MOD が使う観測形式） |
| 推論 → MOD | `action` | `seq`、`index`（選んだ候補）、`label`、`prob`、`candidates`（上位候補と確率）、`hold_ms` |
| 推論 → MOD | `control` | `paused` / `running` / `stopped`（理由つき） |
| 推論 → MOD | `error` | 致命的エラーの内容 |

決まりごと:

- 版が違えば `hello` の時点で切る（黙って別の意味で解釈しない）。
- **同じ `seq` は推論し直さない**（前回と同じ答えを返す）。**同じフレーム（SHA-256）で別 `seq`**
  も同じ答えを返す。これが重複操作の防止になる。
- 一時停止中は推論しない（MOD は待つ）。
- 推論・入力の例外は握りつぶさず、`error` を返して `stopped` にする。
- すべての出来事を JSONL（`~/.local/share/sls2/logs/autopilot-*.jsonl`）と標準出力の両方に残す。

## 5. 実装状況と残り

| 部分 | 状態 |
|---|---|
| プロトコル・状態機械（停止／一時停止／再開／速度／重複防止／例外時停止／ログ） | **実装済み・テスト済み** |
| 推論（観測フレーム → 推奨手・候補スコア・価値） | **実装済み・#1 と共有** |
| ゲームを使わない通し試験（実フレームで 1 ラン） | **実装済み・935 手で勝利を確認** |
| ゲーム側 MOD の読み込みと接続 | **実装済み**（実機での確認は下記） |
| MOD からの公開状態の送信（`probe`） | **実装済み**（実機で確認済み） |
| MOD での観測フレーム生成（戦闘の出牌局面） | **実装済み**（無頭エンジンのエンコーダをそのまま同梱。実機の戦闘では未確認） |
| 選ばれた行動のゲーム内実行（戦闘の出牌・薬水・ターン終了） | **実装済み**（実機の戦闘では未確認） |
| 戦闘の外の意思決定（報酬・経路・イベント・休息・宝箱・カードパック・選牌） | **実装済み**（実機で動作を確認） |
| ショップ | 出るだけ（買い物は未実装） |
| 画面への表示（選択した行動・上位候補・確率） | **実装済み**（実機の画面で確認） |
| 速度制御（ゲーム側） | **実装済み**（Godot の TimeScale。`speed N`） |

### 次の作業（観測フレームの生成）

エンジンの観測エンコーダ（`engine/Sts2Headless/Engine/Obs/**`）は、ゲームの型と
`RunSession` の静的メンバーだけに依存しています。実際、v0.111 系のエンコーダは
**無頭エンジンの外でもコンパイルできる**ことをこの作業で確認しました（`Decision` と
`PendingCardChoice` を足し、`RunSession` の 4 つの静的メンバーを実機向けの薄いシムに
差し替えれば通ります）。`engine/Directory.Build.targets` が `Sts2Bridge` 向けに
`Obs/**` と `Compat/**` を共有する作りになっているのも同じ理由です。

そのあとは、選ばれた候補の添字をゲームの呼び出しへ写す作業を、決定の種類ごとに進めます
（戦闘 → 報酬 → 経路 → ショップ → イベント）。ここが本質的に大きく、実機で 1 戦闘ずつ
確かめる必要があります。

### 速度について

ゲーム内のアニメーション速度は 設定 → Fast mode で変わります。MOD 側からその設定を
切り替えるのは未実装です。いまの `speed` は**推論側が指示する待ち時間（`hold_ms`）**で、
「AI が次の手を出す間隔」を調整します。

## 6. 注意（規約・互換性）

- **オフラインのシングルプレイだけ**で使ってください。マルチプレイやスコアボードに持ち込むと
  他のプレイヤーや記録に影響します。
- MOD はゲーム本体の保証対象外です。ゲームの利用規約と、他の MOD との互換性は自分で確認して
  ください（このリポジトリは MOD を同梱・配布しません）。
- MOD を入れるとセーブの扱いが変わります（初回に未 MOD のセーブを複製します）。
- ゲームを更新したら `./play.sh mod install` で入れ直してください（参照するアセンブリが変わります）。

## 7. 実機での確認（2026-11-24）

| 確認したこと | 結果 |
|---|---|
| MOD がゲームの MOD 機構でビルドできる（ゲーム v0.111.1 のアセンブリ参照） | 成功（`game-mod/AlphaDefectAutopilot`、12.8 KB の DLL） |
| `mods/AlphaDefectAutopilot/` への配置 | 成功（`./play.sh mod install`） |
| ゲーム内での読み込み | 確認済み。ゲームログに `Loading assembly DLL …AlphaDefectAutopilot.dll`、`[AlphaDefect] mod initializer called`、`--- RUNNING MODDED! --- Loaded 1 mods` |
| 接続とハンドシェイク | 確認済み。`[AlphaDefect] connected to 127.0.0.1:18765`、`[AlphaDefect] welcome: run=… obs_format=19`、こちら側のログに `hello … mod=AlphaDefectAutopilot` |
| `probe`（公開状態の送信） | 確認済み。プレイヤーが Ironclad のランを開始したあと、0.5 秒おきに 17 件の probe が届いた（`act 0 / floor 0 / hp 80/80 / gold 99 / deck 10 / combat false`） |
| ブリッジの通し試験（ゲーム不要） | 実フレーム 935 手で A10 を完走（Defect、13.2 秒、勝利） |
| 一時停止・再開・即時停止・速度・重複防止・例外時停止 | ユニットテスト（98 件）で確認 |

**実機で 1 戦闘の自動プレイに成功しました（2026-11-24 23:39、その後も複数回）。画面左上に
「AI: [1] 出 DUALCAST p=0.885 / 候補: …」の表示も出ています（2026-11-25 02:48 に確認）。** 推論サーバの `--autostart`
でゲーム側から Defect のランを開始し、最初の戦闘（TOADPOLES_WEAK）を AI が最後まで打ち切って
勝利しました。ゲームのログに `CHARACTER.DEFECT fought ENCOUNTER.TOADPOLES_WEAK for the first
time and WON` が残っています。報酬画面は実機で捕捉できています（`reward screen: 2 rewards`）。
報酬の取得と経路の選択は、サーバとゲームが途中で落ちたため**実機での完走は未確認**です。
カード報酬は選牌画面の操作が要るため、いまは飛ばします（デッキは育ちません）。

### 対応キャラ（このゲーム版）

ゲーム v0.111.x で動かせるのは **Defect / Regent / Silent** の重みだけです（Ironclad・Fool・
Necrobinder の重みは v0.113.x 向けで、観測形式が違います）。MOD はフレームの形式が重みと
違うと推論側で止まるようにしてあります（黙って別のキャラの数字を読ませない）。

### ポートについて

既定は `127.0.0.1:18765`（`ALPHADEFECT_AUTOPILOT_PORT` で変更）。最初は 8765 にしていたが、
このマシンでは別の常駐アプリ（instant-mic）が 0.0.0.0:8765 を使っていて、127.0.0.1 宛の
接続がそちらに取られる可能性があったため変えた。
