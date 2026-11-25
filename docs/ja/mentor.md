# Mentor 連携の設定（キャラクター・アセンション・シード）— #41

Spire Mentor に画面操作を任せ、判断だけ AlphaDefect のモデルにやらせるモードの設定です。
ランの開始は spire-mentor が `run_config` として外部に問い合わせ、こちらが `start_run` を返します。

## 使い方

```bash
# DEFECT / A0 / 固定シード
./play.sh mentor --character DEFECT --ascension 0 --seed 12345

# 使えるキャラの先頭で開始（auto）
./play.sh mentor --character auto

# ゲーム内で選んで開始（自動開始しない）
./play.sh mentor --character select

# Windows のランチャーから（メニュー 5 番）
launch.cmd

# 設定だけ書いて、あとで起動する
./play.sh mentor --character SILENT --ascension 5 --seed 4242 --no-launch

# いまの設定と、その版で使えるキャラ
./play.sh mentor --status
```

- `--seed` は整数（省略・`random` で乱数）
- `--ascension` は 0〜10
- 未検証のキャラは `--allow-untested` を付けたときだけ開始できます

## 設定の流れ

```
CLI / ランチャー
  └─ local/versions.py で「その版で動くキャラか」「検証済みか」を確認
  └─ mods/AlphaDefectAutopilot/mentor.json に書き出す
       {"character": "SILENT", "ascension": 5, "seed": 4242}
  └─ spire-mentor の autostart を有効化

ゲーム内 MOD（MentorClient）
  └─ mentor.json を読む
  └─ run_config の characters[] / max_ascension と突き合わせる
       ・一覧に無いキャラ → 開始しない（ログに利用可能なキャラを出す）
       ・auto → 一覧の先頭
       ・select → 開始しない
  └─ start_run を返し、決めた内容を推論サーバへ runconfig として送る
```

推論サーバは受け取ったフレームの `character` でモデルを切り替えるため、**実際に始まったキャラと
推論モデルは一致**します（別キャラのモデルで黙って代用しません）。

### 連続ラン・キャラ変更

- 推論サーバは `--character auto` で起動し、**フレームのキャラでモデルを選びます**
- 途中で `mentor.json` を書き換えると、**次のランから**その設定で始まります
- キャラが変わったら、そのキャラのモデルへ**切り替えます**（別キャラのモデルで打ちません）

実機（1 セッション、2 ラン）:

| ラン | 設定 | 結果 |
|---|---|---|
| 1 | DEFECT / A0 / seed 555 | 敗北 floor 17 |
| 2 | SILENT / A0 / seed 777（セッション中に設定変更） | 開始・進行を確認（`runconfig (SILENT, 777)`、probe も Silent） |

## 実機で確認したこと（v0.111.1、2026-11-25）

| 設定 | 結果 |
|---|---|
| `--character DEFECT --ascension 0 --seed 12345` | `answered run_config: DEFECT / A0 / seed=12345`、戦闘・報酬・選牌まで進行（敗北 floor 17） |
| `--character REGENT --ascension 0 --seed 999` | `answered run_config: REGENT / A0 / seed=999`、戦闘開始を確認 |
| `--character SILENT --ascension 5 --seed 4242` | `answered run_config: SILENT / A5 / seed=4242`、戦闘開始を確認 |
| `--character auto` | ゲームが提示した先頭（IRONCLAD）で開始 |
| `--character select` | 開始しない（ゲーム内で選ぶ）。spire-mentor の autopilot はメニュー開始のみ対応のため、このモードでは観戦しない |

### アセンションについて（重要）

`--ascension 5` を指定しても、**ゲーム側のプロファイルがアセンションを解放していない**ため
実際のランは A0 で始まりました。ゲームログ:

```
[StartRunLobby] CHARACTER.SILENT has not revealed the Ascension Epoch, disabling Ascension.
[SimAutoSlayer] Setting ascension level to 5 (panel was 0)
Embarking on a singleplayer SILENT run. Ascension: 0 Seed: 4242
```

つまり指定は spire-mentor まで届いていますが、**ゲームが A0 に落とします**。
A0 以外を試すには、ゲーム内でそのキャラのアセンションを解放したプロファイルが必要です。

## 使えるキャラ（`local/versions.py` が正）

| ゲーム版 | キャラ | 状態 |
|---|---|---|
| v0.111.1（= v0.111.0 系） | DEFECT / REGENT / SILENT | validated |
| v0.113.1 | FOOL / IRONCLAD / NECROBINDER / DEFECT / SILENT | validated |
| v0.113.1 | REGENT | untested（`--allow-untested` が必要） |

未対応の版・キャラ・重み不足は、その場ではっきり断ります（別キャラのモデルで代用しません）。

## 表示（HUD・ログ・計測）

- ゲームログ: `[mentor] answered run_config: SILENT / A0 / seed=31337`
- 推論サーバのログ: `{"event":"runconfig","character":"SILENT","ascension":0,"seed":31337}`
- 状態の要約（probe）にも載ります（HUD・`/status`・計測から見える）:

```json
{"character": "Silent", "mentor_character": "SILENT", "mentor_ascension": 0, "mentor_seed": 31337}
```

シードを指定しない場合は `seed: null`（乱数）として扱い、ゲームが決めた実際のシードは
`run_state` の `seed` に現れます（ログに残ります）。

## 記録

決めた開始設定は、推論サーバのログに `runconfig` として残ります。

```json
{"t": 1795600041.677, "event": "runconfig", "character": "REGENT", "ascension": 0, "seed": 999}
```

ゲーム側のログにも `[mentor] answered run_config: REGENT / A0 / seed=999` として残ります。

## 観戦の間隔（#52）

Mentor モードでも、**判断を先に HUD に出してから、設定した実時間だけ待って実行**します
（Godot のメインスレッドは止めません）。

```bash
# 行動の間隔（ミリ秒）。0 = 最速、500 = 通常、1500 = 観戦、3000 = 学習
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"command":"pacing 1500"}' http://127.0.0.1:18766/control
```

- パネルの「間隔」ボタン、ショートカットでも同じ値を変えられます
- 実行前に HUD へ `[idx] 行動名（1500 ms 待って実行）` を出します
- **一時停止中は実行しません**（再開すると、状態が変わっていなければ実行、変わっていれば破棄）
- **1手実行**は 1 回だけ実行してから自動で一時停止します
- 実行直前に `state_id` を確認し、古い判断は捨てます（`dropped a stale action` とログ）

実測（v0.111.1、DEFECT）:

```
[mentor] chosen: 出 GO_FOR_THE_EYES → EXOSKELETON (wait 1500 ms, combat_action)
[mentor] chosen: 出 SYNCHRONIZE (wait 1500 ms, combat_action)
```

## 既存の設定との関係

- 速度・行動間隔・入場待ち・一時停止・1手実行・手動切替・通常プレイ起動は、これまでどおりです
  （mentor モードでもパネルは動きます）。
- 独自 MOD の `--autostart "キャラ|アセンション|シード"` はそのまま使えます。
  mentor モードでは、同じ意味の設定を `mentor.json` で渡します。

## 注意

- `mentor.json` は `mods/AlphaDefectAutopilot/` に置かれます（MOD を入れ直しても残ります）。
- 設定を消すと MOD の既定（auto / A0 / 乱数）に戻ります。
- 未検証の項目（v0.113.1 の各キャラ、A0 以外の実機）は、確認できるまで「未検証」と扱います。
