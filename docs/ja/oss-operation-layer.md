# 既存OSSの実ゲーム操作基盤の調査（#29）

issue #29 の候補4件を、一次資料（リポジトリ・リリース・コード）で確認した記録です。
**未確認の機能を前提にしていません。** 調査日: 2026-11-25。対象ゲーム: Slay the Spire 2 v0.111.1。

## 1. 候補の実在と基本情報

| リポジトリ | 実在 | 最新 push | ライセンス | ゲーム版の記載 | 接続方式 | 実ゲーム操作 |
|---|---|---|---|---|---|---|
| [tfriedel/spire-mentor](https://github.com/tfriedel/spire-mentor) | あり（21★） | 2026-11-23 | MIT | **v0.111.1（stable）/ v0.113.0 / v0.107.1** を明記 | ゲーム内MODが TCP サーバ（既定 9002） | Autopilot（`SimAutoSlayer`）が全画面を駆動。外部エージェントが全判断を返す |
| [Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP) | あり（547★） | 2026-07-29 | MIT | 明記なし（v0.107 系と推定） | MOD + MCP サーバ | 状態公開と操作（MCP） |
| [barrel10/STS2-Agent-mod](https://github.com/barrel10/STS2-Agent-mod) | あり（0★） | 2026-10-03 | Other（要確認） | 明記なし | ローカル HTTP + MCP | 状態公開と操作 |
| [LightEnding/autoSpire](https://github.com/LightEnding/autoSpire) | あり（2★） | 2026-09-22 | MIT | 明記なし | MOD + HTTP + MCP | 全ループ（戦闘・経路・報酬・店・イベント・宝箱・メニュー） |

判定に使った根拠:

- spire-mentor: `spire-mentor-mod/README.md` に「game build **v0.111.1**, the stable branch, or the **v0.113.0** beta」と明記。
  `SpireMentor.csproj` にも「One DLL for every supported game build: v0.107.1, v0.111.0 (stable as v0.111.1), v0.113.0」。
- autoSpire: README の構成（C# GameHookServer + MCP）と manifest（`autoSpire.json`, `min_game_version` なし）。
- STS2MCP: `McpMod.*.cs` と MCP サーバ。ゲーム版の記載は見当たらず、最終 push も 2026-07-29 で古い。
- STS2-Agent-mod: ライセンスが「Other」。**再利用前に許諾条件の確認が必要**（この調査では採用候補から外した）。

## 2. spire-mentor の構造（最有力）

```
spire-mentor/
├── spire-mentor-mod/     ← ゲーム内MOD（C# / Harmony）。これが「操作基盤」
│   ├── MainFile.cs, SimAutoSlayer*.cs, SimCombatHandler, SimCardSelect*, SimShop*, ...
│   ├── PROTOCOL.md       ← 全メッセージと verb の一次資料（1374 行）
│   └── SpireMentor.csproj
├── sts2-rs/              ← ヘッドレスシミュレータ（Rust）
├── sts2-pycli/           ← シミュレータの Python クライアント
└── PROVENANCE.md         ← ゲーム由来物を含めない方針（公開版にゲームコード・画像・音声なし）
```

- 仕組み: `NGame.IsReleaseGame()` を false にしてゲーム内蔵の **AutoSlay** を有効化し、
  `ICardSelector` の差し替え・`NonInteractiveMode` を使う。`SimAutoSlayer` がメニュー・部屋・画面を
  駆動し、**判断はすべて外部エージェントに問い合わせる**（ランダム代替なし）。
- 対応判断（PROTOCOL.md）: `combat_action` / `map_select` / `card_select` / `card_reward` /
  `card_bundle` / `crystal_sphere` / `rest_site` / `shop` / `event` / `treasure` / `boss_relic` /
  `reward_screen` / `game_over` / `run_complete` / `run_config`。
- `state_id` による古い応答の破棄、`invalid_index` / `stale_state` などの構造化エラー、
  「推測しない」方針が明記されている。
- 依存: **Alchyr/BaseLib**（MIT、v3.4.8 を使用。`min_game_version 0.107.1`）。

## 3. 実機PoC（2026-11-25、v0.111.1）

行ったこと:

1. BaseLib v3.4.8（MIT）を `<game>/mods/BaseLib/` に導入。
2. spire-mentor の MOD を WSL の .NET 9 でビルド（`-p:STS2_GAME_DIR=... -p:BaseLibDir=...`）→
   自動で `<game>/mods/SpireMentor/` に配置された。
3. `SpireMentor.conf` を `auto_start: true` / `client_run_config: true` / `pause_on: ""` にして起動。
4. 最小エージェント [`tools/spire_mentor_probe.py`](../tools/spire_mentor_probe.py) を作成
   （run_config に `start_run` を返し、各判断に合法な `choose` / `play` を返すだけ）。

結果:

| 確認 | 結果 |
|---|---|
| MOD のロード | `=== SpireMentor ready ===`、`Loaded 4 mods` |
| TCP サーバ | `[BridgeServer] Listening on 127.0.0.1:9002` |
| クライアント接続（Windows 側） | 成功。`run_config` を受信 |
| クライアント駆動の開始 | `start_run` を返すとラン開始（`Starting run 1/10`） |
| 画面の駆動 | `reward_screen` / `card_reward` を 50 回以上処理（`Handling screen: NCardRewardSelectionScreen` 等） |

分かった問題点:

1. **うちのMODと同時に読み込めない**: AlphaDefectAutopilot がカードセレクタを差し込んでいるため、
   spire-mentor は `A card selector is already active` でラン開始に失敗する。
   → 併用するなら、どちらか一方だけを `mods/` に置く（切替式）必要がある。
2. **接続の向きが逆**: うちのMODはゲーム→WSL（18765）へ接続するので WSL 側の待ち受けだけで済むが、
   spire-mentor はゲーム側が待ち受け（9002）で、WSL からは `Connection refused`（Windows ファイアウォール）。
   PoC では **Windows 側の Python** で接続した。WSL の推論につなぐには
   (a) ファイアウォール開放、(b) Windows 側に中継を置く、(c) MOD を「外へ接続する」向きに小改造、のいずれかが要る。
   #21 の「クリックだけで動く」を保つなら (c) か (b) が望ましい。
3. 依存 MOD（BaseLib）が必須。導入は MIT のリリースを `mods/BaseLib/` に置くだけ。

## 4. 判断（3案の比較）

| 観点 | 全面置換（spire-mentor を操作基盤に） | 部分再利用（画面駆動・プロトコルを採用） | 既存MODの修正（現状の継続） |
|---|---|---|---|
| 互換性 | ゲーム版は v0.111.1 対応で良いが、**観測が JSON** のため、うちのエンコーダ（ゲームオブジェクト前提）と直接は合わない | 画面駆動の実装（MOD側）を MIT の範囲で移植・参照できる。エンコーダはそのまま | 変更なし |
| 開発コスト | 観測アダプタを一から作る（大） | 停止しやすい画面（鍛冶・選牌・店・遷移）を重点的に移植（中） | 停止原因を個別に潰す（中・継続中） |
| 安定性 | MOD は成熟（never-guess、state_id、構造化エラー） | 同じ成熟部分を利用できる | 実測でボスまで到達。残りは選牌・鍛冶 |
| 保守性 | 外部の更新に追従（BaseLib 依存） | 部分的に追従 | 自前 |
| ライセンス | MIT（PROVENANCE でゲーム物なしを明記） | 同上（移植時は著作権表示と出典を残す） | 自前 |
| 接続 | ゲーム側待ち受け（WSL から届かない） | 外向き接続に小改造して使う | すでに外向き |

**採用案: 部分再利用。** 理由:

- うちの強みは「観測エンコーダ＋学習済みモデル＋操作パネル＋ランチャー」で、これは維持する。
- 停止しているのは **画面駆動（選牌・鍛冶・店・遷移）** で、spire-mentor の該当実装（MIT、v0.111.1 対応、
  never-guess、state_id による古い応答の破棄）はまさにその部分。全面置換すると観測アダプタの作り直しになり、
  モデルの入力（語彙・候補 index・行動の意味）を保証できなくなる。
- 併用はカードセレクタの衝突で不可（実測）。したがって「切替式」で、
  (1) まずは spire-mentor の画面駆動ルールを自前MODへ移植（出典明記）、
  (2) 行き詰まったら spire-mentor を「操作のみ」の実験バックエンドとして起動できるようにする、の順で進める。

## 5. 未検証・残課題

- spire-mentor の **戦闘** を外部エージェントで最後まで戦わせた確認（今回の PoC は報酬画面までの到達）。
- WSL 側の推論から spire-mentor に接続する経路（現状は Windows 側クライアントのみ）。
- うちのモデルの観測を spire-mentor の JSON から再構成する場合の情報充足（隠し情報は使わない）。
- 鍛冶・選牌の停止が、spire-mentor の画面駆動で実際に解消するかの実測（同一 seed での比較）。
- STS2MCP / autoSpire の実機動作（今回はコードと記載の確認まで）。
- STS2-Agent-mod のライセンス条件（「Other」）の確認。

## 6. 出典

- spire-mentor: README.md, PROTOCOL.md, PROVENANCE.md, SpireMentor.csproj, SimConfig.cs, SpireMentor.conf.template
- BaseLib: https://github.com/Alchyr/BaseLib-StS2 （v3.4.8, MIT）
- autoSpire: README.md, autoSpire.json
- STS2MCP: リポジトリ直下の `McpMod.*.cs` ほか
- STS2-Agent-mod: README.md, LICENSE
