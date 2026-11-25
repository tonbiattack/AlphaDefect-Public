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
| 判断の網羅（同一セッション内の実測メッセージ数） | `combat_action` 287 / `reward_screen` 15 / `card_select` 2 / `rest_site` 1 / `treasure` 1 / `map_select` 22 / `event` 10 / `game_over` 3 / `run_complete` 3 |
| ランの完走 | 単純エージェントでもゲームオーバーまで到達し、次のランを開始（`run_complete` 3 回） |

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

## 3.5 変換アダプタの実現可能性（観測エンコーダ・語彙・候補 index を保つ）

要件: 学習済みモデルは置換せず、観測エンコーダ・語彙・合法手の順序・候補 index・行動の意味・
モデル版を保つ。隠し情報は使わない。

分かったこと:

- spire-mentor のクライアント側が受け取るのは **JSON**（`combat_action` には手札の合法性・対象候補・山札・
  敵・オーブ・薬・`available_actions` まで含まれる）。うちのエンコーダは **ゲームのオブジェクト**
  （`RunState` / `Decision` / `CardModel` …）を受け取ってフレームを作るので、JSON から同じバイト列を
  作るには語彙・ID・順序の対応表を一から作ることになり、**同一分布を保証できない**（モデルの入力が変わる）。
- ただし、うちのエンコーダは C# で、いまも MOD 内でコンパイルしている。したがって現実的なのは
  「spire-mentor の MOD 側に、うちのエンコーダを呼ぶ小さなフックを足す」案:
  - 判断の直前に、その MOD が持っているゲームオブジェクトから `Observation.Build(...)` でフレームを作り、
    外部（WSL）へ送る。外部はフレームで推論し、**候補 index** を返す。
  - 候補の順序は、spire-mentor の `hand[]`/`options[]` の並びと、うちの `Decision.Choices` の並びを
    一致させる必要がある（`combat_action` の `card_index` は手札 index、`target_index` は `enemies[]`
    または `valid_target_indices` 基準）。ここは**対応表をテストで固定**できる。
  - これは MIT のフォークで可能（著作権表示と出典を残す）。ただし MOD の更新追従が必要。
- 逆に「外部で JSON → うちのフレーム」は、語彙（2287 キー）・エンティティ順・特徴量の再現が必要で、
  回帰の危険が大きい。**採用しない**。

結論: アダプタは「MOD 側でフレームを作って外部へ送る」形が唯一現実的。PoC の次の段階で、
spire-mentor の MOD にフレーム送信フックを足す（またはうちの MOD に画面駆動を取り込む）方向で検証する。

## 3.6 モデルを接続した PoC（2026-11-25、実機）

**spire-mentor に画面を駆動させ、判断だけを AlphaDefect のモデルに任せる**実験を実装した
（`game-mod/AlphaDefectAutopilot/MentorClient.cs`）。既定は無効で、
`mods/AlphaDefectAutopilot/mentor.txt` があるときだけ有効になる。

仕組み:

1. この MOD はゲーム内で動いているので、ゲームのオブジェクトから**そのまま**うちのフレームを作れる
   （`Observation.Build`。語彙・候補 index・行動の意味は普段と同一）。
2. spire-mentor の TCP（127.0.0.1:9002）に**クライアントとして接続**（ゲーム内なので loopback で届く）。
3. `combat_action` が来たら、うちの戦闘候補を作ってフレームを推論サーバ（WSL, 18765）へ送り、
   返ってきた index を spire-mentor の `play` / `end_turn` / `potion` に翻訳して返す。
4. それ以外の判断（報酬・店・休息など）は、いまは合法な既定（先頭の選択肢）で返す。
   `run_config` には `start_run`（DEFECT / A0）を返す。

実機での結果（v0.111.1、最終: 拒否 0 件）:

| 確認 | 結果 |
|---|---|
| 接続 | `[mentor] connected to spire-mentor on 127.0.0.1:9002` |
| ランの開始 | `answered run_config with start_run` → `Starting run 1/10` |
| モデルの判断が合法に実行される | `{"action":"play","card_index":2,"target_index":0,...}` を返し、spire-mentor が `Playing DEFEND_DEFECT` として実行。`end_turn` も同様 |
| 戦闘の完了 | `Combat finished` まで到達（画面上は「戦闘開始」、左上にうちの推奨、上部に Spire Mentor の Autopilot 表示） |
| 推論サーバ | `action` 75 件超（モデルが判断している） |

分かったこと:

- **`request_id` は JSON 文字列で返す**必要がある（数値だと spire-mentor が捨てる）。
- うちの候補 → 相手の行動の対応は、カードは手札の参照一致で index、対象は敵リストの順で index を取れば一致した。
- この構成なら **うちのエンコーダ・語彙・候補 index・モデルをそのまま**使える（モデル置換なし）。

拡張（同日）:

- `event` は `text_key` の一致でモデルの選択をそのまま返せるようにした（実測 7 件）。
- `card_reward` はカード ID の一致で index を返す（実測 2 件、うち一部は相手側で拒否）。
- `map_select` は座標の一致で index を返す（実測 5 件）。
- 返答待ちの間に次の判断が届いた場合、pending を上書きしないようにした（古い応答の取り違え防止）。
- 返事が要らないメッセージ（error / run_complete / game_over など）には返さない（再送ループの防止）。
- カード報酬の「スキップ」（`CardRewardAlt`）を `{"action":"skip"}` に翻訳。**これが 15〜180 件の
  「card_reward reply rejected」の原因だった**（未対応の種別に `end_turn` を返していた）。
- 推論の待ち時間を 60 秒に延ばし、時間切れ時は判断の種類に合った既定（カード報酬なら skip、
  戦闘なら end_turn）を返す。遅れて届いた返答は捨てる（次の要求に混ざらない）。
- 店（購入・退出）と報酬画面（受け取り・進む）もモデルの候補から返すようにした。
  実機で店 2〜9 件・報酬画面 6〜8 件をモデルが判断し、拒否 0 件を確認。
- 選牌（`card_select`）は、相手の `cards[]` と自分のデッキ（山札・手札・捨て札・除外・デッキ本体）から
  候補を作ってモデルに聞くようにした（`card_select` の残りは既定の「先頭」で返る）。

制約・未対応:

- 報酬・店・休息・選牌などはまだ既定の選択。次は同じ仕組みで候補を対応付ける。
- 有効化すると、こちらの選牌器とドライバは止まる（spire-mentor と衝突するため）。
- 起動は手動（`mentor.txt` を置く）。#21 のランチャーからの切替は未実装。

## 3.7 Mentor 経由の通しラン（2026-11-25、実機）

Mentor モード（`mentor.txt`）で通しランを回した結果:

| 到達 | 内容 |
|---|---|
| 第1幕 | 戦闘・報酬・ショップ・イベント・休息所を通過（モデル判断、拒否 0 件） |
| 第1幕ボス | 撃破して**第2幕へ遷移**（`Entering Boss (Act 2, ...)`） |
| 第2幕以降 | floor 33〜39 まで進行（デッキ 28〜33、hp 62→42、所持金 180） |

- 画面には「Spire Mentor Autopilot」の表示と、うちのモデルの推奨（`AI: [1] 出 HOLOGRAM p=0.314`）が同時に出る。
- このセッションでも**拒否された返答は 0 件**。
- 併用時にこちらの報酬画面パッチが相手の画面に触れないよう、mentor モードでは
  こちらの `NRewardsScreen` / `NCardRewardSelectionScreen` の捕捉を無効化した。

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

- spire-mentor の戦闘を**うちのモデル**で戦わせる確認（PoC は単純エージェント。戦闘の判断自体は
  `combat_action` 287 件で動作を確認済み）。
- **WSL から 9002 へは接続できない**ことを再確認（`127.0.0.1`/`localhost` とも Connection refused）。
  Windows 側クライアントでは接続できる。ファイアウォール開放・Windows 中継・外向き改造のいずれかが必要。
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
