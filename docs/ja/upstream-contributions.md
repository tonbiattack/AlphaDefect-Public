# Fork 元への還元候補（#66）

上流に還元できそうなものの整理。**投稿はまだしていません**（承認後に別作業）。
対象: `hmty-CHEN/AlphaDefect-Public`（上流）と `spire-mentor`（上流: tfriedel）。

## 候補 A: v0.114.0 で公開 Ironclad チェックポイントの観測形式が既定で合わない

| 項目 | 内容 |
|---|---|
| 種別 | バグ報告（互換性） |
| 対象 | 上流 `game-v0.114.0`（2026-11-25 の公開ベータ向け適応） |
| 環境 | ゲーム v0.114.0（Steam public-beta, buildid 26909018）、WSL2、Python 3.12 |
| 根拠 | `weights/MANIFEST_v6.json` の `ironclad_ema_v24782.pt` / `ironclad_raw_v24782.pt` は **`obs_format: 28`**。一方 `game-v0.114.0/engine/Sts2Headless/Engine/Obs/Format.cs` の `FormatFor(Ironclad)` は 2026-11-20 以降 **29**（`FormatAll`） |
| 実際の動作 | 実機ブリッジで、フレームが 29・重みが 28 になり推論が拒否: `観測フレームの形式が重みと違います（フレーム 29 / 重み 28）`。`--character auto`（形式を送らない経路）で発生し、キャラを固定（形式 28 を明示）すると解消 |
| 期待 | 公開チェックポイントを既定のまま動かせる、または読み込み時に形式の不一致を検出して明示的に止まる |
| 最小修正案 | (1) `FormatFor(Ironclad)` を 28 に戻す、または (2) チェックポイントの `obs_format` を読み、reset の `"obs_format"` に反映する（`tools/ability_test.py` / `sts2env` 側で自動）、または (3) 不一致時に明示エラー |
| 再現 | `python -m local play --character IRONCLAD` 相当を v0.114.0 で、形式を指定せず実行 |
| 状態 | 再現確認（フォーク側の計測で発生）。上流そのものでの再現は未実施 |

英語草案（Issue タイトル/本文の骨子）:

> **v0.114.0: default Ironclad observation format (29) does not match the released checkpoint (28)**
> The released `ironclad_ema_v24782.pt` / `ironclad_raw_v24782.pt` declare `obs_format: 28`
> (`weights/MANIFEST_v6.json`), while `FormatFor(Ironclad)` in `game-v0.114.0` returns 29 since
> 2026-11-20. Running the released checkpoint without an explicit format produces format-29 frames
> that the weights reject. Suggest either defaulting `FormatFor(Ironclad)` to 28 for the released
> checkpoint, applying the checkpoint's `obs_format` on reset, or failing with a clear message.

## 候補 B: `StopAutopilot()` が保留中の要求を返さず、ゲームが固まる

| 項目 | 内容 |
|---|---|
| 種別 | バグ報告（堅牢性） |
| 対象 | spire-mentor の `SimAutoSlayer.Stop()`（`StopAutopilot()` 経由） |
| 根拠 | コードコメントに「Do not wait for the run task to unwind: while the selector scope is alive every card choice still blocks on the agent.」とある。実際、外部エージェントが返答を止めると、ハンドラは `await` で待ち続け、キャンセルが観測されず**画面が戻らない**（実機で確認: 手動切り替え直後にゲームが停止） |
| 期待 | 停止時に保留中の ask をキャンセル/失敗させ、ゲームを速やかに人に返す |
| 最小修正案 | `AskAsync` に `Stop` からのキャンセレーショントークンを渡し、停止時に `OperationCanceledException` で抜ける（既に `ct` は渡っているので、停止時に確実にキャンセルが観測されるよう、待機を `ct.WaitHandle` で待つ等） |
| 回避策（フォーク側） | 停止前に保留中の返答を 1 つ送る（`handed the game over to the player`） |
| 状態 | 実機で再現（停止→固まる）。上流での再現は未実施 |

## 候補 C: 進行中のランに autopilot を付け直す API（機能提案 / パッチあり）

| 項目 | 内容 |
|---|---|
| 種別 | 機能提案（小さな追加） |
| 内容 | `SimAutoSlayer.StartAttached()` と `MainFile.StartAutopilotAttached()` を追加。メニューを経由せず、いまの `RunState` に autopilot を付け直す（新しいランを始めない） |
| 根拠 | 現状 `StopAutopilot()` 後の復帰は「メニューから新しいラン」しかない。手動で遊んだ後に同じランへ戻せない |
| 実装 | フォーク `tonbiattack/spire-mentor` PR #3（マージ済み、`1fabf2a`）。`StartAttached()` は `PlayRoomsAsync` を直接走らせ、`run_config`/`start_run` の交渉はしない |
| 検証 | 実機で「手動で遊ぶ → AI に戻す」を通し、同じランでモデルが判断を再開することを確認 |
| 状態 | フォークに実装済み。上流へ PR を出すかは未承認 |

## 還元しないもの（記録）

- 独自 MOD・MentorClient・ランチャーの不具合は上流の対象外（#66 の対象範囲どおり）
- v0.114.0 のヘッドレス selfcheck 通過・勝敗は「こちらの計測結果」であり、上流の主張ではないため報告しない

## 状態の管理

| 候補 | 状態 | リンク |
|---|---|---|
| A（形式不一致） | **投稿済み**（2026-11-26） | hmty-CHEN/AlphaDefect-Public#1 |
| B（停止で戻らない） | **投稿済み**（2026-11-26） | tfriedel/spire-mentor#53 |
| C（attach API） | **上流が独自実装で採用**（PR は上流版に置き換えてクローズ）。フォークは上流 main に同期済み | tfriedel/spire-mentor#54 |
| B'（attach 後の map タイムアウト） | **投稿済み**（2026-11-27、再現 2 回） | tfriedel/spire-mentor#55 |

### 2026-11-27 の追記

- B（停止で固まる）は上流が**真因を特定して修正**（`7f0fa3b1`）: `SimAgentGate.RequestActionAsync` のハートビートがトークンを見ておらず、停止後も 1 コアで回り続けて Not Responding になっていた。修正後は停止時に確実に戻る（10/10 の F8 で確認と上流が報告）
- C（attach）は上流がより完全な実装（`MainFile.AttachAutopilot()` / `spirementor attach`、`AttachRefusal` で理由を返す）を採用。こちらの PR はその版に置き換えられてクローズ
- こちら側の対応: フォークを上流 main に同期（`a7d5f4d2`）、SpireMentor を再ビルドして配置、MOD は `AttachAutopilot()` を優先して呼び（旧 `StartAutopilotAttached` はフォールバック）、手動切り替え時の「返答を 1 つ送る」処理を削除（上流修正で不要に）
- 新たに見つけた不具合: attach 直後に map ハンドラがタイムアウトしてランが中断（#55 として報告）

