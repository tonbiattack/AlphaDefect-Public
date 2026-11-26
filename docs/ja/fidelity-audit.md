# 実ゲーム連携の忠実性監査（#68）

モデルの判断が、実ゲーム連携の都合でどこまで曲げられているかを 1 か所にまとめた記録です。
対象: `game-mod/AlphaDefectAutopilot/MentorClient.cs`（Mentor 連携）、`CombatDriver.cs`（独自 MOD）、
`local/bridge/agent.py`（推論サーバ）。

> 状態: **監査の初版**。表は実装の実態に基づく。実測の不足は「未測定」と書く。

## 1. 意思決定種別ごとの対応

| 種別 | Mentor の要求 | こちらの候補 | 実行 | モデル判断か |
|---|---|---|---|---|
| 戦闘の出牌・対象 | `combat_action`（hand/targets） | `CombatDriver.BuildCombatDecision` | `play`/`end_turn`/`potion` | **モデル** |
| ポーション | `combat_action` の一部 | 同上（`UsablePotions`） | `potion`（slot） | **モデル**（#67 で Mentor では有効化） |
| 報酬画面 | `reward_screen` | `LiveRewards.Build` | `choose`（受け取り／進む） | **モデル** |
| カード報酬 | `card_reward` | 要求の `cards[]` から生成 | `choose`/`skip` | **モデル**（スキップ再発時のみルール、下記） |
| カード選択（1 枚） | `card_select` | 要求の `cards[]` | `choose index` | **モデル** |
| カード選択（複数枚） | `card_select`（min_select>1） | 要求の `cards[]` | `choose indices` | **モデル（1 枚目）+ 残りは候補順** |
| 店 | `shop` | `BuildShopDecisionForMentor` | `choose` | **モデル** |
| 経路 | `map_select` | `BuildMapDecision` | `index` | **モデル** |
| イベント | `event` | `OutOfCombat.BuildEventDecisionForMentor` | `choose index` | **モデル** |
| 休息所 | `rest_site` | `BuildRestDecision` | `choose index` | **モデル** |
| 宝箱 | `treasure` | 要求の `options[]` | `choose index` | **モデル** |
| カードパック | `card_bundle` | 要求の `bundles[]` | `choose index` | モデル（ラベルはパック番号と枚数のみ） |
| 水晶球（ミニゲーム） | `crystal_sphere` | なし | `index`（proceed/先頭セル） | **ルール**（upstream もモデル化していない） |

## 2. ルールで上書きしている箇所（すべてログに理由を出す）

| 箇所 | 内容 | ログ |
|---|---|---|
| カード報酬の再表示 | スキップ後に同じ画面が開き直したら、2 回目はスキップせず先頭カードを取る | `card reward re-opened after a skip; taking the first card instead` |
| 複数枚の選択 | 1 枚目はモデル、残りは候補の並び順で埋める | `card_select needs N cards: asking the model for the first pick` |
| 水晶球 | proceed があればそれ、無ければ先頭セル。同じ state を繰り返す場合はセルを巡回 | `crystal_sphere: rule-based answer` / `same state asked N times` |
| 推論タイムアウト | 種別に応じた既定（戦闘 end_turn / カード報酬 skip / その他 先頭） | `inference timeout; answering with a legal default for <kind>` |
| 未対応種別 | 合法な既定 | `<kind> has no model decision yet (n); replying with a legal default` |
| 古い応答 | 破棄 | `dropped a stale action` |
| 変換不能 | 種別に応じた既定 | `translate failed: ...` |

## 3. 忠実性の集計（実装済み）

ラン終了時に 1 行で出します（ゲームログ・`mod-log`）。

```
[mentor] fidelity: model[combat_action=41,card_reward=4,event=2] rule[card_select=1] rejects=0 timeouts=0 stale=0
```

- `model[...]`: モデルの判断をそのまま実行した回数（種別ごと）
- `rule[...]`: ルールで代替した回数（種別ごと）
- `rejects` / `timeouts` / `stale`: 拒否・タイムアウト・古い応答の破棄

これで「モデル判断忠実実行率」「代替処理率」を、回数制限付きログではなく実数で測れます。

## 4. 未確認・要検証（#68 のチェックリスト対応）

| 項目 | 状態 |
|---|---|
| 同名カード・強化済みカードの取り違え（`ResolveCard`/`CardIndexInMessage` は ID 一致の先頭） | **未検証**（回帰テスト未作成） |
| 店・報酬の index 変換（失敗時 0） | **未検証** |
| 敵の列挙順と `target_index` の対応 | 実機で動作（不一致の報告なし）。**体系的な比較は未実施** |
| カードパックの候補（パック番号と枚数のみ） | 観測エンコーダ側で内容が補われるか**未確認** |
| REGENT 固有（スター・生成カード・ソヴリンブレード） | 実機で動作。**ヘッドレスとの差分比較は未実施** |
| greedy/merge 設定と upstream 評価の差 | **未確認** |
| ポーション使用後の選牌（POTION_OF_BINDING 等）を Mentor で処理できるか | **実機確認中**（#67）。Mentor では有効化した |
| 独自 MOD のポーション | 既定で無効のまま（選牌停止の再発防止）。Mentor とは別扱い |

## 5. 参照

- #67（ポーション）、#28（ヘッドレス差分）、#22（計測）、#33（Mentor 統合）
- `docs/ja/mentor.md`（Mentor 連携の設定・観戦）
