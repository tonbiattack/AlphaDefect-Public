# 第三者コード・データの扱い（#33 / #29）

このリポジトリは、実ゲーム操作の実験で外部の OSS を**利用**しますが、**コードを同梱しません**。
ここに、何を使い、どのライセンスで、何を同梱しないかを記録します。

## 利用している外部ソフトウェア

| 名前 | 用途 | ライセンス | 同梱 | 出典 |
|---|---|---|---|---|
| [Spire Mentor](https://github.com/tfriedel/spire-mentor) | 実ゲームの画面操作・状態同期（mentor モード） | MIT | しない（各自インストール） | リポジトリの `spire-mentor-mod/` |
| [BaseLib](https://github.com/Alchyr/BaseLib-StS2) | Spire Mentor が依存する MOD 基盤 | MIT | しない（各自インストール） | リリースの `BaseLib.dll` / `.pck` / `.json` |
| Harmony（Lib.Harmony） | 独自 MOD のパッチ | MIT | しない（ゲーム同梱/参照） | — |

### 同梱しないもの（重要）

- **ゲーム由来のデータ**: カード・遺物・敵の画像、音声、ゲーム本体のアセンブリ、抽出データ。
  リポジトリには**含めません**（`.gitignore` と `.gitattributes` で除外）。実行時に各自のゲームから読みます。
- **Spire Mentor / BaseLib のコード**: 本リポジトリのコミットには含めません。
  インストールは利用者が自分の環境で行います（`docs/ja/oss-operation-layer.md` の手順）。
- **学習済み重み**: リポジトリの方針に従い、重みは MANIFEST とハッシュのみ（配布は Releases）。

## 参照（コードを読んだだけ）の記録

以下は**読み取りのみ**で、コードのコピーは行っていません。

- Spire Mentor の `PROTOCOL.md`（メッセージ仕様）、`SimCardRewardScreenHandler.cs` /
  `BridgeServer.cs`（`request_id` の文字列要件など、仕様の確認のため）
- 調査記録は `docs/ja/oss-operation-layer.md`

## もし移植する場合のルール

- MIT のコードを移植する場合は、**著作権表示とライセンス文を残す**
  （`LICENSES/` 配下に原文を置き、移植したファイルの先頭に出典を書く）。
- ゲーム本体の逆コンパイル結果・アセットは**移植しない**。
- 移植した範囲と理由を `docs/ja/oss-operation-layer.md` に追記する。

## ライセンス（本リポジトリ自身）

本リポジトリは **MIT**（`LICENSE`）。学習済みモデル・重みの扱いは `LICENSE` と README の記載に従います。
