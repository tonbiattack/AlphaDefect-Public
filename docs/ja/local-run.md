# 訓練済みモデルを自分の PC で動かす（Windows 11 / WSL2・CPU）

このページの手順だけで、**ゲームの画面を開かずに** AlphaDefect のネットワークへ A10（アセンション 10）を 1 ラン完走させ、結果と所要時間を見られます。GPU は要りません。再学習も外部 API も使いません。

```bash
git clone https://github.com/tonbiattack/AlphaDefect-Public.git
cd AlphaDefect-Public
./setup.sh                                              # 初回だけ（数百 MB のダウンロード）
./play.sh --character DEFECT --seed ABC123 --asc 10     # 1 局
```

Windows のエクスプローラーや PowerShell からは `setup.cmd` / `play.cmd` でも同じことをできます（中で WSL の `setup.sh` / `play.sh` を呼びます）。

---

## 1. できること・できないこと

| できる | 説明 |
|---|---|
| A10 を 1 局、CPU で完走 | ゲーム本体と同じルールの headless エンジンで、ネットワークが全判断をします。結果（勝敗・到達層・戦闘数・HP 増減）と所要時間を表示します |
| キャラ・シード・アセンション・探索の有無を指定 | `--character` / `--seed` / `--asc` / `--search` |
| 6 キャラ分の重みの取得と SHA-256 検証 | リリース資産を落として `weights/` の目録と照合します |
| 後続機能のための共通推論 API | `local/infer.py`（観測・合法手・推奨手・候補スコア） |

| できない（はっきり書いておく） | 理由 |
|---|---|
| ゲームの画面で人間と一緒に遊ぶ | 実機ブリッジ MOD（ライブ配信用）はこのリポジトリに入っていません |
| 学習（強化学習）を回す | 学習は 2 台の貸し出しマシン前提です（[docs/03](../03_system_and_infrastructure.md)） |
| v0.111.0 系での探索あり（PIMC） | 外部の [sts2-sim](https://github.com/iRyougi/sts2-sim) のビルドが要ります。この実行系には入れていません |
| GPU 推論 | CPU だけを検証しています（`--device cuda` は未検証） |

---

## 2. 必要なもの

| 項目 | 内容 |
|---|---|
| OS | Windows 11 + WSL2（Ubuntu 22.04 / 24.04）を想定。Linux 単体でも動きます |
| ゲーム | Steam の Slay the Spire 2（下の対応表の版） |
| ディスク | 約 2 GB（ゲーム資産 100 MB、重み 470 MB、.NET SDK 220 MB、PyTorch 300 MB、仮想環境） |
| Python | 3.12（`python3.12`） |
| ネット | 初回のみ（重み・.NET・PyTorch の取得） |
| .NET 9 SDK / uv | 無ければ `setup.sh` が `~/.dotnet` と `~/.local/bin` に入れます（管理者権限不要） |

ゲーム本体と重みは**このリポジトリに含まれません**。どちらも各自の環境に置きます。`setup.sh` はゲームのフォルダを読み取り専用で扱い、コピーとローカライズの抽出だけを行います。

---

## 3. 対応しているゲームのバージョン

ゲームは 2 つの系統に分かれており、系統ごとにコードのツリーが違います。`setup.sh` はインストールされている版を見て自動で選びます。**未対応の版は黙って別のツリーで走らせず、対応表を示して止まります。**

| ゲーム | 使うコード | 動かせるキャラ | 探索あり | 状態 |
|---|---|---|---|---|
| v0.113.1 | リポジトリ直下 | FOOL / IRONCLAD / NECROBINDER、および DEFECT・SILENT（論文 3.2 節・付録 E） | 対応 | 論文の表 3 の 3 キャラ |
| v0.111.0 | `game-v0.111.0/` | DEFECT / REGENT / SILENT | 非対応 | 論文の表 3・表 4 |
| v0.111.1 | `game-v0.111.0/` | DEFECT / REGENT / SILENT | 非対応 | v0.111.0 のパッチ互換（下記の実測で確認） |
| その他（v0.112.x、v0.113.0 など） | — | — | — | **未対応**（明示的に拒否） |

- v0.113.1 の REGENT は重みの系統が違うため「実績なし」扱いです。試すときだけ `--allow-untested` を付けてください。
- v0.111.1 をパッチ互換として扱っている根拠は [8. 検証記録](#8-検証記録2026-11-24) にあります（selfcheck の全項目一致、5 キャラ×12 局のスモーク、語彙の差分 0 件、A10 の完走）。
- Steam の「プロパティ → ベータ」で版を切り替えられます。どの版が入っているかは `./play.sh status` で分かります。

---

## 4. セットアップ（初回だけ）

```bash
./setup.sh
```

順に次を行います。**何度実行しても、済んでいる工程は飛ばします。**

1. ゲームのインストールを探し、`release_info.json` から版を読む（Steam のライブラリを全部回ります）
2. Python 3.12 の仮想環境 `.venv` を作り、numpy と PyTorch(CPU) を入れる
3. .NET 9 SDK を用意する（無ければ `~/.dotnet` に入れる）
4. `data_sts2_windows_x86_64/` のアセンブリと、`.pck` から取り出したローカライズをキャッシュに複製する
5. headless エンジンをビルドする
6. `selfcheck` を走らせる（エンジンがゲームのどの関数を叩いているかを全部確かめる）
7. 6 キャラ分の重みを取得し、SHA-256 を検証する

```text
$ ./setup.sh
ゲーム: /mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2
  バージョン v0.111.1（ブランチ v0.111.1, コミット cc1f9462）
  コード系統: v0.111.0（game-v0.111.0）
  PyTorch: 2.15.0+cpu（導入済み）
  .NET: /home/you/.dotnet/dotnet
  ゲーム資産: アセンブリ 206 ファイルを複製（rsync）
  エンジン: ビルド完了（.../Sts2Headless.dll）
  selfcheck: 通過
  DEFECT: student_e_v165018.pt を取得し SHA-256 を確認しました
  ...
準備完了。`python -m local play --character DEFECT --seed ABC123 --asc 10` で 1 局打てます。
```

### 主なオプション

| オプション | 意味 |
|---|---|
| `--characters DEFECT,SILENT` | 取得する重みを絞る（既定は **6 キャラ全部**。今の版で動かせない分も、Steam のベータを切り替えたときにそのまま使えます） |
| `--force` | 済んでいる工程もやり直す（重みは取り直し） |
| `--skip-selfcheck` | selfcheck を省く（ゲームを更新した直後は省かないこと） |
| `--no-bootstrap` | .NET や uv の自動導入をしない（自分で管理したいとき） |

### キャッシュの場所

```text
~/.local/share/sls2/
  game/v0.111.1/          ゲームのアセンブリと res/localization（エンジンが読む）
  weights/weights-v1/     取得した .pt（SHA-256 検証済み）
  runs/20261124-214719_DEFECT_ABC123_greedy/   実行ごとの結果とリプレイ
  state/                  スタンプ・導入スクリプト
```

`SLS2_HOME` で根を変えられます（`--home` でも同じ）。ゲームが自動検出できない場所にあるときは `SLS2_GAME_PATH`（または `--game-path`）でフォルダを直接指定します。

```bash
SLS2_GAME_PATH="/mnt/d/SteamLibrary/steamapps/common/Slay the Spire 2" ./play.sh status
```

---

## 5. 1 局打つ

```bash
./play.sh --character DEFECT --seed ABC123 --asc 10
```

実測の出力（このリポジトリを WSL2・CPU で動かしたもの）:

```text
DEFECT / A10 / 1 局 / 貪欲
出力: /home/you/.local/share/sls2/runs/20261124-214719_DEFECT_ABC123_greedy
cwd: .../game-v0.111.0
env: STS2_GAME_DIR=.../sls2/game/v0.111.1 DOTNET_ROOT=.../.dotnet
cmd: .../python -I tools/ability_test.py --ckpt .../student_e_v165018.pt --out ... \
     --asc 10-10 --runs 1 --seeds-file .../seeds.json --character DEFECT --envs 1 --device cpu --time-cap 600

  | 版本 165018，本分片 1 局待打
  | 21:47:31 进阶 10：1 局，通关 1，用时 11 秒

=== 結果（12.0 秒）===
[ABC123] 結果: 勝利（全 3 幕クリア）; 到達: 第 3 幕 49 層 / 戦闘 23 回; 最後の戦闘: Boss AEONGLASS_BOSS（第 3 幕 49 層）; HP 増減: -80%
リプレイ: .../A10/replays/v165018_ABC123.jsonl.gz
```

- 幕番号は 1 始まりに直して表示します（エンジン内部は 0 始まり）。
- 「HP 増減」は開始時 HP に対する増減（`-80%` は最大 HP の 8 割を失った、の意味）。
- 全 3 幕をクリアすると `勝利（全 3 幕クリア）`。途中で死ぬと `敗北（第 n 幕・m 層）`。
- リプレイは `replays/*.jsonl.gz`（評価ツールと同じ形式）。

### よく使う形

```bash
./play.sh --character SILENT --seeds AAA111,BBB222 --asc 10 --envs 2   # 2 局を並列
./play.sh --character IRONCLAD --runs 3                                # ランダムなシードで 3 局
./play.sh --character DEFECT --asc 20                                  # アセンションを変える
./play.sh --character DEFECT --seed ABC123 --json                      # 結果を JSON で
./play.sh --character DEFECT --seed ABC123 --dry-run                   # 実行するコマンドを見るだけ
```

### 探索あり（v0.113.1 のみ）

```bash
./play.sh --character FOOL --seed ABC123 --search --sims 32
```

BOSS・エリート戦だけ、局面のコピーを使った情報集約型の木探索（`tools/future/run_search.py`）で打ちます。`--sims` は 1 手あたりのシミュレーション回数で、増やすほど強く・遅くなります（既定 32）。**A10 固定**です（スクリプト側が 10 を直接書いているため）。v0.111.0 系で `--search` を付けると、理由を書いて止まります。

### CPU 時間の目安

| キャラ | 1 局 | 環境 |
|---|---|---|
| Defect（v0.111.0 系の重み） | 11 秒 | 20 コアの WSL2、`--envs 1` |
| Regent（同上） | 16 秒 | 同上 |
| Silent（同上） | 8 秒（2 局並列で 17 秒） | 同上 |

探索ありは 1 手あたり数十回のシミュレーションをするので、桁が変わります（`--sims` を下げるか、`--procs` を増やしてください）。

---

## 6. 共通推論 API

後続機能（推奨行動の表示、カード報酬の選択支援、プレイの比較・振り返り、自動プレイ）は、この API だけを使えば観測形式の違いを気にせずに済みます。

```python
from local.infer import LocalAgent

agent = LocalAgent("DEFECT")          # 重み・エンジン・ゲーム資産はキャッシュから自動で解決
try:
    obs = agent.reset("ABC123", ascension=10)
    print(obs.act_no, obs.floor, obs.hp, obs.decision)      # 1 3 60 event

    for act in agent.legal_actions():    # 合法手（候補）全部
        print(act.index, act.label, act.kind, round(act.prob, 3))

    best = agent.recommend()              # 推奨手（確率最大）
    print("推奨:", best.label, best.prob)

    agent.step(best.index)                # 1 手進める
    print("価値頭:", agent.value())       # この局面から先の見込み
finally:
    agent.close()
```

| メソッド | 返すもの |
|---|---|
| `reset(seed, ascension)` | `Observation`（幕・層・HP・gold・decision・戦闘中か・歩数・終了/勝利） |
| `observation()` | 同じ（`raw` にはエンジンの生の観測） |
| `legal_actions()` / `candidate_scores()` | `Action` の一覧（`index` `label` `kind` `key` `prob` `logit`）。`label` はゲーム自身が付ける名前 |
| `recommend()` | 確率最大の `Action`（`merge=True` なら同一カードをまとめた貪欲） |
| `step(index)` | 実行して次の `Observation` |
| `value()` | 価値頭の評価（探索・比較用） |
| `play(seed, ascension)` | 最後まで貪欲で打って結果 dict を返す（動作確認用） |

観測形式は重みの目録から自動で合わせます（例: Ironclad v24,782 は形式 28、Fool v24,181 は 27）。呼び出し側が `obs_format` を書く必要はありません。

手元ですぐ試すには:

```bash
./play.sh api --character DEFECT --steps 5
```

```text
--- 0 手目: event / 第 1 幕 1 層 / HP 60/75 / 候補 3
  * [0] NEOW.pages.INITIAL.options.LOST_COFFER  p=0.565  (event_option)
    [1] NEOW.pages.INITIAL.options.GOLDEN_PEARL  p=0.413  (event_option)
    [2] NEOW.pages.INITIAL.options.DOWSING_ROD  p=0.022  (event_option)
  推奨: [0] NEOW.pages.INITIAL.options.LOST_COFFER（p=0.565）
```

---

## 7. 仕組み（知りたい人向け）

- **2 つのコードツリー**を版で切り替えます（表 3）。同じ名前の `sts2env` / `sts2rl` を 1 プロセスに混ぜると取り違えるので、`play` は必ず別プロセスで、その版のツリーを cwd にして既存ツール（`tools/ability_test.py` / `tools/future/run_search.py`）を呼びます。論文の評価と同じ経路です。
- 観測形式は重みごとに違うので、`reset` の `--req-extra` に `obs_format` を入れて明示します（`local/recipes.py`）。
- ゲームの版は `release_info.json` を正とします。Steam の `appmanifest` は別アプリの内容になっている壊れた例があるため版判定には使いません。
- 重みの目録（どの版・キャラ・観測形式・SHA-256 か）は `weights/MANIFEST_*.json` と `weights/SHA256SUMS_*.txt` を読みます。目録と実ファイルのハッシュが食い違ったら取得し直します。
- このリポジトリへの変更は 2 か所だけです: `python/sts2env/env.py` と `game-v0.111.0/python/sts2env/env.py` の (1) `reset_run` に `**extra` を追加、(2) `dotnet` の場所を `STS2_DOTNET` → `~/.dotnet` → PATH の順で探す。どちらも既存の呼び出しを壊しません。

---

## 8. 検証記録（2026-11-24）

WSL2 Ubuntu（20 コア / 15 GB）、CPU のみ、ゲームは Steam の **v0.111.1**（コミット `cc1f9462`、2026-11-13）。

| 確認したこと | 結果 |
|---|---|
| `selfcheck`（v0.111.1 のアセンブリ + v0.111.0 のエンジン） | 通過。兼容层 145 か所・パッチ 60 か所すべて一致、語彙は現行ゲームの 2,262 キーを全部カバー、5 キャラ×12 局のスモークで異常 0・カードき 0 |
| 6 キャラ分の重みの取得と SHA-256 | すべて一致（`weights/` の目録と照合） |
| A10 を 1 局完走（Defect、`ABC123`） | 勝利（第 3 幕 49 層、23 戦）、11 秒 |
| A10 を 1 局完走（Regent、`ABC123`） | 勝利（第 3 幕 49 層、23 戦）、16 秒 |
| A10 を 2 局（Silent、`ABC123` / `ZZZZ9`） | 敗北（第 1 幕 17 層）/ 敗北（第 3 幕 49 層）、2 局で 17 秒 |
| 同じシードの再現性（Defect `ABC123` を 3 回） | 3 回とも同じ結果（勝利・49 層・23 戦） |
| 共通推論 API（観測・合法手・推奨手・スコア・価値頭） | 動作確認済み |
| ユニットテスト | 62 件すべて通過（`python -m unittest discover -s local/tests -t .`） |
| v0.113.1 系 | このマシンに v0.113.1 が無いため**1 局は未実行**。ただし (1) `tools/ability_test.py` と `tools/future/run_search.py` がこの環境で読み込めること、(2) `run_search.py` に `play.sh --search` と同じオプション一式を渡して受理されること（0 シードで 0 局）、(3) コマンド組み立てはユニットテスト、を確認しています |

v0.111.1 を v0.111.0 のパッチ互換として扱う判断の根拠は、上の selfcheck（ゲーム側の関数・内容がすべて見つかり、語彙の差分が 0 件）と、A10 の完走です。v0.111.0 と v0.111.1 でゲーム内容が変わっていれば、selfcheck は「缺」を出します。

---

## 9. 困ったとき

| 症状 | 原因と対処 |
|---|---|
| `Slay the Spire 2 が見つかりませんでした` | ゲーム未インストール、WSL からドライブが見えていない（`ls /mnt/c`）、または Steam ライブラリが自動検出できない場所。`SLS2_GAME_PATH` でフォルダを直接指定する |
| `ゲーム vX.Y.Z にはまだ対応していません` | 対応表（3 章）に無い版。Steam のベータで v0.113.1 か v0.111.0 に切り替える |
| `selfcheck が通りませんでした` | ゲームが更新されてエンジンの前提が変わった可能性。`python -m local versions` で対応版を確認し、対応版に切り替える |
| `.NET 9 SDK が見つかりません` | `--no-bootstrap` を付けていないか確認。手動なら `curl -sSL https://dot.net/v1/dotnet-install.sh \| bash -s -- --channel 9.0` |
| `Python 3.12 が見つかりません` / 仮想環境が作れない | `sudo apt install python3.12`（必要なら `python3.12-venv`）。`SLS2_PYTHON` で場所を指定してもよい |
| 重みの `SHA-256 が一致しません` | ダウンロードの破損。もう一度 `./setup.sh`（`.part` から再開します）。配信元を変えるなら `SLS2_RELEASE_REPO` |
| 実行が遅い | `--envs` を増やす（局数が多いとき）。CPU のコア数に合わせる。探索ありは `--sims` を下げる |
| 途中で「中断」になる | 上限（`--time-cap` 600 秒 / 6,000 歩）に当たった、またはエンジンが止まった。同じシードで再現するか確認し、`runs/.../stuck.jsonl` を見る |
| 結果が論文と少し違う | 貪欲は同じシード・同じ重み・同じバッチサイズなら再現しますが、端末が違うと稀に僅差の判断が入れ替わります（論文 付録 D） |

---

## 10. 既知の制約

- ゲーム資産と重みは再配布しません。各自の Steam のゲームと、公開リリースから取得してください。
- 探索ありは v0.113.1 系のみ・A10 固定です。v0.111.0 系の PIMC（表 3 の探索列）は `sts2-sim` のビルドが必要で、この実行系の対象外です。
- 実機（ゲーム画面）で動かすブリッジ MOD は未公開です。この実行系は headless エンジンだけを使います。
- `--device cuda` は未検証です。CPU で十分に速い（1 局 10 秒前後）ので、まず CPU で試してください。
