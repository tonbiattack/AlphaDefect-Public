# Windows ワンクリックランチャー（通常プレイ / AI 観戦）

Windows から**ダブルクリック**で、ふつうに遊ぶモードと、AI に打たせて観戦するモードを選べます。
毎回 WSL2 でコマンドを打つ必要はありません。

```text
リポジトリ\launch.cmd           ← これをダブルクリック（メニューが出ます）
```

メニューは「AI 観戦」と「通常プレイ」の 2 つです。**助言専用モード**（#2 / #3）はまだ実装されて
いないので出していません。

メニューの 1 / 2 でモードを選びます。ショートカットを作っておけば、デスクトップやスタート
メニューから 1 クリックで起動できます。

```bash
./play.sh launch --install-shortcuts    # デスクトップとスタートメニューに 2 つのショートカットを作る
```

| ショートカット | 動作 |
|---|---|
| **AlphaDefect AI 観戦** | MOD を確認（必要ならビルド）→ WSL の推論サーバ起動 → Steam 経由でゲーム起動 → AI が観戦プレイ |
| **AlphaDefect 通常プレイ** | 推論を起動せず、`--nomods` でゲームを起動（MOD は読み込まれない） |

---

## 1. 何をするか

```text
launch.cmd ─ メニュー ─┐
                      ├─ AI 観戦:  推論サーバ起動（18765/18766）→ 準備完了を待つ → Steam でゲーム起動
                      │             → MOD が接続してラン開始 → ゲーム終了でサーバだけ停止
                      └─ 通常プレイ: --nomods 付きでゲーム起動（サーバは起動しない）
```

- ゲームは **`steam.exe -applaunch 2868840 <引数>`** で起動します。実行ファイルを直接起動すると
  Steamworks が初期化されず（ログに `Steamworks is not initialized`）、プロファイルも既定の
  ものになってしまうためです。`-applaunch` なら Steam 経由のまま、この起動だけに `--nomods`
  を渡せます。**Steam 側の起動オプション（プロパティ）は書き換えません。**
- モードの判定・起動内容は次のコマンドで確認できます（実行はしません）。

```bash
./play.sh launch normal --dry-run
./play.sh launch ai --character auto --speed 2 --seed ABC123 --dry-run
```

---

## 2. 通常プレイ（MOD なし）

- 推論サーバは起動しません。ゲームは `--nomods` で起動するので、`mods/` に他の MOD があっても
  読み込まれません（**他の MOD を消したり、設定を書き換えたりはしません**）。
- セーブは通常のプロファイル（`…/SlayTheSpire2/steam/<id>/profile1`）です。
  MOD を入れて遊んだランは別プロファイル（`…/modded/profile1`）に保存されるので、進行データは混ざりません。

---

## 3. AI 観戦

```bash
./play.sh launch ai                      # キャラは auto（ゲーム側の選択に合わせる）
./play.sh launch ai --character DEFECT   # キャラを指定
./play.sh launch ai --speed 2            # 演出速度（Godot の TimeScale）
./play.sh launch ai --seed ABC123        # シード指定
./play.sh launch ai --no-autostart       # 自動でランを始めず、自分で開始する
```

流れ:

1. MOD の導入状態を確認し、**必要時だけ**ビルド・更新（版が変わっていれば入れ直し）
2. 推論サーバを起動（ログ: `~/.local/share/sls2/logs/launcher-*.log`）し、`/status` が応答するまで待つ
3. Steam 経由でゲームを起動 → メインメニューが出た時点でランを開始（**通常のラン開始**。デバッグの
   戦闘直行は使いません）
4. ゲームが終了したら、**このランチャーが起動したサーバだけ**を停止

観戦中は画面左上に「いま選んだ手・確率・上位候補」が出ます。

### 操作（一時停止・再開・停止・速度）

ランチャーを動かしたまま、別のターミナルから:

```bash
./play.sh launch --pause            # 一時停止
./play.sh launch --resume           # 再開
./play.sh launch --set-speed 3      # 速度変更
./play.sh launch --stop             # 推論サーバを止める（ゲームはそのまま）
./play.sh launch --status           # 状態
```

HTTP でも同じことができます（`curl` など）。

```bash
curl -s http://127.0.0.1:18766/status
curl -s -X POST -d '{"command":"pause"}' http://127.0.0.1:18766/control
```

---

## 4. 安全側の作り

| 状況 | 動作 |
|---|---|
| すでにゲームが動いている | **二重起動しない**（そのゲームはそのまま使えます） |
| ポート 18765 / 18766 が他のアプリに使われている | 起動せず、PID を示して案内。**他人のプロセスは止めません** |
| ランチャーの二重起動 | ロックファイル（`~/.local/share/sls2/state/launcher.lock`）で拒否 |
| Steam の起動オプションに `-nomods` がある | AI モードでは MOD が読まれないため、**事前に警告**（設定は書き換えません） |
| ゲームが起動してこない | 120 秒待ってエラー（サーバは自分で停止） |
| ランチャーを Ctrl-C | 自分が起動したサーバを停止 |

異常終了したときは `./play.sh launch --status` で状態を見て、必要なら `--stop` してください。

---

## 5. うまくいかないとき

| 症状 | 対処 |
|---|---|
| ショートカットが動かない | `./play.sh launch --install-shortcuts` で作り直す（`%LOCALAPPDATA%\AlphaDefect\launch-*.cmd` を作り、ショートカットはそれを指します） |
| `ゲームのプロセスが現れませんでした` | Steam が起動・ログインしているか確認。Steam の「プロパティ → 起動オプション」に `-nomods` が無いか確認 |
| `ポート 18765 が別のプロセスに使われています` | そのアプリ（以前の推論サーバなど）を終了するか、`./play.sh launch --stop` |
| 観戦が始まらない | `./play.sh launch --status` で「推論サーバ: 動いています」か確認。ゲーム側の MOD が無効になっていないか（設定 → Modding）確認 |
| 設定を作り直したい | `./play.sh mod install` で MOD を入れ直し（ゲームは終了しておく） |

---

## 6. 検証記録（2026-11-25）

WSL2 Ubuntu、ゲーム v0.111.1。

| 確認 | 結果 |
|---|---|
| AI モード（`launch ai`） | サーバ起動 → 準備完了 → Steam 経由でゲーム起動 → MOD 接続・ラン開始（64 意思決定） |
| ゲーム終了時の後始末 | 「ゲームが終了しました」→ サーバ停止 → ポート解放・プロセス残り無し・ロック解除 |
| 通常モード（`launch normal`） | `--nomods` で起動。MOD 読み込みログ 0 件、プロファイルは `steam/<id>/profile1`、サーバ無し |
| デスクトップのショートカット | 「AlphaDefect AI 観戦.lnk」をダブルクリック相当で実行 → サーバ + ゲーム + MOD 接続を確認 |
| 二重起動 | ゲーム起動中は「ゲームが既に動いています」で起動しない |
| ポート競合 | 他プロセスが 18765 を持つときは起動せず、そのプロセスは止めない |
| 制御 | `pause` / `resume` / `stop` / `speed` を HTTP と CLI の両方で確認 |

未検証: ショートカットを**実際にマウスでダブルクリック**する操作（同じ .lnk を ShellExecute で
起動して確認）。Windows のスタートメニュー検索への反映には時間がかかることがあります。

## GUI ランチャー（推奨）

`launcher.cmd` をダブルクリックすると、**GUI のランチャー**が開きます（PowerShell + WinForms。追加インストール不要）。

- **モード**: 通常プレイ / AI 観戦（独自 MOD）/ Mentor 観戦（spire-mentor）
- **設定**: キャラ（auto / DEFECT / REGENT / SILENT / select）、アセンション 0〜10、シード（乱数可）、速度、間隔（0 / 500 / 1500 / 3000 ms）、ラン数
- **ボタン**: 起動 / 停止 / 状態 / 一時停止・再開 / 1 手 / 手動（推奨のみ）
- 下の欄に実行ログと状態が出ます

デスクトップから使う場合は、`launcher.cmd` のショートカットを作るのが簡単です。

```powershell
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut("$env:USERPROFILE\Desktop\AlphaDefect ランチャー.lnk")
$sc.TargetPath = "\\wsl.localhost\Ubuntu\home\teni2\AlphaDefect-Public\launcher.cmd"
$sc.WorkingDirectory = "\\wsl.localhost\Ubuntu\home\teni2\AlphaDefect-Public"
$sc.Save()
```
