#!/usr/bin/env python3
"""AlphaDefect ランチャー（Windows 用 GUI。Tk。追加インストール不要）。

- 実ゲームの AI 観戦は Spire Mentor 経由を標準（docs/ja/policy.md）
- 起動は必ず「前のセッションを止めてから」行う（連続で押しても固まらない）
- 実行ログはそのまま画面に流す
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

MODES = [
    ("mentor", "Mentor 観戦（推奨）"),
    ("ai", "AI 観戦（独自 MOD）"),
    ("normal", "通常プレイ（MOD なし・ゲームだけ起動）"),
]
PACINGS = ["0", "500", "1500", "3000"]
DEFAULT_REPO = "/home/teni2/AlphaDefect-Public"


def repo_path() -> str:
    """このスクリプトの場所から WSL 側のパスを作る（\\\\wsl.localhost\\Ubuntu\\home\\... → /home/...）。"""
    here = Path(__file__).resolve().parent
    text = str(here).replace("\\", "/")
    m = re.match(r"//wsl(?:\.localhost)?/[^/]+/(.+)$", text, re.IGNORECASE)
    if m:
        return "/" + m.group(1)
    m = re.match(r"([A-Za-z]):/(.+)$", text)
    if m:
        drive, rest = m.group(1).lower(), m.group(2)
        return f"/mnt/{drive}/{rest}"
    return DEFAULT_REPO


REPO = repo_path()


def wsl(command: str, *, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl.exe", "--cd", REPO, "--", "bash", "-lc", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AlphaDefect ランチャー")
        self.geometry("820x620")
        self.minsize(720, 520)

        self.queue: queue.Queue[str] = queue.Queue()
        self.proc: subprocess.Popen[str] | None = None
        self.running_mode: str | None = None

        self.mode = tk.StringVar(value="mentor")
        self.character = tk.StringVar(value="auto")
        self.ascension = tk.IntVar(value=0)
        self.seed = tk.StringVar()
        self.random_seed = tk.BooleanVar(value=True)
        self.speed = tk.DoubleVar(value=1.0)
        self.pacing = tk.StringVar(value="1500")
        self.runs = tk.IntVar(value=1)
        self.no_mods = tk.BooleanVar(value=False)

        self._build()
        self._load_characters()
        self.after(200, self._drain)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.log(f"リポジトリ（WSL）: {REPO}")

    # ---------------------------------------------------------------- UI
    def _build(self) -> None:
        pad = {"padx": 12, "pady": 6}
        frm = ttk.LabelFrame(self, text="モード")
        frm.pack(fill="x", **pad)
        for i, (key, label) in enumerate(MODES):
            ttk.Radiobutton(frm, text=label, value=key, variable=self.mode,
                            command=self._on_mode).grid(row=0, column=i, sticky="w", padx=10, pady=8)

        cfg = ttk.LabelFrame(self, text="設定")
        cfg.pack(fill="x", **pad)
        ttk.Label(cfg, text="キャラ").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.cb_char = ttk.Combobox(cfg, textvariable=self.character, width=18, state="readonly")
        self.cb_char.grid(row=0, column=1, sticky="w", pady=6)
        ttk.Label(cfg, text="アセンション").grid(row=0, column=2, sticky="w", padx=8)
        ttk.Spinbox(cfg, from_=0, to=10, textvariable=self.ascension, width=5).grid(row=0, column=3, sticky="w")
        ttk.Label(cfg, text="シード").grid(row=0, column=4, sticky="w", padx=8)
        ttk.Entry(cfg, textvariable=self.seed, width=14).grid(row=0, column=5, sticky="w")
        ttk.Checkbutton(cfg, text="乱数", variable=self.random_seed).grid(row=0, column=6, sticky="w", padx=6)

        ttk.Label(cfg, text="速度").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(cfg, from_=0.5, to=10.0, increment=0.5, textvariable=self.speed, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(cfg, text="行動間隔（ms）").grid(row=1, column=2, sticky="w", padx=8)
        ttk.Combobox(cfg, textvariable=self.pacing, values=PACINGS, width=8, state="readonly").grid(row=1, column=3, sticky="w")
        ttk.Label(cfg, text="ラン数").grid(row=1, column=4, sticky="w", padx=8)
        ttk.Spinbox(cfg, from_=1, to=20, textvariable=self.runs, width=5).grid(row=1, column=5, sticky="w")

        btn = ttk.Frame(self)
        btn.pack(fill="x", **pad)
        self.b_start = ttk.Button(btn, text="起動", command=self.start)
        self.b_stop = ttk.Button(btn, text="停止", command=self.stop)
        self.b_status = ttk.Button(btn, text="状態", command=self.status)
        self.b_pause = ttk.Button(btn, text="一時停止/再開", command=lambda: self.control("pause-toggle"))
        self.b_step = ttk.Button(btn, text="1 手", command=lambda: self.control("step"))
        self.b_manual = ttk.Button(btn, text="手動（推奨のみ）", command=lambda: self.control("manual"))
        self.b_handover = ttk.Button(btn, text="手動で遊ぶ", command=self.handover)
        self.b_close = ttk.Button(btn, text="ゲームを閉じる", command=self.close_game)
        self.b_return = ttk.Button(btn, text="AI に戻す", command=self.ai_return)
        for b in (self.b_start, self.b_stop, self.b_status, self.b_pause, self.b_step,
                  self.b_manual, self.b_handover, self.b_close, self.b_return):
            b.pack(side="left", padx=4)

        ttk.Checkbutton(self, text="MOD を外して起動（元のセーブで遊ぶ）",
                        variable=self.no_mods).pack(anchor="w", padx=14)

        out = ttk.LabelFrame(self, text="ログ")
        out.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(out, height=14, wrap="none", font=("Consolas", 9))
        scroll = ttk.Scrollbar(out, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=scroll.set)
        self.txt.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._on_mode()

    def _load_characters(self) -> None:
        def work() -> None:
            try:
                r = wsl("./play.sh mentor --choices", timeout=90)
                data = json.loads(r.stdout.strip().splitlines()[-1])
                items = ["auto"]
                for c in data.get("characters", []):
                    items.append(c["id"] if c.get("status") == "validated" else f"{c['id']}（未検証）")
                items.append("select")
                self.queue.put(f"__chars__{json.dumps(items)}")
            except Exception as exc:  # noqa: BLE001 - 失敗しても既定で動かす
                self.queue.put(f"キャラ一覧の取得に失敗（既定を使います）: {exc.__class__.__name__}")
        threading.Thread(target=work, daemon=True).start()

    def _on_mode(self) -> None:
        mentor = self.mode.get() == "mentor"
        state = "readonly" if mentor else "disabled"
        self.cb_char.configure(state="readonly")
        for w in (self.b_pause, self.b_step, self.b_manual):
            w.configure(state="normal" if mentor else "disabled")

    # -------------------------------------------------------------- logging
    def log(self, text: str) -> None:
        self.txt.insert("end", text + "\n")
        self.txt.see("end")

    def _drain(self) -> None:
        try:
            while True:
                line = self.queue.get_nowait()
                if line == "__done__":
                    self._busy(False)
                    continue
                if line.startswith("__chars__"):
                    items = json.loads(line[len("__chars__"):])
                    self.cb_char.configure(values=items)
                    self.character.set(items[0])
                    continue
                self.log(line)
        except queue.Empty:
            pass
        self.after(200, self._drain)

    # -------------------------------------------------------------- actions
    def _busy(self, busy: bool) -> None:
        self.b_start.configure(state="disabled" if busy else "normal")
        self.b_stop.configure(state="normal" if busy else "disabled")

    def _stop_everything(self, quiet: bool = False) -> None:
        """前のセッションを確実に止める（起動し直しで固まらないように）。"""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self.proc = None
        if not quiet:
            self.log("前のセッションを停止します")
        try:
            wsl("./play.sh launch --stop >/dev/null 2>&1; pkill -f 'local mentor'; "
                "pkill -f 'local autopilot'; true", timeout=60)
        except Exception as exc:  # noqa: BLE001
            self.log(f"停止処理でエラー: {exc.__class__.__name__}")
        try:
            marker = Path(r"\\wsl.localhost\Ubuntu" + REPO +
                          "/mods/../game-mod/.mentor.txt")
            _ = marker  # マーカーは CLI が管理する（ここでは触らない）
        except Exception:  # noqa: BLE001
            pass

    def _spawn(self, command: str, label: str) -> None:
        self.log(f"{label}: {command}")
        creation = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            ["wsl.exe", "--cd", REPO, "--", "bash", "-lc", command],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=creation,
        )

        def pump() -> None:
            assert self.proc is not None
            for line in self.proc.stdout or []:
                self.queue.put(line.rstrip())
            code = self.proc.wait()
            self.queue.put(f"（終了: exit {code}）")
            self.queue.put("__done__")
        threading.Thread(target=pump, daemon=True).start()

    def start(self) -> None:
        mode = self.mode.get()
        char = self.character.get().replace("（未検証）", "")
        untested = "（未検証）" in self.character.get()
        seed = "" if self.random_seed.get() else self.seed.get().strip()
        self._busy(True)
        self.log("起動準備: 前のセッションを止めます")
        threading.Thread(target=self._start_worker, args=(mode, char, untested, seed), daemon=True).start()

    def _start_worker(self, mode: str, char: str, untested: bool, seed: str) -> None:
        # 前のセッションを止める → ゲームが残っていたら安全に閉じる → 起動
        self.queue.put("前のセッションを片付けます")
        self._stop_everything(quiet=True)
        try:
            r = wsl("./play.sh launch --game-running", timeout=60)
            if (r.stdout or "").strip().endswith("running"):
                self.queue.put("ゲームが残っているので安全に閉じます")
                r = wsl("./play.sh launch --close-game", timeout=120)
                for line in (r.stdout or "").splitlines():
                    self.queue.put(line)
        except Exception as exc:  # noqa: BLE001
            self.queue.put(f"片付けでエラー: {exc.__class__.__name__}")
        time.sleep(1.0)
        try:
            if mode == "normal":
                self.queue.put("通常プレイ: AI を止めてからゲームだけ起動します")
                r = wsl("./play.sh launch --safe-mode", timeout=90)
                for line in (r.stdout or "").splitlines():
                    self.queue.put(line)
                if self.no_mods.get():
                    r = wsl("./play.sh mod off", timeout=120)
                    for line in (r.stdout or "").splitlines():
                        self.queue.put(line)
                    self.queue.put("MOD を外しました（元のプロファイルで起動します）")
                else:
                    r = wsl("./play.sh mod on", timeout=120)   # 前回外していたら戻す
                    for line in (r.stdout or "").splitlines():
                        self.queue.put(line)
                subprocess.run(["cmd.exe", "/c", "start", "", "steam://rungameid/2868840"],
                               capture_output=True, text=True)
                self.queue.put("__done__")
                return
            r = wsl("./play.sh mod on", timeout=120)   # MOD を外していたら戻す
            for line in (r.stdout or "").splitlines():
                if "戻しました" in line:
                    self.queue.put(line)
            if mode == "ai":
                cmd = f"./play.sh launch ai --character '{char}' --speed {self.speed.get()} --pacing {self.pacing.get()}"
                if seed:
                    cmd += f" --seed '{seed}'"
                self._spawn(cmd, "AI 観戦")
                return
            cmd = f"./play.sh mentor --character '{char}' --ascension {self.ascension.get()} --runs {self.runs.get()}"
            if seed:
                cmd += f" --seed '{seed}'"
            if untested:
                cmd += " --allow-untested"
            self._spawn(cmd, "Mentor 観戦")
        except Exception as exc:  # noqa: BLE001
            self.queue.put(f"起動に失敗: {exc.__class__.__name__}: {exc}")
            self.queue.put("__done__")

    def stop(self) -> None:
        self.log("停止します（ゲームは閉じません）")
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        self._stop_everything()
        self.queue.put("停止しました")
        self.queue.put("__done__")

    def status(self) -> None:
        def work() -> None:
            try:
                r = wsl("./play.sh launch --status", timeout=90)
                self.queue.put((r.stdout or "").strip() or "(状態を取得できません)")
            except Exception as exc:  # noqa: BLE001
                self.queue.put(f"状態の取得に失敗: {exc.__class__.__name__}")
        threading.Thread(target=work, daemon=True).start()

    def ai_return(self) -> None:
        """手動から AI に戻す（いまのランのまま）。"""
        self.log("AI に戻します")

        def work() -> None:
            try:
                r = wsl("./play.sh launch --ai-return", timeout=60)
                for line in (r.stdout or "").splitlines():
                    self.queue.put(line)
            except Exception as exc:  # noqa: BLE001
                self.queue.put(f"AI に戻せませんでした: {exc.__class__.__name__}")
        threading.Thread(target=work, daemon=True).start()

    def close_game(self) -> None:
        """ゲームを安全に閉じる（AI を止めて正常終了 → だめなら強制）。"""
        self.log("ゲームを閉じます")
        self._busy(True)

        def work() -> None:
            try:
                r = wsl("./play.sh launch --close-game", timeout=120)
                for line in (r.stdout or "").splitlines():
                    self.queue.put(line)
            except Exception as exc:  # noqa: BLE001
                self.queue.put(f"終了に失敗: {exc.__class__.__name__}")
            finally:
                self.queue.put("__done__")
        threading.Thread(target=work, daemon=True).start()

    def handover(self) -> None:
        """AI を降りて、そのまま自分で操作する（spire-mentor の autopilot も止める）。"""
        self.log("手動に切り替えます（AI を停止して人に渡します）")

        def work() -> None:
            try:
                r = wsl("./play.sh launch --handover", timeout=60)
                for line in (r.stdout or "").splitlines():
                    self.queue.put(line)
                self.queue.put("終わりました。ゲームの画面をそのまま操作できます。")
            except Exception as exc:  # noqa: BLE001
                self.queue.put(f"切り替えに失敗: {exc.__class__.__name__}")
        threading.Thread(target=work, daemon=True).start()

    def control(self, what: str) -> None:
        command = {"pause-toggle": "pause-toggle", "step": "step", "manual": "manual"}[what]

        def work() -> None:
            try:
                r = wsl(f"./play.sh launch --{command}", timeout=60)
                self.queue.put((r.stdout or "").strip() or "(制御できません)")
            except Exception as exc:  # noqa: BLE001
                self.queue.put(f"制御に失敗: {exc.__class__.__name__}")
        threading.Thread(target=work, daemon=True).start()

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askokcancel("終了", "実行中のセッションを停止して終了しますか？"):
                return
            self._stop_everything(quiet=True)
        self.destroy()


def main() -> int:
    app = Launcher()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
