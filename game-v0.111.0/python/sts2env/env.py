"""引擎进程的封装。每个进程只能跑一局（游戏逻辑是单例），并行靠多进程。"""
from __future__ import annotations

import json
import os
import selectors
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .protocol import Obs, parse

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DLL = REPO / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"


def _find_dotnet() -> Path:
    """dotnet の場所: STS2_DOTNET（明示）→ ~/.dotnet（ローカル実行系が入れる所）→ PATH → システムの既定。"""
    for c in (os.environ.get("STS2_DOTNET"), Path.home() / ".dotnet" / "dotnet", shutil.which("dotnet"),
              "/usr/share/dotnet/dotnet", "/usr/lib/dotnet/dotnet"):
        if c and Path(c).is_file():
            return Path(c)
    return Path.home() / ".dotnet" / "dotnet"


DOTNET = _find_dotnet()
# Rust 引擎（rust-engine/，`sts2-serve frames` 与 C# 的 serve 同协议）；租用机上放 musl 静态版，用 STS2_RUST_BIN 指定
RUST_BIN = Path(os.environ.get("STS2_RUST_BIN", REPO / "rust-engine" / "target" / "release" / "sts2-serve"))
ENGINES = ("csharp", "rust")


class EngineError(RuntimeError):
    """引擎报告的错误（请求非法或游戏逻辑异常）。进程仍可继续使用。"""


def default_engine() -> str:
    """没有显式指定时用哪个引擎：环境变量 STS2_ENGINE（csharp / rust），默认 C#。"""
    return os.environ.get("STS2_ENGINE", "csharp")


class _Process:
    def __init__(self, dll: Path, log: Any, engine: str = "csharp"):
        if engine not in ENGINES:
            raise ValueError(f"未知引擎 {engine}（可选 {ENGINES}）")
        if engine == "rust":
            # Rust 引擎不支持候选文字（labels）、中途开局（act_start）与路线摘要（route_summary），见 frames.rs
            cmd, env = [str(RUST_BIN), "frames"], None
        else:
            cmd = [str(DOTNET), str(dll), "serve"]
            env = dict(os.environ, DOTNET_ROOT=str(DOTNET.parent), DOTNET_CLI_TELEMETRY_OPTOUT="1",
                       DOTNET_TieredPGO="1")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0)

    def send(self, req: dict) -> None:
        self.p.stdin.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        self.p.stdin.flush()

    def recv(self) -> bytes:
        head = self._read(4)
        (n,) = struct.unpack("<i", head)
        body = self._read(n)
        if n >= 4 and struct.unpack_from("<i", body)[0] == -1:
            raise EngineError(body[4:].decode("utf-8"))
        return body

    def _read(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            c = self.p.stdout.read(n - got)
            if not c:
                raise EngineError(f"引擎进程已退出（返回码 {self.p.poll()}）")
            chunks.append(c)
            got += len(c)
        return b"".join(chunks)

    def close(self) -> None:
        try:
            self.send({"op": "close"})
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class Sts2Env:
    """单个引擎进程。

    reset_run(character, ascension, seed)：整局模式，从 Neow 开始。
    reset_combat(spec)：战斗模式，按 CombatSpec 直接开一场战斗，战斗结束即 done。
    step(action)：action 为候选下标（0 .. obs.num_actions-1）。
    """

    def __init__(self, dll: Path = DEFAULT_DLL, labels: bool = False, log: Any = subprocess.DEVNULL, engine: str | None = None):
        self.engine = engine or default_engine()
        self._proc = _Process(dll, log, self.engine)
        self.labels = labels

    def reset_run(self, character: str, ascension: int, seed: str, snapshots: bool = False, **extra) -> Obs:
        # extra: そのまま reset 要求に並べる（obs_format、loop_cut、stall_turns など。Server.cs 冒頭の一覧）
        self._proc.send({"op": "reset", "mode": "run", "character": character, "ascension": ascension,
                         "seed": seed, "labels": self.labels, "snapshots": snapshots, **extra})
        return parse(self._proc.recv())

    def reset_combat(self, spec: dict) -> Obs:
        self._proc.send({"op": "reset", "mode": "combat", "spec": spec, "labels": self.labels})
        return parse(self._proc.recv())

    def step(self, action: int) -> Obs:
        self._proc.send({"op": "step", "action": int(action)})
        return parse(self._proc.recv())

    def vocab(self) -> list[str]:
        self._proc.send({"op": "vocab"})
        return self._proc.recv().decode("utf-8").split("\n")

    def player(self) -> dict:
        """当前玩家快照（{"snapshot": JSON 文本, "act", "floor"}），牌组前瞻用；不走一步。"""
        self._proc.send({"op": "player"})
        return json.loads(self._proc.recv().decode("utf-8"))

    def close(self) -> None:
        self._proc.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


class VecEnv:
    """N 个引擎进程。请求先全部发出、再依次读取，各进程并行计算。"""

    def __init__(self, n: int, dll: Path = DEFAULT_DLL, labels: bool = False, log: Any = subprocess.DEVNULL,
                 engine: str | None = None):
        self._args = (dll, labels, log, engine or default_engine())
        self.envs = [Sts2Env(*self._args) for _ in range(n)]
        self._pending: set[int] = set()
        self._sel: selectors.BaseSelector | None = None
        self._failed: dict[int, EngineError] = {}      # 请求发不出去（进程已退出）的，下一次 recv_ready 当作引擎报错返回

    def __len__(self) -> int:
        return len(self.envs)

    def _all(self, reqs: Sequence[dict | None]) -> list[Obs | None]:
        for e, r in zip(self.envs, reqs):
            if r is not None:
                e._proc.send(r)
        return [parse(e._proc.recv()) if r is not None else None for e, r in zip(self.envs, reqs)]

    def reset(self, reqs: Sequence[dict]) -> list[Obs]:
        """reqs[i] 为 {"mode": "run", "character": ..., ...} 或 {"mode": "combat", "spec": {...}}。"""
        return self._all([dict(r, op="reset", labels=self.envs[i].labels) for i, r in enumerate(reqs)])

    def reset_some(self, reqs: dict[int, dict]) -> dict[int, Obs]:
        """只重置部分进程：reqs 为 {进程下标: 请求}。"""
        full: list[dict | None] = [None] * len(self.envs)
        for i, r in reqs.items():
            full[i] = dict(r, op="reset", labels=self.envs[i].labels)
        out = self._all(full)
        return {i: out[i] for i in reqs}

    def step(self, actions: Sequence[int | None]) -> list[Obs | None]:
        """actions[i] 为 None 的进程本轮不动。"""
        return self._all([None if a is None else {"op": "step", "action": int(a)} for a in actions])

    # ---- 异步接口：各进程各自推进，谁先算完先处理谁，避免同步步进时等最慢的进程

    def send(self, i: int, req: dict) -> None:
        """向第 i 个进程发请求（{"op": "step", "action": a} 或 reset 请求），不等待结果。"""
        if req.get("op") == "reset":
            req = dict(req, labels=self.envs[i].labels)
        try:
            self.envs[i]._proc.send(req)
        except OSError as e:
            # 引擎进程在两次请求之间退出了（被外面结束等）：写管道报 BrokenPipeError。原来异常一路抛出、整轮评测作废；
            # 现在换一个新进程，这一局在下一次 recv_ready 里按引擎报错返回（调用方结束这一局、照常开下一局）
            if self._sel is not None:
                self._respawn(i)
            else:
                self.envs[i] = Sts2Env(*self._args)
            self._failed[i] = EngineError(f"引擎进程已退出，请求发不出去（{type(e).__name__}）")
        self._pending.add(i)

    def send_step(self, i: int, action: int) -> None:
        self.send(i, {"op": "step", "action": int(action)})

    def recv_ready(self, timeout: float | None = None) -> list[tuple[int, Obs | EngineError]]:
        """等到至少一个进程有结果，返回所有已就绪的 (进程下标, 观测)。
        引擎报错时观测换成 EngineError（调用方结束这一局、重新开局即可）；进程已退出的先换一个新进程。"""
        if not self._pending:
            return []
        if self._failed:
            out = [(i, self._failed.pop(i)) for i in list(self._failed) if i in self._pending]
            self._failed.clear()
            for i, _ in out:
                self._pending.discard(i)
            if out:
                return out
        if self._sel is None:
            self._sel = selectors.DefaultSelector()
            for i, e in enumerate(self.envs):
                self._sel.register(e._proc.p.stdout, selectors.EVENT_READ, i)
        out = []
        for key, _ in self._sel.select(timeout):
            i = key.data
            if i in self._pending:
                self._pending.discard(i)
                try:
                    out.append((i, parse(self.envs[i]._proc.recv())))
                except EngineError as e:
                    if self.envs[i]._proc.p.poll() is not None:
                        self._respawn(i)
                    out.append((i, e))
        return out

    def _respawn(self, i: int) -> None:
        self._sel.unregister(self.envs[i]._proc.p.stdout)
        self.envs[i] = Sts2Env(*self._args)
        self._sel.register(self.envs[i]._proc.p.stdout, selectors.EVENT_READ, i)

    @property
    def pending(self) -> set[int]:
        return self._pending

    def close(self) -> None:
        if self._sel is not None:
            self._sel.close()
        for e in self.envs:
            e.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
