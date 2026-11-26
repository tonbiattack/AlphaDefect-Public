"""sts2-sim 搜索服务（engine/Sts2SimServe 的 serve 模式）的 Python 客户端。

协议与 Sts2Headless 相同（标准输入一行 JSON；标准输出 int32 长度 + 内容；出错 int32 −1 + 信息），另加槽位：
  reset（mode=transplant：真实引擎的整局存档 + 下一个房间编号 + 目标格子 + 房间类型）→ 第一个决策的观测帧
  step → 下一个决策的帧；save 槽位 → "ok"（复制当前战斗状态）；load 槽位 → 帧；reseed → 帧（换以后的随机数、重洗抽牌堆）
  state → JSON（候选文字、生命格挡、手牌）；close
进程为 dotnet <dll> serve [词表]（读写同 sts2env.env._Process）。dll 路径依次取参数、环境变量 STS2SIM_DLL、仓库里的默认编译输出；
词表（每行一个键，同真实引擎 vocab 操作的输出）依次取参数、环境变量 STS2SIM_VOCAB，不给时服务端记录的词表 ID 全为 0。

推演要大量往返，send / recv 分开：多条请求可以先一起写进管道再依次读（pipeline），几个进程同时算。
请求很小，一次写进去的总量远小于管道缓冲（64 KB），服务端写回时不会和我们互相等死。
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .env import DOTNET, EngineError, _Process
from .protocol import Obs, parse

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SIM_DLL = REPO / "engine" / "Sts2SimServe" / "bin" / "Release" / "net9.0" / "Sts2SimServe.dll"


def sim_dll(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("STS2SIM_DLL") or DEFAULT_SIM_DLL)


class SimClient:
    """一个 Sts2SimServe 进程：一场战斗的当前状态 + 若干槽位。同步方法每次一个往返；send / recv 用于成批流水。"""

    def __init__(self, dll: str | Path | None = None, log: Any = subprocess.DEVNULL, vocab: str | Path | None = None,
                 tables: str | Path | None = None):
        self.dll = sim_dll(dll)
        if not self.dll.exists():
            raise FileNotFoundError(f"找不到 Sts2SimServe：{self.dll}（参数 --sim-dll 或环境变量 STS2SIM_DLL）")
        vocab = vocab or os.environ.get("STS2SIM_VOCAB")
        tables = tables or os.environ.get("STS2SIM_TABLES")     # 给了就用移植版观测编码器（要 -p:WithObs=true 编译的服务），否则临时编码器
        if tables and not vocab:
            raise ValueError("用移植版观测编码器时词表与 sim_tables.json 都要给")
        self._proc = _Process.__new__(_Process)     # 只借它的读写；命令行多一个词表参数，自己拉起
        env = dict(os.environ, DOTNET_ROOT=str(DOTNET.parent), DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_TieredPGO="1")
        self._proc.p = subprocess.Popen([str(DOTNET), str(self.dll), "serve"] + ([str(vocab)] if vocab else []) + ([str(tables)] if tables else []),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, bufsize=0)
        self.inflight = 0          # 已发出、还没读的请求数
        self.last_raw: bytes | None = None

    # ---- 底层：发、收

    def send(self, req: dict) -> None:
        self._proc.send(req)
        self.inflight += 1

    def recv_raw(self) -> bytes:
        """读一条回复的原始字节；出错时抛 EngineError（进程仍可用；进程退出时也是 EngineError）。"""
        self.inflight -= 1
        return self._proc.recv()

    def drain(self) -> list[bytes | EngineError]:
        """把已发出的请求的回复按顺序全部读完；单条出错不影响后面的。"""
        out: list[bytes | EngineError] = []
        while self.inflight:
            try:
                out.append(self.recv_raw())
            except EngineError as e:
                out.append(e)
                if self._proc.p.poll() is not None:     # 进程没了：后面的回复不会来了
                    out.extend([EngineError("sts2-sim 进程已退出")] * self.inflight)
                    self.inflight = 0
        return out

    def call_raw(self, req: dict) -> bytes:
        if self.inflight:
            raise RuntimeError("还有未读的流水请求，先 drain()")
        self.send(req)
        return self.recv_raw()

    def _frame(self, req: dict) -> Obs:
        self.last_raw = self.call_raw(req)
        return parse(self.last_raw)

    # ---- 协议

    def reset_transplant(self, save: dict, next_room_id: int, target: tuple[int, int] | list[int], room: str,
                         root_frame: bytes | None = None) -> Obs:
        """真实引擎在选路决策时的整局存档（Sts2Headless 的 save 操作）→ 导入 sts2-sim，进 target（[列, 行]）这一格的战斗。
        root_frame：这场战斗第一个决策的真实帧（原始字节），移植版观测编码器从它沿用整局层面的字段。"""
        return self._frame(transplant_req(save, next_room_id, target, room, root_frame))

    def root(self, frame: bytes) -> None:
        """给移植版编码器换根帧（真实引擎在当前决策上的帧）。"""
        r = self.call_raw({"op": "root", "frame": base64.b64encode(frame).decode()})
        if r.strip() != b"ok":
            raise EngineError(f"root 的回复不是 ok：{r[:80]!r}")

    def step(self, action: int) -> Obs:
        return self._frame({"op": "step", "action": int(action)})

    def save(self, slot: int) -> None:
        r = self.call_raw({"op": "save", "slot": int(slot)})
        if r.strip() != b"ok":
            raise EngineError(f"save 槽位 {slot} 的回复不是 ok：{r[:80]!r}")

    def load(self, slot: int) -> Obs:
        return self._frame({"op": "load", "slot": int(slot)})

    def reseed(self, seed: int) -> Obs:
        return self._frame({"op": "reseed", "seed": int(seed) & 0xFFFFFFFF})

    def state(self) -> dict:
        return json.loads(self.call_raw({"op": "state"}).decode("utf-8"))

    def close(self) -> None:
        try:
            self.drain()
        except Exception:
            pass
        self._proc.close()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def transplant_req(save: dict, next_room_id: int, target, room: str, root_frame: bytes | None = None) -> dict:
    req = {"op": "reset", "mode": "transplant", "save": save, "next_room_id": int(next_room_id),
           "target": [int(target[0]), int(target[1])], "room": room}
    if root_frame is not None:
        req["root_frame"] = base64.b64encode(root_frame).decode()
    return req
