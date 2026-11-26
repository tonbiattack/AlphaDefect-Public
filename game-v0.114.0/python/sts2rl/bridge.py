"""双机协同：远程采集机（A）把轨迹送给主线的学习进程（B），并从 B 取最新权重（2026-10-28 起，决策记录 155）。

- B（学习机）：train.py 在 cfg.bridge_port > 0 时起 BridgeServer，只监听 cfg.bridge_bind（内网地址），连接先发 32 字节口令
  （cfg.bridge_token_file 里的十六进制串，两台机器上各放一份，不进仓库），再发一个字节表明用途：
    b"T"：轨迹流——之后连续发送帧，每帧解码成轨迹字典，放进学习进程取数的队列（与本机采集进程的轨迹完全同等对待）；
    b"W"：权重——客户端发 8 字节“我手上的版本”，服务端等到有更新的版本，回发 [版本 8 字节][长度 8 字节][权重文件原样字节]。
- A（采集机）：tools/remote/remote_actors.py 起若干采集进程（sts2rl.actor.run_actor，代码不改），一个线程把权重写到本机
  /dev/shm 的权重文件（采集进程照常每秒查一次），一个线程把采集进程的轨迹编码后送给 B。
- 编码不用 pickle（网络上的 pickle 能执行任意代码）：轨迹字典 → JSON 结构 + 一串 numpy 数组的原始字节，数组按 dtype/shape 还原，
  Ragged 按字段还原；整帧用 zlib 1 级压缩。B 只解码数组与基本类型，不会执行任何代码。
帧格式：b"SLS2" + 头长度（8 字节）+ 负载长度（8 字节）+ 头（JSON）+ 负载（zlib 压缩的数组字节）。
"""
from __future__ import annotations

import hmac
import json
import socket
import struct
import threading
import time
import zlib
from dataclasses import fields
from pathlib import Path

import numpy as np

from .features import Ragged

MAGIC = b"SLS2"
_RAGGED_FIELDS = [f.name for f in fields(Ragged)]
_ALLOWED_DTYPES = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"}
MAX_FRAME = 512 << 20      # 单帧上限（一局整局约几 MB），防止对端发来畸形长度


def load_token(path: str) -> bytes:
    tok = bytes.fromhex(Path(path).read_text().strip())
    if len(tok) != 32:
        raise ValueError("口令应为 32 字节（64 个十六进制字符）")
    return tok


# ---------------------------------------------------------------- 编码
def encode(obj) -> bytes:
    arrays: list[np.ndarray] = []

    def enc(x):
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            arrays.append(a)
            return {"__a": len(arrays) - 1, "dt": a.dtype.str, "sh": list(a.shape)}
        if isinstance(x, Ragged):
            return {"__r": {k: enc(getattr(x, k)) for k in _RAGGED_FIELDS}}
        if isinstance(x, dict):
            return {"__d": [[enc(k), enc(v)] for k, v in x.items()]}
        if isinstance(x, (list, tuple)):
            return [enc(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        raise TypeError(f"轨迹里有不支持的类型 {type(x)}")

    head = json.dumps(enc(obj), ensure_ascii=False, separators=(",", ":")).encode()
    payload = zlib.compress(b"".join(a.tobytes() for a in arrays), 1)
    sizes = json.dumps([a.nbytes for a in arrays]).encode()
    head = struct.pack("<Q", len(sizes)) + sizes + head
    return MAGIC + struct.pack("<QQ", len(head), len(payload)) + head + payload


def decode(head: bytes, payload: bytes):
    (ns,) = struct.unpack_from("<Q", head, 0)
    sizes = json.loads(head[8:8 + ns])
    tree = json.loads(head[8 + ns:])
    raw = zlib.decompress(payload, bufsize=max(1, sum(sizes)))
    if len(raw) != sum(sizes):
        raise ValueError("负载长度与数组大小对不上")
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    def dec(x):
        if isinstance(x, dict):
            if "__a" in x:
                dt = np.dtype(x["dt"])
                if dt.name not in _ALLOWED_DTYPES:
                    raise ValueError(f"不允许的数组类型 {dt}")
                i = int(x["__a"])
                return np.frombuffer(raw, dtype=dt, count=int(np.prod(x["sh"], dtype=np.int64)), offset=int(offs[i])).reshape(x["sh"]).copy()
            if "__r" in x:
                return Ragged(**{k: dec(v) for k, v in x["__r"].items() if k in _RAGGED_FIELDS})
            if "__d" in x:
                return {dec(k): dec(v) for k, v in x["__d"]}
            raise ValueError("未知的对象标记")
        if isinstance(x, list):
            return [dec(v) for v in x]
        return x

    return dec(tree)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("对端关闭了连接")
        got += k
    return bytes(buf)


def recv_frame(sock: socket.socket):
    if _recv_exact(sock, 4) != MAGIC:
        raise ValueError("帧头不对")
    hl, pl = struct.unpack("<QQ", _recv_exact(sock, 16))
    if hl > MAX_FRAME or pl > MAX_FRAME:
        raise ValueError("帧过大")
    return decode(_recv_exact(sock, hl), _recv_exact(sock, pl)), 20 + hl + pl


# ---------------------------------------------------------------- B：服务端
class BridgeServer:
    """学习进程里的桥：收远程轨迹放进 put(item)（阻塞即反压），向远程发最新权重文件。stats() 给日志用。"""

    def __init__(self, bind: str, port: int, token: bytes, put, weights_path: Path):
        self.token, self.put, self.weights_path = token, put, Path(weights_path)
        self.version = -1
        self.lock = threading.Lock()
        self.n_traj = self.n_steps = self.n_bytes = self.n_bad = 0
        self.conns = 0
        self.last_seen = 0.0
        self.sock = socket.create_server((bind, port), reuse_port=False)
        threading.Thread(target=self._accept, daemon=True, name="bridge-accept").start()

    def set_version(self, v: int) -> None:
        self.version = v

    def stats(self) -> dict:
        with self.lock:
            out = {"remote_trajs": self.n_traj, "remote_steps": self.n_steps, "remote_mb": round(self.n_bytes / 2**20, 1),
                   "remote_bad": self.n_bad, "remote_conns": self.conns, "remote_idle_s": round(time.time() - self.last_seen, 1) if self.last_seen else None}
            self.n_traj = self.n_steps = self.n_bytes = 0
        return out

    def _accept(self) -> None:
        while True:
            try:
                c, addr = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c, addr), daemon=True, name=f"bridge-{addr[0]}").start()

    def _serve(self, c: socket.socket, addr) -> None:
        counted = False
        try:
            c.settimeout(30)
            tok = _recv_exact(c, 32)
            if not hmac.compare_digest(tok, self.token):
                print(f"桥：{addr[0]} 口令不对，断开", flush=True)
                return
            role = _recv_exact(c, 1)
            c.settimeout(None)
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.conns += 1
            counted = True
            print(f"桥：{addr[0]} 已连上（{'轨迹' if role == b'T' else '权重'}）", flush=True)
            if role == b"T":
                while True:
                    try:
                        item, nb = recv_frame(c)
                        if not isinstance(item, (list, tuple)) or len(item) != 3 or item[0] != "traj" or not isinstance(item[2], dict):
                            raise ValueError("帧内容不是轨迹")
                    except (ConnectionError, OSError):
                        raise
                    except Exception as e:      # 坏帧：记下来、断开（对端会重连），不影响学习
                        with self.lock:
                            self.n_bad += 1
                        print(f"桥：{addr[0]} 坏帧，断开：{e}", flush=True)
                        return
                    with self.lock:
                        self.n_traj += 1; self.n_steps += int(item[2].get("len", 0)); self.n_bytes += nb; self.last_seen = time.time()
                    self.put(("traj", int(item[1]), item[2]))
            elif role == b"W":
                while True:
                    have = struct.unpack("<q", _recv_exact(c, 8))[0]
                    while self.version <= have:
                        time.sleep(0.2)
                    data = self.weights_path.read_bytes()
                    c.sendall(struct.pack("<qQ", self.version, len(data)) + data)
        except (ConnectionError, OSError) as e:
            print(f"桥：{addr[0]} 断开：{e}", flush=True)
        finally:
            if counted:
                with self.lock:
                    self.conns -= 1
            try:
                c.close()
            except OSError:
                pass


# ---------------------------------------------------------------- A：客户端
def _connect(host: str, port: int, token: bytes, role: bytes) -> socket.socket:
    s = socket.create_connection((host, port), timeout=30)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.sendall(token + role)
    s.settimeout(None)
    return s


def weights_loop(host: str, port: int, token: bytes, path: Path, stop: threading.Event, status: dict) -> None:
    """一直从 B 取最新权重，原子地写到 path（采集进程照常读这个文件）。断线就重连。"""
    have = -1
    while not stop.is_set():
        try:
            s = _connect(host, port, token, b"W")
            while not stop.is_set():
                s.sendall(struct.pack("<q", have))
                v, n = struct.unpack("<qQ", _recv_exact(s, 16))
                if n > MAX_FRAME:
                    raise ValueError("权重过大")
                data = _recv_exact(s, n)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
                have = v; status["version"] = v; status["weights_t"] = time.time()
        except Exception as e:
            status["weights_err"] = f"{time.strftime('%H:%M:%S')} {e}"
            time.sleep(3)


def traj_loop(host: str, port: int, token: bytes, q, stop: threading.Event, status: dict) -> None:
    """从本机采集进程的队列取轨迹，编码后送给 B。断线时停下来重连（采集进程会被队列反压挡住）。"""
    pending = None
    while not stop.is_set():
        try:
            s = _connect(host, port, token, b"T")
            status["connected"] = True
            while not stop.is_set():
                if pending is None:
                    try:
                        item = q.get(timeout=1.0)
                    except Exception:
                        continue
                    if item[0] != "traj":       # 采集进程出错：记在本机，不送给 B（否则 B 的学习进程会因此退出）
                        status["actor_err"] = f"{time.strftime('%H:%M:%S')} 采集进程 {item[1]}：{str(item[2])[-300:]}"
                        status.setdefault("actor_errs", 0); status["actor_errs"] += 1
                        continue
                    pending = encode(item)
                s.sendall(pending)
                status["sent"] = status.get("sent", 0) + 1; status["sent_mb"] = status.get("sent_mb", 0.0) + len(pending) / 2**20
                pending = None
        except Exception as e:
            status["connected"] = False
            status["traj_err"] = f"{time.strftime('%H:%M:%S')} {e}"
            time.sleep(3)
