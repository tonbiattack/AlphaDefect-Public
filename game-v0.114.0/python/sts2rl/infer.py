"""搜索、评测用的推理路径与共用推理进程（2026-11-16，PIMC 推理侧提速；测量与核对见 tools/pimc_infer.py）。

InferEngine：一个网络的推理入口，Ragged → (对数几率 [N, M], 价值 [N])，都是 CPU 上的 float32 numpy。
  - torch.inference_mode + PolicyValueNet.forward_infer（只算策略与价值，与 forward 的这两项逐位相同）；
  - precision："fp32"（默认）、"tf32"（矩阵乘用 TF32）、"bf16"、"fp16"（显卡上用 autocast），输出一律转回 float32；
  - fast：显卡上改用学习端的打包变长注意力（PolicyValueNet._attend_packed，嵌套张量 SDPA，不分组不填充；fp32 下与分组填充差约 1e-5）；
  - compile：torch.compile(forward_infer, dynamic=True)；
  - int8：CPU 上对编码器前馈两层、候选与价值头做动态量化（torch.ao.quantization.quantize_dynamic）；threads：CPU 线程数（0 为不改）；
  - packed：显卡上把整批输入按数据类型拼成几块、各一次拷到显卡（Batch.to 是每个数组各拷一次，约 50 次）。
  实测（reports/PIMC推理侧提速.md）：fp32 以外各项都没有在搜索上带来收益，bf16、int8 偏差大，不要用在搜索上。
serve / InferClient：一台机器一个推理进程，所有搜索分片经 Unix 套接字共用。推理进程收到第一个请求后，
  等到所有已连接的客户端都在等（同步客户端最多一个请求在途），或等满 wait_ms，或凑满 max_rows 帧，就把这些请求合成一批前向，再按请求拆开回复。
  同训练采集进程的 infer_wait_ms 凑批（actor.py），只是跨进程。CPU 吃满的机器上要把推理进程钉在一个单独的快核上（taskset、nice），
  depth 用 1（2 是组批与前向分两个线程，Python 里两线程抢 GIL，实测更慢）。

线上协议（小端）：每条消息 = u32 长度（不含这 4 字节）+ u8 类型 + 内容。
  类型 1 推理请求：u32 N、ΣT、ΣM、标志（第 0 位：带特权特征 tok_priv），接着 modes i32[N]、header i32[N, 34]、
    每帧记录数 i32[N]、每帧候选数 i32[N]、tok_ints i32[ΣT, 5]、tok_feats f32[ΣT, TOK_FEATS]、（tok_priv f32[ΣT, 2]）、cand_ints i32[ΣM, 5]、cand_feats f32[ΣM, CAND_FEATS]（格式 29 起 30、20 项；两端都用 protocol 的常量）。
    回复类型 1：u32 N、宽度 W，对数几率 f32[N, W]（没有的候选为 -inf），价值 f32[N]。
  类型 2 统计：回复类型 2，JSON（批数、帧数、各段耗时）。
  类型 9 出错：回复内容为 UTF-8 的错误文本（推理进程不退出）。
"""
from __future__ import annotations

import contextlib
import json
import os
import selectors
import socket
import struct
import time

import numpy as np

from sts2env.protocol import CAND_FEATS, CAND_INTS, HEADER_INTS, TOK_FEATS, TOK_INTS, TOK_PRIV

from .features import Batch, Ragged, collate_ragged

PRECISIONS = ("fp32", "tf32", "bf16", "fp16")
T_INFER, T_STATS, T_ERROR = 1, 2, 9


def parse_spec(spec: str) -> dict:
    """"fp32" / "tf32" / "bf16" / "fp16"，后面可接 ",int8"、",threads=N"、",nopack"、",fast"、",compile" → InferEngine 的关键字参数（例 "fp32,int8,threads=4"）。"""
    parts = spec.split(",")
    opt: dict = {"precision": parts[0]}
    for p in parts[1:]:
        if p == "int8":
            opt["int8"] = True
        elif p == "nopack":
            opt["packed"] = False
        elif p == "fast":
            opt["fast"] = True
        elif p == "compile":
            opt["compile"] = True
        elif p.startswith("threads="):
            opt["threads"] = int(p[8:])
        elif p:
            raise ValueError(f"不认识的推理选项：{p}")
    return opt


def load_net(ckpt: str, device: str = "cpu", ema: bool = False):
    """按 boss_pimc.Runner 的办法载入检查点（ema 时有滑动平均权重就用它）。"""
    import torch
    from .model import ModelConfig, PolicyValueNet, load_state
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    state = blob.get("ema_state") if ema and blob.get("ema_state") else blob["state"]
    load_state(net, state, blob.get("content_names"))
    return net.to(device)


# ---------------------------------------------------------------- 整批输入一次拷到显卡

def batch_to_packed(b: Batch, device, pin: bool = True) -> dict:
    """Batch.to 的等价物：按数据类型把所有数组拼成一块（int64、float32、bool 各一块），各一次拷到 device，再切回原来的形状。
    显卡上省掉约 50 次小拷贝（每次都要从可分页内存同步拷）；结果与 Batch.to 的逐个拷贝数值相同。"""
    import torch
    arrs = {}
    for k, v in b.__dict__.items():
        if isinstance(v, np.ndarray):
            arrs[k] = v
    for j, (idx, kg) in enumerate(b.groups):
        arrs[f"__group{j}"] = idx
    by: dict = {}
    for k, v in arrs.items():
        by.setdefault(v.dtype.str, []).append(k)
    out = {}
    for dt, keys in by.items():
        flat = np.concatenate([np.ascontiguousarray(arrs[k]).reshape(-1) for k in keys]) if keys else None
        t = torch.from_numpy(flat)
        if pin and str(device).startswith("cuda"):
            t = t.pin_memory()
        t = t.to(device, non_blocking=True)
        off = 0
        for k in keys:
            a = arrs[k]
            out[k] = t[off: off + a.size].view(a.shape)
            off += a.size
    for k, v in b.__dict__.items():
        if not isinstance(v, np.ndarray) and k != "groups":
            out[k] = v
    out["groups"] = [(out.pop(f"__group{j}"), kg) for j, (_, kg) in enumerate(b.groups)]
    return out


# ---------------------------------------------------------------- 推理入口

class InferEngine:
    """一个网络的推理入口。run(r) → (对数几率 float32[N, W], 价值 float32[N])（numpy，CPU）。见模块说明。"""

    def __init__(self, net, device: str = "cpu", precision: str = "fp32", int8: bool = False, threads: int = 0, packed: bool = True,
                 fast: bool = False, compile: bool = False):
        import torch
        if precision not in PRECISIONS:
            raise ValueError(f"precision 只能是 {PRECISIONS}")
        self.torch = torch
        self.device = str(device)
        self.cuda = self.device.startswith("cuda")
        if threads:
            torch.set_num_threads(int(threads))
        net = net.eval()
        net.infer_ready()
        if int8:
            if self.cuda:
                raise ValueError("int8 动态量化只在 CPU 上")
            import copy
            # 只量化算量大的层：编码器各层的前馈两层（注意力的 in_proj 是参数不是 Linear，out_proj 是不可动态量化的 Linear），
            # 战斗与局外的候选打分头、价值头。别的 Linear 多是清零初始化的小映射，content_proj 只对整张内容表算一次再缓存（_content 读它的 .weight）
            spec = {n: torch.ao.quantization.default_dynamic_qconfig for n, m in net.named_modules()
                    if type(m) is torch.nn.Linear and (n.startswith("encoder.") or n.split(".")[0] in ("cand_mlp", "cand_mlp_m", "value_mlp", "value_mlp_m"))}
            net = torch.ao.quantization.quantize_dynamic(copy.deepcopy(net).cpu(), spec, dtype=torch.qint8)
            net.infer_ready()
            self.quantized = sorted(spec)
        self.net = net
        self.precision, self.int8, self.packed = precision, bool(int8), bool(packed)
        self.fast = bool(fast) and self.cuda
        # compile：torch.compile(forward_infer, dynamic=True)，融合逐元素的小算子、减少每批的启动次数（形状不固定，不用 CUDA Graphs）
        self.fwd = torch.compile(net.forward_infer, dynamic=True) if compile else net.forward_infer
        self.compiled = bool(compile)
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        self.n_fwd = self.n_rows = 0
        self.t_collate = self.t_fwd = 0.0

    @contextlib.contextmanager
    def _ctx(self):
        """前向时的设置：自动混合精度、TF32、打包注意力（net.fast）。后两项是进程级 / 网络级的开关，退出时还原，
        同一进程里别的 InferEngine（离线对照时共用一个网络）不受影响。"""
        torch = self.torch
        tf32_old, fast_old = torch.backends.cuda.matmul.allow_tf32, getattr(self.net, "fast", False)
        mha_old = torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.cuda.matmul.allow_tf32 = self.precision == "tf32"
            self.net.fast = self.fast
            if self.int8:       # 编码器层的融合快速路径要读 linear1.weight 等张量，量化后的 Linear 没有，只能走普通路径
                torch.backends.mha.set_fastpath_enabled(False)
            if self.amp is None:
                yield
            else:
                with torch.autocast(device_type="cuda" if self.cuda else "cpu", dtype=self.amp):
                    yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32_old
            self.net.fast = fast_old
            torch.backends.mha.set_fastpath_enabled(mha_old)

    def run_batch(self, b: Batch) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        t0 = time.perf_counter()
        with torch.inference_mode(), self._ctx():
            bd = batch_to_packed(b, self.device) if (self.packed and self.cuda) else b.to(self.device)
            logits, values = self.fwd(bd)
            out = logits.float().cpu().numpy(), values.float().cpu().numpy()
        self.t_fwd += time.perf_counter() - t0
        self.n_fwd += 1
        self.n_rows += len(out[1])
        return out

    def run(self, r: Ragged) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        b = collate_ragged(r)
        self.t_collate += time.perf_counter() - t0
        return self.run_batch(b)


# ---------------------------------------------------------------- 线上编码

_REQ = struct.Struct("<IIII")
_RESP = struct.Struct("<II")


def _send(sock: socket.socket, typ: int, parts: list) -> None:
    n = 1 + sum(len(p) if isinstance(p, (bytes, bytearray, memoryview)) else p.nbytes for p in parts)
    sock.sendall(struct.pack("<IB", n, typ))
    for p in parts:
        sock.sendall(p if isinstance(p, (bytes, bytearray, memoryview)) else memoryview(np.ascontiguousarray(p)).cast("B"))


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("对端关闭了连接")
        got += k
    return buf


def _recv_msg(sock: socket.socket) -> tuple[int, bytearray]:
    n, typ = struct.unpack("<IB", _recv_exact(sock, 5))
    return typ, _recv_exact(sock, n - 1)


def encode_ragged(r: Ragged, with_priv: bool = False) -> list:
    """Ragged → 推理请求的内容（若干段，直接交给 _send）。"""
    N, ST, SC = len(r), len(r.tok_ints), len(r.cand_ints)
    h = r.header
    if h.shape[1] < HEADER_INTS:
        h = np.pad(h, ((0, 0), (0, HEADER_INTS - h.shape[1])))
    cf = r.cand_feats if r.cand_feats.shape[1] >= CAND_FEATS else np.pad(r.cand_feats, ((0, 0), (0, CAND_FEATS - r.cand_feats.shape[1])))
    parts = [_REQ.pack(N, ST, SC, 1 if with_priv else 0), r.modes().astype(np.int32), h.astype(np.int32),
             np.diff(r.tok_off).astype(np.int32), np.diff(r.cand_off).astype(np.int32),
             r.tok_ints.astype(np.int32), r.tok_feats.astype(np.float32)]
    if with_priv:
        parts.append(r.tok_priv.astype(np.float32))
    parts += [r.cand_ints.astype(np.int32), cf.astype(np.float32)]
    return parts


def decode_ragged(buf) -> Ragged:
    N, ST, SC, flags = _REQ.unpack_from(buf, 0)
    off = _REQ.size

    def take(dtype, count, shape):
        nonlocal off
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=off).reshape(shape)
        off += a.nbytes
        return a

    modes = take(np.int32, N, (N,))
    header = take(np.int32, N * HEADER_INTS, (N, HEADER_INTS))
    tn = take(np.int32, N, (N,)); cn = take(np.int32, N, (N,))
    ti = take(np.int32, ST * TOK_INTS, (ST, TOK_INTS))
    tf = take(np.float32, ST * TOK_FEATS, (ST, TOK_FEATS))
    tp = take(np.float32, ST * TOK_PRIV, (ST, TOK_PRIV)) if flags & 1 else np.zeros((ST, TOK_PRIV), np.float32)
    ci = take(np.int32, SC * CAND_INTS, (SC, CAND_INTS))
    cf = take(np.float32, SC * CAND_FEATS, (SC, CAND_FEATS))
    r = Ragged(header, ti, tf, tp, np.concatenate([[0], np.cumsum(tn)]).astype(np.int64), ci, cf,
               np.concatenate([[0], np.cumsum(cn)]).astype(np.int64))
    r.mode = modes.astype(np.int64)
    return r


# ---------------------------------------------------------------- 共用推理进程

class _Conn:
    __slots__ = ("sock", "name")

    def __init__(self, sock, name):
        self.sock, self.name = sock, name


def serve(engine: InferEngine, path: str, wait_ms: float = 2.0, max_rows: int = 1024, log=print, stop=None, depth: int = 1) -> dict:
    """在 Unix 套接字 path 上提供推理，直到 stop()（不给时一直跑）为真。凑批规则见模块说明。返回统计。
    depth=1：主线程收请求、凑批、组批、前向、回复，依次做；
    depth=2：前向与回复交给工作线程。显卡在算第 k 批时，主线程照收请求，等没在途的客户端都到了（或等满 wait_ms、凑满 max_rows）
    就先把第 k+1 批组好排队，第 k 批一完立刻接着算：组批与前向重叠，满载时每批的时间从“组批 + 前向 + 回复”降到两者的较大者。"""
    import queue
    import threading
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
    lis = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    lis.bind(path)
    lis.listen(256)
    lis.setblocking(False)
    sel = selectors.DefaultSelector()
    sel.register(lis, selectors.EVENT_READ, None)
    wake_r, wake_w = socket.socketpair()
    wake_r.setblocking(False)
    sel.register(wake_r, selectors.EVENT_READ, "wake")
    conns: dict[int, _Conn] = {}
    pending: list[tuple[_Conn, Ragged]] = []
    inflight: list[set] = []          # 已交给工作线程、还没回复完的各批的客户端
    to_close: list[_Conn] = []
    first_t = 0.0
    st = {"batches": 0, "rows": 0, "reqs": 0, "t_recv": 0.0, "t_collate": 0.0, "t_fwd": 0.0, "t_send": 0.0, "t_wait": 0.0,
          "rows_hist": {}, "t0": time.time(), "errors": 0, "depth": depth}
    wait = wait_ms / 1000.0
    jobs: queue.Queue = queue.Queue()
    log(f"推理进程就绪：{path}（{engine.device}，{engine.precision}{'，int8' if engine.int8 else ''}，凑批最多等 {wait_ms} 毫秒、最多 {max_rows} 帧，流水深度 {depth}）")

    def close(c: _Conn):
        with contextlib.suppress(Exception):
            sel.unregister(c.sock)
        c.sock.close()
        conns.pop(id(c), None)

    def stats_json() -> bytes:
        s = dict(st)
        s["clients"] = len(conns)
        s["uptime"] = time.time() - st["t0"]
        return json.dumps(s).encode()

    def forward_reply(reqs, b, err):
        t1 = time.perf_counter()
        if err is None:
            try:
                logits, values = engine.run_batch(b)
            except Exception:       # 坏请求不拖垮推理进程：这一批的客户端都收到出错回复
                import traceback
                err = traceback.format_exc()
        if err is not None:
            st["errors"] += 1
            log(f"推理出错：{err}")
        t2 = time.perf_counter()
        i = 0
        for c, q in reqs:
            n = len(q)
            try:
                if err is not None:
                    _send(c.sock, T_ERROR, [err.encode()])
                else:
                    # 每个请求只回到它自己的最大候选数（与单独前向时的宽度相同）
                    w = max(1, int(np.diff(q.cand_off).max(initial=0)))
                    _send(c.sock, T_INFER, [_RESP.pack(n, w), np.ascontiguousarray(logits[i:i + n, :w]), values[i:i + n]])
            except OSError:
                to_close.append(c)
            i += n
        t3 = time.perf_counter()
        st["batches"] += 1; st["rows"] += i; st["reqs"] += len(reqs)
        st["t_fwd"] += t2 - t1; st["t_send"] += t3 - t2
        k = str(1 << max(0, (i - 1).bit_length()))
        st["rows_hist"][k] = st["rows_hist"].get(k, 0) + 1

    def worker():
        while True:
            job = jobs.get()
            if job is None:
                return
            forward_reply(*job)
            wake_w.send(b"x")

    th = None
    if depth > 1:
        th = threading.Thread(target=worker, name="infer-forward", daemon=True)
        th.start()

    def launch():
        """把 pending 组成一批：depth 1 当场算完；depth 2 交给工作线程。"""
        nonlocal pending
        reqs, pending = pending, []
        t0 = time.perf_counter()
        err, b = None, None
        try:
            r = reqs[0][1] if len(reqs) == 1 else Ragged.concat([q[1] for q in reqs])
            b = collate_ragged(r)
        except Exception:
            import traceback
            err = traceback.format_exc()
        st["t_collate"] += time.perf_counter() - t0
        if th is None:
            forward_reply(reqs, b, err)
        else:
            inflight.append({id(c) for c, _ in reqs})
            jobs.put((reqs, b, err))

    try:
        while stop is None or not stop():
            timeout = None
            if pending and len(inflight) < depth:
                timeout = max(0.0, first_t + wait - time.perf_counter())
            elif stop is not None:
                timeout = 0.5       # 工作线程排满时不空转（空转会和工作线程抢 GIL），等它算完的唤醒或新请求
            tw = time.perf_counter()
            events = sel.select(timeout)
            st["t_wait"] += time.perf_counter() - tw
            for key, _ in events:
                if key.data is None:
                    while True:
                        try:
                            s, _ = lis.accept()
                        except BlockingIOError:
                            break
                        s.setblocking(True)
                        c = _Conn(s, f"c{len(conns)}")
                        conns[id(c)] = c
                        sel.register(s, selectors.EVENT_READ, c)
                    continue
                if isinstance(key.data, str):       # "wake"：工作线程算完一批
                    with contextlib.suppress(BlockingIOError):
                        n_done = len(wake_r.recv(4096))
                        del inflight[:n_done]
                    while to_close:
                        close(to_close.pop())
                    continue
                c = key.data
                tr = time.perf_counter()
                try:
                    typ, body = _recv_msg(c.sock)
                except (ConnectionError, OSError):
                    close(c)
                    continue
                if typ == T_INFER:
                    if not pending:
                        first_t = time.perf_counter()
                    pending.append((c, decode_ragged(body)))
                elif typ == T_STATS:
                    _send(c.sock, T_STATS, [stats_json()])
                st["t_recv"] += time.perf_counter() - tr
            if not pending or len(inflight) >= depth:
                continue
            busy = set().union(*inflight) if inflight else set()
            waiting = len({id(c) for c, _ in pending} | busy)
            rows = sum(len(q) for _, q in pending)
            if waiting >= len(conns) or rows >= max_rows or time.perf_counter() >= first_t + wait:
                launch()
    finally:
        if th is not None:
            jobs.put(None)
            th.join(timeout=5)
        sel.close()
        lis.close()
        wake_r.close(); wake_w.close()
        for c in list(conns.values()):
            c.sock.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
    return st


class InferClient:
    """共用推理进程的客户端（同步：一次一个请求）。run(r) → (对数几率 float32[N, W], 价值 float32[N])，W 为这批的最大候选数。"""

    def __init__(self, path: str, timeout: float = 120.0, with_priv: bool = False):
        self.path, self.with_priv = path, with_priv
        t_end = time.time() + timeout
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                self.sock.close()
                if time.time() > t_end:
                    raise
                time.sleep(0.2)
        self.n_req = self.n_rows = 0
        self.t_wait = 0.0

    def run(self, r: Ragged) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        _send(self.sock, T_INFER, encode_ragged(r, self.with_priv))
        typ, body = _recv_msg(self.sock)
        if typ == T_ERROR:
            raise RuntimeError("推理进程出错：" + bytes(body).decode("utf-8", "replace"))
        n, w = _RESP.unpack_from(body, 0)
        logits = np.frombuffer(body, np.float32, n * w, _RESP.size).reshape(n, w)
        values = np.frombuffer(body, np.float32, n, _RESP.size + 4 * n * w)
        self.t_wait += time.perf_counter() - t0
        self.n_req += 1; self.n_rows += n
        return logits, values

    def stats(self) -> dict:
        _send(self.sock, T_STATS, [])
        typ, body = _recv_msg(self.sock)
        return json.loads(bytes(body))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sock.close()
