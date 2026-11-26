"""搜索蒸馏的本机中转（2026-10-27 起）：两台租用机不直接通讯，题目、标签、检查点都经本机转。每 CYCLE 秒一轮：

1. 标签：取走 A 上 ~/distill/out/*.pkl.gz（取到后从 A 删除），推到主线租用机的 run_dir/distill/inbox/，本机留档 ARCHIVE（保留 KEEP_DAYS 天）。
2. 题目：A 上待做的题少于 LOW 道时补到 HIGH 道。题目取自本机求解器目录（主线导出的第二、三幕 Boss 战开打快照：待标注收件箱
   与已标注存档），只取最近 HOURS 小时导出的、没发过的，排除考卷（主线 run_dir/exam.json）里的题，两幕各一半、新的优先。
3. 检查点：主线比 A 正在用的新出 REFRESH_V 版以上时，取主线 latest.pt，去掉优化器状态后推到 A 的 ~/distill/ckpt/latest.pt
   （A 的标签进程每批开始前看一眼，变了就换）。
4. A 上的标签进程不足 WORKERS 个时，按 tools/distill/launch.sh 补起。

    setsid nohup python3 tools/distill/relay.py >> ~/.local/share/sls2/distill/relay.log 2>&1 < /dev/null &

本机守护 tools/remote/local_watch.sh 在 ~/.local/share/sls2/distill/ENABLED 存在时保证本进程在跑；删掉这个文件即停用（同时要停 A 上的标签进程）。
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[2]
LOCAL = HOME / ".local/share/sls2/distill"
STAGE, ARCHIVE = LOCAL / "stage", LOCAL / "archive"
SENT = LOCAL / "sent_ids.txt"
SUMMARY = LOCAL / "labels.jsonl"     # 每个标签文件一行摘要（标签进程写的 .json），看板用
STATE = LOCAL / "state.json"
SOLVER = HOME / ".local/share/sls2/solver"
REMOTE = HOME / ".local/share/sls2/runs/remote"
LABEL_HOST = os.environ.get("LABEL_HOST", "a")
CYCLE = 60
LOW, HIGH = 150, 400
HOURS = 6.0
REFRESH_V = 1000
WORKERS = int(os.environ.get("WORKERS", "16"))
KEEP_DAYS = 3


def log(msg: str) -> None:
    print(time.strftime("%F %T"), msg, flush=True)


def host(name: str) -> dict:
    h = json.loads((REPO / "tools/remote/hosts.json").read_text(encoding="utf-8"))["hosts"][name]
    key = os.path.expanduser(h.get("key") or "")
    h["ssh"] = f"ssh -p {h['port']}" + (f" -i {key} -o IdentitiesOnly=yes" if key else "") + " -o ConnectTimeout=30 -o ServerAliveInterval=30"
    h["addr"] = f"{h['user']}@{h['ssh_host']}"
    return h


def main_host() -> str:
    try:
        return json.loads((REMOTE / "lines.json").read_text(encoding="utf-8"))["main"]["host"]
    except (OSError, ValueError, KeyError, TypeError):
        return "a"


def ssh(h: dict, cmd: str, timeout: float = 120, data: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(h["ssh"]) + [h["addr"], cmd], input=data, capture_output=True, timeout=timeout, check=False)


def rsync(h: dict, args: list[str], timeout: float = 600) -> bool:
    p = subprocess.run(["rsync", "-a", "-e", h["ssh"], *args], capture_output=True, timeout=timeout, check=False)
    if p.returncode != 0:
        log(f"rsync 失败：{p.stderr.decode(errors='replace')[-200:]}")
    return p.returncode == 0


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8"); os.replace(tmp, STATE)


def relay_labels(a: dict, m: dict) -> int:
    """A 上的标签 → 本机暂存 → 主线 inbox；推成功的挪进留档。返回这轮推过去的文件数。"""
    STAGE.mkdir(parents=True, exist_ok=True)
    # 先取摘要再取标签：摘要比标签文件晚写，取到摘要时它的标签文件一定已经写完
    rsync(a, ["--remove-source-files", "--include=*.json", "--exclude=*", f"{a['addr']}:distill/out/", f"{STAGE}/"])
    rsync(a, ["--remove-source-files", "--include=*.pkl.gz", "--exclude=*", f"{a['addr']}:distill/out/", f"{STAGE}/"])
    summ = sorted(STAGE.glob("*.json"))
    if summ:
        with open(SUMMARY, "a", encoding="utf-8") as fh:
            for f in summ:
                try:
                    fh.write(json.dumps(json.loads(f.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
                except (OSError, ValueError):
                    pass
                f.unlink(missing_ok=True)
    files = sorted(STAGE.glob("*.pkl.gz"))
    if not files:
        return 0
    if not rsync(m, [*map(str, files), f"{m['addr']}:{m['run_dir']}/distill/inbox/"]):
        ssh(m, f"mkdir -p {m['run_dir']}/distill/inbox")
        return 0
    day = ARCHIVE / time.strftime("%Y%m%d"); day.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), day / f.name)
    return len(files)


def prune_archive() -> None:
    for d in ARCHIVE.glob("*"):
        if d.is_dir() and time.time() - d.stat().st_mtime > KEEP_DAYS * 86400:
            shutil.rmtree(d, ignore_errors=True)


def t_of(rid: str) -> int:
    """快照编号 r<采集进程>_<导出时刻>_<序号> 里的导出时刻。"""
    try:
        return int(rid.split("_")[1])
    except (IndexError, ValueError):
        return 0


def exam_ids(m: dict, st: dict) -> set[str]:
    """考卷的题号（每小时从主线取一次 exam.json）。"""
    f = LOCAL / "exam.json"
    if not f.exists() or time.time() - st.get("exam_t", 0) > 3600:
        if rsync(m, [f"{m['addr']}:{m['run_dir']}/exam.json", str(f)]):
            st["exam_t"] = time.time()
    try:
        return {x["id"] for x in json.loads(f.read_text(encoding="utf-8"))["items"]}
    except (OSError, ValueError, KeyError):
        return set()


def candidates(skip: set[str]) -> dict[int, list[dict]]:
    """本机求解器目录里最近 HOURS 小时导出的第二、三幕 Boss 快照（没发过、不在考卷里），按幕、新的在前。"""
    pool: dict[str, dict] = {}
    for f in (SOLVER / "inbox").glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            pool[r["id"]] = {"id": r["id"], "act": r["label"]["act"], "encounter": r["label"]["encounter"], "spec": r["spec"]}
        except (OSError, ValueError, KeyError):
            continue
    lab = SOLVER / "labeled.jsonl"
    if lab.exists():
        with open(lab, "rb") as fh:          # 只看末尾一段（新的追加在后面）
            fh.seek(max(0, lab.stat().st_size - 30_000_000))
            for l in fh.read().splitlines()[1:]:
                try:
                    r = json.loads(l)
                    pool.setdefault(r["id"], {"id": r["id"], "act": r["act"], "encounter": r["encounter"], "spec": r["spec"]})
                except (ValueError, KeyError):
                    continue
    newest = max((t_of(i) for i in pool), default=0)
    out: dict[int, list[dict]] = {1: [], 2: []}
    for x in pool.values():
        if x["act"] in out and x["id"] not in skip and newest - t_of(x["id"]) <= HOURS * 3600 and x["spec"].get("player_snapshot"):
            spec = dict(x["spec"]); spec["room"] = "Boss"
            out[x["act"]].append(dict(x, spec=spec))
    for v in out.values():
        v.sort(key=lambda x: -t_of(x["id"]))
    return out


def top_up(a: dict, m: dict, st: dict) -> int:
    p = ssh(a, "mkdir -p distill/problems && ls distill/problems | grep -c '\\.json$' || true")
    try:
        have = int(p.stdout.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0
    if have >= LOW:
        return 0
    sent = set(SENT.read_text(encoding="utf-8").split()) if SENT.exists() else set()
    cand = candidates(sent | exam_ids(m, st))
    need = HIGH - have
    pick = []
    while len(pick) < need and (cand[1] or cand[2]):
        for act in (1, 2):
            if cand[act] and len(pick) < need:
                pick.append(cand[act].pop(0))
    if not pick:
        return 0
    tmp = LOCAL / "out_problems"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    for x in pick:
        (tmp / f"{x['id']}.json").write_text(json.dumps(x, ensure_ascii=False), encoding="utf-8")
    # 先传到临时目录再整体挪进题目目录：标签进程不会认领到传了一半的文件
    if not rsync(a, [f"{tmp}/", f"{a['addr']}:distill/problems/incoming/"]):
        return 0
    ssh(a, "cd distill/problems && mv incoming/*.json . 2>/dev/null; true")
    with open(SENT, "a", encoding="utf-8") as fh:
        fh.write("\n".join(x["id"] for x in pick) + "\n")
    return len(pick)


def main_version(mname: str) -> int | None:
    local = json.loads((REPO / "tools/remote/hosts.json").read_text(encoding="utf-8"))["hosts"][mname]["local"]
    try:
        return int(json.loads((REMOTE / local / "status.json").read_text(encoding="utf-8"))["version"])
    except (OSError, ValueError, KeyError):
        return None


def refresh_ckpt(a: dict, m: dict, mname: str, st: dict) -> None:
    v = main_version(mname)
    if v is None or (st.get("ckpt_version") is not None and v - st["ckpt_version"] < REFRESH_V):
        return
    tmp = LOCAL / "ckpt"; tmp.mkdir(parents=True, exist_ok=True)
    if not rsync(m, [f"{m['addr']}:{m['run_dir']}/latest.pt", str(tmp / "full.pt")]):
        return
    py = REPO / ".venv/bin/python"
    p = subprocess.run([str(py), "-I", "-c", "import sys, torch; b = torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
                        "torch.save({k: b[k] for k in ('version', 'obs_format', 'config', 'state') if k in b}, sys.argv[2]); print(b['version'])",
                        str(tmp / "full.pt"), str(tmp / "slim.pt")], capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        log(f"检查点精简失败：{p.stderr[-300:]}")
        return
    ver = int(p.stdout.strip().splitlines()[-1])
    ssh(a, "mkdir -p distill/ckpt")       # 首次推送时目录还不存在，rsync 不会建上一级目录
    if rsync(a, [str(tmp / "slim.pt"), f"{a['addr']}:distill/ckpt/.incoming.pt"]) and \
            ssh(a, "mv -f distill/ckpt/.incoming.pt distill/ckpt/latest.pt").returncode == 0:
        st["ckpt_version"] = ver; st["ckpt_t"] = time.time()
        log(f"A 的检查点换成主线第 {ver} 版")
    (tmp / "full.pt").unlink(missing_ok=True)


def ensure_workers(a: dict) -> None:
    p = ssh(a, "pgrep -fc 'tools/distill/label_worker.py' || true")
    try:
        n = int(p.stdout.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        return
    if n < WORKERS:
        d = a["deploy"]
        r = ssh(a, f"N={WORKERS} bash {d}/tools/distill/launch.sh", timeout=120)
        log(f"A 上标签进程 {n} 个，补起：{r.stdout.decode(errors='replace').strip()[-200:]}")


def main() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    log("开始")
    st = load_state()
    while True:
        t0 = time.time()
        try:
            mname = main_host()
            a, m = host(LABEL_HOST), host(mname)
            refresh_ckpt(a, m, mname, st)
            ensure_workers(a)
            n_lab = relay_labels(a, m)
            n_prob = top_up(a, m, st)
            prune_archive()
            st["last_ok"] = time.time()
            st["labels_total"] = st.get("labels_total", 0) + n_lab
            save_state(st)
            if n_lab or n_prob:
                log(f"标签文件 {n_lab} 个转给主线 {mname}，补题 {n_prob} 道")
        except Exception as e:       # 网络断了下一轮再试
            log(f"出错：{e!r}")
        time.sleep(max(5, CYCLE - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
