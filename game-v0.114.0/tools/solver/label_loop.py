"""本机标注循环（求解器筛选的 Boss 练习，见 python/sts2rl/solver_pool.py）：
从主线所在的租用机（lines.json 的 main，默认 A；每轮重新读，主线迁走后自动跟过去）取走待标注的 Boss 快照
（run_dir/solver/pending/*.json），用 Combat Solver 逐场求解，把结果追加到主线租用机的 run_dir/solver/labeled.jsonl（采集进程每 30 秒读一次），本机另存一份。

  setsid nohup python3 tools/solver/label_loop.py >> ~/.local/share/sls2/solver/label_loop.log 2>&1 < /dev/null &

环境变量：WORKERS 同时解几场（默认 5），SOLVER_DOP 每场的搜索线程数（默认 4），TIMEOUT 每场限时秒数（默认 120）。
模组文件放在 ~/.local/share/sls2/mods/{CombatSolver,RitsuLib}（从创意工坊目录复制，不进仓库）。
本机引擎必须与租用机同一版本、同一份代码：同一种子才能打出同一场战斗。
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[2]
DLL = REPO / "engine/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"
MODS = HOME / ".local/share/sls2/mods"
LOCAL = HOME / ".local/share/sls2/solver"
INBOX, OUTBOX = LOCAL / "inbox", LOCAL / "outbox.jsonl"
ARCHIVE = LOCAL / "labeled.jsonl"
WORKERS = int(os.environ.get("WORKERS", "5"))   # 3 → 5（2026-10-27：本机 16 核，3 路时负载约 7）
TIMEOUT = float(os.environ.get("TIMEOUT", "120"))
ENV = dict(os.environ, DOTNET_ROOT=str(HOME / ".dotnet"), PATH=f"{HOME / '.dotnet'}:{os.environ['PATH']}", DOTNET_CLI_TELEMETRY_OPTOUT="1",
           COMBAT_SOLVER_DIR=str(MODS / "CombatSolver"), RITSULIB_DIR=str(MODS / "RitsuLib"), SOLVER_DOP=os.environ.get("SOLVER_DOP", "4"))


def log(msg: str) -> None:
    print(time.strftime("%F %T"), msg, flush=True)


def main_remote() -> tuple[str, str, str]:
    """主线所在的租用机（~/.local/share/sls2/runs/remote/lines.json 的 main，文件不存在时为 a；主机登记见 tools/remote/hosts.json）。
    返回 (给 rsync -e 用的 ssh 命令, 用户@主机, 远端运行目录)。主机还没登记地址（pending）时抛 RuntimeError。"""
    hosts = json.loads((REPO / "tools/remote/hosts.json").read_text(encoding="utf-8"))["hosts"]
    try:
        name = json.loads((HOME / ".local/share/sls2/runs/remote/lines.json").read_text(encoding="utf-8"))["main"]["host"]
    except (OSError, ValueError, KeyError, TypeError):
        name = "a"
    h = hosts[name]
    if h.get("pending") or h.get("ssh_host") in (None, "", "待填"):
        raise RuntimeError(f"主线主机 {name} 还没登记地址")
    key = os.path.expanduser(h.get("key") or "")
    ssh = f"ssh -p {h['port']}" + (f" -i {key} -o IdentitiesOnly=yes" if key else "") + " -o ConnectTimeout=30 -o ServerAliveInterval=30"
    return ssh, f"{h['user']}@{h['ssh_host']}", h["run_dir"]


def pull() -> None:
    """取走主线租用机上待标注的快照（取到后从租用机删除；写到一半的 .tmp 不取）。"""
    INBOX.mkdir(parents=True, exist_ok=True)
    try:
        ssh, addr, run = main_remote()
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        log(f"拉取跳过：{e}")
        return
    subprocess.run(["rsync", "-a", "--remove-source-files", "--include=*.json", "--exclude=*", "-e", ssh,
                    f"{addr}:{run}/solver/pending/", f"{INBOX}/"], timeout=300, check=False, capture_output=True)


def solve(path: Path) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    try:
        p = subprocess.run(["dotnet", str(DLL), "solve-test", str(path), "0", "--timeout-s", str(TIMEOUT)], env=ENV,
                           capture_output=True, text=True, timeout=TIMEOUT + 120)
        out = next((json.loads(l) for l in reversed(p.stdout.splitlines()) if l.startswith('{"id"')), None)
        err = None if out else (p.stdout + p.stderr)[-300:]
    except subprocess.TimeoutExpired:
        out, err = None, "进程超时"
    res = {"id": rec["id"], "act": rec["label"]["act"], "encounter": rec["label"]["encounter"], "spec": rec["spec"]}
    if out is None or out.get("error"):
        res.update(win=None, error=(out or {}).get("error") or err)
    else:
        # 搜索超时没有结论时，返回的是未分胜负的最好路线：不算能赢
        res.update(win=bool(out["win"]) and not out["timed_out"], end_hp=out.get("end_hp"), start_hp=out["start_hp"], deck=out["deck"],
                   enemy_hp_left=out.get("enemy_hp_left"), end_turn=out.get("end_turn"), solve_ms=out["solve_ms"], timed_out=out["timed_out"])
    return res


def push() -> bool:
    """把攒下的结果追加到主线租用机的标签文件；失败就留着下次再推。"""
    if not OUTBOX.exists() or OUTBOX.stat().st_size == 0:
        return True
    data = OUTBOX.read_bytes()
    try:
        ssh, addr, run = main_remote()
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        log(f"推送跳过：{e}")
        return False
    p = subprocess.run(shlex.split(ssh) + [addr, f"mkdir -p {run}/solver && cat >> {run}/solver/labeled.jsonl"], input=data, timeout=300,
                       capture_output=True, check=False)
    if p.returncode != 0:
        log(f"推送失败：{p.stderr.decode(errors='replace')[-200:]}")
        return False
    with open(ARCHIVE, "ab") as f:
        f.write(data)
    OUTBOX.unlink()
    return True


# 积压里导出超过这么多小时还没轮到的快照直接丢掉（牌组跟着策略变，旧快照的价值低）。2026-10-27 起：第三幕的导出比本机解得快，
# 第二幕的快照因为第三幕优先一直轮不到（当时积压约 1 万份、中位数 13 小时），第一、二幕 6 小时，第三幕 8 小时
STALE_H = {0: 6, 1: 6, 2: 8}
SYNC_S = 60     # 每隔这么多秒拉一次新快照、推一次结果


def act_of(f: Path) -> int:
    head = f.read_text(encoding="utf-8")[:200]
    return 2 if '"act": 2' in head else 1 if '"act": 1' in head else 0


def main() -> None:
    """持续调度：WORKERS 场同时在解，解完一场马上补一场（原来按 4 × WORKERS 场一批、等整批最慢的那场解完才开下一批）。"""
    LOCAL.mkdir(parents=True, exist_ok=True)
    log(f"开始：{WORKERS} 个进程，每场 {ENV['SOLVER_DOP']} 线程、限时 {TIMEOUT:.0f} 秒")
    acts: dict[Path, int] = {}      # 文件 → 幕（只读一次文件头）
    running: dict = {}              # future → 文件
    queue: list[Path] = []
    last_sync = 0.0
    n_done = n_win = n_err = 0
    t_rep = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        while True:
            if time.time() - last_sync > SYNC_S or not queue:
                last_sync = time.time()
                try:
                    pull()
                except Exception as e:      # 网络断了下次再试
                    log(f"拉取失败：{e}")
                ok = push()
                files = [f for f in INBOX.glob("*.json") if f not in running.values()]
                now = time.time()
                for f in files:
                    if f not in acts:
                        try:
                            acts[f] = act_of(f)
                        except OSError:
                            acts[f] = 0
                stale = [f for f in files if now - f.stat().st_mtime > STALE_H[acts[f]] * 3600]
                if stale:
                    c = [sum(acts[f] == a for f in stale) for a in range(3)]
                    log(f"丢弃 {len(stale)} 份过期快照（第一、二、三幕 {c[0]}、{c[1]}、{c[2]} 份）")
                for f in stale:
                    f.unlink(missing_ok=True); acts.pop(f, None)
                files = [f for f in files if f not in stale]
                # 第三幕优先，其次第二幕，最后第一幕；同一幕先解最新导出的
                files.sort(key=lambda f: (-acts[f], -f.stat().st_mtime))
                queue = files
                if n_done >= WORKERS * 4 or (n_done and time.time() - t_rep > 600):
                    log(f"解了 {n_done} 场（能赢 {n_win}，出错 {n_err}），{time.time() - t_rep:.0f} 秒；积压 {len(queue)}；推送{'成功' if ok else '失败'}")
                    n_done = n_win = n_err = 0; t_rep = time.time()
            while len(running) < WORKERS and queue:
                f = queue.pop(0)
                if f.exists():
                    running[pool.submit(solve, f)] = f
            if not running:
                time.sleep(20)
                continue
            done, _ = wait(running, timeout=30, return_when=FIRST_COMPLETED)
            for fu in done:
                f = running.pop(fu)
                try:
                    r = fu.result()
                except Exception as e:      # 读不了的文件：记一次出错，丢掉
                    r = None
                    log(f"求解异常 {f.name}：{e!r}")
                if r is not None:
                    with open(OUTBOX, "a", encoding="utf-8") as out:
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n_done += 1; n_win += bool(r.get("win")); n_err += r.get("win") is None
                f.unlink(missing_ok=True); acts.pop(f, None)


if __name__ == "__main__":
    sys.exit(main())
