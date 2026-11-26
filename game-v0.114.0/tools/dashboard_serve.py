"""本机看板服务（2026-11-06，用户：“看板只在本机访问……重置通关率、遗物抓取和卡牌抓取的按钮要在本地能够生效，并且不需要输入密钥。”）

只监听 127.0.0.1（Windows 的浏览器经 WSL 的本机转发访问 http://localhost:8765/）：
  GET  /        返回 tools/dashboard_lite.py 生成的本机看板（每次现读文件）
  POST /reset   “重新计数”：把训练整局通关率、选牌、先古遗物三项统计的起点挪到学习端当前的版本——
                写 B 的运行目录 lite_since.json（tools/remote/lite_stats.py 读它，从起点累计），本机另留一份 stats_since.json，
                再立即拉一次统计、重新生成看板。不需要口令：只有本机能连到这个端口。

    setsid nohup python3 tools/dashboard_serve.py > ~/.local/share/sls2/runs/remote/dashboard_serve.log 2>&1 &
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import dashboard_lite as DL  # noqa: E402

PORT = 8765
LOCAL_SINCE = DL.L / "stats_since.json"


def remote(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """在 B 上执行一条命令（连接方式同 tools/remote/hosts.sh 的 host_ssh）。"""
    return subprocess.run(["bash", "-c", f'source "{REPO}/tools/remote/hosts.sh"; $(host_ssh b) "$1"', "_", cmd],
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def reset() -> dict:
    st = remote("cat $HOME/.local/share/sls2/runs/defect/status.json")
    try:
        version = int(json.loads(st.stdout)["version"])
    except (ValueError, KeyError, TypeError):
        raise RuntimeError(f"读不到学习端当前版本：{st.stderr.strip()[:200]}")
    since = {"version": version, "time": time.time(), "text": time.strftime("%m-%d %H:%M")}
    blob = json.dumps(since, ensure_ascii=False)
    w = remote(f"cd $HOME/.local/share/sls2/runs/defect && printf '%s' '{blob}' > lite_since.json.tmp && mv lite_since.json.tmp lite_since.json")
    if w.returncode != 0:
        raise RuntimeError(f"写不进 B：{w.stderr.strip()[:200]}")
    LOCAL_SINCE.write_text(blob, encoding="utf-8")
    # 立即拉一次 B 的统计并重新生成看板（不等常驻拉取的下一轮）
    r = remote("cd /root/sls2-deploy/sls2 && .venv/bin/python -I tools/remote/lite_stats.py train", timeout=120)
    try:
        json.loads(r.stdout)
        tmp = DL.L / "lite_b.json.tmp"; tmp.write_text(r.stdout, encoding="utf-8"); tmp.replace(DL.L / "lite_b.json")
        DL.main()
    except ValueError:
        pass
    return since


class H(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            try:
                self._send(200, DL.OUT.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._send(503, "看板还没生成".encode(), "text/plain; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/reset":
            return self._send(404, b"not found", "text/plain")
        try:
            out = {"ok": True, **reset()}
        except Exception as e:      # 报给页面，不让服务退出
            out = {"ok": False, "error": str(e)}
        self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        sys.stderr.write(time.strftime("%m-%d %H:%M:%S ") + fmt % args + "\n")


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
