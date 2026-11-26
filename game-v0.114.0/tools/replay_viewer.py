"""对局回放播放器的生成：从主线所在的租用机（lines.json 的 main，默认 A）拉回评测存下的回放（通关的局），连同中文名表一起生成一个本地网页。

  python3 tools/replay_viewer.py              # 拉取并生成：Windows 原目录下的 回放/（双击 index.html 打开）与看板公开副本的 replay/
  python3 tools/replay_viewer.py --no-pull --src 目录 --out 目录

名字和描述取自本机游戏文件的中文本地化，美术取自本机的游戏资源包（都不进仓库）。看板公开副本里的那一份随看板推到
香港服务器（2026-10-26 用户决定：回放页带美术和中文名直接放上去，链接不对外公开，只为在手机上看）。
网页不联网：回放数据写成 .js 文件，用 <script> 加载，本机直接用浏览器打开文件即可。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()
LOC = HOME / ".local/share/sls2/game/v0.111.0/res/localization"
LOCAL = HOME / ".local/share/sls2/runs/remote/replays"
OUT = Path("/mnt/c/主存储文件/脚本开发/sls2/回放")
PUBLIC = HOME / ".local/share/sls2/runs/remote/public/replay"     # 看板公开副本的子页面，由 tools/remote/publish_replay.sh 推到香港服务器
KEEP = 300      # 播放器里保留最新的多少局
PULL_N = 2 * KEEP   # 每台、每个回放目录只拉最新的这么多局（pull_recent）


def load_loc(lang: str) -> dict[str, dict]:
    d = LOC / lang
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.json")} if d.exists() else {}


def clean(text: str | None) -> str:
    """去掉富文本标记（[gold]…[/gold] 等）；变量占位与条件文本原样保留，由播放器按游戏的格式语法代入。"""
    if not text:
        return ""
    return re.sub(r"\[/?[a-zA-Z_]+[^\]]*\]", "", text).strip()


def names(vocab: list[str]) -> list[list]:
    """词表编号 → [名字, 描述, 标签, 数值]。找不到中文的用英文，再找不到用键本身。
    标签：卡牌为“类型|稀有度”（Attack、Skill、Power、Curse、Status…），能力为 Buff / Debuff；数值为卡牌的基础数值（描述里的占位）；
    之后是键名和卡牌的基础费用（X 费为 −1）。"""
    zh, en = load_loc("zhs"), load_loc("eng")
    cat_doc = json.loads((REPO / "engine/catalog/v0.111.0/catalog.json").read_text(encoding="utf-8"))
    ents = {e["key"]: e for e in cat_doc["entities"]}

    def get(table: str, key: str) -> str | None:
        return zh.get(table, {}).get(key) or en.get(table, {}).get(key)

    out = []
    for k in vocab:
        name, desc = None, None
        if "." in k and ":" not in k:
            cat, entry = k.split(".", 1)
            table = {"CARD": "cards", "RELIC": "relics", "POTION": "potions", "POWER": "powers", "ORB": "orbs", "ENCHANTMENT": "enchantments",
                     "AFFLICTION": "afflictions", "ENCOUNTER": "encounters", "CHARACTER": "characters"}.get(cat)
            if cat == "MONSTER":
                # 分节的怪物（DECIMILLIPEDE_SEGMENT_FRONT 等）共用一个名字
                name = get("monsters", f"{entry}.name") or get("monsters", f"{re.sub(r'_(FRONT|BACK|MIDDLE|LEFT|RIGHT)$', '', entry)}.name")
            elif cat == "EVENT":
                name = get("events", f"{entry}.title") or get("ancients", f"{entry}.title")
            elif table:
                name = get(table, f"{entry}.title") or get(table, f"{entry}.name")
                # 能力的 smartDescription 带层数占位（{Amount}），description 是不带数的版本
                desc = (get(table, f"{entry}.smartDescription") if cat == "POWER" else None) or get(table, f"{entry}.description")
                if cat == "POWER" and not name:
                    # 卡牌、药水、遗物附带的临时能力（ANTICIPATE_POWER、FLEX_POTION_POWER、HYPERBEAM_FOCUS_DOWN_POWER 等）没有单独的名字，
                    # 用来源的名字：逐段去掉结尾再找
                    parts = entry.removesuffix("_POWER").split("_")
                    for n in range(len(parts), 0, -1):
                        base = "_".join(parts[:n])
                        src = next((t for t in ("cards", "potions", "relics") if get(t, f"{base}.title")), None)
                        if src:
                            name = get(src, f"{base}.title"); desc = desc or get(src, f"{base}.description")
                            break
        elif k.startswith("MOVE:"):
            mon, mv = k[5:].split(".", 1)
            mon = re.sub(r"_(FRONT|BACK|MIDDLE|LEFT|RIGHT)$", "", mon)
            moves = zh.get("monsters", {})
            base = re.sub(r"_?\d+$", "", mv.removesuffix("_MOVE").replace("_MOVE_", "_"))
            name = get("monsters", f"{mon}.moves.{mv.removesuffix('_MOVE')}.title") or get("monsters", f"{mon}.moves.{base}.title")
            if not name:   # 状态机里的招式编号和本地化的键对不上时（SWING_1 对 BIG_SWING），取名字包含它的那一招
                cand = [key for key in moves if key.startswith(f"{mon}.moves.") and key.endswith(".title")
                        and (base in key or key.split(".")[2] in base)]
                name = moves[cand[0]] if cand else {"STUN": "眩晕", "NOTHING": "无行动"}.get(base, base.replace("_", " ").title())
        elif k.startswith("EVENTOPT:"):
            key = k[9:]
            name = get("events", f"{key}.title") or get("ancients", f"{key}.title")
            desc = get("events", f"{key}.description") or get("ancients", f"{key}.description")
            ev = key.split(".")[0]
            evname = get("events", f"{ev}.title") or get("ancients", f"{ev}.title")
            if evname and name:
                name = f"{evname} · {name}"
        elif k.startswith("REST:"):
            name = get("rest_site_ui", f"OPTION_{k[5:]}.name")
            desc = get("rest_site_ui", f"OPTION_{k[5:]}.description")
        elif k.startswith("ALT:"):
            name = get("card_reward_ui", f"OPTION_{k[4:]}.name")
        elif k.startswith("INTENT:"):
            name = get("intents", re.sub(r"(?<!^)(?=[A-Z])", "_", k[7:]).upper() + ".title") or {"Hidden": "未知", "StatusCard": "塞状态牌"}.get(k[7:])
        elif k.startswith("REWARD:"):
            name = {"GoldReward": "金币", "CardReward": "卡牌奖励", "PotionReward": "药水", "RelicReward": "遗物", "SpecialCardReward": "特殊卡牌奖励",
                    "CardRemovalReward": "删牌", "LinkedRewardSet": "成组奖励"}.get(k[7:], k[7:])
        elif k.startswith("MAPNODE:"):
            name = {"Monster": "普通战斗", "Elite": "精英", "Boss": "Boss", "RestSite": "休息处", "Shop": "商店", "Treasure": "宝箱",
                    "Unknown": "问号", "Ancient": "古神", "Unassigned": "未定"}.get(k[8:], k[8:])
        elif k.startswith("SHOP:"):
            name = "删牌服务"
        e = ents.get(k, {})
        tag = f"{e.get('type', '')}|{e.get('rarity', '')}" if k.startswith("CARD.") else e.get("type", "")
        cost = (-1 if e.get("cost_x") else e.get("cost")) if k.startswith("CARD.") else None
        out.append([clean(name) or k, clean(desc), tag, e.get("vars") or {}, k, cost])
    return out


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


def pull_recent(ssh: str, addr: str, rdir: str, n: int = PULL_N) -> subprocess.CompletedProcess:
    """只拉远端目录里最新的 n 个回放（2026-11-03 起，为省中转流量）：原来整目录 rsync，A 上 21 万个文件光清单就 8 MB，
    每天还要下载约 5.4 万个新回放（3.7 GB），播放器却只用最新的 KEEP 局。先在远端按修改时间列出最新 n 个文件名，再 --files-from 只拉这些。"""
    ls = subprocess.run(ssh.split() + [addr, f"cd {rdir} 2>/dev/null && ls -t | grep '\\.jsonl\\.gz$' | head -{n}"],
                        capture_output=True, text=True, timeout=120)
    names = [x for x in ls.stdout.splitlines() if x.endswith(".jsonl.gz") and "/" not in x]
    if not names:
        return subprocess.CompletedProcess(ls.args, 0 if ls.returncode in (0, 1) else ls.returncode, "", ls.stderr)
    lst = LOCAL / ".pull_list.tmp"
    lst.write_text("\n".join(names) + "\n", encoding="utf-8")
    return subprocess.run(["rsync", "-az", f"--files-from={lst}", "-e", ssh, f"{addr}:{rdir}/", f"{LOCAL}/"],
                          capture_output=True, text=True, timeout=600)


def pull() -> None:
    LOCAL.mkdir(parents=True, exist_ok=True)
    try:
        ssh, addr, run = main_remote()
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        print("拉取跳过：", e)
        return
    r = pull_recent(ssh, addr, f"{run}/replays")
    if r.returncode not in (0, 23):
        print("拉取失败：", r.stderr.strip()[:300])
    # 常规评测 2026-10-29 起在协同采集机上跑（tools/remote/remote_eval.py，决策记录 198），通关的回放在那台的 defect-eval/replays/
    hosts = json.loads((REPO / "tools/remote/hosts.json").read_text(encoding="utf-8"))["hosts"]
    for h in hosts.values():
        if h.get("pending") or h.get("ssh_host") in (None, "", "待填"):
            continue
        key = os.path.expanduser(h.get("key") or "")
        hs = f"ssh -p {h['port']}" + (f" -i {key} -o IdentitiesOnly=yes" if key else "") + " -o ConnectTimeout=30 -o ServerAliveInterval=30"
        pull_recent(hs, f"{h['user']}@{h['ssh_host']}", ".local/share/sls2/runs/defect-eval/replays")


def write_if_changed(p: Path, text: str) -> None:
    """内容没变就不写（发布脚本按修改时间判断要推哪些文件）。"""
    if not p.exists() or p.read_text(encoding="utf-8") != text:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(p)


def build(src: Path, out: Path, home: str, keep: int = KEEP) -> None:
    """生成播放器：最新的 keep 局（按时间），其余的数据文件删掉；home 为页面上“返回看板”的链接。"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    vocab = (REPO / "engine/catalog/vocab.txt").read_text(encoding="utf-8").split("\n")
    nm = names(vocab)
    enc = {k.split(".", 1)[1]: nm[i][0] for i, k in enumerate(vocab) if k.startswith("ENCOUNTER.")}
    write_if_changed(out / "names.js", "window.NAMES=" + json.dumps(nm, ensure_ascii=False, separators=(",", ":")) + ";\nwindow.ENC="
                     + json.dumps(enc, ensure_ascii=False) + ";\n")
    heads = []
    for p in src.glob("*.jsonl.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            head = json.loads(f.readline())
        heads.append((head.get("time") or 0, p, head))
    heads.sort(key=lambda x: -x[0])
    manifest = []
    for _, p, head in heads[:keep]:
        rid = p.name.removesuffix(".jsonl.gz")
        target = out / "data" / f"{rid}.js"
        if not target.exists() or target.stat().st_mtime < p.stat().st_mtime:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                lines = f.read().splitlines()
            target.write_text(f"window.REPLAY_LOADED({json.dumps(rid)},{lines[0]},[{','.join(lines[1:])}]);\n", encoding="utf-8")
        manifest.append({"id": rid, **{k: head.get(k) for k in ("version", "seed", "time", "win", "floor", "act", "steps")}})
    ids = {m["id"] for m in manifest}
    for f in (out / "data").glob("*.js"):
        if f.stem not in ids:
            f.unlink()
    write_if_changed(out / "replays.js", "window.REPLAYS=" + json.dumps(manifest, ensure_ascii=False) + ";\nwindow.HOME_URL=" + json.dumps(home) + ";\n")
    page = (REPO / "tools/replay_player.html").read_text(encoding="utf-8")
    write_if_changed(out / "index.html", page)
    # 游戏美术（本机资源包里取，已有的跳过；需要虚拟环境里的 Pillow）
    r = subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "tools/replay_assets.py"), str(out)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-300:])
    print(f"回放 {len(manifest)} 局，输出到 {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--src", default=str(LOCAL))
    ap.add_argument("--out", help="只生成到这个目录（默认生成 Windows 侧的 回放/ 和看板公开副本的 replay/ 两份）")
    a = ap.parse_args()
    if not a.no_pull:
        pull()
    if a.out:
        build(Path(a.src), Path(a.out), "../训练看板.html")
    else:
        build(Path(a.src), OUT, "../训练看板.html")
        build(Path(a.src), PUBLIC, "../")     # 香港服务器上看板在站点根（/），只认 / 不认 /index.html
