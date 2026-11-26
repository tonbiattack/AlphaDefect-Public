"""从 Spire Codex（spire-codex.com，社区上传对局的统计）拉某个角色、某个进阶等级、某个游戏版本的选牌统计，存成看板对照用的 JSON。

    python3 tools/fetch_card_stats.py                     # 默认：故障机器人、单人、A10、v0.111.0
    python3 tools/fetch_card_stats.py --bracket solo:a10:v0.111.0 --character DEFECT

接口：GET https://spire-codex.com/api/runs/metrics/cards?bracket=…&character=…（每次 1 个额度，每小时共 120）。
每张牌：offered / picked 为奖励界面被提供、被选中的次数，pick_rate_by_act 为分幕选取率（%），
win_rate 为牌组里有这张牌的对局胜率（%），lift 为按拿到它的层数校正后的胜率差（个百分点，相对这些玩家自己在该层的期望胜率）。
人类数据带选择偏差，只作对照，不直接当奖励。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "engine" / "card_stats"
KEEP = ("id", "offered", "picked", "pick_rate", "pick_rate_by_act", "win_rate", "win_rate_ci", "lift", "lift_n", "tier")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracket", default="solo:a10:v0.111.0")
    ap.add_argument("--character", default="DEFECT")
    a = ap.parse_args()
    url = f"https://spire-codex.com/api/runs/metrics/cards?bracket={a.bracket}&character={a.character}"
    req = urllib.request.Request(url, headers={"User-Agent": "sls2-card-stats/1"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    rows = [{k: r.get(k) for k in KEEP} for r in d["rows"] if not r.get("upgraded")]
    out = {"source": url, "fetched": time.strftime("%Y-%m-%d %H:%M"), "data_through": d.get("data_through"),
           "bracket": d.get("bracket"), "character": d.get("character"), "character_runs": d.get("character_runs"),
           "character_wins": d.get("character_wins"), "rows": sorted(rows, key=lambda r: r["id"])}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"codex_{a.character.lower()}_{a.bracket.replace(':', '_')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{path.relative_to(REPO)}：{len(rows)} 张牌，{out['character_runs']} 局（胜 {out['character_wins']}），数据截至 {out['data_through']}")


if __name__ == "__main__":
    main()
