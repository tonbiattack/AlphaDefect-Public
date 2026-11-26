"""生成看板选牌对照搜索用的拼音表 tools/card_pinyin.json：{牌 ID: "全拼 首字母"}（只在游戏更新、卡牌中文名变了时重跑）。
    uv run --with pypinyin python tools/gen_card_pinyin.py
看板本身不依赖 pypinyin，只读生成好的表。"""
import json
from pathlib import Path
from pypinyin import lazy_pinyin

LOC = Path.home() / ".local/share/sls2/game/v0.111.0/res/localization/zhs/cards.json"
OUT = Path(__file__).with_name("card_pinyin.json")
cn = {k[:-6]: v for k, v in json.loads(LOC.read_text(encoding="utf-8")).items() if k.endswith(".title")}
out = {}
for k, v in sorted(cn.items()):
    py = [p for p in lazy_pinyin(v) if p.strip()]
    out[k] = "".join(py).lower() + " " + "".join(p[0] for p in py).lower()
OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(len(out), "张牌 →", OUT)
