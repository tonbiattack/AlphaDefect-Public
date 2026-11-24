"""回放播放器用的游戏美术：从本机安装的游戏资源包（.pck）里取卡牌插图、遗物、能力、药水、意图、地图节点和顶栏图标，
缩小后存成 WebP，放在播放器目录的 assets/ 下（不进仓库；看板公开副本里的那份随回放页推到香港服务器）。已有的文件跳过。

  .venv/bin/python tools/replay_assets.py <输出目录>        # 一般由 replay_viewer.py 调用

Godot 的纹理（.ctex，非显存压缩）里直接嵌着一张 WebP 或 PNG；图集里的小图按 .tres 里的区域裁剪。
"""
from __future__ import annotations

import io
import os
import re
import struct
import sys
from pathlib import Path

from PIL import Image

PCK = Path(os.environ.get("STS2_PCK") or "/mnt/c/Program Files (x86)/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.pck")


class Pck:
    def __init__(self, path: Path = PCK):
        self.f = open(path, "rb")
        magic, fmt, *_ , flags = struct.unpack("<4s5I", self.f.read(24))
        if magic != b"GDPC":
            raise ValueError("不是 pck 文件")
        base = struct.unpack("<Q", self.f.read(8))[0]
        if fmt >= 3:
            d = struct.unpack("<Q", self.f.read(8))[0]
        else:
            self.f.read(64); d = self.f.tell()
        rel = base if flags & 2 else 0
        self.f.seek(d)
        self.files: dict[str, tuple[int, int]] = {}
        for _ in range(struct.unpack("<I", self.f.read(4))[0]):
            n = struct.unpack("<I", self.f.read(4))[0]
            p = self.f.read(n).rstrip(b"\0").decode("utf-8").removeprefix("res://")
            off, size = struct.unpack("<QQ", self.f.read(16)); self.f.read(20)
            self.files[p] = (off + rel, size)
        self._img: dict[str, Image.Image] = {}

    def read(self, p: str) -> bytes:
        off, size = self.files[p]
        self.f.seek(off)
        return self.f.read(size)

    def image(self, png: str) -> Image.Image:
        """res 路径下的 .png（经 .import 找到导入后的 .ctex）→ 图像。"""
        if png not in self._img:
            imp = self.read(png + ".import").decode("utf-8")
            ctex = re.search(r'path(?:\.bptc|\.s3tc)?="res://([^"]+\.ctex)"', imp).group(1)
            data = self.read(ctex)
            w, h = struct.unpack("<II", data[8:16])
            data_format, fmt = struct.unpack("<I", data[36:40])[0], struct.unpack("<I", data[48:52])[0]
            bcn = {17: 1, 18: 2, 19: 3, 22: 7}.get(fmt) if data_format == 0 else None
            if bcn:   # 显存压缩（图集）：按 4×4 块解码，宽高补到 4 的倍数再裁回
                pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
                self._img[png] = Image.frombytes("RGBA", (pw, ph), data[52:], "bcn", bcn).crop((0, 0, w, h))
            else:     # 非显存压缩：文件里直接嵌着一张 WebP 或 PNG
                k = min(i for i in (data.find(b"RIFF"), data.find(b"\x89PNG")) if i >= 0)
                self._img[png] = Image.open(io.BytesIO(data[k:])).convert("RGBA")
        return self._img[png]

    def sprite(self, tres: str) -> Image.Image:
        """图集小图（AtlasTexture 的 .tres）→ 裁好的图像。"""
        t = self.read(tres).decode("utf-8")
        atlas = re.search(r'path="res://([^"]+)"', t).group(1)
        x, y, w, h = (float(v) for v in re.search(r"region = Rect2\(([^)]+)\)", t).group(1).split(","))
        return self.image(atlas).crop((int(x), int(y), int(x + w), int(y + h)))


def save(img: Image.Image, out: Path, width: int | None = None, quality: int = 78) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if width and img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.save(out, "WEBP", quality=quality, method=6)


def build(out: Path, vocab: list[str]) -> int:
    pck = Pck()
    files = pck.files
    n = 0

    def put(dest: Path, make, width=None):
        nonlocal n
        if dest.exists():
            return
        try:
            save(make(), dest, width)
            n += 1
        except Exception:
            pass

    portraits = {}
    for p in files:
        # v0.113.0 起新角色愚者的插图在 card_portraits/fool/beta/ 下（多一层，且是显存压缩 BPTC，由 Pck.image 解码）
        m = re.match(r"images/packed/card_portraits/((?:[a-z_]+/)+)([a-z0-9_]+)\.png\.import$", p)
        if m:
            portraits.setdefault(m.group(2), p.removesuffix(".import"))
    # 不在图集里的遗物、能力、药水图（v0.113.0 新加的遗物只有 images/relics/<名>.png 或 images/relics/beta/<名>.png）
    loose = {}
    for p in files:
        m = re.match(r"images/(relics|powers|potions)/(?:beta/)?([a-z0-9_]+)\.png\.import$", p)
        if m:
            loose.setdefault((m.group(1), m.group(2)), p.removesuffix(".import"))
    for key in vocab:
        if "." not in key or ":" in key:
            continue
        cat, entry = key.split(".", 1)
        low = entry.lower()
        if cat == "CARD" and low in portraits:
            put(out / "cards" / f"{entry}.webp", lambda: pck.image(portraits[low]), 240)
        elif cat in ("RELIC", "POWER", "POTION"):
            tres = f"images/atlases/{cat.lower()}_atlas.sprites/{low}.tres"
            if tres in files:
                put(out / cat.lower() / f"{entry}.webp", lambda: pck.sprite(tres), 96)
            elif (cat.lower() + "s", low) in loose:
                put(out / cat.lower() / f"{entry}.webp", lambda: pck.image(loose[(cat.lower() + "s", low)]), 96)
    for p in files:   # 意图（每类第一帧与总图标）、地图节点、Boss 图标、顶栏、能量
        m = re.match(r"images/packed/intents/(intent_[a-z_]+|attack/intent_attack_\d)\.png\.import$", p)
        if m:
            put(out / "intents" / f"{m.group(1).split('/')[-1]}.webp", lambda: pck.image(p.removesuffix(".import")), 96)
        m = re.match(r"images/atlases/ui_atlas\.sprites/(map/icons/map_[a-z_]+|map/placeholder/[a-z_]+|top_bar/top_bar_[a-z]+|card/energy_defect)\.tres$", p)
        if m:
            put(out / "ui" / f"{m.group(1).split('/')[-1]}.webp", lambda: pck.sprite(p), 96)
        # Boss 图标：用战绩页（run_history）那一套，12 个 Boss 齐全、风格一致。地图上的占位图标（map/placeholder）只有 8 个，
        # 祭典兽、女王、贪食者的地图节点是骨骼动画，没有静态图（2026-10-27 回放里这三个 Boss 的图标显示不出来）
        m = re.match(r"images/ui/run_history/([a-z_]+_boss)\.png\.import$", p)
        if m:
            put(out / "ui" / f"{m.group(1)}.webp", lambda: pck.image(p.removesuffix(".import")), 96)
        m = re.match(r"images/atlases/ui_atlas\.sprites/map/ancients/(ancient_node_[a-z]+)\.tres$", p)
        if m:
            put(out / "ui" / f"{m.group(1)}.webp", lambda: pck.sprite(p), 96)
        # 诺奴佩普的地图节点不在图集里，只有单独的图（2026-11-06 看板的先古遗物分享图缺这个图标）
        m = re.match(r"images/packed/map/ancients/(ancient_node_[a-z]+)\.png\.import$", p)
        if m and not (out / "ui" / f"{m.group(1)}.webp").exists():
            put(out / "ui" / f"{m.group(1)}.webp", lambda: pck.image(p.removesuffix(".import")), 96)
    # 分职业的遗物图（美味饼干等，图集里是 <遗物>_<职业>，没有不带职业的那张）：存成 relic/<遗物>_<职业>.webp，分享图按角色取
    for p in files:
        m = re.match(r"images/atlases/relic_atlas\.sprites/([a-z0-9_]+)_(defect|ironclad|necro|regent|silent)\.tres$", p)
        if m:
            put(out / "relic" / f"{m.group(1).upper()}_{m.group(2).upper()}.webp", lambda: pck.sprite(p), 96)
    # 卡牌分享图（tools/share_images.py）的费用角标：各职业的能量球（2026-11-10 起补齐五个职业，故障机器人的图用蓝球）与星辰图标
    for ch in ("regent", "defect", "silent", "ironclad", "necrobinder"):
        put(out / "share" / f"energy_{ch}.webp", lambda ch=ch: pck.sprite(f"images/atlases/ui_atlas.sprites/card/energy_{ch}.tres"), 96)
    put(out / "share" / "star_icon.webp", lambda: pck.image("images/packed/sprite_fonts/star_icon.png"), 96)
    return n


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    vocab = (repo / "engine/catalog/vocab.txt").read_text(encoding="utf-8").split("\n")
    print(f"新增 {build(Path(sys.argv[1]) / 'assets', vocab)} 张图")
