"""ゲームを使わずに、ブリッジの通し試験をするシミュレータ。

無頭エンジン（本物のゲーム規則）で 1 局を回し、その観測フレームをブリッジの
プロトコルにそのまま流して、推論側（Autopilot + ModelDecider）に判断させる。
返ってきた行動でエンジンを進めるので、「MOD が状態を送り、推論が手を返し、
ゲームが実行する」という往復がそのまま試せる。

    python -m local autopilot --simulate --character DEFECT --seed ABC123

ゲーム側の MOD がまだ無い（あるいは実機を汚したくない）ときに、
  * プロトコルと状態機械が正しいか
  * 推論が実フレームを読めるか
  * 1 局が最後まで進むか
を確かめる。フレームは本物なので、MOD 側の実装が合っていれば実機でも同じ経路になる。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import protocol as P
from .agent import frame_meta
from .autopilot import Autopilot


@dataclass
class SimConfig:
    character: str = "DEFECT"
    seed: str = "ABC123"
    ascension: int = 10
    max_steps: int = 6000
    time_cap: float = 900.0
    stop_after: int = 0          # >0 ならその歩数で打ち切る（通し試験の短縮用。1 局未完でよい）


def run(
    ap: Autopilot,
    line,
    cfg: SimConfig,
    labels: bool = True,
    log=print,
) -> dict:
    """エンジンを回して、ブリッジ越しに 1 局を打つ。"""
    dll = line.root / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"
    if not dll.is_file():
        raise FileNotFoundError(f"エンジンが未ビルドです: {dll}（`python -m local setup`）")
    env_mod = _load_env(line)
    parse = env_mod.protocol.parse
    env = env_mod.Sts2Env(dll=dll, labels=labels)
    t0 = time.time()
    seq = 0
    steps = 0
    acts = 0
    last_obs = None
    try:
        raw = env.request({"op": "reset", "mode": "run", "character": cfg.character,
                           "ascension": cfg.ascension, "seed": cfg.seed,
                           "labels": labels, "snapshots": False, **_extra(ap)})
        obs = parse(raw)
        last_obs = obs
        replies = ap.handle(P.state(raw, seq, character=cfg.character, **frame_meta(obs)))
        while True:
            if time.time() - t0 > cfg.time_cap:
                raise TimeoutError(f"シミュレーションが {cfg.time_cap:.0f} 秒を超えました")
            action = next((m for m in replies if m.get("t") == P.S_ACTION), None)
            if action is None:
                if ap.state == "paused":
                    raise RuntimeError("一時停止のままです（シミュレータでは再開できません）")
                raise RuntimeError(f"推論から行動が返りませんでした（state={ap.state}）")
            if action["seq"] != seq:
                raise RuntimeError(f"返ってきた行動の seq が違います（{action['seq']} != {seq}）")
            acts += 1
            raw = env.request({"op": "step", "action": int(action["index"])})
            obs = parse(raw)
            last_obs = obs
            steps += 1
            seq += 1
            if obs.done:
                break
            if cfg.stop_after and steps >= cfg.stop_after:
                break
            if steps >= cfg.max_steps:
                raise TimeoutError(f"{cfg.max_steps} 歩で終わりませんでした")
            replies = ap.handle(P.state(raw, seq, character=cfg.character, **frame_meta(obs)))
    finally:
        try:
            env.close()
        except Exception:                          # noqa: BLE001, S110 - 後始末
            pass

    result = {
        "done": bool(last_obs.done),
        "seed": cfg.seed,
        "character": cfg.character,
        "ascension": cfg.ascension,
        "win": bool(last_obs.victory),
        "floor": int(last_obs.floor),
        "act": int(last_obs.act),
        "act_no": int(last_obs.act) + 1,
        "steps": steps,
        "actions": acts,
        "seconds": round(time.time() - t0, 1),
    }
    if result["done"]:
        ap.handle(P.done(result["win"], result["floor"], result["act"]))
    else:
        # 打ち切り（試験用）: ランはまだ途中なので done は送らない
        ap.stop("sim-stopped")
    return result


def _extra(ap: Autopilot) -> dict:
    """エンジンの reset に足すキー（観測形式）。"""
    return {} if ap.obs_format is None else {"obs_format": int(ap.obs_format)}


_TREES: dict = {}


def _load_env(line):
    """その版のツリーの sts2env を読む（両方のツリーを 1 プロセスに混ぜない）。"""
    import sys

    key = str(line.root)
    if key in _TREES:
        return _TREES[key]
    for name in [m for m in sys.modules if m.startswith(("sts2env", "sts2rl"))]:
        del sys.modules[name]
    py = Path(line.root) / "python"
    if str(py) in sys.path:
        sys.path.remove(str(py))
    sys.path.insert(0, str(py))
    import sts2env

    _TREES[key] = sts2env
    return sts2env
