"""Minimal Spire Mentor client: a legal-move stub agent for the operation-layer PoC.

Connects to the mod's TCP socket (default 127.0.0.1:9002), answers `run_config`
with `start_run`, and answers every decision with a legal choice:

  * combat_action : play the first playable card at its first valid target,
                    otherwise end the turn
  * map_select    : first node
  * everything else: `choose` index 0 (the first offered option), which covers
                    card_reward, reward_screen, shop (0 = leave), rest_site,
                    event, treasure, boss_relic, card_select, card_bundle

Every message is logged as one JSON line so the operation coverage can be
inspected afterwards (this is what the issue's PoC needs to prove).

Usage:
    python tools/spire_mentor_probe.py [--host 127.0.0.1] [--port 9002] [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import socket
import time


def _log(path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _first_enabled(options) -> int | None:
    for opt in options or []:
        if isinstance(opt, dict) and opt.get("enabled", True):
            return int(opt.get("index", 0))
    return None


def _combat_reply(msg: dict) -> dict:
    """Play the first playable card at its first valid target, else end the turn."""
    hand = msg.get("hand") or []
    for idx, card in enumerate(hand):
        if not card.get("playable", False):
            continue
        targets = card.get("valid_target_indices")
        if isinstance(targets, list) and targets:
            return {"action": "play", "card_index": idx, "target_index": int(targets[0])}
        return {"action": "play", "card_index": idx}
    return {"action": "end_turn"}


def reply_for(msg: dict) -> dict:
    """Build a legal reply for one outbound decision message."""
    kind = (msg.get("type") or "").lower()
    state_id = msg.get("state_id")
    if kind == "run_config":
        chars = [c.get("id") for c in (msg.get("characters") or []) if isinstance(c, dict)]
        character = "DEFECT" if "DEFECT" in chars else (chars[0] if chars else "IRONCLAD")
        out = {"action": "start_run", "character": character, "ascension": 0, "seed": None}
        if msg.get("request_id") is not None:
            out["request_id"] = msg["request_id"]
        if msg.get("state_id") is not None:
            out["state_id"] = msg["state_id"]
        return out
    if kind == "combat_action":
        out = _combat_reply(msg)
    elif kind == "map_select":
        nodes = msg.get("nodes") or []
        index = 0
        if nodes and isinstance(nodes[0], dict):
            index = int(nodes[0].get("index", 0))
        out = {"action": "choose", "index": index}
    elif kind in ("card_select",):
        cards = msg.get("cards") or []
        out = {"action": "choose", "index": 0} if cards else {"action": "skip"}
    elif kind in ("card_reward", "card_bundle", "boss_relic", "treasure", "reward_screen",
                  "shop", "rest_site", "event", "crystal_sphere"):
        index = _first_enabled(msg.get("options"))
        if index is None and kind == "card_reward":
            out = {"action": "skip"}
        else:
            out = {"action": "choose", "index": index if index is not None else 0}
    elif kind in ("game_over", "run_complete", "error", "pong"):
        return {}
    else:
        return {}
    if state_id is not None:
        out["state_id"] = state_id
    request_id = msg.get("request_id")
    if request_id is not None:
        out["request_id"] = request_id
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9002)
    ap.add_argument("--log", default="")
    ap.add_argument("--seconds", type=float, default=900.0)
    args = ap.parse_args()

    deadline = time.time() + args.seconds
    sock = None
    while sock is None and time.time() < deadline:
        try:
            sock = socket.create_connection((args.host, args.port), timeout=5)
        except OSError:
            time.sleep(2.0)
    if sock is None:
        _log(args.log, {"event": "no_server", "host": args.host, "port": args.port})
        return 1
    sock.settimeout(5)
    _log(args.log, {"event": "connected", "host": args.host, "port": args.port})
    buf = b""
    seen = {}
    try:
        while time.time() < deadline:
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                kind = msg.get("type")
                seen[kind] = seen.get(kind, 0) + 1
                _log(args.log, {"event": "message", "type": kind, "count": seen[kind],
                                "keys": sorted(msg.keys())[:12]})
                reply = reply_for(msg)
                if reply:
                    sock.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                    _log(args.log, {"event": "reply", "to": kind, "reply": reply})
    finally:
        sock.close()
        _log(args.log, {"event": "done", "counts": seen})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
