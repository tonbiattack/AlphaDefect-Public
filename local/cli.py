"""`python -m local <command>` の入口。

コマンド:
  status     何が揃っていて何が足りないか
  setup      ゲーム資産・重み・エンジンを用意する（冪等）
  selfcheck  エンジンとゲーム資産の噛み合わせを確認する
  play       1 局（または数局）打って結果と所要時間を出す
  api        共通推論 API（local.infer）の動作例
  versions   対応バージョン・キャラ・探索の可否
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
from pathlib import Path

from . import __version__, paths, runner
from . import build as B
from . import weights as W
from .errors import (
    LocalError,
    SetupRequired,
    UnsupportedCharacter,
    UnsupportedFeature,
    UnsupportedVersion,
)
from .recipes import recipe
from .steam import find_install
from .versions import LINES, require_character

# ゲームのシードに使われる文字（英大文字と数字）
_SEED_ALPHABET = string.ascii_uppercase + string.digits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m local", description="AlphaDefect のモデルをローカルで動かす。")
    ap.add_argument("--version", action="version", version=f"local {__version__}")
    ap.add_argument("--home", help="キャッシュの根（既定 ~/.local/share/sls2、環境変数 SLS2_HOME）")
    ap.add_argument("--game-path", help="ゲームのフォルダを直接指定（自動検出できないとき）")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="準備状況を表示する")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("setup", help="ゲーム資産・重み・エンジンを用意する")
    p.add_argument("--characters", help="取得する重み（カンマ区切り。既定は 6 キャラ全部）")
    p.add_argument("--force", action="store_true", help="済んでいる工程もやり直す")
    p.add_argument("--skip-selfcheck", action="store_true")
    p.add_argument("--no-bootstrap", action="store_true", help=".NET や uv の自動導入をしない")

    p = sub.add_parser("selfcheck", help="エンジンとゲーム資産の噛み合わせを確認する")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("play", help="1 局打って結果と所要時間を出す")
    p.add_argument("--character", help="DEFECT / REGENT / SILENT / FOOL / IRONCLAD / NECROBINDER、または auto（ゲーム側の選択に合わせる）")
    p.add_argument("--seed", help="ゲームのシード（例: ABC123）。省略するとランダム")
    p.add_argument("--seeds", help="複数シード（カンマ区切り）")
    p.add_argument("--runs", type=int, help="局数（--seed も --seeds も無いときはランダムなシードをこの数だけ作る）")
    p.add_argument("--asc", type=int, default=10, help="アセンション（既定 10）")
    p.add_argument("--search", action="store_true", help="探索あり（v0.113.1 系のみ。BOSS・エリート戦）")
    p.add_argument("--sims", type=int, default=32, help="探索のシミュレーション回数（既定 32）")
    p.add_argument("--rooms", default="Boss,Elite", help="探索する部屋（既定 Boss,Elite）")
    p.add_argument("--device", default="cpu")
    p.add_argument("--envs", type=int, default=1, help="同時に走らせるエンジン数")
    p.add_argument("--procs", type=int, default=2, help="探索の推演プロセス数")
    p.add_argument("--out", help="出力先（既定 ~/.local/share/sls2/runs/<日時>_...）")
    p.add_argument("--req-extra", help='reset に足すキー（JSON。例: {"loop_cut": true}）')
    p.add_argument("--allow-untested", action="store_true", help="評価実績の無いキャラでも動かす")
    p.add_argument("--dry-run", action="store_true", help="実行するコマンドを表示するだけ")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("api", help="共通推論 API の動作例")
    p.add_argument("--character")
    p.add_argument("--seed")
    p.add_argument("--asc", type=int, default=10)
    p.add_argument("--steps", type=int, default=5, help="何手ぶん表示するか")
    p.add_argument("--top", type=int, default=5, help="候補スコアを上位何件出すか")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("autopilot", help="ゲーム側 MOD とつないで AI に 1 ラン打たせる（観戦用）")
    p.add_argument("--character", help="DEFECT / REGENT / SILENT / FOOL / IRONCLAD / NECROBINDER、または auto（ゲーム側の選択に合わせる）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--speed", type=float, default=1.0, help="速度倍率（MOD に渡す待ち時間が 1/N になる）")
    p.add_argument("--hold-ms", type=int, default=0, help="1 手あたりの待ち時間の基準（ミリ秒、既定 0）")
    p.add_argument("--top", type=int, default=5, help="候補スコアを何件 MOD に渡すか")
    p.add_argument("--merge", action="store_true", help="同じカードをまとめた貪欲")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-states", type=int, default=None, help="この数の状態で自動停止（検証用）")
    p.add_argument("--idle-timeout", type=float, default=0.0, help="無通信で止めるまでの秒数（0 で無効）")
    p.add_argument("--control-port", type=int, default=18766, help="操作口（HTTP）。0 で無効")
    p.add_argument("--exit-on-done", action="store_true", default=True, help="1 ラン終わったら受け口を閉じる")
    p.add_argument("--keep-listening", dest="exit_on_done", action="store_false", help="1 ランの後も次の接続を待つ")
    p.add_argument("--no-stdin", dest="stdin", action="store_false", help="標準入力からの操作を受け付けない")
    p.add_argument("--log", help="JSONL のログの出力先（既定 ~/.local/share/sls2/logs/autopilot-<日時>.jsonl）")
    p.add_argument("--autostart", help='接続時にゲーム側で 1 局始める（"DEFECT|0|ABC123|regular"。encounter は weak/regular/elite/boss）')
    p.add_argument("--simulate", action="store_true", help="ゲームを使わず、無頭エンジンの実フレームで通し試験する")
    p.add_argument("--seed", default="ABC123", help="--simulate のシード")
    p.add_argument("--asc", type=int, default=10, help="--simulate のアセンション")
    p.add_argument("--stop-after", type=int, default=0, help="--simulate でこの歩数で打ち切る（0 は最後まで）")

    p = sub.add_parser("launch", help="Windows から 1 クリックで起動する（通常プレイ / AI 観戦）")
    p.add_argument("mode", nargs="?", choices=["normal", "ai", "mentor"],
                   help="normal=MODなし、ai=推論サーバ＋MOD、mentor=spire-mentor 連携")
    p.add_argument("--character", default="auto", help="AI モードのキャラ（auto でゲーム側の選択に合わせる）")
    p.add_argument("--speed", type=float, default=1.0, help="AI モードの速度（Godot の TimeScale）")
    p.add_argument("--seed", default="", help="AI モードのシード（空ならランダム）")
    p.add_argument("--ascension", type=int, default=0, help="Mentor モードのアセンション 0〜10")
    p.add_argument("--no-autostart", dest="autostart", action="store_false", help="ランを自動で始めない（自分で開始する）")
    p.add_argument("--no-wait", dest="wait", action="store_false", help="ゲームの終了を待たない（サーバは残す）")
    p.add_argument("--dry-run", action="store_true", help="起動内容を表示するだけ")
    p.add_argument("--status", action="store_true", help="いまの状態を表示する")
    p.add_argument("--stop", action="store_true", help="推論サーバを止める")
    p.add_argument("--pause", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--set-speed", type=float, default=None, help="動いている推論サーバの速度を変える")
    p.add_argument("--install-shortcuts", action="store_true", help="デスクトップとスタートメニューにショートカットを作る")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("mentor", help="spire-mentor に画面を任せ、判断だけ AI にやらせる（実験）")
    p.add_argument("--runs", type=int, default=1, help="spire-mentor に回させるランの数")
    p.add_argument("--no-launch", action="store_true", help="ゲームを起動しない（設定だけ）")
    p.add_argument("--log", default="", help="推論サーバのログ（JSONL）")
    p.add_argument("--keep-listening", action="store_true", help="ゲームが終わっても待ち続ける")
    p.add_argument("--idle-timeout", type=float, default=30.0, help="接続が切れてから終了するまで（秒）")
    p.add_argument("--control-port", type=int, default=0, help="操作 HTTP のポート（既定 18766）")
    p.add_argument("--character", default="auto",
                   help="キャラ（auto / select / DEFECT / REGENT / SILENT など。versions.py が正）")
    p.add_argument("--ascension", type=int, default=0, help="アセンション 0〜10")
    p.add_argument("--seed", default=None, help="整数のシード（省略または random で乱数）")
    p.add_argument("--allow-untested", action="store_true", help="未検証のキャラでも開始する")
    p.add_argument("--status", action="store_true", help="いまの開始設定だけを表示する")

    p = sub.add_parser("measure", help="実ゲームで複数ランを回して勝敗・到達階を記録する")
    p.add_argument("--runs", type=int, default=1, help="回すラン数（seed は MEAS1.. または --seeds）")
    p.add_argument("--seeds", default="", help="カンマ区切りの seed（指定すると --runs より優先）")
    p.add_argument("--character", default="DEFECT")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--timeout", type=float, default=900.0, help="1 ランの上限秒数")
    p.add_argument("--out", default="", help="記録先の JSONL（既定 ~/.local/share/sls2/runs/measure.jsonl）")
    p.add_argument("--summary", action="store_true", help="記録済みの集計だけを出す")
    p.add_argument("--backend", choices=["own", "mentor"], default="own",
                   help="own=独自MOD、mentor=spire-mentor に画面を任せモデルで判断")

    p = sub.add_parser("mod", help="ゲーム側 MOD（観戦用）の導入・撤去・状態")
    p.add_argument("action", choices=["install", "remove", "status"], nargs="?", default="status")
    p.add_argument("--force", action="store_true", help="キャッシュがあってもビルドし直す")

    p = sub.add_parser("replay", help="1 局打って、ブラウザのリプレイで AI の判断を観戦する")
    p.add_argument("--character", help="DEFECT / REGENT / SILENT（この版で重みがあるキャラ）")
    p.add_argument("--seed", default="REPLAY1", help="シード（既定 REPLAY1）")
    p.add_argument("--asc", type=int, default=10, help="アセンション（既定 10）")
    p.add_argument("--from", dest="from_", help="既存のリプレイ（.jsonl.gz か、replays/ を含むディレクトリ）")
    p.add_argument("--out", help="再生ページの出力先（既定 ~/.local/share/sls2/runs/<日時>_viewer）")
    p.add_argument("--langs", default="jpn,eng", help="名前の言語の優先順（既定 jpn,eng）")
    p.add_argument("--no-open", action="store_true", help="ブラウザで開かない（生成だけ）")
    p.add_argument("--no-assets", action="store_true", help="ゲーム美術の抽出をしない（軽い）")
    p.add_argument("--device", default="cpu")

    p = sub.add_parser("versions", help="対応バージョンとキャラの一覧")

    a = ap.parse_args(argv)
    if a.home:
        os.environ["SLS2_HOME"] = a.home
    if a.game_path:
        os.environ["SLS2_GAME_PATH"] = a.game_path

    try:
        return _dispatch(a)
    except UnsupportedVersion as e:
        print(f"未対応: {e}", file=sys.stderr)
        return 2
    except UnsupportedCharacter as e:
        print(f"動かせません: {e}", file=sys.stderr)
        return 2
    except SetupRequired as e:
        print(f"準備が必要です: {e}", file=sys.stderr)
        return 3
    except UnsupportedFeature as e:
        print(f"この版では未対応: {e}", file=sys.stderr)
        return 2
    except LocalError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました", file=sys.stderr)
        return 130


def _dispatch(a) -> int:
    return {
        "status": cmd_status,
        "setup": cmd_setup,
        "selfcheck": cmd_selfcheck,
        "play": cmd_play,
        "api": cmd_api,
        "autopilot": cmd_autopilot,
        "replay": cmd_replay,
        "mod": cmd_mod,
        "launch": cmd_launch,
        "measure": cmd_measure,
        "mentor": cmd_mentor,
        "versions": cmd_versions,
    }[a.command](a)


# ---------------------------------------------------------------- status


def cmd_status(a) -> int:
    from .setup import inspect

    st = inspect()
    if a.json:
        print(json.dumps(st.to_dict(), ensure_ascii=False, indent=1))
        return 0
    if st.install is None:
        print("ゲームが見つかりません。", file=sys.stderr)
        for p in st.problems:
            print(p, file=sys.stderr)
        return 1

    print(f"ゲーム: {st.install.path}")
    print(f"  バージョン: {st.install.version}（ブランチ {st.install.branch or '?'}, コミット {st.install.commit or '?'}, {st.install.date or '?'}）")
    if st.line is None:
        print("  未対応のバージョンです。", file=sys.stderr)
        for p in st.problems:
            print(p, file=sys.stderr)
        return 2
    extra = "（パッチ互換。実機で selfcheck と 1 局完走を確認済み）" if st.report and st.report.patch_alias else ""
    print(f"  コード系統: {st.line.key}（{st.line.tree or 'リポジトリ直下'}）{extra}")
    print(f"  探索あり: {'対応' if st.line.search else '非対応'}")
    print(f"  備考: {st.line.note}")
    print("  キャラ:")
    for c, status in (st.report.characters if st.report else {}).items():
        mark = {"validated": "評価実績あり", "untested": "実績なし", "unsupported": "非対応"}[status]
        print(f"    {c:<12} {mark}")
    print(f"キャッシュ: {paths.home()}")
    print(f"  .NET: {st.dotnet or 'なし'}{'（SDK 9 あり）' if st.dotnet_sdk9 else '（SDK 9 なし）'}")
    print(f"  Python: {st.venv_python if st.venv_ok else '仮想環境なし'}")
    print(f"  PyTorch: {st.torch_version or 'なし'}")
    print(f"  ゲーム資産: {'準備済み' if st.game_ready else '未準備'}")
    print(f"  エンジン: {'ビルド済み' if st.engine_built else '未ビルド'}")
    print("  重み:")
    usable = set(st.line.characters)
    for c, w in st.weights.items():
        if w["present"]:
            state = "取得済み" if w.get("size_ok", True) else "サイズ不一致（要再取得）"
        else:
            state = "未取得"
        note = "" if c in usable else "（この版では使わない）"
        print(f"    {c:<12} {state:<22} {w['release']}/{w['file']}{note}")
    if st.ready:
        print("\n準備完了。`python -m local play --character "
              f"{next(iter(st.report.characters)) if st.report else 'DEFECT'} --seed ABC123 --asc 10` で 1 局打てます。")
    else:
        print("\n準備が足りません。`python -m local setup` を実行してください。")
    return 0


# ---------------------------------------------------------------- setup / selfcheck


def cmd_setup(a) -> int:
    from .setup import setup

    install = find_install()
    chars = [c.strip().upper() for c in a.characters.split(",")] if a.characters else None
    st = setup(install, characters=chars, force=a.force, skip_selfcheck=a.skip_selfcheck, bootstrap=not a.no_bootstrap)
    return 0 if st.ready else 1


def cmd_selfcheck(a) -> int:
    from .setup import inspect

    st = inspect()
    if st.install is None or st.line is None:
        print("\n".join(st.problems) or "ゲームが見つかりません", file=sys.stderr)
        return 2
    game_dir = paths.game_dir(st.install.version)
    ok, out = B.selfcheck(st.line, game_dir)
    if a.json:
        print(json.dumps({"ok": ok, "version": st.install.version, "line": st.line.key}, ensure_ascii=False))
    else:
        print(out.rstrip())
        print("\n結果: " + ("通過" if ok else "失敗"))
    return 0 if ok else 1


# ---------------------------------------------------------------- play


def cmd_play(a) -> int:
    from .setup import require_ready

    st = require_ready()
    assert st.install is not None and st.line is not None
    line = st.line
    character = (a.character or _default_character(line)).upper()
    require_character(line, character, allow_untested=a.allow_untested)
    if not line.search and a.search:
        raise UnsupportedFeature(
            f"{line.key} 系ではローカルの探索ありを扱っていません。\n"
            "  探索ありは v0.113.1（愚者・铁甲战士・亡灵契约师）で使えます。\n"
            "  このシステムの探索（PIMC）は外部の sts2-sim のビルドが要るため、このランナーには入れていません。\n"
            "  探索なし（--search を付けない）なら、この版でも論文と同じ貪欲プレイで 1 局打てます。"
        )
    seeds = _seeds(a)
    asset = W.ASSETS[character]
    rec = recipe(line, character, obs_format=asset.obs_format,
                 extra=json.loads(a.req_extra) if a.req_extra else None)
    spec = runner.RunSpec(
        character=character,
        seeds=seeds,
        ascension=a.asc,
        search=a.search,
        sims=a.sims,
        rooms=a.rooms,
        device=a.device,
        envs=a.envs,
        procs=a.procs,
        out=Path(a.out) if a.out else None,
        time_cap=rec.time_cap,
    )
    plan = runner.plan(line, spec, rec, paths.venv_python(), W.weight_path(asset), paths.game_dir(st.install.version),
                       st.dotnet, dry_run=a.dry_run)

    if a.dry_run:
        print(plan.display())
        return 0

    print(f"{character} / A{a.asc} / {len(seeds)} 局 / {'探索あり（' + str(a.sims) + ' 回）' if a.search else '貪欲'}")
    print(f"出力: {plan.out}")
    print(plan.display())
    print()
    outcome = runner.execute(plan)
    print()
    if a.json:
        print(json.dumps({"runs": outcome.results, "seconds": round(outcome.seconds, 1), "returncode": outcome.returncode, "out": str(plan.out)},
                         ensure_ascii=False, indent=1))
    else:
        _print_runs(outcome)
    if outcome.returncode != 0:
        print(f"\n評価ツールが終了コード {outcome.returncode} で止まりました。最後の出力:\n{outcome.tail}", file=sys.stderr)
        return 1
    missing = [s for s in seeds if runner.pick(outcome, s) is None]
    if missing:
        print(f"\n結果が出ていないシード: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def _print_runs(outcome: runner.RunOutcome) -> None:
    print(f"=== 結果（{outcome.seconds:.1f} 秒）===")
    for r in outcome.results:
        print(f"[{r.get('seed')}] " + "; ".join(runner.describe(r)))
    paths_ = runner.replay_paths(outcome.plan.out)
    if paths_:
        print(f"リプレイ: {paths_[0]}" + (f"（ほか {len(paths_) - 1} 件）" if len(paths_) > 1 else ""))


def _default_character(line) -> str:
    from .versions import character_status

    for c in line.characters:
        if character_status(line, c) == "validated":
            return c
    return line.characters[0]


def _seeds(a) -> list[str]:
    if a.seeds:
        seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    elif a.seed:
        seeds = [a.seed.strip()]
    else:
        n = a.runs or 1
        seeds = ["".join(random.choice(_SEED_ALPHABET) for _ in range(10)) for _ in range(n)]
    if a.runs and a.runs != len(seeds):
        raise LocalError(f"--runs {a.runs} とシードの数 {len(seeds)} が合いません（--runs を省くか、シードを {a.runs} 個指定してください）")
    if not seeds:
        raise LocalError("シードが空です")
    return seeds


# ---------------------------------------------------------------- api


def cmd_api(a) -> int:
    from .infer import LocalAgent
    from .setup import require_ready

    st = require_ready()
    assert st.install is not None and st.line is not None
    character = (a.character or _default_character(st.line)).upper()
    require_character(st.line, character, allow_untested=True)
    seed = a.seed or "".join(random.choice(_SEED_ALPHABET) for _ in range(10))
    agent = LocalAgent(character, install=st.install, game_dir=paths.game_dir(st.install.version))
    try:
        obs = agent.reset(seed, a.asc)
        for i in range(max(1, a.steps)):
            actions = agent.candidate_scores()
            best = agent.recommend()
            if a.json:
                print(json.dumps({"step": i, "observation": obs.to_dict(), "recommended": best.to_dict(),
                                  "candidates": [x.to_dict() for x in actions]}, ensure_ascii=False))
            else:
                print(f"--- {i} 手目: {obs.decision} / 第 {obs.act_no} 幕 {obs.floor} 層 / HP {obs.hp}/{obs.max_hp} / 候補 {obs.num_actions}")
                for x in sorted(actions, key=lambda v: -v.prob)[: max(1, a.top)]:
                    mark = "*" if x.index == best.index else " "
                    print(f"  {mark} [{x.index}] {x.label}  p={x.prob:.3f}  ({x.kind})")
                print(f"  推奨: [{best.index}] {best.label}（p={best.prob:.3f}）")
            if obs.done:
                break
            obs = agent.step(best.index)
        print(f"\nシード {seed} / {character} / A{a.asc}：{obs.floor} 層（{obs.decision}）")
    finally:
        agent.close()
    return 0


# ---------------------------------------------------------------- autopilot


def cmd_autopilot(a) -> int:
    """ゲーム側 MOD とつないで AI に打たせる（--simulate ならゲーム不要で通し試験）。"""
    import time as _time

    from .bridge import Autopilot, SessionLog
    from .bridge.agent import ModelDecider
    from .setup import require_ready

    st = require_ready()
    assert st.install is not None and st.line is not None
    line = st.line
    character = (a.character or _default_character(line)).upper()
    if character != "AUTO":
        require_character(line, character, allow_untested=True)
    if character == "AUTO":
        from .bridge.agent import AutoDecider

        decider = AutoDecider(install=st.install, device=a.device, top=a.top, merge=a.merge)
    else:
        decider = ModelDecider.load(character, device=a.device, install=st.install, top=a.top, merge=a.merge)
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(a.log) if a.log else paths.home() / "logs" / f"autopilot-{stamp}_{character}.jsonl"
    log = SessionLog(log_path)
    run_id = f"{stamp}-{character}"
    ap = Autopilot(
        decider,
        character=character,
        run_id=run_id,
        log=log,
        hold_ms=a.hold_ms,
        max_states=a.max_states,
        merge=a.merge,
        obs_format=decider.obs_format,
        autostart=a.autostart,
    )
    print(f"[autopilot] キャラ {character} / ゲーム {st.install.version} / 形式 {decider.obs_format} / ログ {log_path}", flush=True)
    try:
        if a.simulate:
            return _run_simulation(ap, line, a, log_path)
        from .bridge.server import ServerConfig, serve

        cfg = ServerConfig(host=a.host, port=a.port, control_port=a.control_port,
                           idle_timeout=a.idle_timeout, exit_on_done=a.exit_on_done, stdin=a.stdin)
        return serve(ap, cfg)
    finally:
        log.close()


def _run_simulation(ap, line, a, log_path: Path) -> int:
    """無頭エンジンの実フレームで、ブリッジ越しに 1 局打つ（ゲーム不要の通し試験）。"""
    from .bridge.sim import SimConfig, run

    sim_character = ap.character
    if sim_character == "AUTO":
        sim_character = _default_character(line)   # 通し試験では具体キャラを使う
    cfg = SimConfig(character=sim_character, seed=a.seed, ascension=a.asc, stop_after=a.stop_after)
    print(f"[autopilot] ゲームを使わずに通し試験します（seed {cfg.seed} / A{cfg.ascension}）", flush=True)
    ap.start()
    try:
        result = run(ap, line, cfg, log=lambda s: print(s, flush=True))
    except Exception as e:                          # noqa: BLE001 - 失敗はそのまま報告する
        print(f"[autopilot] 通し試験が失敗しました: {e}", file=sys.stderr)
        ap.stop("simulation-error")
        print("[autopilot] " + json.dumps(ap.summary(), ensure_ascii=False), flush=True)
        return 1
    print()
    print(f"=== 通し試験の結果（{result['seconds']:.1f} 秒 / 行動 {result['actions']} 回）===")
    print(f"シード {result['seed']} / {result['character']} / A{result['ascension']}: "
          + ("勝利（全 3 幕クリア）" if result["win"] else f"第 {result['act_no']} 幕 {result['floor']} 層で終了")
          + ("" if result["done"] else "（打ち切り）"))
    print("ログ: " + str(log_path))
    return 0


# ---------------------------------------------------------------- replay


def cmd_replay(a) -> int:
    from . import replay as R
    from .setup import inspect, require_ready

    langs = tuple(x for x in (a.langs or "jpn,eng").split(",") if x)
    if a.from_:
        st = inspect()
        if st.install is None or st.line is None:
            print("\n".join(st.problems) or "ゲームが見つかりません", file=sys.stderr)
            return 2
        records = R.collect(Path(a.from_))
        if not records:
            print(f"リプレイが見つかりません: {a.from_}", file=sys.stderr)
            return 1
    else:
        st = require_ready()
        assert st.install is not None and st.line is not None
        character = (a.character or _default_character(st.line)).upper()
        require_character(st.line, character, allow_untested=True)
        if not R._load_viewer():
            return 1
        rec = R.record(st.line, st.install, character, a.seed, a.asc, device=a.device)
        if not rec.ok:
            print("リプレイが残りませんでした（ability_test の出力を確認してください）", file=sys.stderr)
            return 1
        records = rec.replays
    out = Path(a.out) if a.out else R.default_out(
        (a.character or "run").upper() if not a.from_ else "from", a.seed or "records")
    print(f"記録: {len(records)} 局")
    R.summarize(records)
    html = R.build(records, st.install, st.line, out, langs=langs, assets=not a.no_assets)
    print(f"再生ページ: {html}")
    if not a.no_open:
        R.open_in_browser(html)
    return 0


# ---------------------------------------------------------------- mod


def cmd_mod(a) -> int:
    from . import mod as M
    from .steam import find_install

    install = find_install()
    if a.action == "status":
        st = M.status(install)
        print(f"ゲーム: {install.path}（{install.version}）")
        print(f"MOD: {'導入済み' if st.installed else '未導入'} {st.path}")
        if st.installed:
            print(f"  ビルド時の版: {st.built_for or '?'}" + ("（今のゲームと違います。入れ直してください）" if st.stale else ""))
        print()
        print(M.gameplay_note())
        return 0
    if a.action == "remove":
        removed = M.remove(install)
        if not removed:
            print("MOD は入っていません。")
        return 0
    if M.game_running():
        st = M.status(install)
        if st.installed:
            print("ゲームが動いています。MOD を入れ替えるにはゲームを終了してください。", file=sys.stderr)
            return 1
        print("ゲームが動いています（いまは未導入なので、置くだけにします。次の起動から効きます）", file=sys.stderr)
    M.install(install, force=a.force, allow_running=True)
    print()
    print(M.ensure_enabled_hint(install))
    print()
    print(M.gameplay_note())
    return 0


# ---------------------------------------------------------------- launch


def cmd_launch(a) -> int:
    from . import launcher as L

    if a.install_shortcuts:
        L.install_shortcuts()
        return 0
    if a.status or (a.mode is None and not (a.stop or a.pause or a.resume or a.set_speed is not None)):
        print(L.status_text())
        if a.mode is None and not a.json:
            print("\n使い方: python -m local launch normal|ai [--character auto] [--speed 2] [--seed ABC]")
        return 0
    if a.stop:
        return L.stop_server()
    if a.pause or a.resume or a.set_speed is not None:
        cmd = "pause" if a.pause else ("resume" if a.resume else f"speed {a.set_speed}")
        got = L.control(cmd)
        print(json.dumps(got or {"error": "推論サーバが応答しません"}, ensure_ascii=False))
        return 0 if got else 1
    if a.mode == "mentor":
        return _start_mentor(character=a.character, ascension=a.ascension, seed=a.seed,
                             runs=1, launch=not a.dry_run)
    return L.start(a.mode, character=a.character, speed=a.speed, seed=a.seed,
                   autostart=a.autostart, wait=a.wait, dry_run=a.dry_run)


# ---------------------------------------------------------------- measure


def _start_mentor(character: str, ascension: int, seed, runs: int, launch: bool,
                  log: str = "", keep_listening: bool = True, idle_timeout: float = 0.0,
                  allow_untested: bool = False) -> int:
    """spire-mentor に画面を任せ、判断だけをこのリポジトリのモデルにやらせる（共通処理）。"""
    import subprocess
    import time

    from . import measure as M
    from . import mentorcfg as MC
    from . import paths
    from . import steam as _steam

    install = _steam.find_install()
    if install is None:
        print("ゲームが見つかりません（Steam のインストール先を確認してください）。")
        return 1
    spire = install.path / "mods" / "SpireMentor" / "SpireMentor.conf"
    if not spire.is_file():
        print("spire-mentor が入っていません（docs/ja/oss-operation-layer.md の手順で導入してください）。")
        return 1
    try:
        choice = MC.resolve(install.version, character, ascension, seed, allow_untested=allow_untested)
    except (LocalError, UnsupportedVersion) as e:
        print(f"開始できません: {e}", file=sys.stderr)
        print("使えるキャラ: " + ", ".join(f"{c}({st})" for c, st in MC.available_characters(install.version)))
        return 2
    MC.write(install, choice)
    if choice.character == "select":
        M._set_spire_autostart(install, False, max(1, runs))
        print("開始しません（ゲーム内でキャラを選んで開始してください）。"
              "このモードでは spire-mentor の autopilot は動きません。")
    else:
        M._set_spire_autostart(install, True, max(1, runs))
    flag = M._mentor_flag(install)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1", encoding="utf-8")
    print(f"spire-mentor: {runs} ラン / {install.version}")
    print("開始設定: " + MC.describe(install))
    if not launch:
        return 0

    from . import launcher as L

    # こちらは「待ち受けだけ」の推論サーバ。ランの開始と画面遷移は spire-mentor が行う。
    if not L.game_running():
        M._set_spire_autostart(install, True, max(1, runs))
    # 推論サーバはフレームのキャラでモデルを選ぶ（auto）。連続ランでキャラが変わっても追随する。
    argv = [str(paths.venv_python()), "-m", "local", "autopilot", "--character", "auto", "--no-stdin"]
    if keep_listening:
        argv.append("--keep-listening")
    argv += ["--idle-timeout", str(idle_timeout)]
    if log:
        argv += ["--log", log]
    server_log = paths.home() / "logs" / "mentor-server.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    with server_log.open("ab") as fh:
        server_proc = subprocess.Popen(argv, cwd=str(paths.REPO), stdout=fh, stderr=subprocess.STDOUT,
                                       start_new_session=True)
    print(f"推論サーバを起動しました（PID {server_proc.pid}、ログ {server_log}）")
    try:
        for _ in range(60):
            if L.control("status") is not None:
                break
            time.sleep(1.0)
        L.launch_game(install, [])
        print("ゲームを起動しました（spire-mentor がランを始めます）。")
        # Steam 経由の起動はプロセスが現れるまで少し間がある。すぐ「終わった」と判断しない。
        spawn_deadline = time.time() + 120
        while time.time() < spawn_deadline and not L.game_running():
            time.sleep(2.0)
        while L.game_running():
            time.sleep(3.0)
    except KeyboardInterrupt:
        print("\n中断しました", file=sys.stderr)
    finally:
        try:
            os.killpg(os.getpgid(server_proc.pid), 15)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return 0


def cmd_mentor(a) -> int:
    """spire-mentor に画面操作を任せ、判断だけをこのリポジトリのモデルにやらせる。"""
    from . import mentorcfg as MC
    from . import steam as _steam

    install = _steam.find_install()
    if install is None:
        print("ゲームが見つかりません（Steam のインストール先を確認してください）。")
        return 1
    if a.status:
        print(MC.describe(install))
        print("使えるキャラ: " + ", ".join(f"{c}({st})" for c, st in MC.available_characters(install.version)))
        return 0
    return _start_mentor(
        character=a.character,
        ascension=a.ascension,
        seed=a.seed,
        runs=a.runs,
        launch=not a.no_launch,
        log=a.log,
        keep_listening=a.keep_listening,
        idle_timeout=a.idle_timeout,
        allow_untested=a.allow_untested,
    )


def cmd_measure(a) -> int:
    from . import measure as M

    out = Path(a.out) if a.out else M.DEFAULT_OUT
    if a.summary:
        rows = []
        if out.is_file():
            for line in out.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        print(M.summarize(rows))
        return 0
    seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = [f"MEAS{i}" for i in range(1, a.runs + 1)]
    try:
        if a.backend == "mentor":
            import json as _json

            from . import launcher as _L
            from . import steam as _steam

            install = _steam.find_install()
            rows = M.run_once_mentor(install, runs_in_session=max(1, a.runs), timeout=a.timeout)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(_json.dumps(row, ensure_ascii=False) + "\n")
            _ = _L
            print()
            print(M.summarize(rows))
            return 0
        rows = M.measure(seeds, a.character, a.ascension, timeout=a.timeout, out=out)
    except KeyboardInterrupt:
        print("\n中断しました（ここまでの記録は残っています）")
        return 130
    print()
    print(M.summarize(rows))
    return 0


# ---------------------------------------------------------------- versions


def cmd_versions(a) -> int:
    print("対応しているゲームの版:")
    for key, line in LINES.items():
        print(f"  {key}")
        print(f"    コード: {line.tree or 'リポジトリ直下'}")
        print(f"    キャラ: {', '.join(line.characters)}")
        print(f"    探索あり: {'対応' if line.search else '非対応'}")
        print(f"    備考: {line.note}")
    from .versions import PATCH_COMPAT

    for patch, base in PATCH_COMPAT.items():
        print(f"  {patch}（{base} のパッチ互換。実機で確認済み）")
    print("\nどの版も、モデルは CPU で動きます。ゲーム資産と重みはリポジトリに含まれません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
