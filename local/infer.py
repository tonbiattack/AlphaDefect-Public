"""後続機能（推奨行動の表示・カード報酬の選択支援・プレイ比較・自動プレイなど）が使う共通推論 API。

ここだけを使えば、観測の読み方や観測形式の違いを各機能が気にしなくてよい。

    from local.infer import LocalAgent

    agent = LocalAgent("DEFECT")            # 重み・エンジン・ゲーム資産はキャッシュから
    obs = agent.reset("ABC123", ascension=10)
    print(obs["act"], obs["floor"], obs["hp"])
    for a in agent.legal_actions():
        print(a.index, a.label, a.prob)
    best = agent.recommend()
    agent.step(best.index)

返す値はすべて素の Python（dict / dataclass）なので、UI や別プロセスから使える。
ネットワークとエンジンはこのクラスが持ち、`close()` でエンジンを終了する。

観測形式の扱い: 重みは学習時の観測形式でしか意味を持たない（例: 铁甲战士 v24,782 は 28）。
LocalAgent は重みの目録（weights.ASSETS）から形式を知り、実行時の既定と違えば
reset の req-extra に obs_format を入れる。呼び出し側は何も指定しなくてよい。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from . import weights as W
from .errors import LocalError, SetupRequired
from .recipes import default_format
from .steam import find_install, read_install
from .versions import line_for


@dataclass
class Action:
    """合法手 1 つ。index がそのまま env.step() に渡す値。"""

    index: int
    label: str
    kind: str
    key: str
    prob: float = 0.0
    logit: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "kind": self.kind,
            "key": self.key,
            "prob": round(self.prob, 6),
            "logit": round(self.logit, 4),
        }


@dataclass
class Observation:
    """1 つの決定点。"""

    act: int
    floor: int
    hp: int
    max_hp: int
    gold: int
    decision: str
    in_combat: bool
    combats_won: int
    steps: int
    done: bool
    victory: bool
    num_actions: int
    raw: Any = field(repr=False, default=None)

    @property
    def act_no(self) -> int:
        """1 始まりの幕番号（エンジンの act は 0 始まり）。"""
        return self.act + 1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


_LOADED_TREE: Path | None = None


def _load_tree(tree: Path):
    """選んだコードツリーの python パッケージ（sts2env / sts2rl）を読み込む。

    2 つのツリーは同じ名前のパッケージを持つので、別のツリーを読み込み済みなら
    そのモジュールを捨ててから入れ替える（1 プロセスで両方を混ぜない）。
    """
    global _LOADED_TREE
    py = tree / "python"
    if not py.is_dir():
        raise LocalError(f"コードツリーに python/ がありません: {tree}")
    if _LOADED_TREE is not None and _LOADED_TREE != py:
        for name in [m for m in sys.modules if m.startswith(("sts2env", "sts2rl"))]:
            del sys.modules[name]
    _LOADED_TREE = py
    if str(py) not in sys.path:
        sys.path.insert(0, str(py))
    import sts2env
    import sts2rl

    # サブモジュールは明示的に読み込む（パッケージ属性は import しただけでは生えない）
    import sts2rl.actor
    import sts2rl.catalog
    import sts2rl.features
    import sts2rl.greedy
    import sts2rl.model

    return sts2env, sts2rl


def _load_net(ckpt: Path, torch):
    """チェックポイントからネットワークを組む（ability_test.load_net と同じ扱い）。

    チェックポイントより後に追加された語彙 ID の埋め込みは 0 にし、内容先験は
    特徴名で対応付ける（古い版で学習したネットワークを新しい版で使うため）。
    """
    import inspect

    from sts2rl.catalog import Catalog
    from sts2rl.model import ModelConfig, PolicyValueNet, load_state, zero_new_vocab

    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    net = PolicyValueNet(ModelConfig(**blob["config"]["model"])).eval()
    # v0.111.0 系の load_state は content_names を受け取らない（内容先験の対応付けは後から入った）
    if len(inspect.signature(load_state).parameters) >= 3:
        load_state(net, blob["state"], blob.get("content_names"))
    else:
        load_state(net, blob["state"])
    old_n = blob.get("vocab_n") or 1873
    vocab_n = Catalog().vocab_size
    if old_n < vocab_n:
        with torch.no_grad():
            zero_new_vocab(net, old_n, vocab_n)
    return net, blob


class Policy:
    """チェックポイントと推論だけを持つ部分（エンジンは持たない）。

    LocalAgent（ヘッドレスを自分で回す）も、ゲーム側 MOD とつなぐブリッジも、
    推論はここを共有する。観測フレームは python/sts2env/protocol.py の Obs。
    """

    def __init__(
        self,
        character: str,
        ckpt: Path | str | None = None,
        device: str = "cpu",
        install=None,
        tree: Path | None = None,
        log=print,
    ):
        self.character = character.upper()
        self.device = device
        self._log = log
        install = install or find_install()
        self.install = install
        self.line = line_for(install.version) if tree is None else None
        root = Path(tree) if tree is not None else self.line.root
        self.version = install.version

        asset = W.ASSETS.get(self.character)
        if asset is None:
            raise LocalError(f"未知のキャラ: {character}（{', '.join(W.ASSETS)}）")
        self.asset = asset
        self.ckpt = Path(ckpt) if ckpt else W.weight_path(asset)
        if not self.ckpt.is_file():
            raise SetupRequired(f"重みがありません: {self.ckpt}\n  `python -m local setup` を実行してください。")

        self.sts2env, self.sts2rl = _load_tree(root)
        import torch

        self.torch = torch
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.net, self.blob = _load_net(self.ckpt, torch)
        self.net = self.net.to(device)
        self.vocab = _load_vocab(root)

    # -- 推論 ----------------------------------------------------------

    def score(self, obs) -> tuple[Any, Any, float]:
        """観測 1 つ分の (確率, ロジット, 価値)。ネットワークは 1 回だけ通す。"""
        from sts2rl.features import Ragged, collate_ragged

        # 整局モードの一局は、戦闘中もずっと MODE_RUN のヘッドで打つ（evaluator.play と同じ）。
        mode = self.sts2rl.actor.MODE_RUN
        rg = Ragged.from_obs([obs], [mode])
        with self.torch.no_grad():
            logits, value = self.net(collate_ragged(rg).to(self.device))
        logits = logits[0].float().cpu().numpy()
        probs = self.torch.softmax(self.torch.tensor(logits), -1).numpy()
        return probs, logits, float(value[0].float().cpu())

    def actions(self, obs) -> list[Action]:
        """観測 1 つ分の合法手（確率・ロジットつき）。"""
        probs, logits, _ = self.score(obs)
        out = []
        for i in range(obs.num_actions):
            label = obs.labels[i] if i < len(obs.labels) else ""
            if not label:
                label = self.sts2rl.actor.cand_name(obs, i, self.vocab)
            kind = self.sts2env.ActionKind(int(obs.cand_ints[i][0])).name.lower()
            try:
                key = self.sts2rl.actor.choice_key(obs, i, self.vocab)
            except Exception:                      # noqa: BLE001 - 語彙に無い対象でもラベルだけで動く
                key = label
            out.append(Action(i, label, kind, str(key), float(probs[i]), float(logits[i])))
        return out

    def recommend(self, obs, merge: bool = False) -> Action:
        """推奨手（確率最大。merge なら同じカードをまとめた貪欲）。"""
        actions = [a for a in self.actions(obs) if a.logit > float("-inf")]
        if not actions:
            raise LocalError("合法手がありません（観測が壊れている可能性があります）")
        if merge:
            # 同じカードをまとめた貪欲（sts2rl.greedy.merged_argmax）。返すのはその添字の候補。
            idx = self.sts2rl.greedy.merged_argmax(obs, self.score(obs)[0])
            keep = [a for a in actions if a.index == idx]
            if keep:
                return keep[0]
        return max(actions, key=lambda a: (a.prob, -a.index))


class LocalAgent:
    """1 つのゲーム版・1 キャラの推論をまとめたもの（ヘッドレスの 1 局を自分で回す）。"""

    def __init__(
        self,
        character: str,
        ckpt: Path | str | None = None,
        device: str = "cpu",
        game_dir: Path | None = None,
        labels: bool = True,
        merge: bool = False,
        install=None,
        log=print,
    ):
        install = install or find_install()
        self.policy = Policy(character, ckpt=ckpt, device=device, install=install, log=log)
        self.character = self.policy.character
        self.device = device
        self.merge = merge
        self.line = self.policy.line
        self.version = install.version
        self.asset = self.policy.asset
        self.game_dir = game_dir or paths.game_dir(install.version)
        self.sts2env = self.policy.sts2env
        self.sts2rl = self.policy.sts2rl
        self.torch = self.policy.torch
        self.vocab = self.policy.vocab
        self._obs = None
        self._scores_cache = None

        dll = self.line.root / "engine" / "Sts2Headless" / "bin" / "Release" / "net9.0" / "Sts2Headless.dll"
        if not dll.is_file():
            raise SetupRequired(f"エンジンが未ビルドです: {dll}\n  `python -m local setup` を実行してください。")
        self._dll = dll
        self.env = self.sts2env.Sts2Env(dll=dll, labels=labels)

    # ------------------------------------------------------------ 局面

    def reset(self, seed: str, ascension: int = 10) -> Observation:
        """新しい 1 局を始める（seed はゲーム内のシード文字列）。"""
        o = self.env.reset_run(self.character, ascension, seed, **self._req_extra())
        self._set_obs(o)
        return self.observation()

    def _req_extra(self) -> dict:
        extra = {}
        if self.asset.obs_format != default_format(self.line, self.character):
            extra["obs_format"] = self.asset.obs_format
        return extra

    def step(self, index: int) -> Observation:
        """合法手を 1 つ実行する。"""
        if self._obs is None:
            raise LocalError("reset() を先に呼んでください")
        if not 0 <= int(index) < self._obs.num_actions:
            raise LocalError(f"候補の番号が範囲外です: {index}（0〜{self._obs.num_actions - 1}）")
        o = self.env.step(int(index))
        self._set_obs(o)
        return self.observation()

    def observation(self) -> Observation:
        o = self._require_obs()
        return Observation(
            act=int(o.act),
            floor=int(o.floor),
            hp=int(o.hp),
            max_hp=int(o.max_hp),
            gold=int(o.gold),
            decision=self.sts2env.DecisionKind(int(o.decision)).name.lower(),
            in_combat=bool(o.in_combat),
            combats_won=int(o.combats_won),
            steps=int(o.steps),
            done=bool(o.done),
            victory=bool(o.victory),
            num_actions=int(o.num_actions),
            raw=o,
        )

    def legal_actions(self) -> list[Action]:
        """今の決定点で選べる手（候補）全部。"""
        return self.policy.actions(self._require_obs())

    def candidate_scores(self) -> list[Action]:
        """候補ごとのスコア（方策の確率とロジット）。合法手と同じ並び。"""
        return self.legal_actions()

    def recommend(self) -> Action:
        """推奨手（確率最大）。"""
        return self.policy.recommend(self._require_obs(), merge=self.merge)

    def value(self) -> float:
        """価値頭の評価（この局面から先の見込み。探索や比較の道具に使う）。"""
        return self.policy.score(self._require_obs())[2]

    # ------------------------------------------------------------ 内部

    def _set_obs(self, o) -> None:
        self._obs = o

    def _require_obs(self):
        if self._obs is None:
            raise LocalError("reset() を先に呼んでください")
        return self._obs

    # ------------------------------------------------------------ 便利

    def play(self, seed: str, ascension: int = 10, max_steps: int = 6000, on_step=None) -> dict:
        """貪欲で 1 局最後まで打つ（動作確認と簡単な評価用）。"""
        import time

        t0 = time.time()
        ob = self.reset(seed, ascension)
        steps = 0
        while not ob.done and steps < max_steps:
            a = self.recommend()
            if on_step:
                on_step(ob, a)
            ob = self.step(a.index)
            steps += 1
        return {
            "seed": seed,
            "ascension": ascension,
            "character": self.character,
            "win": bool(ob.victory),
            "floor": ob.floor,
            "act": ob.act,
            "act_no": ob.act_no,
            "steps": steps,
            "stuck": steps >= max_steps and not ob.done,
            "s": round(time.time() - t0, 1),
        }

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:      # noqa: BLE001, S110 - 終了処理。エンジンが既に死んでいても握りつぶす
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _load_vocab(tree: Path) -> list[str]:
    """語彙ファイル（1 行 1 キー、行番号が ID）を読む。ラベルの綴りに使うだけ。"""
    path = tree / "engine" / "catalog" / "vocab.txt"
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def loads_install(path: str | Path):
    """ゲームのフォルダを直接指定してインストール情報を読む。"""
    return read_install(Path(path))
