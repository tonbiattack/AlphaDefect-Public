"""战斗课程：生成 CombatSpec（技术路线 5.2 节的第 1–2 步）。

每场战斗随机抽取：幕、遭遇类别、牌组（初始牌组 + 按稀有度抽的若干张卡、随机升级、随机删去基础牌）、
遗物、药水、开场血量。“进度”0→1 控制牌组规模和后期遭遇的比例，由训练端按表现调节。
另有固定比例的 Boss 名额（boss_fraction），不受进度限制，牌组按“打到这一幕结尾”配，见 _boss。
第 3 步（用整局策略实际打出的牌组和局面）：Boss 名额和各幕的普通战、精英战练习（real_fractions）优先用整局里攒下的真实快照，见 SnapshotPools；
另有随机构筑（mix_fraction）用幕末规模的随机牌组。按进度出题只占剩下的份额。
真实快照的 Boss 练习另有一部分用整局中途（选路线时）的快照，结果给牌组打分头当标签，见 SnapshotPools.probe。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .catalog import Catalog

# 各角色的初始牌组（游戏 CharacterModel.StartingDeck，v0.111.0）
STARTING_DECK = {
    "DEFECT": ["STRIKE_DEFECT"] * 4 + ["DEFEND_DEFECT"] * 4 + ["ZAP", "DUALCAST"],
    "REGENT": ["STRIKE_REGENT"] * 4 + ["DEFEND_REGENT"] * 4 + ["FALLING_STAR", "VENERATE"],
    # 静默猎手（Silent.StartingDeck，猎人观测调研.md 第一节）：打击 ×5、防御 ×5、中和、生存者
    "SILENT": ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5 + ["NEUTRALIZE", "SURVIVOR"],
    # Fool（v0.113.0 新角色，Fool.StartingDeck）：打击 ×5、防御 ×4、愚者射击
    "FOOL": ["STRIKE_FOOL"] * 5 + ["DEFEND_FOOL"] * 4 + ["FOOLISH_SHOT"],
    # 铁甲战士（Ironclad.StartingDeck，铁甲与亡灵契约师观测调研.md 第一节）：打击 ×5、防御 ×4、痛击
    "IRONCLAD": ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"],
    # 亡灵契约师（Necrobinder.StartingDeck，v0.113.1 反编译源码核对）：打击 ×4、防御 ×4、护卫、出击
    "NECROBINDER": ["STRIKE_NECROBINDER"] * 4 + ["DEFEND_NECROBINDER"] * 4 + ["BODYGUARD", "UNLEASH"],
}


@dataclass
class CurriculumConfig:
    character: str = "DEFECT"
    ascension: int = 10
    pool: str = "DEFECT_CARD_POOL"
    relic_pools: tuple[str, ...] = ("SHARED_RELIC_POOL", "DEFECT_RELIC_POOL")
    potion_pools: tuple[str, ...] = ("SHARED_POTION_POOL", "DEFECT_POTION_POOL")

    @staticmethod
    def for_character(character: str, ascension: int = 10) -> "CurriculumConfig":
        """按角色取卡池、遗物池、药水池（目录里的池名为 角色_CARD_POOL 等）；故障机器人与默认值相同。"""
        if character not in STARTING_DECK:
            raise ValueError(f"没有 {character} 的初始牌组（curriculum.STARTING_DECK）")
        return CurriculumConfig(character, ascension, f"{character}_CARD_POOL", ("SHARED_RELIC_POOL", f"{character}_RELIC_POOL"),
                                ("SHARED_POTION_POOL", f"{character}_POTION_POOL"))


class CombatCurriculum:
    def __init__(self, catalog: Catalog, cfg: CurriculumConfig | None = None, seed: int = 0):
        self.cfg = cfg or CurriculumConfig()
        self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.cards = {r: catalog.card_pool(c.pool, (r,)) for r in ("Common", "Uncommon", "Rare")}
        self.relics = {r: catalog.relic_pool(c.relic_pools, (r,)) for r in ("Common", "Uncommon", "Rare", "Shop")}
        self.potions = catalog.potion_pool(c.potion_pools, ("Common", "Uncommon", "Rare"))
        self.counter = 0

    # Boss 名额里各幕的比例。6 : 3 : 1 → 第 2,866 版起 8 : 1.5 : 0.5（第一幕是卡点，后两幕用估计牌组一场未胜）
    # → 第 8,360 版起 5 : 4.5 : 0.5：第一幕 Boss 胜率已到约八成，第二幕 Boss 成了新卡点，且有了整局里的真实牌组可练
    # → 构筑训练起 2.5 : 6.5 : 1：第一幕 Boss 的真实练习胜率已近七成，名额让给第二幕
    # → 第 9,400 版回退后 3.5 : 5.5 : 1：第一幕练习减半之后，第一幕 Boss 的真实练习胜率从 0.67 一路掉到 0.36，退回一部分
    # → 改为可配置（TrainConfig.boss_act_probs，默认仍是 3.5 : 5.5 : 1），实例上的 boss_act_probs 优先
    BOSS_ACT_PROBS = (0.35, 0.55, 0.10)

    def sample(self, progress: float, boss_fraction: float = 0.0, pools: "SnapshotPools | None" = None,
               real_share: float = 0.0, real_fractions=(0.0, 0.0, 0.0), probe_share: float = 0.0, elite_shares=(0.0, 0.0, 0.0),
               solver=None, solver_share: float = 0.0, mix_fraction: float = 0.0, mix_elite_share: float = 0.0,
               second_share: float = 0.0, chain_share: float = 0.0) -> dict:
        """progress ∈ [0, 1]：0 只打第一幕普通怪、初始牌组；1 覆盖三幕全部遭遇、成型牌组。
        boss_fraction：按这个比例改打 Boss（spec 带 slot="boss"，不计入进度调节）；池里有这一幕的真实快照时，按 real_share 改用快照
        （一个数对三幕都适用，也可按幕给三个数），
        其中 probe_share 用整局中途的快照（spec 带 "_view"：快照那一刻的局面视图，采集进程取走后给牌组打分当标签）。
        real_fractions[幕]：按这个比例（占全部战斗）用这一幕的真实快照重打当时那场普通战或精英战（slot="act1"/"act2"/"act3"，不计入进度调节），
        其中 elite_shares[幕] 打精英战（精英战在整局里远少于普通战，按自然比例只占一成多）；这一幕的池还空着就改打第二幕，第二幕也空就打随机构筑。
        mix_fraction：按这个比例打随机构筑（slot="mix"）：牌组按幕末规模随机配（同 _boss），打该幕的普通战或精英战，
        让模型练到自己不抓的牌；其中 mix_elite_share 打精英战。
        以上都抽完剩下的份额按进度出题（slot 为 None，参与进度调节）。
        solver（solver_pool.SolverPool）：用真实快照的 Boss 名额里，按 solver_share 改打求解器判定能赢的局面（spec 带 "cert"）。
        second_share：第三幕的 Boss 名额里改打“第二个 Boss”真实快照（A10 最后一幕连打两个，带着第一场打完后的血量）的比例。
        chain_share（决策记录 187）：第三幕第一个 Boss 的真实快照里按这个比例连打（spec 带 next_encounter：打赢接着打第二个，按两场结算）。
        第三幕的两类真实快照按遭遇的输率加权抽（record / loss_power，见 _pick）。
        几个比例都为 0 时随机数序列与以前相同（评测的固定战斗不变）。"""
        if boss_fraction > 0 and self.rng.random() < boss_fraction:
            act = int(self.rng.choice(3, p=getattr(self, "boss_act_probs", self.BOSS_ACT_PROBS)))
            if act == 2 and second_share > 0 and pools is not None and pools.boss2 and self.rng.random() < second_share:
                return self._real(pools.boss2, "boss", weight_key=lambda e: ("b2", e["encounter"]))
            rs = real_share[act] if isinstance(real_share, (list, tuple)) else real_share
            if pools is not None and pools.boss[act] and self.rng.random() < rs:
                if solver is not None and self.rng.random() < solver_share:
                    self.counter += 1
                    spec = solver.sample(act, self.rng, self.counter)
                    if spec is not None:
                        return spec
                if pools.probe[act] and self.rng.random() < probe_share:
                    # 中途快照没有对应的 Boss 战：从这一幕最近打过的 Boss 里随机挑一个
                    boss = pools.boss[act][int(self.rng.integers(len(pools.boss[act])))]
                    return self._real(pools.probe[act], "boss", encounter=boss["encounter"])
                if act == 2 and chain_share > 0:
                    firsts = [e for e in pools.boss[2] if e.get("next")]
                    if firsts and self.rng.random() < chain_share:
                        return self._real(firsts, "boss", weight_key=lambda e: ("chain", e["encounter"], e["next"]), chain=True)
                return self._real(pools.boss[act], "boss")
            return self._boss(act)
        if sum(real_fractions) + mix_fraction > 0:
            u, pick = self.rng.random() * (1 - boss_fraction), None
            for act, f in enumerate(real_fractions):
                if u < f:
                    pick = act
                    break
                u -= f
            if pick is not None:
                for act in dict.fromkeys((pick, 1)):
                    spec = self._fight(pools, act, elite_shares[act])
                    if spec is not None:
                        return spec
                return self._mix(mix_elite_share)
            if u < mix_fraction:
                return self._mix(mix_elite_share)
        r, p = self.rng, float(np.clip(progress, 0, 1))
        act = int(r.choice(3, p=self._act_probs(p)))
        enc = str(r.choice(["weak", "regular", "elite", "boss"], p=self._enc_probs(p, act)))
        # 牌组成长：约等于到该幕为止拿到的卡数
        n_added = int(r.integers(0, 1 + round(p * (4 + 6 * act))))
        deck = [{"id": cid, "upgrades": 0} for cid in STARTING_DECK[self.cfg.character]]
        if self.cfg.ascension >= 5:   # 进阶 5（AscendersBane）起初始牌组带一张不可移除的诅咒
            deck.append({"id": "ASCENDERS_BANE", "upgrades": 0})
        for _ in range(int(r.integers(0, 1 + round(p * (1 + 2 * act))))):   # 删去基础牌
            basics = [i for i, d in enumerate(deck) if d["id"].startswith(("STRIKE_", "DEFEND_"))]   # 诅咒不可删
            if len(basics) > 4:
                deck.pop(int(r.choice(basics)))
        for _ in range(n_added):
            rarity = str(r.choice(["Common", "Uncommon", "Rare"], p=[0.55, 0.35, 0.10]))
            deck.append({"id": str(r.choice(self.cards[rarity])), "upgrades": 0})
        up_prob = p * (0.15 + 0.2 * act)
        for d in deck:
            if r.random() < up_prob:
                d["upgrades"] = 1
        n_relics = int(r.integers(0, 1 + round(p * (1 + 3 * act))))
        relics = []
        for _ in range(n_relics):
            rarity = str(r.choice(["Common", "Uncommon", "Rare", "Shop"], p=[0.45, 0.3, 0.15, 0.1]))
            if self.relics[rarity]:
                relics.append(str(r.choice(self.relics[rarity])))
        potions = [str(x) for x in r.choice(self.potions, size=int(r.integers(0, 1 + round(p * 3))))]
        hp_frac = float(r.uniform(0.45, 1.0)) if p > 0 else 1.0
        self.counter += 1
        return {
            "character": self.cfg.character, "ascension": self.cfg.ascension, "seed": f"C{self.counter}_{int(r.integers(1 << 30))}",
            "act": act, "encounter": enc, "deck": deck, "relics": sorted(set(relics)), "potions": potions,
            "hp_fraction": hp_frac,
        }

    # 按学习潜力选题（决策记录 226）：p̂ 的经验分位（第 122,056 版在 2,100 个第三幕第一个 Boss 快照上，价值头估值经逻辑函数换成 p̂，
    # tools/health/p_calib.py），按价值估计在池里的排序分位查表
    PHAT_QUANTILES = [0.0179, 0.0226, 0.0307, 0.0398, 0.049, 0.0657, 0.0818, 0.1002, 0.1118, 0.1297, 0.1453, 0.1605, 0.1775, 0.1941, 0.2163, 0.2315, 0.2492, 0.2641, 0.2805, 0.2955, 0.309, 0.333, 0.3503, 0.3671, 0.3803, 0.4058, 0.4196, 0.445, 0.4625, 0.4725, 0.4943, 0.5091, 0.5322, 0.5486, 0.5671, 0.5845, 0.6064, 0.6199, 0.6358, 0.6522, 0.6748, 0.6874, 0.7024, 0.7142, 0.7263, 0.7377, 0.7492, 0.7656, 0.778, 0.788, 0.7995, 0.807, 0.8172, 0.8259, 0.833, 0.8411, 0.8477, 0.8578, 0.8642, 0.8737, 0.8791, 0.8841, 0.8922, 0.8959, 0.9024, 0.9095, 0.9145, 0.9206, 0.9252, 0.929, 0.9326, 0.9381, 0.9429, 0.9453, 0.9487, 0.9516, 0.9542, 0.9588, 0.9619, 0.9641, 0.9669, 0.9696, 0.9714, 0.9728, 0.9748, 0.9764, 0.9778, 0.9794, 0.9808, 0.9824, 0.9842, 0.9856, 0.9874, 0.9884, 0.9895, 0.9909, 0.9925, 0.9932, 0.9943, 0.9954, 0.9974]
    pq_weight, pq_power, pq_drop, pq_mix = False, 3.0, 0.15, 0.1

    def _pick_pq(self, pool) -> dict:
        """连打练习按学习潜力抽：池里有价值估计的按排序分位查 p̂，权重 ∝ [p̂(1−p̂)]^pq_power，分位低于 pq_drop 的为 0；
        还没有估计的给平均权重；再混 pq_mix 的等概率。"""
        vs = [e.get("v") for e in pool]
        known = sorted(v for v in vs if v is not None)
        if len(known) < 8:
            return pool[int(self.rng.integers(len(pool)))]
        q = self.PHAT_QUANTILES
        def w_of(v):
            r = (np.searchsorted(known, v, side="left") + 0.5) / len(known)
            if r < self.pq_drop:
                return 0.0
            p = q[min(100, int(r * 100))]
            return (p * (1 - p)) ** self.pq_power
        w = np.array([w_of(v) if v is not None else np.nan for v in vs], dtype=float)
        mean_w = np.nanmean(w) if np.isfinite(w).any() else 1.0
        w = np.where(np.isnan(w), mean_w, w)
        w = (1 - self.pq_mix) * w / max(w.sum(), 1e-12) + self.pq_mix / len(w)
        return pool[int(self.rng.choice(len(pool), p=w / w.sum()))]

    # 按遭遇的输率加权抽第三幕 Boss 快照（决策记录 187）：每个键（第二个 Boss 为 ("b2", 遭遇)，连打为 ("chain", 第一个, 第二个)）
    # 记最近练习的输率（指数平均，没数据时按 0.5），抽中的概率正比于 输率 ** loss_power；loss_power 为 0 时等概率
    loss_power = 0.0
    LOSS_EMA = 0.02

    def record(self, key: tuple, win: bool) -> None:
        if not hasattr(self, "loss"):
            self.loss = {}
        self.loss[key] = self.loss.get(key, 0.5) * (1 - self.LOSS_EMA) + self.LOSS_EMA * (0.0 if win else 1.0)

    def _pick(self, pool, weight_key=None) -> dict:
        if weight_key is None or self.loss_power <= 0 or len(pool) < 2:
            return pool[int(self.rng.integers(len(pool)))]
        loss = getattr(self, "loss", {})
        w = np.array([max(0.05, loss.get(weight_key(e), 0.5)) ** self.loss_power for e in pool])
        return pool[int(self.rng.choice(len(pool), p=w / w.sum()))]

    def _real(self, pool, slot: str, encounter: str | None = None, weight_key=None, chain: bool = False) -> dict:
        """整局里的真实局面：快照 + 当时那场战斗的遭遇；只有抽牌、敌人行动等随机数不同。
        Boss 练习带上快照那一刻的局面视图（"_view"，不发给引擎），练习结果就是这副牌的打分标签。"""
        e = self._pick_pq(pool) if chain and self.pq_weight else self._pick(pool, weight_key)
        self.counter += 1
        spec = {"character": self.cfg.character, "ascension": self.cfg.ascension, "seed": f"S{self.counter}_{int(self.rng.integers(1 << 30))}",
                "act": e["act"], "encounter": encounter or e["encounter"], "room": "Boss" if encounter else e["room"], "player_snapshot": e["snap"], "slot": slot}
        if encounter:
            spec["probe"] = True
        if e.get("second"):
            spec["second"] = True
            if e.get("prev"):
                spec["prev_encounter"] = e["prev"]
        if chain:
            spec["next_encounter"] = e["next"]
            if e.get("v") is not None:
                spec["_pv"] = e["v"]      # 快照入池时的价值估计（决策记录 226），不发给引擎，练习结束与输赢一起记下
        if slot == "boss" and e.get("view") is not None:
            spec["_view"] = e["view"]
        return spec

    def chain_problem(self, problems: list, weights: np.ndarray) -> dict:
        """第三幕双 Boss 专项（决策记录 287）：按题目权重抽一题连打，每场换新种子。"""
        e = problems[int(self.rng.choice(len(problems), p=weights))]
        self.counter += 1
        spec = {"character": self.cfg.character, "ascension": self.cfg.ascension, "seed": f"P{self.counter}_{int(self.rng.integers(1 << 30))}",
                "act": 2, "encounter": e["encounter"], "room": "Boss", "player_snapshot": e["snap"], "slot": "boss",
                "next_encounter": e["next"], "_pid": e.get("id")}
        return spec

    def _boss(self, act: int) -> dict:
        """幕末 Boss：牌组按“打到这一幕结尾”的正常规模配（估计值：第一幕约拿 5–9 张牌、1–3 个遗物，后面每幕再多 6–7 张、2–3 个），
        与进度无关。按进度配的牌组在进度 0.3 时最多只多 1 张牌，远弱于整局里真打到 Boss 时的牌组。"""
        deck, relics, potions = self._act_end_deck(act)
        self.counter += 1
        return {
            "character": self.cfg.character, "ascension": self.cfg.ascension, "seed": f"C{self.counter}_{int(self.rng.integers(1 << 30))}",
            "act": act, "encounter": "boss", "deck": deck, "relics": relics, "potions": potions,
            "hp_fraction": float(self.rng.uniform(0.5, 1.0)), "room": "Boss", "slot": "boss",
        }

    # 随机构筑各幕的比例
    MIX_ACT_PROBS = (0.4, 0.4, 0.2)

    def _mix(self, elite_share: float) -> dict:
        """随机构筑：幕末规模的随机牌组（同 _boss）打这一幕的普通战或精英战。真实快照里只有模型自己抓的牌，这一类让它练到别的牌。"""
        act = int(self.rng.choice(3, p=self.MIX_ACT_PROBS))
        enc = "elite" if self.rng.random() < elite_share else "regular"
        deck, relics, potions = self._act_end_deck(act)
        self.counter += 1
        return {
            "character": self.cfg.character, "ascension": self.cfg.ascension, "seed": f"M{self.counter}_{int(self.rng.integers(1 << 30))}",
            "act": act, "encounter": enc, "deck": deck, "relics": relics, "potions": potions,
            "hp_fraction": float(self.rng.uniform(0.5, 1.0)), "room": "Elite" if enc == "elite" else "Monster", "slot": "mix",
        }

    def _fight(self, pools: "SnapshotPools | None", act: int, elite_share: float) -> dict | None:
        """这一幕的真实快照重打当时那场普通战或精英战；池还空着时返回 None。"""
        if pools is None or not (pools.monster[act] or pools.elite[act]):
            return None
        elite = bool(pools.elite[act]) and (not pools.monster[act] or self.rng.random() < elite_share)
        return self._real(pools.elite[act] if elite else pools.monster[act], f"act{act + 1}")

    def _act_end_deck(self, act: int) -> tuple[list, list, list]:
        """打到这一幕结尾的正常规模：牌组、遗物、药水。"""
        r = self.rng
        deck = [{"id": cid, "upgrades": 0} for cid in STARTING_DECK[self.cfg.character]]
        if self.cfg.ascension >= 5:
            deck.append({"id": "ASCENDERS_BANE", "upgrades": 0})
        for _ in range(int(r.integers(0, 2 + act))):     # 删牌：商店、事件
            basics = [i for i, d in enumerate(deck) if d["id"].startswith(("STRIKE_", "DEFEND_"))]
            if len(basics) > 4:
                deck.pop(int(r.choice(basics)))
        for _ in range(int(r.integers(5 + 6 * act, 10 + 7 * act))):
            rarity = str(r.choice(["Common", "Uncommon", "Rare"], p=[0.55, 0.35, 0.10]))
            deck.append({"id": str(r.choice(self.cards[rarity])), "upgrades": 0})
        for d in deck:                                    # 升级：休息处锻造、事件
            if r.random() < 0.15 + 0.2 * act:
                d["upgrades"] = 1
        relics = []
        for _ in range(int(r.integers(1 + 2 * act, 4 + 3 * act))):
            rarity = str(r.choice(["Common", "Uncommon", "Rare", "Shop"], p=[0.45, 0.3, 0.15, 0.1]))
            if self.relics[rarity]:
                relics.append(str(r.choice(self.relics[rarity])))
        potions = [str(x) for x in r.choice(self.potions, size=int(r.integers(0, 3)))]
        return deck, sorted(set(relics)), potions

    @staticmethod
    def _act_probs(p: float) -> list[float]:
        w = np.array([1.0, p, p * p]) + 1e-9
        return list(w / w.sum())

    @staticmethod
    def _enc_probs(p: float, act: int) -> list[float]:
        w = np.array([1.0, 0.5 + p, 0.4 * p, 0.25 * p]) if act == 0 else np.array([0.3, 1.0, 0.5, 0.3])
        return list(w / w.sum())


class SnapshotPools:
    """整局里攒下的真实局面（每个采集进程一份）：选路线时的玩家快照 + 紧接着那场战斗的遭遇（引擎的 Snapshot.Extra）。
    boss[幕]：各幕 Boss 战前；monster[幕] / elite[幕]：各幕每场普通战、精英战前；
    probe[幕]：整局中途随机抽的选路线时刻（不管下一场打什么），拿去打这一幕的 Boss，给牌组打分提供“牌还没攒齐”时的标签。
    每个池只留最近 size 份，跟着策略的牌组一起变。快照附带那一刻的局面视图（features.deck_view），牌组打分用。"""

    def __init__(self, size: int):
        self.boss = [deque(maxlen=size) for _ in range(3)]
        self.probe = [deque(maxlen=size) for _ in range(3)]
        self.monster = [deque(maxlen=size) for _ in range(3)]
        self.elite = [deque(maxlen=size) for _ in range(3)]
        self.boss2 = deque(maxlen=size)      # 第三幕第二个 Boss 战前（同时也在 boss[2] 里）

    def add(self, combat: dict, snap: str, view=None, second: bool = False, next_encounter: str | None = None, prev_encounter: str | None = None) -> dict:
        e = {"act": combat["act"], "encounter": combat["encounter"], "room": combat["room"], "snap": snap, "view": view}
        if next_encounter:      # 第三幕第一个 Boss：紧接着的第二个 Boss（连打练习用，决策记录 187）
            e["next"] = next_encounter
        if prev_encounter:      # 第二个 Boss：本幕已经打赢的第一个（观测头部 16，决策记录 194）
            e["prev"] = prev_encounter
        if combat["room"] == "Boss" and combat["act"] < 3:
            self.boss[combat["act"]].append(e)
            if second and combat["act"] == 2:
                e["second"] = True
                self.boss2.append(e)
        elif combat["act"] < 3 and combat["room"] == "Monster":
            self.monster[combat["act"]].append(e)
        elif combat["act"] < 3 and combat["room"] == "Elite":
            self.elite[combat["act"]].append(e)
        return e

    def add_probe(self, act: int, snap: str, view) -> None:
        if act < 3:
            self.probe[act].append({"act": act, "encounter": None, "room": "Boss", "snap": snap, "view": view})
