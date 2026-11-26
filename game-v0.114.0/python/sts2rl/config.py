"""训练配置。所有规模参数都可在命令行覆盖，便于在租用机上按标定结果调整。"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field

from .model import ModelConfig


# 人类 A10 Defect 赢局在第 0–49 层的平均牌数（knowledgedemon.com/api/analytics/deck-size-curve，赢局 799 局，第 49 层只有 7 局）
DECK_SIZE_REF = (11.7, 12.6, 13.2, 13.9, 14.7, 15.3, 15.8, 16.3, 16.7, 16.7, 17.2, 17.7, 18.2, 18.9, 19.5, 19.5, 20.5, 20.3, 21.0, 21.6,
                 22.1, 22.7, 23.3, 23.7, 24.1, 23.9, 24.3, 24.7, 25.1, 25.7, 26.3, 26.4, 27.2, 27.4, 27.9, 28.5, 29.0, 29.5, 30.0, 30.3,
                 30.3, 30.7, 31.3, 31.6, 32.2, 32.8, 33.1, 33.1, 33.1, 33.6)


# 各角色的牌组规模参考（牌组势能 deck_over_coef 用）；没有人类数据的角色退回故障机器人的（deck_over_coef 为 0 时用不到，储君训练即如此）
DECK_SIZE_REFS = {"DEFECT": DECK_SIZE_REF}


def deck_size_ref(character: str) -> tuple:
    return DECK_SIZE_REFS.get(character, DECK_SIZE_REF)


@dataclass
class TrainConfig:
    run_dir: str = "runs/defect_combat"
    # 角色（2026-11-06 起可配，储君观测调研.md）：整局、战斗课程、评测、求解器导出都用它；卡池、遗物池、药水池按角色取（curriculum.CurriculumConfig.for_character）。
    # 储君的帧是观测格式 20（引擎按角色定），模型要 n_zones ≥ 19（已知牌堆顶的区域 16–18）
    character: str = "DEFECT"
    # 规模（租用机上按 calibrate.py 的结果设置）
    actors: int = 2                 # 采集进程数
    envs_per_actor: int = 6         # 每个采集进程管理的引擎进程数
    actor_device: str = "cpu"       # 采集进程的推理设备：cpu 或 cuda
    learner_device: str = "cpu"     # 学习进程的设备
    actor_threads: int = 1          # 采集进程的 torch 线程数（CPU 推理时）
    learner_threads: int = 4
    infer_wait_ms: float = 2.0      # 采集进程凑批的最长等待
    # PPO
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 3
    batch_steps: int = 16384        # 每次更新使用的决策数
    minibatch: int = 2048
    # 梯度累积（决策记录 225）：每 grad_accum 个小批累积一次梯度再走一步优化，有效小批量 = minibatch × grad_accum；显存与每样本计算量不变。
    # Adam 的噪声温度约为 学习率 ÷ √有效小批量（Malladi 等 2022），学习率不变时累积 ×2 约等于把噪声温度降到 0.71 倍
    grad_accum: int = 1
    lr: float = 3e-4
    # 参数的指数滑动平均（决策记录 223）：学习端每版更新后 ema ← d·ema + (1 − d)·当前参数，随权重文件发布（ema_state）、随检查点保存（ema）；
    # 采集照用当前参数，评测打开 eval_use_ema 时用它。0 为不做。0.998 约等于最近 500 版的平均
    ema_decay: float = 0.0
    # 优化器（储君决策记录第 25 条）："adam" 为原来的；"muon" 时隐藏层的二维权重（编码器的注意力与前馈、各头的隐藏层、中间投影，
    # 见 sts2rl/muon.py 的 select）改用 Muon，其余参数照旧 Adam。muon_lr 按 Liu 等 2025 的口径：更新 = muon_lr × 0.2·√max(行, 列) × 正交化的动量，
    # 每步均方根约 0.2 × muon_lr；None 时同 lr。动量 muon_momentum（插值式、Nesterov），Newton–Schulz 迭代 muon_ns_steps 次。
    # 第一次切到 Muon 时动量接着 Adam 的一阶矩；之后存进检查点的 muon
    optimizer: str = "adam"
    muon_lr: float | None = None
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    opt_warmup: int = 50           # 检查点不带优化器状态（模型扩容）时，学习率逐步升上去的更新次数
    ent_coef: float = 0.01
    # 局外决策（选路线、抓牌、事件、商店、休息处）的熵系数，其余步仍用 ent_coef（2026-10-25 起）。
    # 局外策略几乎是确定的（熵约 0.16，全部决策约 0.72）：从不跳过抓卡、休息处从不升级、商店从不删卡，
    # 不去试别的选项，单独的局外价值头也学不到它们更好
    ent_coef_macro: float = 0.05
    # 地图决策（选路）的熵系数，None 沿用 ent_coef_macro；ent_map_until 为生效到第几版（不含），之后自动回到 ent_coef_macro，None 为一直生效。
    # 2026-10-27 用户决定：路线摘要上线后两条线临时调到 0.05、持续 750 版，让模型多试不同路线
    ent_coef_map: float | None = None
    ent_map_until: int | None = None
    # 局外决策的强制探索（2026-10-28 起，决策记录 167）：整局训练里的局外决策按类别以 explore_probs 的概率不按策略选，
    # 改为在其他合法选项里均匀随机选一个（古神三选一只在带遗物的选项里选）。熵奖励对选中率接近 0 的选项几乎没有梯度，
    # 只有真去试，价值网络才见得到“拿了这件遗物、这张牌”之后的局面。被改的那一步是离策略的：不进策略梯度、价值目标取原值，
    # 前面各步的优势在这一步截断（用这一步的价值估计自举）。类别：ancient、card_reward、shop、rest、event、card_select、map；
    # 空字典为关闭。explore_until 为生效到第几版（不含），None 为一直生效
    explore_probs: dict = field(default_factory=dict)
    # 改选换成哪一项（决策记录 202）：{类别: [温度 T, 均匀份额 ε]} 的类别不再均匀挑，替代项按 (1 − ε)·π^(1/T) 归一 + ε·均匀 抽，
    # 多落在策略排第二、第三的选项上，比率 π/μ 不再那么极端；ε 留给策略几乎不选的选项。不在表里的类别照旧均匀挑；空字典为全部均匀
    explore_temper: dict = field(default_factory=dict)
    explore_until: int | None = None
    # 改选纳入行为分布（决策记录 197）：采集端在会改选的界面上按行为分布 μ 记出手概率（actor.explore_logmu），学习端比率用 π_新 / μ，
    # 改选的步正常进策略梯度、价值目标用实际回报（前面各步的优势仍在改选处截断）。False 为原来的办法
    explore_offpolicy: bool = False
    # 战斗里的定向强制探索（决策记录 238）：{牌: {房间: 概率}}，房间为 Boss、Elite、Monster；空字典为关闭，行为与原来完全相同。
    # 每个回合只在第一个“名单牌有能打的出牌候选”的战斗决策抽一次签：候选里每张牌独立以自己的概率举手，有人举手就在举手的牌里均匀选一张；
    # 策略抽中的已经是名单牌时不改。名单牌要求只有一个出牌候选（有目标的牌、手里有两张同名的都不算）。行为分布见 actor.combat_explore_logmu，
    # 必须同时开 explore_offpolicy（否则采集端与学习端都拒绝启动）。combat_explore_gate：门关时这一回合不改，loss_frac 为“现在结束回合预计掉血”
    # 不低于当前生命的这个比例，kill 为本回合能杀光全部敌人。combat_explore_until 为生效到第几版（不含）；combat_explore_cut 为在战斗改选处截断前面的优势（默认不截断）
    combat_explore: dict = field(default_factory=dict)
    combat_explore_gate: dict = field(default_factory=lambda: {"loss_frac": 0.5, "kill": True})
    combat_explore_until: int | None = None
    combat_explore_cut: bool = False
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    # 只读价值诊断：独立哈希抽样，不消费训练 RNG，不改 loss/目标/优化器。
    value_probe: bool = False
    value_probe_every: int = 20          # 每多少次更新记录分层矩并重评小样本；启动首批也记录
    value_probe_samples: int = 128       # 当前/旧/去特权预测比较的局外样本数（另取 1/4 战斗步）
    value_probe_grad_every: int = 200    # 只读共享梯度探针；启动首批也记录
    value_probe_episode_frac: float = 0.02  # 另存完整奖励时间轴与局外原始观测的整局比例
    value_probe_max_gb: float = 8.0      # 远端未搬回的 NPZ 上限；满时暂停 NPZ，矩统计仍继续
    learner_bf16: bool = True       # 学习时前向用 bf16 混合精度：实测快约 23%，概率比偏差平均 0.2%、p99 1.5%（远小于裁剪 0.2）
    # 学习端提速（2026-10-27，决策记录 144）：learner_fast 打开打包变长注意力与 index_select（PolicyValueNet.fast，数学等价）；
    # learner_compile 再对 PPO 小批循环里的前向套 torch.compile（dynamic=True，进程启动后第一遍多约 30 秒编译）。
    # 3090 上用 B 导出的真实数据离线测：两项合起来每遍快约 1.5–1.7 倍。默认关，与原实现完全一样
    learner_fast: bool = False
    learner_compile: bool = False
    collate_workers: int = 1        # 学习时组批的线程数（2026-10-28，决策记录 158）：去掉主机同步后学习快了，单线程组批跟不上，主线程每批等约 1.2 秒
    target_kl: float = 0.02         # 一遍训练后近似 KL 超过 1.5 倍此值就停止本轮更新
    max_policy_lag: int = 3         # 轨迹的策略版本比当前落后超过这个数就丢弃
    # 主线 B 上 3 → 6（第 77,622 版起，决策记录 177，用户：第一幕只有约 84% 入训，要推到 95–99%）：改之前近 300 版各幕入训都只有约 79%，
    # 被排除的几乎全是 B 本机采集进程的步（本机送来的约 56% 入训，A 送来的 99%）。落后的步仍按采集时的旧对数概率算概率比并截断
    # （PPO 本来就这样处理离策略），学习端的速度不变。收到时落后几版的分布记在 lag_hist、run_lag_act
    # 整局分段提交（2026-10-28，决策记录 169）：整局没送出的部分满这么多步后，在下一个选路线决策前切一段先送，
    # 下一段第一步的价值作自举。原来整局打完才送，长局开头的步大多排到过旧（第一幕只有约 34% 入训）。None 为整局打完才送
    run_segment_steps: int | None = None
    queue_steps: float = 1.5        # 轨迹队列容量（以 batch_steps 的倍数计）；满时采集进程暂停，避免产生会被丢弃的数据
    # 主线 B 上 1.5 → 0.75（第 78,004 版起，决策记录 178）：开双机协同时本机采集的轨迹要先后排两道这么大的队列（采集队列、取数缓冲），
    # A 的只排后一道，本机的步因此常落后 4–8 版；减半后整局各幕入训 92.7% → 100%，学习端每版从 3.50 秒变成 3.30 秒
    # 双机协同时本机采集队列只留这么多条（决策记录 197：本机轨迹先过这一级、再过取数缓冲，远程的只过后一级；
    # 实测本机送出时只落后 0–2 版，过旧都出在排队上）。None 为按 queue_steps 算
    local_queue_items: int | None = None
    queue_item_steps: int = 50      # 队列按条数限容，这是换算用的每条轨迹平均步数（实测每局约 50 步；原来按 16 算，队列实际能囤约 4.7 批，4090 机上 44% 的步数排到过旧被丢）
    prefetch_steps: float = 0.0     # 学习进程后台预取的轨迹量（以 batch_steps 的倍数计），取轨迹与学习重叠；0 为关闭（主线程直接从队列取）。
                                    # 4090 机实测：取数时间 1.26→0.95 秒，但学习线程争 GIL，学习 2.67→3.05 秒，总体略慢，所以默认关
    # 中途开局（技术路线第十四节，2026-10-26）：整局按比例从第一层、第二幕开头、第三幕开头开始。开幕快照取自本采集进程
    # 正常整局换幕时导出的（引擎 ActStart），每幕留 act_start_pool 份，池子还空时退回从第一层开。默认全部从第一层开（关闭）
    run_start_probs: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    act_start_pool: int = 200
    actor_nice: int = 5             # 采集进程与引擎的 nice 增量（学习进程不变），CPU 吃满时让学习先跑
    # 奖励（战斗模式）：胜 +1、负 -1，加上血量变化与药水消耗
    win_reward: float = 1.0
    lose_reward: float = -1.0
    hp_coef: float = 1.0            # × (结束血量 − 开始血量) / 最大生命
    potion_coef: float = 0.05       # 每用一瓶药水的代价（战斗课程；Boss 战不扣，见 actor._finish）
    # 整局（A10 Defect）：每个采集进程中这一比例的引擎打整局，其余打战斗课程。
    # 0.3 → 0.5（构筑训练起）：战斗课程已稳定，重心移到整局里的路线、抓牌、事件、商店
    run_fraction: float = 0.5
    floor_reward: float = 0.02      # 每多到一层
    # 打过幕末 Boss（进入下一幕）的额外奖励。第一幕 Boss 是主要卡点，打过它比整个第一幕的爬层奖励（16 × 0.02）还值钱；
    # 第三幕 Boss 即通关，见 victory_reward
    act1_boss_reward: float = 0.5
    act2_boss_reward: float = 0.5   # 0.3 → 0.5（第 8,360 版起）：第二幕 Boss 成了新卡点
    victory_reward: float = 1.0     # 通关
    # 第三幕打赢第一个 Boss（A10 最后一幕连打两个）的里程碑奖励；0 为不给。2026-10-26 计划阶段 0 设为 0.5（写在训练配置里）
    act3_boss_reward: float = 0.0
    act3_boss2_reward: float = 0.0  # 打赢第三幕第二个 Boss（即通关）的里程碑奖励，在 victory_reward 之外另加（2026-10-26 起）
    # 训练加血与通关余血奖励（2026-11-20，铁甲战士决策记录第 24 条；有意引入的有偏信号）：
    # train_bonus_hp：只有训练采集的 reset 带 "bonus_hp"（引擎 HeadlessRun.BonusHp），整局与战斗练习建局时最大生命、当前生命各加这么多；
    #   评测、实机、搜索、各种测试脚本都不带，与原来完全相同。0 为关闭（请求与原来完全相同）。
    # victory_hp_bonus：整局通关且终局血量大于 victory_hp_margin 时，在通关奖励之外再加这么多（记在最后一步）；0 为关闭。
    #   余血门槛取同加血量的 20：通关时多出来的血不止这 20 点，才大致算“不加血也能赢”。日志 run_choices 记 VICTORY_HP:ok / VICTORY_HP:low 两类通关局数
    train_bonus_hp: int = 0
    victory_hp_bonus: float = 0.0
    victory_hp_margin: int = 20
    # 第三幕中途余血奖励（2026-11-20，铁甲战士决策记录第 25 条；有偏信号）：整局里打赢第三幕第一个 Boss、接着打第二个时，
    # 最后一步再加 boss3_hp_coef × clip((血量 − 本局加血) / (最大生命 − 本局加血), 0, 1)，即按“不加血时还剩几成血”给，满血为系数本身。
    # 血量取打完第一个 Boss 后的第一帧（两个 Boss 之间没有别的决策，就是进第二个 Boss 时的血量）。0 为关闭。
    # 日志 run_choices 记 BOSS3_HP:n（场数）、BOSS3_HP:hp（血量合计）、BOSS3_HP:frac（比例合计）
    boss3_hp_coef: float = 0.0
    # 血量塑形（第 9,400 版回退后起）：整局里血量每变化 1 点记 run_hp_shaping / 开局最大生命，记在导致变化的那一步。
    # 死亡时血量为 0，所以整局合计约为固定的 −run_hp_shaping，不改变最优策略，只把“掉血不好、回血好”提前告诉策略。
    # 跑偏那一段的评测里，牌数、遗物数都没变，到第一幕 Boss 时的血量从 0.82 掉到 0.69，第一幕 Boss 胜率随之从 0.76 掉到 0.60
    # 血量的边际价值（2026-10-25 起，用户决定）：势能改为 run_hp_shaping × (最大生命 / 开局最大生命) × V(血量比例)，
    # V(x) = x + hp_concavity × x(1 − x)，每点血的价值从满血时的 0.5 倍升到空血时的 1.5 倍（相对线性）。整局合计仍约为 −run_hp_shaping。
    # 0.2 → 0.3：平均每点血的权重提高一半，满血附近掉血的代价（0.15）比原来（0.2）略低，半血以下明显更高。
    # 休息回 30% 最大生命：血量 80% 时 +0.036，70% 时 +0.059，50% 时 +0.077，30% 时 +0.095
    run_hp_shaping: float = 0.3
    hp_concavity: float = 0.5
    # 白丢药水的惩罚（2026-10-25 起，用户决定）：引擎只在需要腾位置领新药水时才提供丢药水（奖励界面、商店，药水栏满），
    # 整局里丢了药水、直到离开这个场景（下一个选路线决策或对局结束）都没领到新药水，每瓶记这么多惩罚在丢药水的那一步
    potion_discard_penalty: float = 0.1
    # 牌组规模与升级的塑形（2026-10-25 起，用户决定）：整局里每到一个战斗外的决策，把势能
    #   Φ = −deck_over_coef × max(0, 牌数 − (DECK_SIZE_REF[层] + deck_size_margin)) + upgrade_coef × 已升级的卡数
    # 的变化记在上一步：超出上限后每多拿一张 −deck_over_coef（删一张 +），升级一张 +upgrade_coef（删掉已升级的卡 −）。
    # 上限取人类的数据：knowledgedemon.com 的 deck-size-curve（Defect、A10、赢局 799 局在各层的平均牌数，2026-10-25 取），加余量。
    # 升级 0.05 → 0.08（2026-10-25 起，用户决定：升级要比相应的回血值钱）：对照上面休息的收益，血量约 46% 以上升级更划算，
    # 更低时休息更划算（原来 0.05 对线性的 +0.06，只有约 75% 血以上才划算）
    # 0.08 → 0.04（第 16,340 版之后，用户决定：现阶段多休息更强，但仍保留一点升级）：每局升级从 2.3 涨到 3.6 次，
    # 到第二幕 Boss 时剩血从 65% 掉到 56%。0.04 时约 77% 血以上升级更划算，更低时休息更划算
    # 主线 B 上升级（第 158 条）、稀有卡早已设为 0；牌组超限惩罚 deck_over_coef 0.03 → 0（第 68,850 版起，决策记录 174，用户：“取消卡组大小奖惩”），
    # 牌组势能从此恒为 0。这里的默认值不变，改动在 B 的 train_config.json
    deck_over_coef: float = 0.03
    deck_size_margin: float = 2.0
    upgrade_coef: float = 0.04
    # 战损、药水与遗物（2026-10-25 起，用户决定）：整局里
    #   非 Boss 战结束时按这场的净掉血（开打时血量 − 结束时血量，战斗中回的血抵掉一部分）再扣 combat_dmg_coef × 掉血 / 开打时最大生命，
    #   不经过血量势能，所以休息回血的收益不变；高血量时掉 10% 血合计约 −0.028（血量势能 −0.018 + 这里 −0.01）。真实扣分，改变最优策略：战斗中少掉血本身成了目标；
    #   非 Boss 战里每用一瓶药水扣 run_potion_coef（和战斗课程的 potion_coef 一致），高血量时约要省下 20% 的血才划算，把药水留给 Boss 战；
    #   遗物势能 relic_coef × 手上遗物数，拿到一件 +relic_coef，失去扣回，补上精英战（以及宝箱、商店、事件）的回报：
    #   高血量时精英战掉血少于约 20% 净收益为正。
    combat_dmg_coef: float = 0.1
    run_potion_coef: float = 0.05
    # 精英战用药水不扣分（整局的 run_potion_coef 与战斗课程的 potion_coef 都不扣；2026-10-27 起 A 上的对照实验，用户决定）
    elite_potion_free: bool = False
    relic_coef: float = 0.05
    # 金币奖励改回由策略自己领（2026-10-28，决策记录 160；原来引擎自动领，见决策记录 68）。manual_gold_start 为 None 时照旧自动领；
    # 给定版本号后平稳过渡：从这一版起的 manual_gold_span 版内，每局开局以 (版本 − start) / span 的概率关掉自动领，之后全部由策略领。
    # 评测在过渡期内照旧自动领（曲线前后可比），过渡结束后改为由策略领
    manual_gold_start: int | None = None
    manual_gold_span: int = 2000
    # 跳过无负面奖励的惩罚（2026-10-28，决策记录 160）：整局里奖励界面点离开时还剩的无负面遗物（actor.NO_DOWNSIDE_RELICS）与金币、
    # 宝箱里跳过无负面遗物，每件在这一步记 −relic_skip_penalty / −gold_skip_penalty。系数从 skip_penalty_start 版起
    # skip_penalty_span 版内线性降到 0，之后自行退出（只为把“拿”这个习惯扶起来，不长期改变奖励）；skip_penalty_start 为 None 时不扣
    relic_skip_penalty: float = 0.0
    gold_skip_penalty: float = 0.0
    skip_penalty_start: int | None = None
    skip_penalty_span: int = 4000
    # 用了不扣分的药水（整局普通战与战斗课程都免扣；2026-10-27 起，用户决定）：药水形状的石头只造成 15 点伤害，不值得留到 Boss 战，
    # 普通战里用掉还能腾出药水栏
    potion_free: list = field(default_factory=lambda: ["POTION_SHAPED_ROCK"])
    # 稀有卡（金卡，目录 rarity 为 Rare，含无色稀有；2026-10-25 起，用户决定）：牌组势能里每张 +rare_coef，删掉扣回。
    # 牌组超出上限时抓一张稀有卡仍净得 +0.02（0.05 − 0.03），即稀有卡在牌多时也值得抓；普通、罕见卡不加分
    rare_coef: float = 0.05
    # 固定种子评测（单独进程）：每隔 eval_every_s 秒（0 为连续），用当前策略贪心打 eval_runs 局整局。
    # 评测只用训练闲置的资源：推理在 CPU 上，进程与引擎以 SCHED_IDLE 调度，CPU 被训练占用时自动让路。
    eval_every_s: float = 900.0
    eval_runs: int = 48
    eval_workers: int = 4           # 评测工作进程数，每个进程管 eval_envs 个引擎、在 CPU 上单线程推理
    eval_device: str = "cpu"        # 评测的网络推理放在哪里：cpu 或 cuda（显卡有空时省下 CPU 给采集；推理按 FP32、关 TF32）
    eval_cpu_cores: float = 5.0     # 评测（含引擎）最多用几个核：用 cgroup 硬限制。超线程上再多会和训练抢物理核
    # 单局步数上限：策略可能在某个决策点来回打转（引擎的管道模式不限步数），超过就按卡住结束、记为失败
    max_steps_run: int = 6000
    max_steps_combat: int = 600
    # 整局里单场战斗的步数上限（2026-11-18，亡灵契约师决策记录第 4 条补记二）：从开打算起超过就按卡住结束整局、记为失败。
    # 整局只在选路前切段提交（run_segment_steps），一场打转几千步的战斗会整段一次交给学习端；None 为不限（与原来相同）
    # 常规评测（evaluator.run_evaluator）也用它：整局评测里连续在战斗中超过这么多步按卡住结束（第 10 条）
    max_steps_fight: int | None = None
    # 局面重复截断（2026-11-07，用户，见引擎 RunSession.LoopCut）：同一回合里同一局面（手牌、玩家生命与格挡）第 4 次出现、
    # 敌人生命加格挡几乎没变（不到 1%）时只给“结束回合”。采集与评测的 reset 都带上；False 时与原来完全相同
    loop_cut: bool = False
    # 跨回合无进展判输（2026-11-07，储君决策记录第 13 条，见引擎 RunSession.StallTurns）：连续这么多个玩家回合没有对敌人造成生命伤害，
    # 这场战斗判输（整局以死亡结束）。治模型剩一只打不疼的弱敌时有攻击牌也不打、一回合接一回合拖下去。采集与评测的 reset 都带上；0 为关，与原来完全相同
    stall_turns: int = 0
    # 评测用的判输线（2026-11-07，储君决策记录第 13 条补记，用户：“调低一点，但是别太狠，到8回合吧。”训练 8、评测 20）：
    # 训练按概率出手，拖不满 20 回合，吃不到拖延的惩罚；训练调低、评测照旧。None 为与 stall_turns 相同
    eval_stall_turns: int | None = None
    # 超长局也要入训（2026-11-07，用户：“这种超长局也要纳入训练，就是即使版本过旧，也不能因为过旧被抛掉”）：一份轨迹（整局的一段或一场战斗）
    # 有这么多步以上时，里面的步不按 max_policy_lag 判过旧（版本差为负、即回退作废的时间线上的照旧不要）；None 为不放宽
    long_keep_steps: int | None = None
    # 评测单局的时长上限（秒），超过按卡住结束；原来写死 600
    eval_time_cap: float = 600.0
    eval_envs: int = 4
    eval_nice: int | None = None    # None：评测以 SCHED_IDLE 调度（只用空闲 CPU）；给整数时为普通调度、nice 加这么多（决策记录 198）
    # 评测种子的口径名（写进 eval.jsonl 的 seed_set，看板按它认 P1 评测）；空时按种子数写“N 个分层 RT 种子”。remote_eval.py 按种子池名设，
    # 单机训练时在配置里直接给（储君暂时在 A 上单机训练，2026-11-06）
    eval_seed_set: str = ""
    eval_seeds: list | None = None  # 评测种子；None 为 EVAL000…（eval_runs 个）。决策记录 198 起主线用 tools/remote/eval_seeds.json 里按难度分层抽的 RT 种子
    eval_merge_equiv: bool = False  # 评测的贪心按等价动作合并（两张一模一样的牌合计概率，sts2rl.greedy，决策记录 205）；False 为逐候选取最大
    eval_use_ema: bool = False      # 评测用权重文件里的滑动平均权重 ema_state（决策记录 223）；文件里没有这一项时照常用当前参数
    eval_keep_runs: int = 0         # 每轮评测的逐局结果（复测的格式）留最近多少轮到 run_dir/eval_runs/（另留最近 2 轮的权重），复测同一版本时沿用，0 为不留（决策记录 203）
    eval_combats: int = 128         # 每轮评测另打的单场战斗（课程按进度 0.5 抽题，看板评测明细的“战斗评测”）；决策记录 198 起主线为 0
    # 评测的每局整局（不论输赢）另存一份回放到 run_dir/eval_replays/（2026-10-29 起，决策记录 175）：给实时胜率模型（tools/winprob/）攒数据。
    # 只多写磁盘（回放本来就逐步记着，原来只存通关的局到 replays/），不加算力；每局约 40 KB
    eval_replay_all: bool = False
    # 训练采集里从第一层开打的整局按这个比例抽样另存回放到 run_dir/run_replays/（决策记录 236）：新随机种子、不论输赢，给实时胜率模型攒数据；
    # 只多算抽中的局每步的 replay.describe，目录里只留最近 run_replay_keep 局，更旧的自动删掉
    run_replay_frac: float = 0.0
    run_replay_keep: int = 20000
    exam_every: int = 500           # 固定考卷（运行目录的 exam.json）每隔多少版打一遍；没有 exam.json 或为 0 就不打（决策记录 197 起主线不打）
    calib_log: bool = True          # 评测整局的 Boss、精英战开打状态写 calib.jsonl（离线拟合资源边际价值用；决策记录 197 起主线关）
    card_log_every: int = 1         # 训练日志里选牌、古神遗物的计数每隔多少次更新写一次（中间累计）
    best_keep: int = 5              # 评测成绩前几名的权重留档到 run_dir/best/（见 evaluator._archive_best）
    # 评测偏高的权重另外留档到 run_dir/high/（2026-10-31 用户：“以后评测偏高的，也要自动备份”，决策记录 208）：
    # 通关率比之前 eval_high_window 轮（同局数）的平均高出 eval_high_margin 就留，按时间留最近 eval_high_keep 份（每份约 72 MB）
    eval_high_margin: float = 0.03
    eval_high_window: int = 20
    eval_high_keep: int = 40
    # 课程
    progress_init: float = 0.0
    progress_step: float = 0.02
    progress_up_winrate: float = 0.75
    progress_down_winrate: float = 0.45
    boss_fraction: float = 0.2      # 战斗课程里固定打 Boss 的比例（幕末牌组，不受进度限制，不计入进度调节）。0.15 → 0.2（构筑训练起）
    boss_win_bonus: float = 0.5     # 战斗课程的 Boss 名额打赢时，在 win_reward 之外再加
    # Boss 名额里各幕的权重（会归一化）；默认同 CombatCurriculum.BOSS_ACT_PROBS
    boss_act_probs: tuple = (0.35, 0.55, 0.10)
    # 战斗课程里打输的 Boss 战，按打掉的敌人血量比例（开场敌人最大生命合计为分母）加 boss_dmg_coef × 比例；0 为不加
    boss_dmg_coef: float = 0.0
    # Boss 名额里改用整局真实快照的比例（这一幕的快照池还空着时用估计牌组）；可按幕给 [第一幕, 第二幕, 第三幕]。
    # 2026-10-27 第三幕改为 1.0：估计牌组打第三幕 Boss 几乎赢不了（主线换新奖励以来约 1.2 万场只赢 8 场），练不出东西
    boss_real_share: float | list = 0.75
    # 路线摘要（观测格式版本 7，2026-10-27 起，用户决定两条线都开）：地图决策时给每个现在可前往的节点加一条记录，
    # 汇总从它出发到本幕 Boss 的全部路线（各类房间最少最多几个、精英与休息处的取舍、精英前有没有火堆），见 观测格式.md
    route_summary: bool = False
    # 采集用的引擎（2026-10-27 起可选）：csharp 为游戏本体（Sts2Headless），rust 为 rust-engine/ 的重写引擎（快、与 C# 逐步对拍
    # 约 99.4% 一致；不支持 route_summary 与中途开局 run_start_probs）。评测进程不看这一项，始终用 C#（准绳）
    engine: str = "csharp"
    # 战斗课程里用各幕真实快照重打普通战、精英战的比例（占全部战斗，不计入进度调节）及其中打精英战的比例。
    # 第二幕 0.15 → 0.35（构筑训练起）→ 0.4；第一、三幕 0 → 0.15、0.1（真实局面扩到三幕起）
    act1_fraction: float = 0.15
    act1_elite_share: float = 0.5
    act2_fraction: float = 0.4
    act2_elite_share: float = 0.5   # 0.4 → 0.5：精英战在整局里只占一成多，第二幕精英每局死 0.23 次、与普通战相当
    act3_fraction: float = 0.1
    act3_elite_share: float = 0.4
    # 随机构筑：幕末规模的随机牌组打普通战、精英战，让模型练到自己不抓的牌（取代按进度出题，后者只剩下兜底的份额）
    mix_fraction: float = 0.15
    mix_elite_share: float = 0.4
    probe_prob: float = 0.2         # 整局里每次选路线时，以这个概率把当时的快照存进中途快照池（SnapshotPools.probe）
    probe_share: float = 0.3        # 用真实快照的 Boss 名额里，改用中途快照的比例：结果给牌组打分当“牌还没攒齐”时的标签
    # 房间结果记录（2026-10-27 起，决策记录 146，见 room_log.py）：整局里真打的战斗与用真实快照重打的练习战，按这个比例抽样记下
    # 开打时的玩家快照、遭遇与结果，给“房间结果模型”（辅助选路搜索）当训练数据。0 为不记
    room_log: float = 0.0
    # 双机协同（2026-10-28 起，决策记录 155，见 bridge.py）：学习进程在内网地址 bridge_bind:bridge_port 上收远程采集机送来的轨迹、
    # 向它发最新权重；bridge_token_file 是 32 字节口令（十六进制）。bridge_port 为 0 时不开
    bridge_port: int = 0
    bridge_bind: str = ""
    bridge_token_file: str = ""
    snapshot_pool: int = 256        # 每个采集进程每个快照池保留最近多少份
    # 求解器筛选的 Boss 练习（2026-10-25 起，见 solver_pool.py）：本机用 Combat Solver 标注真实快照能不能赢，
    # 真实快照的 Boss 名额里 solver_share 改打“能赢”的局面，模型还没赢过的优先
    solver_share: float = 0.5
    # 第三幕的 Boss 名额里，按这个比例改打“第二个 Boss”的真实快照（A10 最后一幕连打两个，带着第一场打完后的血量；2026-10-26 起）。
    # 自然出现的比例约一成（第一个 Boss 的胜率），0 为不额外加
    second_boss_share: float = 0.0
    # 第三幕 Boss 连打练习（2026-10-29，决策记录 187）：第三幕第一个 Boss 的真实快照里按这个比例连打——打赢后带着当时的血量、药水
    # 接着打这一局的第二个 Boss，只按两场的总结果结算（中途不扣血、不计药水）；0 为不连打
    boss_chain_share: float = 0.0
    # 第三幕两类真实快照（第二个 Boss、连打）按遭遇的输率加权抽：概率正比于 最近练习输率 ** boss_loss_power（各采集进程自己记）；0 为等概率
    boss_loss_power: float = 0.0
    # 第三幕连打练习按学习潜力选题（决策记录 226，默认关，代码备好未上线）：快照入池时记下整局里第一个 Boss 开打那一步的价值估计 v；
    # 抽题时按 v 在池里的排序分位查 p̂ 的分布（curriculum.PHAT_QUANTILES，第 122,056 版可学性测量拟合），权重 ∝ [p̂(1−p̂)]^boss_pq_power，
    # 分位低于 boss_pq_drop 的不抽，另混 boss_pq_mix 的等概率；不用固定的“估值 → p”换算，价值头整体偏移不影响。
    # 打开时连打不再按输率加权（第二个 Boss 单打不变）。练习结束把 (v, 输赢) 记进 run_dir/boss_pq.jsonl，供重新校准
    boss_pq_weight: bool = False
    boss_pq_power: float = 3.0
    boss_pq_drop: float = 0.15
    boss_pq_mix: float = 0.1
    # 第三幕双 Boss 专项（决策记录 287）：题目文件（jsonl，相对路径按运行目录），每行 {"id", "encounter", "next", "snap", "w"}——
    # 第三幕第一个 Boss 战前的玩家快照与这一局的两个 Boss，w 为抽题权重（缺省 1）。给了就让所有战斗课程的引擎只打这些题的连打
    # （同 boss_chain_share 的连打：打赢第一个带着血量、药水接着打第二个，按两场结算），每场换新种子；文件改了（按修改时间）自动重读。
    # 空为不用（主线）
    chain_problems: str = ""
    # 专项训练的日程（决策记录 287，版本号为绝对值，专项从第 0 版起；默认全关，主线不变）：
    # weight_decay：解耦权重衰减（AdamW 式，每次更新后 p ← p·(1 − 学习率 × wd)），只作用于二维及以上的参数（不含偏置、LayerNorm）；
    # lr_decay_until / lr_final_frac：学习率从第 0 版的 lr 线性降到这一版的 lr × lr_final_frac，之后保持；
    # combat_anchor_until：锚定 KL 的系数从 combat_anchor_coef 线性降到这一版的 0，之后不再算老师；
    # value_warmup_until：这一版之前只更新价值一侧的参数（价值头、特权编码、整局预生成内容、辅助头、打分头），主干与策略头不动
    weight_decay: float = 0.0
    lr_decay_until: int | None = None
    lr_final_frac: float = 1.0
    combat_anchor_until: int | None = None
    value_warmup_until: int = 0
    # 胜率势能塑形（2026-10-29，决策记录 189）：胜率小模型（winprob_online，结构同直播侧栏的实时胜率模型）在学习进程里与主模型一起在线训练，
    # 整局每一步的奖励加 Φ(下一步) − Φ(这一步)，Φ = 系数 × 预测通关概率，对局结束时 Φ = 0（基于势能的塑形，不改变最优策略）。
    # wp_shaping 为目标系数（0 为关闭，不训练也不塑形）；在线训练满 wp_warmup 次更新后用 wp_ramp 次更新从 0 线性升到目标
    wp_shaping: float = 0.0
    wp_init: str = ""               # 初始权重（离线训好的胜率模型）；检查点里已有在线训练的模型时用检查点的
    wp_warmup: int = 100
    wp_ramp: int = 400
    wp_lr: float = 3e-4
    wp_iters: int = 1               # 每次主模型更新训练胜率模型几步（升系数仍按更新次数算，见 wp_warmup / wp_ramp）
    wp_batch: int = 4096
    wp_buffer: int = 300000         # 学习进程保留最近多少个样本（每局每层最多 wp_per_floor 步，权重 1 / 留下的步数）
    wp_per_floor: int = 3
    # 进度势能（2026-10-29，决策记录 191）：旧的爬层与中途 Boss 奖励改写成势能——Φ = prog_floor × 层数 + prog_boss[0、1、2] ×（打过第一幕
    # Boss、第二幕 Boss、第三幕第一个 Boss），每一步加 Φ(下一步) − Φ(这一步)，对局结束（输赢都算）时 Φ = 0。活着时与旧奖励逐步相同，
    # 死时把攒下的进度分一次扣回，每局合计只剩通关奖励减开局的势能，不改变最优策略。与胜率势能（wp_shaping）分开计、可同时用
    prog_floor: float = 0.0
    prog_boss: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # 构筑价值势能（2026-11-03，决策记录 252，情况二）：冻结的构筑价值模型（离线训好，tools/buildvalue.py、sts2rl/buildvalue_model.py）
    # 在整局的每个战斗外决策估计这一局最后通关的概率 P，Φ = 系数 × P；Φ(这一个) − Φ(上一个) 记在上一步，对局结束时 Φ = 0。
    # 每局合计 = −Φ(开局)，不改变最优策略，只把“这次选择让构筑变好还是变坏”的信号提前到当场。战斗中不更新
    # （战斗里的变化多半是这场仗的运气；第 189 条的胜率势能每回合都更新，噪声大）。
    # 系数按权重版本从 bv_start 起用 bv_ramp 版线性升到 bv_shaping；bv_start 为 None 时直接用满额。bv_shaping 为 0 时关闭
    bv_shaping: float = 0.0
    bv_model: str = ""
    bv_start: int | None = None
    bv_ramp: int = 300
    solver_same_seed: float = 0.5   # 其中用求解器解过的那个种子的比例（其余换新种子）
    solver_export_every_s: list = field(default_factory=lambda: [1800.0, 120.0, 120.0])   # 每个采集进程导出各幕 Boss 快照的最短间隔（秒）；第三幕起于 2026-10-26，第一幕 600 → 1800 同日
    solver_pending_max: int = 400   # 待标注的快照超过这么多就暂停导出（本机标注跟不上时）
    solver_pool: int = 600          # 每个采集进程每幕保留最近多少个“能赢”的局面
    # 非战斗决策（整局里战斗之外的选路线、抓牌、事件、商店、休息处）：优势只沿这些决策的序列计算，
    # 中间的战斗步的奖励并入下一段。原来与战斗步一起按 λ=0.95 逐步衰减，一场战斗几十步，抓牌几乎只看得到下一场战斗。
    macro_lam: float = 0.95         # 非战斗决策序列上的 λ（约 20 个决策 ≈ 5 层）
    # 局外决策的优势跨段计算（2026-11-03，决策记录 255；用户：“局外决策的优势跨段计算，不在段边界截断”）：整局分段提交（run_segment_steps）时，
    # 原来每段末尾用下一段第一个决策的价值自举，局外决策链每约 4 层就被截断一次。开了以后学习端先把每段的局外决策存起来（按采集端的局编号 gid），
    # 等这一局最后一段到了再沿整条局外决策链算 λ 优势，放进当批训练；战斗步照旧按段及时训练。局外步因此要等整局打完才入训（一局约 7 版），
    # 过旧判断单独用 macro_defer_lag；pending 超过 macro_defer_ttl 秒还没等到最后一段的整局丢掉（采集端重启、连接断开等）。关时与原来完全相同
    macro_cross_segment: bool = False
    macro_defer_lag: int = 12
    macro_defer_ttl: float = 1800.0
    # 只训局外（决策记录 267，默认关）：combat_greedy 时整局里的战斗一律取概率最大的动作（局外照常采样）；macro_only_pi 时策略损失与熵奖励只算局外步，
    # 战斗步只按 combat_value_frac 抽样训价值头与辅助头；combat_anchor_coef > 0 时战斗步另加 KL(老师 ‖ 当前)，老师为 combat_anchor_ckpt 里冻结的权重
    # （上线那一刻的网络），让共享主干被局外梯度改动时战斗输出不漂。
    combat_greedy: bool = False
    macro_only_pi: bool = False
    combat_value_frac: float = 1.0
    combat_anchor_coef: float = 0.0
    combat_anchor_ckpt: str = ""
    # 搜索标签的 KL 辅助损失（2026-11-17 训练中用搜索的试点，sts2rl/search_kl.py；标签由 tools/future/search_stream.py 产出）：search_kl_coef > 0 时
    # 每个 PPO 小批另取 search_kl_batch 个搜索过的局面，损失加 search_kl_coef × KL(搜索改进后的分布 ‖ 当前策略)。标签从 search_kl_dir 下的 labels_*.jsonl 持续读入，
    # 最多留 search_kl_buffer 个；search_kl_max_age > 0 时只用最近这么多版的网络产的标签；缓冲少于 search_kl_min 个时不加。默认关，与原来完全相同
    search_kl_coef: float = 0.0
    search_kl_dir: str = ""
    search_kl_batch: int = 128
    search_kl_buffer: int = 60000
    search_kl_max_age: int = 0
    search_kl_min: int = 1024
    search_kl_compiled: bool = True     # 标签批的前向走学习端编译的 fwd（learner_compile 时）；False 为不编译的 net.forward_all
    # 采集端用搜索出手（2026-11-17，局外适应搜索战斗，sts2rl/actor_search.py）：search_collect_rooms 非空时（如 "Boss,Elite"），整局里这些房间的战斗照部署方式出手——
    # 首选概率 ≥ search_collect_tau 取最大，犹豫时做 search_collect_sims 次模拟的整局树搜索（λ search_collect_lam，每个采集进程另开 search_collect_procs 个推演引擎）；
    # 这些步记为 det，学习端不进策略梯度、回报照常（局外决策学的是战斗有搜索时的价值）。默认空，与原来完全相同。宜配 envs_per_actor 1（换局要从头重放）
    search_collect_rooms: str = ""
    search_collect_sims: int = 32
    search_collect_procs: int = 8
    search_collect_tau: float = 0.9
    search_collect_lam: float = 0.06
    search_collect_min_visits: int = 4      # 最终选法只看访问数不少于它的候选（8 次模拟时要 2，否则几乎没有合格的候选、搜索不改选）
    search_collect_dll: str = ""            # 搜索器的引擎程序集（空为部署目录里的；要带战后奖励确定化补丁 e0797a2 的）
    # 整局里这些房间的战斗按贪心确定性出手（不做搜索），同样记为 det（不进策略梯度、回报照常）：局外决策在接近部署强度的战斗下学
    # （2026-11-17 选路探针：训练时战斗采样出手比部署的贪心弱，局外偏回避精英）。与 search_collect_rooms 同时给出时，后者的房间做搜索。默认空，与原来相同
    det_greedy_rooms: str = ""
    # 同种子成组采集（决策记录 267，默认 1 为关）：每个采集进程把同一个种子连续发给 run_group 局整局；学习端（要 macro_cross_segment）等同组的局都打完，
    # 局外决策的优势减去同组其他局在同一局面（帧签名相同）上的优势的平均（同前缀留一基线：别的局的结果与本局这一步的动作无关，不引入偏差）；
    # 没有同局面同伴的步照用原来的优势。组超过 group_ttl 秒没等齐就按已到的局放行
    run_group: int = 1
    group_ttl: float = 900.0
    # 局外优势按段自举（决策记录 269，要 macro_cross_segment）：局外步照样等整局、按组放行，但优势按原来不跨段的办法算——每段末尾用下一段第一个决策的价值自举，
    # 与第 267 条之前的局外优势相同（本机检验只差 float32 舍入，约 1e-7）。第 267 条为了按组放行打开了跨段计算，局外回报一路累积到整局结束，局外价值均方误差从约 0.21 升到约 0.43
    macro_seg_adv: bool = False
    # 战斗阶段（决策记录 270，默认全关；与第 267 条的局外阶段对称）：macro_greedy 时整局里的局外决策一律取概率最大的动作（强制探索、温度改选也不做）；
    # combat_only_pi 时策略损失与熵奖励只算战斗步，局外步只训价值；combat_anchor_rows 选锚定 KL 拴哪一类步（"combat" 为第 267 条的拴战斗，"macro" 为拴局外，"all" 为全部步——决策记录 274 续训时拴在模型池蒸馏出的网络上，免得翻动吃掉集成收益）；
    # combat_fight_boot 时战斗步的优势在每场战斗打完处截断，最后一步用战后那一步的价值自举（战斗的目标统一成“整局胜率”，战后的局外决策不再混进来）
    macro_greedy: bool = False
    combat_only_pi: bool = False
    combat_anchor_rows: str = "combat"
    combat_fight_boot: bool = False
    # combat_fight_boot_map（愚者决策记录第 8 条，要 combat_fight_boot）：截断点从战后第一步挪到战后第一次选路（或下一场战斗开打），
    # 领奖、选牌界面的估值不再拿来给战斗自举（那里的局外估值局部偏低，会让“打完”吃亏、拖着不打完占便宜）。默认关，原来的截断不变
    combat_fight_boot_map: bool = False
    # combat_fight_center（同一条，要上面两项）：每批按幕把“战斗最后一步的优势”（算到战后第一次选路）的平均从这场战斗各步里减掉（按 (γλ)^步数 衰减），
    # 去掉两个价值头在战斗收尾处的系统落差带来的“早打完 / 晚打完”偏置；只减平均，不增加方差。日志 fight_end 记各幕 [场数, 去均值前的平均]
    combat_fight_center: bool = False
    # 免调度 Adam（Schedule-Free，Defazio 等 2024；python/sts2rl/schedulefree.py，2026-11-15）：Adam 那部分参数换成免调度更新，Muon 不变。
    # sf_lr 为它的学习率（None 时同 lr），sf_warmup 为预热步数（按梯度步算），平均权重 c_k = lr_max^sf_power × k^sf_r。检查点另存 opt_sf 与平均点 sf_x
    sf_adam: bool = False
    sf_lr: float | None = None
    sf_warmup: int = 640
    sf_beta1: float = 0.9
    sf_beta2: float = 0.999
    sf_power: float = 2.0
    sf_r: float = 0.0
    # Muon 也做免调度平均（愚者决策记录第 16 条）：β 用 sf_beta1，预热、平均权重同上；Muon 自己的动量照留
    sf_muon: bool = False
    # 从当前平均点重新开始免调度平均（愚者决策记录第 19 条）：填一个标记字符串。续训时检查点里记的标记（sf_restart）与它不同，
    # 就把参数（y）与 z 都换成平均点 x、累计权重清零，之后 x 只平均这以后的 z（步数照留、不再预热，二阶矩与 Muon 动量照留）；
    # 做过之后标记存进检查点，守护重启不会再做。空串为不做。用途：改学习率、梯度累积等之后，评测的 x 只反映改后的轨迹
    sf_restart: str = ""
    # 里程碑奖励逐步撤掉（决策记录 271）：milestone_from 那一版起用 milestone_ramp 版把层数奖励、第一二幕 Boss、第三幕第一个 Boss 的奖励线性降到 0
    # （按每局开局时的版本算，同一局的系数不变）；撤掉的部分在通关时一次补回（按这一局原本能拿到的里程碑合计），所以通关局的回报不变，
    # 只有输局的回报降低，训练目标逐步变成“只看通关”。None 为不撤（原来的奖励）
    milestone_from: int | None = None
    milestone_ramp: int = 500
    # 分类价值头（HL-Gauss，决策记录 261；要 model.value_hl）：交叉熵按 vf_coef × hl_coef 计入；用的价值从数值头逐步换成分类头的期望（hl_beta 0 → 1），
    # 数值头的均方误差按 (1 − hl_beta) 计入。换用有门槛：hl_start 版起，分类头与数值头均方误差之比（每版一次，指数平均 0.1）不超过 hl_gate 的那一版，
    # hl_beta 加 1 / hl_ramp；超过门槛就停在原值（不回退）。hl_beta 存在检查点里，重启接着走。hl_start 为 None 时一直用数值头（只训练分类头）。
    # 软标签的高斯宽度为 hl_sigma × 桶宽
    hl_coef: float = 0.5
    hl_start: int | None = None
    hl_ramp: int = 300
    hl_gate: float = 1.05
    hl_sigma: float = 0.75
    # 分类头交叉熵重新传回主干时的慢开（决策记录 261 补记四）：hl_coef_from 那一版起用 hl_coef_ramp 版把系数从 0 升到 hl_coef；None 为一直用 hl_coef。
    # hl_scalar_full：数值头的均方误差一直按全权重训练主干（不随 β 降），交叉熵只作辅助
    hl_coef_from: int | None = None
    hl_coef_ramp: int = 300
    hl_scalar_full: bool = False
    # 战斗短程辅助头（决策记录 261；要 model.combat_aux）：aux_coef × (本场输赢的交叉熵 + aux_hp_w × 剩余掉血的均方误差)，只在有标签的战斗步上
    aux_coef: float = 0.25
    aux_hp_w: float = 4.0
    # 分类价值头、战斗辅助头的参数各自单独裁剪梯度，不算进全局范数（决策记录 261 补记二）
    head_own_clip: bool = False
    # 候选条件化的未来读数头（愚者决策记录第 25 条；要 model.future_head）：fh_coef × 读数均方误差（各维按滑动标准差归一，只在有标签的战斗步、
    # 真实所选的候选上）。读数为走这一步之后 fh_h 步（战斗内；战斗先结束则到结束为止）的 [fh_h 步优势（奖励之和 + 末端价值 − 这一步价值）、
    # 血量变化 / 最大生命、敌人剩余生命比例变化、其间打赢、其间输掉]，见 actor.future_labels。头（fh_pred、fh_res）用单独的 Adam（fh_lr）、
    # 单独裁剪，不进免调度 Adam 与 Muon；fh_res（读数 → 候选打分残差）只由策略损失训练
    fh_coef: float = 0.0
    fh_h: int = 12
    fh_lr: float = 1e-4
    # 跳过稳赢战斗（reports/跳过稳赢战斗方案.md，猎人决策记录第 14 条；默认全关，行为与原来完全相同）：
    # co_heads 时学习端另训结果头（model.CombatOutcomeHeads：赢下后的战后血量分布、每瓶药水用掉的概率），读主干的 detach 副本、单独的 Adam（co_lr），
    # 随检查点保存、随权重文件发给采集进程；co_init 为初值（tools/combat_model.py 训好的冻结主干模型，只取它的 hp、pot 两个头），检查点里已有结果头时不用。
    # skip_easy 时采集端在整局里选路后直接开打的 skip_rooms 战斗的第一帧，辅助头预测胜率不低于 skip_p、遭遇不在 skip_blacklist（按前缀）、
    # 这一局不是“完全真打”的局（开局时以 skip_real_runs 的概率指定）时，以 skip_prob 的概率不打：按结果头抽战后血量（加上按（幕，房间）
    # 滚动估计的校准偏差，速率 skip_hp_ema）、按药水头逐瓶抽签，发引擎的 resolve 请求直接结束；这一帧不进轨迹，整局分段的步数计数补上这类战斗的平均步数。
    # 身上有药水的战斗从 skip_pot_from 版起才跳（None 为不限）；skip_from 为从第几版起跳（None 为不限）
    # 机会事件的运气校正（AIVAT 式控制变量；铁甲战士决策记录第 20 条，默认关，行为与原来完全相同）：
    # 采集端在 luck_cv_rooms 的战斗里选了“结束回合”时，先在引擎里存下局面（槽位 luck_cv_slot），按公开信息换 luck_cv_k 次随机数
    # （reseed，保留已知的抽牌堆顶）各走一遍这个动作，用同一个网络估出“平均运气下”的下一步价值 V̄，再恢复局面走真实的那一步。
    # 学习端算 GAE 时这一步的 δ 用 V̄ 代替真实下一步的价值 V(s')：期望不变，去掉的是敌方回合与抽牌的运气 γ(V(s') − V̄)。
    # 真实那一步或任一推演打完了这场战斗（局结束、离开战斗）时不校正（这类结局在给定局面与动作时几乎是确定的，见第 20 条）。
    # luck_cv_rooms：逗号分隔的房间类型，"boss" 只在 Boss 战（整局的 Boss 房间、练习的 Boss 名额与连打），"boss,elite" 加上精英战，"all" 为全部战斗；
    # 日志每版记 luck（步数、运气大小、δ 校正前后）
    luck_cv: bool = False
    luck_cv_k: int = 4
    luck_cv_rooms: str = "boss"
    luck_cv_slot: int = 7
    luck_cv_apply: bool = True      # False 时只量不用：采集端照样推演、学习端日志照样记运气有多大，优势与价值目标不改
    co_heads: bool = False
    co_lr: float = 1e-3
    co_init: str = ""
    skip_easy: bool = False
    skip_p: float = 0.995
    skip_prob: float = 0.8
    skip_real_runs: float = 0.25
    skip_rooms: list = field(default_factory=lambda: ["Monster", "Elite"])
    skip_blacklist: list = field(default_factory=lambda: ["SCROLLS_OF_BITING", "THIEVING_HOPPER", "GREMLIN_MERC"])
    skip_pot_from: int | None = None
    skip_from: int | None = None
    skip_hp_ema: float = 0.02
    # 阶段式策略梯度（PPG，Cobbe 等 2021；决策记录 266，默认关）：ppg_every > 0 时每版从入训步里抽 ppg_keep 存下（局面、回报、辅助头标签），
    # 每 ppg_every 版做一次辅助阶段：先固定分块记下当时的策略概率，再训 ppg_epochs 遍“价值损失（同 PPO 里的数值头、分类头、辅助头）
    # + ppg_beta_clone × KL(当时的策略 ‖ 现在的策略)”，让价值多训几遍而策略输出不动；ppg_pi_vf 为策略阶段 PPO 损失里价值项的倍数（1 为不变）
    ppg_every: int = 0
    ppg_keep: float = 0.25
    ppg_epochs: int = 2
    ppg_beta_clone: float = 1.0
    ppg_pi_vf: float = 1.0
    # 梯度分解（诊断用，决策记录 261 补记）：打开时第一次更新只算各项损失对主干的梯度（grad_decomp_mb 个小批），写 run_dir/grad_decomp.json 后退出，不更新参数
    grad_decomp: bool = False
    grad_decomp_mb: int = 8
    # 策略损失里非战斗决策的权重（优势与战斗步分开标准化）。2.0 → 1.0（第 9,400 版回退后起）：
    # 局外决策有了自己的策略头和价值头，不再需要靠加权抢梯度；加权放大了噪声大的局外优势，是跑偏的嫌疑之一
    macro_pi_coef: float = 1.0
    # 牌组打分：网络的一个输出头，从局面视图（牌组、遗物、药水、血量）预测打这一幕 Boss 的成绩，
    # 标签来自真实快照的 Boss 练习：打赢为 1，打输为 phi_dmg_weight × 打掉的 Boss 血量比例。
    # 整局里同一幕内相邻两个非战斗决策之间打分的变化 × deck_shaping 记为奖励，记在前一个决策上（抓了好牌、删了坏牌、休息回血都直接得分）
    # 1.0 → 0（第 9,400 版回退后起）：标签来自当前策略自己打 Boss 练习的成绩，策略一变标准就跟着漂，
    # 可能形成“越练越差”的循环；局外决策的价值头本身就在估计同一件事。打分头仍按标签训练，只用于看板上的校准诊断
    deck_shaping: float = 0.0
    phi_dmg_weight: float = 0.6
    phi_buffer: int = 20000         # 学习进程保留最近多少条打分标签
    phi_batch: int = 512            # 每次更新用多少条标签训练一步打分头
    phi_coef: float = 1.0           # 打分损失（二元交叉熵）的权重
    # 打分头改为独立诊断器（决策记录 179，用户 2026-10-29 排为下一项单项改动）：编码器不回传打分损失的梯度，
    # 打分头用自己的 Adam；主干参数与主优化器状态不受它影响。False 为原来的共享梯度做法
    phi_detach: bool = False
    phi_warmup: int = 150           # 打分头训练满这么多步之后才开始给奖励
    phi_holdout: bool = False       # 牌组打分的区分能力诊断（新标签进训练前先打分，记 phi_fresh）；2026-10-26 用户决定先关掉，之后再做
    # 搜索蒸馏（2026-10-27 起，见 distill.py）：A 上的 Boss 战确定化搜索产出标签，每次 PPO 更新之后抽一批训练一步策略。
    # distill_on 打开收标签与一致率诊断；distill_coef 为 0 时只诊断、不训练
    distill_on: bool = False
    distill_coef: float = 1.0       # 加权交叉熵的权重（单独一步，与牌组打分头同样的做法）
    distill_batch: int = 128        # 每次更新抽多少条
    distill_same_w: float = 0.2     # 没改选的标签的权重（改选了的为 1）
    distill_max_age: int = 2000     # 标签比当前版本旧超过这么多版就丢掉（产出标签的检查点版本计）
    distill_cap: int = 60000        # 缓冲最多保留多少条（每条在内存里约 9 KB）
    distill_poll_s: float = 30.0    # 每隔多少秒收一次新标签
    # 其他
    total_steps: int = 10_000_000
    save_every: int = 20            # 每隔多少次更新存检查点
    log_every: int = 1
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)

    @staticmethod
    def from_args(argv: list[str] | None = None) -> "TrainConfig":
        cfg = TrainConfig()
        p = argparse.ArgumentParser(description="战斗课程 + 整局 PPO（角色见 --character）")
        p.add_argument("--config", help="JSON 配置文件（命令行参数优先）")
        for f in dataclasses.fields(TrainConfig):
            if f.name == "model":
                continue
            p.add_argument("--" + f.name.replace("_", "-"), type=type(getattr(cfg, f.name)), default=None)
        for f in dataclasses.fields(ModelConfig):
            p.add_argument("--model-" + f.name.replace("_", "-"), type=type(getattr(cfg.model, f.name)), default=None)
        a = p.parse_args(argv)
        if a.config:
            raw = json.load(open(a.config, encoding="utf-8"))
            for k, v in raw.items():
                if k == "model":
                    for mk, mv in v.items():
                        setattr(cfg.model, mk, mv)
                else:
                    setattr(cfg, k, v)
        for k, v in vars(a).items():
            if v is None or k == "config":
                continue
            if k.startswith("model_"):
                setattr(cfg.model, k[len("model_"):], v)
            else:
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def cut_flags(self, evaluating: bool = False) -> dict:
        """reset 请求里的两个截断开关（局面重复截断、跨回合无进展判输）；evaluating 时判输线用 eval_stall_turns（没给时同 stall_turns）。
        都关时为空，请求与原来完全相同。"""
        st = self.eval_stall_turns if evaluating and self.eval_stall_turns is not None else self.stall_turns
        return {**({"loop_cut": True} if self.loop_cut else {}), **({"stall_turns": int(st)} if st and st > 0 else {})}
