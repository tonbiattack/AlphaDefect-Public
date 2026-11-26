"""训练看板：从训练运行目录生成一个独立的 HTML，离线可看，按本机 1920×1080 屏幕排版。

首屏回答“现在什么水平、有没有在进步”：四个关键指标（平均层数、打过第一幕 Boss、打过第二幕 Boss、通关率，
各带近 12 次合计的变化和迷你趋势线）、通关之路（层数分布 + 按幕分段的进度条）、一行里程碑摘要；双机时顶部每台主机一张训练线小卡，
首屏下方是对照实验的摘要。往下是进展趋势；训练诊断、卡点诊断、评测明细是可折叠的分区（手机上默认收起，开合记在本机）。

    python3 tools/dashboard.py                 # 生成一次
    python3 tools/dashboard.py --loop          # 常驻：输入变了就重新生成，至少每 60 秒一次；代码更新后自动重启
    python3 tools/dashboard.py --hosts H.json --lines L.json --remote DIR --out X.html --public DIR   # 测试：换数据来源与输出

多机：主机登记 tools/remote/hosts.json、训练线登记 ~/.local/share/sls2/runs/remote/lines.json（规则见“多机与训练线”一节）。
主线的数据画在原有各面板上（迁移前后接起来）；有对照线时首屏下方加“对照实验”，按分叉后的版本数对齐比较评测。
两个文件都不存在时与单机时完全一样：只读主机 a 的目录 defect。

每台主机的数据目录（~/.local/share/sls2/runs/remote/<local>，由 tools/remote/pull.sh 从租用机拉回；a 为 defect）：
    log.jsonl      每次 PPO 更新一行（训练指标）
    eval.jsonl     固定种子评测（贪心）：整局层数分布、各幕到达率、通关率、战斗课程胜率
    resource.jsonl 机器资源（30 秒一次）
    status.json    训练进程状态（每次更新写一次）
    watchdog.json  租用机守护的状态
    eval_progress.json  评测进行中的进度
    ../dashboard_note.txt  当前配置说明（可选）
写（Windows 侧项目目录）：
    训练看板.html        完整页面（首次打开用）
    训练看板.data.js     页面内容；打开着的页面发现更新后原地替换，不整页重载
    训练看板.stamp.js    版本戳，页面每 2 秒查一次
同样的三个文件另写一份到 ~/.local/share/sls2/runs/remote/public（index.html、data.js、stamp.js），
由 tools/remote/publish.sh 推到香港服务器，公网看板 https://dashboard.example 。
只用标准库。
"""
from __future__ import annotations

import argparse
import base64
import bisect
import copy
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
import time
from value_probe_cards import value_probe_section
from pathlib import Path

HOME = Path.home()
REMOTE = HOME / ".local/share/sls2/runs/remote"          # 各主机的拉取目录、拉取状态、lines.json 都在这里
HOSTS_FILE = Path(__file__).resolve().parent / "remote/hosts.json"
LINES_FILE = REMOTE / "lines.json"
NOTE_NAME = "dashboard_note.txt"                          # 在 REMOTE 下
OUT = Path("/mnt/c/主存储文件/脚本开发/sls2/训练看板.html")
VALUE_PROBE_ROOT = OUT.parent / "runs/value_probe"
# 常规评测改到协同采集机 A 上跑（tools/remote/remote_eval.py，决策记录 198）的时刻：主线的评测记录这之前读主线主机的 eval.jsonl，
# 之后读本机拉取的 A 的评测目录；评测进度、评测报错也改读那里。种子、局数（192）与打法不变，前后可比
EVAL_DIR = REMOTE / "defect-eval"
# 评测口径的新起点（决策记录 198、199）：10-30 06:57（B 时间）起评测改为 384 个按难度分层的 RT 种子，与之前的 EVAL 种子不可比。
# 看板上评测相关的图表、表格、首屏数字只用这之后的评测；之前的评测留在各机的 eval.jsonl 里存档，不再显示
EVAL_SINCE_T = 1793314636
EVAL_MOVE_T: float | None = 1793312727.0      # B 10-30 06:25 重启、停掉本机评测的时刻
STAMP = "训练看板.stamp.js"
DATA = "训练看板.data.js"
DATA2 = "训练看板.data2.js"     # 对调视角的内容脚本（公开副本里叫 data2.js）；页面只下载当前视角的那一份
# 公开副本（ASCII 文件名）：tools/remote/publish.sh 把它推到香港服务器，由 https://dashboard.example 分发
PUBLIC = HOME / ".local/share/sls2/runs/remote/public"
# 单人 A10 常规 49 层：第一幕 17 层（先古 + 15 个房间 + Boss）、第二幕 16、第三幕 15 + 第二个 Boss。
# 第二幕的地图偶尔多两行（两排宝箱），总层数最多 51。通关之路按幕分开排：每幕常规层数 + 预留的加长层，
# 加长层排在本幕 Boss 之前、用浅色标出，Boss 总在各幕末尾（评测的 pos 记录本幕第几层，见 evaluator.run_pos）
FULL_FLOORS = 49
MAX_FLOOR = 51
ACTS = [(1, 17, "第一幕"), (18, 33, "第二幕"), (34, 49, "第三幕")]
# 通关之路的排布：(幕名, 常规层数（含先古和 Boss）, 加长层数, Boss 数, Boss 标签)
LAYOUT = [("第一幕", 17, 0, 1, "第一幕 Boss"), ("第二幕", 16, 2, 1, "第二幕 Boss"), ("第三幕", 16, 0, 2, "终局")]
N_SLOTS = sum(n + e for _, n, e, _, _ in LAYOUT)


def slot_of(act: int, k: int, boss: int) -> int:
    """（幕, 本幕第几层, 第几个 Boss）→ 通关之路上的格子（1 起）：Boss 排在本幕最后，其余按本幕层数，超出的落在加长层里。"""
    act = max(0, min(act, len(LAYOUT) - 1))
    base = sum(n + e for _, n, e, _, _ in LAYOUT[:act])
    _, n, e, nb, _ = LAYOUT[act]
    if boss:
        return base + n + e - nb + min(boss, nb)
    return base + max(1, min(k, n + e - nb))


def legacy_pos(f: int) -> tuple[int, int, int]:
    """没有 pos 的旧评测：总层数按常规层数换算成（幕, 本幕第几层, 第几个 Boss）。超过常规的 49 层说明第二幕加长了，先减掉加长的两层。"""
    if f > FULL_FLOORS:
        f -= 2
    for a, (_, n, _, nb, _) in enumerate(LAYOUT):
        if f <= n or a == len(LAYOUT) - 1:
            return a, f, max(0, min(f, n) - (n - nb))
        f -= n
    return 0, f, 0


def floor_slot(v: float) -> float:
    """连续的总层数（平均、前 10% 等）→ 通关之路上的位置：按常规层数分段线性换算，第二幕 Boss 之前跨过加长层。"""
    pts, F, S = [(0.0, 0.0)], 0, 0
    for _, n, e, nb, _ in LAYOUT:
        pts += [(F + n - nb, S + n - nb), (F + n, S + n + e)]
        F, S = F + n, S + n + e
    v = min(max(v, 0.0), pts[-1][0])
    for (f0, s0), (f1, s1) in zip(pts, pts[1:]):
        if v <= f1:
            return s0 + (s1 - s0) * (v - f0) / max(f1 - f0, 1e-9)
    return float(S)


def skull(x: float, y: float) -> str:
    """迷你骷髅头（约 10×11），标在 Boss 所在格的上方。"""
    return (f'<g class="skull" transform="translate({x:.1f},{y:.1f})"><path d="M-5,0.5A5,5 0 1 1 5,0.5V3H3V5.5H-3V3H-5Z"/>'
            f'<circle class="sk-e" cx="-2" cy="0.4" r="1.45"/><circle class="sk-e" cx="2" cy="0.4" r="1.45"/>'
            f'<path class="sk-t" d="M-1,3.4V5.5M1,3.4V5.5"/></g>')
# 评测口径分界：引擎修正后同一种子的商店、可选动作都变了，前后的评测成绩不能直接比较。（时间戳, 说明）
EVAL_BREAKS = [(1792878510, "引擎修正"), (1792930182, "升级到 v0.111.0"), (1793313770, "评测 192 → 384 局（多出的 EVAL192–383 更难）"), (1793314636, "评测种子改为 384 个按难度分层的 RT 种子"), (1793570879, "评测与复测改用随机种子池 P1")]
# 训练方案的重要改动：评测口径不变，前后可以直接比较，只在图上标线、在评测表里插一行，便于对照改动前后。（时间戳, 说明）
CHANGES = [(1792918403, "真实局面练习"), (1792920028, "构筑训练"), (1792926893, "回退到第 9,400 版·局外双头"), (1792927442, "局外探索·金币自动领"), (1792933190, "求解器筛选 Boss 练习"), (1792945898, "药水限制·牌组塑形"), (1792946295, "Boss 战用药水不扣分"), (1792948665, "血量边际价值·升级 0.08"), (1792951087, "战损·药水·遗物·金卡"), (1792952338, "本幕 Boss 条件输入"), (1792962695, "真实局面扩到三幕"), (1792964229, "升级 0.08 → 0.04"), (1792980540, "第三幕 Boss 练习·输局部分分"), (1793000833, "水晶球棋盘进观测"), (1793001577, "地图连线·选牌说明"), (1793001931, "事件数值·卡包·栏满药水·任务标记"), (1793002387, "求解器扩到第三幕"), (1793003463, "第三幕第二个 Boss 练习"), (1793004385, "一幕练习减半·去金卡·通关 2.0"), (1793005678, "局外熵 0.05 → 0.03"), (1793016082, "第三幕 Boss 奖励 0.5·考卷与标定"), (1793017532, "引擎：放弃战斗后开不起来"), (1793018800, "引擎：毛皮大衣练习开不了战"), (1793020204, "队列限容·采集降优先级"), (1793028713, "合入中途开局 60:25:15"), (1793036452, "合入挪练习份额·升级 0.02·考卷 250 版"), (1793040780, "采纳分叉：学习率 1.5e-4·第三幕奖励 1.0·遗物 0.03"), (1793046814, "第三幕 Boss 全用真实快照·第二个 Boss 35%"), (1793052067, "路线摘要·精英前火堆"), (1793052932, "选路熵系数 0.05（750 版）"), (1793057680, "B 扩容转为主线"), (1793058393, "精英战用药不扣分"), (1793071575, "第三幕 Boss 练习份额 0.09 → 0.21"), (1793088455, "休息处数值·满血不休息"), (1793099576, "考卷第 2 版"), (1793101741, "观测 9·内容先验·去掉血扣分·升级 0.01·遗物 0.02·二 Boss +0.5"), (1793109932, "观测 10·回合结束预演"), (1793110587, "牌堆张数"), (1793121721, "搜索蒸馏"), (1793122914, "Boss 练习：一幕 0.06·三幕 0.67·二 Boss 55%"), (1793125299, "停求解器练习·关蒸馏"), (1793129901, "学习端提速：打包注意力·编译"), (1793148034, "双机协同：A 远程采集"), (1793148673, "B 本机采集 6 → 3"), (1793150180, "B 本机采集 3 → 4"), (1793183623, "关用药扣分·升级·遗物奖励"), (1793184349, "学习端去同步"), (1793185239, "组批 2 线程"), (1793192620, "回退到第 59,098 版·观测 11 与价值头特权·无负面遗物自动拿·本机采集 8"), (1793195002, "观测 12·回合结束预演补能力与遗物·计算型卡牌数值·Boss 招式力量与格挡"), (1793201864, "关整局血量塑形"), (1793205382, "局外强制探索·局外熵 0.05"), (1793210775, "入训统计（不改训练）"), (1793215122, "整局分段提交 64 步"), (1793228846, "NaN：回退到第 67,920 版·非有限梯度跳过"), (1793232408, "关牌组超限惩罚"), (1793265644, "过旧阈值 3 → 6 版"), (1793267080, "队列 1.5 → 0.75 批"), (1793269311, "附魔效果进内容先验"), (1793273310, "观测 13·汇总前小网络·打分头独立"), (1793274566, "Boss 部分分：多形态·召唤物"), (1793276851, "整局只留通关奖励·选路熵 0.1·预演修正·第二个 Boss 单独嵌入"), (1793277860, "局外熵 0.05 → 0.08·选路跟随"), (1793279369, "观测 14：遗物状态·选牌来源·遭遇历史·局外熵 0.05"), (1793279744, "回退到第 80,200 版（设置不变）"), (1793280738, "第三幕 Boss 连打练习·按输率加权·二 Boss 0.55 → 0.3"), (1793282557, "战斗熵 0.01 → 0.006"), (1793285951, "胜率势能塑形（系数升到 3.5）"), (1793288049, "再次回退到第 80,200 版（设置不变）"), (1793288541, "进度势能·回退到第 80,200 版"), (1793291462, "恢复旧奖励·战斗熵 0.01·关势能·回退到第 80,200 版"), (1793299204, "观测 15：连打对齐·疯狂科学·缆线·占卜·出怪池·掉血修正·二 Boss 2.0"), (1793300341, "费用有效期·能力已触发·副本标记"), (1793307612, "版本差非负·best 按通关率·落后拆分统计"), (1793311286, "改选纳入行为分布 μ·本机队列 64 条·停旧考卷与打分头"), (1793313658, "评测移到 A·B 每进程 36 个引擎、A 8 个采集进程"), (1793388449, "局外学习诊断（不改训练）"), (1793389354, "整局 75%·停第一二幕真实局面与随机构筑普通战"), (1793395646, "每批 2 遍·学习率 2.25e-4·事件与先古之民温度改选"), (1793403072, "观测 16·贪心合并等价动作·快照只收紧接的战斗"), (1793409870, "评测取消合并贪心"), (1793437179, "观测 17·删牌次数与价格·能力先后·回手资格·迭代"), (1793440801, "观测 18·全部卡牌奖励可见·问号房实际概率·木桥候选池·下回合抽牌"), (1793468563, "学习率 1.1e-4·每批 65,536 步"), (1793472498, "评测与复测改用滑动平均权重（这之后评测偏高，不是模型突然变强）"), (1793477518, "梯度累积 ×2（有效小批量 8,192）"), (1793487407, "观测 19·候选活过本回合·买不起的商品·事件数值·学习潜力选题"), (1793527866, "学习率 1.65e-4"), (1793532862, "回退到第 136,503 版·学习率 1.1e-4"), (1793533041, "梯度累积 ×4·滑动平均 0.999"), (1793533232, "关局外强制探索"), (1793533946, "滑动平均重新起算（之后头几百版评测偏噪）"), (1793547047, "梯度裁剪阈值 1.0 → 2.0"), (1793554096, "Boss 战定向探索：碎片整理·电容器·回声形态"), (1793603476, "学习率 1.1e-4 → 5.5e-5·恢复 A 的采集"), (1793710631, "停训诊断后恢复·构筑价值势能 2.0（300 版升满）")]
# 遭遇的中文名：直接读本机游戏安装里的本地化文件（不复制进仓库）；读不到时用遭遇 ID
ENC_LOC = HOME / ".local/share/sls2/game/v0.111.0/res/localization/zhs/encounters.json"
# 内容目录（遗物所属的先古之民、遗物说明里的数值）；set_game_version 换游戏版本时一起换
CATALOG = Path(__file__).resolve().parents[1] / "engine/catalog/v0.111.0/catalog.json"
RANGES = [("all", "全部", None), ("24h", "最近 24 小时", 86400), ("6h", "最近 6 小时", 6 * 3600)]
MAX_PTS = 300          # 每条线最多画的点数；更多时按横轴分桶取平均
VREF_LABELS = 0        # 曲线图上写出文字的改动标线条数：改动多、又集中在最近一段，写字总会压住曲线，所以默认不写，悬停时在提示框里看
MAX_MARKERS = 40       # 评测点超过这个数就不画圆点

# 里程碑，未达成的按这里的顺序排，已达成的按达成先后排在前面：（名称，是否达成，当前值，目标）
MILESTONES = [
    ("平均到达第 10 层", lambda e: e["floor_mean"] >= 10, lambda e: f'{e["floor_mean"]:.1f} 层', "10 层"),
    ("首次打过第一幕 Boss", lambda e: e["reach_act2"] > 0, lambda e: f'{e["reach_act2"]:.0%}', "> 0%"),
    ("平均到达第一幕 Boss", lambda e: e["floor_mean"] >= 17, lambda e: f'{e["floor_mean"]:.1f} 层', "17 层"),
    ("过半打过第一幕 Boss", lambda e: e["reach_act2"] >= 0.5, lambda e: f'{e["reach_act2"]:.0%}', "50%"),
    ("首次打过第二幕 Boss", lambda e: e["reach_act3"] > 0, lambda e: f'{e["reach_act3"]:.0%}', "> 0%"),
    ("首次通关", lambda e: e["win_rate"] > 0, lambda e: f'{e["win_rate"]:.1%}', "> 0%"),
    ("平均到达第二幕 Boss", lambda e: e["floor_mean"] >= 33, lambda e: f'{e["floor_mean"]:.1f} 层', "33 层"),
    ("通关率 10%", lambda e: e["win_rate"] >= 0.10, lambda e: f'{e["win_rate"]:.1%}', "10%"),
    ("通关率 25%", lambda e: e["win_rate"] >= 0.25, lambda e: f'{e["win_rate"]:.1%}', "25%"),
    ("通关率 50%", lambda e: e["win_rate"] >= 0.50, lambda e: f'{e["win_rate"]:.1%}', "50%"),
]


# ---------------------------------------------------------------- 读数据

_jsonl_cache: dict[str, tuple[int, bytes, list[dict]]] = {}     # 路径 → (已解析的字节数, 那之前最后 4 KB, 记录)


def _loads_interned(line: bytes):
    """解析一行并让所有层级的键名共用同一份字符串（sys.intern）：主线日志十几万行、每行上百个键，逐行各存一份键名约占四成内存
    （11-02 实测每行 21.7 → 12.5 KB）。"""
    return json.loads(line, object_pairs_hook=lambda pairs: {sys.intern(k): v for k, v in pairs})


def _parse_lines(data: bytes, out: list[dict], post=None, loads=json.loads) -> int:
    """把完整的行解析进 out，返回解析到的字节数（最后一行没写完时不算它，下次再读）。post(记录) 在解析后就地精简记录。"""
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        line = line.strip()
        if line:
            try:
                r = loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if post is not None:
                post(r)
            out.append(r)
    return end


# 训练日志里看板不用的字段（入训、落后、改选一步差等诊断，离线脚本另读原文件），解析时就去掉；选牌计数（run_cards）看板只用最近
# 1,000 次更新，超过 LOG_CARDS_KEEP_S 秒的也去掉；先古遗物、死因、Boss 过关（run_ancient、run_deaths、run_bosses）只从“训练统计的起点”
# （stats_since，重新计数时上一段已留档）起累计，更早的也去掉。常驻进程按记录缓存，这些字段原来占了大半内存
# （2026-10-29 实测 573 MB 的日志解析后常驻约 3.9 GB，决策记录 197）
LOG_UNUSED = ("act1_monster", "act1_elite", "act2_monster", "act2_elite", "mix_monster", "explore_td", "run_fresh_kind", "explore_runs", "run_fresh_act", "run_lag_act", "lag_hist", "lag_split", "rank_lag", "run_boss_pots",
              "phi_calib", "phi_fresh", "solver_labels", "solver_pending", "boss_cert", "boss_probe_n", "phi_new",
              # 11-02 起（用户：减少看板内存）：局外学习诊断与战斗定向探索的统计，看板不用，每行约 1.3 KB
              "macro_diag", "combat_explore",
              # 11-02 再清一批：看板源码（含内嵌的网页脚本）里一处都没出现的字段
              "episodes", "dropped", "stale_steps", "return", "hp_delta", "ep_len", "run_steps", "run_floor_max", "run_reach_act2",
              "run_reach_act3", "run_return", "run_shaped", "run_hp_shaped", "run_pot_wasted", "run_deck_shaped", "run_dmg_shaped",
              "run_pot_cost", "run_skip_cost", "run_relic_shaped", "macro_frac", "run_segments", "run_elite", "shaping", "phi_steps",
              "phi_buf", "wp_coef", "wp_steps", "wp_rounds", "wp_loss", "wp_buf", "run_wp_shaped", "run_pg_shaped", "remote_trajs",
              "remote_steps", "remote_mb", "remote_bad", "remote_conns", "remote_idle_s", "progress", "lag", "t_collect", "t_learn",
              "t_prep", "t_ppo", "t_mb_wait", "loss_pi", "loss_v", "clip_frac", "loss_v_macro", "ent_coef_map", "nan_skipped")
# 课程分配（cur）只有最近几百次更新的明细表用全部项；旧记录只留走势图要的这几项（training_charts 的 cur3）
CUR_KEEP = ("boss3_chain", "boss3", "boss3_probe")
# 只用最近几百次更新的字段：超过 LOG_CARDS_KEEP_S 的旧记录里丢掉（选牌统计 run_cards 原来就这样处理）
LOG_RECENT_ONLY = ("run_cards", "macro_eff", "boss_deck")
LOG_CARDS_KEEP_S = 6 * 3600


def _slim_log_rec(r: dict, since_t: float) -> None:
    for k in LOG_UNUSED:
        r.pop(k, None)
    t = r.get("time", 0)
    if t < time.time() - LOG_CARDS_KEEP_S:
        for k in LOG_RECENT_ONLY:
            r.pop(k, None)
        c = r.get("cur")
        if c:
            r["cur"] = {k: c[k] for k in CUR_KEEP if k in c}
    fl = r.get("run_floors")
    if fl and isinstance(fl, list) and all(isinstance(f, int) and 0 <= f < 256 for f in fl):
        r["run_floors"] = bytes(fl)     # 逐局层数（0–255）存成字节串，用法只是取长度和逐个遍历
    if t < since_t:
        for k in ("run_ancient", "run_deaths", "run_bosses"):
            r.pop(k, None)


def read_jsonl(p: Path) -> list[dict]:
    """读 JSON Lines。常驻进程里按路径缓存已解析的部分：文件只是在末尾追加时（原来那段的最后 4 KB 没变），只解析新增的行；
    否则（被换成别的文件、截短）整份重读。拉取用 rsync 会换新文件，所以不靠 inode 判断。返回新列表，记录对象与缓存共用。"""
    key = str(p)
    post = (lambda r, t=stats_since()["t"]: _slim_log_rec(r, t)) if p.name == "log.jsonl" else None
    loads = _loads_interned if p.name == "log.jsonl" else json.loads
    try:
        with open(p, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            old = _jsonl_cache.get(key)
            if old and old[0] <= size:
                n, tail, recs = old
                f.seek(max(0, n - len(tail)))
                if f.read(len(tail)) == tail:
                    if n == size:
                        return list(recs)
                    data = f.read()
                    recs = list(recs)
                    n += _parse_lines(data, recs, post, loads)
                    f.seek(max(0, n - 4096)); _jsonl_cache[key] = (n, f.read(min(n, 4096)), recs)
                    return list(recs)
            f.seek(0)
            recs: list[dict] = []
            n, buf = 0, b""
            while chunk := f.read(64 << 20):     # 分块读，不把几百 MB 的整份文件和它的逐行拷贝同时放进内存
                buf += chunk
                k = _parse_lines(buf, recs, post, loads); n += k; buf = buf[k:]
            f.seek(max(0, n - 4096)); _jsonl_cache[key] = (n, f.read(min(n, 4096)), recs)
            return list(recs)
    except FileNotFoundError:
        _jsonl_cache.pop(key, None)
        return []


def rewind(recs: list[dict]) -> list[dict]:
    """从检查点续训时训练量会回退：丢掉被覆盖的那段记录，只保留最终那条时间线。"""
    out: list[dict] = []
    for r in recs:
        t = r.get("total_steps") or 0
        while out and (out[-1].get("total_steps") or 0) >= t and t > 0:
            out.pop()
        out.append(r)
    return out


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------- 多机与训练线
# 主机登记 tools/remote/hosts.json：{"hosts": {名: {"label", "local", "pending", ...}}}，本机拉取目录为 REMOTE/<local>。
# 训练线登记 REMOTE/lines.json：{"main": {"host", "since", "note"},
#                                "challenger": null 或 {"host", "branch_version", "branch_time", "main_host", "change", "note"}}。
# hosts.json 不存在时只有主机 a（目录 defect）；lines.json 不存在时主线在 a 上、没有对照。两者都没有就是单机时的老样子。
# 拉取状态：主机 a 沿用 REMOTE/pull.ok，其他主机为 REMOTE/pull-<主机名>.ok（tools/remote/pull.sh 写，内容为 unix 秒）。
#
# 主线拼接（主线换到新主机后，看板要能看到主线的完整历史）：
#   旧主机 = main.prev_host；没写时取对照所在的主机（迁移后旧主机改跑对照）；再没有、且主线已不在 a 上时取 a。
#   - log / eval / resource：旧主机只取 time < main.since 的记录，主线主机只取 time >= since 的记录，按时间先后接起来，
#     再用 rewind 按累计决策数去掉被覆盖的段。新主机从迁移时的检查点续训，版本号和 total_steps 接着记，所以两段正好首尾相接；
#     若新主机从更早的检查点续训，旧主机多出来的那段会被丢掉，和单机回退续训时一样。
#   - 旧主机在 since 之后的记录属于对照线（或迁移后滞后的评测），一律不进主线。
#   - 例外：新主机的文件里已有 since 之前的记录（migrate.sh 搬过去的、截到检查点的历史）时，直接用新主机的整份，不再拼旧主机的
#     （旧主机分叉成对照后运行目录换新，本机拉回的副本里已没有迁移前的历史）。resource.jsonl 不搬，仍按上面的规则拼。
#   - status / config / watchdog / eval_progress 只看主线主机；“已训练”时长从两台机器 status 里最早的 started_at 算起。
#   - 改动标线（CHANGES）自动加一条“迁移到<主线主机标签>”，时刻为 since。
# 对照线：对照主机上 time >= branch_time 的记录（同一台机器在 since 之前属于主线，也一并排除），再 rewind。
#   对照从主线第 branch_version 版的检查点续训，版本号接着记；横轴用“分叉后版本数” = version − branch_version
#   （两台机器吞吐不同，按时间对齐没有意义）。主线取 version >= branch_version 的评测，分叉点那次评测是两条线共同的起点。

DEFAULT_HOSTS = {"a": {"label": "3090 机", "local": "defect"}}


def load_hosts(path: Path) -> dict:
    """{主机名: 登记信息}；文件不存在或读不出时只有主机 a。"""
    h = read_json(path).get("hosts")
    return h if isinstance(h, dict) and h else dict(DEFAULT_HOSTS)


def load_lines(path: Path, hosts: dict) -> dict:
    """训练线登记，补齐默认值：{"main": {...}, "challenger": dict 或 None}。指向未登记（或待租）主机的项当作不存在。"""
    d = read_json(path)
    main = d.get("main") if isinstance(d.get("main"), dict) else {}
    main = {**main, "host": main.get("host") if main.get("host") in hosts else "a"}
    ch = d.get("challenger")
    if not isinstance(ch, dict) or ch.get("host") not in hosts or hosts[ch["host"]].get("pending") or ch.get("branch_version") is None:
        ch = None
    return {"main": main, "challenger": ch}


def host_dir(remote: Path, hosts: dict, h: str) -> Path:
    return remote / hosts.get(h, {}).get("local", "defect" if h == "a" else f"defect-{h}")


def pull_time(remote: Path, h: str) -> float | None:
    """这台主机最近一次拉取成功的时刻（主机 a 用 pull.ok，其他用 pull.<主机名>.ok）。"""
    names = ["pull.ok"] if h == "a" else [f"pull-{h}.ok", f"pull.{h}.ok"]      # tools/remote/pull.sh 写 pull-<主机>.ok
    for name in names:
        try:
            return float((remote / name).read_text().strip() or 0)
        except (OSError, ValueError):
            continue
    return None


def host_label(hosts: dict, h: str | None) -> str:
    return str(hosts.get(h, {}).get("label") or f"主机 {h}") if h else ""


class Sources:
    """一次生成用到的数据来源：主机、训练线、各目录。每次生成重新读 hosts.json / lines.json（迁移、分叉随时会改它们）。"""

    def __init__(self, remote: Path, hosts_file: Path, lines_file: Path, run: Path | None = None):
        if run is not None:              # 旧用法 --run DIR：只读这一个目录，当作单机
            remote, hosts, lines = run.parent, {"a": {"label": "", "local": run.name}}, {"main": {"host": "a"}, "challenger": None}
        else:
            hosts = load_hosts(hosts_file)
            lines = load_lines(lines_file, hosts)
        self.remote, self.hosts, self.lines = remote, hosts, lines
        self.files = [hosts_file, lines_file] if run is None else []
        m = lines["main"]
        self.main = m["host"]
        self.since = m.get("since")
        ch = lines["challenger"]
        prev = m.get("prev_host") or (ch["host"] if ch and ch["host"] != self.main else None) or ("a" if self.main != "a" else None)
        self.prev = prev if self.since and prev in hosts and prev != self.main else None
        self.ch = ch
        self.ch_host = ch["host"] if ch else None
        self.history = read_json(lines_file).get("history") or [] if run is None else []   # 已结束的训练线（迁走的主线、结束的对照）
        self.pre: Sources | None = None     # 对调视角时：分叉前的主线记录从原视角的主线取（见 flipped）
        self.marks: list[tuple[float, str]] = []    # 只属于这个视角的改动标线（对调视角里本线自己的改动）
        self._cache: dict = {}

    def dir(self, h: str) -> Path:
        return host_dir(self.remote, self.hosts, h)

    def dirs(self) -> list[Path]:
        """参与生成的目录（常驻模式据此判断输入是否变了）。"""
        hs = [self.main] + [h for h in (self.prev, self.ch_host) if h] + [h for h, v in self.hosts.items() if not v.get("pending")]
        return list(dict.fromkeys(self.dir(h) for h in hs))

    def jsonl(self, h: str, name: str) -> list[dict]:
        """同一台机器的同一个文件一次生成只读一遍（旧主机的日志既给主线的迁移前、又给对照线）。"""
        key = (h, name)
        if key not in self._cache:
            self._cache[key] = read_jsonl(self.dir(h) / name)
        return self._cache[key]

    def main_recs(self, name: str) -> list[dict]:
        """主线的记录（未 rewind）：迁移前取旧主机 time < since，迁移后取主线主机 time >= since；没有迁移时就是主线主机的全部记录。
        评测（eval.jsonl）在 EVAL_MOVE_T 之后取 A 的评测目录（决策记录 198）。"""
        recs = self._main_recs(name)
        if name == "eval.jsonl" and EVAL_MOVE_T and self.files:
            key = ("eval-dir", name)
            if key not in self._cache:
                self._cache[key] = read_jsonl(EVAL_DIR / name)
            recs = [r for r in recs if r.get("time", 0) < EVAL_MOVE_T] + [r for r in self._cache[key] if r.get("time", 0) >= EVAL_MOVE_T]
        return recs

    def eval_dir(self) -> Path:
        """评测进度、报错文件所在的目录。"""
        return EVAL_DIR if EVAL_MOVE_T and self.files and time.time() >= EVAL_MOVE_T else self.dir(self.main)

    def _main_recs(self, name: str) -> list[dict]:
        cur = self.jsonl(self.main, name)
        if not self.prev:
            return cur
        t = self.since
        # migrate.sh 会把截到检查点的 log、eval 一起搬到新主机；旧主机之后若被分叉成对照，运行目录换新，本机拉回的副本里就没有迁移前的历史了。
        # 所以主线主机自己带着迁移前的记录时直接用它（2026-10-26 迁到 4090 机后发现看板只剩迁移之后的一段）
        if cur and cur[0].get("time", t) < t:
            return cur
        pre = self.pre._main_recs(name) if self.pre else self.jsonl(self.prev, name)
        return [r for r in pre if r.get("time", 0) < t] + [r for r in cur if r.get("time", 0) >= t]

    def ch_recs(self, name: str) -> list[dict]:
        """对照线的记录（未 rewind）：对照主机上分叉之后的记录。"""
        if not self.ch:
            return []
        t0 = max(self.ch.get("branch_time") or 0, (self.since or 0) if self.ch_host == self.prev else 0)
        return [r for r in self.jsonl(self.ch_host, name) if r.get("time", 0) >= t0]

    def migration(self) -> tuple[float, str] | None:
        """（迁移时刻, 标线文字），没有迁移时为 None。"""
        lab = host_label(self.hosts, self.main)
        return (float(self.since), "迁移到" + (" " if lab[:1].isascii() else "") + lab) if self.prev and not self.pre else None

def rolling(xs: list[float | None], k: int) -> list[float | None]:
    out, win = [], []
    for x in xs:
        if x is not None:
            win.append(x)
            if len(win) > k:
                win.pop(0)
        out.append(sum(win) / len(win) if win else None)
    return out


def run_window(log: list[dict], k: int) -> tuple[list[float | None], list[float | None]]:
    """训练整局（采样）最近 k 次更新合计的前 10% 层数和通关率（百分比）；旧记录没有逐局层数时为 None。
    层数用直方图滑动统计，每个点只花几十次运算。"""
    return memo(("runwin", id(log), len(log), k), lambda: _run_window(log, k), log)


def _run_window(log: list[dict], k: int) -> tuple[list[float | None], list[float | None]]:
    from collections import deque
    hist = [0] * (MAX_FLOOR + 2); win = deque(); n = w = 0
    p90s, wins = [], []
    for r in log:
        fl = r.get("run_floors")
        if fl is None:
            p90s.append(None); wins.append(None); continue
        rw = round((r.get("run_win_rate") or 0) * len(fl))
        win.append((fl, rw)); n += len(fl); w += rw
        for f in fl: hist[min(f, MAX_FLOOR + 1)] += 1
        if len(win) > k:
            ofl, orw = win.popleft(); n -= len(ofl); w -= orw
            for f in ofl: hist[min(f, MAX_FLOOR + 1)] -= 1
        if not n:
            p90s.append(None); wins.append(None); continue
        need, acc, p = n * 0.1, 0, 0
        for f in range(MAX_FLOOR + 1, -1, -1):
            acc += hist[f]
            if acc >= need:
                p = f; break
        p90s.append(p); wins.append(100 * w / n)
    return p90s, wins


_series_memo: dict = {}


def memo(key: tuple, fn, *keep):
    """一次 build 里同样的序列只算一次：曲线按 3 个时间范围 × 桌面、手机各画一份，序列本身与它们无关。
    键里用列表的 id（与长度）区分不同的数据；keep 为键里取了 id 的对象，随结果一起留着，保证这一轮里它们不被回收、id 不被别的列表复用。
    build 开头清空。"""
    if key not in _series_memo:
        _series_memo[key] = (fn(), keep)
    return _series_memo[key][0]


def pooled(log: list[dict], get, k: int) -> list[float | None]:
    """胜率（百分比）：最近 k 次更新的胜场合计 / 场数合计。get(记录) 返回 (胜场, 场数)，旧记录没有这一项时返回 None，画为空。
    get 多为调用处现写的 lambda：按它的代码与闭包里的值区分（同一处、同样参数的 lambda 结果相同）。"""
    cl = repr([c.cell_contents for c in get.__closure__ or ()])
    return memo(("pooled", id(log), len(log), get.__code__, cl, k), lambda: _pooled(log, get, k), log)


def _pooled(log: list[dict], get, k: int) -> list[float | None]:
    from collections import deque
    out, win, sw, sn = [], deque(), 0, 0     # 滑动窗口里的胜场、场数合计（逐条加减，不再每步重算）
    for r in log:
        x = get(r)
        if x is not None:
            win.append(x); sw += x[0]; sn += x[1]
            if len(win) > k:
                a, b = win.popleft(); sw -= a; sn -= b
        out.append(100 * sw / sn if x is not None and sn else None)
    return out


# 非 Boss 的练习在日志里的键（如 act3_elite），值为 [胜场, 场数]。决策记录 201 起只剩第三幕真实局面与随机构筑精英战；
# 第一、二幕真实局面与随机构筑普通战停练，历史存档在原目录，看板不再画（日志里这几项为 [0, 0]，读日志时去掉，见 LOG_UNUSED）


def fights(r: dict, keys: list[str]) -> tuple[int, int] | None:
    """一条日志里几类练习合计的 (胜场, 场数)；旧记录没有这些键时为 None。"""
    xs = [r[k] for k in keys if k in r]
    return (sum(x[0] for x in xs), sum(x[1] for x in xs)) if xs and keys[0] in r else None


def pooled_rate(log: list[dict], act: int, k: int, real: bool = False) -> list[float | None]:
    """课程 Boss 胜率；real 时只算用整局真实快照的 Boss 名额。"""
    w, n = ("boss_real_wins", "boss_real_n") if real else ("boss_wins", "boss_n")
    return pooled(log, lambda r: (r[w][act], r[n][act]) if n in r else None, k)


def choice_stats(ch: dict, runs: int) -> dict:
    """局外行为（actor.choice_key 的计数）折算成几个可读的量：每局次数，及几个选择的比例（0–1，分母为 0 时为 None）。"""
    return choice_stats_from_sums({pre: sum(v for k, v in ch.items() if k.startswith(pre)) for pre in _CHOICE_PREFIXES}, runs)


def choice_stats_from_sums(g: dict, runs: int) -> dict:
    """同 choice_stats，输入为各前缀（_CHOICE_PREFIXES）下的计数合计。"""
    rest, heal, smith = g["REST_SITE:REST_OPTION:"], g["REST_SITE:REST_OPTION:REST:HEAL"], g["REST_SITE:REST_OPTION:REST:SMITH"]
    pick, skip, moves = g["CARD_REWARD:PICK_CARD_REWARD:"], g["CARD_REWARD:CARD_REWARD_ALT:ALT:SKIP"], g["MAP:MAP_MOVE:"]
    elite, removal = g["MAP:MAP_MOVE:MAPNODE:Elite"], g["SHOP:BUY:SHOP:CARD_REMOVAL"]
    per = lambda x: x / runs if runs else None
    return {"elite": per(elite), "rest_site": per(g["MAP:MAP_MOVE:MAPNODE:RestSite"]),
            "shop": per(g["MAP:MAP_MOVE:MAPNODE:Shop"]), "unknown": per(g["MAP:MAP_MOVE:MAPNODE:Unknown"]),
            "pick": per(pick), "skip": per(skip), "buy": per(g["SHOP:BUY:"] - removal), "remove": per(removal),
            "heal_share": heal / rest if rest else None, "smith_share": smith / rest if rest else None,
            "skip_share": skip / (pick + skip) if pick + skip else None, "elite_share": elite / moves if moves else None}


_CHOICE_PREFIXES = ("REST_SITE:REST_OPTION:", "REST_SITE:REST_OPTION:REST:HEAL", "REST_SITE:REST_OPTION:REST:SMITH", "CARD_REWARD:PICK_CARD_REWARD:",
                    "CARD_REWARD:CARD_REWARD_ALT:ALT:SKIP", "MAP:MAP_MOVE:", "MAP:MAP_MOVE:MAPNODE:Elite", "MAP:MAP_MOVE:MAPNODE:RestSite",
                    "MAP:MAP_MOVE:MAPNODE:Shop", "MAP:MAP_MOVE:MAPNODE:Unknown", "SHOP:BUY:", "SHOP:BUY:SHOP:CARD_REMOVAL")


def pooled_choices(log: list[dict], key: str, k: int) -> list[float | None]:
    """训练对局的局外行为：最近 k 次更新结束的整局合计后，choice_stats 的 key 项（比例按百分比）。旧记录没有时画为空。
    每条记录先按 choice_stats 用到的前缀汇总成几个数，再滑动累加；同一份日志、同一个 k 的结果在一次 build 里只算一次（memo）。"""
    out = memo(("choices", id(log), len(log), k), lambda: _choice_series(log, k), log).get(key)
    return out if out is not None else [None] * len(log)


_prefix_idx: dict[str, list[int]] = {}     # 局外选择的键 → 它匹配的 _CHOICE_PREFIXES 下标


def _choice_series(log: list[dict], k: int) -> dict[str, list]:
    """pooled_choices 用：choice_stats 每一项的序列。"""
    from collections import deque
    tot = [0] * (len(_CHOICE_PREFIXES) + 1); win: deque = deque(); series: dict[str, list] = {}
    for r in log:
        if "run_choices" not in r:
            for name in series: series[name].append(None)
            series.setdefault("_pending", []).append(None)
            continue
        vec = [0] * (len(_CHOICE_PREFIXES) + 1)
        for kk, v in r["run_choices"].items():
            idx = _prefix_idx.get(kk)
            if idx is None:
                idx = _prefix_idx[kk] = [j for j, pre in enumerate(_CHOICE_PREFIXES) if kk.startswith(pre)]
            for j in idx:
                vec[j] += v
        vec[-1] = r.get("run_episodes") or 0
        win.append(vec); tot = [a + b for a, b in zip(tot, vec)]
        if len(win) > k:
            old = win.popleft(); tot = [a - b for a, b in zip(tot, old)]
        stats = choice_stats_from_sums(dict(zip(_CHOICE_PREFIXES, tot[:-1])), tot[-1])
        for name, v in stats.items():
            if name not in series:      # 第一次出现的项，前面补空（默认值不能写在 setdefault 里：那样每次都会新建一个和日志一样长的列表）
                series[name] = [None] * len(series.get("_pending", []))
            series[name].append(None if v is None else 100 * v)
        series.setdefault("_pending", []).append(None)
    return series



def route_panel(ev: list[dict], k: int = 10) -> str:
    """评测里各幕的路线：实际走的精英、休息处、前面有火堆的精英，与这一幕开头能做到的最多值（最近 k 次评测合计，只算走到 Boss 的幕）。"""
    es = [e for e in ev if e.get("route")][-k:]
    if not es:
        return '<div class="empty" style="height:160px">打开路线摘要后的评测开始记录后显示</div>'
    tot: dict[str, list[int]] = {}
    for e in es:
        for act, v in e["route"].items():
            acc = tot.setdefault(act, [0] * len(v))
            for j, x in enumerate(v):
                acc[j] += x
    rows = []
    for act in sorted(tot, key=int):
        n, el, rest, cov, max_e, max_r, max_cov, max_r_e = tot[act]
        f = lambda x: f"{x / n:.2f}"
        rows.append(f'<tr><td>第{"一二三"[int(act)] if int(act) < 3 else int(act) + 1}幕</td><td>{n}</td><td>{f(el)} / {f(max_e)}</td>'
                    f'<td>{f(rest)} / {f(max_r_e)}</td><td>{f(cov)} / {f(max_cov)}</td><td>{f(max_r)}</td></tr>')
    return ('<table><thead><tr><th>幕</th><th>局数</th><th>精英<br>实际 / 最多</th><th>休息处<br>实际 / 同样精英数下最多</th>'
            '<th>前面有火堆的精英<br>实际 / 最多</th><th>最多休息处</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            f'<p class="note">最近 {len(es)} 次评测，平均每幕；“最多”按这一幕第一次选路时全部路线算（路线摘要），实际值不超过它。'
            '“前面有火堆”指上一个精英（或本幕开头）之后经过了休息处。只算走到 Boss 的幕。</p>')


def macro_panel(ev: list[dict], k: int = 4) -> str:
    """评测（贪心、固定种子）的局外行为：最近 k 次与再往前 k 次对照。"""
    es = [e for e in ev if "choices" in e]
    if not es:
        return '<div class="empty" style="height:200px">新评测开始记录局外行为后显示</div>'
    def agg(xs):
        tot: dict = {}
        for e in xs:
            for kk, v in e["choices"].items():
                tot[kk] = tot.get(kk, 0) + v
        return choice_stats(tot, sum(e["runs"] for e in xs))
    cur, prev = agg(es[-k:]), (agg(es[-2 * k:-k]) if len(es) > k else None)
    rows = [("进精英", "elite", "{:.1f} 次"), ("进休息处", "rest_site", "{:.1f} 次"), ("进商店", "shop", "{:.1f} 次"), ("进未知房间", "unknown", "{:.1f} 次"),
            ("抓卡", "pick", "{:.1f} 张"), ("跳过抓卡", "skip", "{:.1f} 次"), ("商店购买", "buy", "{:.1f} 件"), ("商店删卡", "remove", "{:.2f} 次"),
            ("休息处选休息", "heal_share", "pct"), ("休息处选升级", "smith_share", "pct")]
    f = lambda v, fmt: "—" if v is None else (pct(v, 0) if fmt == "pct" else fmt.format(v))
    body = "".join(f'<tr><td>{name}</td><td>{f(cur[key], fmt)}</td>' + (f'<td>{f(prev[key], fmt)}</td>' if prev else "") + '</tr>'
                   for name, key, fmt in rows)
    head = f'<th>最近 {min(k, len(es))} 次</th>' + (f'<th>再往前 {min(k, len(es) - k)} 次</th>' if prev else "")
    return (f'<table><thead><tr><th>每局</th>{head}</tr></thead><tbody>{body}</tbody></table>'
            '<p class="note">次数为平均每局；两个比例的分母是休息处的全部选择。</p>')


DECK_BIN_NAMES = ["≤15", "16–20", "21–25", "26–30", "31–35", "≥36"]     # 与 actor.DECK_BINS 对应
# 社区选牌统计（tools/fetch_card_stats.py 从 Spire Codex 拉的单人 A10 故障机器人、同一游戏版本）；牌的中文名读本机游戏安装的本地化
CARD_STATS = Path(__file__).resolve().parents[1] / "engine/card_stats/codex_defect_solo_a10_v0.111.0.json"
CARD_LOC = HOME / ".local/share/sls2/game/v0.111.0/res/localization/zhs/cards.json"
_card_ref: dict | None = None


CARD_PINYIN = Path(__file__).with_name("card_pinyin.json")     # tools/gen_card_pinyin.py 生成：{牌: "全拼 首字母"}
_card_pinyin: dict | None = None


def card_pinyin() -> dict:
    """选牌对照搜索用的拼音（读不到时为空，搜索只认中文名和 ID）。"""
    global _card_pinyin
    if _card_pinyin is None:
        try:
            _card_pinyin = json.loads(CARD_PINYIN.read_text(encoding="utf-8"))
        except Exception:
            _card_pinyin = {}
    return _card_pinyin


def card_ref() -> tuple[dict, dict, dict]:
    """(社区数据 {牌: 行}, 社区数据的概要, 中文名 {牌: 名})；读不到的部分为空。"""
    global _card_ref
    if _card_ref is None:
        _card_ref = {}
        try:
            d = json.loads(CARD_STATS.read_text(encoding="utf-8"))
            _card_ref["rows"] = {r["id"]: r for r in d["rows"]}
            _card_ref["meta"] = {k: v for k, v in d.items() if k != "rows"}
        except Exception:
            pass
        try:
            loc = json.loads(CARD_LOC.read_text(encoding="utf-8"))
            _card_ref["cn"] = {k[:-6]: v for k, v in loc.items() if k.endswith(".title")}
        except Exception:
            pass
    return _card_ref.get("rows", {}), _card_ref.get("meta", {}), _card_ref.get("cn", {})



RELIC_LOC = ENC_LOC.parent / "relics.json"
_RELIC_CN: dict[str, str] | None = None


def relic_cn(name: str) -> str:
    """遗物的中文名（本机游戏安装里的本地化文件，读不到时用 ID）。"""
    global _RELIC_CN
    if _RELIC_CN is None:
        try:
            loc = json.loads(RELIC_LOC.read_text(encoding="utf-8"))
            _RELIC_CN = {k[: -len(".title")]: v for k, v in loc.items() if k.endswith(".title")}
        except (OSError, ValueError):
            _RELIC_CN = {}
    return _RELIC_CN.get(name) or name.replace("_", " ").title()


ANCIENTS = ["NEOW", "OROBAS", "PAEL", "TEZCATARA", "NONUPEIPE", "VAKUU", "TANX", "DARV"]
_ANCIENT_OF: dict[str, str] | None = None


def ancient_of(relic: str) -> str:
    """遗物所属的先古之民：内容目录里先古之民事件引用的遗物（效果签名 ref:RELIC.X）；达弗给的是以前的 Boss 遗物，目录里只引用了尘封之书，
    其余对不上任何先古之民的都归达弗。"""
    global _ANCIENT_OF
    if _ANCIENT_OF is None:
        _ANCIENT_OF = {}
        try:
            cat = json.loads(CATALOG.read_text(encoding="utf-8"))["entities"]
            for e in cat:
                if e["category"] == "EVENT" and e["key"].removeprefix("EVENT.") in ANCIENTS:
                    for sig in e.get("signature", []):
                        if sig.startswith("ref:RELIC."):
                            _ANCIENT_OF.setdefault(sig[len("ref:RELIC."):], e["key"].removeprefix("EVENT."))
        except (OSError, ValueError, KeyError):
            pass
    return _ANCIENT_OF.get(relic, "DARV")


_ANCIENT_CN: dict[str, str] | None = None


def ancient_cn(key: str) -> str:
    global _ANCIENT_CN
    if _ANCIENT_CN is None:
        try:
            loc = json.loads((ENC_LOC.parent / "ancients.json").read_text(encoding="utf-8"))
            _ANCIENT_CN = {k[: -len(".title")]: v for k, v in loc.items() if k.endswith(".title")}
        except (OSError, ValueError):
            _ANCIENT_CN = {}
    return _ANCIENT_CN.get(key) or key.title()


_RELIC_TIP: dict[str, str] | None = None


def _fmt_desc(text: str, vars_: dict) -> str:
    """游戏的本地化说明（SmartFormat 写法）→ 纯文字：{名字} 和 {名字:格式()} 代入内容目录里的真实数值；
    {X.StringValue:cond:有值时|没值时} 取没值时的通用说法；颜色标签 [gold] 等去掉。"""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch != "{":
            out.append(ch); i += 1; continue
        depth, j = 0, i
        while j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        body = text[i + 1:j]; i = j + 1
        name, _, fmt = body.partition(":")
        if fmt.startswith("cond:"):
            alt = fmt[len("cond:"):].split("|", 1)
            out.append(_fmt_desc(alt[1] if len(alt) > 1 else "", vars_)); continue
        v = vars_.get(name)
        num = None if v is None else (int(v) if float(v).is_integer() else v)
        if fmt.startswith("energyIcons"):
            arg = fmt[len("energyIcons("):-1]
            # energyPrefix:energyIcons(1) 是一个单独的能量图标：前面紧跟数字时读作“费”（“耗能增加 1 费”），否则读作“能量”
            if num is None and name == "energyPrefix":
                out.append("\x01")         # 去掉颜色标签之后再看前面是不是数字
            else:
                out.append(f"{num}点能量" if num is not None else f"{arg}点能量")
        elif fmt.startswith("percentMore") and num is not None:
            out.append(f"{round((float(num) - 1) * 100):g}")       # 倍率 1.25 → 多 25%
        elif fmt.startswith("starIcons") or name == "singleStarIcon":
            out.append(f"{num if num is not None else 1}颗星")
        else:
            out.append(str(num) if num is not None else "X")      # 内容目录里没有这个数值（少数遗物在运行时才算）
    text = re.sub(r"\[/?[a-z_]+(=[^\]]*)?\]", "", "".join(out))
    return re.sub(r"\x01", "能量", re.sub(r"(\d)\x01", r"\1费", text))


def relic_tip(name: str) -> str:
    """遗物的效果说明（本机游戏的中文本地化，数值按内容目录代入）；读不到为空。"""
    global _RELIC_TIP
    if _RELIC_TIP is None:
        _RELIC_TIP = {}
        try:
            loc = json.loads(RELIC_LOC.read_text(encoding="utf-8"))
            cat = json.loads(CATALOG.read_text(encoding="utf-8"))["entities"]
            vars_ = {e["key"].removeprefix("RELIC."): e.get("vars") or {} for e in cat if e["category"] == "RELIC"}
            for k, v in loc.items():
                if k.endswith(".description"):
                    r = k[: -len(".description")]
                    _RELIC_TIP[r] = _fmt_desc(v, vars_.get(r, {})).strip()
        except (OSError, ValueError, KeyError):
            pass
    return _RELIC_TIP.get(name, "")


def set_game_version(version: str, catalog: Path | None = None) -> None:
    """换成另一个游戏版本的中文名与内容目录（2026-11-13，愚者在 v0.113.0 上训练，看板代码在主线、主线的游戏版本仍是 v0.111.0）：
    遭遇、卡牌、遗物、先古之民的本地化读 ~/.local/share/sls2/game/<版本>/res/localization/zhs/，内容目录读 catalog（默认本仓库的
    engine/catalog/<版本>/catalog.json），并清掉已读的缓存。读不到时各处照旧退回 ID。"""
    global ENC_LOC, CARD_LOC, RELIC_LOC, CATALOG, _card_ref, _RELIC_CN, _ANCIENT_OF, _ANCIENT_CN, _RELIC_TIP, _enc_cn
    loc = HOME / ".local/share/sls2/game" / version / "res/localization/zhs"
    ENC_LOC, CARD_LOC, RELIC_LOC = loc / "encounters.json", loc / "cards.json", loc / "relics.json"
    CATALOG = Path(catalog) if catalog else Path(__file__).resolve().parents[1] / "engine/catalog" / version / "catalog.json"
    _card_ref = _RELIC_CN = _ANCIENT_OF = _ANCIENT_CN = _RELIC_TIP = _enc_cn = None


# 先古遗物分享图的样式与绘制（不进 f-string 模板，原样拼在面板后面）
# 先古遗物分享图的样式（拼在面板里）；绘制函数 saInit 在页面脚本里，首屏与每次原地刷新后各调一次
ANC_SHARE = """<style>
.sa{text-align:center}
.sa-img{display:block;width:100%;max-width:760px;margin:0 auto;border-radius:12px;box-shadow:0 6px 24px rgba(0,0,0,.25);min-height:240px;background:#141826}
.sa-save{font:inherit;font-size:15px;font-weight:600;margin-top:12px;padding:9px 24px;border-radius:999px;border:0;background:#fb7299;color:#fff;cursor:pointer}
</style>"""
ANC_SHARE_JS = r"""function saInit(scope){scope.querySelectorAll('.sa').forEach(function(root){
var D=JSON.parse(root.querySelector('.sa-data').textContent);
var W=1080, P=32, G=20, CW=(W-2*P-2*G)/3, FONT='"PingFang SC","Microsoft YaHei","Noto Sans CJK SC","Source Han Sans SC",sans-serif';
var ROW=48, AH=60, AP=12, ABG=14, CH=72, HEAD=344, FOOT=70;
var imgs={}, ready=Promise.all(Object.keys(D.icons).map(function(k){return new Promise(function(res){var im=new Image();im.onload=function(){imgs[k]=im;res()};im.onerror=res;im.src=D.icons[k]})}));
function f(px,w){return (w||'700')+' '+px+'px '+FONT}
function rr(c,x,y,w,h,r){c.beginPath();c.moveTo(x+r,y);c.arcTo(x+w,y,x+w,y+h,r);c.arcTo(x+w,y+h,x,y+h,r);c.arcTo(x,y+h,x,y,r);c.arcTo(x,y,x+w,y,r);c.closePath()}
function icon(c,k,x,y,s,fb){var im=imgs[k];if(im){var z=Math.min(s/im.width,s/im.height);c.drawImage(im,x+(s-im.width*z)/2,y+(s-im.height*z)/2,im.width*z,im.height*z)}
  else if(fb){c.fillStyle='#2e3550';c.beginPath();c.arc(x+s/2,y+s/2,s/2,0,7);c.fill();c.fillStyle='#ffd166';c.font=f(s*.5);c.textAlign='center';c.fillText(fb,x+s/2,y+s/2+2);c.textAlign='left'}}
function fit(c,t,px,maxw,w){var s=px;c.font=f(s,w);while(c.measureText(t).width>maxw&&s>18){s-=1;c.font=f(s,w)}}
function face(c,x,y,d){var im=imgs['@face'];if(!im)return false;c.save();c.beginPath();c.arc(x+d/2,y+d/2,d/2,0,7);c.closePath();c.clip();c.drawImage(im,x,y,d,d);c.restore();
  c.lineWidth=Math.max(4,d/28);c.strokeStyle='#fb7299';c.beginPath();c.arc(x+d/2,y+d/2,d/2,0,7);c.stroke();return true}
function col(r){return r>=.5?'#ffd166':r>=.15?'#e8ebf3':'#7d849a'}
function colH(act){var h=CH;act.ancients.forEach(function(g){h+=AH+g.relics.length*ROW+AP+ABG});return h}
function draw(){
  var bodyH=Math.max.apply(null,D.acts.map(colH)), h=HEAD+bodyH+FOOT;
  var cv=document.createElement('canvas');cv.width=W;cv.height=h;var c=cv.getContext('2d');c.textBaseline='middle';
  var bg=c.createLinearGradient(0,0,0,h);bg.addColorStop(0,'#1d1533');bg.addColorStop(.12,'#121624');bg.addColorStop(1,'#0d101b');c.fillStyle=bg;c.fillRect(0,0,W,h);
  // 斜向水印（先画，内容盖在上面）
  c.save();c.translate(W/2,h/2);c.rotate(-Math.PI/7);c.fillStyle='rgba(255,255,255,.045)';c.font=f(40);c.textAlign='center';
  for(var yy=-h;yy<h;yy+=210)for(var xx=-W*1.3;xx<W*1.3;xx+=430)c.fillText('B站 @'+D.up,xx+((yy/210)%2?215:0),yy);c.restore();
  // 头部：直播标识与账号、标题、胜率、说明
  rr(c,P,40,176,58,29);c.fillStyle='#fb7299';c.fill();c.fillStyle='#fff';c.beginPath();c.arc(P+30,69,9,0,7);c.fill();c.font=f(32);c.fillText('直播中',P+50,71);
  c.fillStyle='#fb7299';c.font=f(42);c.fillText('B站 @'+D.up,P+196,71);
  var FD=210, hasF=face(c,W-P-FD,52,FD), TW=hasF?W-2*P-FD-24:W-2*P;
  c.fillStyle='#fff';fit(c,D.title,68,TW,'800');c.fillText(D.title,P,160);
  // 进阶等级后面跟 NoSL 标签（描边胶囊），再接胜率
  c.fillStyle='#ffd166';c.font=f(56,'800');c.fillText(D.asc,P,236);var lx=P+c.measureText(D.asc).width+18;
  c.font=f(36,'800');var tw=c.measureText(D.tag).width;rr(c,lx,206,tw+32,60,30);c.fillStyle='#ffd166';c.fill();c.fillStyle='#1d1533';c.fillText(D.tag,lx+16,238);lx+=tw+32+18;
  c.fillStyle='#ffd166';c.font=f(56,'800');c.fillText('· '+D.rate,lx,236);
  c.fillStyle='#a3a9bb';c.font=f(28,'500');c.fillText(D.sub,P,290);
  // 三栏：每幕一栏，按先古之民分块
  D.acts.forEach(function(act,a){
    var x=P+a*(CW+G), y=HEAD;
    c.fillStyle='#fff';c.font=f(40,'800');c.fillText(act.n,x+4,y+28);c.fillStyle=['#7fb2ff','#b48cff','#ff8fab'][a];c.fillRect(x+4,y+54,CW-8,4);
    y+=CH;
    act.ancients.forEach(function(g){
      var bh=AH+g.relics.length*ROW+AP;rr(c,x,y,CW,bh,18);c.fillStyle='rgba(32,37,58,.94)';c.fill();
      icon(c,g.k,x+12,y+10,42,g.n.charAt(0));var cs=g.c.toLocaleString('zh-CN')+' 次';c.font=f(22,'500');var cw=c.measureText(cs).width;c.fillStyle='#fff';fit(c,g.n,32,CW-80-cw-12,'800');var tw=c.measureText(g.n).width;c.fillText(g.n,x+62,y+32);c.fillStyle='#8f96ab';c.font=f(22,'500');c.fillText(cs,x+62+tw+12,y+35);
      var ry=y+AH;
      g.relics.forEach(function(r){
        icon(c,r.k,x+10,ry+4,40,null);
        var nx=x+56, pr=x+CW-12, nw=CW-56-12-86;
        c.fillStyle='#eef0f6';fit(c,r.n,27,nw,'600');c.fillText(r.n,nx,ry+18);
        rr(c,nx,ry+36,nw,6,3);c.fillStyle='#2b3049';c.fill();
        if(r.r>0){rr(c,nx,ry+36,Math.max(6,nw*r.r),6,3);c.fillStyle=col(r.r);c.fill()}
        c.fillStyle=col(r.r);c.font=f(30,'800');c.textAlign='right';c.fillText(Math.round(r.r*100)+'%',pr,ry+25);c.textAlign='left';
        ry+=ROW});
      y+=bh+ABG});
  });
  // 最短一栏下面的空白放直播间卡片；底部写数据来源
  var hs=D.acts.map(colH), m=hs.indexOf(Math.min.apply(null,hs)), px=P+m*(CW+G), py=HEAD+hs[m]+10, ph=Math.min(640,HEAD+bodyH-py);
  if(ph>=300){rr(c,px,py,CW,ph,22);var pg=c.createLinearGradient(0,py,0,py+ph);pg.addColorStop(0,'#fb7299');pg.addColorStop(1,'#c4507a');c.fillStyle=pg;c.fill();
    var cx=px+CW/2, fd=Math.min(150,ph*.3), t=py+28, ok=face(c,cx-fd/2,t,fd);if(ok)t+=fd+14;var rest=py+ph-t, k=function(q){return t+rest*q};
    c.textAlign='center';c.fillStyle='#fff';c.font=f(28,'700');c.fillText('B站直播间',cx,k(.1));
    fit(c,D.up,60,CW-40,'800');c.fillText(D.up,cx,k(.3));c.fillStyle='rgba(255,255,255,.35)';c.fillRect(px+40,k(.45),CW-80,2);
    c.fillStyle='#fff';fit(c,'AI 故障机器人',34,CW-40,'800');c.fillText('AI 故障机器人',cx,k(.58));var ln='杀戮尖塔 2 · '+D.asc+' · '+D.tag;fit(c,ln,28,CW-40,'600');c.fillText(ln,cx,k(.72));
    c.fillStyle='#fff3c4';c.font=f(46,'800');c.fillText(D.rate,cx,k(.88));c.textAlign='left'}
  else{rr(c,P,h-FOOT-70,W-2*P,64,32);c.fillStyle='#fb7299';c.fill();c.fillStyle='#fff';c.font=f(34,'800');c.textAlign='center';c.fillText('B站直播间 · '+D.up,W/2,h-FOOT-37);c.textAlign='left'}
  c.fillStyle='#8a90a3';c.font=f(24,'500');c.textAlign='center';c.fillText(D.foot,W/2,h-34);c.textAlign='left';
  return cv}
var img=root.querySelector('.sa-img'), url=null;if(window.SA_LAST)img.src=window.SA_LAST;      // 原地刷新后先用上一张占位，画好再换
ready.then(function(){url=draw().toDataURL('image/png');img.src=url;window.SA_LAST=url});
root.querySelector('.sa-save').onclick=function(){if(!url)return;var l=document.createElement('a');l.href=url;l.download='AI故障机器人_先古遗物选取率.png';document.body.appendChild(l);l.click();l.remove()};
})}
"""


ASSETS = OUT.parent / "回放/assets"      # 回放播放器从本机游戏资源包提取的图标（tools/replay_assets.py）
# 头部第二行：进阶等级、紧跟的标签（NoSL：不存档读档重来）、胜率
SHARE_TITLE, SHARE_ASC, SHARE_TAG, SHARE_RATE, SHARE_UP = "杀戮尖塔 2 · AI 故障机器人", "进阶 10", "NoSL", "胜率 75%+", "辉弥天野"
SHARE_FACE = HOME / ".local/share/sls2/share/up_face.jpg"   # B 站头像（space.bilibili.com/159141914，256 像素）


def _icon_uri(path: Path) -> str | None:
    try:
        return f"data:image/{'jpeg' if path.suffix == '.jpg' else 'webp'};base64," + base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


def ancient_detail(log: list[dict], since_t: float, since_text: str) -> str:
    """先古之民三选一遗物的具体数据：训练整局（带探索）自训练统计起点的合计，按幕分三栏、栏里按先古之民分块，组内按选取率从高到低，
    每件遗物给选取率与被选/被提供次数（分享图里只有选取率）。"""
    es = [e for e in log if e.get("run_ancient") and e["time"] >= since_t]
    if not es:
        return f'<div class="empty" style="height:160px">自 {esc(since_text)} 起累计，新记录到达后显示</div>'
    tot = ancient_totals(es, since_t)
    scr = tot.pop("_SCREEN", [0] * 6)
    acts: list[dict[str, list[tuple[str, int, int]]]] = [{}, {}, {}]
    for n, v in tot.items():
        for a in range(3):
            if v[a]:
                acts[a].setdefault(ancient_of(n), []).append((n, v[a], v[3 + a]))
    cols = []
    for a, groups in enumerate(acts):
        blocks = []
        for g in sorted(groups, key=lambda g: -sum(o for _, o, _ in groups[g])):
            items = sorted(groups[g], key=lambda x: (-(x[2] / x[1] if x[1] else 0), -x[1]))
            rows = "".join(f'<div class="arow" data-tip="{esc(relic_tip(n))}"><span class="an">{esc(relic_cn(n))}</span><div class="fbar"><i style="width:{100 * p_ / o if o else 0:.0f}%"></i></div>'
                           f'<span class="av">{pct(p_ / o if o else None, 0)}<span class="fn">{p_:,}/{o:,}</span></span></div>' for n, o, p_ in items)
            blocks.append(f'<div class="ablk"><div class="ah"><b>{esc(ancient_cn(g))}</b><span class="fn">约 {sum(o for _, o, _ in items) // 3:,} 次</span></div>{rows}</div>')
        cols.append(f'<div class="acol" style="--ac:var(--act{a + 1})"><div class="acth"><b>第{"一二三"[a]}幕</b><span class="fn">{scr[a]:,} 次三选一</span></div>'
                    + ("".join(blocks) or '<div class="empty" style="height:80px">还没有记录</div>') + '</div>')
    sub = f'自第 {es[0]["version"]:,} 版（{since_text}）起累计（训练整局，带探索）：共 {sum(scr[:3]):,} 个先古之民三选一界面。'
    return f'<p class="note">{esc(sub)}</p><div class="ancacts">{"".join(cols)}</div>'


def ancient_panel(log: list[dict], since_t: float, since_text: str) -> str:
    """先古之民三选一遗物的分享图：训练整局（带探索）自训练统计起点的合计，三幕合成一张 1080 宽的图（浏览器画布绘制、图标内嵌），
    每幕一栏、栏里按先古之民分块、块内按选取率从高到低；页面上显示生成的图片（手机可长按保存），另有保存按钮。"""
    es = [e for e in log if e.get("run_ancient") and e["time"] >= since_t]
    if not es:
        return f'<div class="empty" style="height:160px">自 {esc(since_text)} 起累计，新记录到达后显示</div>'
    tot = ancient_totals(es, since_t)
    tot.pop("_SCREEN", None)
    # run_ancient 每件遗物为 [第一、二、三幕被提供, 第一、二、三幕被选]；达弗第二、三幕都会出现，两幕各算各的
    acts: list[dict[str, list[tuple[str, int, int]]]] = [{}, {}, {}]
    for n, v in tot.items():
        for a in range(3):
            if v[a] >= 20:
                acts[a].setdefault(ancient_of(n), []).append((n, v[a], v[3 + a]))
    icons: dict[str, str] = {}
    if (u := _icon_uri(SHARE_FACE)):
        icons["@face"] = u
    data = {"acts": [], "icons": icons, "title": SHARE_TITLE, "asc": SHARE_ASC, "tag": SHARE_TAG, "rate": SHARE_RATE, "up": SHARE_UP, "sub": "先古遗物选取率：三选一时各遗物被选取的比例",
            "foot": f"数据：AI 训练对局，第 {es[0]['version']:,} 版起，截至 {time.strftime('%Y-%m-%d')}"}
    for a, groups in enumerate(acts):
        anc = []
        for g in sorted(groups, key=lambda g: -sum(o for _, o, _ in groups[g])):
            if (u := _icon_uri(ASSETS / f"ui/ancient_node_{g.lower()}.webp")):
                icons["@" + g] = u
            rel = []
            for n, o, p_ in sorted(groups[g], key=lambda x: -(x[2] / x[1])):
                if n not in icons and (u := _icon_uri(ASSETS / f"relic/{n}.webp")):
                    icons[n] = u
                rel.append({"k": n, "n": relic_cn(n), "r": round(p_ / o, 4)})
            anc.append({"k": "@" + g, "n": ancient_cn(g), "c": sum(o for _, o, _ in groups[g]) // 3, "relics": rel})
        data["acts"].append({"n": f"第{'一二三'[a]}幕", "ancients": anc})
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return ('<div class="sa"><img class="sa-img" alt="先古遗物选取率（三幕）"><button type="button" class="sa-save">保存图片</button>'
            f'<script type="application/json" class="sa-data">{blob}</script>{ANC_SHARE}</div>')



def card_panel(ev: list[dict], log: list[dict] | None = None, k_log: int = 1000, min_offered: int = 20, min_share: float = 0.005, top: int = 12) -> str:
    """选牌与社区数据对照，按“我们的选取率 − 社区选取率”分多拿、少拿两段。默认用最近 k_log 次更新里训练整局的选牌（run_cards，
    2026-10-29 起记录，带探索、种子各不相同）；还没有时退回自开始记录起的全部评测（固定种子、贪心，提供的牌重复多，误差大）。"""
    # 2026-10-29 起选牌计数每 card_log_every 次更新才写一次（决策记录 197）：按“最近 k_log 次更新”覆盖的时间窗口取，与写入频率无关
    t0 = log[-k_log]["time"] if log and len(log) >= k_log else 0.0
    rs = [x for x in (log or []) if x.get("run_cards") and x["time"] >= t0]
    es = [e for e in ev if e.get("cards")]
    if not rs and not es:
        return '<div class="empty" style="height:200px">新评测开始记录选牌后显示</div>'
    tot: dict[str, list[int]] = {}
    for c in ([x["run_cards"] for x in rs] if rs else [e["cards"] for e in es]):
        for n, v in c.items():
            acc = tot.setdefault(n, [0] * len(v))
            for j, x in enumerate(v):
                acc[j] += x
    ref, meta, cn = card_ref()
    scr = tot.pop("_SCREEN", [0] * 6)
    # 按出现频率筛：被提供次数至少占选牌界面数的 min_share（且不少于 min_offered 次）。棱彩宝石等让奖励里偶尔出现别的职业的牌，
    # 这类牌每张只占约 0.2% 以下，本职业卡池的牌每张约 1% 以上
    floor_n = max(min_offered, min_share * sum(scr[:3]))
    rate = lambda p, o: p / o if o else None
    items, full = [], []
    for n, v in tot.items():
        r = ref.get(n)
        if r and sum(v[:3]) >= floor_n and r.get("pick_rate") is not None:
            ours = rate(sum(v[3:]), sum(v[:3]))
            items.append((ours - r["pick_rate"] / 100, n, v, r))
        if n != "_SCREEN" and sum(v[:3]) > 0:      # 完整列表：提供过的每一张牌（含没有社区数据、出现很少的）
            ok = r is not None and r.get("pick_rate") is not None
            full.append(((sum(v[3:]) / sum(v[:3]) - r["pick_rate"] / 100) if ok else None, n, v, r if ok else {}))
    pys = card_pinyin()
    def rows(sel, full=False):
        out = []
        for d, n, v, r in sel:
            acts = " · ".join("—" if not v[a] else f"{100 * v[3 + a] / v[a]:.0f}" for a in range(3))
            lift = "—" if r.get("lift") is None else f'{r["lift"]:+.1f}'
            cr = "—" if r.get("pick_rate") is None else f'{r["pick_rate"]:.0f}%'
            name = cn.get(n) or n.replace("_", " ").title()
            ours = rate(sum(v[3:]), sum(v[:3]))
            # 完整列表：搜索用的键（中文名、ID、全拼、首字母，空格分隔）与各列排序用的数值（没有为空）
            tr, tv = ('<tr>', [""] * 7) if not full else (
                f'<tr data-s="{esc((name + " " + n.lower().replace("_", "") + " " + pys.get(n, "")).lower())}">',
                [f' data-v="{x}"' if x is not None else "" for x in
                 (None, sum(v[:3]), ours, (v[3] / v[0]) if v[0] else None, r.get("pick_rate"), d, r.get("lift"))])
            out.append(f'{tr}<td>{esc(name)}</td><td{tv[1]}>{sum(v[:3])}</td>'
                       f'<td{tv[2]}>{pct(ours, 0)}</td><td class="fn"{tv[3]}>{acts}</td>'
                       f'<td{tv[4]}>{cr}</td>'
                       f'<td{tv[5]}>{"—" if d is None else f"{100 * d:+.0f}"}</td><td{tv[6]}>{lift}</td></tr>')
        return "".join(out)
    head = ('<thead><tr><th>牌</th><th>提供</th><th>我们<br>选取率</th><th>分幕<br>一·二·三</th><th>社区<br>选取率</th>'
            '<th>差<br>(点)</th><th>社区<br>lift</th></tr></thead>')
    items.sort()
    full.sort(key=lambda x: -sum(x[2][:3]))
    more, less = [x for x in items[::-1] if x[0] > 0][:top], [x for x in items if x[0] < 0][:top]
    skip = " · ".join("—" if not scr[a] else pct(scr[3 + a] / scr[a], 0) for a in range(3))
    who = (f'最近 {len(rs)} 次更新里训练整局的选牌合计（第 {rs[0].get("version", "—")}–{rs[-1].get("version", "—")} 版，{scr[0] + scr[1] + scr[2]:,} 个选牌界面，种子各不相同、按概率采样）' if rs
           else f'{len(es)} 次评测合计（{sum(e["runs"] for e in es):,} 局，固定种子、贪心；训练里的选牌记录还没有）')
    src = (f'社区：Spire Codex 单人 A10 故障机器人 {meta.get("bracket", "").split(":")[-1]}，{meta.get("character_runs", 0):,} 局'
           f'（胜 {meta.get("character_wins", 0):,}），数据截至 {str(meta.get("data_through", ""))[:10]}') if meta else "社区数据缺失"
    return (f'<div class="cg2"><div><div class="subh">我们多拿</div><div class="tscroll"><table>{head}<tbody>{rows(more)}</tbody></table></div></div>'
            f'<div><div class="subh">我们少拿</div><div class="tscroll"><table>{head}<tbody>{rows(less)}</tbody></table></div></div></div>'
            f'<div class="cfull"><div class="cfbar"><div class="subh">全部 {len(full)} 张牌（含没有社区数据、出现很少的牌；点表头排序）</div>'
            f'<input class="cfq" type="search" placeholder="搜索：中文、拼音或首字母" aria-label="搜索卡牌" autocomplete="off"><span class="cfn"></span></div>'
            f'<div class="tscroll tall"><table>{head}<tbody>{rows(full, True)}</tbody></table></div></div>'
            f'<p class="note">我们：{who}，只列被提供次数至少占选牌界面 {100 * min_share:.1f}%（本次为 {floor_n:,.0f} 次）的牌，偶尔出现的牌（如棱彩宝石带来的其他职业的牌）不统计；'
            f'选牌界面跳过率（第一·二·三幕）{skip}。{src}。选取率 = 被选中次数 / 被提供次数；'
            'lift 为社区里拿了这张牌的对局、按拿牌时的层数校正后的胜率差（个百分点），人类数据带选择偏差，只作参照。'
            + ('训练按概率采样（带探索），选取率比贪心温和。</p>' if rs else '贪心策略的选取率比人类更极端（接近 0 或 100%）。</p>'))


CUR_CN = {"run": "整局", "mid": "中途开局整局", "boss1": "第一幕 Boss", "boss1_probe": "第一幕 Boss · 中途牌组", "boss2": "第二幕 Boss",
          "boss2_probe": "第二幕 Boss · 中途牌组", "boss3": "第三幕第一个 Boss", "boss3_chain": "第三幕连打两个 Boss", "boss3_second": "第三幕第二个 Boss",
          "boss3_probe": "第三幕 Boss · 中途牌组", "act1_monster": "第一幕普通战", "act1_elite": "第一幕精英", "act2_monster": "第二幕普通战",
          "act2_elite": "第二幕精英", "act3_monster": "第三幕普通战", "act3_elite": "第三幕精英", "mix_monster": "随机构筑普通战", "mix_elite": "随机构筑精英"}
DEC_CN = {"MAP": "选路", "CARD_REWARD": "抓牌", "REWARDS": "领奖励", "EVENT": "事件", "SHOP": "商店", "REST_SITE": "休息处", "CARD_SELECT": "选卡",
          "TREASURE": "宝箱", "MINIGAME": "小游戏", "BUNDLE": "卡包"}


def curriculum_panel(log: list[dict], k: int = 200) -> str:
    """课程分配（cur：各类练习 [局数, 收到步数, 入训步数, 胜场, 战损合计]）与局外有效样本（macro_eff：按决策类型 [入训, 多选, 改选,
    多选 |优势| 合计, 多选 权重 × |优势| 合计]），最近 k 次更新合计；两项都是 2026-10-29 起记录（决策记录 197）。"""
    rs = [r for r in log[-k:] if "cur" in r]
    if not rs:
        return '<div class="empty" style="height:120px">新记录开始后显示</div>'
    cur: dict[str, list[float]] = {}
    me: dict[str, list[float]] = {}
    for r in rs:
        for key, v in r["cur"].items():
            acc = cur.setdefault(key, [0.0] * 5)
            for j, x in enumerate(v): acc[j] += x
        for key, v in (r.get("macro_eff") or {}).items():
            acc = me.setdefault(key, [0.0] * 5)
            for j, x in enumerate(v): acc[j] += x
    tot = sum(v[2] for v in cur.values()) or 1
    def rate(w, n):
        return f'{pct(w / n, 0)}<span class="fn">{int(n):,} 局</span>' if n else "—"
    rows = "".join(
        f'<tr><td>{esc(CUR_CN.get(key, key))}</td><td>{pct(v[2] / tot, 1)}</td><td>{pct(v[2] / v[1], 0) if v[1] else "—"}</td>'
        f'<td>{rate(v[3], v[0]) if not key.startswith(("run", "mid")) else "—"}</td>'
        f'<td>{f"{100 * v[4] / v[0]:+.0f}%" if v[0] and not key.startswith(("run", "mid")) else "—"}</td></tr>'
        for key, v in sorted(cur.items(), key=lambda kv: -kv[1][2]))
    g = sum(v[4] for v in me.values()) or 1
    n_all = sum(v[0] for v in me.values()) or 1
    rows2 = "".join(
        f'<tr><td>{esc(DEC_CN.get(key, key))}</td><td>{pct(v[0] / n_all, 1)}</td><td>{pct(v[1] / v[0], 0) if v[0] else "—"}</td>'
        f'<td>{int(v[2]):,}</td><td>{pct(v[4] / g, 1)}</td></tr>'
        for key, v in sorted(me.items(), key=lambda kv: -kv[1][4]))
    return ('<div class="cg2"><div><div class="subh">各类练习</div><div class="tscroll"><table><thead><tr><th>练习</th><th>占入训步数</th><th>入训率</th><th>胜率</th><th>平均战损</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '</div><div><div class="subh">局外决策</div><div class="tscroll"><table><thead><tr><th>局外决策</th><th>占局外步数</th><th>多选</th><th>强制改选</th><th>策略梯度份额</th></tr></thead>'
            f'<tbody>{rows2}</tbody></table></div></div></div>'
            f'<p class="note">最近 {len(rs)} 次更新合计。战损为战斗练习里血量变化占最大生命的平均（负为掉血）；整局一行的步数含分段送出的部分。'
            '多选：有两个以上合法选项的步，只有一个选项的步只训练价值头；策略梯度份额：多选步的策略权重 × |标准化优势| 占全部局外的比例，近似各类决策实际推动策略的分量。</p>')


def deck_size_panel(log: list[dict], ev: list[dict], k_log: int = 300, k_ev: int = 10) -> str:
    """按牌组张数分档：真实快照 Boss 练习的胜率（最近 k_log 次更新合计），评测里各节点的分布与 Boss 过关率（最近 k_ev 次评测合计）。"""
    nb = len(DECK_BIN_NAMES)
    ls = [r["boss_deck"] for r in log[-k_log:] if "boss_deck" in r]
    es = [e["deck_bins"] for e in ev[-k_ev:] if "deck_bins" in e]
    if not ls and not es:
        return '<div class="empty" style="height:160px">新记录开始后显示</div>'
    prac = [[[sum(x[a][k][j] for x in ls) for j in (0, 1)] for k in range(nb)] for a in (0, 1)]
    evb = {key: [[sum(x[key][k][j] for x in es) for j in (0, 1)] for k in range(nb)] for key in ("boss1", "boss2")}
    a2 = [sum(x["act2_first"][k] for x in es) for k in range(nb)]
    def rate(w, n, unit):
        return f'{pct(w / n, 0)}<span class="fn">{n} {unit}</span>' if n else "—"
    def share(n, tot):
        return f'{pct(n / tot, 0)}<span class="fn">{n} 局</span>' if tot and n else "—"
    t1, t2 = sum(n for _, n in evb["boss1"]), sum(a2)
    rows = "".join(
        f'<tr><td>{name}</td><td>{rate(*prac[0][k], "场")}</td><td>{share(evb["boss1"][k][1], t1)}</td><td>{rate(*evb["boss1"][k], "局")}</td>'
        f'<td>{share(a2[k], t2)}</td><td>{rate(*prac[1][k], "场")}</td><td>{rate(*evb["boss2"][k], "局")}</td></tr>'
        for k, name in enumerate(DECK_BIN_NAMES))
    return ('<div class="tscroll"><table><thead><tr><th>牌组张数</th><th>第一幕 Boss<br>练习胜率</th><th>第一幕 Boss<br>评测分布</th>'
            '<th>第一幕 Boss<br>评测过关率</th><th>第二幕第一场<br>评测分布</th><th>第二幕 Boss<br>练习胜率</th><th>第二幕 Boss<br>评测过关率</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            f'<p class="note">练习：整局在这一幕 Boss 前留下的真实快照（不含中途快照）重打 Boss，最近 {len(ls)} 次更新合计；'
            f'评测：最近 {len(es)} 次评测合计，固定种子、贪心。张数含诅咒。场数、局数少于约 30 的比例只作参考；'
            '牌多的局往往也是走得更远的局，两者混在一起，看的是同一幕内不同档之间的差别。</p>')


ROOM_CN = {"Monster": "普通", "Elite": "精英", "Boss": "Boss"}


_enc_cn: dict | None = None


def enc_name(key: str) -> tuple[str, str]:
    """评测死因的键 "幕:房间类型:遭遇" → (遭遇中文名, "第 N 幕 · 房间")。没有中文名时，遭遇 ID 去掉 _WEAK/_NORMAL 等后缀、按词首字母大写。"""
    global _enc_cn
    if key.count(":") != 2:
        return key, ""
    if _enc_cn is None:
        try:
            _enc_cn = json.loads(ENC_LOC.read_text(encoding="utf-8"))
        except Exception:
            _enc_cn = {}
    act, room, enc = key.split(":")
    name = _enc_cn.get(f"{enc}.title")
    if not name:
        for suf in ("_WEAK", "_NORMAL", "_ELITE", "_BOSS"):
            enc = enc.removesuffix(suf)
        name = enc.replace("_", " ").title()
    return name, f"第 {int(act) + 1} 幕 · {ROOM_CN.get(room, room)}"


# 训练统计（各 Boss 过关率、训练死因、先古遗物选取率）的起算节点：三项共用一个时刻（按时间，两条线对齐），记在 REMOTE/stats_since.json。
# 重置（看板“卡点诊断”旁的按钮，或 python3 tools/dashboard.py --reset-stats）时把上一段的累计留档到原目录的 训练统计留档/，再以重置时刻为新起点。
# 更早的留档（只有死因与 Boss 过关率）在原目录根下：训练死因与Boss过关率_*.json。文件缺失时用下面的旧起点（2026-10-27 09:27）
STATS_FILE = REMOTE / "stats_since.json"
RESET_REQ = REMOTE / "reset_requests.log"      # publish.sh 从香港服务器取回的重置请求（每行：请求时刻 来源地址）
STATS_ARCHIVE = OUT.parent / "训练统计留档"
DIAG_SINCE_T = 1793089620
DIAG_SINCE_TEXT = "2026-10-27 09:27"


def stats_since() -> dict:
    """当前起算节点：{"t": 时刻, "text": 显示用时刻, "req": 已处理的最后一个重置请求时刻, "history": [以往各段]}。"""
    try:
        d = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        if d.get("t"):
            return d
    except (OSError, ValueError):
        pass
    return {"t": DIAG_SINCE_T, "text": DIAG_SINCE_TEXT, "req": 0, "history": []}


# 事件“饥饿蘑菇”的两个选项各带一件遗物，actor.ancient_offer 把它当成先古之民三选一记了下来；它们不属于任何先古之民，统计里去掉
NOT_ANCIENT = ("BIG_MUSHROOM", "FRAGRANT_MUSHROOM")


def ancient_totals(log: list[dict], since_t: float) -> dict[str, list[int]]:
    tot: dict[str, list[int]] = {}
    for e in log:
        if not e.get("run_ancient") or e["time"] < since_t:
            continue
        for n, v in e["run_ancient"].items():
            acc = tot.setdefault(n, [0] * len(v))
            for j, x in enumerate(v):
                acc[j] += x
    gone = [tot.pop(n) for n in NOT_ANCIENT if n in tot]
    if gone and "_SCREEN" in tot:       # 界面数也去掉蘑菇事件：两件遗物每次同时被提供，没选遗物的次数 = 提供 − 两件被选
        scr = tot["_SCREEN"]
        for a in range(3):
            offered = max(v[a] for v in gone)
            scr[a] -= offered
            scr[3 + a] -= offered - sum(v[3 + a] for v in gone)
    return tot


def reset_stats(src: "Sources", now: float | None = None, why: str = "") -> Path:
    """把当前一段（起算节点到 now）的三项累计按线写一份留档，再把起算节点挪到 now。返回留档路径。"""
    now = now or time.time()
    cur = stats_since()
    t0 = cur["t"]
    # 统一起点之前先古遗物一直从开始记录起累计：第一次重置时它的留档从头算
    anc_t0 = t0 if STATS_FILE.exists() else 0
    lines = {f"主线 {src.main}（{host_label(src.hosts, src.main)}）": rewind(src.main_recs("log.jsonl"))}
    if src.ch:
        lines[f"对照 {src.ch_host}（{host_label(src.hosts, src.ch_host)}）"] = rewind(src.ch_recs("log.jsonl"))
    t_txt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    out: dict = {"说明": f"训练整局（带探索）的死因、各 Boss 过关率与先古遗物选取，{t_txt(t0)} 起的累计，截至 {t_txt(now)}。"
                        f"之后看板从 {t_txt(now)} 起重新累计（三项共用一个起点，两条线按时间对齐）。" + (why and f"重置原因：{why}。") +
                        ("先古遗物这一段从开始记录起算（统一起点之前它没有单独的起点）。" if not anc_t0 else "")}
    for name, log in lines.items():
        seg = [r for r in log if t0 <= r["time"] < now]
        deaths, bosses, n, v0 = train_diag(seg, t0)
        aseg = [r for r in log if anc_t0 <= r["time"] < now and r.get("run_ancient")]
        anc = ancient_totals(aseg, anc_t0)
        scr = anc.pop("_SCREEN", [0] * 6)
        out[name] = {
            "起始版本": v0, "截止版本": seg[-1]["version"] if seg else None, "训练整局数": n,
            "死因（局数，占全部训练整局的比例）": {f"{a}（{b}）": [v, round(v / n, 4) if n else None]
                                     for (a, b), v in ((enc_name(k), v) for k, v in sorted(deaths.items(), key=lambda x: -x[1]))},
            "各 Boss 过关（打赢场数, 场数, 过关率）": {f"{enc_name(f'{k.split(':')[0]}:Boss:{k.split(':', 1)[1]}')[0]}（第 {int(k.split(':')[0]) + 1} 幕）":
                                              [w, m, round(w / m, 4) if m else None] for k, (w, m) in sorted(bosses.items())},
            "先古遗物（被提供次数, 被选次数, 选取率）": {f"{relic_cn(r)}（{ancient_cn(ancient_of(r))}）": [sum(v[:3]), sum(v[3:]), round(sum(v[3:]) / sum(v[:3]), 4) if sum(v[:3]) else None]
                                             for r, v in sorted(anc.items(), key=lambda x: (ancient_of(x[0]), -sum(x[1][:3])))},
            "先古之民三选一界面数（总数, 没选遗物的次数）": [sum(scr[:3]), sum(scr[3:])],
            "先古遗物的起止版本": [aseg[0]["version"], aseg[-1]["version"]] if aseg else None,
        }
    STATS_ARCHIVE.mkdir(parents=True, exist_ok=True)
    arch = STATS_ARCHIVE / f"训练统计_{time.strftime('%Y-%m-%d %H%M', time.localtime(t0))}起_至{time.strftime('%Y-%m-%d %H%M', time.localtime(now))}.json"
    arch.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    hist = cur.get("history", []) + [{"t": t0, "text": cur["text"], "end": now, "archive": arch.name, "why": why}]
    new = {"t": now, "text": t_txt(now), "req": max(cur.get("req", 0), now), "history": hist}
    tmp = STATS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, STATS_FILE)
    return arch


def poll_reset(src: "Sources") -> None:
    """看板按钮的重置请求（香港服务器记下、publish.sh 取回到 RESET_REQ）：有比上次处理过的更新的请求就重置一次。
    同一批里连点几下只算一次；请求时刻早于当前起点的忽略。"""
    try:
        reqs = [float(l.split()[0]) for l in RESET_REQ.read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError, IndexError):
        return
    cur = stats_since()
    new = [r for r in reqs if r > max(cur.get("req", 0), cur["t"])]
    if new:
        arch = reset_stats(src, why="看板按钮")
        d = stats_since(); d["req"] = max(new); STATS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已按看板按钮重置训练统计，留档 {arch}", flush=True)


def train_diag(log: list[dict], since_t: float) -> tuple[dict, dict, int, int | None]:
    """训练整局从 since_t 时刻起累计的死因（局数）、各 Boss 的 [打赢场数, 场数]、整局数和实际开始累计的版本。"""
    deaths: dict[str, int] = {}
    bosses: dict[str, list[int]] = {}
    n, v0 = 0, None
    for r in log:
        if "run_deaths" not in r or r["time"] < since_t:
            continue
        v0 = r["version"] if v0 is None else v0
        n += r.get("run_episodes") or 0
        for key, v in r["run_deaths"].items():
            deaths[key] = deaths.get(key, 0) + v
        for key, (w, m) in r["run_bosses"].items():
            t = bosses.setdefault(key, [0, 0]); t[0] += w; t[1] += m
    return deaths, bosses, n, v0


def deaths_panel(deaths: dict, n: int, top: int = 10) -> str:
    """训练整局里输掉的局死在哪场战斗（占全部训练整局的比例）。"""
    if not n:
        return '<div class="empty" style="height:200px">训练开始记录死因后显示</div>'
    items = sorted(deaths.items(), key=lambda x: -x[1])[:top]
    peak = items[0][1] if items else 1
    rows = "".join(f'<div class="drow"><div class="dname"><b>{esc(name)}</b><small>{esc(where)}</small></div>'
                   f'<div class="fbar"><i style="width:{100 * v / peak:.1f}%"></i></div><div class="dval">{pct(v / n, 1)}<span class="fn">{v:,} 局</span></div></div>'
                   for name, where, v in ((*enc_name(key), v) for key, v in items))
    return f'<div class="funnel">{rows}</div>'


def boss_panel(bosses: dict) -> str:
    """训练整局按 Boss 的过关率（打赢场数 / 场数），按幕、再按场数排列。"""
    if not bosses:
        return '<div class="empty" style="height:200px">训练开始按 Boss 记录后显示</div>'
    items = sorted(bosses.items(), key=lambda x: (int(x[0].split(":")[0]), -x[1][1]))
    rows, act_prev = [], None
    for key, (w, n) in items:
        act = int(key.split(":")[0])
        if act != act_prev:     # 按幕分组，组头写一次“第 N 幕”，各行不再重复
            rows.append(f'<div class="dgrp">第{"一二三"[act] if act < 3 else act + 1}幕</div>'); act_prev = act
        name = enc_name(f"{act}:Boss:{key.split(':', 1)[1]}")[0]
        rows.append(f'<div class="drow"><div class="dname"><b>{esc(name)}</b></div>'
                    f'<div class="fbar"><i style="width:{100 * w / max(n, 1):.1f}%"></i></div><div class="dval">{pct(w / max(n, 1), 0)}<span class="fn">{w:,}/{n:,} 场</span></div></div>')
    return f'<div class="funnel">{"".join(rows)}</div>'


def deck_panel(ev: list[dict], k: int = 4) -> str:
    """最近 k 次评测里，打到几个节点时的牌组规模（局数加权平均）。"""
    es = [e for e in ev if "deck_at" in e][-k:]
    if not es:
        return '<div class="empty" style="height:200px">新评测开始记录后显示</div>'
    rows = []
    for key, name in (("boss1", "第一幕 Boss"), ("act2_first", "第二幕第一场"), ("boss2", "第二幕 Boss"),
                      ("act3_first", "第三幕第一场"), ("boss3", "第三幕第一个 Boss"), ("boss3b", "第三幕第二个 Boss")):
        xs = [e["deck_at"][key] for e in es if e["deck_at"].get(key, [0])[0] > 0]
        n = sum(x[0] for x in xs)
        avg = [sum(x[0] * x[j] for x in xs) / n for j in (1, 2, 3)] if n else None
        rows.append(f'<tr><td>{name}</td><td>{n}</td>' + (f'<td>{avg[0]:.1f}</td><td>{avg[1]:.1f}</td><td>{pct(avg[2], 0)}</td>' if avg else '<td>—</td>' * 3) + '</tr>')
    return ('<table><thead><tr><th>节点</th><th>局数</th><th>牌数</th><th>遗物</th><th>开场血量</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            '<p class="note">对照：战斗课程估计牌组的规模（池里没有真实快照时使用）为第一幕 Boss 15–20 张牌、2–4 个遗物，'
            '第二幕 Boss 22–27 张牌、4–7 个遗物，开场血量 50%–100%。</p>')


ABILITY = HOME / ".local/share/sls2/ability_test"      # 千种子复测与加测的结果（tools/remote/retest.sh）


def retest_rows(k: int = 8) -> tuple[list[dict], str | None]:
    """最近 k 次复测（state.json 里的定时复测）与加测（结果目录带标签的计划外测法，compare.json），按完成时间从新到旧；
    另返回正在跑的那次（retest.log 最后一条“开始复测”之后还没有“完成”）。"""
    rows = []
    for r in (read_json(ABILITY / "retest/state.json").get("runs") or []):
        d = Path(r.get("dir", ""))
        try:
            t = (d / "compare.json").stat().st_mtime
        except OSError:
            continue
        vs = r.get("vs") or {}
        rows.append({"t": t, "v": r["version"], "kind": "定时", "win": r["win_rate"], "pm": r.get("pm"), "floor": r.get("floor_mean"),
                     "vs": vs.get("version"), "a": vs.get("only_this"), "b": vs.get("only_prev"), "p": vs.get("p")})
    for c in ABILITY.glob("v*-*/compare.json"):
        tag = c.parent.name.split("-", 1)[1]
        m = read_json(c)
        if not m.get("models"):
            continue
        x, pr = m["models"][0], (m.get("pairs") or [{}])[0]
        ci = x.get("ci") or [x["win_rate"], x["win_rate"]]
        rows.append({"t": c.stat().st_mtime, "v": x["version"], "kind": "加测" if tag == "计划外" else "加测·" + tag.replace("计划外-", ""),
                     "win": x["win_rate"], "pm": (ci[1] - ci[0]) / 2, "floor": x.get("floor_mean"),
                     "vs": pr.get("b"), "a": pr.get("only_a"), "b": pr.get("only_b"), "p": pr.get("p")})
    rows.sort(key=lambda r: -r["t"])
    running = None
    try:
        lines = (ABILITY / "retest/retest.log").read_text(encoding="utf-8").splitlines()[-200:]
        last = max((i for i, l in enumerate(lines) if "开始复测第" in l), default=None)
        if last is not None and not any("完成" in l for l in lines[last:]):
            running = lines[last][:19] + " " + lines[last].split("开始复测", 1)[1].split("，主线")[0]
    except OSError:
        pass
    return rows[:k], running


def retest_panel(log: list[dict] | None = None) -> str:
    """复测与加测：千种子（11-01 起为种子池 P1，之前 RT0000–RT0999）贪心，与上一次逐局配对。log 给定时版本旁注上大致的累计决策数（与训练图的横轴同一单位）。"""
    rows, running = retest_rows()
    vs_steps = sorted((r["version"], r["total_steps"]) for r in (log or []) if r.get("version") and r.get("total_steps"))

    def steps_at(v: int) -> str:
        i = bisect.bisect_right(vs_steps, (v, float("inf"))) - 1
        return "" if i < 0 or v - vs_steps[i][0] > 200 else f' <span class="fn">约 {vs_steps[i][1] / 1e8:.1f} 亿</span>'

    if not rows and not running:
        return '<div class="empty" style="height:120px">还没有复测结果</div>'
    def pair(r):
        if r["a"] is None:
            return "—"
        sig = ' style="color:var(--s1);font-weight:600"' if r["p"] is not None and r["p"] < 0.05 else ""
        return f'<span{sig}>{r["a"]} : {r["b"]}（p = {r["p"]:.3f}）</span> <span class="fn">对第 {r["vs"]:,} 版</span>'
    body = "".join(f'<tr><td>{time.strftime("%m-%d %H:%M", time.localtime(r["t"]))}</td><td>{esc(r["kind"])}</td><td>第 {r["v"]:,} 版{steps_at(r["v"])}</td>'
                   f'<td><b>{pct(r["win"])}</b>{"" if r["pm"] is None else f" ± {100 * r["pm"]:.1f}"}</td>'
                   f'<td>{"—" if r["floor"] is None else f"{r['floor']:.2f}"}</td><td>{pair(r)}</td></tr>' for r in rows)
    run = f'<p class="note">正在测：{esc(running)}</p>' if running else ""
    # 复测快照的通关并集（tools/retest_union.py，决策记录 247）：允许读档重来的支线版本要看这个数
    u = read_json(ABILITY / "retest/union.json")
    if u.get("all"):
        a, w4 = u["all"], u.get("last4")
        bf = f'，含并集补测 {u["backfill"]} 个' if u.get("backfill") else ""
        run += (f'<p class="note">种子池 {esc(u.get("pool", ""))} 的 {a["snapshots"]} 个复测快照{bf}：至少一个通关 <b>{pct(a["union"])}</b>'
                + (f'（最近 4 个 {pct(w4["union"])}）' if w4 else "")
                + f'，全部通关 {a["all"]}，都没通关 {a["none"]}（共 {a["seeds"]} 个种子，单个平均 {pct(a["mean_single"])}）</p>')
        if u.get("probe"):     # 首通探测（tools/unsolved_probe.py，决策记录 250）：只打全局未通关的种子，首通的快照立即进直播快照池
            pr = u["probe"]
            last = "；最近：" + "，".join(f'第 {f["version"]:,} 版 · {esc(f["seed"])}' for f in pr["firsts"][-3:][::-1]) if pr["firsts"] else ""
            run += f'<p class="note">首通探测：已测 {pr["probes"]} 个快照，首通 {pr["first_n"]} 个种子{last}</p>'
    return (f'{run}<div class="tscroll"><table><thead><tr><th>完成</th><th>类型</th><th>版本（累计决策）</th><th>通关</th><th>平均层数</th><th>逐局配对（只有这一版通关 : 只有对照通关）</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


# ---------------------------------------------------------------- 格式

def cn_count(n: float | None) -> str:
    """中文计数：多少亿多少万，精确到万（不足一万照写）。"""
    if n is None:
        return "—"
    n = int(n)
    if n < 10_000:
        return f"{n}"
    wan = n // 10_000
    yi, rest = divmod(wan, 10_000)
    if yi:
        return f"{yi} 亿 {rest} 万" if rest else f"{yi} 亿"
    return f"{wan} 万"


def cn_count_k(n: float | None) -> str:
    """中文计数（局数用）：以万为单位、精确到千，如 13.4 万；过亿写成“X 亿 Y.Y 万”，不足一万照写。"""
    if n is None:
        return "—"
    n = int(n)
    if n < 10_000:
        return f"{n:,}"
    yi, rest = divmod(n, 100_000_000)
    wan = f"{rest // 1000 / 10:.1f} 万"
    return f"{yi} 亿 {wan}" if yi else wan


def fmt_dur(s: float) -> str:
    s = int(max(0, s))
    d, h, m = s // 86400, s % 86400 // 3600, s % 3600 // 60
    return f"{d} 天 {h} 小时" if d else (f"{h} 小时 {m} 分" if h else f"{m} 分")


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def esc(s) -> str:
    return html.escape(str(s))


def when(e: dict) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(e["time"]))


def delta_html(cur: float | None, ref: float | None, unit: str, scale: float = 1.0, digits: int = 1, label: str = "") -> str:
    if cur is None or ref is None:
        return ""
    dv = (cur - ref) * scale
    if abs(dv) < 10 ** -digits / 2:
        return f'<span class="dl flat">持平{esc(label)}</span>'
    cls, arrow = ("up", "▲") if dv > 0 else ("down", "▼")
    return f'<span class="dl {cls}">{arrow} {abs(dv):.{digits}f}{esc(unit)}{esc(label)}</span>'


# ---------------------------------------------------------------- 折线图

def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    step = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(step))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= step), default=mag * 10)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


def downsample(p: list[tuple[float, float]], n: int = MAX_PTS) -> list[tuple[float, float]]:
    """按横轴等宽分桶取平均，保留首尾两点的原值，曲线形状不变、点数有上限。"""
    if len(p) <= n:
        return p
    if n < 3:           # 很短的一段（如刚开始记录的新指标）只配到 2 个点：保留首尾
        return [p[0], p[-1]]
    x0, x1 = p[0][0], p[-1][0]
    if x1 <= x0:
        return p[-n:]
    w = (x1 - x0) / (n - 2)
    mid, cur, bx, by, prev = [], None, [], [], None
    for x, y in p[1:-1]:
        if prev is not None and x < prev:       # 横轴不单调（少见）：按桶号整理后再取平均
            return [p[0]] + _bucket_mean(p[1:-1], x0, w, n) + [p[-1]]
        prev = x
        b = min(n - 3, int((x - x0) / w))
        if b != cur:                # 横轴单调时同一个桶的点是连着的，顺序收集即可，不用建字典再排序
            if bx:                  # 用 sum() 求和（3.12 起是补偿求和），与原来逐桶 sum 的结果逐位相同
                mid.append((sum(bx) / len(bx), sum(by) / len(by)))
            cur, bx, by = b, [], []
        bx.append(x); by.append(y)
    if bx:
        mid.append((sum(bx) / len(bx), sum(by) / len(by)))
    return [p[0]] + mid + [p[-1]]


def _bucket_mean(pts: list[tuple[float, float]], x0: float, w: float, n: int) -> list[tuple[float, float]]:
    buckets: dict[int, list] = {}
    for x, y in pts:
        buckets.setdefault(min(n - 3, int((x - x0) / w)), []).append((x, y))
    return [(sum(a for a, _ in b) / len(b), sum(c for _, c in b) / len(b)) for _, b in sorted(buckets.items())]


def x_scale(kind: str, lo: float, hi: float) -> tuple[float, str, str]:
    """横轴单位随范围切换：返回（除数，轴标题，提示框里的单位）。"""
    if kind == "steps":
        return (1e8, "累计决策数（亿）", "亿") if hi >= 1e8 else (1e4, "累计决策数（万）", "万")
    if kind == "ver":                     # 对照实验：分叉后经过的版本数
        return 1, "分叉后版本数", "版"
    return (86400, "开训后天数", "天") if hi - lo > 3 * 86400 else (3600, "开训后小时数", "小时")


def spread(vals: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """一维标注防重叠：排序后依次推开，保证间距，整体不出 [lo, hi]。"""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = list(vals)
    prev = None
    for i in order:
        v = max(out[i], lo)
        if prev is not None and v - prev < gap:
            v = prev + gap
        out[i] = v; prev = v
    nxt = None      # 右端超出时从右往左回推：只挪挤在一起的那几个，不把最左边的推出画面
    for i in reversed(order):
        v = min(out[i], hi)
        if nxt is not None and nxt - v < gap:
            v = nxt - gap
        out[i] = max(v, lo); nxt = out[i]
    return out


def _sorted_times(ts: list) -> bool:
    """时间戳没有空、且不减（一次 build 里同一份列表只查一遍）。"""
    return memo(("sorted", id(ts), len(ts)), lambda: all(t is not None for t in ts) and all(a <= b for a, b in zip(ts, ts[1:])), ts)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """通关率 p（n 局）的 Wilson 95% 置信区间。"""
    if not n:
        return 0.0, 1.0
    c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); d = 1 + z * z / n
    return (c - r) / d, (c + r) / d


def _pick(xs: list, ys: list, ts: list, since: float | None) -> list[tuple[float, float]]:
    """一条线在时间范围内、有值的点。时间单调时二分找起点，只看这一段。"""
    if since is not None and ts and _sorted_times(ts):
        i = bisect.bisect_left(ts, since)
        xs, ys, ts = xs[i:], ys[i:], ts[i:]
    return [(x, y) for x, y, t in zip(xs, ys, ts) if y is not None and (since is None or t is None or t >= since)]


def line_chart(label: str, series: list[dict], *, w: int = 600, h: int = 240, y_fmt: str = "{:.1f}", y_min: float | None = None,
               y_max: float | None = None, refs: list[tuple[float, str]] = (), bands: list[tuple[float, float, str, str]] = (),
               x_kind: str = "steps", since: float | None = None, x_origin: float = 0.0,
               vrefs: list[tuple[float, str]] = (), max_pts: int = MAX_PTS) -> str:
    """series: [{"name", "short", "var", "xs", "ys", "ts", "markers", "fmt", "gap", "bold", "lo", "hi"}]；一个纵轴，最多 3 条线。
    lo、hi：与 ys 对齐的区间下沿、上沿（如通关率的 95% 置信区间），画成同色的半透明带。
    xs 为原始横坐标（累计决策数，或时间戳）；ts 为各点的时间戳，用于按时间范围筛选（since）。
    bands: 纵向分区底色 [(y0, y1, 文字, css 变量)]。vrefs: 竖直参考线 [(原始横坐标, 文字)]。"""
    L, R, T, B = 50, 104, 14, 34
    picked = []
    for s in series:
        ts = s.get("ts") or [None] * len(s["xs"])
        picked.append((s, memo(("pick", id(s["xs"]), id(s["ys"]), id(ts), since), lambda: _pick(s["xs"], s["ys"], ts, since), s["xs"], s["ys"], ts)))
    allp = [pt for _, p in picked for pt in p]
    if not allp:
        return f'<div class="empty" style="height:{h}px">这段时间暂无数据</div>'
    rx0, rx1 = min(x for x, _ in allp), max(x for x, _ in allp)
    div, x_label, x_unit = x_scale(x_kind, rx0, rx1)
    if x_kind != "steps":                 # 时间轴：从开训（或第一次采样）算起
        picked = [(s, [((x - x_origin), y) for x, y in p]) for s, p in picked]
        rx0, rx1 = rx0 - x_origin, rx1 - x_origin
    x0, x1 = rx0 / div, rx1 / div
    if x1 - x0 < 1e-9:
        x1 = x0 + 1
    ys = [y for _, y in allp] + [r[0] for r in refs]
    lo = min(ys) if y_min is None else y_min
    hi = max(ys) if y_max is None else y_max
    if hi - lo < 1e-9:
        hi = lo + 1
    ticks = nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    sx = lambda x: L + (x - x0) / (x1 - x0) * (w - L - R)
    sy = lambda y: T + (1 - (y - lo) / (hi - lo)) * (h - T - B)
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(label)}">']
    for y0, y1, txt, var in bands:
        a, b = max(lo, y0), min(hi, y1)
        if a < b:
            g.append(f'<rect x="{L}" y="{sy(b):.1f}" width="{w - L - R}" height="{sy(a) - sy(b):.1f}" fill="var({var})"/>')
            g.append(f'<text class="bandt" x="{L + 8}" y="{sy(b) + 15:.1f}">{esc(txt)}</text>')
    for t in ticks:
        g.append(f'<line class="grid" x1="{L}" x2="{w - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        g.append(f'<text class="tick" x="{L - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{esc(y_fmt.format(t))}</text>')
    for xt in nice_ticks(x0, x1, max(4, w // 150)):
        if sx(xt) < w - R - (12 * len(x_label) + 10 if x_kind == "ver" else 115):        # 最右端留给轴标题
            g.append(f'<text class="tick" x="{sx(xt):.1f}" y="{h - 12}" text-anchor="middle">{xt:,g}</text>')
    g.append(f'<text class="tick" x="{w - R}" y="{h - 12}" text-anchor="end">{esc(x_label)}</text>')
    g.append(f'<line class="axis" x1="{L}" x2="{w - R}" y1="{sy(lo):.1f}" y2="{sy(lo):.1f}"/>')
    for yv, txt in refs:
        if lo <= yv <= hi:
            g.append(f'<line class="ref" x1="{L}" x2="{w - R}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}"/>')
            g.append(f'<text class="reft" x="{w - R - 6}" y="{sy(yv) - 5:.1f}" text-anchor="end">{esc(txt)}</text>')
    placed = []     # 已放的标线文字 (左端, 右端, 行)：两条线挨得近时，后一条的文字下移一行，不叠在一起
    # 改动多了文字会挤成一团：只给范围内最近 VREF_LABELS 条写字，更早的只画淡线；悬停（手机上点按）时提示框里写靠近的那条改动
    shown = sorted((xv, txt) for xv, txt in vrefs if x0 <= xv / div <= x1)
    labeled = []    # 从最近的往前挑，和已挑的相距 90 像素以上才写字，最多 VREF_LABELS 条
    for xv, _ in reversed(shown):
        if len(labeled) < VREF_LABELS and all(abs(sx(xv / div) - sx(o / div)) >= 90 for o in labeled):
            labeled.append(xv)
    for xv, txt in shown:
        px = sx(xv / div)
        if xv not in labeled:
            g.append(f'<line class="ref faint" x1="{px:.1f}" x2="{px:.1f}" y1="{T}" y2="{h - B}"/>')
            continue
        g.append(f'<line class="ref" x1="{px:.1f}" x2="{px:.1f}" y1="{T}" y2="{h - B}"/>')
        left = px > (w - R) * 0.7      # 靠右时文字放在线的左边，不压线尾标注
        tw = 11 * len(txt) + 6           # 文字宽度的粗略估计（中文约 11 像素一个字）
        a0, a1 = (px - tw, px) if left else (px, px + tw)
        row = 0
        while any(r == row and a0 < b1 and b0 < a1 for b0, b1, r in placed):
            row += 1
        placed.append((a0, a1, row))
        g.append(f'<text class="reft" x="{px + (-6 if left else 6):.1f}" y="{T + 14 + 14 * row}" text-anchor="{"end" if left else "start"}">{esc(txt)}</text>')
    meta = {"x0": x0, "x1": x1, "L": L, "R": R, "W": w, "xdiv": div, "xkind": x_kind, "xunit": x_unit, "series": [],
            "vrefs": [[round(xv / div, 4), txt] for xv, txt in shown]}
    ends = []
    for s, _ in picked:
        if s.get("lo") is None:
            continue
        ts = s.get("ts") or [None] * len(s["xs"])
        lo_p, hi_p = _pick(s["xs"], s["lo"], ts, since), _pick(s["xs"], s["hi"], ts, since)
        if len(lo_p) >= 2:
            pts = [(sx(x / div), sy(min(hi, y))) for x, y in hi_p] + [(sx(x / div), sy(max(lo, y))) for x, y in reversed(lo_p)]
            g.append(f'<path class="ciband" d="M{" L".join(f"{a:.1f},{b:.1f}" for a, b in pts)} Z" fill="var({s["var"]})"/>')
    for s, p in picked:
        if not p:
            continue
        # 采样中断（如训练暂停）超过 gap 时断开曲线，不把空白连成直线
        gap = s.get("gap")
        segs, cur = [], []
        for x, y in p:
            if cur and gap is not None and x - cur[-1][0] > gap:
                segs.append(cur); cur = []
            cur.append((x, y))
        segs.append(cur)
        d_segs = [downsample([(x / div, y) for x, y in sg], max(2, max_pts * len(sg) // len(p))) for sg in segs]
        d = " ".join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(y):.1f}' for sg in d_segs for i, (x, y) in enumerate(sg))
        g.append(f'<path class="line{" bold" if s.get("bold") else ""}" d="{d}" stroke="var({s["var"]})"/>')
        if s.get("markers") and len(p) <= MAX_MARKERS:
            for x, y in p:
                g.append(f'<circle class="dot" cx="{sx(x / div):.1f}" cy="{sy(y):.1f}" r="3.5" fill="var({s["var"]})"/>')
        ends.append((s, p[-1]))
        meta["series"].append({"name": s["name"], "var": s["var"], "fmt": s.get("fmt", y_fmt),
                               "pts": [[round(x, 5), round(y, 4)] for sg in d_segs for x, y in sg]})
    # 线尾直接标注（文字用文字色，旁边的色块标识系列），互相错开
    for (s, (_, ly)), yy in zip(ends, spread([sy(ly) for _, (_, ly) in ends], 15, T + 6, h - B)):
        g.append(f'<rect x="{w - R + 8}" y="{yy - 4:.1f}" width="10" height="3" rx="1.5" fill="var({s["var"]})"/>')
        g.append(f'<text class="direct" x="{w - R + 22}" y="{yy + 4:.1f}">{esc(s.get("short", s["name"]))} {esc(s.get("fmt", y_fmt).format(ly))}</text>')
    g.append(f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{h - B}" style="display:none"/>')
    g.append("</svg>")
    legend = "".join(f'<span class="lg"><i style="background:var({s["var"]})"></i>{esc(s["name"])}</span>' for s, p in picked if p) if len(series) > 1 else ""
    return (f'<div class="legend">{legend}</div><div class="chartwrap" data-chart=\'{esc(json.dumps(meta, ensure_ascii=False))}\'>'
            + "".join(g) + '<div class="tip" style="display:none"></div></div>')


# ---------------------------------------------------------------- 首屏组件

def journey(ev: list[dict], w: int = 1180, h: int = 296, cid: str = "trackclip") -> str:
    """通关之路：上方是最近一次评测停在各层的局数，下方是按幕分段的进度条（见 LAYOUT），平均层数实心填充。"""
    if not ev:
        return '<div class="empty" style="height:250px">等待第一次评测</div>'
    last, first = ev[-1], ev[0]
    L, R = 16, 16
    WIN = N_SLOTS + 1                                         # 最右边加一格“通关”（2026-10-27 起）：通关的局单独放这一格
    cw = (w - L - R) / WIN
    fx = lambda s: L + (s - 1) * cw                           # 第 s 格的左边
    cx = lambda v: L + floor_slot(v) * cw                     # 连续层数 → 横坐标（v 层走完的位置）
    ht, hb = 40, 184                                          # 分布区上下沿
    ty, th = 198, 28                                          # 进度条
    # 每格的含义（悬停提示）与是否加长层
    info, extra, bosses = [""] * (N_SLOTS + 1), set(), []
    base = 0
    for name, n, e, nb, blabel in LAYOUT:
        for k in range(1, n - nb + 1):
            info[base + k] = f"{name}第 {k} 层"
        for k in range(1, e + 1):
            info[base + n - nb + k] = f"{name}加长层（第 {n - nb + k} 层，只在地图加长时出现）"; extra.add(base + n - nb + k)
        for b in range(1, nb + 1):
            info[base + n + e - nb + b] = f"{blabel}" if nb == 1 else f"{blabel}第 {b} 个 Boss"
            bosses.append((base + n + e - nb + b, blabel, b == 1, b == nb))
        base += n + e
    # 最近 12 次评测合计（2026-10-27 起；单次 192 局噪声大）
    recent = ev[-KPI_K:]
    pos = [p_ for e in recent for p_ in (e.get("pos") or [legacy_pos(f) for f in e.get("floors", [])])]
    counts = [0] * (WIN + 1)
    for a, k, b in pos:
        counts[slot_of(a, k, b)] += 1
    # 通关的局在 pos 里记在第三幕第二个 Boss 那一格（和死在那里的局一起），挪到“通关”格
    wins = min(sum(round(e.get("win_rate", 0) * e.get("runs", 0)) for e in recent), counts[N_SLOTS])
    counts[N_SLOTS] -= wins; counts[WIN] = wins
    info.append("通关")
    top = max(counts) or 1
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart journey" role="img" aria-label="通关之路">']
    # 加长层：分布区里浅色底纹
    for s_ in sorted(extra):
        g.append(f'<rect class="xcol" x="{fx(s_):.1f}" y="{ht}" width="{cw:.1f}" height="{hb - ht}"/>')
    # Boss 所在格：竖虚线贯穿分布区，顶上一个迷你骷髅头；连打两个时只在第一个旁边写字
    small = cw < 12
    for s_, txt, first_b, last_b in bosses:
        x = fx(s_) + cw / 2
        g.append(f'<line class="ref" x1="{x:.1f}" x2="{x:.1f}" y1="{ht - 8}" y2="{ty + th}"/>')
        if last_b or not small:
            g.append(skull(x, ht - 20))
        if first_b:
            g.append(f'<text class="reft" x="{x - 9:.1f}" y="{ht - 15}" text-anchor="end">{esc(txt)}</text>')
    # 层数分布（柱）
    n_runs = max(1, len(pos))
    for s_ in range(1, WIN + 1):
        c = counts[s_]
        if not c:
            continue
        x = fx(s_) + min(2, cw * 0.15); bw = cw - 2 * min(2, cw * 0.15)
        y = hb - (hb - ht) * c / top
        r = min(3, bw / 2)
        g.append(f'<path class="hbar{" hx" if s_ in extra else ""}{" hwin" if s_ == WIN else ""}" d="M{x:.1f},{hb} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} '
                 f'Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{hb} Z"><title>{esc(info[s_])}：{c} 局（{c / n_runs:.0%}）</title></path>')
    g.append(f'<line class="axis" x1="{L}" x2="{w - R}" y1="{hb}" y2="{hb}"/>')
    # 进度条：三幕分段底色，加长层浅色 + 平均层数填充
    base = 0
    for i, (name, n, e, nb, _) in enumerate(LAYOUT):
        x0, x1 = fx(base + 1), fx(base + n + e) + cw
        g.append(f'<rect x="{x0 + (1 if i else 0):.1f}" y="{ty}" width="{x1 - x0 - (1 if i else 0) - 1:.1f}" height="{th}" '
                 f'rx="{6 if i == 0 else 0}" fill="var(--act{i + 1})"/>')
        if e:
            g.append(f'<rect class="xseg" x="{fx(base + n - nb + 1):.1f}" y="{ty}" width="{e * cw:.1f}" height="{th}">'
                     f'<title>{esc(name)}加长层：地图多两行（两排宝箱）时才有</title></rect>')
        base += n + e
    # 最右边的“通关”格
    g.append(f'<rect class="winseg" x="{fx(WIN):.1f}" y="{ty}" width="{cw:.1f}" height="{th}" rx="6"><title>通关：{wins} 局（{wins / max(1, len(pos)):.0%}）</title></rect>')
    g.append(f'<text class="wint" x="{fx(WIN) + cw / 2:.1f}" y="{ty + 19}" text-anchor="middle">★</text>')
    g.append(f'<text class="reft" x="{fx(WIN) + cw / 2:.1f}" y="{ht - 15}" text-anchor="middle">通关</text>')
    fl_all = sorted(f for e in recent for f in e["floors"])
    mean = sum(fl_all) / len(fl_all) if fl_all else last["floor_mean"]
    p90 = fl_all[int(0.9 * (len(fl_all) - 1))] if fl_all else last["floor_p90"]
    g.append(f'<clipPath id="{cid}"><rect x="{L}" y="{ty}" width="{w - L - R}" height="{th}" rx="6"/></clipPath>')
    g.append(f'<rect class="fill" x="{L}" y="{ty}" width="{cx(mean) - L:.1f}" height="{th}" clip-path="url(#{cid})"/>')
    base = 0
    for name, n, e, nb, _ in LAYOUT:
        lx = fx(base + 1) + 10
        on = cx(mean) >= lx + 96               # 文字整段落在填充里时用反白
        g.append(f'<text class="actt{" on" if on else ""}" x="{lx:.1f}" y="{ty + 19}">{esc(name)}<tspan class="actr"> {n} 层{f"（加长 +{e}）" if e else ""}</tspan></text>')
        base += n + e
    # 标记：起点、平均、前 10%、最高（最高按最远那局所在的格子；旧评测没有 pos，按常规层数换算）
    far = max((max(slot_of(*p_) for p_ in (e.get("pos") or [legacy_pos(e["floor_max"])])), e["floor_max"]) for e in ev)
    if any(e.get("win_rate", 0) > 0 for e in ev):
        far = (WIN, "通关")              # 有局通关过：最高就是通关，不写层数（51 层之类的数反而让人看不懂）
    marks = [("起点", cx(first["floor_mean"]), first["floor_mean"], "m-start"), ("平均", cx(mean), mean, "m-mean"),
             ("前 10%", cx(p90), p90, "m-p90"), ("最高", L + (far[0] - 0.5) * cw if far[0] == WIN else L + far[0] * cw, far[1], "m-max")]
    xs = spread([x for _, x, _, _ in marks], 86, L + 30, w - R - 30)
    for (name, x, v, cls), lx in zip(marks, xs):
        g.append(f'<line class="mk {cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{ty - 4}" y2="{ty + th + 6}"/>')
        g.append(f'<path class="mk {cls}" d="M{x:.1f},{ty + th + 6} L{lx:.1f},{ty + th + 18}" fill="none"/>')
        val = v if isinstance(v, str) else f"{v:.1f}" if cls in ("m-start", "m-mean") else f"{v:.0f}"      # 平均带一位小数，前 10% 与最高是整数层
        g.append(f'<text class="mkt {cls}" x="{lx:.1f}" y="{ty + th + 34}" text-anchor="middle">{esc(name)} <tspan class="mkv">{val}</tspan></text>')
    g.append("</svg>")
    return "".join(g)


def hour_ago(ev: list[dict]) -> dict | None:
    """约 1 小时前的那次评测（没有就取第一次），用于“近 1 小时”的变化。"""
    if len(ev) < 2:
        return None
    t = ev[-1]["time"] - 3600
    older = [e for e in ev[:-1] if e["time"] <= t]
    return older[-1] if older else ev[0]


KPIS = [("floor_mean", "平均到达层数", "{:.1f}", 1.0, " 层", " / 49 层"),
        ("reach_act2", "打过第一幕 Boss", "{:.0f}%", 100.0, " 个百分点", ""),
        ("reach_act3", "打过第二幕 Boss", "{:.0f}%", 100.0, " 个百分点", ""),
        ("win_rate", "通关率", "{:.1f}%", 100.0, " 个百分点", "")]
KPI_K = 12        # “近 12 次”：单次评测 192 局，噪声大（打过第二幕 Boss 的标准误约 3.5 个百分点），看趋势用多次合计


def kpi_tiles(ev: list[dict]) -> str:
    """首屏的四个关键指标（2026-10-27 起）：大字为最近 12 次评测合计（单次 192 局噪声大），小字为最近一次；
    再给出与之前 12 次合计的差，以及迷你趋势线（每次评测一个点）。"""
    if not ev:
        return '<div class="empty" style="height:120px">等待第一次评测</div>'
    last = ev[-1]
    recent, before = ev[-KPI_K:], ev[-2 * KPI_K:-KPI_K]

    def pooled_v(es: list[dict], key: str) -> float | None:
        den = sum(e["runs"] for e in es)
        return sum(e[key] * e["runs"] for e in es) / den if den else None
    out = []
    for key, name, fmt, scale, unit, of in KPIS:
        r, b = pooled_v(recent, key), pooled_v(before, key)
        n = sum(e["runs"] for e in recent)
        # 有效数字（2026-10-27 起各加一位）：打过第一、二幕 Boss 一位小数，通关率不到 10% 时两位小数
        pd_ = (2 if r < 0.1 else 1) if key == "win_rate" else 1
        val = fmt.format(r * scale) if key == "floor_mean" else f"{r * scale:.{pd_}f}%"
        lv = last[key]
        ld = (2 if lv < 0.1 else 1) if key == "win_rate" else 1
        last_txt = f"{lv * 100:.{ld}f}%" + f"（{round(lv * last['runs'])} / {last['runs']} 局）"
        if key == "floor_mean":
            fl = [f for e in recent for f in e["floors"]]
            sub = f"近 {len(recent)} 次 {n:,} 局 · 中位 {statistics.median(fl):.0f} 层"
            last_txt = f"{lv:.1f} 层"
        else:
            sub = f"近 {len(recent)} 次 {round(r * n):,} / {n:,} 局"
        trend = (f'<div class="kt">最近一次 {last_txt}' + (f' · 比之前 {len(before)} 次 {delta_html(r, b, unit, scale, 1, "")}' if b is not None else "") + '</div>')
        out.append(f'<section class="card kpi"><div class="kl">{esc(name)}</div>'
                   f'<div class="kv" data-k="kpi-{key}">{val}<span class="of">{esc(of)}</span></div>'
                   f'<div class="ks">{esc(sub)}</div>{trend}{sparkline([e[key] for e in ev])}</section>')
    return "".join(out)


def milestones(ev: list[dict]) -> str:
    """里程碑：一行摘要（进度点、已达成数、下一个目标），完整列表折叠起来。"""
    # 每个里程碑首次达成的那次评测（下标），评测按版本先后排，下标先后就是达成时的步数先后
    first = [next((j for j, e in enumerate(ev) if ok(e)), None) for _, ok, _, _ in MILESTONES]
    order = sorted(range(len(MILESTONES)), key=lambda i: (first[i] is None, first[i] or 0, i))
    states = [ev[first[i]] if first[i] is not None else None for i in order]
    out, nxt_marked, nxt, dots = [], False, None, []
    for i, ((name, ok, cur, target), hit) in enumerate(zip([MILESTONES[i] for i in order], states)):
        if hit:
            cls, icon = "done", "✓"
            body = f'<div class="msw">第 {hit["version"]:,} 版 · {when(hit)}</div><div class="msw">{cn_count(hit.get("total_steps"))} 步</div>'
        elif not nxt_marked and ev:
            nxt_marked = True
            cls, icon = "next", str(i + 1)
            nxt = (name, cur(ev[-1]), target)
            body = f'<div class="msw">现在 <b>{esc(cur(ev[-1]))}</b></div><div class="msw">目标 {esc(target)}</div>'
        else:
            cls, icon, body = "todo", str(i + 1), f'<div class="msw">目标 {esc(target)}</div>'
        dots.append(f'<i class="{cls}" title="{esc(name)}"></i>')
        out.append(f'<div class="ms {cls}"><div class="msdot">{icon}</div><div class="msn">{esc(name)}</div>{body}</div>')
    done = sum(1 for s_ in states if s_)
    last_done = max((s_ for s_ in states if s_), key=lambda e: e["version"], default=None)
    nxt_html = (f'<span class="msnext">下一个：<b>{esc(nxt[0])}</b>　现在 {esc(nxt[1])}，目标 {esc(nxt[2])}</span>' if nxt else
                '<span class="msnext">全部达成</span>')
    recent_html = (f'<span class="msw">最近达成：{esc(MILESTONES[order[states.index(last_done)]][0])}（第 {last_done["version"]:,} 版）</span>'
                   if last_done else "")
    return (f'<details class="sec msbox card" data-sec="ms"><summary><span class="ptitle">里程碑</span>'
            f'<span class="msdots">{"".join(dots)}</span><span class="psub" data-k="ms">已达成 {done} / {len(MILESTONES)}</span>'
            f'{nxt_html}{recent_html}<span class="more">全部</span></summary>'
            f'<div class="msrow">{"".join(out)}</div></details>')


def sparkline(vals: list[float], w: int = 120, h: int = 28) -> str:
    vals = [v for v in vals if v is not None][-60:]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    pts = " ".join(f"{i / (len(vals) - 1) * (w - 4) + 2:.1f},{h - 3 - (v - lo) / (hi - lo) * (h - 6):.1f}" for i, v in enumerate(vals))
    return f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}"/></svg>'


# ---------------------------------------------------------------- 多机状态与对照实验

def change_text(ch: dict, short: bool = False) -> str:
    change = ch.get("change") or {}
    if not isinstance(change, dict):
        return str(change)

    practice = {"boss_fraction", "boss_real_share", "act1_fraction", "act2_fraction", "act3_fraction", "mix_fraction",
                "act1_elite_share", "act2_elite_share", "act3_elite_share", "boss_act_probs", "second_boss_share"}
    names = {"boss_fraction": "Boss 名额", "boss_real_share": "Boss 用真实快照", "act1_fraction": "第一幕真实局面", "act2_fraction": "第二幕真实局面",
             "act3_fraction": "第三幕真实局面", "mix_fraction": "随机构筑", "act2_elite_share": "第二幕其中精英",
             "act1_elite_share": "第一幕其中精英", "act3_elite_share": "第三幕其中精英", "second_boss_share": "第三幕第二个 Boss"}
    other_names = {"upgrade_coef": "升级系数", "relic_coef": "遗物系数", "act3_boss_reward": "第三幕 Boss 奖励", "act3_boss2_reward": "第二个 Boss 奖励", "model": "模型", "epochs": "每批训练遍数",
                   "potion_coef": "药水系数", "combat_dmg_coef": "非 Boss 战掉血系数", "lr": "学习率", "seed": "随机种子",
                   "deck_over_coef": "牌组超限系数"}
    parts = []
    pk = [k for k in change if k in practice]
    if pk:      # 战斗练习份额：短写法只写“挪练习份额”，长写法列出各项的新值
        if short:
            parts.append("挪练习份额")
        else:
            bits = [f'{names[k]} ' + (" : ".join(f"{x:.0%}" for x in change[k]) if isinstance(change[k], list) else f'{change[k]:.0%}') for k in pk if k in names]
            if "boss_act_probs" in change:
                bits.append("Boss 各幕 " + " : ".join(f"{x:.0%}" for x in change["boss_act_probs"]))
            parts.append("练习份额：" + "，".join(bits))

    def one(k, v) -> str:
        if k == "run_start_probs" and isinstance(v, list) and len(v) == 3:
            if short:
                return "中途开局 {:.0f} : {:.0f} : {:.0f}".format(*(100 * x for x in v))
            return "中途开局（整局从第一层 {:.0%}、第二幕 {:.0%}、第三幕 {:.0%} 开）".format(*v)
        if k in other_names:
            return f"{other_names[k]} {v}"
        return f"{k} = {json.dumps(v, ensure_ascii=False)}"
    parts += [one(k, v) for k, v in change.items() if k not in practice]
    return ("＋" if short else "；").join(parts)


def mid_start_table(recs: list[dict]) -> str:
    """中途开局的统计：按开局位置的局数、实际占比、平均到达层数、通关（开幕快照池刚开始是空的，起初全部从第一层开）。"""
    n0 = sum(r.get("run_episodes") or 0 for r in recs)
    f0 = sum((r.get("run_floor_mean") or 0) * (r.get("run_episodes") or 0) for r in recs)
    w0 = sum((r.get("run_win_rate") or 0) * (r.get("run_episodes") or 0) for r in recs)
    mid = {a: [0, 0, 0, 0] for a in ("1", "2")}
    for r in recs:
        for a, v in (r.get("mid_runs") or {}).items():
            if a in mid:
                mid[a] = [x + y for x, y in zip(mid[a], v)]
    tot = n0 + sum(v[0] for v in mid.values())
    if not tot:
        return ""
    rows = [("第一层", n0, f0 / n0 if n0 else None, round(w0))] + [
        (f"第{'二三'[int(a) - 1]}幕开头", v[0], v[2] / v[0] if v[0] else None, v[3]) for a, v in mid.items()]
    body = "".join(f'<tr><td>{name}</td><td>{n:,}</td><td>{n / tot:.0%}</td><td>{"—" if f is None else f"{f:.1f}"}</td><td>{w:,}</td></tr>'
                   for name, n, f, w in rows)
    return ('<div class="ptitle" style="margin-top:12px">中途开局</div>'
            '<div class="psub">对照线分叉后的训练整局（带探索）按开局位置统计；开幕快照池刚开始是空的，起初会全部从第一层开</div>'
            f'<table class="cmp"><thead><tr><th>开局位置</th><th>局数</th><th>占比</th><th>平均到达层数</th><th>通关</th></tr></thead><tbody>{body}</tbody></table>')


def ep_counts(recs: list[dict]) -> tuple[int, int]:
    """训练记录里累计的（整局数, 单场战斗数）；整局含中途开局（mid_runs）。"""
    return (sum((r.get("run_episodes") or 0) + sum(v[0] for v in (r.get("mid_runs") or {}).values()) for r in recs),
            sum((r.get("combat_episodes") or 0) + (r.get("boss_episodes") or 0) for r in recs))


def ledger(src: Sources, log: list[dict], now: float) -> tuple[list[float], dict[str, list[float]]]:
    """累计训练量 [决策数, 整局数, 战斗场数]：全部（主线的完整历史 + 每条对照线分叉之后的部分）与各机器（在这台机器上实际练的）。
    主线的记录按训练线历史里迁走的时刻归到当时的机器；对照线：已结束的从本机存档读（fork.sh --end 存），在跑的读拉回的目录，
    都只算分叉时刻之后、扣掉分叉点之前的决策数。"""
    total, per = [0.0, 0.0, 0.0], {}

    def add(h: str, d: float, runs: float, combats: float) -> None:
        v = per.setdefault(h, [0.0, 0.0, 0.0])
        for i, x in enumerate((d, runs, combats)):
            v[i] += x; total[i] += x
    ends = sorted((e["ended"], e["host"]) for e in src.history if e.get("role") == "main" and e.get("host") and e.get("ended"))
    # 被主线采纳的对照线（adopt.sh）：它分叉后的记录已拼进主线日志，那段时间实际在对照主机上练
    adopted = [(e.get("branch_time") or 0, e["ended"], e["host"]) for e in src.history if e.get("adopted") and e.get("ended")]
    prev = 0
    for r in log:
        ts = r.get("total_steps") or 0
        t_ = r.get("time", 0)
        h = next((hh for b0, b1, hh in adopted if b0 <= t_ < b1), None) or next((hh for t, hh in ends if t_ < t), src.main)
        add(h, max(0, ts - prev), *ep_counts([r])); prev = ts
    st = read_json(src.dir(src.main) / "status.json")
    if st.get("total_steps", 0) > prev:      # status 比最近一条记录新一点
        add(src.main, st["total_steps"] - prev, 0, 0)

    def base_steps(bv: int) -> int:
        return next((r.get("total_steps") or 0 for r in reversed(log) if r.get("version", 0) <= bv), 0)
    segs = [(e["host"], e.get("branch_version"), e.get("branch_time") or 0, e.get("ended") or 9e18, read_jsonl(src.remote / e["archive"] / "log.jsonl"))
            for e in src.history if e.get("role") in ("challenger", "discarded") and e.get("archive") and not e.get("adopted")]
    if src.ch:
        segs.append((src.ch_host, src.ch.get("branch_version"), src.ch.get("branch_time") or 0, 9e18, src.jsonl(src.ch_host, "log.jsonl")))
    for h, bv, t0, t1, recs in segs:
        recs = rewind([r for r in recs if t0 <= r.get("time", 0) < t1])
        if not recs or bv is None:
            continue
        add(h, max(0, (recs[-1].get("total_steps") or 0) - base_steps(int(bv))), *ep_counts(recs))
        if h == src.ch_host and src.ch and t0 == (src.ch.get("branch_time") or 0):
            cst = read_json(src.dir(h) / "status.json")
            if cst.get("total_steps", 0) > (recs[-1].get("total_steps") or 0):
                add(h, cst["total_steps"] - recs[-1]["total_steps"], 0, 0)
    return total, per


COLLECTOR_FRESH_S = 600


def collector_status(src: Sources, h: str, now: float) -> dict | None:
    """协同采集机（tools/remote/remote_actors.py）写的 collector.json；10 分钟内更新过才算在跑。"""
    st = read_json(src.dir(h) / "collector.json")
    return st if st and now - st.get("time", 0) < COLLECTOR_FRESH_S else None


def remote_rate(log: list[dict], batch: int, k: int = 8) -> float | None:
    """最近 k 版里，学习用到的步数中来自协同采集机的部分折成每秒决策数（日志的 remote_used，2026-10-28 起）。"""
    rs = [r for r in log[-k:] if "remote_used" in r and r.get("sps")]
    if not rs:
        return None
    return statistics.mean(r["sps"] * r["remote_used"] / max(1, batch) for r in rs)


def lines_panel(src: Sources, now: float, per_host: dict[str, list[float]], log: list[dict] | None = None, batch: int = 32768) -> tuple[str, list[dict], dict[str, str]]:
    """各主机一张小卡：在跑哪条线、机器、当前版本、每秒决策数、上次同步（页面上每秒更新）；对照线另写改了什么、分叉后练了多少版。
    只有一台主机且没有对照时不显示。"""
    hs = [h for h, v in src.hosts.items() if not v.get("pending")]
    if len(hs) < 2 and not src.ch:
        return "", [], {}
    hs.sort(key=lambda h: (h != src.main, h != src.ch_host, h))     # 主线在前
    out, live, cards = [], [], {}
    for h in hs:
        st = read_json(src.dir(h) / "status.json")
        age = now - st.get("time", 0) if st else None
        state = "idle" if age is None else ("ok" if age < 300 else ("warn" if age < 1800 else "bad"))
        role = "主线" if h == src.main else ("对照" if h == src.ch_host else "空闲")
        co = collector_status(src, h, now) if role == "空闲" else None
        if co:      # 协同采集机：速度是它送来、被学习用上的决策数；版本是它手上的权重版本
            role = "协同采集"
            rr = remote_rate(log or [], batch)
            ok = co.get("connected") and now - co.get("weights_t", 0) < 120
            state = "ok" if ok else "warn"
            speed = f'{rr:,.0f}<small> 决策/秒</small>' if rr is not None else "<small>等待数据</small>"
            ver = f'权重第 {co.get("version", 0):,} 版'
            note = (f'{co.get("alive", 0)}/{co.get("actors", 0)} 个采集进程 × {co.get("envs", 0)} 个引擎 · 已送 {cn_count_k(co.get("sent", 0))}段轨迹 · '
                    f'{co.get("sent_mb", 0) / 1024:.1f} GB' + ("" if co.get("connected") else " · <b>未连上学习机</b>"))
            if h in per_host:
                d, rn, cb = per_host[h]
                note += f" · 以前在这台机器上练过 {cn_count(d)} 次决策"
            live.append({"id": h, "pull_t": pull_time(src.remote, h)})
            cards[h] = (f'<div class="line"><div class="lh"><span class="role">{role}</span>'
                        f'<span class="pill {state}"><i></i>{esc(host_label(src.hosts, h))}</span></div>'
                        f'<div class="lv">{speed}</div><div class="lm"><span class="ch">{ver}</span><span class="ch">同步 <span data-live="hsync-{esc(h)}">—</span></span>'
                        f'<span>{note}</span></div></div>')
            out.append(cards[h])
            continue
        if st:
            speed = f'{st.get("sps", 0):,.0f}<small> 决策/秒</small>' if state == "ok" else f'<small>{fmt_dur(age)}未更新</small>'
            ver = f'第 {st.get("version", 0):,} 版'
        else:
            speed, ver = "<small>尚无数据</small>", ""
        if role == "对照" and src.ch:
            bv = int(src.ch["branch_version"])
            note = f'改动 <b>{esc(change_text(src.ch, short=True))}</b> · 分叉后 {max(0, st.get("version", bv) - bv):,} 版' if st else ""
        elif role == "主线" and src.ch and st:
            note = f'分叉后 {max(0, st.get("version", 0) - int(src.ch["branch_version"])):,} 版'
        else:
            note = ""
        if role == "主线" and st and state == "ok":
            rr = remote_rate(log or [], batch)
            if rr:
                role = "主线 · 学习 + 采集"
                note = (note + " · " if note else "") + f"其中本机采集约 {max(0, st.get('sps', 0) - rr):,.0f}、协同采集约 {rr:,.0f} 决策/秒"
        if h in per_host:      # 在这台机器上实际练过的（不含分叉前继承来的）
            d, rn, cb = per_host[h]
            note += ("" if not note else " · ") + f"本机累计 {cn_count(d)} 次决策、{cn_count_k(rn)}局整局、{cn_count_k(cb)}场战斗"
        live.append({"id": h, "pull_t": pull_time(src.remote, h)})
        cards[h] = (f'<div class="line {"main" if h == src.main else ""}"><div class="lh"><span class="role">{role}</span>'
                   f'<span class="pill {state}"><i></i>{esc(host_label(src.hosts, h))}</span></div>'
                   f'<div class="lv">{speed}</div><div class="lm"><span class="ch">{ver}</span><span class="ch">同步 <span data-live="hsync-{esc(h)}">—</span></span>'
                   f'{f"<span>{note}</span>" if note else ""}</div></div>')
        out.append(cards[h])
    return f'<div class="lines">{"".join(out)}</div>', live, cards


def ev_metrics(e: dict) -> dict:
    """一次评测的几个对比指标，(分子, 分母) 形式，便于按局数合计。第三幕 Boss 为评测里第三幕各场 Boss 战（含连打的第二个）的打赢场数 / 场数。"""
    n = e.get("runs") or 0
    b3 = [v for k, v in (e.get("bosses") or {}).items() if k.startswith("2:")]
    return {"floor": (e["floor_mean"] * n, n), "act3": (e["reach_act3"] * n, n), "win": (e["win_rate"] * n, n),
            "b3": (sum(w for w, _ in b3), sum(m for _, m in b3)) if "bosses" in e else None}


CMP_METRICS = [("floor", "平均层数", "{:.1f}", 1.0, " 层"), ("act3", "打过第二幕 Boss", "{:.1f}%", 100.0, " 个百分点"),
               ("b3", "第三幕 Boss 胜率", "{:.1f}%", 100.0, " 个百分点"), ("win", "通关率", "{:.1f}%", 100.0, " 个百分点")]
CMP_WINDOW = 200      # 对照曲线每个点合计前多少版内的评测
CMP_SEG = 500         # 合计对比表每段的版本数


def pool_metric(es: list[tuple[int, dict]], key: str) -> float | None:
    xs = [m[key] for _, m in es if m[key] is not None]
    den = sum(b for _, b in xs)
    return sum(a for a, _ in xs) / den if den else None


def challenger_section(src: Sources, log: list[dict], ev: list[dict], now: float) -> str:
    """对照实验：主线与对照按分叉后的版本数对齐，比较评测（曲线 + 每 500 版合计）。没有对照时返回空串。"""
    ch = src.ch
    if not ch:
        return ""
    bv = int(ch["branch_version"])
    ch_log = rewind(src.ch_recs("log.jsonl"))
    ch_ev = clean_eval(src.ch_recs("eval.jsonl"))
    # 对照的版本号按约定从分叉版本接着记；万一是从 0 重新数的，就直接当作分叉后版本数
    ch_base = bv if not ch_log or ch_log[0].get("version", 0) >= bv else 0
    lines = [("主线", src.main, [(e["version"] - bv, ev_metrics(e)) for e in ev if e["version"] > bv],
              (log[-1]["version"] - bv) if log else 0, "--s1"),
             ("对照", src.ch_host, [(e["version"] - ch_base, ev_metrics(e)) for e in ch_ev if e["version"] - ch_base > 0],
              (ch_log[-1]["version"] - ch_base) if ch_log else 0, "--s2")]
    anchor = next((e for e in reversed(ev) if e["version"] <= bv), None)      # 分叉点（或之前最近）的那次主线评测：两条线共同的起点
    a_m = ev_metrics(anchor) if anchor else None

    def series(key: str, scale: float, fmt: str) -> list[dict]:
        out = []
        for name, h, es, _, var in lines:
            xs, ys = [], []
            if a_m and a_m[key] is not None and a_m[key][1]:
                xs.append(0); ys.append(scale * a_m[key][0] / a_m[key][1])
            j = 0
            for i, (x, _) in enumerate(es):
                while es[j][0] <= x - CMP_WINDOW:
                    j += 1
                v = pool_metric(es[j:i + 1], key)
                if v is not None:
                    xs.append(x); ys.append(scale * v)
            out.append({"name": f"{name} · {host_label(src.hosts, h)}", "short": name, "var": var, "xs": xs, "ys": ys, "markers": True,
                        "fmt": fmt, "bold": name == "主线"})
        return out

    def chart(key: str, title: str, fmt: str, scale: float, mobile: bool) -> str:
        size = dict(w=400, h=260, max_pts=MAX_PTS // 2) if mobile else dict(w=460, h=250)
        return line_chart(f"对照实验 · {title}", series(key, scale, fmt), **size, y_fmt=fmt, x_kind="ver")

    charts = [
        f'<section class="card"><div class="ptitle">{esc(title)}</div><div class="psub">每点为前 {CMP_WINDOW} 版内的评测合计，横轴为分叉后版本数</div>'
        f'<div class="dv">{chart(key, title, fmt, scale, False)}</div><div class="mv">{chart(key, title, fmt, scale, True)}</div></section>'
        for key, title, fmt, scale, _ in CMP_METRICS]

    # 每 CMP_SEG 版一段的合计对比；至少列到 1,500 版，哪条线练得更远就跟着往后加
    far = max([1500] + [es[-1][0] for _, _, es, _, _ in lines if es])
    rows = []
    for lo in range(0, far, CMP_SEG):
        hi = lo + CMP_SEG
        seg = [[x for x in es if lo < x[0] <= hi] for _, _, es, _, _ in lines]
        vals = [[pool_metric(sg, key) for key, *_ in CMP_METRICS] for sg in seg]
        for k, ((name, h, _, _, var), sg, vs) in enumerate(zip(lines, seg, vals)):
            head = f'<td rowspan="3">{lo:,}–{hi:,}</td>' if k == 0 else ""
            cells = "".join(f'<td>{"—" if v is None else fmt.format(v * scale)}</td>' for v, (_, _, fmt, scale, _) in zip(vs, CMP_METRICS))
            rows.append(f'<tr>{head}<td class="cl"><i style="background:var({var})"></i>{name}</td><td>{len(sg)}</td>'
                        f'<td>{sum(m["floor"][1] for _, m in sg):,}</td>{cells}</tr>')
        diff = "".join(f'<td>{delta_html(b, a, unit, scale, 1) or "—"}</td>' for a, b, (_, _, _, scale, unit) in zip(*vals, CMP_METRICS))
        rows.append(f'<tr class="cdiff"><td class="cl">差值（对照 − 主线）</td><td></td><td></td>{diff}</tr>')
    table = ('<div class="tscroll"><table class="cmp"><thead><tr><th>分叉后版本</th><th>线</th><th>评测次数</th><th>局数</th>'
             + "".join(f'<th>{esc(t)}</th>' for _, t, *_ in CMP_METRICS) + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
             '<p class="note">“0–500”指分叉后第 1–500 版，以此类推（不含分叉点那次评测），各指标按局数合计。评测每次约 192 局，'
             '一段里只有几次评测时，差几个百分点多半是噪声。</p>')

    # 摘要：两条线都练到的范围（分叉后 1 到 common 版）内各自的评测合计，差值是对照减主线
    common = min((es[-1][0] if es else 0) for _, _, es, _, _ in lines)
    pools = [[x for x in es if 0 < x[0] <= common] for _, _, es, _, _ in lines]
    overlap = all(pools)
    if not overlap:      # 刚分叉时两条线的评测版本区间还没重叠：先各用分叉后的全部评测，注明范围不同
        pools = [list(es) for _, _, es, _, _ in lines]
    srows = []
    for key, title, fmt, scale, unit in CMP_METRICS:
        a_, b_ = (pool_metric(p_, key) for p_ in pools)
        # 比例类指标给差值的标准误：分母按各自的口径（到达率、通关率按局数，第三幕 Boss 胜率按 Boss 战场数）
        se = ""
        if key != "floor" and a_ is not None and b_ is not None:
            na, nb = (sum(m[key][1] for _, m in p_ if m[key] is not None) for p_ in pools)
            if na and nb:
                se = f' <span class="meta">± {scale * math.sqrt(a_ * (1 - a_) / na + b_ * (1 - b_) / nb):.1f}</span>'
        srows.append(f'<tr><td>{esc(title)}</td>' + "".join(f'<td>{"—" if v is None else fmt.format(v * scale)}</td>' for v in (a_, b_))
                     + f'<td>{delta_html(b_, a_, unit, scale, 1) or "—"}{se}</td></tr>')
    n_ev = [len(p_) for p_ in pools]
    summary = ('<table class="cmp sum"><thead><tr><th>指标</th>' + "".join(
        f'<th><i style="background:var({var})"></i>{name}</th>' for name, _, _, _, var in lines) + '<th>差值</th></tr></thead>'
               f'<tbody>{"".join(srows)}</tbody></table>'
               + (f'<p class="note">分叉后第 1–{common:,} 版（两条线都练到的范围），' if overlap else
                  '<p class="note">两条线分叉后的评测版本区间还没有重叠，暂时各用分叉后的全部评测，范围不同、只作参考；')
               + f'主线评测 {n_ev[0]} 次、对照 {n_ev[1]} 次，按局数合计。'
               '评测每次约 192 局，次数少时差几个百分点多半是噪声；两台机器吞吐不同，按分叉后版本数对齐，不按时间。</p>'
               ) if all(pools) else '<div class="empty" style="height:90px">等两条线在分叉后都有评测</div>'
    bt = ch.get("branch_time")
    chips = [("改动", change_text(ch) or "—"), ("分叉点", f'主线第 {bv:,} 版' + (f' · {time.strftime("%m-%d %H:%M", time.localtime(bt))}' if bt else ""))]
    for name, h, es, n_ver, _ in lines:
        chips.append((f"{name} · {host_label(src.hosts, h)}", f"分叉后 {max(0, n_ver):,} 版 · 评测 {len(es)} 次"))
    chips_html = '<div class="cmpchips">' + "".join(f'<div><div class="lsl">{esc(k)}</div><div class="cmpv">{esc(v)}</div></div>' for k, v in chips) + '</div>'
    first2, rest2 = "".join(charts[:2]), "".join(charts[2:])

    # 分叉线自己的平均到达层数（与主线那张同样的四条线），横轴为分叉后版本数
    def floor_chart(mobile: bool) -> str:
        xe = [e["version"] - ch_base for e in ch_ev]
        xl = [r["version"] - ch_base for r in ch_log]
        p90_run = run_window(ch_log, 20)[0]
        top = max([e["floor_p90"] for e in ch_ev] + [x for x in p90_run if x] + [10])
        bands = [(0, 17, "第一幕", "--band1"), (17, 33, "第二幕", "--band2"), (33, MAX_FLOOR, "第三幕", "--band3")]
        size = dict(w=400, h=260, max_pts=MAX_PTS // 2) if mobile else dict(w=1200, h=300)
        return line_chart("分叉线 · 平均到达层数", [
            {"name": "评测平均（贪心，固定种子）", "short": "评测平均", "var": "--s1", "xs": xe, "ys": [e["floor_mean"] for e in ch_ev], "markers": True, "bold": True},
            {"name": "评测前 10%", "short": "前 10%", "var": "--s2", "xs": xe, "ys": [e["floor_p90"] for e in ch_ev], "markers": True},
            {"name": "训练对局平均（采样，第一层开局，最近 10 次更新滑动）", "short": "训练对局", "var": "--s3", "xs": xl,
             "ys": rolling([r.get("run_floor_mean") for r in ch_log], 10)},
            {"name": "训练对局前 10%（采样，第一层开局，最近 20 次更新合计）", "short": "训练前 10%", "var": "--s4", "xs": xl, "ys": p90_run},
            # 训练对局通关率按千分号画在同一纵轴上（纵轴读数即 ‰），用户要求（2026-10-26）
            {"name": "训练对局通关率（‰，纵轴读数即千分之几；第一层开局，最近 50 次更新合计）", "short": "训练通关 ‰", "var": "--s5", "xs": xl,
             "ys": [None if v is None else 10 * v for v in run_window(ch_log, 50)[1]], "fmt": "{:.1f}‰"},
        ], **size, y_fmt="{:.0f}", y_min=0, y_max=min(MAX_FLOOR, max(20, top * 1.25)), bands=bands, x_kind="ver")
    floor_card = (f'<div class="row"><section class="card"><div class="ptitle">分叉线 · 平均到达层数</div>'
                  f'<div class="psub">对照线（{esc(change_text(ch, short=True))}）自己的评测与训练对局，横轴为分叉后版本数；'
                  f'训练对局只算从第一层开的局，中途开局另见上表；训练通关率按千分号画在同一纵轴上（读数 5 即 5‰）。与主线的对比见上面两张图</div>'
                  f'<div class="dv">{floor_chart(False)}</div><div class="mv">{floor_chart(True)}</div></section></div>') if ch_ev or ch_log else ""
    mid_html = mid_start_table(ch_log) if any(r.get("mid_runs") for r in ch_log) or "run_start_probs" in (ch.get("change") or {}) else ""
    return (f'<div class="sechead"><h2>对照实验</h2><span class="psub">{esc(ch.get("note") or "")}</span></div>'
            f'<div class="row g3"><section class="card"><div class="ptitle">目前为止</div><div class="psub">对照线只改一项配置，其余与主线相同</div>'
            f'{chips_html}{summary}{mid_html}</section>{first2}</div>{floor_card}'
            f'<details class="sec" data-sec="cmp"><summary class="sechead sub"><h3>更多对比</h3><span class="psub">第三幕 Boss 胜率、通关率曲线，每 {CMP_SEG} 版分段合计表</span></summary>'
            f'<div class="row g3">{rest2}<section class="card"><div class="ptitle">分段合计对比</div>'
            f'<div class="psub">每 {CMP_SEG} 版一段，主线与对照各自的评测合计</div>{table}</section></div></details>')


MCTS_DIR = REMOTE / "defect" / "mcts"     # A 上 Boss 战搜索试验（tools/mcts_eval.py）的结果，pull.sh 从 A 的 ~/mcts 拉来


def mcts_runs(kind: str = "mcts") -> list[dict]:
    """搜索试验的各批（文件名 <批名>_s<分片>.jsonl / .meta.json），按开始时间排。每批：说明、结果行、各分片的题数。"""
    runs: dict[str, dict] = {}
    if not MCTS_DIR.exists():
        return []
    for m in MCTS_DIR.glob("*.meta.json"):
        name = m.name[: -len(".meta.json")]
        tag = name.rsplit("_s", 1)[0]
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("kind", "mcts") != kind:
            continue
        r = runs.setdefault(tag, {"tag": tag, "meta": meta, "items": 0, "total": 0, "start": meta.get("start", 0), "rows": [], "shards": 0})
        r["items"] += meta.get("items", 0); r["total"] = max(r["total"], meta.get("total", 0)); r["shards"] += 1
        if meta.get("start", 0) >= r["meta"].get("start", 0):
            r["meta"] = meta        # 改过参数接着跑时，以最近一次启动的说明为准
        r["start"] = min(r["start"], meta.get("start", r["start"]))
        rows = read_jsonl(MCTS_DIR / f"{name}.jsonl") if (MCTS_DIR / f"{name}.jsonl").exists() else []
        r["rows"].extend(rows)
    for r in runs.values():
        if r["total"]:
            r["items"] = r["total"]     # 新版说明文件记了全批题数（分片数改过也不会重复计）
    return sorted(runs.values(), key=lambda r: r["start"])


def mcts_stats(rows: list[dict]) -> dict:
    """按题配对：贪心与搜索的胜率、差（搜索 − 贪心）与标准误、不一致的题数、赢了时的剩余血量、每题推演次数。"""
    n = len(rows)
    if not n:
        return {"n": 0}
    g = [int(r["greedy"]["win"]) for r in rows]
    sr = [int(r["search"]["win"]) for r in rows]
    d = [b - a for a, b in zip(g, sr)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(n - 1, 1))
    hp = lambda k: [r[k]["hp"] / max(1, r[k]["max_hp"]) for r in rows if r[k]["win"]]
    hg, hs = hp("greedy"), hp("search")
    return {"n": n, "g": sum(g) / n, "s": sum(sr) / n, "d": m, "se": sd / math.sqrt(n),
            "s_only": sum(1 for x in d if x > 0), "g_only": sum(1 for x in d if x < 0),
            "hp_g": sum(hg) / len(hg) if hg else None, "hp_s": sum(hs) / len(hs) if hs else None,
            "sims": sum(r["search"].get("sims", 0) for r in rows) / n,
            "searched": sum(r["search"].get("searched", 0) for r in rows) / max(1, sum(r["search"].get("decisions", 0) for r in rows))}


DISTILL = HOME / ".local/share/sls2/distill"       # 搜索蒸馏的本机中转目录（tools/distill/relay.py）


def distill_section(now: float, log: list[dict]) -> str:
    """A 产搜索标签、主线学（2026-10-27 起，决策记录 131）：产出速度、改选比例、A 用的检查点、主线缓冲，
    以及新标签进训练之前、当前网络的首选与搜索的选择一致的比例（改选处的一致率上升，说明网络在没见过的局面上学到了搜索的选择）。"""
    summ = []
    try:
        summ = [json.loads(l) for l in (DISTILL / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    except (OSError, ValueError):
        pass
    try:
        st = json.loads((DISTILL / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}

    def tile(name, val, sub):
        return f'<section class="card kpi"><div class="kl">{esc(name)}</div><div class="kv">{val}</div><div class="ks">{esc(sub)}</div></section>'
    h1 = [x for x in summ if x["t"] >= now - 3600]
    h3 = [x for x in summ if x["t"] >= now - 3 * 3600]
    tot = sum(x["labels"] for x in summ)
    lab3, ch3 = sum(x["labels"] for x in h3), sum(x["changed"] for x in h3)
    by = {k: [sum(x["by_act"][k][0] for x in h3), sum(x["by_act"][k][1] for x in h3)] for k in ("1", "2")}
    win_txt = "，".join(f'第{"一二三"[int(k)]}幕 Boss {pct(w / n if n else None, 0)}（{n} 题）' for k, (n, w) in by.items() if n)
    dl = [r for r in log if "distill_n" in r]
    last = dl[-1] if dl else {}
    tiles = (tile("标签产出", f'{sum(x["labels"] for x in h1):,}<span class="of">条/小时</span>',
                  f'最近一小时 {sum(x["problems"] for x in h1)} 道题；累计 {tot:,} 条')
             + tile("改选比例", pct(ch3 / lab3 if lab3 else None, 1), f"最近 3 小时：搜索过的决策里搜索换了动作的比例。搜索打赢：{win_txt or '—'}")
             + tile("A 用的检查点", f'第 {st["ckpt_version"]:,} 版' if st.get("ckpt_version") else "—",
                    f'主线每新出 1,000 版换一次；{time.strftime("%H:%M", time.localtime(st["ckpt_t"]))} 换上' if st.get("ckpt_t") else "等第一次换上")
             + tile("主线缓冲", f'{last["distill_n"][0]:,}<span class="of">条</span>' if last else "—",
                    f'其中改选 {last["distill_n"][1]:,} 条；已训练 {last.get("distill_steps", 0):,} 步' if last else "主线收到标签后显示"))
    # 新标签进训练前的一致率：按 200 版一档合计
    bins: dict[int, list[int]] = {}
    for r in dl:
        f_ = r.get("distill_fresh") or [0, 0, 0, 0]
        if f_[0] or f_[2]:
            b = bins.setdefault(r["version"] // 200 * 200, [0, 0, 0, 0])
            for j in range(4):
                b[j] += f_[j]
    pts = [(v, b[1] / b[0], b[0]) for v, b in sorted(bins.items()) if b[0] >= 20]
    if len(pts) >= 2:
        w, h = 520, 160
        x0, x1 = pts[0][0], pts[-1][0]
        xs = lambda v: 40 + (w - 60) * (v - x0) / max(1, x1 - x0)
        hi = max(0.3, max(y for _, y, _ in pts) * 1.15)
        ys = lambda y: 10 + (h - 34) * (hi - y) / hi
        path = " ".join(f"{'M' if i == 0 else 'L'}{xs(v):.1f},{ys(y):.1f}" for i, (v, y, _) in enumerate(pts))
        dots = "".join(f'<circle cx="{xs(v):.1f}" cy="{ys(y):.1f}" r="3.5" fill="var(--s1)"><title>第 {v:,}–{v + 199:,} 版：{100 * y:.1f}%（{n} 条改选标签）</title></circle>' for v, y, n in pts)
        grid = "".join(f'<line x1="40" x2="{w - 20}" y1="{ys(t):.1f}" y2="{ys(t):.1f}" stroke="var(--grid)"/><text x="34" y="{ys(t) + 4:.1f}" font-size="11" fill="var(--muted)" text-anchor="end">{100 * t:.0f}%</text>'
                       for t in [k / 10 for k in range(0, int(hi * 10) + 1, max(1, int(hi * 10) // 4))])
        svg = (f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto">{grid}<path d="{path}" fill="none" stroke="var(--s1)" stroke-width="2"/>{dots}'
               f'<text x="40" y="{h - 6}" font-size="11" fill="var(--muted)">第 {x0:,} 版</text><text x="{w - 20}" y="{h - 6}" font-size="11" fill="var(--muted)" text-anchor="end">第 {x1:,} 版</text></svg>')
    else:
        svg = '<div class="empty" style="height:160px">主线收到两档以上的新标签后显示</div>'
    same = [b for b in bins.values() if b[2]]
    same_txt = pct(sum(b[3] for b in same) / sum(b[2] for b in same), 1) if same else "—"
    agree_card = (f'<section class="card"><div class="ptitle">改选处的一致率 · 新标签进训练前</div><div class="psub">新到的标签先不训练，'
                  f'看当前网络的首选与搜索选的是否一样（产出标签的网络在这些地方选的都是别的，所以从 0 起步），每 200 版一档。'
                  f'没改选处的一致率（网络仍选原来那一项）合计 {same_txt}</div>{svg}</section>')
    acc = [r["distill_acc"][0] for r in dl[-100:] if r.get("distill_acc") and r["distill_acc"][0] is not None]
    loss = [r["distill_loss"] for r in dl[-100:] if r.get("distill_loss") is not None]
    chips = (f'<span class="chip"><b>交叉熵</b> {statistics.mean(loss):.3f}</span>' if loss else "") + \
            (f'<span class="chip"><b>改选标签命中</b> {pct(statistics.mean(acc), 1)}</span>' if acc else "")
    train_card = (f'<section class="card"><div class="ptitle">训练这一步</div><div class="psub">最近 100 版：每次 PPO 更新后抽 128 条训练一步，'
                  f'改选了的权重 1、没改选的 0.2（命中为训练这一步之前）</div>'
                  f'{f"<div class=chips>{chips}</div>" if chips else "<div class=empty style=height:60px>主线开始训练蒸馏后显示</div>"}'
                  '<p class="note">效果以固定考卷（第二、三幕 Boss 各 256 题，每 250 版一次）前后对比为准，见下方“固定考卷”。</p></section>')
    sub = ("A 用确定化搜索（价值估值叶子版：第三幕考卷 +3.5±1.7、第二幕 +6.2±1.7）打主线导出的第二、三幕 Boss 战开打局面（不含考卷的题），"
           "每个搜索过的决策记下搜索的选择；经本机转给主线，主线每次 PPO 更新后多训练一步，让策略往搜索的选择靠")
    return (f'<div class="blockhead bc" id="blk-mcts"><i></i><h2>A · 搜索蒸馏</h2><span class="meta">{esc(sub)}</span></div>'
            f'<div class="row kpis">{tiles}</div><div class="row">{agree_card}{train_card}</div>')


def a_section(now: float, log: list[dict] | None = None) -> str:
    """A 上的试验只显示正在做的那一个：搜索评测、牌组前瞻或搜索蒸馏（2026-10-27 用户：“看板上把前两个 A 的实验先清掉，只看当前的实验”；
    2026-10-28 起已结束的也不显示）。
    以前各批的结果仍在 A 的 ~/mcts 与本机拉取目录里，决策记录第 125 条有汇总。"""
    if (DISTILL / "ENABLED").exists():
        return distill_section(now, log or [])
    # 2026-10-28 用户：“看板上移去之前的A相关的实验的内容。”——已经结束的搜索评测、牌组前瞻不再显示，
    # 只有结果文件最近 A_ACTIVE_S 秒内还在更新（正在跑）时才显示
    if not MCTS_DIR.exists():
        return ""
    newest = max((f.stat().st_mtime for f in MCTS_DIR.glob("*.jsonl")), default=0.0)
    if now - newest > A_ACTIVE_S:
        return ""
    cand = [(r["start"], k) for k in ("mcts", "lookahead") for r in mcts_runs(k)[-1:]]
    if not cand:
        return ""
    return look_section(now) if max(cand)[1] == "lookahead" else mcts_section(now)


A_ACTIVE_S = 2 * 3600


def look_section(now: float) -> str:
    """A 上牌组前瞻的评测（tools/lookahead_eval.py）：每个种子打两遍整局（纯贪心、贪心 + 选牌前瞻），按种子配对比较。"""
    runs = mcts_runs("lookahead")
    if not runs:
        return ""
    cur = runs[-1]
    rows, meta = cur["rows"], cur["meta"]
    p = meta.get("params", {})
    n, total = len(rows), max(cur["items"], 1)
    ts = sorted(r["t"] for r in rows if "t" in r)
    rate = (n - 1) / max(ts[-1] - ts[0], 1) * 3600 if len(ts) > 1 else 0.0
    left = "已完成" if n >= total else (f"约 {(total - n) / rate:.1f} 小时后完成" if rate else "等待第一批结果")

    def paired(key: str, scale: float = 1.0) -> tuple[float, float, float, float]:
        g = [float(r["greedy"][key]) for r in rows]; l_ = [float(r["look"][key]) for r in rows]
        d = [y - x for x, y in zip(g, l_)]
        m = sum(d) / max(1, len(d)); sd = math.sqrt(sum((v - m) ** 2 for v in d) / max(1, len(d) - 1))
        return sum(g) / max(1, len(g)) * scale, sum(l_) / max(1, len(l_)) * scale, m * scale, sd / math.sqrt(max(1, len(d))) * scale

    def tile(name, val, sub):
        return f'<section class="card kpi"><div class="kl">{esc(name)}</div><div class="kv">{val}</div><div class="ks">{esc(sub)}</div></section>'
    if n:
        tiles = [tile("进度", f'{n}<span class="of">/ {total} 个种子</span>', f"{rate:.0f} 个/小时，{left}")]
        for key, name, sc, unit in (("floor", "平均层数", 1.0, "层"), ("reach2", "进第二幕", 100.0, "个百分点"), ("reach3", "进第三幕", 100.0, "个百分点"), ("win", "通关", 100.0, "个百分点")):
            g, l_, m, se = paired(key, sc)
            fmt = (lambda x: f"{x:.1f}") if key == "floor" else (lambda x: f"{x:.0f}%")
            tiles.append(tile(name, f'{"+" if m >= 0 else "−"}{abs(m):.1f}<span class="of">± {se:.1f} {unit}</span>', f"贪心 {fmt(g)} → 前瞻 {fmt(l_)}"))
        dec = sum(r["look"].get("decisions", 0) for r in rows); ch = sum(r["look"].get("changed", 0) for r in rows)
        tiles = "".join(tiles)
        sub = (f'每个种子打两遍整局（纯贪心、贪心 + 选牌前瞻），按种子配对；模型第 {meta.get("version", "?"):,} 版。前瞻：卡牌奖励的每个选项（拿某张牌或跳过）'
               f'用“当前牌组 + 这张牌”满血打本幕 Boss {p.get("n")} 场，与网络原本的选择逐个种子配对，平均差超过 {p.get("z")} 个标准误才改。'
               f'共 {dec:,} 次选牌、改了 {ch:,} 次（{pct(ch / dec if dec else None, 0)}）')
    else:
        tiles = tile("进度", f'0<span class="of">/ {total} 个种子</span>', left)
        sub = "牌组前瞻的评测"
    return (f'<div class="blockhead bc" id="blk-look"><i></i><h2>A · 选牌的牌组前瞻</h2><span class="meta">{esc(sub)}</span></div>'
            f'<div class="row kpis k5">{tiles}</div>')


def mcts_section(now: float) -> str:
    """A 上 Boss 战确定化搜索与贪心在固定考卷上的配对比较：进度、速度、胜率差、各 Boss 明细，以及之前各批的汇总。"""
    runs = mcts_runs()
    if not runs:
        return ""
    cur = runs[-1]
    rows, meta = cur["rows"], cur["meta"]
    st = mcts_stats(rows)
    p = meta.get("params", {})
    acts = "、".join("第" + "一二三"[a] + "幕" for a in meta.get("acts", []))
    ts = sorted(r["t"] for r in rows if "t" in r)
    done, total = len(rows), max(cur["items"], 1)
    t_last = ts[-1] if ts else max((f.stat().st_mtime for f in MCTS_DIR.glob(f"{cur['tag']}_s*.jsonl")), default=now)
    rate = done / max(t_last - cur["start"], 1) * 3600 if done else 0.0
    eta = (total - done) / rate * 3600 if rate and done < total else 0
    left = "已完成" if done >= total else (f"约 {eta / 3600:.1f} 小时后完成" if rate else "等待第一批结果")

    def tile(name, val, sub):
        return f'<section class="card kpi"><div class="kl">{esc(name)}</div><div class="kv">{val}</div><div class="ks">{esc(sub)}</div></section>'
    if st["n"]:
        sign = "+" if st["d"] >= 0 else "−"
        z = st["d"] / st["se"] if st["se"] else 0.0
        verdict = ("搜索明显更好" if z >= 2 else "搜索明显更差" if z <= -2 else "还分不出高下")
        tiles = (tile("进度", f'{done}<span class="of">/ {total} 题</span>', f"{rate:.0f} 题/小时，{left}")
                 + tile("贪心胜率", pct(st["g"], 1), f'赢时剩血 {pct(st["hp_g"], 0) if st["hp_g"] is not None else "—"}')
                 + tile("搜索胜率", pct(st["s"], 1), f'赢时剩血 {pct(st["hp_s"], 0) if st["hp_s"] is not None else "—"}')
                 + tile("差（搜索 − 贪心）", f'{sign}{abs(st["d"]) * 100:.1f}<span class="of">± {st["se"] * 100:.1f} 个百分点</span>',
                        f'{verdict}；搜索多赢 {st["s_only"]} 题、少赢 {st["g_only"]} 题'))
    else:
        tiles = tile("进度", f'0<span class="of">/ {total} 题</span>', left)
    # 各 Boss：贪心与搜索的胜率并排
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(f'{r["act"]}:Boss:{r["encounter"]}', []).append(r)
    brow = []
    for key, rs in sorted(by.items(), key=lambda x: (-len(x[1]), x[0])):
        s2 = mcts_stats(rs)
        name = enc_name(key)[0]
        brow.append(f'<div class="drow"><div class="dname"><b>{esc(name)}</b><small>{s2["n"]} 题</small></div>'
                    f'<div><div class="fbar" title="贪心"><i style="width:{100 * s2["g"]:.1f}%;background:var(--s4)"></i></div>'
                    f'<div class="fbar" style="margin-top:3px" title="搜索"><i style="width:{100 * s2["s"]:.1f}%"></i></div></div>'
                    f'<div class="dval">{pct(s2["g"], 0)} → {pct(s2["s"], 0)}<span class="fn">贪心 → 搜索</span></div></div>')
    boss_card = (f'<section class="card"><div class="ptitle">各 Boss 胜率</div><div class="psub">上条（橙）贪心、下条（蓝）搜索</div>'
                 f'<div class="funnel">{"".join(brow) or "<div class=empty>等待结果</div>"}</div></section>')
    # 累计差随题数的变化（看结论是否稳定）
    pts = []
    acc = 0
    for k, r in enumerate(sorted(rows, key=lambda r: r.get("t", 0)), 1):
        acc += int(r["search"]["win"]) - int(r["greedy"]["win"])
        pts.append((k, acc))
    if len(pts) >= 2:
        w, h = 520, 150
        xs = lambda k: 30 + (w - 40) * (k - 1) / max(1, len(pts) - 1)
        lo, hi = min(0, min(v for _, v in pts)), max(1, max(v for _, v in pts))
        ys = lambda v: 10 + (h - 30) * (hi - v) / max(hi - lo, 1)
        path = " ".join(f"{'M' if i == 0 else 'L'}{xs(k):.1f},{ys(v):.1f}" for i, (k, v) in enumerate(pts))
        svg = (f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto"><line x1="30" x2="{w - 10}" y1="{ys(0):.1f}" y2="{ys(0):.1f}" stroke="var(--muted)" stroke-dasharray="3 3"/>'
               f'<path d="{path}" fill="none" stroke="var(--s1)" stroke-width="2"/>'
               f'<text x="30" y="{h - 4}" font-size="11" fill="var(--muted)">第 1 题</text><text x="{w - 10}" y="{h - 4}" font-size="11" fill="var(--muted)" text-anchor="end">第 {len(pts)} 题</text>'
               f'<text x="26" y="{ys(0) + 4:.1f}" font-size="11" fill="var(--muted)" text-anchor="end">0</text>'
               f'<text x="{w - 10}" y="{ys(pts[-1][1]) - 6:.1f}" font-size="12" fill="var(--ink2)" text-anchor="end">{pts[-1][1]:+d}</text></svg>')
    else:
        svg = '<div class="empty" style="height:150px">等待结果</div>'
    trend_card = (f'<section class="card"><div class="ptitle">累计净胜题数</div><div class="psub">按完成先后：搜索赢而贪心输记 +1，反过来记 −1；在 0 线以上说明搜索领先</div>{svg}</section>')
    hist = ""
    sub = (f'固定考卷 {acts} Boss 题，模型第 {meta.get("version", "?"):,} 版；每题贪心、搜索各打一遍按题配对。搜索：网络最想选的不到 {p.get("tau")} 时，'
           f'取概率前 {p.get("k")} 的候选，在换了随机数、重洗了抽牌堆的“可能世界”里{f"贪心往下走 {p.get('leaf')} 步、用价值头估这个局面" if p.get("leaf") else "贪心打完比结果（" + ("输了按打掉的敌人血量给部分分" if p.get("score") == "dmg" else "只看输赢") + "）"}，逐轮减半（首轮每个 {p.get("sims0")} 次，共 {p.get("rounds")} 轮）{f"，与贪心那一项逐个世界配对比较、平均差超过 {p.get('z')} 个标准误才换" if p.get("paired") else "，选平均分最高的"}。'
           f'平均每题推演 {st.get("sims", 0):.0f} 次，{pct(st.get("searched", 0), 0)} 的决策用了搜索。批名 {cur["tag"]}，{cur["shards"]} 个进程')
    return (f'<div class="blockhead bc" id="blk-mcts"><i></i><h2>A · Boss 战搜索试验</h2><span class="meta">{esc(sub)}</span></div>'
            f'<div class="row kpis">{tiles}</div><div class="row">{trend_card}{boss_card}</div>{hist}')


# ---------------------------------------------------------------- 页面

def clean_eval(recs: list[dict]) -> list[dict]:
    ev = rewind(recs)
    ev = [e for i, e in enumerate(ev) if i + 1 == len(ev) or ev[i + 1].get("version") != e.get("version")]   # 同一版本只留最后一次
    for e in ev:        # 早期的战斗评测误为同一场战斗反复打，不作数
        if not e.get("combat_eval"):
            e["combat_win_rate"] = e["combat_hp_delta"] = None
    return ev


def drop_abandoned(ev: list[dict], raw_log: list[dict]) -> list[dict]:
    """A 上的评测记录没有训练量，rewind 管不到：按训练日志里的回退点（续训后版本号倒退）丢掉被覆盖那段时间线上的评测，
    即回退之前、版本号高于回退到的版本的那些（2026-11-01 回退到第 136,503 版后，作废时间线的第 136,542–136,951 版评测混在图里，线往回折）。
    之后再按版本号做一次单调化，同一条线上的点不再往回走。"""
    cuts, prev = [], None
    for r in raw_log:
        v = r.get("version")
        if prev is not None and v is not None and v <= prev:
            cuts.append((r.get("time", 0), v - 1))
        if v is not None:
            prev = v
    out: list[dict] = []
    for e in ev:
        if any(e.get("time", 0) < t and e.get("version", 0) > v0 for t, v0 in cuts):
            continue
        while out and out[-1].get("version", 0) >= e.get("version", 0):
            out.pop()
        out.append(e)
    return out


def build(src: Sources, shared: dict | None = None) -> str:
    """返回页面主体 HTML（#app），实时数字所需的数据放在它的 data-live 上。原有各面板都画主线（见“多机与训练线”）。"""
    _series_memo.clear()
    run = src.dir(src.main)
    raw_log = src.main_recs("log.jsonl")
    log = rewind(raw_log)
    ev = drop_abandoned([e for e in clean_eval(src.main_recs("eval.jsonl")) if e["time"] >= EVAL_SINCE_T], raw_log)
    if log:     # A 上的评测（决策记录 198）读的是采集用的权重，记录里没有训练量：按版本号从训练日志补上，否则点都落在横轴 0 处
        lv = [r["version"] for r in log]
        for e in ev:
            if not e.get("total_steps"):
                e["total_steps"] = log[min(bisect.bisect_left(lv, e["version"]), len(log) - 1)]["total_steps"]
    res = src.main_recs("resource.jsonl")
    st = read_json(run / "status.json")
    cfg = read_json(run / "config.json")
    wd = read_json(run / "watchdog.json")
    prog = read_json(src.eval_dir() / "eval_progress.json")
    now = time.time()
    note_p = src.remote / NOTE_NAME
    note = note_p.read_text(encoding="utf-8").strip() if note_p.exists() else ""
    mig = src.migration()
    changes = sorted(CHANGES + ([mig] if mig else []) + src.marks)
    # 开训时刻：迁移后取两台机器 status 里最早的 started_at（新机器若没有沿用旧的 status.json，也不会把迁移当成开训）
    started = min([x for x in (st.get("started_at"), src.prev and read_json(src.dir(src.prev) / "status.json").get("started_at")) if x] or [now])

    # ---- 状态
    age = now - st.get("time", 0) if st else 1e9
    if not st:
        state, state_txt = "idle", "尚无数据"
    elif age < 300:
        state, state_txt = "ok", "训练中"
    elif age < 1800:
        state, state_txt = "warn", f"{int(age // 60)} 分钟未更新"
    else:
        state, state_txt = "bad", f"已停止或失联（{fmt_dur(age)}未更新）"
    pull_t = pull_time(src.remote, src.main)
    pull_age = now - pull_t if pull_t is not None else None
    if st and age >= 300:
        if pull_age is not None and pull_age > 300:
            state_txt += f"；本机已 {fmt_dur(pull_age)}拉不到租用机数据"
        elif wd and wd.get("stop_flag"):
            state, state_txt = "idle", "已手动停止"

    last = ev[-1] if ev else None
    first = ev[0] if ev else None
    # 评测进程出错：报错文件比最近一次评测新（评测进程启动时会删掉旧的报错文件）
    alert = ""
    err_p = src.eval_dir() / "eval_error.txt"
    if err_p.exists() and (not last or err_p.stat().st_mtime > last["time"]):
        tail = [l for l in err_p.read_text(encoding="utf-8", errors="replace").strip().splitlines() if l.strip()][-1:]
        alert = (f'<div class="alert"><b>评测进程出错</b>，最近一次评测停在{f"第 {last['version']:,} 版（{when(last)}）" if last else "开训之前"}；'
                 f'{esc(tail[0][:160]) if tail else ""}</div>')
    ref = hour_ago(ev)
    sps = statistics.mean([r["sps"] for r in log[-8:]]) if log else 0
    # 评测进度：进行中的那一轮（结果已写出的不算）
    ev_prog = None
    if prog and (not last or prog.get("version") != last.get("version")) and prog.get("done", 0) < prog.get("total", 1):
        ev_prog = {k: prog.get(k) for k in ("version", "done", "total", "runs", "start", "done_runs", "done_combats")}
    if shared and "ledger" in shared:      # 账本是全局事实，对调视角沿用原视角算的
        all_tot, per_host = shared["ledger"]
    else:
        all_tot, per_host = ledger(src, log, now) if (src.ch or src.history) else ([st.get("total_steps") or 0, *ep_counts(log)], {})
        if shared is not None:
            shared["ledger"] = (all_tot, per_host)
    ch_st = read_json(src.dir(src.ch_host) / "status.json") if src.ch else {}
    ch_sps = ch_st.get("sps", 0) if ch_st and now - ch_st.get("time", 0) < 300 else 0
    live = {"steps": all_tot[0], "sps": (sps + ch_sps) if state == "ok" else 0, "t": st.get("time", now),
            "started": started if st else now, "pull_t": pull_t if pull_t is not None else now, "ev": ev_prog}
    hosts_html, host_live, host_cards = lines_panel(src, now, per_host, log, int(cfg.get("batch_steps") or 32768))
    # 协同采集机的资源采样（与主线机的画在同一张占用图上）
    co_hosts = [h for h, v in src.hosts.items() if h != src.main and not v.get("pending") and collector_status(src, h, now)]
    co_res = {h: [r for r in src.jsonl(h, "resource.jsonl") if r.get("time", 0) > now - 30 * 86400] for h in co_hosts}
    viewbar = ""
    if host_live:
        live["hosts"] = host_live

    # ---- 顶栏实时数字
    n_runs, n_combats = all_tot[1], all_tot[2]      # 分叉前的累计 + 分叉后各条线的；整局含中途开局
    stats = [
        ("训练量", f'<span data-live="steps">{cn_count(live["steps"])}</span><small> 次决策</small>'),
        ("已训练局数", f'<span class="ch">{cn_count_k(n_runs)}<small> 局整局</small></span><small class="sep"> · </small>'
                      f'<span class="ch">{cn_count_k(n_combats)}<small> 场战斗</small></span>'),
        *([] if hosts_html else [("速度", f'{sps:,.0f}<small> 决策/秒</small>')]),     # 双机时速度写在各训练线的卡片上
        ("已训练", f'<span data-live="elapsed">{fmt_dur(live["t"] - live["started"])}</span>'),
        ("评测", '<span data-live="eval">—</span><span class="evbar"><i data-live="evbar"></i></span>'),
        ("数据同步", '<span data-live="sync">—</span>'),
    ]
    topstats = "".join(f'<div class="ls"><div class="lsl">{esc(k)}</div><div class="lsv">{v}</div></div>' for k, v in stats)

    # ---- 首屏：四个关键指标 + 通关之路（层数分布）
    if last:
        rec = ev[-KPI_K:]
        jsub = (f'最近 {len(rec)} 次评测合计（第 {rec[0]["version"]:,}–{last["version"]:,} 版，{sum(e["runs"] for e in rec):,} 局固定种子整局，贪心）；'
                f'柱为停在各层的局数，进度条填到平均层数；最近一次第 {last["version"]:,} 版，{when(last)}')
    else:
        jsub = "等待第一次评测"
    hero = (f'<div class="row kpis">{kpi_tiles(ev)}</div>'
            f'<section class="card hero"><div class="ptitle">通关之路</div><div class="psub">{esc(jsub)}</div>'
            f'<div class="dv">{journey(ev)}</div><div class="mv">{journey(ev, w=420, cid="trackclip-m")}</div></section>')

    # ---- 曲线：每张图按时间范围各生成一份，页面上切换
    ex = [e.get("total_steps") or 0 for e in ev]; et = [e["time"] for e in ev]
    lx = [r["total_steps"] for r in log]; lt = [r["time"] for r in log]
    roll = lambda key, k, scale=1.0: memo(("roll", id(log), len(log), key, k, scale),
                                          lambda: rolling([None if r.get(key) is None else r[key] * scale for r in log], k), log)
    run_p90 = run_window(log, 20)[0]
    run_win_50 = run_window(log, 50)[1]
    top_floor = max([e["floor_p90"] for e in ev] + [r.get("run_floor_mean") or 0 for r in log] + [x for x in run_p90 if x] + [10])
    y_top = min(MAX_FLOOR, max(20, top_floor * 1.25))
    bands = [(0, 17, "第一幕", "--band1"), (17, 33, "第二幕", "--band2"), (33, MAX_FLOOR, "第三幕", "--band3")]
    reach_max = min(100, max(10, max([e["reach_act2"] * 100 for e in ev] + [0]) * 1.1))
    t_origin = (started if st.get("started_at") else None) or (res[0]["time"] if res else 0)
    # 分界线画在分界时刻的训练量处
    vrefs = [(next((r["total_steps"] for r in reversed(log) if r["time"] <= bt), 0), txt) for bt, txt in EVAL_BREAKS + changes if log and log[-1]["time"] > bt]

    health_t0 = next((r["time"] for r in log if r.get("health")), None)

    def charts(since: float | None, mobile: bool = False) -> dict[str, str]:
        # 手机版：约 400 宽的画布，缩到手机屏宽时文字仍接近原大；点数减半，控制页面体积
        m = dict(w=400, h=260, max_pts=MAX_PTS // 2) if mobile else {}
        ev_since = max(since or 0, EVAL_SINCE_T)     # 评测相关的两张图从评测口径的新起点画起，训练对局的线也一起对齐
        c = {}
        c["floor"] = line_chart("到达层数", [
            {"name": "评测平均（贪心，固定种子）", "short": "评测平均", "var": "--s1", "xs": ex, "ts": et, "ys": [e["floor_mean"] for e in ev], "markers": True, "bold": True},
            {"name": "评测前 10%", "short": "前 10%", "var": "--s2", "xs": ex, "ts": et, "ys": [e["floor_p90"] for e in ev], "markers": True},
            {"name": "训练对局平均（采样，最近 10 次更新滑动）", "short": "训练对局", "var": "--s3", "xs": lx, "ts": lt, "ys": roll("run_floor_mean", 10)},
            {"name": "训练对局前 10%（采样，最近 20 次更新合计）", "short": "训练前 10%", "var": "--s4", "xs": lx, "ts": lt, "ys": run_p90},
        ], **(m or dict(w=600, h=320)), y_fmt="{:.0f}", y_min=0, y_max=y_top, bands=bands, since=ev_since, vrefs=vrefs)
        c["reach"] = line_chart("通关率与各幕到达率", [
            {"name": "打过第一幕 Boss", "short": "第一幕", "var": "--s1", "xs": ex, "ts": et, "ys": [e["reach_act2"] * 100 for e in ev], "markers": True},
            {"name": "打过第二幕 Boss", "short": "第二幕", "var": "--s2", "xs": ex, "ts": et, "ys": [e["reach_act3"] * 100 for e in ev], "markers": True},
            {"name": "通关（评测，阴影为 95% 置信区间）", "short": "通关", "var": "--s3", "xs": ex, "ts": et, "ys": [e["win_rate"] * 100 for e in ev],
             "lo": [wilson(e["win_rate"], e["runs"])[0] * 100 for e in ev], "hi": [wilson(e["win_rate"], e["runs"])[1] * 100 for e in ev],
             "markers": True, "bold": True},
            # 训练对局通关率 2026-10-26 起按千分号画，2026-10-28 用户要求改回百分号
            {"name": "训练对局通关（采样，最近 50 次更新合计）", "short": "训练通关", "var": "--s4", "xs": lx, "ts": lt,
             "ys": run_win_50, "fmt": "{:.1f}%"},
        ], **(m or dict(w=1200, h=340)), y_fmt="{:.0f}%", y_min=0, y_max=reach_max, since=ev_since, vrefs=vrefs)
        c["combat"] = line_chart("第一、二幕 Boss 练习胜率", [
            {"name": "第一幕 Boss（固定名额，最近 10 次更新合计）", "short": "第一幕 Boss", "var": "--s3", "xs": lx, "ts": lt,
             "ys": pooled_rate(log, 0, 10)},
            {"name": "第二幕 Boss（固定名额，最近 10 次更新合计）", "short": "第二幕 Boss", "var": "--s4", "xs": lx, "ts": lt,
             "ys": pooled_rate(log, 1, 10)},
        ], **m, y_fmt="{:.0f}%", y_min=0, y_max=100, since=since, vrefs=vrefs)
        c["real"] = line_chart("非 Boss 练习胜率", [
            {"name": "第三幕普通战与精英（真实局面，最近 20 次更新合计）", "short": "第三幕", "var": "--s1", "xs": lx, "ts": lt,
             "ys": pooled(log, lambda r: fights(r, ["act3_monster", "act3_elite"]), 20)},
            {"name": "随机构筑精英战", "short": "随机构筑精英", "var": "--s2", "xs": lx, "ts": lt,
             "ys": pooled(log, lambda r: fights(r, ["mix_elite"]), 20)},
        ], **m, y_fmt="{:.0f}%", y_min=0, y_max=100, since=since)
        cur3 = lambda key: (lambda r: (lambda c: [c[3], c[0]] if c and c[0] else None)((r.get("cur") or {}).get(key)))
        c["cert"] = line_chart("第三幕 Boss 练习", [
            {"name": "连打两个 Boss（真实快照，最近 20 次更新合计）", "short": "连打", "var": "--s1", "xs": lx, "ts": lt, "ys": pooled(log, cur3("boss3_chain"), 20)},
            {"name": "第二个 Boss 单打（残血，真实快照）", "short": "第二个", "var": "--s2", "xs": lx, "ts": lt, "ys": pooled(log, lambda r: r.get("boss_second"), 20)},
            {"name": "第一个 Boss 单打", "short": "第一个", "var": "--s3", "xs": lx, "ts": lt, "ys": pooled(log, cur3("boss3"), 20)},
            {"name": "中途牌组挑战 Boss", "short": "中途牌组", "var": "--s4", "xs": lx, "ts": lt, "ys": pooled(log, cur3("boss3_probe"), 20)},
        ], **m, y_fmt="{:.0f}%", y_min=0, y_max=100, since=since, vrefs=vrefs)
        c["macro"] = line_chart("局外行为", [
            {"name": "休息处选休息（最近 20 次更新合计）", "short": "选休息", "var": "--s1", "xs": lx, "ts": lt, "ys": pooled_choices(log, "heal_share", 20), "fmt": "{:.0f}%"},
            {"name": "休息处选升级", "short": "选升级", "var": "--s2", "xs": lx, "ts": lt, "ys": pooled_choices(log, "smith_share", 20), "fmt": "{:.0f}%"},
            {"name": "卡牌奖励选跳过", "short": "跳过抓卡", "var": "--s3", "xs": lx, "ts": lt, "ys": pooled_choices(log, "skip_share", 20), "fmt": "{:.0f}%"},
            {"name": "路线选精英（占全部移动）", "short": "选精英", "var": "--s4", "xs": lx, "ts": lt, "ys": pooled_choices(log, "elite_share", 20), "fmt": "{:.0f}%"},
        ], **m, y_fmt="{:.0f}%", y_min=0, y_max=100, since=since, vrefs=vrefs)
        c["sps"] = line_chart("吞吐", [
            {"name": "决策/秒（最近 5 次滑动）", "short": "决策/秒", "var": "--s1", "xs": lx, "ts": lt, "ys": roll("sps", 5), "fmt": "{:,.0f}"},
            *([{"name": "其中协同采集机送来的（最近 5 次滑动）", "short": "协同采集", "var": "--s2", "xs": lx, "ts": lt,
                "ys": rolling([r["sps"] * r["remote_used"] / max(1, int(cfg.get("batch_steps") or 32768)) if "remote_used" in r and r.get("sps") else None for r in log], 5),
                "fmt": "{:,.0f}"}] if any("remote_used" in r for r in log) else []),
        ], **m, y_fmt="{:,.0f}", y_min=0, since=since)
        c["ent"] = line_chart("策略熵", [
            {"name": "全部决策", "short": "全部", "var": "--s1", "xs": lx, "ts": lt, "ys": roll("entropy", 5), "fmt": "{:.2f}"},
            {"name": "局外决策", "short": "局外", "var": "--s2", "xs": lx, "ts": lt, "ys": roll("entropy_macro", 5), "fmt": "{:.2f}"},
            {"name": "地图决策（选路，2026-10-27 起记录）", "short": "选路", "var": "--s3", "xs": lx, "ts": lt, "ys": roll("entropy_map", 5), "fmt": "{:.2f}"},
        ], **m, y_fmt="{:.2f}", y_min=0, since=since, vrefs=vrefs)
        # 学习端健康（决策记录 220–221）：log.jsonl 的 health 字段，只从有这项统计的那一版画起
        h_since = max(since or 0, health_t0) if health_t0 else since
        hv = lambda f: [f(r["health"]) if r.get("health") else None for r in log]
        c["hgn"] = line_chart("裁剪前梯度范数", [
            {"name": "各小批的平均（最近 20 次更新滑动）", "short": "梯度范数", "var": "--s1", "xs": lx, "ts": lt, "ys": rolling(hv(lambda h: h.get("grad_norm")), 20), "fmt": "{:.2f}"},
        ], **m, y_fmt="{:.1f}", y_min=0, refs=[(float(cfg.get("max_grad_norm") or 1.0), "裁剪阈值")], since=h_since)
        c["hclip"] = line_chart("被裁剪的小批比例", [
            {"name": "裁剪前梯度范数超过阈值的小批（最近 20 次更新滑动）", "short": "被裁剪", "var": "--s2", "xs": lx, "ts": lt,
             "ys": rolling(hv(lambda h: 100 * h["grad_clip_frac"] if h.get("grad_clip_frac") is not None else None), 20), "fmt": "{:.0f}%"},
        ], **m, y_fmt="{:.0f}%", y_min=0, y_max=100, since=h_since)
        c["hev"] = line_chart("价值解释方差", [
            {"name": "战斗（最近 20 次更新滑动）", "short": "战斗", "var": "--s1", "xs": lx, "ts": lt, "ys": rolling(hv(lambda h: (h.get("ev") or {}).get("combat")), 20), "fmt": "{:.3f}"},
            {"name": "局外", "short": "局外", "var": "--s2", "xs": lx, "ts": lt, "ys": rolling(hv(lambda h: (h.get("ev") or {}).get("macro")), 20), "fmt": "{:.3f}"},
        ], **m, y_fmt="{:.2f}", y_max=1.0, since=h_since)
        c["hupd"] = line_chart("每版更新量占权重", [
            {"name": f"{nm}（最近 20 次更新滑动）" if i == 0 else nm, "short": sh, "var": f"--s{i + 1}", "xs": lx, "ts": lt,
             "ys": rolling(hv(lambda h, g=g: 1000 * h["upd"][g] if g in (h.get("upd") or {}) else None), 20), "fmt": "{:.2f}‰"}
            for i, (g, nm, sh) in enumerate((("L0", "编码器第 0 层", "第 0 层"), ("L5", "编码器第 5 层", "第 5 层"),
                                             ("cand_mlp", "战斗打分头", "打分头"), ("value_mlp", "战斗价值头", "价值头")))
        ], **m, y_fmt="{:.1f}‰", y_min=0, since=h_since)
        c["kl"] = line_chart("近似 KL", [
            {"name": "近似 KL", "short": "KL", "var": "--s1", "xs": lx, "ts": lt, "ys": roll("approx_kl", 5), "fmt": "{:.3f}"},
        ], **m, y_fmt="{:.3f}", y_min=0, refs=[(0.03, "提前停止线")], since=since)
        if res:
            rx = [r["time"] for r in res]
            c["res"] = line_chart("机器占用率", [
                {"name": "CPU", "short": "CPU", "var": "--s1", "xs": rx, "ts": rx, "ys": rolling([r.get("cpu_util") for r in res], 4), "fmt": "{:.0f}%", "gap": 300},
                {"name": "显卡", "short": "显卡", "var": "--s2", "xs": rx, "ts": rx, "ys": rolling([r.get("gpu_util") for r in res], 4), "fmt": "{:.0f}%", "gap": 300},
                {"name": "内存（占总量）", "short": "内存", "var": "--s3", "xs": rx, "ts": rx,
                 "ys": [100 * r["mem_gb"] / r["mem_total_gb"] for r in res], "fmt": "{:.0f}%", "gap": 300},
                *[ser for h, rs in co_res.items() if rs for ser in (
                    {"name": f"{host_label(src.hosts, h)} CPU（协同采集）", "short": f"{h} CPU", "var": "--s4", "xs": [r["time"] for r in rs], "ts": [r["time"] for r in rs],
                     "ys": rolling([r.get("cpu_util") for r in rs], 4), "fmt": "{:.0f}%", "gap": 300},
                    {"name": f"{host_label(src.hosts, h)} 显卡（协同采集）", "short": f"{h} 显卡", "var": "--s5", "xs": [r["time"] for r in rs], "ts": [r["time"] for r in rs],
                     "ys": rolling([r.get("gpu_util") for r in rs], 4), "fmt": "{:.0f}%", "gap": 300})],
            ], **{**m, "h": 212}, y_fmt="{:.0f}%", y_min=0, y_max=100, x_kind="time", since=since, x_origin=t_origin)
        else:
            c["res"] = '<div class="empty" style="height:212px">暂无资源数据</div>'
        return c

    variants = {key: (charts(None if sec is None else now - sec), charts(None if sec is None else now - sec, mobile=True)) for key, _, sec in RANGES}

    def slot(name: str) -> str:
        return "".join(f'<div class="rg" data-r="{key}"><div class="dv">{variants[key][0][name]}</div><div class="mv">{variants[key][1][name]}</div></div>'
                       for key, _, _ in RANGES)

    def panel(title: str, sub: str, body: str, cls: str = "") -> str:
        return f'<section class="card {cls}"><div class="ptitle">{esc(title)}</div><div class="psub">{esc(sub)}</div>{body}</section>'

    since = stats_since()
    # 学习端健康（决策记录 220–221）：最近 20 版的平均读数
    hrec = [r["health"] for r in log if r.get("health")]
    if hrec:
        last = hrec[-20:]
        avg = lambda f: (lambda xs: sum(xs) / len(xs) if xs else None)([x for x in map(f, last) if x is not None])
        hv0 = next(r["version"] for r in log if r.get("health"))
        h_chips = "".join(f'<span class="chip"><b>{esc(k)}</b> {esc(v)}</span>' for k, v in [
            ("统计起于", f"第 {hv0:,} 版（已 {len(hrec):,} 版）"),
            ("梯度范数", f'{avg(lambda h: h.get("grad_norm")):.2f}（阈值 {float(cfg.get("max_grad_norm") or 1.0):g}）'),
            ("被裁剪", f'{100 * avg(lambda h: h.get("grad_clip_frac")):.0f}%'),
            ("解释方差 战斗", f'{avg(lambda h: (h.get("ev") or {}).get("combat")):.3f}'),
            ("局外", f'{avg(lambda h: (h.get("ev") or {}).get("macro")):.3f}'),
            ("每版更新 第 0 层", f'{1000 * avg(lambda h: (h.get("upd") or {}).get("L0")):.2f}‰')])
        health_body = f'<div class="chips">{h_chips}</div>'
    else:
        health_body = '<div class="empty" style="height:60px">学习端还没有这项统计（决策记录 220 起才记）</div>'
    td_deaths, td_bosses, td_n, td_v0 = train_diag(log, since["t"])
    td_since = f"自第 {td_v0:,} 版（{since['text']}）起累计" if td_v0 is not None else f"自 {since['text']} 起累计，新记录到达后显示"
    lc = max(changes) if changes else None
    last_change = f"。最近一次：{lc[1]}（{time.strftime('%m-%d %H:%M', time.localtime(lc[0]))}）" if lc else ""
    range_buttons = "".join(f'<button data-r="{key}" onclick="setRange(\'{key}\')">{esc(txt)}</button>' for key, txt, _ in RANGES)
    lr = res[-1] if res else {}
    res_chips = "".join(f'<span class="chip"><b>{esc(k)}</b> {esc(v)}</span>' for k, v in [
        ("CPU", f'{lr.get("cpu_util", 0):.0f}%'), ("内存", f'{lr.get("mem_gb", 0):.0f} / {lr.get("mem_total_gb", 0):.0f} GB'),
        ("显卡", f'{lr.get("gpu_util", 0):.0f}%'), ("显存", f'{lr.get("gpu_mem_gb", 0):.1f} / {lr.get("gpu_mem_total_gb", 0):.0f} GB'),
        ("温度", f'{lr.get("gpu_temp", 0):.0f}°C'), ("磁盘可用", f'{wd.get("disk_free_gb", "—")} GB'),
    ]) if lr else ""
    if lr and co_res:
        res_chips = f'<span class="chip"><b>{esc(host_label(src.hosts, src.main))}</b> 学习 + 采集</span>' + res_chips
    for h, rs in co_res.items():     # 协同采集机的最近一次采样
        if rs:
            x = rs[-1]
            res_chips += (f'<span class="chip"><b>{esc(host_label(src.hosts, h))}</b> 协同采集 · CPU {x.get("cpu_util", 0):.0f}% · 显卡 {x.get("gpu_util", 0):.0f}% · '
                          f'显存 {x.get("gpu_mem_gb", 0):.1f} / {x.get("gpu_mem_total_gb", 0):.0f} GB · 内存 {x.get("mem_gb", 0):.0f} / {x.get("mem_total_gb", 0):.0f} GB</span>')

    def brk(e: dict) -> str:   # 评测表里口径分界、训练改动处各插一行说明（表按时间倒序）
        nxt = next((x["time"] for x in ev if x["time"] > e["time"]), 0)
        return "".join(f'<tr class="brk"><td colspan="11">{esc(msg)}</td></tr>' for bt, msg in
                       [(bt, f"以上为{txt}之后的评测（新口径），与以下不可直接比较") for bt, txt in EVAL_BREAKS] +
                       [(bt, f"以上为{txt}之后的评测（口径不变）" if mig and bt == mig[0] else f"以上为{txt}上线之后的评测（口径不变）") for bt, txt in changes]
                       if e["time"] < bt <= nxt)
    rows = "".join(
        brk(e) +
        f'<tr><td>{e["version"]}</td><td>{cn_count(e.get("total_steps"))}</td><td>{when(e)}</td><td>{e["runs"]}</td>'
        f'<td>{e["floor_mean"]:.1f}</td><td>{e["floor_median"]:.0f}</td><td>{e["floor_p90"]:.0f}</td><td>{e["floor_max"]}</td>'
        f'<td>{pct(e["reach_act2"], 0)}</td><td>{pct(e["reach_act3"], 0)}</td><td>{pct(e["win_rate"], 1)}</td></tr>'
        for e in reversed(ev[-200:]))
    table = ('<div class="tscroll"><table><thead><tr><th>版本</th><th>训练量（步）</th><th>时间</th><th>局数</th><th>平均层数</th><th>中位</th><th>前 10%</th>'
             f'<th>最高</th><th>过第一幕</th><th>过第二幕</th><th>通关</th></tr></thead><tbody>{rows}</tbody></table></div>'
             ) if ev else '<div class="empty" style="height:120px">暂无评测</div>'

    machine = [f'{st.get("actors", "?")} 个采集进程 × {st.get("envs_per_actor", "?")} 个引擎',
               f'网络 d={st.get("model", {}).get("d", "?")}、{st.get("model", {}).get("layers", "?")} 层',
               f'当前第 {st.get("version", "?")} 版'] if st else []

    # 有分支时按“主线一块、分支一块”排：各块开头是块标题和这条线的卡片，主线块里是首屏、趋势、诊断、评测明细，
    # 分支块里是对照实验；最后是指标说明。没有分支时照旧一路排下来
    split = bool(src.ch)
    meta_html = " · ".join(f'<span class="ch">{esc(m)}</span>' for m in machine)
    if split:
        others = "".join(c for h, c in host_cards.items() if h not in (src.main, src.ch_host))
        top_lines = f'<div class="lines">{others}</div>' if others else ""
        blk_main = (f'<div class="blockhead bm" id="blk-main"><i></i><h2>主线 · {esc(host_label(src.hosts, src.main))}</h2><span class="meta">{meta_html}</span></div>'
                    f'<div class="lines">{host_cards.get(src.main, "")}</div>')
        blk_ch = (f'<div class="blockhead bc" id="blk-ch"><i></i><h2>分支 · {esc(host_label(src.hosts, src.ch_host))}</h2></div>'
                  f'<div class="lines">{host_cards.get(src.ch_host, "")}</div>'
                  f'{challenger_section(src, log, ev, now)}')
    else:
        top_lines, blk_main, blk_ch = hosts_html, "", ""
    body = f"""
<header class="top">
  <div class="brand"><h1>杀戮尖塔2 · Defect A10 训练</h1><span class="pill {state}"><i></i>{esc(state_txt)}</span>
    {"" if split else f'<div class="meta">{meta_html}</div>'}</div>
  <div class="livestrip">{topstats}</div>
  <button class="theme" onclick="toggleTheme()" title="切换深浅色">◐</button>
</header>
{alert}{viewbar}{top_lines}{blk_main}{hero}
{milestones(ev)}
{a_section(now, log)}
<div class="sechead"><h2>进展趋势</h2><span class="seg">{range_buttons}</span><span class="psub">鼠标停在图上（手机上点按或横向拖动）看具体数值；竖虚线是训练改动，靠近时提示框里写名称{last_change}</span></div>
<div class="row g3">
  {panel("通关率与各幕到达率", "评测：384 个按难度分层的 RT 种子，贪心策略；通关的阴影为每次评测的 95% 置信区间（Wilson）。训练对局带探索，通关率低得多", slot("reach"), "span2")}
  {panel("平均到达层数", "同一批评测；训练对局带探索且混有战斗课程，数值偏低。底色为三幕所在层数段", slot("floor"))}
</div>
<div class="row g3">
  {panel("复测与加测", "千种子（11-01 起为随机种子池 P1，之前 RT0000–RT0999；换池前后不配对）进阶 10、纯贪心；定时复测每 500 版一次（11-02 起，之前每 1,000 版），加测为计划外的单独复测。配对 p < 0.05 时加色", retest_panel(log), "span3")}
</div>
<details class="sec" data-sec="health" data-open="desktop"><summary class="sechead"><h2>学习端健康</h2><span class="psub">每次更新的梯度、裁剪、价值拟合与参数更新幅度；用来判断平台期是不是步长、价值头或可塑性的问题（决策记录 219–221）</span></summary>
<div class="row g3">
  {panel("读数", "最近 20 次更新的平均。被裁剪的比例长期在一半以上，说明步长主要由梯度裁剪阈值决定；每版更新量与相邻检查点净位移的比值见决策记录 221", health_body, "span3")}
  {panel("裁剪前梯度范数", "每次更新各小批裁剪前的全局梯度范数的平均；虚线为 max_grad_norm，超过就按比例缩到线上", slot("hgn"))}
  {panel("被裁剪的小批比例", "裁剪前梯度范数超过阈值的小批占比", slot("hclip"))}
  {panel("价值解释方差", "采集时旧价值对本批自举目标的解释方差，战斗与局外分开；下方诊断同时显示绝对误差与目标方差，不能仅凭下降判断价值头退化", slot("hev"))}
  {panel("每版更新量占权重", "这一版更新前后参数差的范数除以更新前的范数（千分号）；长期偏大而检查点之间净位移很小，说明多数更新在来回抵消", slot("hupd"), "span2")}
</div>
</details>
{value_probe_section(VALUE_PROBE_ROOT, panel, line_chart)}
<details class="sec" data-sec="diag"><summary class="sechead"><h2>训练诊断</h2><span class="psub">战斗课程、课程分配与局外样本、吞吐、策略熵与 KL、机器占用</span></summary>
<div class="row g3">
  {panel("第一、二幕 Boss 练习胜率", "单场战斗练习（占四分之一的引擎）：Boss 77%（三幕 6 : 27 : 67，多用整局留下的真实快照；第三幕见“卡点诊断”），第三幕真实局面 12%，随机构筑精英 11%", slot("combat"))}
  {panel("课程分配与局外样本", "最近 200 次更新合计：各类练习占入训步数的比例与成绩；局外决策按类型的多选步数、强制改选与策略梯度份额（单选步只训价值头）", curriculum_panel(log), "span3")}
  {panel("吞吐", "每秒学习用上的决策数（含收集与学习）；双机协同时另画其中协同采集机送来的部分", slot("sps"))}
  {panel("策略熵", "逐步下降说明策略在变得确定；骤降可能意味着过早收敛。局外决策（选路线、抓牌、事件、商店、休息处）单独一条", slot("ent"))}
  {panel("近似 KL", "每次更新的策略变化幅度；单遍超过 0.03 会提前结束本轮更新", slot("kl"))}
  {panel("机器占用率", "30 秒一次采样，按 2 分钟滑动平均；采样中断处断开。双机协同时主线机（学习 + 采集）与协同采集机画在一起", '<div class="chips">' + res_chips + '</div>' + slot("res"))}
</div>
</details>
<details class="sec" data-sec="stuck" data-open="desktop"><summary class="sechead"><h2>卡点诊断</h2><button type="button" class="rstbtn" data-rst data-since="{esc(since['text'])}" data-v0="{td_v0 or ''}" title="各 Boss 过关率、训练死因、先古遗物选取率从现在起重新累计"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9"/><path d="M13.6 1.8v2.9h-2.9"/></svg>重新计数<span class="rsts">自 {esc(since['text'][5:])}</span></button><span class="psub">Boss 过关率、死因、关键节点的牌组、真实局面练习、局外行为</span></summary>
<div class="row g3">
  {panel("各 Boss 过关率 · 训练", f"训练整局{td_since}，打到这个 Boss 的场次里打赢的比例；第一幕有两种（各 3 个 Boss），第三幕连打两个 Boss 各算一场", boss_panel(td_bosses))}
  {panel("训练死因", f"训练整局{td_since}，输掉的局死在哪场战斗；比例占全部训练整局（{td_n:,} 局，带探索）", deaths_panel(td_deaths, td_n))}
  {panel("打到关键节点时的牌组", "最近 4 次评测；开场血量为进入这场战斗时占最大生命的比例", deck_panel(ev))}
  {panel("非 Boss 练习胜率", "第三幕真实局面：用整局里打到这一步时的真实牌组、遗物、药水和血量重打当时那场普通战或精英战；随机构筑精英战：幕末规模的随机牌组，练到自己不抓的牌。第一、二幕真实局面与随机构筑普通战已停（胜率 97–99.6%），历史存档在原目录", slot("real"))}
  {panel("牌组大小与 Boss 胜率", "按牌组张数分档：真实快照 Boss 练习的胜率，以及评测里各节点的牌组张数分布和 Boss 过关率", deck_size_panel(log, ev), "span2")}
  {panel("第三幕 Boss 练习", "第三幕 Boss 的几类练习：连打两个 Boss、残血打第二个、单打第一个、中途牌组挑战 Boss（胜场 / 场数，最近 20 次更新合计）", slot("cert"))}
  {panel("局外行为 · 训练对局", "带探索的训练整局里几类局外选择的比例", slot("macro"))}
  {panel("局外行为 · 评测", "固定种子、贪心策略的评测整局，平均每局的局外选择", macro_panel(ev))}
  {panel("路线 · 评测", "评测整局每一幕实际走的路线，与这一幕能做到的最多值对照", route_panel(ev))}
  {panel("选牌对照 · 训练", "近期训练整局里每张牌被提供时选它的比例，与人类玩家（同版本、单人 A10 故障机器人）对照；按两者之差排出我们多拿、少拿的牌", card_panel(ev, log), "span3")}
  {panel("先古遗物选取率 · 训练", "涅奥与各幕开头的先古之民给出三件遗物时，模型选取各遗物的比例与次数（训练整局，带探索）；按幕分三栏，栏里按先古之民分组，组内按选取率从高到低", ancient_detail(log, since["t"], since["text"]), "span3")}
  {panel("先古遗物选取率 · 分享图", "涅奥与各幕开头的先古之民给出三件遗物时 AI 选取各遗物的比例（训练对局），三幕合为一张图，可保存分享；手机上长按图片亦可保存", ancient_panel(log, since["t"], since["text"]), "span3")}
</div></details>
<details class="sec" data-sec="detail"><summary class="sechead"><h2>评测明细</h2><span class="psub">每次评测的数字</span></summary>
<div class="row g3">
  {panel("评测明细", f"共 {len(ev)} 次，新的在前", table, "span3")}
</div></details>
{blk_ch}
<details class="sec" data-sec="defs" id="blk-defs"><summary class="sechead"><h2>指标说明</h2><span class="psub">评测、训练对局、奖励等的口径</span></summary>
<div class="row g3">
  <section class="card defs span3"><div class="ptitle">指标说明</div>
    {f'<p class="note">{esc(note)}</p>' if note else ""}
    <dl>
    <dt>评测</dt><dd>固定种子的 A10 Defect 整局，贪心策略（每步取最高分动作），从 Neow 打到终局，前后可比。2026-10-29 起在协同采集机 A 上用显卡推理，每轮 384 局；种子是从千种子复测的 RT0000–RT0999 里按难度分层抽的（tools/eval_seeds.py），通关率与千种子复测大体一致（抽样误差约 ±2 个百分点），之前是 EVAL000–191（192 局）。2026-11-01 起改用种子池 P1（tools/seed_pool.py：密码学随机源抽的 1,000 个种子字符串，评测取其中均匀随机抽的 384 个，与复测照旧重叠 384 局），与之前的分层 RT 种子不可直接比较。通关率 = 打赢终局第二个 Boss 的比例；“近 12 次”为最近 12 轮评测按局数合计。</dd>
    <dt>到达层数</dt><dd>死亡或通关时的层数，常规满分 49：第 17 层第一幕 Boss，第 33 层第二幕 Boss，第 48、49 层终局两个 Boss。第二幕地图偶尔多两层，最多到 51 层。</dd>
    <dt>训练对局</dt><dd>四分之三的引擎打整局（都从第一层开），四分之一打单场战斗练习，都带探索。局外决策另有强制改选（按类别 3%–10% 的概率换成其他选项），2026-10-29 起按实际出手的概率做重要性校正后进策略梯度。</dd>
    <dt>战斗练习</dt><dd>Boss 77%（三幕 6 : 27 : 67，第三幕全部用整局留下的真实快照，其中连打两个 Boss、残血打第二个、中途牌组挑战各有份额，见“课程分配与局外样本”）；第三幕真实局面的普通战与精英战 12%（其中精英 40%）；随机构筑精英战 11%。第一、二幕真实局面与随机构筑普通战 10-30 起停练。</dd>
    <dt>奖励</dt><dd>战斗练习：赢 +1、输 −1（Boss 输局按打掉的血量给部分分）。整局：每层 +0.02；打过第一、二幕 Boss 各 +0.5，第三幕第一个 Boss +1.5，第二个 +2.0，通关另 +2.0；白丢药水 −0.1。血量、牌组、升级、遗物等塑形与胜率、进度势能都关着。</dd>
    <dt>训练</dt><dd>PPO，学习率 1.5×10⁻⁴，每版 32,768 步，每批 3 遍；局外决策（选路线、抓牌、事件、商店、休息处）有自己的策略头与价值头，熵系数局外 0.05、战斗 0.01。6 层 × 宽 256 的模型，4090 机学习并采集，3090 机协同采集。</dd>
    <dt>对照实验</dt><dd>两条线按分叉后的版本数对齐，不按时间。2026-10-28 起两台机器协同训练一条主线，没有对照线时这一区不显示。</dd>
    <dt>路线</dt><dd>2026-10-27 起地图决策带路线摘要：每个下一步出发到本幕 Boss 的全部路线里，各类房间最少、最多几个，精英与休息处的取舍，以及精英前面有没有火堆（上一个精英之后经过了休息处）。面板比较每幕实际走的与这一幕开头能做到的最多值，只算走到 Boss 的幕。</dd>
    <dt>选牌对照</dt><dd>训练整局里每个选牌奖励界面记下提供的牌和选中的牌（跳过后又回来只算最后一次），取最近 1,000 次更新合计；评测是固定种子，提供的牌高度重复，只在训练记录还没有时退回评测。社区数据来自 Spire Codex 玩家上传的对局，只作参照，不进奖励。</dd>
    <dt>训练统计的起点</dt><dd>各 Boss 过关率、训练死因、先古遗物选取率三项共用一个起点（当前 {since['text']}，两条线按同一时刻起算），都是带探索的训练整局，比评测低。用“卡点诊断”旁的“重新计数”按钮重置：上一段的累计按线留档到原目录的“训练统计留档”，再从按下的时刻重新累计。</dd>
    </dl></section>
</div></details>"""
    return f'<div id="app" data-live=\'{esc(json.dumps(live))}\'>{body}</div>'


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Defect 训练看板</title><style>
:root{color-scheme:light;--page:#f4f4f1;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e6e5df;--axis:#c3c2b7;
--ring:rgba(11,11,11,.08);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#8e5bd8;--good:#0ca30c;--warn:#fab219;--bad:#d03b3b;--up:#006300;--down:#b42828;
--act1:#cde2fb;--act2:#b7d3f6;--act3:#9ec5f4;--band1:rgba(42,120,214,.035);--band2:rgba(42,120,214,.075);--band3:rgba(42,120,214,.12);--flash:rgba(42,120,214,.18)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.09);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#a383e8;--up:#0ca30c;--down:#e66767;
--act1:#0d366b;--act2:#104281;--act3:#184f95;--band1:rgba(57,135,229,.04);--band2:rgba(57,135,229,.08);--band3:rgba(57,135,229,.13);--flash:rgba(57,135,229,.25)}}
:root[data-theme=dark]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--ring:rgba(255,255,255,.09);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#a383e8;--up:#0ca30c;--down:#e66767;
--act1:#0d366b;--act2:#104281;--act3:#184f95;--band1:rgba(57,135,229,.04);--band2:rgba(57,135,229,.08);--band3:rgba(57,135,229,.13);--flash:rgba(57,135,229,.25)}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1904px;margin:0 auto;padding:14px 16px 40px}
.card{background:var(--surface);border-radius:14px;padding:14px 18px;box-shadow:0 0 0 1px var(--ring)}
.row{display:grid;gap:14px;margin-top:14px}.g3{grid-template-columns:repeat(3,1fr)}.span2{grid-column:span 2}.span3{grid-column:1/-1}
.row>*{min-width:0}
h1{font-size:20px;margin:0;font-weight:650;letter-spacing:.01em}h2{font-size:15px;margin:0;font-weight:600}
.ptitle{font-weight:600;font-size:15px}.psub{color:var(--muted);font-size:12px;margin:1px 0 6px}
.top{display:flex;align-items:center;gap:24px;padding:4px 2px}
.brand{display:flex;flex-wrap:wrap;align-items:center;gap:4px 12px;width:520px}.brand .meta{width:100%;color:var(--muted);font-size:12px}
.pill{display:inline-flex;align-items:center;gap:7px;padding:2px 11px;border-radius:999px;font-size:13px;box-shadow:0 0 0 1px var(--ring);background:var(--surface)}
.pill i{width:8px;height:8px;border-radius:50%;display:inline-block}.pill.ok i{background:var(--good);animation:pulse 2s infinite}
.pill.warn i{background:var(--warn)}.pill.bad i{background:var(--bad)}.pill.idle i{background:var(--muted)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(12,163,12,.5)}70%{box-shadow:0 0 0 7px rgba(12,163,12,0)}100%{box-shadow:0 0 0 0 rgba(12,163,12,0)}}
.livestrip{flex:1;display:grid;grid-auto-flow:column;grid-auto-columns:minmax(0,auto);background:var(--surface);border-radius:14px;box-shadow:0 0 0 1px var(--ring)}
.ls{padding:8px 16px;border-left:1px solid var(--grid)}.ls:first-child{border-left:none}
.lsl{color:var(--muted);font-size:12px}.lsv{font-size:20px;font-weight:650;white-space:nowrap}.lsv small{font-size:12px;font-weight:500;color:var(--ink2)}
.evbar{display:block;height:4px;border-radius:2px;background:var(--grid);margin-top:3px;overflow:hidden}.evbar i{display:block;height:100%;width:0;background:var(--s1);transition:width .6s}
.evt{font-size:16px}.ch{white-space:nowrap}.sync-ok{color:var(--up)}.sync-warn{color:var(--down)}.lsv .sub{font-size:12px;font-weight:500;color:var(--ink2)}
.theme{width:34px;height:34px;border-radius:50%;border:none;background:var(--surface);box-shadow:0 0 0 1px var(--ring);color:var(--ink2);cursor:pointer;font-size:16px}
.herohead{display:flex;justify-content:space-between;align-items:flex-start}
.bignum{font-size:46px;font-weight:700;line-height:1;text-align:right;letter-spacing:-.02em}.bignum .of{font-size:16px;font-weight:500;color:var(--ink2);letter-spacing:0}
.bigsub{display:flex;gap:12px;justify-content:flex-end;margin-top:6px;font-size:12px}
.dl{color:var(--ink2)}.dl.up{color:var(--up)}.dl.down{color:var(--down)}.dl.flat{color:var(--muted)}
.journey .hbar{fill:var(--s1);opacity:.55}.journey .hbar:hover{opacity:.9}
.journey .fill{fill:var(--s1)}.actt{font-size:12px;font-weight:600;fill:var(--ink)}.actr{font-weight:400;fill:var(--ink2)}
.actt.on,.actt.on .actr{fill:#fff}
.journey .xseg{fill:var(--surface);opacity:.55}.journey .xcol{fill:var(--act1);opacity:.35}.journey .hbar.hx{opacity:.3}.journey .hbar.hwin{fill:var(--s4);opacity:.85}.journey .winseg{fill:var(--s4)}.journey .wint{fill:#fff;font-size:13px;font-weight:700}
.skull path:first-child{fill:var(--ink2)}.skull .sk-e{fill:var(--surface)}.skull .sk-t{stroke:var(--surface);stroke-width:.8;fill:none}
.mk{stroke:var(--ink2);stroke-width:1.5}.mk.m-mean{stroke:var(--ink);stroke-width:2.5}.mk.m-start{stroke-dasharray:3 3}
.mkt{font-size:12px;fill:var(--ink2)}.mkt.m-mean{font-size:14px;font-weight:650;fill:var(--ink)}.mkv{font-weight:650}
.fbar{height:14px;background:var(--grid);border-radius:7px;overflow:hidden}.fbar i{display:block;height:100%;background:var(--s1);border-radius:7px;transition:width .8s}
.drow{display:grid;grid-template-columns:140px 1fr 116px;column-gap:12px;align-items:center;padding:4px 0;border-bottom:1px solid var(--grid)}.drow:last-child{border-bottom:none}
.dname{min-width:0;line-height:1.3}.dname b,.dname small{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.dname b{font-weight:600;font-size:13px}.dname small{font-size:11px;color:var(--muted)}
.dgrp{font-size:11px;font-weight:600;color:var(--muted);padding:12px 0 2px}.dgrp:first-child{padding-top:0}.drow .fbar{height:10px}.dval{font-size:14px;font-weight:650;text-align:right;line-height:1.2}.fval{font-size:20px;font-weight:650;text-align:right;line-height:1.2}.fn{display:block;font-size:11px;color:var(--muted);font-weight:400}
.fd{grid-column:2;font-size:11px;min-height:0}
.spark{width:100%;height:28px;display:block}.spark polyline{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.fs{font-size:20px;font-weight:650;line-height:1.3}.fs small{font-size:12px;font-weight:500;color:var(--ink2)}
.msbox{margin-top:14px;padding:12px 18px 14px}.mshead{display:flex;gap:12px;align-items:baseline}
.msrow{display:grid;grid-template-columns:repeat(10,1fr);margin-top:8px}
.ms{position:relative;text-align:center;padding:0 6px}.msdot{position:relative;z-index:1;width:30px;height:30px;margin:0 auto 6px;border-radius:50%;
display:flex;align-items:center;justify-content:center;font-weight:650;font-size:13px;background:var(--surface);box-shadow:0 0 0 2px var(--grid);color:var(--muted)}
.ms.done .msdot{background:var(--s1);color:#fff;box-shadow:none}.ms.next .msdot{box-shadow:0 0 0 2px var(--s1);color:var(--s1);animation:ring 2.4s infinite}
@keyframes ring{0%{box-shadow:0 0 0 2px var(--s1),0 0 0 2px var(--flash)}70%{box-shadow:0 0 0 2px var(--s1),0 0 0 10px rgba(0,0,0,0)}100%{box-shadow:0 0 0 2px var(--s1)}}
.msn{font-weight:600;font-size:13px}.ms.todo .msn{color:var(--muted);font-weight:500}.msw{font-size:11px;color:var(--muted)}.ms.next .msw b{color:var(--ink)}
.sechead{display:flex;align-items:center;gap:14px;margin:22px 2px 0}.sechead .psub{margin:0}
.seg{display:inline-flex;border-radius:999px;box-shadow:0 0 0 1px var(--ring);background:var(--surface);overflow:hidden}
.seg button{border:none;background:none;color:var(--ink2);font:inherit;font-size:12px;padding:3px 12px;cursor:pointer}
.blockhead{display:flex;align-items:center;flex-wrap:wrap;gap:4px 12px;margin:30px 2px 0;padding-top:14px;border-top:2px solid var(--grid);scroll-margin-top:8px}
.blockhead h2{font-size:18px;font-weight:650}.blockhead i{width:10px;height:10px;border-radius:50%;flex:none}.blockhead.bm i{background:var(--s1)}.blockhead.bc i{background:var(--s2)}
.blockhead .meta{color:var(--muted);font-size:12px}.blockhead+.lines{margin-top:8px;grid-template-columns:minmax(0,640px)}#blk-main{margin-top:16px}
.jump{display:inline-flex;gap:12px;font-size:12px}.jump a{color:var(--ink2);text-decoration:none;border-bottom:1px dashed var(--axis)}.jump a:hover{color:var(--ink)}
.viewbar{display:flex;align-items:center;flex-wrap:wrap;gap:6px 12px;margin:12px 2px 0}.viewbar .psub{margin:0;flex:1;min-width:0}.viewbar .seg button.on{background:var(--ink);color:var(--page)}
.rg{display:none}body[data-range=all] .rg[data-r=all],body[data-range="24h"] .rg[data-r="24h"],body[data-range="6h"] .rg[data-r="6h"]{display:block}
body[data-range=all] .seg button[data-r=all],body[data-range="24h"] .seg button[data-r="24h"],body[data-range="6h"] .seg button[data-r="6h"]{background:var(--ink);color:var(--page)}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2)}.lg i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.chartwrap{position:relative;touch-action:pan-y}.mv{display:none}.chart{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}.ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4}.ref.faint{opacity:.5}.tvr{color:var(--muted);margin-top:3px}tr.brk td{color:var(--muted);font-size:12px;text-align:center;border-top:2px dashed var(--grid)}
.tick{fill:var(--muted);font-size:12px}.reft{fill:var(--muted);font-size:12px;paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.bandt{fill:var(--muted);font-size:12px}.direct{fill:var(--ink2);font-size:13px}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.ciband{opacity:.16;stroke:none}.line.bold{stroke-width:3}.dot{stroke:var(--surface);stroke-width:2}.cross{stroke:var(--axis);stroke-width:1}
.empty{color:var(--muted);display:flex;align-items:center;justify-content:center}
.tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--ring),0 4px 14px rgba(0,0,0,.14);
border-radius:8px;padding:6px 10px;font-size:12px;white-space:nowrap;z-index:2}.tip b{font-weight:600}.tip i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 6px}.chip{font-size:12px;color:var(--ink2);background:var(--page);border-radius:6px;padding:1px 8px}.chip b{font-weight:600;color:var(--ink)}
.ancacts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px 28px}@media (max-width:980px){.ancacts{grid-template-columns:1fr}}.acol{min-width:0}.acth{display:flex;justify-content:space-between;align-items:baseline;gap:8px;border-top:3px solid var(--ac);padding-top:6px;margin-bottom:10px}.acth b{font-size:15px}.acth .fn{display:inline}.ablk{break-inside:avoid;margin-bottom:14px}.ah{display:flex;justify-content:space-between;align-items:baseline;gap:8px;border-bottom:1px solid var(--grid);padding-bottom:3px;margin-bottom:4px}.ah .fn{display:inline}.arow{position:relative;display:grid;grid-template-columns:minmax(0,1fr) 64px 92px;column-gap:8px;align-items:center;padding:2px 0;font-size:13px}.arow .fbar{height:8px}.an{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.av{text-align:right;font-weight:600}.av .fn{display:inline;margin-left:4px}.arow[data-tip]:not([data-tip=""]):hover::after{content:attr(data-tip);position:absolute;left:0;top:100%;z-index:20;width:max-content;max-width:300px;padding:6px 9px;border-radius:6px;background:var(--card,#fff);color:var(--ink,#222);border:1px solid var(--grid);box-shadow:0 4px 14px rgba(0,0,0,.18);font-size:12px;font-weight:400;line-height:1.5;white-space:normal;pointer-events:none}.arow:hover .an{color:var(--s1)}.tscroll{max-height:330px;overflow:auto}.tscroll.tall{max-height:640px}.cfull{margin-top:14px}.cfbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin-bottom:6px}.cfbar .subh{margin:0;flex:1 1 auto}.cfq{flex:0 1 220px;min-width:0;font:inherit;font-size:13px;padding:4px 8px;border:1px solid var(--grid);border-radius:6px;background:transparent;color:inherit}.cfn{font-size:12px;color:var(--muted)}.cfull th{cursor:pointer;user-select:none;white-space:nowrap}.cfull th[data-dir="1"]::after{content:" ▲"}.cfull th[data-dir="-1"]::after{content:" ▼"}@media (hover:hover){}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:4px 10px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:500;position:sticky;top:0;background:var(--surface)}
th:first-child,td:first-child{text-align:left}
.defs{color:var(--ink2);font-size:13px}.defs.span3 dl{columns:2 420px;column-gap:36px;margin:0}.defs dt{break-after:avoid}.defs dd{break-inside:avoid}.defs dt{font-weight:600;color:var(--ink);margin-top:8px}.defs dd{margin:0}.note{margin:6px 0;color:var(--ink2)}
.flash{animation:flash 1.8s ease-out}@keyframes flash{0%{background:var(--flash)}100%{background:transparent}}
/* 多机：各主机状态条、对照实验 */
.hosts{display:flex;flex-wrap:wrap;gap:8px 10px;margin-top:10px}
.host{display:flex;flex-wrap:wrap;align-items:center;gap:2px 12px;padding:4px 14px 4px 5px;background:var(--surface);border-radius:999px;box-shadow:0 0 0 1px var(--ring);font-size:13px;color:var(--ink2)}
.host .pill{box-shadow:none;background:var(--page);font-weight:600;color:var(--ink)}.hrole{font-weight:600;color:var(--ink)}.hrole.main{color:var(--s1)}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
.cmpchips{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 14px;margin:8px 0}.cmpv{font-size:14px;font-weight:600}
table.cmp td.cl{text-align:left;white-space:nowrap}.cl i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
table.cmp tr.cdiff td{border-bottom:2px solid var(--grid);font-size:12px}table.cmp td[rowspan]{vertical-align:top;text-align:left;font-weight:600}
.alert{margin-top:10px;padding:8px 14px;border-radius:10px;background:color-mix(in srgb,var(--bad) 12%,var(--surface));color:var(--ink);box-shadow:0 0 0 1px var(--bad);font-size:13px}.alert b{color:var(--bad)}
/* 首屏：关键指标块 */
.kpis{grid-template-columns:repeat(4,minmax(0,1fr))}.kpis.k5{grid-template-columns:repeat(5,minmax(0,1fr))}.kpi{padding:12px 18px 10px;display:flex;flex-direction:column;gap:2px}
.kl{font-size:13px;font-weight:600;color:var(--ink2)}.kv{font-size:34px;font-weight:700;line-height:1.15;letter-spacing:-.01em}
.kv .of{font-size:14px;font-weight:500;color:var(--ink2);letter-spacing:0;margin-left:2px}.ks{font-size:12px;color:var(--muted)}
.kt{font-size:12px;color:var(--ink2);margin-top:2px}.kpi .spark{margin-top:auto;padding-top:6px;height:30px}
.hero{margin-top:14px}
/* 训练线：每台主机一张小卡 */
.lines{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;margin-top:10px}
.line{background:var(--surface);border-radius:12px;box-shadow:0 0 0 1px var(--ring);padding:8px 14px;display:grid;
  grid-template-columns:auto 1fr;grid-template-areas:"h v" "m m";align-items:center;gap:2px 14px}
.line.main{box-shadow:0 0 0 1.5px var(--s1)}.lh{grid-area:h;display:flex;align-items:center;gap:8px}.lv{grid-area:v;justify-self:end;font-size:18px;font-weight:650}
.lv small{font-size:12px;font-weight:500;color:var(--ink2)}.lm{grid-area:m;display:flex;flex-wrap:wrap;gap:2px 14px;font-size:12px;color:var(--ink2)}.lm b{color:var(--ink);font-weight:600}
.role{font-size:12px;font-weight:700;padding:1px 8px;border-radius:6px;background:var(--page);color:var(--ink2)}.line.main .role{background:var(--s1);color:#fff}
/* 可折叠分区：summary 就是分区标题 */
details.sec>summary{list-style:none;cursor:pointer}details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary.sechead::before{content:"";width:8px;height:8px;border-right:2px solid var(--ink2);border-bottom:2px solid var(--ink2);
  transform:rotate(-45deg);transition:transform .15s;margin:0 2px 0 4px;flex:none}
details.sec[open]>summary.sechead::before{transform:rotate(45deg)}details.sec>summary.sechead:hover h2,details.sec>summary.sechead:hover h3{text-decoration:underline}
.sechead.sub{margin-top:12px}.sechead h3{font-size:14px;margin:0;font-weight:600}
/* 里程碑：一行摘要，展开看全部 */
.msbox{margin-top:14px;padding:10px 18px}.msbox>summary{display:flex;flex-wrap:wrap;align-items:center;gap:4px 14px}
.msdots{display:inline-flex;gap:4px}.msdots i{width:10px;height:10px;border-radius:50%;background:var(--grid)}.msdots i.done{background:var(--s1)}
.msdots i.next{box-shadow:0 0 0 2px var(--s1);background:var(--surface)}.msnext{font-size:13px;color:var(--ink2)}.msnext b{color:var(--ink)}
.msbox .more{margin-left:auto;font-size:12px;color:var(--s1)}.msbox[open] .more{display:none}.msbox .msrow{grid-template-columns:repeat(5,1fr);row-gap:14px;margin-top:12px}
.msbox .ms{text-align:left;display:grid;grid-template-columns:28px minmax(0,1fr);column-gap:10px;padding:0}
.msbox .ms>:not(.msdot){grid-column:2}.msbox .msdot{grid-row:1/span 3;width:26px;height:26px;margin:0;font-size:12px}
/* 选牌对照：两张表并排，手机上下排 */
.cg2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.subh{font-size:13px;font-weight:600;margin:4px 0 6px}
@media (max-width:760px){.cg2{grid-template-columns:1fr}}
/* 对照实验摘要表 */
table.sum th i{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}table.sum td:first-child{font-weight:600}
/* 窄屏：顶栏统计换到第二行 */
@media (max-width:1500px){.top{flex-wrap:wrap;gap:10px 16px}.brand{width:auto;flex:1}.livestrip{order:3;flex-basis:100%}}
/* 平板：首屏与趋势改为上下排，诊断两列，里程碑两行 */
@media (max-width:1200px){.g3{grid-template-columns:repeat(2,minmax(0,1fr))}
.livestrip{grid-auto-flow:row;grid-template-columns:repeat(3,minmax(0,1fr))}.ls:nth-child(3n+1){border-left:none}.ls:nth-child(n+4){border-top:1px solid var(--grid)}.kpis,.kpis.k5{grid-template-columns:repeat(2,minmax(0,1fr))}.msbox .msrow{grid-template-columns:repeat(3,1fr)}
.msrow{grid-template-columns:repeat(5,1fr);row-gap:16px}.g4{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3>section:not(.span2) .dv{display:none}.g3>section:not(.span2) .mv{display:block}}
/* 手机：单栏，图表换成手机版尺寸，里程碑改为竖排 */
@media (max-width:760px){body{font-size:13px}main{padding:10px 12px 28px}.row{gap:10px;margin-top:10px}.card{padding:12px 14px;border-radius:12px}
.dv{display:none}.mv{display:block}
.g3,.g4{grid-template-columns:minmax(0,1fr)}.span2{grid-column:auto}
.hosts{gap:8px}.host{width:100%;border-radius:12px;font-size:12px;gap:2px 10px}.cmpchips{gap:6px 12px}
h1{font-size:17px;margin-bottom:5px}.brand{display:block;min-width:0;text-wrap:pretty}.brand .meta{display:inline;font-size:11px;line-height:1.7}.pill{font-size:12px;padding:0 9px;gap:5px;margin-right:6px;vertical-align:1px}.theme{align-self:flex-start;flex:none}
.livestrip{grid-template-columns:repeat(2,minmax(0,1fr));border-radius:12px}.ls{padding:7px 12px;border-left:none;border-top:1px solid var(--grid)}
.ls:nth-child(-n+2){border-top:none}.ls:nth-child(even){border-left:1px solid var(--grid)}.lsv{font-size:17px;white-space:normal;line-height:1.35}.lsv .sep{display:none}.lsv .ch{display:block}.evt{font-size:15px}.evbar{margin-top:6px}
.herohead{flex-direction:column;gap:6px}.bignum{font-size:38px;text-align:left}.bigsub{justify-content:flex-start;flex-wrap:wrap}
.drow{grid-template-columns:112px minmax(0,1fr) 88px;column-gap:8px}

.msbox{padding:10px 14px}.msbox .msrow{grid-template-columns:minmax(0,1fr);row-gap:10px}.msnext{flex-basis:100%}.msbox .msw{font-size:11px}.kpis{gap:10px}.kpi{padding:10px 12px 8px}.kv{font-size:26px}.kv .of{font-size:12px}.kl{font-size:12px}.ks,.kt{font-size:11px}.lines{grid-template-columns:minmax(0,1fr);gap:8px}.lv{font-size:16px}
.sechead{flex-wrap:wrap;gap:6px 12px;margin-top:18px}.sechead .psub{flex-basis:100%}.viewbar .psub{flex-basis:100%;font-size:11px}
.legend{font-size:11px;gap:2px 10px}.tip{white-space:normal;max-width:220px}
table{font-size:12px}th,td{padding:4px 6px}.tscroll{max-height:300px}}
.replay-btn{position:fixed;right:18px;bottom:18px;z-index:50;display:inline-flex;align-items:center;gap:6px;padding:9px 16px;border-radius:22px;
  background:var(--surface);color:var(--ink);border:1px solid var(--axis);box-shadow:0 6px 18px #0005;text-decoration:none;font-weight:700;font-size:14px}
.replay-btn:hover{border-color:var(--ink2)}
.rstbtn{display:inline-flex;align-items:center;gap:6px;padding:3px 12px 3px 9px;border-radius:999px;border:none;background:var(--surface);color:var(--ink);
  box-shadow:0 0 0 1px var(--ring);font:inherit;font-size:12px;font-weight:600;cursor:pointer;white-space:nowrap;transition:box-shadow .15s,background .15s}
.rstbtn svg{width:13px;height:13px;fill:none;stroke:var(--s1);stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;transition:transform .4s}
.rstbtn:hover{box-shadow:0 0 0 1px var(--s1);background:color-mix(in srgb,var(--s1) 7%,var(--surface))}.rstbtn:hover svg{transform:rotate(-180deg)}
.rstbtn .rsts{font-weight:500;color:var(--muted);padding-left:7px;border-left:1px solid var(--grid)}
dialog.rstdlg{border:none;border-radius:14px;padding:0;width:min(420px,calc(100vw - 32px));background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--ring),0 18px 48px rgba(0,0,0,.28)}
dialog.rstdlg::backdrop{background:rgba(0,0,0,.35);backdrop-filter:blur(2px)}
.rstdlg form{padding:20px 22px 18px;display:flex;flex-direction:column;gap:12px}.rstdlg h3{margin:0;font-size:16px}
.rstdlg p{margin:0;font-size:13px;color:var(--ink2);line-height:1.6}.rstdlg p b{color:var(--ink);font-weight:600}
.rstdlg ul{margin:0;padding-left:18px;font-size:13px;color:var(--ink2)}
.rstdlg label{font-size:12px;color:var(--ink2);display:flex;flex-direction:column;gap:4px}
.rstdlg input[type=password]{font:inherit;font-size:14px;padding:7px 10px;border-radius:8px;border:1px solid var(--axis);background:var(--page);color:var(--ink)}
.rstdlg input[type=password]:focus{outline:2px solid var(--s1);outline-offset:-1px;border-color:transparent}
.rstdlg .rem{flex-direction:row;align-items:center;gap:6px}
.rstdlg .act{display:flex;justify-content:flex-end;gap:8px;margin-top:4px}
.rstdlg .act button{font:inherit;font-size:13px;font-weight:600;padding:6px 16px;border-radius:8px;border:none;cursor:pointer}
.rstdlg .cancel{background:var(--page);color:var(--ink2);box-shadow:0 0 0 1px var(--ring)}.rstdlg .ok{background:var(--s1);color:#fff}.rstdlg .ok:disabled{opacity:.5;cursor:default}
.rstdlg .msg{min-height:1.2em;font-size:12px;color:var(--ink2)}.rstdlg .msg.bad{color:var(--down)}.rstdlg .msg.good{color:var(--up)}
</style></head><body data-range="all"><main>__BODY__</main>
<a class="replay-btn" href="__REPLAY__" title="查看评测里通关对局的逐步回放">▶ 对局回放</a>
<dialog class="rstdlg" id="rstdlg"><form method="dialog" onsubmit="return rstSend(event)">
<h3>重新计数训练统计</h3>
<p>以下三项将从<b>现在</b>起重新累计（两条线同一时刻）：</p>
<ul><li>各 Boss 过关率 · 训练</li><li>训练死因</li><li>先古遗物选取率 · 训练</li></ul>
<p>当前一段（<b id="rstfrom"></b>起）的累计会按线留档到原目录的“训练统计留档”，不会丢失。</p>
<label>口令<input type="password" id="rstpw" autocomplete="current-password" required></label>
<label class="rem"><input type="checkbox" id="rstrem">在这台设备上记住口令</label>
<div class="msg" id="rstmsg"></div>
<div class="act"><button type="button" class="cancel" onclick="document.getElementById('rstdlg').close()">取消</button><button type="submit" class="ok" id="rstok">确认重置</button></div>
</form></dialog>
<script>
function toggleTheme(){var r=document.documentElement,d=r.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
r.dataset.theme=d==='dark'?'light':'dark';try{localStorage.setItem('sls2theme',r.dataset.theme)}catch(e){}}
try{var t=localStorage.getItem('sls2theme');if(t)document.documentElement.dataset.theme=t}catch(e){}
// 训练统计重新计数：按钮在“卡点诊断”的标题行里（点它不开合分区）；口令由香港服务器校验，请求记下后本机约半分钟内取回并重置
document.addEventListener('click',function(e){var b=e.target.closest&&e.target.closest('[data-rst]');if(!b)return;e.preventDefault();e.stopPropagation();
  var d=document.getElementById('rstdlg'),m=document.getElementById('rstmsg'),pw=document.getElementById('rstpw'),ok=document.getElementById('rstok');
  document.getElementById('rstfrom').textContent=(b.dataset.v0?'第 '+(+b.dataset.v0).toLocaleString()+' 版，':'')+b.dataset.since;
  ok.disabled=false;m.className='msg';m.textContent='';try{var k=localStorage.getItem('sls2rst');if(k){pw.value=k;document.getElementById('rstrem').checked=true}}catch(_){}
  if(location.protocol==='file:'){ok.disabled=true;m.textContent='本机文件版不能重置，请在 dashboard.example 上操作'}
  d.showModal()},true);
function rstSend(e){e.preventDefault();var m=document.getElementById('rstmsg'),ok=document.getElementById('rstok'),pw=document.getElementById('rstpw').value,
  rem=document.getElementById('rstrem').checked;ok.disabled=true;m.className='msg';m.textContent='提交中…';
  fetch('/api/reset-stats',{method:'POST',cache:'no-store',credentials:'omit',headers:{'Authorization':'Basic '+btoa(unescape(encodeURIComponent('reset:'+pw)))}}).then(function(r){
    if(r.ok){try{rem?localStorage.setItem('sls2rst',pw):localStorage.removeItem('sls2rst')}catch(_){}
      m.className='msg good';m.textContent='已提交。约半分钟内生效，按钮上的起点会变成现在的时刻。';setTimeout(function(){document.getElementById('rstdlg').close()},2500)}
    else{ok.disabled=false;m.className='msg bad';m.textContent=r.status===429||r.status===503?'尝试太频繁，请一分钟后再试':(r.status===401||r.status===403?'口令不对':'服务器返回 '+r.status)}
  }).catch(function(){ok.disabled=false;m.className='msg bad';m.textContent='连不上服务器，稍后再试'});return false}
function setRange(r){document.body.dataset.range=r;try{localStorage.setItem('sls2range',r)}catch(e){}}
try{var rr=localStorage.getItem('sls2range');if(rr)document.body.dataset.range=rr}catch(e){}
function cn(n){n=Math.floor(n);if(n<1e4)return String(n);var w=Math.floor(n/1e4),y=Math.floor(w/1e4),r=w%1e4;return y?(y+' 亿'+(r?' '+r+' 万':'')):(w+' 万')}
function dur(s){s=Math.max(0,Math.floor(s));var d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?d+' 天 '+h+' 小时':(h?h+' 小时 '+m+' 分':m+' 分')}
// 可折叠分区：没点过时按屏宽给默认（data-open="desktop" 的在宽屏上展开），点过的记在本机；原地刷新后照样恢复
function initSecs(root){var wide=matchMedia('(min-width:761px)').matches;
  root.querySelectorAll('details.sec').forEach(function(d){var k='sls2sec:'+d.dataset.sec,v=null;try{v=localStorage.getItem(k)}catch(e){}
    d._restoring=true;d.open=v!==null?v==='1':(d.dataset.open==='desktop'&&wide);setTimeout(function(){d._restoring=false},0);
    d.addEventListener('toggle',function(){if(d._restoring)return;try{localStorage.setItem(k,d.open?'1':'0')}catch(e){}})})}
initSecs(document);
var hovering=false;
function initCharts(root){root.querySelectorAll('.chartwrap').forEach(function(w){
  var m=JSON.parse(w.dataset.chart),svg=w.querySelector('svg'),tip=w.querySelector('.tip'),cross=svg.querySelector('.cross');
  function fmt(f,v){if(f.indexOf('{:,')===0)return Math.round(v).toLocaleString();var d=(f.match(/\.(\d)f/)||[0,1])[1];return v.toFixed(+d)+(f.indexOf('%')>=0?'%':'')}
  function xl(x){return m.xkind==='steps'?cn(x*m.xdiv)+' 步':(+x.toFixed(2))+' '+m.xunit}
  function hide(){hovering=false;clearTimeout(w._t);tip.style.display='none';cross.style.display='none'}
  w._hide=hide;
  function show(e){hovering=true;var b=svg.getBoundingClientRect(),k=m.W/b.width,px=(e.clientX-b.left)*k;
    if(px<m.L||px>m.W-m.R){tip.style.display='none';cross.style.display='none';return}
    var x=m.x0+(px-m.L)/(m.W-m.L-m.R)*(m.x1-m.x0),rows=[],bx=null;
    m.series.forEach(function(s){var best=null;s.pts.forEach(function(p){if(!best||Math.abs(p[0]-x)<Math.abs(best[0]-x))best=p});
      if(best){if(bx===null)bx=best[0];rows.push('<i style="background:var('+s.var+')"></i>'+s.name+'：<b>'+fmt(s.fmt,best[1])+'</b>')}});
    var cx=m.L+(bx-m.x0)/(m.x1-m.x0)*(m.W-m.L-m.R);cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.style.display='';
    var near=(m.vrefs||[]).filter(function(v){return Math.abs(v[0]-x)<=(m.x1-m.x0)*0.015}).map(function(v){return '<div class="tvr">┆ '+v[1]+'</div>'});
    tip.innerHTML='<div style="color:var(--muted)">'+xl(bx)+'</div>'+rows.join('<br>')+near.join('');tip.style.display='';
    var left=(cx/k)+12;if(left+tip.offsetWidth>b.width)left=Math.max(0,(cx/k)-tip.offsetWidth-12);tip.style.left=left+'px';tip.style.top='8px'}
  // 触摸：点按或横向拖动显示，8 秒后或点到图外时收起（提示显示时暂停原地刷新）
  function touch(e){show(e);clearTimeout(w._t);w._t=setTimeout(hide,8000)}
  w.addEventListener('pointermove',function(e){if(e.pointerType==='mouse')show(e);else if(e.buttons)touch(e)});
  w.addEventListener('pointerdown',function(e){if(e.pointerType!=='mouse')touch(e)});
  w.addEventListener('pointerleave',function(e){if(e.pointerType==='mouse')hide()});
})}
document.addEventListener('pointerdown',function(e){if(e.pointerType==='mouse')return;
  document.querySelectorAll('.chartwrap').forEach(function(w){if(w._hide&&!w.contains(e.target))w._hide()})});
initCharts(document);
__SAJS__
saInit(document);
// 选牌对照的完整列表：搜索（中文、拼音、首字母，按顺序含有输入的各字即算命中）与点表头排序；原地刷新后按 CF 恢复
var CF={q:'',c:1,d:-1};
function cfHit(s,q){if(!q)return true;return s.split(' ').some(function(f){if(f.indexOf(q)>=0)return true;var i=0;for(var j=0;j<f.length&&i<q.length;j++)if(f[j]===q[i])i++;return i===q.length})}
function cfApply(root){try{cfDo(root)}catch(e){}}     // 出错也不影响看板的其余脚本
function cfDo(root){var w=root.querySelector('.cfull');if(!w)return;var tb=w.querySelector('tbody'),rs=[].slice.call(tb.rows),q=CF.q.trim().toLowerCase().replace(/\s+/g,''),n=0;
  var inp=w.querySelector('.cfq');if(inp&&inp.value!==CF.q)inp.value=CF.q;
  w.querySelectorAll('th').forEach(function(th,i){if(i===CF.c)th.dataset.dir=CF.d;else delete th.dataset.dir});
  rs.sort(function(a,b){var x=a.cells[CF.c],y=b.cells[CF.c];
    if(CF.c===0)return CF.d*x.textContent.localeCompare(y.textContent,'zh');
    var u=x.dataset.v,v=y.dataset.v;if(u===undefined)return v===undefined?0:1;if(v===undefined)return -1;return CF.d*(u-v)});
  rs.forEach(function(r){var ok=cfHit(r.dataset.s||'',q);r.hidden=!ok;n+=ok;tb.appendChild(r)});
  var c=w.querySelector('.cfn');if(c)c.textContent=q?n+' 张':''}
document.addEventListener('input',function(e){if(e.target.classList&&e.target.classList.contains('cfq')){CF.q=e.target.value;cfApply(document)}});
document.addEventListener('click',function(e){var th=e.target.closest&&e.target.closest('.cfull th');if(!th)return;var i=[].indexOf.call(th.parentNode.children,th);
  CF.d=CF.c===i?-CF.d:(i===0?1:-1);CF.c=i;cfApply(document)});
function cfBusy(){var a=document.activeElement;return !!(a&&a.classList&&a.classList.contains('cfq'))}
cfApply(document);
// 实时数字：训练量按速度外推（最多外推 30 秒，只增不减），时长、评测进度、同步状态每秒更新
var shownSteps=0;
function tick(){var app=document.getElementById('app');if(!app)return;var L=JSON.parse(app.dataset.live),now=Date.now()/1000;
  var q=function(k){return app.querySelector('[data-live="'+k+'"]')};
  var dt=Math.min(30,Math.max(0,now-L.t));shownSteps=Math.max(shownSteps,L.steps+L.sps*dt);
  if(q('steps'))q('steps').textContent=cn(shownSteps);
  if(q('elapsed'))q('elapsed').textContent=dur(L.t+(L.sps?dt:0)-L.started);
  if(q('eval')){var e=L.ev;
    if(e){var nc=e.total-e.runs,dr=e.done_runs!=null?e.done_runs:Math.min(e.done,e.runs),dc=e.done_combats!=null?e.done_combats:Math.max(0,e.done-e.runs),
      ph='<span class="ch">整局 '+dr+'/'+e.runs+'</span><span class="sep"> · </span><span class="ch">战斗 '+dc+'/'+nc+'</span>';
      q('eval').innerHTML='<span class="evt">'+ph+'</span><span class="sub"><span class="sep"> · </span><span class="ch">第 '+e.version+' 版</span></span>';q('evbar').style.width=(100*e.done/e.total).toFixed(1)+'%'}
    else{q('eval').innerHTML='<span class="sub">等待新版本</span>';q('evbar').style.width='0'}}
  if(q('sync')){var sa=now-L.pull_t,ta=now-L.t;
    q('sync').innerHTML=sa>300?'<span class="sync-warn">已 '+dur(sa)+'未同步</span>':
      (ta>420?'<span class="sync-warn">训练 '+dur(ta)+'无新数据</span>':'<span class="sync-ok">● 正常</span><span class="sub"> 每 1 分钟</span>')}
  (L.hosts||[]).forEach(function(h){var el=q('hsync-'+h.id);if(!el)return;if(h.pull_t==null){el.textContent='—';return}
    var a=now-h.pull_t;el.textContent=(a<60?Math.max(0,Math.round(a))+' 秒':dur(a))+'前';el.className=a>60?'sync-warn':''})}
tick();setInterval(tick,1000);
// 原地刷新：每 2 秒查版本戳，变了就加载内容脚本，替换页面主体（鼠标在图上时先不换）
// 内容脚本：首屏是 lines.json 的主线（def）；2026-10-28 起只有这一个视角（files 只有一项）
(function(){var mine='__STAMP__',code='__CODE__',pending=false,shown='__VIEW__',def='__VIEW__',files=__FILES__,pick=null;
  function want(){return pick&&files[pick]?pick:def}
  function swap(force){var d=window.SLS2_DATA;if(!d)return;
    if(d.code!==code){location.reload();return}      // 看板代码更新了：整页重载，换上新的脚本和样式
    files=d.files||files;def=d.def||def;var v=want();
    if(d.view!==v||(d.stamp===mine&&v===shown))return;     // 下载到的不是所选视角（刚切换或角色刚对调）：等下一次按新的 files 下载
    if(!force&&(hovering||document.hidden||cfBusy())){pending=true;return}
    var same=v===shown,app=document.getElementById('app'),old={};app.querySelectorAll('[data-k]').forEach(function(e){old[e.dataset.k]=e.textContent});
    var sc={};app.querySelectorAll('.tscroll').forEach(function(e,i){var t=e.closest('section'),h=t&&t.querySelector('.ptitle');sc[(h?h.textContent:'')+'#'+i]=[e.scrollTop,e.scrollLeft]});   // 卡片里的滚动位置：刷新后放回
    var y=scrollY;app.outerHTML=d.html;mine=d.stamp;shown=v;pending=false;initSecs(document.getElementById('app'));scrollTo(0,y);
    document.getElementById('app').querySelectorAll('.tscroll').forEach(function(e,i){var t=e.closest('section'),h=t&&t.querySelector('.ptitle'),p=sc[(h?h.textContent:'')+'#'+i];if(p){e.scrollTop=p[0];e.scrollLeft=p[1]}});document.body.dataset.swaps=(+document.body.dataset.swaps||0)+1;var na=document.getElementById('app');initCharts(na);saInit(na);cfApply(na);tick();
    if(same)na.querySelectorAll('[data-k]').forEach(function(e){if(old[e.dataset.k]!==undefined&&old[e.dataset.k]!==e.textContent)e.classList.add('flash')})}
  function load(src,cb){var s=document.createElement('script');s.src=src+'?'+Date.now();s.onload=s.onerror=function(){s.remove();cb&&cb()};document.head.appendChild(s)}
  var busy=null;      // 正在下载的内容脚本：同一个文件下载中不再重复发起
  function loadView(cb){var f=files[want()];if(!f||busy===f)return;busy=f;load(f,function(){busy=null;cb&&cb()})}
  if(want()!==shown)loadView(function(){swap(true)});
  var tried='';     // 同一个版本戳、同一个视角只下载一次（文件缺失或角色刚对调时不反复下载）
  setInterval(function(){if(pending){swap();return}load('__STAMPJS__',function(){var s=window.SLS2_STAMP,k=s+'|'+want();
    if(s&&(s!==mine||want()!==shown)&&k!==tried){tried=k;loadView(swap)}})},2000);
})();
</script></body></html>"""
PAGE = PAGE.replace("__SAJS__", ANC_SHARE_JS)


def write(src: Sources, out: Path, public: Path = PUBLIC) -> None:
    # 2026-10-28 起双机协同训练一条主线，裁掉主线 / 支线视角切换（用户：“之前的切换主线支线视角这种功能可以裁掉了”），只生成主线一个视角
    body = build(src)
    alts: dict = {}
    stamp = f"{time.time():.3f}"
    code = hashlib.md5(PAGE.encode("utf-8")).hexdigest()[:8]
    for d, page_name, data_name, data2_name, stamp_name in ((out.parent, out.name, DATA, DATA2, STAMP), (public, "index.html", "data.js", "data2.js", "stamp.js")):
        d.mkdir(parents=True, exist_ok=True)
        files = {src.main: data_name, **{h: data2_name for h in alts}}      # 视角 → 内容脚本
        data = lambda html_, view: "window.SLS2_DATA=" + json.dumps({"stamp": stamp, "code": code, "html": html_, "view": view, "def": src.main, "files": files},
                                                                   ensure_ascii=False) + ";"
        page = (PAGE.replace("__BODY__", body).replace("__STAMP__", stamp).replace("__CODE__", code).replace("__VIEW__", src.main)
                .replace("__FILES__", json.dumps(files, ensure_ascii=False)).replace("__STAMPJS__", stamp_name)
                .replace("__REPLAY__", "回放/index.html" if d == out.parent else "replay/index.html"))
        for name, text in ((page_name, page), (data_name, data(body, src.main)), *((data2_name, data(h_, h)) for h, h_ in alts.items())):
            p = d / name
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)
        (d / stamp_name).write_text(f"window.SLS2_STAMP='{stamp}';", encoding="utf-8")   # 最后写版本戳：页面看到新戳时内容已就绪


def input_sig(src: Sources) -> tuple:
    """输入文件的（名字, 修改时间, 大小）：各主机目录里的文件，加上 hosts.json、lines.json。"""
    ps = [p for d in src.dirs() if d.exists() for p in sorted(d.glob("*")) if p.is_file()] + [p for p in src.files if p.exists()]
    if (VALUE_PROBE_ROOT / "summary.json").exists(): ps.append(VALUE_PROBE_ROOT / "summary.json")
    return tuple((str(p), p.stat().st_mtime, p.stat().st_size) for p in ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="只读这一个运行目录（旧用法，当作单机，不看 hosts.json / lines.json）")
    ap.add_argument("--remote", default=str(REMOTE), help="各主机拉取目录的上级目录（含 pull*.ok、lines.json）")
    ap.add_argument("--hosts", default=str(HOSTS_FILE)); ap.add_argument("--lines", default=None, help="默认 <remote>/lines.json")
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--public", default=str(PUBLIC)); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--reset-stats", metavar="原因", nargs="?", const="手动", default=None,
                    help="把各 Boss 过关率、训练死因、先古遗物选取率的起点挪到现在（上一段留档到原目录的 训练统计留档/），然后生成一次")
    a = ap.parse_args()
    remote = Path(a.remote)
    mk = lambda: Sources(remote, Path(a.hosts), Path(a.lines) if a.lines else remote / "lines.json", Path(a.run) if a.run else None)
    out, public = Path(a.out), Path(a.public)
    if a.reset_stats is not None:
        print("留档：", reset_stats(mk(), why=a.reset_stats))
    if not a.loop:
        write(mk(), out, public)
        return
    sig_prev, last = None, 0.0
    me = Path(__file__).resolve()
    watched = (me, Path(value_probe_section.__code__.co_filename))
    my_mtime = tuple(p.stat().st_mtime_ns for p in watched)
    while True:
        if tuple(p.stat().st_mtime_ns for p in watched) != my_mtime:  # 主文件或卡片模块更新时重新加载
            os.execv(sys.executable, [sys.executable, str(me)] + sys.argv[1:])
        try:
            src = mk()                          # 每轮重读主机与训练线登记：迁移、分叉之后自动换数据来源
            poll_reset(src)
            sig = input_sig(src) + tuple((str(f), f.stat().st_mtime) for f in (STATS_FILE,) if f.exists())
        except Exception as e:
            print("读取登记失败：", e, flush=True)
            time.sleep(2)
            continue
        if sig != sig_prev or time.time() - last > 60:
            try:
                write(src, out, public)
            except Exception as e:     # 不让一次坏数据中断常驻
                print("生成失败：", e, flush=True)
            sig_prev, last = sig, time.time()
        time.sleep(2)


if __name__ == "__main__":
    main()
