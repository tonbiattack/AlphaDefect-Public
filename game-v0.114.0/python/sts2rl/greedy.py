"""按等价动作合并的贪心（第十二份外部评审，决策记录 205）。

策略给每个候选一个概率。手里有两张一模一样的牌时，“打这张牌”被拆成两个候选、各占一份概率：
两张启动流程各 31.7%、结束回合 36.7%，逐候选取最大会选结束回合，而“打一张启动流程”合计 63.4%。
训练按分布采样，同类候选被抽中的总机会就是合计概率，不受影响；只有贪心出手（评测、复测、双 Boss 卷、直播）会偏。

这里把观测上完全分不出的候选并成一类：候选的类型、动作键、特征都相同；候选引用的对象 A（牌、药水等）按内容比较
——记录类型、词表 ID、区域、附加编号、全部记录特征，以及挂在它上面的全部附加记录（附魔失效、苦痛、费用有效期、重放、副本等，
各自的 ID 与特征）；对象 B（目标生物、地图节点等）必须是同一条记录，不按内容合并（两只一样的怪站位不同，打谁不一样）。
合计概率最大的一类里，执行下标最小的那个候选。
"""
from __future__ import annotations

import numpy as np


def _content(o, j: int, kids: dict[int, list[int]]) -> tuple:
    """记录 j 的内容键：类型、ID、区域、附加编号、全部特征，加上挂在它上面的附加记录（按内容排序）。"""
    ti, tf = o.tok_ints, o.tok_feats
    att = tuple(sorted((int(ti[k, 0]), int(ti[k, 1]), tf[k].tobytes()) for k in kids.get(j, ())))
    return (int(ti[j, 0]), int(ti[j, 1]), int(ti[j, 2]), int(ti[j, 4]), tf[j].tobytes(), att)


def equivalence_classes(o) -> list[int]:
    """每个候选所属的类（类号为这一类里下标最小的候选）。"""
    n = o.num_actions
    ti = o.tok_ints
    kids: dict[int, list[int]] = {}
    for k in np.flatnonzero(ti[:, 3] >= 0):
        kids.setdefault(int(ti[k, 3]), []).append(int(k))
    ci, cf = o.cand_ints, o.cand_feats
    first: dict[tuple, int] = {}
    cls = []
    for c in range(n):
        a, b = int(ci[c, 1]), int(ci[c, 2])
        key = (int(ci[c, 0]), int(ci[c, 3]), cf[c].tobytes(), _content(o, a, kids) if a >= 0 else None, b)
        cls.append(first.setdefault(key, c))
    return cls


def merged_argmax(o, probs: np.ndarray) -> int:
    """按类合计概率取最大的一类，返回这一类里下标最小的候选；没有重复候选时与逐候选 argmax 相同。"""
    n = o.num_actions
    p = np.asarray(probs[:n], dtype=np.float64)
    cls = equivalence_classes(o)
    if len(set(cls)) == n:
        return int(np.argmax(p))
    tot = np.zeros(n)
    np.add.at(tot, np.asarray(cls), p)
    return int(np.argmax(tot))
