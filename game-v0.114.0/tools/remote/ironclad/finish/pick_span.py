# 从 keep/ 的备份里按版本均匀挑集成成员（2026-11-20，铁甲战士决策记录第 18 条；故障机器人的教训：均匀间隔、跨度越大越好，不要取最近连续的几个）。
# 免调度那段（第 22,444–22,729 版的旧时间线，备份文件名带 20261119-2 且版本 > 22443）不算。成员路径一行一个打到标准输出，挑选摘要打到标准错误。
#   python3 -I pick_span.py <起始版本> <结束版本> <个数>
import glob, os, re, sys
lo, hi, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
K = {}
for p in glob.glob("/root/.local/share/sls2/runs/ironclad/keep/v*.pt"):
    b = os.path.basename(p); v = int(re.match(r"v(\d+)-", b).group(1))
    if v > 22443 and "20261119-2" in b:
        continue
    if lo <= v <= hi:
        K.setdefault(v, p)
vs = sorted(K)
n = min(n, len(vs))        # 可选的不够时全取（第 24 条补记的 R5：加血后只有几份备份）
pick = []
for i in range(n):
    t = lo + (hi - lo) * i / max(1, n - 1)
    pick.append(min((x for x in vs if x not in pick), key=lambda x: abs(x - t)))
pick.sort()
print(f"{len(vs)} 份可选；挑出 {pick}", file=sys.stderr)
print("\n".join(K[v] for v in pick))
