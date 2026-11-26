using System.Diagnostics;

namespace Sts2Headless;

/// <summary>
/// 局面复制的进程内自检（`dotnet Sts2Headless.dll clone-test 子命令 ...`，局面复制.md 第四节）。
///   census   [combat|run] [局数]          对象图规模：拍快照记了多少对象、按类型与程序集分布、静态根、拍与恢复的耗时
///   roundtrip [combat|run] [局数] [步数]   往返：随机走到随机一步拍快照 → 随机走若干步记帧（一半先 reseed）→ 恢复 → 同样的动作再走一遍，逐字节比帧；
///                                          再恢复一次比根帧。同一进程里头部也该完全相同（头部 12、14 是静态字段，一并恢复）
///   bench    题目.json [题数] [世界数]      引擎侧测速（不含网络）：boss_pimc 的题目，开战后随机走到战斗中途作根，每个根若干个世界，
///                                          复制（恢复 + reseed + 首步 + 往下 12 步，每步编码）对原做法（开局 + 重放 + reseed + 首步 + 往下 12 步），
///                                          两条路的帧逐字节比较（跳过头部 12、14）；最后在重放打乱过局面之后再恢复一遍核对
///   audit    [combat|run] [局数] [步数]    漏记审计：同一处各拍一份“全记”（叶子也往下走）和正式的快照，走若干步后，
///                                          查被改过、却不在正式快照里的对象（正式快照恢复时会漏掉它们）
/// 有问题时退出码为 1。
/// </summary>
public static class CloneTest
{
    static string[] Chars = { "IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER", "FOOL" };
    static readonly string[] Encs = { "weak", "regular", "elite", "boss" };

    public static int Run(List<string> pos)
    {
        string what = pos.ElementAtOrDefault(1) ?? "roundtrip";
        bool combat = (pos.ElementAtOrDefault(2) ?? "combat") == "combat";
        int n = int.Parse(pos.ElementAtOrDefault(3) ?? "20");
        int steps = int.Parse(pos.ElementAtOrDefault(4) ?? "40");
        if (pos.ElementAtOrDefault(5) is { } only) Chars = only.Split(',');     // 可选：只测这些角色
        return what switch
        {
            "census" => Census(combat, n),
            "roundtrip" => Roundtrip(combat, n, steps),
            "audit" => Audit(combat, n, steps),
            "bench" => Bench(pos.ElementAtOrDefault(2) ?? "", n, steps),
            _ => throw new ArgumentException("clone-test census|roundtrip|audit [combat|run] [局数] [步数] | bench 题目.json [题数] [每个根的世界数]"),
        };
    }

    /// <summary>可选的牌组（环境变量 CLONE_TEST_DECK="卡牌ID*张数,..."）：专门覆盖某类局面（比如把牌放到抽牌堆顶的牌，测已知牌堆顶）。</summary>
    static readonly List<CardSpec>? Deck = Environment.GetEnvironmentVariable("CLONE_TEST_DECK") is { Length: > 0 } dk
        ? dk.Split(',').SelectMany(x => { var a = x.Split('*'); return Enumerable.Repeat(a[0], a.Length > 1 ? int.Parse(a[1]) : 1); }).Select(id => new CardSpec { Id = id }).ToList()
        : null;

    static RunSession Start(bool combat, int k) => combat
        ? RunSession.StartCombat(new CombatSpec { Character = Chars[k % Chars.Length], Seed = $"CT{k}", Act = k % 3, Encounter = Encs[k % 4], Deck = Deck })
        : RunSession.Start(Chars[k % Chars.Length], 10, $"CT{k}");

    static int Pick(Decision d, Random rng)
    {
        if (d.Kind == DecisionKind.Combat && d.Choices.Count > 1)
            return rng.NextDouble() < 0.15 ? d.Choices.Count - 1 : rng.Next(d.Choices.Count - 1);
        return rng.Next(d.Choices.Count);
    }

    static bool Over(Decision d) => d.Kind is DecisionKind.GameOver or DecisionKind.Stuck || d.Choices.Count == 0;

    static byte[] Frame(RunSession run, Decision d) => Observation.Build(run, d, false).Serialize();

    /// <summary>随机走到第 pre 步（整局模式下尽量停在战斗里），返回当前决策。</summary>
    static Decision Advance(RunSession run, Random rng, int pre, bool preferCombat)
    {
        var d = run.GetDecision();
        for (int i = 0; i < pre && !Over(d); i++) { run.Choose(d, Pick(d, rng)); d = run.GetDecision(); }
        for (int i = 0; preferCombat && i < 400 && !Over(d) && d.Kind != DecisionKind.Combat; i++) { run.Choose(d, Pick(d, rng)); d = run.GetDecision(); }
        return d;
    }

    static int Census(bool combat, int n)
    {
        var rng = new Random(1);
        var byType = new Dictionary<Type, int>();
        double cap = 0, res = 0; long objs = 0, visited = 0;
        int done = 0;
        for (int k = 0; k < n; k++)
        {
            using var run = Start(combat, k);
            var d = Advance(run, rng, rng.Next(combat ? 30 : 400), !combat);
            if (Over(d)) continue;
            Frame(run, d);
            var s = HeapSnapshot.Capture(run, d);
            if (k == 0) { HeapSnapshot.Capture(run, d); s = HeapSnapshot.Capture(run, d); }    // 第一次含找根与生成代码，不计
            var s2 = HeapSnapshot.Capture(run, d);
            double r = s2.Restore();
            var reps = new List<double>();
            for (int rep = 0; rep < 20; rep++) reps.Add(s2.Restore());
            reps.Sort();
            r = reps[reps.Count / 2];
            var caps = new List<(double all, double walk)>();
            for (int rep = 0; rep < 20; rep++) { var sx = HeapSnapshot.Capture(run, d); caps.Add((sx.CaptureMs, HeapSnapshot.WalkMs)); }
            caps.Sort();
            Console.WriteLine($"    拍快照（20 次中位）{caps[10].all:F2} ms，其中遍历 {caps[10].walk:F2} ms");
            cap += s2.CaptureMs; res += r; objs += s2.Objects; visited += s2.Visited; done++;
            foreach (var p in s2.Plans) byType[p.Type] = byType.GetValueOrDefault(p.Type) + 1;
            Console.WriteLine($"第 {k} 局 {(combat ? "战斗" : "整局")} 第 {RunSession.State.TotalFloor} 层 {d.Kind}：记 {s2.Objects} 个对象（走到 {s2.Visited}），拍 {s2.CaptureMs:F2} ms，恢复（20 次中位）{r:F3} ms，最快 {reps[0]:F3} ms");
        }
        Console.WriteLine($"找根 {CloneRoots.DiscoverMs:F0} ms：可写静态字段 {CloneRoots.Writable.Length} 个，只读 {CloneRoots.ReadOnly.Length} 个；静态初始化失败 {CloneRoots.InitFailures.Count} 个，有静态字段的泛型定义 {CloneRoots.GenericDefinitions.Count} 个");
        foreach (var f in CloneRoots.InitFailures.Take(30)) Console.WriteLine("  初始化失败 " + f);
        foreach (var g in CloneRoots.GenericDefinitions.Take(40)) Console.WriteLine("  泛型定义 " + g);
        var ts = CloneRoots.ThreadStatics().ToList();
        Console.WriteLine($"线程静态字段 {ts.Count} 个（不进快照；只能是一次调用内的临时量、单调计数或与局面无关的缓存，逐个确认）：");
        foreach (var f in ts) Console.WriteLine($"  {f.DeclaringType!.FullName}.{f.Name}");
        if (done > 0) Console.WriteLine($"平均：记 {objs / done} 个对象（走到 {visited / done}），拍 {cap / done:F2} ms，恢复 {res / done:F2} ms；类型计划 {TypePlans.Count} 个");
        Console.WriteLine("按类型（全部局合计，前 50）：");
        foreach (var (t, c) in byType.OrderByDescending(kv => kv.Value).Take(50)) Console.WriteLine($"  {c,8}  {Name(t)}");
        // 前 12 类对象各取一个，打印它是从哪里走到的
        var w = HeapWalker.Shared;
        {
            using var run = Start(combat, 0);
            var d = Advance(run, rng, 10, !combat);
            w.Parent = new Dictionary<object, object>(ReferenceEqualityComparer.Instance);
            var parents = w.Parent;
            var sp = HeapSnapshot.Capture(run, d);
            foreach (var (t, _) in byType.OrderByDescending(kv => kv.Value).Take(12))
            {
                var o = sp.LiveObjects.FirstOrDefault(x => x.GetType() == t);
                if (o == null) continue;
                var chain = new List<string>();
                for (int hop = 0; hop < 10 && o != null; hop++) { chain.Add(Name(o.GetType())); o = parents.GetValueOrDefault(o); }
                Console.WriteLine($"  来路 {string.Join(" ← ", chain)}");
            }
        }
        Console.WriteLine("按程序集：");
        foreach (var g in byType.GroupBy(kv => kv.Key.Assembly.GetName().Name).OrderByDescending(g => g.Sum(kv => kv.Value)))
            Console.WriteLine($"  {g.Sum(kv => kv.Value),8}  {g.Key}");
        Console.WriteLine("叶子类型：");
        foreach (var g in TypePlans.All.Where(p => p.Leaf).GroupBy(p => p.Why)) Console.WriteLine($"  {g.Key}：{g.Count()} 种");
        return 0;
    }

    static string Name(Type t) => t.IsGenericType ? $"{t.Name.Split('`')[0]}<{string.Join(",", t.GetGenericArguments().Select(Name))}>" : t.IsArray ? Name(t.GetElementType()!) + "[]" : t.Name;

    /// <summary>这一局随机打到底一共几步（同一进程里同样的开局与随机数，第二遍走法完全相同）。</summary>
    static int Length(bool combat, int k, int seed, int cap)
    {
        using var run = Start(combat, k);
        var rng = new Random(seed);
        var d = run.GetDecision();
        int n = 0;
        for (; n < cap && !Over(d); n++) { run.Choose(d, Pick(d, rng)); d = run.GetDecision(); }
        return n;
    }

    /// <summary>从当前决策随机走 steps 步（可先 reseed），返回动作与每步之后的帧。</summary>
    static (List<int> acts, List<byte[]> frames, Decision end) Walk(RunSession run, Decision d, Random rng, int steps, uint? reseed)
    {
        var acts = new List<int>(); var frames = new List<byte[]>();
        if (reseed is { } s) { RunSession.Reseed(s); d = run.GetDecision(); frames.Add(Frame(run, d)); }
        for (int i = 0; i < steps && !Over(d); i++)
        {
            int a = Pick(d, rng);
            acts.Add(a);
            run.Choose(d, a);
            d = run.GetDecision();
            frames.Add(Frame(run, d));
        }
        return (acts, frames, d);
    }

    /// <summary>照给定动作走一遍（可先 reseed），返回帧；候选对不上时返回出错说明。</summary>
    static (List<byte[]> frames, string? why) Replay(RunSession run, Decision d, List<int> acts, uint? reseed)
    {
        var frames = new List<byte[]>();
        if (reseed is { } s) { RunSession.Reseed(s); d = run.GetDecision(); frames.Add(Frame(run, d)); }
        foreach (int a in acts)
        {
            if (Over(d) || a >= d.Choices.Count) return (frames, $"第 {frames.Count} 步候选对不上（{d.Kind}，{d.Choices.Count} 项，要走 {a}）");
            run.Choose(d, a);
            d = run.GetDecision();
            frames.Add(Frame(run, d));
        }
        return (frames, null);
    }

    static string? Compare(List<byte[]> a, List<byte[]> b, ref int compared)
    {
        if (a.Count != b.Count) return $"帧数 {a.Count} 对 {b.Count}";
        for (int i = 0; i < a.Count; i++)
        {
            compared++;
            if (!a[i].AsSpan().SequenceEqual(b[i])) return $"第 {i} 帧不同（{FirstDiff(a[i], b[i])}）";
        }
        return null;
    }

    /// <summary>
    /// 往返：在随机一局里均匀抽一个决策点拍快照，然后
    ///   A：（一半先 reseed）随机走 steps 步，记动作与帧；
    ///   恢复 → 照 A 的动作走，帧要与 A 逐字节相同；
    ///   恢复 → 换一个 reseed、随机走另一条路（像搜索的另一个世界）；
    ///   恢复 → 再照 A 走，帧仍要相同（恢复前走过别的路也不影响）；
    ///   恢复 → 根帧要与拍快照前编的根帧相同。
    /// </summary>
    static int Roundtrip(bool combat, int n, int steps)
    {
        int bad = 0, compared = 0, trials = 0, restores = 0, knownTop = 0;
        double cap = 0, res = 0; long objs = 0;
        var kinds = new Dictionary<DecisionKind, int>();
        for (int k = 0; k < n; k++)
        {
            int seedA = 1000 + k;
            int len = Length(combat, k, seedA, combat ? 3000 : 6000);
            if (len < 2) continue;
            var pick = new Random(77 + k);
            int pre = pick.Next(len);
            using var run = Start(combat, k);
            var rngA = new Random(seedA);
            var d0 = run.GetDecision();
            for (int i = 0; i < pre && !Over(d0); i++) { run.Choose(d0, Pick(d0, rngA)); d0 = run.GetDecision(); }
            if (Over(d0)) continue;
            kinds[d0.Kind] = kinds.GetValueOrDefault(d0.Kind) + 1;
            var f0 = Frame(run, d0);
            if (RunSession.State.Players[0].PlayerCombatState?.DrawPile is { } dp && Observation.DrawTop.Known(dp) > 0) knownTop++;
            var snap = HeapSnapshot.Capture(run, d0);
            uint? rsA = pick.Next(2) == 0 ? (uint)pick.Next() : null;
            var (acts, framesA, endA) = Walk(run, d0, new Random(seedA * 31 + 7), steps, rsA);
            string? why = null;
            res += snap.Restore(); restores++;
            var (fb, wb) = Replay(run, (Decision)snap.Extra[1]!, acts, rsA);
            why = wb ?? Compare(framesA, fb, ref compared);
            if (why == null)
            {
                res += snap.Restore(); restores++;
                Walk(run, (Decision)snap.Extra[1]!, new Random(seedA * 31 + 8), steps, (uint)pick.Next());
                res += snap.Restore(); restores++;
                var (fc, wc) = Replay(run, (Decision)snap.Extra[1]!, acts, rsA);
                why = wc ?? Compare(framesA, fc, ref compared);
                if (why != null) why = "走过另一条路之后：" + why;
            }
            if (why == null)
            {
                res += snap.Restore(); restores++;
                compared++;
                var g0 = Frame(run, (Decision)snap.Extra[1]!);
                if (!f0.AsSpan().SequenceEqual(g0)) why = $"恢复后根帧不同（{FirstDiff(f0, g0)}）";
            }
            trials++; cap += snap.CaptureMs; objs += snap.Objects;
            if (why != null) { bad++; Console.WriteLine($"✗ 第 {k} 局（共 {len} 步，第 {pre} 步拍快照，{d0.Kind} {d0.Context}，reseed {rsA?.ToString() ?? "无"}）：{why}"); }
            else if (k < 5 || k % 20 == 0)
                Console.WriteLine($"第 {k} 局：共 {len} 步，第 {pre} 步拍快照（{d0.Kind}，第 {RunSession.State.TotalFloor} 层），记 {snap.Objects} 个对象，拍 {snap.CaptureMs:F2} ms；A 走 {acts.Count} 步、{framesA.Count} 帧，结束于 {endA.Kind}");
        }
        Console.WriteLine($"{(combat ? "战斗" : "整局")}模式 {trials} 局（拍快照处：{string.Join("、", kinds.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key} {kv.Value}"))}），比了 {compared} 帧，不同 {bad} 局；"
            + $"平均记 {(trials > 0 ? objs / trials : 0)} 个对象，拍 {(trials > 0 ? cap / trials : 0):F2} ms，恢复 {(restores > 0 ? res / restores : 0):F3} ms（含首次）；晚初始化的类型 {CloneRoots.LateInit.Count} 个；"
            + $"拍快照时抽牌堆顶有已知牌的 {knownTop} 局，已知顶部核对不一致 {Observation.DrawTop.Mismatch} 次");
        foreach (var t in CloneRoots.LateInit) Console.WriteLine($"  晚初始化 {t}");
        return bad == 0 ? 0 : 1;
    }

    static string FirstDiff(byte[] a, byte[] b)
    {
        if (a.Length != b.Length) return $"长度 {a.Length} 对 {b.Length}";
        for (int i = 0; i < a.Length; i += 4)
            if (BitConverter.ToInt32(a, i) != BitConverter.ToInt32(b, i)) return $"第 {i / 4} 个 int32：{BitConverter.ToInt32(a, i)} 对 {BitConverter.ToInt32(b, i)}";
        return "相同";
    }

    static int Bench(string path, int nItems, int worlds)
    {
        const int Leaf = 12;
        var opts = new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true, PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.SnakeCaseLower };
        var items = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path)).RootElement.GetProperty("items");
        RunSession.LoopCut = true; RunSession.StallTurns = 20;      // 同 boss_pimc 的 EXTRA
        double tRestore = 0, tStart = 0, tReplay = 0, tRollClone = 0, tRollReplay = 0, tCapture = 0;
        int nWorlds = 0, roots = 0, bad = 0, histSum = 0;
        long frames = 0;
        // 会话的先后同服务端：开新局前先释放当前会话（构造会话时要往全局挂选牌器，旧的还挂着会冲突）；
        // 恢复快照时不释放，全局状态已经整体回到拍快照时，当前会话就是快照里的那个
        RunSession? cur = null;
        RunSession Fresh(CombatSpec sp) { cur?.Dispose(); cur = null; return cur = RunSession.StartCombat(sp); }
        static bool Same(byte[] a, byte[] b) => a.Length == b.Length && a.AsSpan(0, 48).SequenceEqual(b.AsSpan(0, 48)) && a.AsSpan(52, 4).SequenceEqual(b.AsSpan(52, 4)) && a.AsSpan(60).SequenceEqual(b.AsSpan(60));
        // 一个世界：reseed + 首步 + 往下 Leaf 步（每步编码，动作由世界自己的随机数决定，两条路相同）
        List<byte[]> Roll(RunSession run, uint seed, int first, int wseed, out double ms)
        {
            var sw = Stopwatch.StartNew();
            var fr = new List<byte[]>();
            RunSession.Reseed(seed);
            var d = run.GetDecision();
            run.Choose(d, Math.Min(first, d.Choices.Count - 1));
            d = run.GetDecision();
            fr.Add(Frame(run, d));
            var rng = new Random(wseed);
            for (int i = 0; i < Leaf && !Over(d); i++) { run.Choose(d, Pick(d, rng)); d = run.GetDecision(); fr.Add(Frame(run, d)); }
            ms = sw.Elapsed.TotalMilliseconds;
            return fr;
        }
        for (int it = 0; it < Math.Min(nItems, items.GetArrayLength()); it++)
        {
            var spec = System.Text.Json.JsonSerializer.Deserialize<CombatSpec>(items[it].GetProperty("spec"), opts)!;
            var rng = new Random(it);
            // 先随机打完一遍记下动作，在有多个候选的决策点里均匀抽一个作根，再重开、照走到根
            var run = Fresh(spec);
            var all = new List<int>(); var multi = new List<int>();
            var d = run.GetDecision();
            for (int i = 0; i < 3000 && !Over(d); i++) { if (d.Choices.Count > 1) multi.Add(i); int a = Pick(d, rng); all.Add(a); run.Choose(d, a); d = run.GetDecision(); }
            if (multi.Count == 0) continue;
            var hist = all.Take(multi[rng.Next(multi.Count)]).ToList();
            run = Fresh(spec);
            d = run.GetDecision();
            foreach (int a in hist) { run.Choose(d, a); d = run.GetDecision(); }
            if (Over(d) || d.Choices.Count < 2) { Console.WriteLine($"  题 {it} 跳过：照走 {hist.Count} 步后 {d.Kind} {d.Context}（{d.Choices.Count} 项）"); continue; }
            Frame(run, d);
            var snap = HeapSnapshot.Capture(run, d);
            tCapture += snap.CaptureMs; roots++; histSum += hist.Count;
            var ws = Enumerable.Range(0, worlds).Select(w => ((uint)rng.Next(), rng.Next(d.Choices.Count), rng.Next())).ToList();
            var clone = new List<List<byte[]>>();
            foreach (var (seed, first, wseed) in ws)
            {
                tRestore += snap.Restore();
                var r = cur = (RunSession)snap.Extra[0]!;
                clone.Add(Roll(r, seed, first, wseed, out double ms)); tRollClone += ms;
            }
            for (int w = 0; w < ws.Count; w++)
            {
                var (seed, first, wseed) = ws[w];
                var sw = Stopwatch.StartNew();
                var r2 = Fresh(spec);
                tStart += sw.Elapsed.TotalMilliseconds; sw.Restart();
                var d2 = r2.GetDecision();
                foreach (int a in hist) { r2.Choose(d2, a); d2 = r2.GetDecision(); }
                tReplay += sw.Elapsed.TotalMilliseconds;
                var fr = Roll(r2, seed, first, wseed, out double ms); tRollReplay += ms;
                nWorlds++;
                frames += fr.Count;
                if (fr.Count != clone[w].Count || fr.Zip(clone[w]).Any(x => !Same(x.First, x.Second))) { bad++; Console.WriteLine($"✗ 题 {it} 世界 {w}：复制与重放的帧不同"); }
            }
            // 重放把局面打乱过以后，再从同一份快照恢复，结果仍要一样
            snap.Restore(); var rr = cur = (RunSession)snap.Extra[0]!;
            var again = Roll(rr, ws[0].Item1, ws[0].Item2, ws[0].Item3, out _);
            if (again.Count != clone[0].Count || again.Zip(clone[0]).Any(x => !Same(x.First, x.Second))) { bad++; Console.WriteLine($"✗ 题 {it}：重开别的局之后再恢复，帧不同"); }
        }
        cur?.Dispose();
        double w0 = Math.Max(1, nWorlds);
        Console.WriteLine($"{roots} 个根（平均已走 {(double)histSum / Math.Max(1, roots):F1} 步），{nWorlds} 个世界，比了 {frames} 帧，不同 {bad}");
        Console.WriteLine($"  拍快照 平均 {tCapture / Math.Max(1, roots):F2} ms/根");
        Console.WriteLine($"  复制：恢复 {tRestore / w0:F3} ms + 推演 {tRollClone / w0:F2} ms = {(tRestore + tRollClone) / w0:F2} ms/世界");
        Console.WriteLine($"  重放：开局 {tStart / w0:F2} ms + 重放 {tReplay / w0:F2} ms + 推演 {tRollReplay / w0:F2} ms = {(tStart + tReplay + tRollReplay) / w0:F2} ms/世界");
        Console.WriteLine($"  引擎侧每个世界加速 {(tStart + tReplay + tRollReplay) / Math.Max(1e-9, tRestore + tRollClone):F2} 倍");
        return bad == 0 ? 0 : 1;
    }

    static int Audit(bool combat, int n, int steps)
    {
        var missed = new Dictionary<string, int>();
        var missedStatics = new Dictionary<string, int>();
        int trials = 0;
        long changedProd = 0, totalProd = 0;
        for (int k = 0; k < n; k++)
        {
            var rng = new Random(5000 + k);
            int len = Length(combat, k, 5000 + k, combat ? 3000 : 6000);
            if (len < 2) continue;
            int pre = new Random(99 + k).Next(len);
            using var run = Start(combat, k);
            var d = run.GetDecision();
            for (int i = 0; i < pre && !Over(d); i++) { run.Choose(d, Pick(d, rng)); d = run.GetDecision(); }
            if (Over(d)) continue;
            Frame(run, d);
            var prod = HeapSnapshot.Capture(run, d);
            HeapSnapshot full;
            ClonePolicy.AuditAll = true; TypePlans.Reset();
            try { full = HeapSnapshot.Capture(run, d); }
            finally { ClonePolicy.AuditAll = false; TypePlans.Reset(); }
            var inProd = new HashSet<object>(prod.LiveObjects, ReferenceEqualityComparer.Instance);
            if (rng.Next(2) == 0) { RunSession.Reseed((uint)rng.Next()); d = run.GetDecision(); Frame(run, d); }
            for (int i = 0; i < steps && !Over(d); i++) { run.Choose(d, Pick(d, rng)); d = run.GetDecision(); Frame(run, d); }
            foreach (var (o, p) in full.Changed())
                if (!inProd.Contains(o))
                {
                    string key = Name(p.Type) + (o is MegaCrit.Sts2.Core.Models.AbstractModel m ? $"（{(m.IsCanonical ? "规范" : "可变")}模型 {m.Id.Entry}）" : "")
                        + "：" + string.Join("、", full.ChangedFieldNames(o));
                    missed[key] = missed.GetValueOrDefault(key) + 1;
                }
            changedProd += prod.Changed().Count(); totalProd += prod.Objects;
            foreach (var r in prod.ChangedStatics()) missedStatics[$"{r.Field.DeclaringType!.FullName}.{r.Field.Name}"] = missedStatics.GetValueOrDefault($"{r.Field.DeclaringType!.FullName}.{r.Field.Name}") + 1;
            trials++;
            prod.Restore();
        }
        Console.WriteLine($"正式快照里走若干步后被改过的对象 {changedProd} / {totalProd}（{100.0 * changedProd / Math.Max(1, totalProd):F1}%；范围化快照能省的恢复时间不超过其余部分）");
        Console.WriteLine($"{(combat ? "战斗" : "整局")}模式 {trials} 局：被改过、正式快照没记的对象 {missed.Values.Sum()} 个（{missed.Count} 种）");
        foreach (var (k, c) in missed.OrderByDescending(kv => kv.Value).Take(60)) Console.WriteLine($"  {c,6}  {k}");
        Console.WriteLine($"走动中值变过的可写静态字段 {missedStatics.Count} 个（这些都在快照里，会写回；列出来供核对）：");
        foreach (var (k, c) in missedStatics.OrderByDescending(kv => kv.Value).Take(60)) Console.WriteLine($"  {c,6}  {k}");
        // 只有“原为空、后来填上”的字段（模板的惰性缓存：卡池的 _allCards、规范能力的 _dynamicVars 等，值与时间线无关）不算问题
        int real = missed.Keys.Count(k => k.Split('：').Last().Split('、').Any(f => !f.EndsWith("（惰性，原为空）")));
        Console.WriteLine(real == 0 ? "漏记审计通过（只有模板的惰性缓存）" : $"漏记审计不通过：{real} 种对象有非惰性的改动");
        return real == 0 ? 0 : 1;
    }
}
