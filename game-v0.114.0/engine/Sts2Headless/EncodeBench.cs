using System.Diagnostics;
using System.Text.Json;

namespace Sts2Headless;

/// <summary>
/// 观测编码与 step 的确定性微基准（2026-11-16，引擎提速）：
///   bench-encode 题目.json 动作.json [--n 题数] [--reps 5] [--passes 3] [--fmt 格式] [--privileged] [--steps 每题最多步数]
/// 题目为 tools/future/boss_pimc.py build 的 items（每项的 spec 是 CombatSpec），动作为 tools/perf/boss_equal.py --save-traj 录下的动作序列。
/// 每题按 boss_pimc 的设置开战（战斗模式，loop_cut、stall_turns 20，不带特权；--privileged 时带），照动作序列一步步走：
///   每个决策编码 reps 次（Observation.Build + Serialize，同 Server 的 encode），取这一帧的中位数；
///   step = run.Choose（同 Server 的 step），decision = run.GetDecision（同 Server 的 decision）。
/// 整个序列跑 passes 遍（另加一遍预热不计），报告各遍合计的中位数，以及全部帧（第一遍、第一次编码；头部 12 累计胜场清零）的 SHA-256：
/// 改编码器前后这个哈希必须相同（逐字节不变的快速自查；正式对拍仍用 tools/engine_equal.py 与 tools/perf/boss_equal.py）。
/// 同一帧重复编码得到的字节不同、动作越界时都报出来。
/// </summary>
public static class EncodeBench
{
    static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct Timespec { public long Sec, Nsec; }
    [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "clock_gettime")]
    static extern int ClockGetTime(int clk, out Timespec ts);
    /// <summary>本线程用掉的 CPU 时间（毫秒，CLOCK_THREAD_CPUTIME_ID）：机器上别的进程抢 CPU 时比墙钟稳（被抢走的时间不算）。</summary>
    static double CpuMs() { ClockGetTime(3, out var t); return t.Sec * 1e3 + t.Nsec / 1e6; }

    static string Arg(string[] args, string name, string def)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : def;
    }

    public static int Run(string[] args, List<string> pos)
    {
        var itemsDoc = JsonDocument.Parse(File.ReadAllText(pos[1]));
        var trajDoc = JsonDocument.Parse(File.ReadAllText(pos[2]));
        int n = int.Parse(Arg(args, "--n", "16")), reps = int.Parse(Arg(args, "--reps", "5")), passes = int.Parse(Arg(args, "--passes", "3"));
        int maxSteps = int.Parse(Arg(args, "--steps", "100000"));
        string fmt = Arg(args, "--fmt", "");
        var specs = new Dictionary<int, CombatSpec>();
        foreach (var it in itemsDoc.RootElement.GetProperty("items").EnumerateArray())
            specs[it.GetProperty("id").GetInt32()] = it.GetProperty("spec").Deserialize<CombatSpec>(JsonOptions)!;
        var trajs = trajDoc.RootElement.GetProperty("trajs").EnumerateArray()
            .Select(t => (id: t.GetProperty("id").GetInt32(), acts: t.GetProperty("actions").EnumerateArray().Select(x => x.GetInt32()).Take(maxSteps).ToArray()))
            .Take(n).ToList();

        RunSession.LoopCut = true; RunSession.StallTurns = 20;
        if (args.Contains("--micro")) return Micro(specs[trajs[0].id], trajs[0].acts, int.Parse(Arg(args, "--at", "20")));
        // --off 开关,开关：整个进程关掉这些提速（与不带的进程对照进程级的影响：GC、JIT 分层）
        foreach (var nm in Arg(args, "--off", "").Split(',', StringSplitOptions.RemoveEmptyEntries))
            if (Switches.TryGetValue(nm, out var sw0)) sw0(false); else { Console.WriteLine($"未知开关 {nm}"); return 2; }
        if (Arg(args, "--ab", "") is { Length: > 0 } ab) return RunAB(specs, trajs, ab.Split(','), reps, passes);
        if (args.Contains("--step-only")) return StepOnly(specs, trajs, passes, int.Parse(Arg(args, "--enc-every", "0")));
        Observation.RouteSummary = false; Observation.FutureContent = false; Observation.Oracle = false;
        Observation.Privileged = args.Contains("--privileged");
        Observation.FormatOverride = fmt.Length > 0 ? int.Parse(fmt) : null;

        // --watch：给几个热点方法计次（Profiler.Watch），只统计编码期间的次数，报每帧平均（计时有额外开销，只看次数）
        var encCounts = new Dictionary<string, (double ms, int n)>();
        if (args.Contains("--watch"))
        {
            var t = typeof(MegaCrit.Sts2.Core.Combat.CombatState);
            Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Combat.CombatState), "IterateHookListeners"));
            Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Runs.RunState), "IterateHookListeners"));
            Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.CardModel), "UpdateDynamicVarPreview"));
            Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.CardModel), "CanPlay", Type.EmptyTypes));
            Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.CardModel), "GetKeywordsWithSources"));
            Profiler.Watch(HarmonyLib.AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Models.CardModel), "Keywords"));
            foreach (var name in new[] { "ModifyDamage", "ModifyHpLost", "ModifyBlock", "ModifyCardPlayCount", "ModifyAttackHitCount", "ModifyXValue", "ShouldDraw", "ModifyEnergyCostInCombat", "ModifyKeywordsInCombat" })
                foreach (var m in typeof(MegaCrit.Sts2.Core.Hooks.Hook).GetMethods().Where(m => m.Name == name)) Profiler.Watch(m);
            Profiler.Watch(HarmonyLib.AccessTools.PropertyGetter(t, "HittableEnemies"));
        }
        var encTotals = new List<double>(); var stepTotals = new List<double>(); var decTotals = new List<double>();
        var encCpuTotals = new List<double>(); var stepCpuTotals = new List<double>();
        string? firstHash = null; int frames = 0, unstable = 0, desync = 0;
        long bytesTotal = 0, tokTotal = 0, candTotal = 0;
        var perFrame = new List<double>();      // 最后一遍每帧的中位数（看分布）
        for (int p = 0; p <= passes; p++)
        {
            using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
            double enc = 0, step = 0, dec = 0, encCpu = 0, stepCpu = 0;
            var tc = new double[reps];
            long encAlloc = 0, stepAlloc = 0; var pause0 = GC.GetTotalPauseDuration(); double encPause = 0;
            int fr = 0;
            var ts = new double[reps];
            if (p == passes) perFrame.Clear();

            var gc0 = GC.CollectionCount(0);
            foreach (var (id, acts) in trajs)
            {
                using var run = RunSession.StartCombat(specs[id]);
                var sw = new Stopwatch();
                for (int k = 0; ; k++)
                {
                    sw.Restart();
                    var d = run.GetDecision();
                    dec += sw.Elapsed.TotalMilliseconds;
                    byte[]? first = null;
                    var before = p == 1 && Profiler.Totals.Count > 0 ? new Dictionary<string, (double ms, int n)>(Profiler.Totals) : null;
                    for (int r = 0; r < reps; r++)
                    {
                        long a0 = GC.GetAllocatedBytesForCurrentThread(); var gp0 = GC.GetTotalPauseDuration();
                        double c0 = CpuMs();
                        sw.Restart();
                        var o = Observation.Build(run, d, false);
                        var bytes = o.Serialize();
                        ts[r] = sw.Elapsed.TotalMilliseconds;
                        tc[r] = CpuMs() - c0;
                        if (r == 0) { encAlloc += GC.GetAllocatedBytesForCurrentThread() - a0; encPause += (GC.GetTotalPauseDuration() - gp0).TotalMilliseconds; }
                        if (r == 0)
                        {
                            first = bytes;
                            if (p == 0) { tokTotal += o.NumTokens; candTotal += o.NumCands; bytesTotal += bytes.Length; }
                        }
                        else if (p == 0 && !bytes.AsSpan().SequenceEqual(first)) unstable++;
                    }
                    if (before != null)
                        foreach (var (key, (ms, cnt)) in Profiler.Totals)
                        {
                            var (bms, bn) = before.GetValueOrDefault(key);
                            var (ems, en) = encCounts.GetValueOrDefault(key);
                            encCounts[key] = (ems + (ms - bms) / reps, en + (cnt - bn) / reps);
                        }
                    Array.Sort(ts); Array.Sort(tc);
                    double med = ts[reps / 2];
                    enc += med; encCpu += tc[reps / 2]; fr++;
                    if (p == passes) perFrame.Add(med);
                    // 头部 12 是本进程累计胜场（UiFlows.CombatsWon），每遍不同；哈希时清零，各遍可以互相核对
                    first![48] = first[49] = first[50] = first[51] = 0;
                    hash.AppendData(BitConverter.GetBytes(first.Length)); hash.AppendData(first);
                    if (k >= acts.Length || d.Kind is DecisionKind.GameOver or DecisionKind.Stuck || d.Choices.Count == 0) break;
                    int a = acts[k];
                    if (a >= d.Choices.Count) { desync++; break; }
                    long s0 = GC.GetAllocatedBytesForCurrentThread();
                    double sc0 = CpuMs();
                    sw.Restart();
                    run.Choose(d, a);
                    step += sw.Elapsed.TotalMilliseconds;
                    stepCpu += CpuMs() - sc0;
                    stepAlloc += GC.GetAllocatedBytesForCurrentThread() - s0;
                }
            }
            string h = Convert.ToHexString(hash.GetHashAndReset())[..16];
            if (p == 0) { firstHash = h; frames = fr; }
            else if (h != firstHash) Console.WriteLine($"  第 {p} 遍的帧哈希 {h} 与第 0 遍 {firstHash} 不同（编码有残留状态？）");
            Console.WriteLine($"  第 {p} 遍{(p == 0 ? "（预热，不计）" : "")}：{fr} 帧，编码 {enc:F0} ms（每帧 {enc / fr:F3}，CPU {encCpu / fr:F3}），step {step:F0} ms（CPU {stepCpu:F0}），decision {dec:F0} ms，GC0 {GC.CollectionCount(0) - gc0} 次，GC 暂停 {(GC.GetTotalPauseDuration() - pause0).TotalMilliseconds:F0} ms（编码第一次里 {encPause:F0} ms），每帧编码分配 {encAlloc / fr / 1024.0:F0} KB，每步 step 分配 {stepAlloc / Math.Max(1, fr - trajs.Count) / 1024.0:F0} KB");
            if (p > 0) { encTotals.Add(enc); stepTotals.Add(step); decTotals.Add(dec); encCpuTotals.Add(encCpu); stepCpuTotals.Add(stepCpu); }
        }
        static double Med(List<double> xs) { var s = xs.OrderBy(x => x).ToList(); return s[s.Count / 2]; }
        int steps = frames - trajs.Count;
        perFrame.Sort();
        Console.WriteLine($"帧哈希 {firstHash}  {trajs.Count} 题 {frames} 帧（平均 {tokTotal / (double)frames:F0} 条记录、{candTotal / (double)frames:F1} 个候选、{bytesTotal / frames} 字节）  重复编码不同 {unstable}  动作越界 {desync}");
        Console.WriteLine($"编码 每帧 {Med(encTotals) / frames:F3} ms（中位帧 {perFrame[perFrame.Count / 2]:F3}，p90 {perFrame[(int)(perFrame.Count * 0.9)]:F3}，最大 {perFrame[^1]:F3}）  " +
                          $"step 每步 {Med(stepTotals) / Math.Max(1, steps):F3} ms  decision 每次 {Med(decTotals) / frames:F3} ms  （{passes} 遍中位数，每帧编码 {reps} 次取中位数）");
        foreach (var (key, (ms, cnt)) in encCounts.OrderByDescending(kv => kv.Value.n))
            Console.WriteLine($"  每帧编码调用 {key,-50} {cnt / (double)frames,8:F1} 次（计时含开销 {ms / frames:F3} ms）");
        Console.WriteLine($"线程 CPU 时间：编码 每帧 {Med(encCpuTotals) / frames:F3} ms  step 每步 {Med(stepCpuTotals) / Math.Max(1, steps):F3} ms");
        return unstable + desync > 0 ? 1 : 0;
    }

    /// <summary>--micro：走到第 at 步，单测几个游戏原语的耗时（每次调用的微秒数），看钩子监听者遍历本身多贵。</summary>
    static int Micro(CombatSpec spec, int[] acts, int at)
    {
        using var run = RunSession.StartCombat(spec);
        for (int k = 0; k < at && k < acts.Length; k++) { var d0 = run.GetDecision(); run.Choose(d0, acts[k]); }
        var d = run.GetDecision();
        var rs = RunSession.State;
        var me = rs.Players[0];
        var cs = (MegaCrit.Sts2.Core.Combat.CombatState)me.Creature.CombatState!;
        var hand = me.PlayerCombatState!.Hand.Cards.OrderByDescending(x => x.DynamicVars.ContainsKey("Damage") ? 1 : 0).ToList();
        var foe = cs.Enemies.FirstOrDefault(e => e.IsAlive);
        int nl = cs.IterateHookListeners().Count(), nr = rs.IterateHookListeners(cs).Count();
        Console.WriteLine($"第 {at} 步：战斗监听者 {nl}，整局监听者 {nr}，手牌 {hand.Count}，候选 {d.Choices.Count}");
        bool print = false;
        void T(string name, int n, Action f)
        {
            if (!print) { var w = Stopwatch.StartNew(); while (w.ElapsedMilliseconds < 300) f(); return; }
            var best = double.MaxValue;
            for (int rep = 0; rep < 5; rep++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < n; i++) f();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / n);
            }
            Console.WriteLine($"  {name,-48} {best,8:F2} µs");
        }
        int sink = 0;
        for (int round = 0; round < 2; round++, print = true) {
        T("CombatState.IterateHookListeners 遍历", 20000, () => { foreach (var x in cs.IterateHookListeners()) sink++; });
        T("RunState.IterateHookListeners(cs) 遍历", 10000, () => { foreach (var x in rs.IterateHookListeners(cs)) sink++; });
        if (hand.Count > 0)
        {
            var c = hand[0];
            T($"Keywords（{c.Id.Entry}）", 20000, () => sink += c.Keywords.Count);
            T($"UpdateDynamicVarPreview(null)", 5000, () => c.UpdateDynamicVarPreview(MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode.Normal, null, c.DynamicVars));
            if (foe != null) T($"UpdateDynamicVarPreview(foe)", 5000, () => c.UpdateDynamicVarPreview(MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode.Normal, foe, c.DynamicVars));
            T($"CanPlay", 5000, () => sink += c.CanPlay() ? 1 : 0);
            T($"EnergyCost.GetWithModifiers(All)", 20000, () => sink += c.EnergyCost.GetWithModifiers(MegaCrit.Sts2.Core.Entities.Cards.CostModifiers.All));
            T($"Hook.ModifyDamage(All)", 5000, () => sink += (int)MegaCrit.Sts2.Core.Hooks.Hook.ModifyDamage(rs, cs, foe, me.Creature, 6m, MegaCrit.Sts2.Core.ValueProps.ValueProp.Move, c, null, MegaCrit.Sts2.Core.Hooks.ModifyDamageHookType.All, MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode.None, out _));
            T($"Hook.ModifyCardPlayCount", 20000, () => sink += MegaCrit.Sts2.Core.Hooks.Hook.ModifyCardPlayCount(cs, c, 1, foe, out _));
            c.UpdateDynamicVarPreview(MegaCrit.Sts2.Core.Entities.Cards.CardPreviewMode.Normal, null, c.DynamicVars);
        }
        T("HittableEnemies", 20000, () => sink += cs.HittableEnemies.Count);
        T("Observation.Build + Serialize", 200, () => sink += Observation.Build(run, d, false).Serialize().Length);
        }
        Console.WriteLine($"（{sink & 1}）");
        return 0;
    }

    /// <summary>可在运行时开关的提速（--ab 用）。</summary>
    static readonly Dictionary<string, Action<bool>> Switches = new()
    {
        ["pool"] = on => HookListenerPool.Enabled = on,
        ["memo"] = on => Observation.MemoEnabled = on,
        ["slug"] = on => SlugifyCache.Enabled = on,
        ["freeze"] = on => HookListenerPool.FreezeEnabled = on,
    };

    /// <summary>
    /// --ab 开关,开关：同一进程里交替关（A）、开（B）这些提速，对照耗时，并核对两边的帧逐字节相同（同一局面上）。
    ///   编码：每帧交替 A、B 各编码 reps 次，各取中位数；两边每次的字节都要与 A 的第一次相同。
    ///   step / decision：每题按 A、B 各走一遍整条动作序列（题号奇偶交替先后），每帧照常编码一次（不计时），两边的帧哈希要相同。
    /// 交替的粒度细，机器负载的起伏两边一样，比两次进程分开测稳得多。报 passes 遍的中位数。
    /// </summary>
    static int RunAB(Dictionary<int, CombatSpec> specs, List<(int id, int[] acts)> trajs, string[] names, int reps, int passes)
    {
        foreach (var nm in names) if (!Switches.ContainsKey(nm)) { Console.WriteLine($"未知开关 {nm}（可选 {string.Join(",", Switches.Keys)}）"); return 2; }
        if (names.Contains("pool") && !HookListenerPool.Installed) Console.WriteLine("注意：钩子监听者列表复用没装（STS2_NO_PERF / 指纹不符），pool 开关无效");
        void Set(bool on) { foreach (var nm in names) Switches[nm](on); }
        var encA = new List<double>(); var encB = new List<double>(); var stA = new List<double>(); var stB = new List<double>(); var dcA = new List<double>(); var dcB = new List<double>();
        int mism = 0, hashMism = 0, frames = 0, steps = 0;
        for (int p = 0; p <= passes; p++)
        {
            double ea = 0, eb = 0, sa = 0, sb = 0, da = 0, db = 0;
            int fr = 0, st = 0;
            var ta = new double[reps]; var tb = new double[reps];
            // 编码：步子按 B 走（开着提速），每帧交替编码
            Set(true);
            foreach (var (id, acts) in trajs)
            {
                using var run = RunSession.StartCombat(specs[id]);
                for (int k = 0; ; k++)
                {
                    var d = run.GetDecision();
                    byte[]? refBytes = null;
                    for (int r = 0; r < reps; r++)
                    {
                        for (int side = 0; side < 2; side++)
                        {
                            bool on = (r + side) % 2 == 1;      // A、B 轮流先
                            Set(on);
                            var sw = Stopwatch.StartNew();
                            var bytes = Observation.Build(run, d, false).Serialize();
                            double t = sw.Elapsed.TotalMilliseconds;
                            (on ? tb : ta)[r] = t;
                            if (refBytes == null) refBytes = bytes;
                            else if (!bytes.AsSpan().SequenceEqual(refBytes)) mism++;
                        }
                    }
                    Set(true);
                    Array.Sort(ta); Array.Sort(tb);
                    ea += ta[reps / 2]; eb += tb[reps / 2]; fr++;
                    if (k >= acts.Length || d.Kind is DecisionKind.GameOver or DecisionKind.Stuck || d.Choices.Count == 0) break;
                    run.Choose(d, acts[k]);
                }
            }
            // step / decision：每题 A、B 各走一遍
            for (int i = 0; i < trajs.Count; i++)
            {
                var (id, acts) = trajs[i];
                string? h0 = null;
                for (int side = 0; side < 2; side++)
                {
                    bool on = (i + side) % 2 == 1;
                    Set(on);
                    using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
                    using var run = RunSession.StartCombat(specs[id]);
                    var sw = new Stopwatch();
                    for (int k = 0; ; k++)
                    {
                        sw.Restart();
                        var d = run.GetDecision();
                        if (on) db += sw.Elapsed.TotalMilliseconds; else da += sw.Elapsed.TotalMilliseconds;
                        var bytes = Observation.Build(run, d, false).Serialize();
                        bytes[48] = bytes[49] = bytes[50] = bytes[51] = 0;      // 头部 12 累计胜场
                        hash.AppendData(bytes);
                        if (k >= acts.Length || d.Kind is DecisionKind.GameOver or DecisionKind.Stuck || d.Choices.Count == 0) break;
                        sw.Restart();
                        run.Choose(d, acts[k]);
                        if (on) sb += sw.Elapsed.TotalMilliseconds; else sa += sw.Elapsed.TotalMilliseconds;
                        if (side == 0) st++;
                    }
                    string h = Convert.ToHexString(hash.GetHashAndReset());
                    if (h0 == null) h0 = h; else if (h != h0) hashMism++;
                }
            }
            Set(true);
            Console.WriteLine($"  第 {p} 遍{(p == 0 ? "（预热，不计）" : "")}：{fr} 帧  编码 A {ea / fr:F3} / B {eb / fr:F3} ms（B/A {eb / ea:F3}）  step A {sa / st:F3} / B {sb / st:F3}（{sb / sa:F3}）  decision A {da / fr:F3} / B {db / fr:F3}");
            if (p == 0) { frames = fr; steps = st; continue; }
            encA.Add(ea); encB.Add(eb); stA.Add(sa); stB.Add(sb); dcA.Add(da); dcB.Add(db);
        }
        static double Med(List<double> xs) { var s = xs.OrderBy(x => x).ToList(); return s[s.Count / 2]; }
        Console.WriteLine($"A/B（{string.Join(",", names)}）{trajs.Count} 题 {frames} 帧 {steps} 步：编码每帧 A {Med(encA) / frames:F3} → B {Med(encB) / frames:F3} ms（×{Med(encB) / Med(encA):F3}）  " +
                          $"step 每步 A {Med(stA) / steps:F3} → B {Med(stB) / steps:F3} ms（×{Med(stB) / Med(stA):F3}）  decision A {Med(dcA) / frames:F3} → B {Med(dcB) / frames:F3}");
        Console.WriteLine($"A、B 帧不同 {mism} 次（同一局面上逐字节），整条序列帧哈希不同 {hashMism} 题");
        return mism + hashMism > 0 ? 1 : 0;
    }

    /// <summary>--step-only：只走动作序列、不编码（--enc-every k：每 k 步编码一次，不计时），看 step / decision 本身的耗时，与编码在两步之间做了多少无关。</summary>
    static int StepOnly(Dictionary<int, CombatSpec> specs, List<(int id, int[] acts)> trajs, int passes, int encEvery)
    {
        var st = new List<double>(); var dc = new List<double>();
        int steps = 0, frames = 0;
        for (int p = 0; p <= passes; p++)
        {
            double s = 0, dd = 0; int n = 0, f = 0;
            int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2); var gp = GC.GetTotalPauseDuration(); long al = GC.GetAllocatedBytesForCurrentThread();
            double stepGcPause = 0;
            foreach (var (id, acts) in trajs)
            {
                using var run = RunSession.StartCombat(specs[id]);
                var sw = new Stopwatch();
                for (int k = 0; ; k++)
                {
                    sw.Restart();
                    var d = run.GetDecision();
                    dd += sw.Elapsed.TotalMilliseconds; f++;
                    if (encEvery > 0 && k % encEvery == 0) Observation.Build(run, d, false).Serialize();
                    if (k >= acts.Length || d.Kind is DecisionKind.GameOver or DecisionKind.Stuck || d.Choices.Count == 0) break;
                    var gps = GC.GetTotalPauseDuration();
                    sw.Restart();
                    run.Choose(d, acts[k]);
                    s += sw.Elapsed.TotalMilliseconds; n++;
                    stepGcPause += (GC.GetTotalPauseDuration() - gps).TotalMilliseconds;
                }
            }
            Console.WriteLine($"  第 {p} 遍{(p == 0 ? "（预热，不计）" : "")}：step 每步 {s / n:F3} ms，decision 每次 {dd / f:F3} ms，GC 0/1/2 {GC.CollectionCount(0) - g0}/{GC.CollectionCount(1) - g1}/{GC.CollectionCount(2) - g2} 次，暂停 {(GC.GetTotalPauseDuration() - gp).TotalMilliseconds:F0} ms（step 里 {stepGcPause:F0}），分配 {(GC.GetAllocatedBytesForCurrentThread() - al) / 1048576.0:F0} MB");
            if (p > 0) { st.Add(s); dc.Add(dd); }
            steps = n; frames = f;
        }
        static double Med(List<double> xs) { var s = xs.OrderBy(x => x).ToList(); return s[s.Count / 2]; }
        Console.WriteLine($"只走动作（每 {encEvery} 步编码一次）：step 每步 {Med(st) / steps:F3} ms，decision 每次 {Med(dc) / frames:F3} ms");
        return 0;
    }
}
