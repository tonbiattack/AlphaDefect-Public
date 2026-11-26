using System.Diagnostics;
using System.Runtime.CompilerServices;
using Sts2Headless;

// 管道模式：标准输出只用于协议帧，其余输出都写标准错误。必须在碰到游戏程序集之前设置：
// v0.111.0 起 sts2.dll 的模块初始化器（Sentry）会往 Console.Out 打一行字，晚了就会混进协议帧，Python 端读到乱的帧长后一直等
if (args.FirstOrDefault(a => !a.StartsWith('-')) == "serve") Console.SetOut(Console.Error);
GameAssemblies.Register();
Cli.Run(args);

static class Cli
{
    static readonly string[] Chars = { "IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER", "FOOL" };   // v0.113.0 加了第六个角色 FOOL

    [MethodImpl(MethodImplOptions.NoInlining)]
    public static void Run(string[] args)
    {
        GodotShim.Verbose = args.Contains("-v");
        bool god = args.Contains("--god");
        string mode = args.Where(a => !a.StartsWith('-')).FirstOrDefault() ?? "one";
        // 自检（游戏更新后第一步，见 引擎模块与游戏更新适配.md）：补丁找不到目标时登记下来继续，最后一次列全
        if (mode == "selfcheck") GameCompat.SelfCheck = true;
        if (args.Contains("--first-chance"))
            AppDomain.CurrentDomain.FirstChanceException += (_, e) =>
            {
                if (e.Exception is not NullReferenceException) return;
                var st = new StackTrace(e.Exception, true);
                Console.Error.WriteLine("[first-chance] " + string.Join(" | ", st.GetFrames().Take(3).Select(f => $"{f.GetMethod()?.DeclaringType?.Name}.{f.GetMethod()?.Name}@IL{f.GetILOffset():X}")));
            };
        var pos = args.Where(a => !a.StartsWith('-')).ToList();
        var sw = Stopwatch.StartNew();
        if (mode == "serve")
        {
            Bootstrap.Init();
            Vocab.Load();
            if (Vocab.Count <= 2)
            {
                // 只剩 <pad>、<unk>：词表没载入，帧里的内容全是 <unk>，训练、评测都会悄悄出错，不如直接退出（见 Vocab.Load）
                Console.Error.WriteLine($"[serve] 词表没有载入（{Vocab.DefaultPath}），退出");
                Environment.Exit(3);
            }
            Console.Error.WriteLine($"[serve] 就绪，启动 {sw.ElapsedMilliseconds} ms，词表 {Vocab.Count} 项");
            Server.Run();
            return;
        }
        Bootstrap.Init();
        Vocab.Load();
        Console.WriteLine($"启动 {sw.ElapsedMilliseconds} ms");

        switch (mode)
        {
            case "one":
            {
                string c = pos.ElementAtOrDefault(1) ?? "DEFECT";
                var r = PlayOne(c, 10, pos.ElementAtOrDefault(2) ?? "M1TEST1", int.Parse(pos.ElementAtOrDefault(3) ?? "1"), god, trace: true);
                Console.WriteLine(r);
                break;
            }
            case "runs":
            {
                int n = int.Parse(pos.ElementAtOrDefault(1) ?? "20");
                foreach (string c in Chars)
                {
                    var rs = Enumerable.Range(0, n).Select(i => PlayOne(c, 10, $"RUN{i}", i, god, trace: false)).ToList();
                    var stuck = rs.Where(r => r.Stuck).ToList();
                    Console.WriteLine($"{c,-12} 局数 {n} 胜 {rs.Count(r => r.Won)} 卡住 {stuck.Count} 平均层数 {rs.Average(r => r.Floor):F1} 平均决策 {rs.Average(r => r.Steps):F0} 每局 {rs.Average(r => r.Ms):F0} ms 每决策 {rs.Sum(r => r.Ms) / rs.Sum(r => r.Steps):F3} ms");
                    foreach (var r in stuck.Take(3)) Console.WriteLine($"    卡住：{r.Seed} {r.Last}");
                    if (GodotShim.Errors.Count > 0)
                    {
                        Console.WriteLine($"    游戏报错 {GodotShim.Errors.Values.Sum()} 条：");
                        foreach (var kv in GodotShim.Errors.OrderByDescending(kv => kv.Value).Take(5)) Console.WriteLine($"      {kv.Value}× {kv.Key}");
                        GodotShim.Errors.Clear();
                    }
                    Console.WriteLine($"    决策分布：{string.Join(" ", rs.SelectMany(r => r.Counts).GroupBy(kv => kv.Key).OrderBy(g => g.Key).Select(g => $"{g.Key}={g.Sum(kv => kv.Value)}"))}");
                    Console.WriteLine($"    少见流程：{string.Join(" ", rs.SelectMany(r => r.Tags).GroupBy(kv => kv.Key).OrderBy(g => g.Key).Select(g => $"{g.Key}={g.Sum(kv => kv.Value)}"))}");
                }
                break;
            }
            case "determinism":
                foreach (string c in Chars)
                {
                    var a = PlayOne(c, 10, "DET42", 7, god, false);
                    var b = PlayOne(c, 10, "DET42", 7, god, false);
                    var x = PlayOne(c, 10, "DET43", 7, god, false);
                    Console.WriteLine($"{c,-12} 同种子一致：{a.Fingerprint == b.Fingerprint}（{a.Fingerprint[..16]}，{a.Steps} 步）  换种子不同：{a.Fingerprint != x.Fingerprint}");
                }
                break;
            case "audit-testmode":
                foreach (string m in FidelityPatches.Audit(typeof(MegaCrit.Sts2.Core.Runs.RunManager).Assembly)) Console.WriteLine(m);
                break;
            case "catalog":
            {
                // 追加词表并导出内容目录：catalog [版本]
                string version = pos.ElementAtOrDefault(1) ?? AppContext.GetData("Sts2GameVersion") as string ?? "v0.111.0";      // 默认取编译时的游戏版本（engine/game_version.txt）
                Vocab.Load();
                int added = Catalog.ExtendVocab();
                Vocab.Save(Vocab.DefaultPath);
                string dir = Path.Combine(Path.GetDirectoryName(Vocab.DefaultPath)!, version);
                Directory.CreateDirectory(dir);
                Catalog.Export(Path.Combine(dir, "catalog.json"), version);
                Console.WriteLine($"词表 {Vocab.Count} 项（新增 {added}），目录写入 {dir}");
                break;
            }
            case "sim-tables":
                // sts2-sim 观测移植查的静态表（卡牌、遗物、能力、药水、附魔的 DynamicVars）：sim-tables [输出路径]
                SimTables.Export(pos.ElementAtOrDefault(1) ?? "sim_tables.json");
                break;
            case "obs-tables":
                // Rust 观测编码查的静态表：obs-tables [输出路径]
                ObsTables.Export(pos.ElementAtOrDefault(1) ?? "obs_tables.json");
                break;
            case "solve-test":
            {
                // 用 Combat Solver 模组试解一场战斗（只用于评估）：solve-test 配置.jsonl 行号 [--timeout-s 秒]，见 SolverHarness
                int ti = Array.IndexOf(args, "--timeout-s");
                SolverHarness.Run(pos[1], int.Parse(pos[2]), ti >= 0 ? double.Parse(args[ti + 1]) : 600);
                break;
            }
            case "bench-combat":
            {
                // 进程内随机打战斗（含观测编码），用于剖析：bench-combat [场数]
                int n = int.Parse(pos.ElementAtOrDefault(1) ?? "300");
                string? only = pos.ElementAtOrDefault(2);   // 可选：只测一个角色
                bool encode = !args.Contains("--no-encode");
                var rng = new Random(0);
                string[] encs = { "weak", "regular", "elite", "boss" };
                var swb = Stopwatch.StartNew();
                int steps = 0;
                for (int k = 0; k < n; k++)
                {
                    using var run = RunSession.StartCombat(new CombatSpec { Character = only ?? Chars[k % 5], Seed = $"P{k}", Act = k % 3, Encounter = encs[k % 4] });
                    for (var d = run.GetDecision(); d.Kind != DecisionKind.GameOver && d.Kind != DecisionKind.Stuck; d = run.GetDecision())
                    {
                        if (encode) Observation.Build(run, d, false).Serialize();
                        int j = d.Choices.Count > 1 && rng.NextDouble() > 0.15 ? rng.Next(d.Choices.Count - 1) : d.Choices.Count - 1;
                        run.Choose(d, j);
                        steps++;
                    }
                }
                Console.WriteLine($"{n} 场 {steps} 步，{swb.Elapsed.TotalSeconds:F1} s，每步 {swb.Elapsed.TotalMilliseconds / steps:F3} ms");
                break;
            }
            case "bench-encode":
                // 观测编码与 step 的确定性微基准：bench-encode 题目.json 动作.json [--n 题数] [--reps 5] [--passes 3] [--fmt 格式] [--privileged]（见 EncodeBench）
                Environment.ExitCode = EncodeBench.Run(args, pos);
                break;
            case "mapgen-check":
            {
                // 对 N 个种子 × 3 幕生成地图，输出摘要的哈希与耗时；分别在有/无性能补丁（STS2_NO_PERF=1）下运行后比较
                int n = int.Parse(pos.ElementAtOrDefault(1) ?? "200");
                var swm = Stopwatch.StartNew();
                using var sha = System.Security.Cryptography.SHA256.Create();
                var all = new System.Text.StringBuilder();
                double genMs = 0;
                for (int k = 0; k < n; k++)
                {
                    var t = Pump.Instance.Start(() => HeadlessRun.CreateRun("DEFECT", 10, $"MAP{k}"));
                    if (t.IsFaulted) throw t.Exception!;
                    var st = MegaCrit.Sts2.Core.Runs.RunManager.Instance.DebugOnlyGetState()!;
                    for (int a = 0; a < st.Acts.Count; a++)
                    {
                        st.CurrentActIndex = a;
                        var g = Stopwatch.StartNew();
                        var map = st.Act.CreateMap(st, replaceTreasureWithElites: false);
                        genMs += g.Elapsed.TotalMilliseconds;
                        all.Append(MapGenPatch.Digest(map)).Append('#');
                    }
                    MegaCrit.Sts2.Core.Runs.RunManager.Instance.CleanUp();
                }
                string digest = Convert.ToHexString(sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(all.ToString())))[..16];
                Console.WriteLine($"{n} 个种子 × 3 幕，地图摘要 {digest}，平均每幕生成 {genMs / (3 * n):F2} ms");
                break;
            }
            case "probe":
            {
                Console.WriteLine($"NGame.Instance={(MegaCrit.Sts2.Core.Nodes.NGame.Instance == null ? "null" : "fake")}");
                try { MegaCrit.Sts2.Core.Nodes.NGame.Instance!.ScreenShakeTrauma(MegaCrit.Sts2.Core.Nodes.Vfx.Utilities.ShakeStrength.Strong); Console.WriteLine("ScreenShakeTrauma ok"); }
                catch (Exception e) { Console.WriteLine(e); }
                break;
            }
            case "profile-start":
            {
                var t = new Dictionary<string, double>();
                var rm = typeof(MegaCrit.Sts2.Core.Runs.RunManager);
                foreach (string name in new[] { "GenerateMap", "SetActInternal", "GenerateRooms", "EnterMapCoordInternal", "EnterRoomInternal" })
                    if (HarmonyLib.AccessTools.Method(rm, name) is { } mm) Profiler.Watch(mm);
                Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.EventModel), "BeginEvent"));
                Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Localization.LocString), "GetFormattedText"));
                Profiler.Watch(HarmonyLib.AccessTools.Method(typeof(MegaCrit.Sts2.Core.Models.ActModel), "CreateMap") ?? HarmonyLib.AccessTools.Method(rm, "GenerateMap"));
                foreach (var ty in new[] { typeof(MegaCrit.Sts2.Core.Map.StandardActMap), typeof(MegaCrit.Sts2.Core.Map.MapPathPruning), typeof(MegaCrit.Sts2.Core.Map.MapPostProcessing) })
                    foreach (var mm in ty.GetMethods(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.DeclaredOnly))
                        if (!mm.IsAbstract && !mm.ContainsGenericParameters && mm.GetMethodBody() != null && !mm.Name.StartsWith("get_") && !mm.Name.StartsWith("<")) { try { Profiler.Watch(mm); } catch { } }
                for (int i = 0; i < 50; i++)
                {
                    var sw2 = Stopwatch.StartNew();
                    if (MegaCrit.Sts2.Core.Runs.RunManager.Instance.IsInProgress) MegaCrit.Sts2.Core.Runs.RunManager.Instance.CleanUp();
                    t["清理"] = t.GetValueOrDefault("清理") + sw2.Elapsed.TotalMilliseconds; sw2.Restart();
                    var task = Pump.Instance.Start(() => HeadlessRun.StartNew("IRONCLAD", 10, $"P{i}", t));
                    if (task.IsFaulted) throw task.Exception!;
                }
                foreach (var kv in t) Console.WriteLine($"{kv.Key,-12} 平均 {kv.Value / 50:F1} ms");
                foreach (var kv in Profiler.Totals.OrderByDescending(kv => kv.Value.ms)) Console.WriteLine($"  {kv.Key,-40} 共 {kv.Value.ms / 50:F1} ms/局，{kv.Value.n / 50.0:F0} 次/局");
                break;
            }
            case "selfcheck":
                // 游戏更新后的自检：selfcheck [每个角色的整局数，默认 2] [--all 连找到的也列出]，有问题时退出码为 1
                Environment.ExitCode = SelfCheck.Run(int.Parse(pos.ElementAtOrDefault(1) ?? "2"), args.Contains("--all"));
                break;
            case "clone-test":
                // 局面复制的进程内自检：clone-test census|roundtrip|audit [combat|run] [局数] [步数]（Engine/Clone/局面复制.md）
                Environment.ExitCode = CloneTest.Run(pos);
                break;
            case "audit-ui":
                foreach (string m in UiLogicAudit.Run(typeof(MegaCrit.Sts2.Core.Runs.RunManager).Assembly)) Console.WriteLine(m);
                break;
            default:
                Console.WriteLine("用法：one [角色] [种子] | runs [每角色局数] | determinism | audit-testmode  [--god] [-v]");
                break;
        }
    }

    record Result(string Seed, bool Won, bool Stuck, int Floor, int Steps, double Ms, string Fingerprint, string Last, Dictionary<DecisionKind, int> Counts, Dictionary<string, int> Tags)
    {
        public override string ToString() => $"{Seed} 胜={Won} 卡住={Stuck} 层数={Floor} 决策={Steps} 耗时={Ms:F0} ms 结束于：{Last}";
    }

    static Result PlayOne(string character, int ascension, string seed, int agentSeed, bool god, bool trace)
    {
        var sw = Stopwatch.StartNew();
        using var run = RunSession.Start(character, ascension, seed);
        var driver = new RandomDriver(agentSeed) { Trace = trace, God = god };
        Decision last;
        try { last = driver.Play(run); }
        catch (Exception e)
        {
            last = new Decision(DecisionKind.Stuck) { Context = "异常：" + e.GetBaseException().Message };
            if (trace) Console.WriteLine(e);
        }
        sw.Stop();
        var s = RunSession.State;
        var p = s.Players[0];
        string summary = $"hp={p.Creature.CurrentHp}/{p.Creature.MaxHp} gold={p.Gold} deck={string.Join(",", p.Deck.Cards.Select(c => c.Id.Entry))} relics={string.Join(",", p.Relics.Select(r => r.Id.Entry))}";
        string fp = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(string.Join("\n", driver.Log) + summary)));
        if (trace) Console.WriteLine(summary);
        return new Result(seed, last.Kind == DecisionKind.GameOver && run.IsVictory, last.Kind == DecisionKind.Stuck, s.TotalFloor, run.Steps, sw.Elapsed.TotalMilliseconds, fp, last.ToString(), driver.Counts, driver.Tags);
    }
}
