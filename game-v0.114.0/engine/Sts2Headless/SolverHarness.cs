using System.Diagnostics;
using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;

namespace Sts2Headless;

/// <summary>
/// 试解：用创意工坊的 Combat Solver 模组（MIT 许可）求解一场战斗，记录耗时、内存和结果。只用于评估，不参与训练。
///   solve-test 配置.jsonl 行号 [--timeout-s 秒]
/// 配置文件每行 {"label": ..., "spec": CombatSpec}（与管道模式的 combat reset 相同的字段）。
/// 模组不随仓库分发：COMBAT_SOLVER_DIR 指向 CombatSolver.dll 所在目录，RITSULIB_DIR 指向它依赖的 RitsuLib 模组目录。
///
/// 模组的入口（Entry.Initialize）会读写游戏用户目录、打 Harmony 补丁、订阅 RitsuLib 的战斗事件、挂界面，这里都不做，
/// 只补上求解需要的三样：日志（Entry.Logger）、默认设置（SolverSettings 未加载时即为默认值）、显示服务器名（按 headless 处理）。
/// 然后照 SolverController 发起搜索的顺序构造根局面、显示名、伤害记录和搜索策略，调用 CombatSearchCoordinator.Solve。
/// 求解器知道本局的全部随机数（抽牌顺序、怪物行动），结果是“完全信息下”的打法，只能当上界或诊断用。
/// 每场单独开一个进程运行，峰值内存才不受上一场影响。
/// </summary>
public static class SolverHarness
{
    static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    public static void Run(string specPath, int index, double timeoutS)
    {
        using var doc = JsonDocument.Parse(File.ReadLines(specPath).Skip(index).First());
        var label = doc.RootElement.TryGetProperty("label", out var l) ? l.Clone() : default;
        string? id = doc.RootElement.TryGetProperty("id", out var idEl) ? idEl.GetString() : null;
        var spec = doc.RootElement.GetProperty("spec").Deserialize<CombatSpec>(JsonOptions)!;
        var asm = LoadSolver();
        Type T(string name) => asm.GetType("CombatSolver." + name, throwOnError: true)!;
        const BindingFlags Any = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance;
        object? Call(Type t, string method, params object?[] args) =>
            t.GetMethods(Any).Single(m => m.Name == method && m.GetParameters().Length == args.Length).Invoke(null, args);

        // 求解需要的运行环境
        string logDir = SolverLogDir();
        PruneOldLogs(logDir);
        var logger = Activator.CreateInstance(T("CombatSolverLog"), Any, null, new object[] { logDir }, null);
        T("Entry").GetProperty("Logger", Any)!.SetValue(null, logger);
        T("SolverController").GetProperty("DisplayServerNameProvider", Any)!.SetValue(null, (Func<string>)(() => "headless"));
        // 每场的搜索并行度（默认按逻辑核数：16 核及以上为 8）；同时跑几场时用 SOLVER_DOP 调小，避免抢核
        if (int.TryParse(Environment.GetEnvironmentVariable("SOLVER_DOP"), out int dop))
        {
            // UnattendedTestRunner 的静态属性只读，转发到私有的 Host 对象（ProtocolHost）上，设在 Host 上
            var host = T("UnattendedTestRunner").GetField("Host", Any)!.GetValue(null)!;
            host.GetType().GetProperty("SearchMaxDegreeOfParallelismOverride", Any)!.SetValue(host, (int?)dop);
        }

        using var run = RunSession.StartCombat(spec);
        var decision = run.GetDecision();
        // 开打时的选择（如开局选牌的遗物）求解器不接：选第一项推进到出牌阶段，记下选了什么
        var pre = new List<string>();
        for (int k = 0; decision.Kind is not (DecisionKind.Combat or DecisionKind.GameOver) && decision.Choices.Count > 0 && k < 30; k++)
        {
            pre.Add($"{decision} → {decision.Choices[0].Label}");
            run.Choose(decision, 0);
            decision = run.GetDecision();
        }
        var state = CombatManager.Instance.DebugOnlyGetState() ?? throw new InvalidOperationException("开局后不在战斗中：" + decision.Kind);
        var me = state.Players[0];

        // 控制器的准备步骤要求在“主线程”上调用（NGame.IsMainThread：第一次调用它的线程记为主线程）。
        // NGame 属于界面层，引擎启动时把它的静态调用置空了（UiCallStubber），IsMainThread 永远返回 false；
        // 这里撤掉这一个方法的置空补丁，并把当前线程记为主线程。之后不再推进游戏，不影响游戏逻辑
        var isMain = typeof(MegaCrit.Sts2.Core.Nodes.NGame).GetMethod("IsMainThread", Any)!;
        var harmony = new HarmonyLib.Harmony("sts2headless.solver-harness");
        harmony.Unpatch(isMain, HarmonyLib.HarmonyPatchType.All, "*");
        typeof(MegaCrit.Sts2.Core.Nodes.NGame).GetField("_mainThreadId", Any)!.SetValue(null, Environment.CurrentManagedThreadId);
        // 显示名（只用于界面和日志）要给模型库里所有能力取名，测试专用的能力（如 MOCK_GATED_TURN_START_POWER）在本地化表里没有条目：
        // 查不到时返回键名，不抛异常
        harmony.Patch(typeof(MegaCrit.Sts2.Core.Localization.LocTable).GetMethod("GetRawText", Any, new[] { typeof(string) })!,
            finalizer: new HarmonyLib.HarmonyMethod(typeof(SolverHarness).GetMethod(nameof(MissingLocAsKey), Any)));

        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        long rssBefore = proc.WorkingSet64, allocBefore = GC.GetTotalAllocatedBytes(precise: true);
        var sw = Stopwatch.StartNew();
        var settings = Call(T("SolverSettings"), "Capture");
        var policy = Call(T("SolverController"), "CaptureSearchPolicy", settings, state, false, null, null);
        // 软预算 = 限时（求解器到点自己收尾，交出目前最好的路线：已找到赢法的就带着压到的最低掉血）；硬取消延后到 1.5 倍再加 30 秒，
        // 只防卡死。原来两者都是 120 秒，到点被硬取消，抛 OperationCanceledException，拿不到路线
        policy!.GetType().GetField("<BudgetOverrideMilliseconds>k__BackingField", Any)!.SetValue(policy, (int?)(int)(timeoutS * 1000));
        var root = Call(T("CombatRootSnapshot"), "Capture", state, false);
        var names = Call(T("SolverDisplayNames"), "Capture", state);
        var damage = Call(T("BattleDamageTracker"), "Observe", state);
        double setupMs = sw.Elapsed.TotalMilliseconds;
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutS * 1.5 + 30));
        object? result = null; string? error = null;
        try { result = Call(T("CombatSearchCoordinator"), "Solve", root, names, damage, policy, cts.Token, null); }
        catch (TargetInvocationException e) { error = (e.InnerException ?? e).GetType().Name + ": " + (e.InnerException ?? e).Message; }
        double solveMs = sw.Elapsed.TotalMilliseconds - setupMs;
        proc.Refresh();

        var outcome = new Dictionary<string, object?>
        {
            ["id"] = id,
            ["label"] = label.ValueKind == JsonValueKind.Undefined ? null : label,
            ["encounter"] = state.Encounter?.Id.Entry, ["act"] = spec.Act,
            ["start_hp"] = me.Creature.CurrentHp, ["max_hp"] = me.Creature.MaxHp, ["deck"] = me.Deck.Cards.Count,
            ["setup_ms"] = Math.Round(setupMs, 1), ["solve_ms"] = Math.Round(solveMs, 1), ["timed_out"] = cts.IsCancellationRequested, ["budget_ms"] = (int)(timeoutS * 1000),
            ["rss_before_mb"] = rssBefore >> 20, ["rss_after_mb"] = proc.WorkingSet64 >> 20, ["rss_peak_mb"] = proc.PeakWorkingSet64 >> 20,
            ["allocated_mb"] = (GC.GetTotalAllocatedBytes(precise: true) - allocBefore) >> 20,
            ["error"] = error,
            ["policy"] = policy?.ToString(),
            ["pre_choices"] = pre,
            ["start_decision"] = decision.ToString(),
        };
        if (result != null)
        {
            object? Get(object o, string p) => o.GetType().GetProperty(p, Any)!.GetValue(o);
            var snap = Get(result, "Snapshot")!;
            bool playerDead = (bool)Get(snap, "PlayerDead")!, enemiesDead = (bool)Get(snap, "AllEnemiesDead")!;
            outcome["win"] = enemiesDead && !playerDead;
            outcome["player_dead"] = playerDead;
            outcome["end_hp"] = Get(snap, "PlayerHp");
            outcome["end_turn"] = Get(snap, "Turn");
            outcome["enemy_hp_left"] = Get(snap, "EnemyHp");
            outcome["boundary"] = Get(snap, "BoundaryReason")?.ToString();
            outcome["end_max_hp"] = Get(snap, "PlayerMaxHp");
            // 最好路线上用掉的药水（连打第二个 Boss 时从快照里去掉）：结果的 PotionUses，每项取 PotionId（没有这个属性时记整项的文本）
            var used = new List<string>();
            if (Get(result, "PotionUses") is System.Collections.IEnumerable uses)
                foreach (var u in uses)
                    used.Add(u.GetType().GetProperty("PotionId", Any)?.GetValue(u) as string ?? u.ToString()!);
            outcome["potions_used"] = used;
            outcome["expanded_nodes"] = Get(result, "ExpandedNodes");
            outcome["total_expanded_nodes"] = Get(result, "TotalExpandedNodes");
        }
        Console.WriteLine(JsonSerializer.Serialize(outcome));
    }

    /// <summary>
    /// 求解器日志目录：$XDG_CACHE_HOME（未设置时为 ~/.cache）下的 sls2/combat_solver_logs。
    /// 不放 /tmp：WSL 的 /tmp 是内存盘，每场 1–3MB，批量标注几天就能占满
    /// </summary>
    static string SolverLogDir()
    {
        string? cache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (string.IsNullOrEmpty(cache)) cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        return Path.Combine(cache, "sls2", "combat_solver_logs");
    }

    /// <summary>
    /// 删掉一天前的日志：求解器每个进程建一个子目录。按修改时间删，不按个数删，
    /// 同时跑的几个进程才不会删到彼此正在写的目录（每场限时远短于一天）
    /// </summary>
    static void PruneOldLogs(string logDir)
    {
        if (!Directory.Exists(logDir)) return;
        var cutoff = DateTime.UtcNow - TimeSpan.FromDays(1);
        foreach (string d in Directory.EnumerateDirectories(logDir))
        {
            if (Directory.GetLastWriteTimeUtc(d) >= cutoff) continue;
            // 别的进程可能同时在删同一个目录
            try { Directory.Delete(d, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    static Exception? MissingLocAsKey(Exception? __exception, string key, ref string __result)
    {
        if (__exception is MegaCrit.Sts2.Core.Localization.LocException) { __result = key; return null; }
        return __exception;
    }

    /// <summary>从模组目录加载 Combat Solver 及其依赖（RitsuLib 按游戏版本分的兼容目录优先）。</summary>
    static Assembly LoadSolver()
    {
        string solverDir = Environment.GetEnvironmentVariable("COMBAT_SOLVER_DIR") ?? throw new InvalidOperationException("未设置 COMBAT_SOLVER_DIR");
        string? ritsu = Environment.GetEnvironmentVariable("RITSULIB_DIR");
        var dirs = new List<string> { solverDir };
        if (ritsu != null)
            dirs.AddRange(new[] { Path.Combine(ritsu, "compat", "0.111.0"), ritsu, Path.Combine(ritsu, "shared") });
        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            foreach (string d in dirs)
            {
                string p = Path.Combine(d, name.Name + ".dll");
                if (File.Exists(p)) return ctx.LoadFromAssemblyPath(p);
            }
            return null;
        };
        return AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(solverDir, "CombatSolver.dll"));
    }
}
