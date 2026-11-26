using System.Diagnostics;
using System.Reflection;
using Godot;
using HarmonyLib;

namespace Sts2Headless;

/// <summary>
/// 没有 Godot 引擎时，GodotSharp 的原生函数指针都是空的，调用即段错误且不留信息。
/// 这里做两件事：
/// 1. 护栏：给 NativeFuncs 的每个方法加前缀，改为抛出带名字的托管异常，便于定位缺哪个替身；
/// 2. 替身：游戏逻辑层用到的少量 Godot 静态接口（日志、命令行、时间等）直接返回无界面下的合理值。
/// 补丁只打在 GodotSharp 的公开接口上，不碰 sts2.dll，游戏更新后通常不用改。
/// </summary>
public static class GodotShim
{
    public static readonly Harmony H = new("sts2headless.godot");
    static readonly Stopwatch Clock = Stopwatch.StartNew();
    public static bool Verbose;

    public static void Install()
    {
        InstallNativeGuard();
        Replace(typeof(GD), nameof(GD.Print), nameof(Swallow));
        Replace(typeof(GD), nameof(GD.PrintErr), nameof(PrintErrToConsole));
        Replace(typeof(GD), nameof(GD.PrintRich), nameof(Swallow));
        Replace(typeof(GD), nameof(GD.PushError), nameof(PrintErrToConsole));
        Replace(typeof(GD), nameof(GD.PushWarning), nameof(Swallow));
        Replace(typeof(OS), nameof(OS.GetCmdlineArgs), nameof(EmptyStrings));
        Replace(typeof(OS), nameof(OS.GetCmdlineUserArgs), nameof(EmptyStrings));
        Replace(typeof(OS), nameof(OS.HasFeature), nameof(False));
        Replace(typeof(OS), nameof(OS.IsDebugBuild), nameof(False));
        Replace(typeof(OS), nameof(OS.GetName), nameof(OsName));
        Replace(typeof(OS), nameof(OS.GetEnvironment), nameof(GetEnv));
        Replace(typeof(OS), nameof(OS.GetExecutablePath), nameof(ExePath));
        Replace(typeof(Engine), nameof(Engine.IsEditorHint), nameof(False));
        Replace(typeof(Engine), nameof(Engine.HasSingleton), nameof(False));
        Replace(typeof(Time), nameof(Time.GetTicksMsec), nameof(TicksMsec));
        Replace(typeof(Time), nameof(Time.GetTicksUsec), nameof(TicksUsec));
        H.Patch(AccessTools.Method(typeof(Callable), nameof(Callable.CallDeferred)),
            prefix: new HarmonyMethod(typeof(GodotShim).GetMethod(nameof(CallDeferred), BindingFlags.NonPublic | BindingFlags.Static)));
        // StringName / NodePath：界面类的静态构造里大量创建（方法名、输入动作名），无头下只做空占位
        var skipCtor = new HarmonyMethod(typeof(GodotShim).GetMethod(nameof(SkipCtor), BindingFlags.NonPublic | BindingFlags.Static));
        H.Patch(AccessTools.Constructor(typeof(StringName), new[] { typeof(string) }), prefix: skipCtor);
        H.Patch(AccessTools.Constructor(typeof(NodePath), new[] { typeof(string) }), prefix: skipCtor);
        // 帧耗时：逻辑层有约 20 处用 ((SceneTree)Engine.GetMainLoop()).Root.GetProcessDeltaTime() 决定是否插入动画等待，
        // 给一个假的主循环和根窗口，帧耗时返回很大的值，等待条件永远不成立
        Replace(typeof(Engine), nameof(Engine.GetMainLoop), nameof(FakeMainLoop));
        H.Patch(AccessTools.PropertyGetter(typeof(SceneTree), nameof(SceneTree.Root)),
            prefix: new HarmonyMethod(typeof(GodotShim).GetMethod(nameof(FakeRoot), BindingFlags.NonPublic | BindingFlags.Static)));
        H.Patch(AccessTools.Method(typeof(Node), nameof(Node.GetProcessDeltaTime)),
            prefix: new HarmonyMethod(typeof(GodotShim).GetMethod(nameof(HugeDelta), BindingFlags.NonPublic | BindingFlags.Static)));
        // 假界面对象上偶尔会调用的 Godot 节点方法
        H.Patch(AccessTools.Method(typeof(CanvasItem), nameof(CanvasItem.SetVisible)), prefix: skipCtor);
        // 资源：无头下不加载任何贴图、场景、音频
        Replace(typeof(ResourceLoader), nameof(ResourceLoader.Exists), nameof(False));
        foreach (MethodInfo m in typeof(ResourceLoader).GetMethods(BindingFlags.Static | BindingFlags.Public).Where(m => m.Name == nameof(ResourceLoader.Load)))
            H.Patch(m.IsGenericMethodDefinition ? m.MakeGenericMethod(typeof(Resource)) : m,
                prefix: new HarmonyMethod(typeof(GodotShim).GetMethod(nameof(ReturnNull), BindingFlags.NonPublic | BindingFlags.Static)));
        Replace(typeof(ProjectSettings), nameof(ProjectSettings.GlobalizePath), nameof(GlobalizePath));
        Replace(typeof(DirAccess), nameof(DirAccess.DirExistsAbsolute), nameof(DirExists));
        Replace(typeof(DirAccess), nameof(DirAccess.MakeDirAbsolute), nameof(MakeDir));
        Replace(typeof(DirAccess), nameof(DirAccess.MakeDirRecursiveAbsolute), nameof(MakeDir));
        Replace(typeof(Godot.FileAccess), nameof(Godot.FileAccess.FileExists), nameof(FileExists));
    }

    /// <summary>
    /// 用一张“陷阱表”初始化 NativeFuncs：每个原生函数指针都指向 Trap。
    /// 这些方法大多是 AggressiveInlining，Harmony 拦不住，只能在指针这一层兜底。
    /// </summary>
    static unsafe void InstallNativeGuard()
    {
        Type nf = typeof(GodotObject).Assembly.GetType("Godot.NativeInterop.NativeFuncs")!;
        Type cb = nf.GetNestedType("UnmanagedCallbacks", BindingFlags.NonPublic)!;
        int size = (int)typeof(System.Runtime.CompilerServices.Unsafe).GetMethod("SizeOf")!.MakeGenericMethod(cb).Invoke(null, null)!;
        nint table = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
        delegate* unmanaged<void> trap = &Trap;
        for (int i = 0; i < size / sizeof(nint); i++) ((nint*)table)[i] = (nint)trap;
        nf.GetMethod("Initialize")!.Invoke(null, new object[] { table, size });
        if (Verbose) Console.Error.WriteLine($"[shim] 原生陷阱表：{size / sizeof(nint)} 项");
    }

    [System.Runtime.InteropServices.UnmanagedCallersOnly]
    static void Trap()
    {
        Console.Error.WriteLine("无头模式下调用了 Godot 原生接口，调用栈：");
        Console.Error.WriteLine(new StackTrace(1, true).ToString());
        Console.Error.Flush();
        System.Environment.Exit(97);
    }

    /// <summary>把 type 上所有名为 name 的方法（含重载）替换为 GodotShim 上的 replacement 前缀。</summary>
    static void Replace(Type type, string name, string replacement)
    {
        var prefix = new HarmonyMethod(typeof(GodotShim).GetMethod(replacement, BindingFlags.NonPublic | BindingFlags.Static));
        foreach (MethodInfo m in type.GetMethods(BindingFlags.Static | BindingFlags.Public).Where(m => m.Name == name))
            H.Patch(m, prefix: prefix);
    }

    static bool Swallow() => false;
    /// <summary>游戏日志里的错误，按“异常类型 + 第一个出错位置”归类计数。无头运行中的每一条错误都可能意味着逻辑被中断。</summary>
    public static readonly Dictionary<string, int> Errors = new();
    /// <summary>累计错误条数与最近一条（RunSession 据此把出错的对局按卡住结束：TaskHelper.RunSafely 等只记日志、不抛出）。</summary>
    public static int ErrorTotal;
    public static string LastError = "";
    /// <summary>最近一条错误的全文（含调用栈），卡住时写进引擎日志留现场（LastError 只是归类用的前 120 字）。</summary>
    public static string LastErrorFull = "";
    /// <summary>游戏自己只记日志、流程照常往下走的错误：仍按类计数留档（Errors、BenignTotal），但不计入 ErrorTotal，不判卡住。
    /// “Tried to pop model … from stack of player choice context”：出牌包装（CardModel.OnPlayWrapper）在玩家死亡等情况下提前返回，
    /// 没把牌出栈，外层钩子出栈时对不上；PlayerChoiceContext.PopModel 只记一条错误。低语耳环（瓦库代打第一回合）、
    /// 乱战（回合开始自动打出牌堆顶）代打时玩家被打死就会出现，原来整局被判卡住（2026-10-27 查明，训练里约占卡住的九成），
    /// 真游戏里接着进入死亡结算。</summary>
    static readonly string[] BenignPrefixes = { "[ERROR] Tried to pop model " };
    public static int BenignTotal;

    static bool PrintErrToConsole(object[] __args)
    {
        string text = string.Join(" ", FlattenArgs(__args));
        if (Verbose) Console.Error.WriteLine("[godot] " + text);
        var lines = text.Split('\n');
        string key = lines[0].Length > 120 ? lines[0][..120] : lines[0];
        string? at = lines.Skip(1).Select(l => l.Trim()).FirstOrDefault(l => l.StartsWith("at "));
        if (at != null) key += " " + at;
        Errors[key] = Errors.GetValueOrDefault(key) + 1;
        if (BenignPrefixes.Any(p => text.TrimStart().StartsWith(p))) { BenignTotal++; return false; }
        ErrorTotal++; LastError = key; LastErrorFull = text;
        return false;
    }
    static IEnumerable<object?> FlattenArgs(object[] args) =>
        args.SelectMany(a => a is object[] arr ? arr : new[] { a });
    static bool EmptyStrings(ref string[] __result) { __result = Array.Empty<string>(); return false; }
    static bool False(ref bool __result) { __result = false; return false; }
    static bool OsName(ref string __result) { __result = "Windows"; return false; }
    static bool GetEnv(string variable, ref string __result) { __result = System.Environment.GetEnvironmentVariable(variable) ?? ""; return false; }
    static bool ExePath(ref string __result) { __result = Path.Combine(GameAssemblies.Dir, "SlayTheSpire2.exe"); return false; }
    /// <summary>“下一帧调用”改为排到消息泵队尾。</summary>
    static bool CallDeferred(ref Callable __instance, Variant[] args)
    {
        Delegate d = __instance.Delegate ?? throw new NotSupportedException("CallDeferred 目标不是托管委托");
        if (args.Length != 0) throw new NotSupportedException("CallDeferred 带参数");
        Pump.Instance.Post(_ => d.DynamicInvoke(), null);
        return false;
    }

    static bool SkipCtor() => false;
    static readonly SceneTree MainLoop = FakeUi.Create<SceneTree>();
    static readonly Window RootWindow = FakeUi.Create<Window>();
    static bool FakeMainLoop(ref MainLoop __result) { __result = MainLoop; return false; }
    static bool FakeRoot(ref Window __result) { __result = RootWindow; return false; }
    static bool HugeDelta(ref double __result) { __result = 1e9; return false; }
    static bool ReturnNull(ref object? __result) { __result = null; return false; }
    static bool GlobalizePath(string path, ref string __result) { __result = VirtualFs.Map(path); return false; }
    static bool DirExists(string path, ref bool __result) { __result = Directory.Exists(VirtualFs.Map(path)); return false; }
    static bool MakeDir(string path, ref Error __result) { Directory.CreateDirectory(VirtualFs.Map(path)); __result = Error.Ok; return false; }
    static bool FileExists(string path, ref bool __result) { __result = File.Exists(VirtualFs.Map(path)); return false; }
    static bool TicksMsec(ref ulong __result) { __result = (ulong)Clock.ElapsedMilliseconds; return false; }
    static bool TicksUsec(ref ulong __result) { __result = (ulong)(Clock.ElapsedTicks * 1_000_000 / Stopwatch.Frequency); return false; }
}
