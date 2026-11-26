using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Debug;
using System.Text.Json;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2Headless;

/// <summary>
/// sts2.dll 内部需要替换的少数方法。每一项都写明原因；游戏更新后若目标方法消失，Install 会直接报错。
/// </summary>
public static class GameShim
{
    static readonly Harmony H = new("sts2headless.game");

    public static void Install()
    {
        // 命令行参数：静态构造里用 Godot 字典解析，无头下直接视为没有参数
        Prefix(typeof(CommandLineHelper), nameof(CommandLineHelper.HasArg), nameof(ReturnFalse));
        Prefix(typeof(CommandLineHelper), nameof(CommandLineHelper.TryGetValue), nameof(TryGetValueNone));
        Prefix(typeof(CommandLineHelper), nameof(CommandLineHelper.GetValue), nameof(ReturnNull));
        // 崩溃上报：静态构造需要 Godot，全部置空
        NoOp(typeof(SentryService), "AddBreadcrumb", "CaptureException", "CaptureMessage", "SetTag", "SetExtra");
        // 存档：训练不写盘
        NoOp(typeof(SaveManager), "SaveRun", "SaveSettings", "SaveProfile", "SaveRunHistory", "SaveProgressFile", "SavePrefsFile");
        NoOp(typeof(RunManager), "WriteReplay");   // 战斗回放写盘
        // 纯表现层命令（对白、思考气泡、特效、音效）：整类置空，返回默认值
        foreach (Type t in new[] { typeof(TalkCmd), typeof(ThinkCmd), typeof(VfxCmd), typeof(SfxCmd) }) SkipAll(t);
        // 本地化：原实现用 Godot 的 FileAccess/DirAccess 读 res://localization，改为读提取到磁盘的文件
        Prefix(typeof(LocManager), "LoadTable", nameof(LocLoadTable));
        Prefix(typeof(LocManager), "ListLocalizationFiles", nameof(LocListFiles));
        Prefix(typeof(LocManager), "LoadLocCompletionFile", nameof(LocLoadCompletion));
        NoOp(typeof(LocManager), "TriggerLocaleChange");   // 只通知界面刷新文字
        // Kaiser Crab Boss（Crusher / Rocket）的逻辑直接驱动战斗背景节点做动画，拿不到节点就抛异常：给一个假背景对象
        foreach (Type t in new[] { typeof(MegaCrit.Sts2.Core.Models.Monsters.Crusher), typeof(MegaCrit.Sts2.Core.Models.Monsters.Rocket) })
            GameCompat.Patch(H, GameCompat.Getter(t, "Background", "凯撒蟹的假战斗背景"), $"{t.Name}.Background",
                prefix: GameCompat.Own(typeof(GameShim), nameof(FakeCrabBackground)));
        // 逻辑层里有不带 ?. 的音频调用（如 Rocket.BeforeDeath 中的 NAudioManager.Instance.PlayOneShot）：单例给假对象，其方法由 UiCallStubber 置空
        FakeSingleton<MegaCrit.Sts2.Core.Nodes.Audio.NAudioManager>();
        // 地图界面：LordsParasol 在删牌前后不带 ?. 地调用 NMapScreen.Instance.SetTravelEnabled
        FakeSingleton<MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen>();
        // NCombatRoom 保持为 null（与开发者单元测试一致）。逻辑层里少数不带 ?. 的访问，若是方法的最后一句，就吞掉空引用：
        // SoulNexus.AfterDeath：取消订阅后设置死亡动画
        SwallowNullReference(GameCompat.Method(typeof(MegaCrit.Sts2.Core.Models.Monsters.SoulNexus), "AfterDeath", "灵魂枢纽死亡时的界面空引用", new[] { typeof(MegaCrit.Sts2.Core.Entities.Creatures.Creature) }));
        // 游戏根节点：逻辑层有不带 ?. 的 NGame.Instance.ScreenShakeTrauma(...)（如 Amalgamator 事件）
        FakeSingleton<MegaCrit.Sts2.Core.Nodes.NGame>();
        // 调试音频：事件里有不带 ?. 的 NDebugAudioManager.Instance.Play(...)（DenseVegetation、JungleMazeAdventure 等），
        // 它的 Instance 是 NGame.Instance?.DebugAudio，给假 NGame 挂上一个假的调试音频对象
        AccessTools.Field(typeof(MegaCrit.Sts2.Core.Nodes.NGame), "<DebugAudio>k__BackingField")!
            .SetValue(MegaCrit.Sts2.Core.Nodes.NGame.Instance, FakeUi.Create<MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager>());
        UiCallStubber.Exclude.Add(AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Audio.Debug.NDebugAudioManager), "Instance"));
        // 界面上下文：事件中开战时调用 ActiveScreenContext.Instance.Update()；它是普通类、按需创建，保留真实实例，Update 由 UiCallStubber 置空
        UiCallStubber.Exclude.Add(AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext.ActiveScreenContext), "Instance"));
        // 记录游戏自己判定的胜负：胜利后剧情会“击杀”玩家，不能用玩家是否存活判断
        GameCompat.Patch(H, GameCompat.Method(typeof(RunManager), nameof(RunManager.OnEnded), "记录游戏判定的胜负"), "RunManager.OnEnded",
            prefix: GameCompat.Own(typeof(GameShim), nameof(RecordOutcome)));
        // 测试分支调用选牌器时固定传 min=0（可跳过），正式界面只有 canSkip 时才能跳过：把 canSkip 传给选牌器
        GameCompat.Patch(H, GameCompat.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseACardScreen), "三选一能否跳过"), "CardSelectCmd.FromChooseACardScreen",
            prefix: GameCompat.Own(typeof(GameShim), nameof(ChooseACardMinSelect)));
        // 事件界面：Trial.Accept 里有不带 ?. 的 NEventRoom.Instance.Layout.RemoveNodesOnPortrait()
        FakeSingleton<MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom>();
        UiCallStubber.FakeReturn.Add(AccessTools.PropertyGetter(typeof(MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom), "Layout"));
        // Trial 在肖像上加特效：从资源实例化场景，纯表现
        GameCompat.Patch(H, GameCompat.Method(typeof(MegaCrit.Sts2.Core.Models.Events.Trial), "AddVfxAnchoredToPortrait", "审判事件的肖像特效"), "Trial.AddVfxAnchoredToPortrait",
            prefix: GameCompat.Own(typeof(GameShim), nameof(SkipOriginal)));
        if (System.Environment.GetEnvironmentVariable("STS2_NO_PERF") == null) { Step("性能补丁", () => PerfPatches.Install(H)); Step("地图生成补丁", () => MapGenPatch.Install(H)); }
        Step("保真补丁", () => FidelityPatches.Install(H));
        Step("无头修补", () => HeadlessFixes.Install(H));
        Step("开幕快照", () => ActStart.Install(H));
        Step("假界面", () => FakeScreens.Install(H));
        Step("事件选项记录", () => EventOptionTracker.Install(H));
        Step("已知抽牌堆顶", () => Observation.DrawTop.Install(H));      // 已知的抽牌堆顶部（观测格式 20）：只记牌堆操作，不改游戏状态
        Step("有目标伤害预览口径", () => Observation.RawPreview.Install(H));      // v0.114.0：编码的有目标伤害预览不过格挡与掉血钩子（见 Obs/RawPreview.cs）
        Step("跳过延时", () => DelayPatch.Install(H));
        Step("界面调用置空", () => UiCallStubber.Install(H, typeof(CommandLineHelper).Assembly));
    }

    /// <summary>装一组补丁。正常模式下出错照旧抛出（引擎启动失败）；自检模式（GameCompat.SelfCheck）下记下来继续装后面的，好一次查全。</summary>
    static void Step(string name, Action install)
    {
        if (!GameCompat.SelfCheck) { install(); return; }
        try { install(); }
        catch (Exception e) { GameCompat.Patch(H, null, $"{name}：{e.GetBaseException().GetType().Name}：{e.GetBaseException().Message}"); }
    }

    static void SkipAll(Type type)
    {
        foreach (var m in type.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.DeclaredOnly).Where(m => !m.ContainsGenericParameters))
        {
            if (m.ReturnType == typeof(Task) || (m.ReturnType.IsGenericType && m.ReturnType.GetGenericTypeDefinition() == typeof(Task<>)))
                throw new NotSupportedException($"{type.Name}.{m.Name} 返回 Task，需要单独处理");
            H.Patch(m, prefix: new HarmonyMethod(typeof(GameShim).GetMethod(nameof(SkipOriginal), BindingFlags.NonPublic | BindingFlags.Static)));
        }
    }

    static bool SkipOriginal() => false;

    /// <summary>
    /// 吞掉方法抛出的空引用异常。只用于“界面访问是方法最后一句”的同步方法（逐个读过源码确认），否则会跳过后续逻辑。
    /// </summary>
    static void SwallowNullReference(MethodInfo? m)
    {
        if (m == null) throw new MissingMethodException("SwallowNullReference 的目标方法不存在，游戏可能已更新");
        if (m.ReturnType != typeof(void)) throw new NotSupportedException($"{m.Name} 不是同步 void 方法");
        H.Patch(m, finalizer: new HarmonyMethod(typeof(GameShim).GetMethod(nameof(DropNullReference), BindingFlags.NonPublic | BindingFlags.Static)));
    }

    static Exception? DropNullReference(Exception? __exception) => __exception is NullReferenceException ? null : __exception;

    /// <summary>最近一局的结果：null 表示尚未结束。</summary>
    public static bool? LastRunVictory { get; set; }
    static void RecordOutcome(bool isVictory) => LastRunVictory = isVictory;

    static void ChooseACardMinSelect(bool canSkip)
    {
        AgentCardSelector.NextMinSelect = canSkip ? 0 : 1;
        AgentCardSelector.NextInfo = new SelectInfo(null, null, "FromChooseACardScreen", null);
    }

    /// <summary>
    /// 让界面单例 T.Instance 返回一个假对象。自动属性直接写后备字段：它的 getter 会被 JIT 内联进调用方，打补丁不生效；
    /// 计算属性（如 NCombatRoom.Instance => NRun.Instance?.CombatRoom）不会被内联，给 getter 打补丁。
    /// </summary>
    static void FakeSingleton<T>() where T : class
    {
        var field = GameCompat.OptionalField(typeof(T), "<Instance>k__BackingField", "界面单例的后备字段（没有时改给 getter 打补丁）");
        if (field != null)
        {
            field.SetValue(null, FakeUi.Create<T>());
            UiCallStubber.Exclude.Add(AccessTools.PropertyGetter(typeof(T), "Instance"));
            return;
        }
        GameCompat.Patch(H, GameCompat.Getter(typeof(T), "Instance", "界面单例换成假对象"), $"{typeof(T).Name}.Instance",
            prefix: new HarmonyMethod(typeof(FakeInstance<T>).GetMethod(nameof(FakeInstance<T>.Prefix))));
    }

    static class FakeInstance<T> where T : class
    {
        static readonly T Value = FakeUi.Create<T>();
        public static bool Prefix(ref T __result) { __result = Value; return false; }
    }

    static readonly MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground CrabBackground = FakeUi.Create<MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground>();
    static bool FakeCrabBackground(ref MegaCrit.Sts2.Core.Nodes.Vfx.Backgrounds.NKaiserCrabBossBackground __result) { __result = CrabBackground; return false; }

    /// <summary>把方法改成什么都不做：void 直接返回，Task 返回已完成任务，bool 返回 false。</summary>
    static void NoOp(Type type, params string[] names)
    {
        foreach (string name in names)
        {
            var methods = type.GetMethods(BindingFlags.Static | BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                .Where(m => m.Name == name).ToList();
            if (methods.Count == 0) { GameCompat.Patch(H, null, $"{type.FullName}.{name}（置空）"); continue; }
            foreach (var m in methods)
            {
                string replacement = m.ReturnType == typeof(void) ? nameof(Skip)
                    : m.ReturnType == typeof(Task) ? nameof(CompletedTask)
                    : m.ReturnType == typeof(bool) ? nameof(ReturnFalse)
                    : throw new NotSupportedException($"{type.Name}.{name} 返回 {m.ReturnType}");
                GameCompat.Patch(H, m, $"{type.Name}.{name}（置空）", prefix: GameCompat.Own(typeof(GameShim), replacement));
            }
        }
    }

    static void Prefix(Type type, string name, string replacement)
    {
        var methods = type.GetMethods(BindingFlags.Static | BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
            .Where(m => m.Name == name).ToList();
        if (methods.Count == 0) { GameCompat.Patch(H, null, $"{type.FullName}.{name}（替换为 {replacement}）"); return; }
        var prefix = GameCompat.Own(typeof(GameShim), replacement);
        foreach (var m in methods) GameCompat.Patch(H, m, $"{type.Name}.{name}（替换为 {replacement}）", prefix: prefix);
    }

    static bool LocLoadTable(string path, ref Dictionary<string, string> __result)
    {
        __result = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(VirtualFs.Map(path)))!;
        return false;
    }

    static bool LocListFiles(string path, ref IEnumerable<string> __result)
    {
        __result = Directory.GetFiles(VirtualFs.Map(path), "*.json").Select(Path.GetFileName).ToArray()!;
        return false;
    }

    static bool LocLoadCompletion(LocManager __instance)
    {
        var counts = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(VirtualFs.Map("res://localization/completion.json")));
        AccessTools.Field(typeof(LocManager), "_languageKeyCount").SetValue(__instance, counts);
        return false;
    }

    static bool Skip() => false;
    static bool CompletedTask(ref Task __result) { __result = Task.CompletedTask; return false; }
    static bool ReturnFalse(ref bool __result) { __result = false; return false; }
    static bool ReturnNull(ref object? __result) { __result = null; return false; }
    static bool TryGetValueNone(ref string? value, ref bool __result) { value = null; __result = false; return false; }
}
