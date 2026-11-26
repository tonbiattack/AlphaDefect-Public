using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

namespace Sts2Headless;

/// <summary>
/// 局面复制的“接触面”：哪些静态字段是根、哪些类型当叶子（不往下走、不记）。游戏更新时复核这里（局面复制.md 第五节）。
///
/// 原则：宁可多记（多记只是慢一点，恢复写回的是同样的值），不可漏记（漏记的可变状态在恢复后会带着死去那条时间线的值）。
/// 当叶子的只有两类：
///   1. 不可变或与局面无关的运行时对象：字符串、反射对象、流与编码器、线程与执行上下文、Harmony 与 System.Text.Json 的内部对象；
///   2. 游戏的模板：规范模型（AbstractModel 且 IsMutable 为假，玩的时候游戏自己断言不许改）、本地化表。
/// 第 2 类是为了快，是否真的没被改由 clone-test 的“漏记审计”核对（拍快照后走若干步，查叶子背后的对象有没有被改）。
/// </summary>
public static class ClonePolicy
{
    /// <summary>根所在的程序集：游戏本体与引擎本身。BCL、Harmony、GodotSharp、SmartFormat 等的静态字段不进根（它们不存局面）。</summary>
    public static IEnumerable<Assembly> RootAssemblies => new[] { typeof(MegaCrit.Sts2.Core.Runs.RunManager).Assembly, typeof(ClonePolicy).Assembly };

    static readonly Assembly[] LeafAssemblies =
    {
        typeof(HarmonyLib.Harmony).Assembly,
        typeof(System.Text.Json.JsonSerializer).Assembly,
        typeof(System.Text.RegularExpressions.Regex).Assembly,
    };

    /// <summary>类型整个当叶子的原因；不是叶子时返回 null。</summary>
    public static string? LeafType(Type t)
    {
        if (t == typeof(string)) return "字符串";
        if (t == typeof(MegaCrit.Sts2.Core.Models.ModelId)) return "模型 ID（不可变的记录）";
        if (typeof(Attribute).IsAssignableFrom(t)) return "特性（元数据）";
        if ((t.IsArray && typeof(MemberInfo).IsAssignableFrom(t.GetElementType()))
            || (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(List<>) && typeof(MemberInfo).IsAssignableFrom(t.GetGenericArguments()[0])))
            return "反射对象的数组与列表（反射缓存）";
        if (t == typeof(GameCompat.Need)) return "兼容层登记";
        if (typeof(MemberInfo).IsAssignableFrom(t) || typeof(Assembly).IsAssignableFrom(t) || typeof(Module).IsAssignableFrom(t)
            || typeof(ParameterInfo).IsAssignableFrom(t) || t == typeof(Pointer)) return "反射对象";
        if (typeof(Stream).IsAssignableFrom(t) || typeof(TextWriter).IsAssignableFrom(t) || typeof(TextReader).IsAssignableFrom(t)
            || typeof(System.Text.Encoding).IsAssignableFrom(t) || typeof(System.Text.Encoder).IsAssignableFrom(t) || typeof(System.Text.Decoder).IsAssignableFrom(t))
            return "流与编码器";
        if (typeof(Thread).IsAssignableFrom(t) || typeof(ExecutionContext).IsAssignableFrom(t) || typeof(TaskScheduler).IsAssignableFrom(t)
            || typeof(System.Runtime.InteropServices.SafeHandle).IsAssignableFrom(t) || typeof(WaitHandle).IsAssignableFrom(t))
            return "线程、调度与句柄";
        if (typeof(HeapSnapshot).IsAssignableFrom(t) || t == typeof(HeapWalker) || t == typeof(TypePlan) || t == typeof(CloneSlots)) return "快照自身";
        if (LeafAssemblies.Contains(t.Assembly) || typeof(System.Text.RegularExpressions.Regex).IsAssignableFrom(t)) return "Harmony、JSON、正则的内部对象（含游戏里源生成的正则）";
        if (t == typeof(WeakReference) || (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(WeakReference<>)))
            return "弱引用（句柄由运行时管，游戏逻辑不用；正则的替换缓存在用）";
        // ConditionalWeakTable 不当叶子：引擎的“已知牌堆顶”（DrawTop）、PerfPatches 的生物列表缓存都存在里面，由 TypePlans 按键值对存取。
        // 反向核对用：环境变量 STS2_CLONE_WEAK_LEAF=1 时退回当叶子（最初的写法），clone-test 应当查出已知牌堆顶的不同
        if (WeakLeaf && t.IsGenericType && t.GetGenericTypeDefinition() == typeof(System.Runtime.CompilerServices.ConditionalWeakTable<,>)) return "弱引用表（反向核对：当叶子）";
        if (t.Namespace is "Godot" && (t == typeof(Godot.StringName) || t == typeof(Godot.NodePath))) return "Godot 名字（无头下是空占位）";
        if (!AuditAll && t == typeof(MegaCrit.Sts2.Core.Localization.LocTable)) return "本地化表";
        return null;
    }

    static readonly bool WeakLeaf = Environment.GetEnvironmentVariable("STS2_CLONE_WEAK_LEAF") == "1";

    /// <summary>漏记审计（clone-test audit）：第 2 类叶子（规范模型、本地化表）也往下走、也记。</summary>
    public static bool AuditAll;

    /// <summary>按实例判断的叶子：规范模型（游戏的模板）。</summary>
    public static Func<object, bool>? LeafInstance(Type t) =>
        !AuditAll && typeof(MegaCrit.Sts2.Core.Models.AbstractModel).IsAssignableFrom(t) ? static o => ((MegaCrit.Sts2.Core.Models.AbstractModel)o).IsCanonical : null;

    /// <summary>
    /// 没初始化的类型能不能由我们去跑它的静态初始化：Godot 对象与界面层（MegaCrit.Sts2.Core.Nodes 下）不能——
    /// 它们的静态初始化会调 Godot 原生接口，无头下进程直接退出（GodotShim.Trap）。
    /// 新版游戏里别的类型也这样时，clone-test 第一次拍快照就会退出（退出码 97，日志里有调用栈），照它补在这里。
    /// </summary>
    public static bool ForceInit = false;    // 原型：先不主动初始化（RichTextUtil 等非界面类的静态初始化也会调 Godot 原生接口），只取已初始化的，晚初始化的记进 LateInit
    public static bool SafeToInitialize(Type t) => ForceInit &&
        !typeof(Godot.GodotObject).IsAssignableFrom(t) && !(t.Namespace ?? "").StartsWith("MegaCrit.Sts2.Core.Nodes") && !(t.Namespace ?? "").StartsWith("Godot");

    /// <summary>不进根的静态字段（快照自身的存放处）。</summary>
    public static bool SkipStatic(FieldInfo f) =>
        f.DeclaringType == typeof(CloneSlots) || f.DeclaringType == typeof(HeapWalker) || f.DeclaringType == typeof(TypePlans) || f.DeclaringType == typeof(HeapSnapshot)
        || f.DeclaringType == typeof(CloneRoots) || f.FieldType.IsPointer || f.FieldType.IsByRefLike
        || f.IsDefined(typeof(ThreadStaticAttribute)) || f.DeclaringType!.Name.StartsWith("<PrivateImplementationDetails>");
}

/// <summary>静态根：游戏与引擎程序集里全部类型的静态字段（第一次拍快照时找一遍，并把这些类型的静态初始化都跑完）。</summary>
public static class CloneRoots
{
    public sealed class StaticRoot
    {
        public required FieldInfo Field { get; init; }
        public required Func<object?> Get { get; init; }
        public Action<object?>? Set { get; init; }
    }

    static StaticRoot[]? _writable, _readonly;
    /// <summary>可写的静态字段：记下值、恢复时写回。</summary>
    public static StaticRoot[] Writable => _writable!;
    /// <summary>只读的静态字段：不能写回（JIT 可能已把值当常量），只从它们出发往下走。</summary>
    public static StaticRoot[] ReadOnly => _readonly!;
    /// <summary>静态初始化跑不起来的类型（自检报告用）。</summary>
    public static readonly List<string> InitFailures = new();
    /// <summary>有静态字段的泛型类型定义（封闭类型在运行时才有，拿不到；自检报告用）。</summary>
    public static readonly List<string> GenericDefinitions = new();
    public static double DiscoverMs { get; private set; }

    /// <summary>拍快照时还没初始化的类型（不主动去初始化：有的类型静态初始化会调 Godot 原生接口，无头下进程直接退出）。</summary>
    static readonly List<Type> _pending = new();
    /// <summary>拍快照之后才初始化的类型（这些类型的静态字段不在那份快照里），自检与日志用。</summary>
    public static readonly List<string> LateInit = new();

    /// <summary>
    /// 找静态根：游戏与引擎程序集里有静态字段的类型。
    /// 已经初始化的直接取。没初始化的非界面类型先把静态初始化跑完：不然它在拍快照之后才第一次被用到，静态字段不在快照里，
    /// 恢复后会留着死去那条时间线的值。界面类（Godot 对象、MegaCrit.Sts2.Core.Nodes 下）不去初始化：它们的静态初始化会调 Godot 原生接口，
    /// 无头下进程直接退出（GodotShim.Trap）；它们以后被初始化了，下次拍快照时补进根（RefreshLate），并记进 LateInit。
    /// 只在第一次拍快照时做，不用复制功能的进程不受影响。跑静态初始化时游戏的报错不算这一局的错：GodotShim 的计数原样还原。
    /// </summary>
    public static void EnsureDiscovered()
    {
        if (_writable != null) { RefreshLate(); return; }
        var sw = System.Diagnostics.Stopwatch.StartNew();
        TypeInit.Validate();
        // 引擎自己的类型先全部初始化（同 selfcheck 的第一步，安全）：不然像 Observation.DrawTop（已知牌堆顶的弱引用表）这样
        // 到第一场战斗才初始化的，在战斗前拍的快照里就没有它的静态字段（2026-11-16 整局模式的复制保真里列出）。游戏的类型仍不去碰
        foreach (var t in typeof(ClonePolicy).Assembly.GetTypes())
            if (!t.ContainsGenericParameters && t.Namespace == "Sts2Headless" && t != typeof(TypeInit) && t.DeclaringType != typeof(TypeInit))
                try { RuntimeHelpers.RunClassConstructor(t.TypeHandle); } catch { }
        var writable = new List<StaticRoot>();
        var ro = new List<StaticRoot>();
        int errTotal = GodotShim.ErrorTotal, benign = GodotShim.BenignTotal;
        string lastErr = GodotShim.LastError, lastFull = GodotShim.LastErrorFull;
        foreach (var asm in ClonePolicy.RootAssemblies)
        {
            Type[] types;
            try { types = asm.GetTypes(); }
            catch (ReflectionTypeLoadException e) { types = e.Types.Where(x => x != null).ToArray()!; }
            foreach (var t in types)
            {
                var fields = StaticFields(t);
                if (fields.Count == 0) continue;
                if (t.ContainsGenericParameters) { GenericDefinitions.Add($"{t.FullName}：{string.Join("、", fields.Select(f => f.Name))}"); continue; }
                if (!TypeInit.IsInitialized(t))
                {
                    if (TypeInit.Failed(t)) { InitFailures.Add($"{t.FullName}：静态初始化曾经出错"); continue; }
                    if (!ClonePolicy.SafeToInitialize(t)) { _pending.Add(t); continue; }
                    try { RuntimeHelpers.RunClassConstructor(t.TypeHandle); }
                    catch (Exception e) { InitFailures.Add($"{t.FullName}：{e.GetBaseException().Message}"); continue; }
                }
                Add(fields, writable, ro);
            }
        }
        GodotShim.ErrorTotal = errTotal; GodotShim.BenignTotal = benign; GodotShim.LastError = lastErr; GodotShim.LastErrorFull = lastFull;
        _writable = writable.ToArray();
        _readonly = ro.ToArray();
        DiscoverMs = sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>上次以来被游戏初始化了的界面类：补进根，并记下（它们在之前的快照里没有）。</summary>
    static void RefreshLate()
    {
        List<StaticRoot>? w = null, r = null;
        for (int i = _pending.Count - 1; i >= 0; i--)
        {
            var t = _pending[i];
            if (!TypeInit.IsInitialized(t)) continue;
            _pending.RemoveAt(i);
            LateInit.Add(t.FullName!);
            Console.Error.WriteLine($"[clone] 类型 {t.FullName} 在之前的快照之后才初始化，从这次起进根{(LateWritable(t) ? "（有可写静态字段，要核查）" : "")}");
            w ??= _writable!.ToList(); r ??= _readonly!.ToList();
            Add(StaticFields(t), w, r);
        }
        if (w != null) { _writable = w.ToArray(); _readonly = r!.ToArray(); }
    }

    /// <summary>晚初始化的类型有没有要核查的静态字段：可写的（编译器生成的委托缓存 &lt;&gt;9__ 除外），或只读但指向可变集合的。</summary>
    public static bool LateWritable(Type t) => StaticFields(t).Any(f =>
        (!f.IsInitOnly && !f.Name.StartsWith("<>9__")) || (f.IsInitOnly && typeof(System.Collections.ICollection).IsAssignableFrom(f.FieldType)));

    /// <summary>恢复时调用：有类型在拍快照之后才初始化时，这份快照管不到它们的静态字段（只记日志，自检里会报出来）。</summary>
    public static void CheckLate()
    {
        foreach (var t in _pending)
            if (TypeInit.IsInitialized(t)) { RefreshLate(); break; }
    }

    /// <summary>
    /// 根程序集里的线程静态字段（[ThreadStatic]）：不进快照（读的是当前线程的值）。它们在决策点之间不能带局面——
    /// 只能是一次调用之内的临时量、单调递增的计数、或与局面无关的缓存。钩子监听者列表池原来放在线程静态字段里，
    /// 停在钩子迭代中间的决策点（赌博筹码、工具箱选牌）拍快照、恢复后两处共用一个列表（2026-11-16，提交 b855217 改成普通静态字段）。
    /// clone-test census 列出全部，游戏更新或引擎改动后逐个确认。
    /// </summary>
    public static IEnumerable<FieldInfo> ThreadStatics() => ClonePolicy.RootAssemblies.SelectMany(a =>
    {
        try { return a.GetTypes(); } catch (ReflectionTypeLoadException e) { return e.Types.Where(x => x != null)!; }
    }).SelectMany(t => t!.GetFields(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
      .Where(f => f.IsDefined(typeof(ThreadStaticAttribute)));

    static List<FieldInfo> StaticFields(Type t) =>
        t.GetFields(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
            .Where(f => !f.IsLiteral && !ClonePolicy.SkipStatic(f)).ToList();

    static void Add(List<FieldInfo> fields, List<StaticRoot> writable, List<StaticRoot> ro)
    {
        foreach (var f in fields)
        {
            var root = new StaticRoot { Field = f, Get = Getter(f), Set = f.IsInitOnly ? null : Setter(f) };
            (f.IsInitOnly ? ro : writable).Add(root);
        }
    }

    static Func<object?> Getter(FieldInfo f)
    {
        var dm = new DynamicMethod("clone_sget_" + f.Name, typeof(object), Type.EmptyTypes, typeof(CloneRoots).Module, skipVisibility: true);
        var il = dm.GetILGenerator();
        il.Emit(OpCodes.Ldsfld, f);
        if (f.FieldType.IsValueType) il.Emit(OpCodes.Box, f.FieldType);
        il.Emit(OpCodes.Ret);
        return (Func<object?>)dm.CreateDelegate(typeof(Func<object?>));
    }

    static Action<object?> Setter(FieldInfo f)
    {
        var dm = new DynamicMethod("clone_sset_" + f.Name, null, new[] { typeof(object) }, typeof(CloneRoots).Module, skipVisibility: true);
        var il = dm.GetILGenerator();
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(f.FieldType.IsValueType ? OpCodes.Unbox_Any : OpCodes.Castclass, f.FieldType);
        il.Emit(OpCodes.Stsfld, f);
        il.Emit(OpCodes.Ret);
        return (Action<object?>)dm.CreateDelegate(typeof(Action<object?>));
    }
}

/// <summary>服务端的快照槽位（Server 的 snapshot / restore）。</summary>
public static class CloneSlots
{
    static readonly Dictionary<int, HeapSnapshot> _slots = new();
    public static int Count => _slots.Count;
    public static void Put(int slot, HeapSnapshot s) => _slots[slot] = s;
    public static HeapSnapshot Get(int slot) => _slots.TryGetValue(slot, out var s) ? s : throw new InvalidOperationException($"槽位 {slot} 没有快照");
    public static bool Drop(int slot) => _slots.Remove(slot);
    public static void Clear() => _slots.Clear();
}
