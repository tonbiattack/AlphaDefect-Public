using System.Reflection;
using System.Reflection.Emit;

namespace Sts2Headless;

/// <summary>
/// 局面复制的核心：原地快照（设计与取舍见 engine/Sts2Headless/Engine/Clone/局面复制.md）。
///
/// 拍快照（Capture）：从静态根（CloneRoots：游戏与引擎程序集的全部静态字段）和调用方给的额外根出发，走遍对象图；
/// 每个可变对象用 MemberwiseClone 浅拷贝一份——拷贝里的引用仍指向原来的对象。可写的静态字段记下当时的值。
/// 恢复（Restore）：把每份拷贝的全部字段原地写回原对象（数组用 Array.Copy），再写回静态字段。
///
/// 对象身份始终不变，所以：按身份散列的字典与集合、闭包与委托、挂起的 async 状态机与 TaskCompletionSource、
/// 游戏里互相引用的模型都不用重建，恢复后指向的还是同一批对象、内容回到拍快照时的样子；拍快照之后新建的对象，
/// 恢复后没有人再引用它们，由 GC 回收。快照只读，可以恢复任意多次。
///
/// 前提：拍快照时引擎停在决策点（消息泵空闲，游戏在等玩家输入，只有主线程在跑游戏逻辑）。
/// 哪些类型不往下走、不记（字符串、反射对象、流、规范模型……）由 ClonePolicy 决定；它是游戏更新时要复核的“接触面”。
/// </summary>
public sealed class HeapSnapshot
{
    readonly object[] _live;
    readonly object[] _copy;
    readonly TypePlan[] _plan;
    readonly object?[] _statics;          // CloneRoots.Writable 里每个静态字段拍快照时的值
    /// <summary>调用方给的额外根（服务端的会话、当前决策等），恢复后原样交回。</summary>
    public readonly object?[] Extra;

    public int Objects => _live.Length;
    /// <summary>最近一次拍快照里遍历对象图（含读静态字段）用的毫秒数（剖析用）。</summary>
    public static double WalkMs;
    public int Visited { get; }
    public double CaptureMs { get; }

    HeapSnapshot(object[] live, object[] copy, TypePlan[] plan, object?[] statics, object?[] extra, int visited, double ms)
    {
        _live = live; _copy = copy; _plan = plan; _statics = statics; Extra = extra; Visited = visited; CaptureMs = ms;
    }

    /// <summary>拍快照。extra 里的对象一并走到、恢复时一并恢复（它们的字段），Extra 原样交回。</summary>
    public static HeapSnapshot Capture(params object?[] extra)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        CloneRoots.EnsureDiscovered();
        var w = HeapWalker.Shared;
        w.Begin();
        var writable = CloneRoots.Writable;
        var statics = new object?[writable.Length];
        for (int i = 0; i < writable.Length; i++) w.Push(statics[i] = writable[i].Get());
        foreach (var r in CloneRoots.ReadOnly) w.Push(r.Get());
        foreach (var x in extra) w.Push(x);
        w.Run();
        WalkMs = sw.Elapsed.TotalMilliseconds;
        int n = w.Captured.Count;
        var live = new object[n];
        var copy = new object[n];
        var plan = new TypePlan[n];
        for (int i = 0; i < n; i++)
        {
            var o = w.Captured[i];
            live[i] = o;
            plan[i] = w.CapturedPlans[i];
            copy[i] = plan[i].Capturer is { } cap ? cap(o) : TypePlans.Shallow(o);
            // 带终结器的对象（WeakReference、Godot 对象等）：拷贝不能被终结，否则它会释放与原对象共用的句柄
            if (plan[i].HasFinalizer) GC.SuppressFinalize(copy[i]);
        }
        int visited = w.VisitedCount;
        var keepParents = w.Parent;
        w.End();
        w.Parent = keepParents;
        return new HeapSnapshot(live, copy, plan, statics, (object?[])extra.Clone(), visited, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>恢复到拍快照时的样子（快照本身不变，可反复恢复）。返回耗时（毫秒）。</summary>
    public double Restore()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var live = _live; var copy = _copy; var plan = _plan;
        for (int i = 0; i < live.Length; i++) plan[i].CopyInto!(copy[i], live[i]);
        var writable = CloneRoots.Writable;
        int ns = Math.Min(writable.Length, _statics.Length);      // 之后补进根的界面类在这份快照里没有值（CloneRoots.LateInit）
        for (int i = 0; i < ns; i++) writable[i].Set!(_statics[i]);
        CloneRoots.CheckLate();
        return sw.Elapsed.TotalMilliseconds;
    }

    /// <summary>自检用：拍快照以来字段被改过的对象（逐字段比较拷贝与原对象：引用比身份，值比内容）。</summary>
    internal IEnumerable<(object Live, TypePlan Plan)> Changed()
    {
        for (int i = 0; i < _live.Length; i++)
            if (!TypePlans.SameFields(_plan[i], _copy[i], _live[i])) yield return (_live[i], _plan[i]);
    }

    /// <summary>自检用：某个对象拍快照以来变了的字段名（反射读，读不了的字段标“?”）。</summary>
    internal IEnumerable<string> ChangedFieldNames(object o)
    {
        int i = Array.FindIndex(_live, x => ReferenceEquals(x, o));
        if (i < 0 || _plan[i].Type.IsArray || _plan[i].Capturer != null) yield break;
        foreach (var f in _plan[i].Fields)
        {
            bool diff, lazy = false;
            try { var a = f.GetValue(_copy[i]); var b = f.GetValue(o); diff = f.FieldType.IsValueType ? !Equals(a, b) : !ReferenceEquals(a, b); lazy = a is null && !f.FieldType.IsValueType; }
            catch { diff = true; }
            if (diff) yield return lazy ? f.Name + "（惰性，原为空）" : f.Name;
        }
    }

    /// <summary>自检用：拍快照以来值变了的可写静态字段。</summary>
    internal IEnumerable<CloneRoots.StaticRoot> ChangedStatics()
    {
        var writable = CloneRoots.Writable;
        for (int i = 0; i < Math.Min(writable.Length, _statics.Length); i++)
        {
            object? now = writable[i].Get(), then = _statics[i];
            bool same = writable[i].Field.FieldType.IsValueType ? Equals(now, then) : ReferenceEquals(now, then);
            if (!same) yield return writable[i];
        }
    }

    internal IReadOnlyList<object> LiveObjects => _live;
    internal IReadOnlyList<TypePlan> Plans => _plan;
}

/// <summary>对象图的遍历器：去重（按身份）、按类型取处理方式，记下要拍的对象。单线程，跨快照复用以免反复分配。</summary>
public sealed class HeapWalker
{
    internal static readonly HeapWalker Shared = new();
    readonly HashSet<object> _seen = new(ReferenceEqualityComparer.Instance);
    readonly Stack<object> _todo = new();
    internal readonly List<object> Captured = new();
    internal readonly List<TypePlan> CapturedPlans = new();
    /// <summary>自检（clone-test census）用：每个对象是从谁那里走到的。平时为 null。</summary>
    internal Dictionary<object, object>? Parent;
    object? _current;

    internal int VisitedCount => _seen.Count;
    readonly List<(object Table, TypePlan Plan)> _weak = new();
    /// <summary>走到过（弱引用表只存键走得到的条目）。</summary>
    internal bool Seen(object o) => _seen.Contains(o);

    internal void Begin() { _seen.Clear(); _todo.Clear(); Captured.Clear(); CapturedPlans.Clear(); _weak.Clear(); _current = null; _pushType = null; }

    /// <summary>拍完后清掉引用，免得遍历器自己让上一局的对象一直活着。</summary>
    internal void End() { _seen.Clear(); _todo.Clear(); Captured.Clear(); CapturedPlans.Clear(); _weak.Clear(); _current = null; }

    /// <summary>交给遍历器一个被引用的对象（生成的 Visit 代码调用）。</summary>
    public void Push(object? o)
    {
        if (o is null || o is string) return;
        // 整个类型都是叶子的（反射对象、ModelId、装箱基元……）在这里就挡掉，不进去重集合（拍快照的大半时间花在走叶子上）
        var t = o.GetType();
        if (!ReferenceEquals(t, _pushType)) { _pushType = t; var p = TypePlans.Get(t); _pushLeaf = p.Leaf; _pushLeafIf = p.LeafIf; }
        if (_pushLeaf || (_pushLeafIf != null && _pushLeafIf(o))) return;
        if (_seen.Add(o))
        {
            _todo.Push(o);
            if (Parent != null && _current != null) Parent[o] = _current;
        }
    }
    Type? _pushType;
    bool _pushLeaf;
    Func<object, bool>? _pushLeafIf;
    internal void ForgetTypeCache() => _pushType = null;

    /// <summary>
    /// 走完整张图。弱引用表（ConditionalWeakTable）先不往下走：别处都走完以后，只把“键在别处也走得到”的条目的值交给遍历器，
    /// 反复做到没有新条目为止（值又可能让别的键走得到）。键已经死了、只是 GC 还没回收的条目（比如上一场战斗的牌堆）不进快照，
    /// 否则会把整场旧战斗拖进来、还让它一直活着。
    /// </summary>
    internal void Run()
    {
        while (true)
        {
            Walk();
            bool more = false;
            foreach (var (t, p) in _weak)
                foreach (var (k, v) in p.Pairs!(t))
                    if (_seen.Contains(k) && v != null && !_seen.Contains(v)) { Push(v); more = true; }
            if (!more) break;
        }
    }

    void Walk()
    {
        Type? lastType = null;
        TypePlan? last = null;
        while (_todo.Count > 0)
        {
            var o = _todo.Pop();
            var t = o.GetType();
            var p = ReferenceEquals(t, lastType) ? last! : TypePlans.Get(t);
            lastType = t; last = p;
            if (p.Leaf || (p.LeafIf != null && p.LeafIf(o))) continue;
            if (p.Pairs != null) { _weak.Add((o, p)); Captured.Add(o); CapturedPlans.Add(p); continue; }
            _current = o;
            p.Visit?.Invoke(o, this);
            if (p.Capture) { Captured.Add(o); CapturedPlans.Add(p); }
        }
        _current = null;
    }
}

/// <summary>一个类型在快照里怎么处理。</summary>
public sealed class TypePlan
{
    public readonly Type Type;
    /// <summary>不往下走、也不记（字符串、反射对象、流……，见 ClonePolicy）。</summary>
    public bool Leaf;
    /// <summary>按实例判断要不要当叶子（规范模型：游戏的模板，玩的时候不改）。</summary>
    public Func<object, bool>? LeafIf;
    /// <summary>记下内容、恢复时写回。委托等不可变对象只往下走、不记。</summary>
    public bool Capture;
    /// <summary>把它直接引用的对象交给遍历器（经过结构体字段也算）。没有引用字段时为 null。</summary>
    public Action<object, HeapWalker>? Visit;
    /// <summary>把第一个参数（拷贝）的全部字段写进第二个参数（原对象）。</summary>
    public Action<object, object>? CopyInto;
    /// <summary>自检用的逐字段比较。</summary>
    public FieldInfo[] Fields = Array.Empty<FieldInfo>();
    /// <summary>不用浅拷贝、自己记内容的类型（ConditionalWeakTable：记下键值对，恢复时清空再放回）。为 null 时用 MemberwiseClone。</summary>
    public Func<object, object>? Capturer;
    /// <summary>弱引用表：列出全部键值对。</summary>
    public Func<object, IEnumerable<(object Key, object? Value)>>? Pairs;
    /// <summary>自己记内容的类型的比较（拷贝、原对象）。</summary>
    public Func<object, object, bool>? Equal;
    /// <summary>有终结器：拷贝出来的那份要 GC.SuppressFinalize（它与原对象共用句柄之类的资源）。</summary>
    public bool HasFinalizer;
    public string Why = "";

    public TypePlan(Type t) { Type = t; }
}

/// <summary>按类型生成并缓存 TypePlan：用 IL 生成遍历与写回的代码（跳过可见性检查，能读写私有字段与只读字段）。</summary>
public static class TypePlans
{
    static readonly Dictionary<Type, TypePlan> _cache = new();
    static readonly MethodInfo PushMethod = typeof(HeapWalker).GetMethod(nameof(HeapWalker.Push))!;
    static readonly Func<object, object> _memberwiseClone = (Func<object, object>)Delegate.CreateDelegate(
        typeof(Func<object, object>), typeof(object).GetMethod("MemberwiseClone", BindingFlags.Instance | BindingFlags.NonPublic)!);

    public static int Count => _cache.Count;
    public static IEnumerable<TypePlan> All => _cache.Values;

    public static object Shallow(object o) => _memberwiseClone(o);

    /// <summary>清掉缓存的类型计划（漏记审计切换策略时用）。</summary>
    internal static void Reset() { _cache.Clear(); HeapWalker.Shared.ForgetTypeCache(); }

    public static TypePlan Get(Type t)
    {
        if (_cache.TryGetValue(t, out var p)) return p;
        p = Build(t);
        _cache[t] = p;
        return p;
    }

    static TypePlan Build(Type t)
    {
        var p = new TypePlan(t);
        if (ClonePolicy.LeafType(t) is { } why) { p.Leaf = true; p.Why = why; return p; }
        p.LeafIf = ClonePolicy.LeafInstance(t);
        if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(System.Runtime.CompilerServices.ConditionalWeakTable<,>)) { WeakTable(p); return p; }
        p.HasFinalizer = HasFinalizer(t);
        if (t.IsArray)
        {
            var e = t.GetElementType()!;
            p.Capture = true;
            p.CopyInto = static (s, d) => Array.Copy((Array)s, (Array)d, ((Array)s).Length);
            if (!t.IsSZArray)
                p.Visit = HasRefs(e) ? static (o, w) => { foreach (var x in (Array)o) w.Push(x); } : null;
            else if (!e.IsValueType)
                p.Visit = static (o, w) => { foreach (var x in (object?[])o) w.Push(x); };
            else if (HasRefs(e))
                p.Visit = StructArrayVisit(t, e);
            return p;
        }
        if (t.IsValueType)
        {
            // 装箱的结构体：基元、枚举的箱子没人原地改，当叶子；其余（装箱的枚举器等）照常记
            if (t.IsPrimitive || t.IsEnum || !AllInstanceFields(t).Any()) { p.Leaf = true; p.Why = "装箱的基元或枚举"; return p; }
            p.Fields = AllInstanceFields(t).ToArray();
            p.Capture = true;
            p.Visit = HasRefs(t) ? ObjectVisit(t) : null;
            p.CopyInto = BoxCopy(t);
            return p;
        }
        p.Fields = AllInstanceFields(t).ToArray();
        if (p.Fields.Length == 0) { p.Leaf = true; p.Why = "没有实例字段"; return p; }
        p.Visit = p.Fields.Any(f => !f.FieldType.IsValueType || HasRefs(f.FieldType)) ? ObjectVisit(t) : null;
        // 委托创建后不再改字段（合并、移除都生成新委托）：只往下走（目标对象），不记
        p.Capture = !typeof(Delegate).IsAssignableFrom(t);
        if (p.Capture) p.CopyInto = ObjectCopy(t, p.Fields);
        return p;
    }

    /// <summary>
    /// ConditionalWeakTable&lt;K,V&gt;：内部容器靠依赖句柄挂着键值，浅拷贝不对。拍快照时记下全部键值对（值照常往下走、照常记），
    /// 恢复时 Clear 再逐个 AddOrUpdate 放回。只有引擎在用（DrawTop 的已知牌堆顶、PerfPatches 的生物列表缓存），条目很少。
    /// </summary>
    static void WeakTable(TypePlan p)
    {
        var t = p.Type;
        var kv = typeof(KeyValuePair<,>).MakeGenericType(t.GetGenericArguments());
        var key = kv.GetProperty("Key")!; var val = kv.GetProperty("Value")!;
        var clear = t.GetMethod("Clear")!; var put = t.GetMethod("AddOrUpdate")!;
        IEnumerable<(object, object?)> All(object o)
        {
            foreach (var e in (System.Collections.IEnumerable)o) yield return (key.GetValue(e)!, val.GetValue(e));
        }
        // 只记键走得到的条目（HeapWalker.Run 的说明）；拍快照时遍历器还没清空，可以问它
        object[] Pairs(object o)
        {
            var list = new List<object?>();
            foreach (var (k, v) in All(o)) if (HeapWalker.Shared.Seen(k)) { list.Add(k); list.Add(v); }
            return list.ToArray()!;
        }
        p.Capture = true;
        p.Pairs = All;
        p.Capturer = o => Pairs(o);
        p.CopyInto = (pairs, live) =>
        {
            clear.Invoke(live, null);
            var a = (object[])pairs;
            for (int i = 0; i < a.Length; i += 2) put.Invoke(live, new[] { a[i], a[i + 1] });
        };
        p.Equal = (pairs, live) =>
        {
            var now = All(live).ToList();
            var a = (object[])pairs;
            if (now.Count != a.Length / 2) return false;
            for (int i = 0; i < a.Length; i += 2) if (!now.Any(x => ReferenceEquals(x.Item1, a[i]) && ReferenceEquals(x.Item2, a[i + 1]))) return false;
            return true;
        };
        p.Why = "弱引用表（按键值对存取）";
    }

    static bool HasFinalizer(Type t)
    {
        for (var c = t; c != null && c != typeof(object); c = c.BaseType)
            if (c.GetMethod("Finalize", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly, null, Type.EmptyTypes, null) != null) return true;
        return false;
    }

    /// <summary>类型（含各级基类）的全部实例字段。</summary>
    internal static IEnumerable<FieldInfo> AllInstanceFields(Type t)
    {
        for (var c = t; c != null && c != typeof(object) && c != typeof(ValueType); c = c.BaseType)
            foreach (var f in c.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                yield return f;
    }

    static readonly Dictionary<Type, bool> _hasRefs = new();

    /// <summary>值类型里有没有引用（直接的或经过嵌套结构体的）。</summary>
    internal static bool HasRefs(Type t)
    {
        if (!t.IsValueType) return true;
        if (t.IsPrimitive || t.IsEnum || t.IsPointer) return false;
        if (_hasRefs.TryGetValue(t, out bool r)) return r;
        _hasRefs[t] = false;     // 结构体不会包含自己；先占位
        r = t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic).Any(f => !f.FieldType.IsPointer && (!f.FieldType.IsValueType || HasRefs(f.FieldType)));
        _hasRefs[t] = r;
        return r;
    }

    /// <summary>从类型出发、经过结构体字段、终点是引用类型字段的各条路径。</summary>
    static void RefPaths(Type t, List<FieldInfo> prefix, List<FieldInfo[]> paths)
    {
        var fields = t.IsValueType ? t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic) : AllInstanceFields(t).ToArray();
        foreach (var f in fields)
        {
            var ft = f.FieldType;
            if (ft.IsPointer || ft.IsByRefLike) continue;
            prefix.Add(f);
            if (!ft.IsValueType) paths.Add(prefix.ToArray());
            else if (HasRefs(ft)) RefPaths(ft, prefix, paths);
            prefix.RemoveAt(prefix.Count - 1);
        }
    }

    static void EmitPath(ILGenerator il, FieldInfo[] path)
    {
        for (int i = 0; i < path.Length - 1; i++) il.Emit(OpCodes.Ldflda, path[i]);
        il.Emit(OpCodes.Ldfld, path[^1]);
    }

    static DynamicMethod NewMethod(string name, Type[] args) =>
        new("clone_" + name, null, args, typeof(TypePlans).Module, skipVisibility: true);

    /// <summary>对象（或装箱的结构体）：把每个引用字段交给遍历器。</summary>
    static Action<object, HeapWalker>? ObjectVisit(Type t)
    {
        var paths = new List<FieldInfo[]>();
        RefPaths(t, new List<FieldInfo>(), paths);
        if (paths.Count == 0) return null;
        var dm = NewMethod("visit_" + t.Name, new[] { typeof(object), typeof(HeapWalker) });
        var il = dm.GetILGenerator();
        var self = il.DeclareLocal(t.IsValueType ? t.MakeByRefType() : t);
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(t.IsValueType ? OpCodes.Unbox : OpCodes.Castclass, t);
        il.Emit(OpCodes.Stloc, self);
        foreach (var path in paths)
        {
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, self);
            EmitPath(il, path);
            il.Emit(OpCodes.Call, PushMethod);
        }
        il.Emit(OpCodes.Ret);
        return (Action<object, HeapWalker>)dm.CreateDelegate(typeof(Action<object, HeapWalker>));
    }

    /// <summary>结构体数组（如 Dictionary 的 Entry[]）：逐个元素把引用字段交给遍历器。</summary>
    static Action<object, HeapWalker>? StructArrayVisit(Type arr, Type e)
    {
        var paths = new List<FieldInfo[]>();
        RefPaths(e, new List<FieldInfo>(), paths);
        if (paths.Count == 0) return null;
        var dm = NewMethod("visitarr_" + e.Name, new[] { typeof(object), typeof(HeapWalker) });
        var il = dm.GetILGenerator();
        var a = il.DeclareLocal(arr);
        var i = il.DeclareLocal(typeof(int));
        var top = il.DefineLabel();
        var test = il.DefineLabel();
        il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Castclass, arr); il.Emit(OpCodes.Stloc, a);
        il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Stloc, i);
        il.Emit(OpCodes.Br, test);
        il.MarkLabel(top);
        foreach (var path in paths)
        {
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, a); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldelema, e);
            EmitPath(il, path);
            il.Emit(OpCodes.Call, PushMethod);
        }
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I4_1); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);
        il.MarkLabel(test);
        il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, a); il.Emit(OpCodes.Ldlen); il.Emit(OpCodes.Conv_I4);
        il.Emit(OpCodes.Blt, top);
        il.Emit(OpCodes.Ret);
        return (Action<object, HeapWalker>)dm.CreateDelegate(typeof(Action<object, HeapWalker>));
    }

    /// <summary>对象：逐个实例字段 dst.f = src.f（含只读字段与各级基类的私有字段；引用字段经写屏障）。</summary>
    static Action<object, object> ObjectCopy(Type t, FieldInfo[] fields)
    {
        var dm = NewMethod("copy_" + t.Name, new[] { typeof(object), typeof(object) });
        var il = dm.GetILGenerator();
        var src = il.DeclareLocal(t);
        var dst = il.DeclareLocal(t);
        il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Castclass, t); il.Emit(OpCodes.Stloc, src);
        il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Castclass, t); il.Emit(OpCodes.Stloc, dst);
        foreach (var f in fields)
        {
            il.Emit(OpCodes.Ldloc, dst);
            il.Emit(OpCodes.Ldloc, src);
            il.Emit(OpCodes.Ldfld, f);
            il.Emit(OpCodes.Stfld, f);
        }
        il.Emit(OpCodes.Ret);
        return (Action<object, object>)dm.CreateDelegate(typeof(Action<object, object>));
    }

    /// <summary>装箱的结构体：把整个值拷进箱子。</summary>
    static Action<object, object> BoxCopy(Type t)
    {
        var dm = NewMethod("copybox_" + t.Name, new[] { typeof(object), typeof(object) });
        var il = dm.GetILGenerator();
        il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Unbox, t);
        il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Unbox, t); il.Emit(OpCodes.Ldobj, t);
        il.Emit(OpCodes.Stobj, t);
        il.Emit(OpCodes.Ret);
        return (Action<object, object>)dm.CreateDelegate(typeof(Action<object, object>));
    }

    /// <summary>自检用：两份同类型对象的字段是否相同（引用比身份；其余逐个基本字段按位比较，浮点不受 NaN 影响）。比较器按类型用 IL 生成。</summary>
    internal static bool SameFields(TypePlan p, object a, object b)
    {
        if (p.Equal != null) return p.Equal(a, b);
        if (!_equal.TryGetValue(p.Type, out var eq)) _equal[p.Type] = eq = BuildEqual(p.Type);
        return eq(a, b);
    }

    static readonly Dictionary<Type, Func<object, object, bool>> _equal = new();

    /// <summary>从类型出发、经过结构体字段、终点是引用或基本类型字段的全部路径（比较用）。</summary>
    static void LeafPaths(Type t, List<FieldInfo> prefix, List<FieldInfo[]> paths, bool root)
    {
        var fields = root && !t.IsValueType ? AllInstanceFields(t).ToArray() : t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        foreach (var f in fields)
        {
            var ft = f.FieldType;
            prefix.Add(f);
            if (!ft.IsValueType || ft.IsPrimitive || ft.IsEnum || ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) paths.Add(prefix.ToArray());
            else LeafPaths(ft, prefix, paths, false);
            prefix.RemoveAt(prefix.Count - 1);
        }
    }

    /// <summary>栈顶是值的地址时，按类型大小把它当整数读出来（按位比较）。</summary>
    static void EmitLoadBits(ILGenerator il, Type t)
    {
        if (t.IsEnum) t = Enum.GetUnderlyingType(t);
        if (t.IsPointer || t == typeof(IntPtr) || t == typeof(UIntPtr)) { il.Emit(OpCodes.Ldind_I); return; }
        int size = System.Runtime.InteropServices.Marshal.SizeOf(t == typeof(bool) ? typeof(byte) : t == typeof(char) ? typeof(ushort) : t);
        il.Emit(size switch { 1 => OpCodes.Ldind_U1, 2 => OpCodes.Ldind_U2, 4 => OpCodes.Ldind_I4, _ => OpCodes.Ldind_I8 });
    }

    /// <summary>emitRoot 把“根”放上栈（类时是对象引用，结构体时是地址），再沿路径取到叶子的值。</summary>
    static void EmitLeaf(ILGenerator il, Action emitRoot, FieldInfo[] path)
    {
        emitRoot();
        for (int i = 0; i < path.Length - 1; i++) il.Emit(OpCodes.Ldflda, path[i]);
        var last = path[^1];
        if (!last.FieldType.IsValueType) il.Emit(OpCodes.Ldfld, last);
        else { il.Emit(OpCodes.Ldflda, last); EmitLoadBits(il, last.FieldType); }
    }

    static Func<object, object, bool> BuildEqual(Type t)
    {
        if (t.IsArray && !t.IsSZArray)
            return static (a, b) =>
            {
                var x = ((Array)a).GetEnumerator(); var y = ((Array)b).GetEnumerator();
                while (x.MoveNext() && y.MoveNext()) if (!(x.Current?.GetType().IsValueType == true ? Equals(x.Current, y.Current) : ReferenceEquals(x.Current, y.Current))) return false;
                return true;
            };
        var dm = new DynamicMethod("clone_eq_" + t.Name, typeof(bool), new[] { typeof(object), typeof(object) }, typeof(TypePlans).Module, skipVisibility: true);
        var il = dm.GetILGenerator();
        var no = il.DefineLabel();
        if (t.IsArray)
        {
            var e = t.GetElementType()!;
            var a = il.DeclareLocal(t); var b = il.DeclareLocal(t); var i = il.DeclareLocal(typeof(int));
            var top = il.DefineLabel(); var test = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Castclass, t); il.Emit(OpCodes.Stloc, a);
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Castclass, t); il.Emit(OpCodes.Stloc, b);
            il.Emit(OpCodes.Ldloc, a); il.Emit(OpCodes.Ldlen); il.Emit(OpCodes.Ldloc, b); il.Emit(OpCodes.Ldlen); il.Emit(OpCodes.Bne_Un, no);
            il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Stloc, i); il.Emit(OpCodes.Br, test);
            il.MarkLabel(top);
            if (!e.IsValueType)
            {
                foreach (var arr in new[] { a, b }) { il.Emit(OpCodes.Ldloc, arr); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldelem_Ref); }
                il.Emit(OpCodes.Bne_Un, no);
            }
            else if (e.IsPrimitive || e.IsEnum || e.IsPointer || e == typeof(IntPtr) || e == typeof(UIntPtr))
            {
                foreach (var arr in new[] { a, b }) { il.Emit(OpCodes.Ldloc, arr); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldelema, e); EmitLoadBits(il, e); }
                il.Emit(OpCodes.Bne_Un, no);
            }
            else
            {
                var paths = new List<FieldInfo[]>();
                LeafPaths(e, new List<FieldInfo>(), paths, false);
                foreach (var path in paths)
                {
                    foreach (var arr in new[] { a, b }) EmitLeaf(il, () => { il.Emit(OpCodes.Ldloc, arr); il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldelema, e); }, path);
                    il.Emit(OpCodes.Bne_Un, no);
                }
            }
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I4_1); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);
            il.MarkLabel(test);
            il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldloc, a); il.Emit(OpCodes.Ldlen); il.Emit(OpCodes.Conv_I4); il.Emit(OpCodes.Blt, top);
        }
        else
        {
            var paths = new List<FieldInfo[]>();
            LeafPaths(t, new List<FieldInfo>(), paths, true);
            var a = il.DeclareLocal(t.IsValueType ? t.MakeByRefType() : t); var b = il.DeclareLocal(t.IsValueType ? t.MakeByRefType() : t);
            il.Emit(OpCodes.Ldarg_0); il.Emit(t.IsValueType ? OpCodes.Unbox : OpCodes.Castclass, t); il.Emit(OpCodes.Stloc, a);
            il.Emit(OpCodes.Ldarg_1); il.Emit(t.IsValueType ? OpCodes.Unbox : OpCodes.Castclass, t); il.Emit(OpCodes.Stloc, b);
            foreach (var path in paths)
            {
                EmitLeaf(il, () => il.Emit(OpCodes.Ldloc, a), path);
                EmitLeaf(il, () => il.Emit(OpCodes.Ldloc, b), path);
                il.Emit(OpCodes.Bne_Un, no);
            }
        }
        il.Emit(OpCodes.Ldc_I4_1); il.Emit(OpCodes.Ret);
        il.MarkLabel(no);
        il.Emit(OpCodes.Ldc_I4_0); il.Emit(OpCodes.Ret);
        return (Func<object, object, bool>)dm.CreateDelegate(typeof(Func<object, object, bool>));
    }
}
