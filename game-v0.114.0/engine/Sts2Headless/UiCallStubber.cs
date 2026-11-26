using System.Diagnostics;
using System.Reflection;
using HarmonyLib;

namespace Sts2Headless;

/// <summary>
/// 逻辑层偶尔直接调用界面层的方法（如 NHoverTipSet.Clear()、NCombatRoom.Instance、Boss 背景的动画）。
/// 扫描 sts2.dll 中非界面命名空间的全部方法，找出它们调用的界面层方法与属性，统一改为“什么都不做、返回默认值”：
/// Task 返回已完成任务，集合返回空集合，其余返回 null / 0 / false。
/// 无头下界面单例本来就不存在，返回 null 与游戏里“界面未加载”的分支一致。
/// </summary>
public static class UiCallStubber
{
    static readonly string[] UiNamespaces = { "MegaCrit.Sts2.Core.Nodes", "MegaCrit.sts2.Core.Nodes", "MegaCrit.Sts2.addons", "MegaCrit.Sts2.Core.Bindings", "MegaCrit.Sts2.Core.Audio" };
    public static readonly SortedSet<string> Stubbed = new();
    /// <summary>不置空的方法（如已写入假对象的单例 getter）。</summary>
    public static readonly HashSet<MethodBase> Exclude = new();
    /// <summary>置空后返回假对象而不是 null 的方法（逻辑层会在返回值上继续调用，且没有 ?. 保护）。</summary>
    public static readonly HashSet<MethodBase> FakeReturn = new();

    public static bool IsUi(Type? t)
    {
        for (; t != null; t = t.DeclaringType)
            if (t.Namespace != null && UiNamespaces.Any(ns => t.Namespace.StartsWith(ns))) return true;
        return false;
    }

    public static void Install(Harmony h, Assembly game)
    {
        var sw = Stopwatch.StartNew();
        var targets = new HashSet<MethodBase>();
        foreach (Type t in game.GetTypes())
        {
            if (IsUi(t)) continue;
            foreach (MethodBase m in t.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                         .Cast<MethodBase>().Concat(t.GetConstructors(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)))
            {
                if (m.GetMethodBody()?.GetILAsByteArray() is not { } il) continue;
                foreach (MethodBase callee in IlScan.Callees(il, m.Module, m))
                    if (callee is MethodInfo { IsAbstract: false } && !callee.ContainsGenericParameters && IsUi(callee.DeclaringType))
                        targets.Add(callee);   // 静态与实例方法都置空：实例来自 FakeUi 的假对象或根本拿不到
            }
        }
        var prefix = new HarmonyMethod(typeof(UiCallStubber).GetMethod(nameof(SkipOriginal), BindingFlags.NonPublic | BindingFlags.Static));
        foreach (MethodBase m in targets)
        {
            if (Harmony.GetPatchInfo(m) != null || Exclude.Contains(m)) continue;   // 已由 FakeScreens、FakeSingleton 等专门接管
            Type ret = ((MethodInfo)m).ReturnType;
            if (FakeReturn.Contains(m))
            {
                h.Patch(m, prefix: new HarmonyMethod(typeof(FakeValue<>).MakeGenericType(ret).GetMethod("Prefix")));
                Stubbed.Add($"{m.DeclaringType!.Name}.{m.Name}（假对象）");
                continue;
            }
            h.Patch(m, prefix: EmptyValue.CanProvide(ret)
                ? new HarmonyMethod(typeof(EmptyValue<>).MakeGenericType(ret).GetMethod("Prefix"))
                : prefix);
            Stubbed.Add($"{m.DeclaringType!.Name}.{m.Name}");
        }
        if (GodotShim.Verbose) Console.Error.WriteLine($"[shim] 界面静态调用置空：{targets.Count} 个，用时 {sw.ElapsedMilliseconds} ms");
    }

    /// <summary>跳过原方法；返回值保持默认（null / 0 / false）。</summary>
    static bool SkipOriginal() => false;
}

/// <summary>集合类型的“空值”：资源路径列表之类的界面静态属性返回空集合，调用方 AddRange / foreach 不会出错。</summary>
public static class EmptyValue
{
    public static bool CanProvide(Type t) => Create(t) != null;

    public static object? Create(Type t)
    {
        if (t == typeof(Task)) return Task.CompletedTask;
        if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Task<>))
        {
            Type r = t.GetGenericArguments()[0];
            object? value = r.IsValueType ? Activator.CreateInstance(r) : Create(r);
            return typeof(Task).GetMethod(nameof(Task.FromResult))!.MakeGenericMethod(r).Invoke(null, new[] { value });
        }
        if (t == typeof(string) || !typeof(System.Collections.IEnumerable).IsAssignableFrom(t)) return null;
        if (t.IsArray) return Array.CreateInstance(t.GetElementType()!, 0);
        Type? elem = t.IsGenericType ? t.GetGenericArguments().FirstOrDefault() : null;
        if (t.IsInterface && elem != null && t.IsAssignableFrom(elem.MakeArrayType())) return Array.CreateInstance(elem, 0);
        if (!t.IsAbstract && t.GetConstructor(Type.EmptyTypes) != null) return Activator.CreateInstance(t);
        return null;
    }
}

public static class EmptyValue<T>
{
    static readonly T Value = (T)EmptyValue.Create(typeof(T))!;
    public static bool Prefix(ref T __result) { __result = Value; return false; }
}

public static class FakeValue<T> where T : class
{
    static readonly T Value = FakeUi.Create<T>();
    public static bool Prefix(ref T __result) { __result = Value; return false; }
}

/// <summary>简易 IL 扫描：列出方法体中 call/callvirt/newobj/ldftn 引用的方法。</summary>
public static class IlScan
{
    public static IEnumerable<MethodBase> Callees(byte[] il, Module module, MethodBase owner)
    {
        Type[]? typeArgs = owner.DeclaringType?.IsGenericType == true ? owner.DeclaringType.GetGenericArguments() : null;
        Type[]? methodArgs = owner is MethodInfo { IsGenericMethod: true } mi ? mi.GetGenericArguments() : null;
        for (int i = 0; i + 4 < il.Length; i++)
        {
            byte op = il[i];
            bool isCall = op == 0x28 || op == 0x6F || op == 0x73;           // call / callvirt / newobj
            bool isFtn = op == 0xFE && i + 5 < il.Length && il[i + 1] == 0x06; // ldftn
            if (!isCall && !isFtn) continue;
            int token = BitConverter.ToInt32(il, isFtn ? i + 2 : i + 1);
            if ((token >> 24) is not (0x06 or 0x0A or 0x2B)) continue;      // MethodDef / MemberRef / MethodSpec
            MethodBase? m = null;
            try { m = module.ResolveMethod(token, typeArgs, methodArgs); } catch { }
            if (m != null) yield return m;
        }
    }
}
