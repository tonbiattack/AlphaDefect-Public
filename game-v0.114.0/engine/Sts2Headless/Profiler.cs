using System.Diagnostics;
using System.Reflection;
using HarmonyLib;

namespace Sts2Headless;

/// <summary>调试用：给若干方法计时（同步部分；async 方法只计到第一个挂起点）。</summary>
public static class Profiler
{
    public static readonly Dictionary<string, (double ms, int n)> Totals = new();
    static readonly Harmony H = new("sts2headless.profiler");
    [ThreadStatic] static Stack<long>? _starts;

    public static void Watch(MethodBase m)
    {
        H.Patch(m, prefix: new HarmonyMethod(typeof(Profiler).GetMethod(nameof(Pre), BindingFlags.NonPublic | BindingFlags.Static)),
            finalizer: new HarmonyMethod(typeof(Profiler).GetMethod(nameof(Post), BindingFlags.NonPublic | BindingFlags.Static)));
    }

    static void Pre() => (_starts ??= new()).Push(Stopwatch.GetTimestamp());

    static Exception? Post(MethodBase __originalMethod, Exception? __exception)
    {
        double ms = Stopwatch.GetElapsedTime(_starts!.Pop()).TotalMilliseconds;
        string k = $"{__originalMethod.DeclaringType?.Name}.{__originalMethod.Name}";
        var (t, n) = Totals.GetValueOrDefault(k);
        Totals[k] = (t + ms, n + 1);
        return __exception;
    }
}
