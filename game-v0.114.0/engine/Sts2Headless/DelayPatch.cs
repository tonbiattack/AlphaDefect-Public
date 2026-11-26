using System.Reflection;
using HarmonyLib;

namespace Sts2Headless;

/// <summary>
/// 游戏逻辑里的 Task.Delay 是真实时间等待（如 Amalgamator 事件里的 300 ms）：既拖慢模拟，又让续延在不确定的时刻回到消息泵。
/// 直接给 Task.Delay 打补丁：在消息泵线程（即游戏逻辑）上调用时立即完成，其他线程照常。
/// 不改调用方：给 async 状态机的 MoveNext 打 Harmony 补丁会让它行为异常（M1 实测，连原样转译都会出空引用）。
/// </summary>
public static class DelayPatch
{
    public static int Skipped;
    static long _item = -1;
    static int _streak;

    public static void Install(Harmony h)
    {
        var prefix = new HarmonyMethod(typeof(DelayPatch).GetMethod(nameof(Prefix), BindingFlags.NonPublic | BindingFlags.Static));
        foreach (MethodInfo m in typeof(Task).GetMethods(BindingFlags.Public | BindingFlags.Static).Where(m => m.Name == nameof(Task.Delay)))
            h.Patch(m, prefix: prefix);
    }

    static bool Prefix(ref Task __result)
    {
        if (System.Environment.CurrentManagedThreadId != Pump.Instance.OwnerThread) return true;
        // 等待循环（如 ActionExecutor.WaitForUnpause 的 while (_isPaused) await Task.Delay(50)）遇到立即完成的 Delay 会同步空转、永不让出线程。
        // 同一条续延里连续跳过太多次就抛出，让该任务出错（RunSession 按卡住结束这一局），而不是挂死整个引擎
        if (Pump.Instance.Executed != _item) { _item = Pump.Instance.Executed; _streak = 0; }
        if (++_streak > 100_000) throw new InvalidOperationException("Task.Delay 在同一条续延里被跳过超过 10 万次，疑似忙等");
        Skipped++;
        __result = Task.CompletedTask;
        return false;
    }
}
