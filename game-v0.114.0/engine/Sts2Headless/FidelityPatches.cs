using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2Headless;

/// <summary>
/// 测试模式里有少数分支改变了游戏内容（不只是跳过动画）。这些方法执行期间临时关闭 TestMode，走正式游戏的分支。
/// 这几处判断都在方法开头、第一个 await 之前，同步执行，所以前缀关、后缀开是准确的。
/// 不用 IL 转译：给 async 状态机的 MoveNext 打补丁会让它行为异常（见 DelayPatch）；TestMode.IsOn 是自动属性，会被内联，也不能给 getter 打补丁。
/// 每次游戏更新都要用 `audit-testmode` 重新列出所有 TestMode 分支，人工确认这张表是否需要增减。
/// </summary>
public static class FidelityPatches
{
    public static readonly List<string> Patched = new();

    public static void Install(Harmony h)
    {
        // 大锅：测试模式给固定的 5 瓶药水，正式游戏随机
        RealMode(h, GameCompat.Method(typeof(Cauldron), "GenerateRewards", "坩埚按正式模式发奖励"));
        // 召唤铃：测试模式给固定的 3 个遗物，正式游戏按稀有度随机
        RealMode(h, GameCompat.Method(typeof(CallingBell), "GenerateRewards", "召唤之铃按正式模式发奖励"));
        // 商店离开：测试模式不写本层历史（未买的卡和遗物）
        RealMode(h, GameCompat.Method(typeof(MerchantRoom), nameof(MerchantRoom.Exit), "离开商店按正式模式"));
        // 三选一卡包：测试模式永远选第一个；正式分支弹出选择界面，由 FakeScreens 接管成决策点
        RealMode(h, GameCompat.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseABundleScreen), "选卡包按正式模式"));
        // 商店药水定价：测试模式价格固定，正式游戏乘 0.95–1.05 的随机系数。少抽这一次随机数，之后所有商店的内容和价格都与原版同种子不一致
        RealMode(h, GameCompat.Method(typeof(MerchantPotionEntry), nameof(MerchantPotionEntry.CalcCost), "药水价格按正式模式"));
    }

    static void RealMode(Harmony h, MethodInfo? m)
    {
        GameCompat.Patch(h, m, "FidelityPatches 的目标方法（游戏可能已更新）",
            prefix: GameCompat.Own(typeof(FidelityPatches), nameof(Off)), finalizer: GameCompat.Own(typeof(FidelityPatches), nameof(On)));
        if (m != null) Patched.Add($"{m.DeclaringType!.FullName}.{m.Name}");
    }

    static void Off() => TestMode.IsOn = false;
    static Exception? On(Exception? __exception) { TestMode.IsOn = true; return __exception; }

    /// <summary>列出逻辑层所有读取 TestMode 的方法，用于游戏更新后人工复核。</summary>
    public static IEnumerable<string> Audit(Assembly game)
    {
        var isOn = AccessTools.PropertyGetter(typeof(TestMode), nameof(TestMode.IsOn));
        var isOff = AccessTools.PropertyGetter(typeof(TestMode), nameof(TestMode.IsOff));
        return game.GetTypes().Where(t => !UiCallStubber.IsUi(t)).SelectMany(t =>
                t.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
                    .Where(m => !m.IsAbstract && !m.ContainsGenericParameters && m.GetMethodBody()?.GetILAsByteArray() is { } il
                                && IlScan.Callees(il, m.Module, m).Any(c => c == isOn || c == isOff))
                    .Select(m => $"{t.FullName}.{m.Name}"))
            .OrderBy(s => s, StringComparer.Ordinal);
    }
}
