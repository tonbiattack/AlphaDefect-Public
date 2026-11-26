using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.TreasureRelicPicking;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>记录最近一次事件选项的执行任务，用来识别“选项执行到一半挂起”的引擎缺陷，避免反复选同一选项。</summary>
public static class EventOptionTracker
{
    public static Task? Pending { get; set; }

    public static void Install(HarmonyLib.Harmony h) =>
        GameCompat.Patch(h, GameCompat.Method(typeof(MegaCrit.Sts2.Core.Events.EventOption), nameof(MegaCrit.Sts2.Core.Events.EventOption.Chosen), "事件选项的结算任务"),
            "EventOption.Chosen", postfix: GameCompat.Own(typeof(EventOptionTracker), nameof(Postfix)));

    static void Postfix(Task __result) => Pending = __result;
}

/// <summary>
/// 正式游戏里由界面节点完成、但会改变游戏状态的流程，在无头下照原样复刻。
/// 依据是 `audit-ui` 清单（UiLogicAudit）；游戏更新后对比清单，新增的界面→逻辑调用要在这里或 RunSession 中补上。
/// </summary>
public sealed class UiFlows : IDisposable
{
    public UiFlows()
    {
        CombatManager.Instance.CombatWon += OnCombatWon;
        RunManager.Instance.TreasureRoomRelicSynchronizer.RelicsAwarded += OnRelicsAwarded;
    }

    public void Dispose()
    {
        CombatManager.Instance.CombatWon -= OnCombatWon;
        RunManager.Instance.TreasureRoomRelicSynchronizer.RelicsAwarded -= OnRelicsAwarded;
    }

    /// <summary>NCombatUi.OnCombatWon：发战斗奖励；不给奖励的战斗直接“继续”（ProceedWithoutRewards，原有 1 秒等待省略）。</summary>
    /// <summary>本进程内获胜的战斗数（导出到观测头部，供训练端判断战斗结束）。</summary>
    public static int CombatsWon { get; private set; }

    static void OnCombatWon(CombatRoom room)
    {
        CombatsWon++;
        if (room.Encounter.ShouldGiveRewards) TaskHelper.RunSafely(room.OfferRoomEndRewards());
        else TaskHelper.RunSafely(RunManager.Instance.ProceedFromTerminalRewardsScreenAsync());   // v0.113.0 原名改成同步包装，异步版加了 Async
    }

    /// <summary>NTreasureRoomRelicCollection.AnimateRelicAwards 的逻辑部分：按结果类型排序后发放遗物，其余玩家的抓取袋移到备用池。</summary>
    /// v0.113.0 加了 isFinished（Blitz 模式单人时宝箱分两轮挑；普通模式只有一轮，恒为 true），发放逻辑本身没变。
    static void OnRelicsAwarded(List<RelicPickingResult> results, bool isFinished)
    {
        var state = RunManager.Instance.DebugOnlyGetState()!;
        results.Sort((a, b) => a.type.CompareTo(b.type));
        foreach (var r in results)
        {
            var relic = r.relic.ToMutable();
            if (r.type != RelicPickingResultType.Skipped)
                TaskHelper.RunSafely(RelicCmd.Obtain(relic, r.player));
            foreach (var p in state.Players)
                if (p != r.player) p.RelicGrabBag.MoveToFallback(r.relic);
        }
    }
}
