using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// 戦闘の外の意思決定（報酬・経路）。無頭エンジンの RunSession にある判断と実行を、
/// 実機のゲームでもそのまま使えるように移したもの。
///
/// 報酬はエンジンの RewardsFlow をそのまま使う（RewardsSet の testSelector を差し替える
/// 仕組みで、エンジンでも同じものを使っている）。こちらが選んだら
/// RunManager.RewardsSetSynchronizer.SelectLocalReward でゲーム側に反映される。
///
/// まだ無いもの: イベント・ショップ・休息・宝箱・選牌。そこはゲームが待つので、
/// その局面ではフレームを送らずに止まる（プレイヤーが操作すれば続く）。
/// </summary>
internal static class OutOfCombat
{
    static bool _installed;

    public static void Install()
    {
        if (_installed) return;
        RewardsFlow.Install();          // RewardsSet.testSelector = RewardsFlow.Select
        _installed = true;
        Bridge.LogLine("reward flow installed (test selector)");
    }

    public static void ResetForNewRun() => RewardsFlow.Reset();

    /// <summary>いま判断すべき戦闘外の決定。無ければ null。</summary>
    public static Decision? Build(RunState run, Player me)
    {
        if (LiveCardReward.Active && LiveCardReward.Build() is { } cardReward) return cardReward;
        if (LiveRewards.Active && LiveRewards.Build(me) is { } rewards) return rewards;
        if (RewardsFlow.Pending is { } pr) return RewardsDecision(me, pr);
        if (run.CurrentRoom is EventRoom && EventDecision(run, me) is { } ev) return ev;
        // 戦闘が終わって報酬も閉じたあとは、部屋のオブジェクトがまだ CombatRoom のことがある。
        // エンジンと同じく、他にすることが無ければ地図の選択に落とす。
        if (run.CurrentRoom is MapRoom or CombatRoom || run.CurrentRoom is null) return MapDecision(run, me);
        return null;
    }

    /// <summary>エンジンの RewardsDecision と同じ並び（領得できる報酬、薬水、離開）。</summary>
    static Decision RewardsDecision(Player me, PendingRewards pr)
    {
        var d = new Decision(DecisionKind.Rewards) { Context = pr.IsTerminal ? "战斗奖励" : "额外奖励" };
        // カード報酬は画面（NCardRewardSelectionScreen）の選択待ちになる。画面の操作はまだ
        // 実装していないので、いまは候補から外して「取らない」で進む（デッキは育たない）。
        var available = pr.Available.Where(r => r is not CardReward).ToList();
        bool cardOnly = pr.Available.Count > 0 && available.Count == 0;
        foreach (var r in available)
            d.Add(ActionKind.TakeReward, "领 " + RewardLabel(r), () => pr.Take(r), a: r);
        if (!pr.Set.DisallowSkipping || available.Count == 0)
            d.Add(ActionKind.Leave, "离开", pr.Proceed);
        if (cardOnly && pr.Set.DisallowSkipping)
            Bridge.LogLine("card reward cannot be skipped here; the game will wait for the player");
        return d;
        AddPotionActions(me, d);
        if (!me.HasOpenPotionSlots && pr.Set.Rewards.Any(r => r is PotionReward && !r.SuccessfullySelected))
            AddPotionDiscards(me, d);
    }

    /// <summary>イベントの選択（エンジンの EventDecision と同じ。Neow もここで進む）。</summary>
    static Decision? EventDecision(RunState run, Player me)
    {
        var ev = RunManager.Instance.EventSynchronizer.GetLocalEvent();
        if (ev == null || ev.IsFinished) return null;
        var d = new Decision(DecisionKind.Event) { Context = ev.Id.Entry };
        var opts = ev.CurrentOptions;
        for (int i = 0; i < opts.Count; i++)
        {
            var opt = opts[i];
            if (opt.IsLocked || ExcludedEventOptions.Contains(opt.TextKey)) continue;
            int k = i;
            // NEventRoom.OptionButtonClicked: 「進む」系は Chosen を直接、それ以外は同期器経由
            if (opt.IsProceed)
                d.Add(ActionKind.EventOption, opt.TextKey, () => opt.Chosen(), a: opt);
            else
                d.Add(ActionKind.EventOption, opt.TextKey, () => RunManager.Instance.EventSynchronizer.ChooseLocalOption(k), a: opt);
        }
        if (d.Choices.Count == 0) return null;
        AddPotionActions(me, d);
        return d;
    }

    /// <summary>選ばない選択肢（エンジンと同じ。TRIAL の DOUBLE_DOWN は放棄確認だけなので出さない）。</summary>
    static readonly HashSet<string> ExcludedEventOptions = new() { "TRIAL.pages.REJECT.options.DOUBLE_DOWN" };

    static Decision MapDecision(RunState run, Player me)
    {
        var cur = run.CurrentMapPoint;
        var d = new Decision(DecisionKind.Map)
        {
            Context = $"第 {run.CurrentActIndex + 1} 幕 {(cur == null ? "起点" : $"({cur.coord.col},{cur.coord.row})")}",
        };
        foreach (var p in RunSessionMap.Travelable(run).OrderBy(p => p.coord.col))
        {
            var dest = p;
            d.Add(ActionKind.MapMove, $"→ ({p.coord.col},{p.coord.row}) {p.PointType}", () =>
            {
                var vote = new MapVote
                {
                    coord = dest.coord,
                    mapGenerationCount = RunManager.Instance.MapSelectionSynchronizer.MapGenerationCount,
                };
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                    new VoteForMapCoordAction(me, run.MapLocation, vote));
            }, a: p);
        }
        AddPotionActions(me, d);
        return d;
    }

    static void AddPotionActions(Player me, Decision d)
    {
        foreach (var p in UsablePotions(me))
        {
            PotionModel potion = p;
            if (potion.IsValidTarget(me.Creature) || potion.IsValidTarget(null))
                d.Add(ActionKind.UsePotion, "用 " + potion.Id.Entry, () => potion.EnqueueManualUse(null), a: potion);
        }
    }

    static void AddPotionDiscards(Player me, Decision d)
    {
        if (!me.CanUseOrRemovePotions) return;
        var slots = me.PotionSlots;
        for (int i = 0; i < slots.Count; i++)
        {
            if (slots[i] is not { IsQueued: false } potion) continue;
            uint slot = (uint)i;
            d.Add(ActionKind.DiscardPotion, "丢弃 " + potion.Id.Entry, () =>
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                    new DiscardPotionGameAction(me, slot, CombatInProgress)), a: potion);
        }
    }

    static bool CombatInProgress
    {
        get
        {
            try { return MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress; }
            catch { return false; }
        }
    }

    static IEnumerable<PotionModel> UsablePotions(Player me) =>
        me.CanUseOrRemovePotions && !me.Creature.IsDead
            ? me.Potions.Where(p => !p.IsQueued && p.PassesCustomUsabilityCheck &&
                                    p.Usage == PotionUsage.AnyTime)
            : Enumerable.Empty<PotionModel>();

    static string RewardLabel(Reward r) => r switch
    {
        GoldReward => "金币",
        PotionReward p => "药水 " + p.Potion?.Id.Entry,
        RelicReward rr => "遗物 " + rr.Relic?.Id.Entry,
        CardReward => "卡牌奖励",
        _ => r.GetType().Name,
    };
}
