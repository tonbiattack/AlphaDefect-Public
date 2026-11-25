using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// 実機の報酬画面（NRewardsScreen）を AI から操作する。
///
/// 無頭エンジンは RewardsSet.testSelector を差し替えて報酬を処理するが、あれは
/// TestMode のときだけ使われる（ゲーム本体では画面が出る）。実機では画面そのものを
/// 捕まえて、ボタンと同じ処理（RewardsSetSynchronizer.SelectLocalReward と
/// 画面の「進む」ボタン）を呼ぶ。
///
/// カード報酬は選牌画面（NCardRewardSelectionScreen）の操作が要るので、いまは候補に
/// 出さずに飛ばす。
/// </summary>
internal static class LiveRewards
{
    public static RewardsSet? Set { get; private set; }
    public static NRewardsScreen? Screen { get; private set; }
    public static bool IsTerminal { get; private set; }

    public static void Capture(RewardsSet set, NRewardsScreen screen, bool isTerminal)
    {
        Set = set;
        Screen = screen;
        IsTerminal = isTerminal;
        Bridge.LogLine($"reward screen: {set.Rewards.Count} rewards (terminal={isTerminal})");
    }

    static readonly HashSet<object> _logged = new();

    public static void Forget()
    {
        Set = null;
        Screen = null;
    }

    public static bool Active =>
        Set != null && Screen != null && Godot.GodotObject.IsInstanceValid(Screen);

    /// <summary>報酬画面の候補（取れる報酬と「進む」）。</summary>
    public static Decision? Build(Player me)
    {
        if (!Active || Set is null) return null;
        var set = Set!;
        var d = new Decision(DecisionKind.Rewards) { Context = IsTerminal ? "战斗奖励" : "额外奖励" };
        foreach (var r in set.Rewards)
        {
            if (r.SuccessfullySelected || r is CardReward) continue;   // カード報酬は飛ばす
            var reward = r;
            d.Add(ActionKind.TakeReward, "领 " + Label(reward), () => Take(reward), a: reward);
        }
        // 「進む」は、取り残しが無いときだけ候補にする。まだ取れる報酬があるうちは
        // 画面のボタンが無効なことがあり、押しても閉じない（同じ手の繰り返しになる）。
        bool nothingLeft = set.Rewards.All(r => r.SuccessfullySelected);
        if (!_logged.Contains(set))
        {
            _logged.Add(set);
            Bridge.LogLine("reward candidates: " + string.Join(", ", set.Rewards.Select(r => (r.SuccessfullySelected ? "*" : "") + Label(r))));
        }
        if (nothingLeft) d.Add(ActionKind.Leave, "离开", Leave);
        return d;
    }

    static void Take(Reward r)
    {
        try
        {
            _ = RunManager.Instance.RewardsSetSynchronizer.SelectLocalReward(r);
            Bridge.LogLine("took reward: " + Label(r));
        }
        catch (Exception e)
        {
            Bridge.LogLine("take reward failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    /// <summary>画面の「進む / スキップ」ボタンを、押したときと同じ信号で押す。</summary>
    static void Leave()
    {
        try
        {
            var screen = Screen;
            if (screen == null || !Godot.GodotObject.IsInstanceValid(screen)) return;
            var button = screen.GetNodeOrNull<NProceedButton>("ProceedButton");
            if (button == null) { Bridge.LogLine("proceed button not found"); return; }
            button.EmitSignal(NClickableControl.SignalName.Released, button);
            Bridge.LogLine("pressed proceed");
            Forget();                     // 閉じた前提で手放す（多重押しを避ける）
        }
        catch (Exception e)
        {
            Bridge.LogLine("proceed failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    static string Label(Reward r) => r switch
    {
        GoldReward => "金币",
        PotionReward p => "药水 " + p.Potion?.Id.Entry,
        RelicReward rr => "遗物 " + rr.Relic?.Id.Entry,
        CardReward => "卡牌奖励",
        _ => r.GetType().Name,
    };
}

/// <summary>報酬画面が出た瞬間を捕まえる（画面と RewardsSet を覚えておく）。</summary>
[HarmonyPatch(typeof(NRewardsScreen), nameof(NRewardsScreen.ShowScreen))]
internal static class RewardScreenPatch
{
    [HarmonyPostfix]
    static void Postfix(RewardsSet set, bool isTerminal, NRewardsScreen __result)
    {
        try
        {
            LiveRewards.Capture(set, __result, isTerminal);
        }
        catch (Exception e)
        {
            Bridge.LogLine("reward capture failed: " + e.GetType().Name);
        }
    }
}
