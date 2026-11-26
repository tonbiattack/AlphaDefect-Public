using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;

namespace Sts2Headless;

/// <summary>不运行构造函数的界面节点对象：没有对应的 Godot 原生对象，只用来承接被置空或被接管的方法调用。</summary>
public static class FakeUi
{
    public static T Create<T>()
    {
        var o = (T)RuntimeHelpers.GetUninitializedObject(typeof(T));
        GC.SuppressFinalize(o!);
        return o;
    }
}

/// <summary>卡牌奖励界面上的一次选择：选一张牌，或选一个替代项（跳过、重抽等）。</summary>
public sealed class PendingCardReward(IReadOnlyList<CardCreationResult> cards, IReadOnlyList<CardRewardAlternative> alternatives)
{
    public IReadOnlyList<CardCreationResult> Cards { get; } = cards;
    public IReadOnlyList<CardRewardAlternative> Alternatives { get; } = alternatives;
    internal readonly TaskCompletionSource<int?> Tcs = new();
    /// <summary>index &lt; Cards.Count 选牌，否则选第 index - Cards.Count 个替代项。</summary>
    public void Resolve(int index) => Tcs.SetResult(index);
}

/// <summary>三选一卡包。</summary>
public sealed class PendingBundle(IReadOnlyList<IReadOnlyList<CardModel>> bundles)
{
    public IReadOnlyList<IReadOnlyList<CardModel>> Bundles { get; } = bundles;
    internal readonly TaskCompletionSource<IEnumerable<IReadOnlyList<CardModel>>> Tcs = new();
    public void Resolve(int index) => Tcs.SetResult(new[] { Bundles[index] });
}

/// <summary>
/// 卡牌奖励和卡包在正式游戏里走异步界面（await screen.OptionSelected()），测试模式下则是同步接口或固定选第一个。
/// 这里提供“假界面对象”：不运行构造函数（避开 Godot 原生调用），只接管逻辑层会调用的几个方法，
/// 让游戏按正式流程 await 我们的挂起任务。重抽、可多选等原有行为因此全部保留。
/// </summary>
public static class FakeScreens
{
    public static PendingCardReward? CardReward { get; private set; }
    /// <summary>进行中的 Crystal Sphere 小游戏（NCrystalSphereScreen 在正式游戏里驱动它）。</summary>
    public static MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? CrystalSphere { get; private set; }
    public static PendingBundle? Bundle { get; private set; }

    static readonly NCardRewardSelectionScreen RewardScreen = Uninitialized<NCardRewardSelectionScreen>();
    static readonly NChooseABundleSelectionScreen BundleScreen = Uninitialized<NChooseABundleSelectionScreen>();
    static IReadOnlyList<CardCreationResult> _rewardCards = Array.Empty<CardCreationResult>();
    static IReadOnlyList<CardRewardAlternative> _rewardAlternatives = Array.Empty<CardRewardAlternative>();
    static IReadOnlyList<IReadOnlyList<CardModel>> _bundles = Array.Empty<IReadOnlyList<CardModel>>();

    static T Uninitialized<T>() => FakeUi.Create<T>();

    /// <summary>开新局时清掉上一局遗留的挂起状态。</summary>
    public static void Reset() { CardReward = null; Bundle = null; CrystalSphere = null; }

    public static void Install(Harmony h)
    {
        Patch(h, typeof(NCardRewardSelectionScreen), nameof(NCardRewardSelectionScreen.ShowScreen), nameof(ShowRewardScreen));
        Patch(h, typeof(NCardRewardSelectionScreen), nameof(NCardRewardSelectionScreen.RefreshOptions), nameof(RefreshRewardOptions));
        Patch(h, typeof(NCardRewardSelectionScreen), nameof(NCardRewardSelectionScreen.OptionSelected), nameof(RewardOptionSelected));
        Patch(h, typeof(NCardRewardSelectionScreen), nameof(NCardRewardSelectionScreen.GetCardHolder), nameof(NoCardHolder));
        Patch(h, typeof(NChooseABundleSelectionScreen), nameof(NChooseABundleSelectionScreen.ShowScreen), nameof(ShowBundleScreen));
        Patch(h, typeof(NChooseABundleSelectionScreen), nameof(NChooseABundleSelectionScreen.CardsSelected), nameof(BundleSelected));
        Patch(h, typeof(MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen), "ShowScreen", nameof(ShowCrystalSphere));
        // 核对用：STS2_FORCE_CRYSTAL_SPHERE=1 时，问号房间抽到的事件一律换成 Crystal Sphere（训练和评测不设）
        if (Environment.GetEnvironmentVariable("STS2_FORCE_CRYSTAL_SPHERE") == "1")
            GameCompat.Patch(h, GameCompat.Method(typeof(MegaCrit.Sts2.Core.Models.ActModel), "PullNextEvent", "核对用：强制占卜事件"), "ActModel.PullNextEvent",
                postfix: GameCompat.Own(typeof(FakeScreens), nameof(ForceCrystalSphere)));
    }

    static void ForceCrystalSphere(ref MegaCrit.Sts2.Core.Models.EventModel __result) =>
        __result = MegaCrit.Sts2.Core.Models.ModelDb.Event<MegaCrit.Sts2.Core.Models.Events.CrystalSphere>();

    static void Patch(Harmony h, Type t, string name, string prefix) =>
        GameCompat.Patch(h, GameCompat.Method(t, name, "无头的选牌、卡包、占卜界面"), $"{t.Name}.{name}", prefix: GameCompat.Own(typeof(FakeScreens), prefix));

    static bool ShowRewardScreen(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> extraOptions, ref NCardRewardSelectionScreen? __result)
    {
        _rewardCards = options;
        _rewardAlternatives = extraOptions;
        __result = RewardScreen;
        return false;
    }

    static bool RefreshRewardOptions(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> extraOptions)
    {
        _rewardCards = options;
        _rewardAlternatives = extraOptions;
        return false;
    }

    static bool RewardOptionSelected(ref Task<int?> __result)
    {
        var p = new PendingCardReward(_rewardCards.ToList(), _rewardAlternatives.ToList());
        CardReward = p;
        __result = Clear(p.Tcs.Task, () => CardReward = null);
        return false;
    }

    static bool NoCardHolder(ref NCardHolder? __result) { __result = null; return false; }

    static bool ShowBundleScreen(IReadOnlyList<IReadOnlyList<CardModel>> bundles, ref NChooseABundleSelectionScreen __result)
    {
        _bundles = bundles;
        __result = BundleScreen;
        return false;
    }

    static bool BundleSelected(ref Task<IEnumerable<IReadOnlyList<CardModel>>> __result)
    {
        var p = new PendingBundle(_bundles);
        Bundle = p;
        __result = Clear(p.Tcs.Task, () => Bundle = null);
        return false;
    }

    static readonly MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen CrystalScreen = Uninitialized<MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen>();

    static bool ShowCrystalSphere(MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame grid,
        ref MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere.NCrystalSphereScreen __result)
    {
        CrystalSphere = grid;
        grid.Finished += () => CrystalSphere = null;
        __result = CrystalScreen;
        return false;
    }

    static async Task<T> Clear<T>(Task<T> t, Action clear)
    {
        try { return await t; }
        finally { clear(); }
    }
}
