using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2Headless;

/// <summary>一次“从若干张牌里选 min..max 张”的待决选择。</summary>
/// <summary>
/// 界面上看得到的选牌用途：提示语（CardSelectorPrefs.Prompt 的表名与键）、发起的选牌命令（CardSelectCmd 的方法名，决定从哪里选）、
/// 附魔选牌要附的附魔。选牌器接口本身不带这些，由 HeadlessFixes 在选牌命令上取。
/// </summary>
public sealed record SelectInfo(string? PromptTable, string? PromptKey, string Method, MegaCrit.Sts2.Core.Models.EnchantmentModel? Enchantment,
    MegaCrit.Sts2.Core.Models.AbstractModel? Source = null);

public sealed class PendingCardChoice(IReadOnlyList<CardModel> options, int min, int max, bool cancelable, SelectInfo? info = null)
{
    public SelectInfo? Info { get; } = info;
    public IReadOnlyList<CardModel> Options { get; } = options;
    public int Min { get; } = min;
    public int Max { get; } = max;
    /// <summary>界面上有关闭按钮（CardSelectorPrefs.Cancelable）：可以一张不选、返回空集。</summary>
    public bool Cancelable { get; } = cancelable;
    internal readonly TaskCompletionSource<IEnumerable<CardModel>> Tcs = new();

    public void Resolve(IEnumerable<int> indices) => Tcs.SetResult(indices.Select(i => Options[i]).ToList());
}

/// <summary>
/// 接管游戏所有选牌界面：游戏 await 这里返回的任务，任务挂起即形成一个决策点，由驱动方 Resolve 后继续。
/// </summary>
public sealed class AgentCardSelector : ICardSelector
{
    public PendingCardChoice? Pending { get; private set; }

    /// <summary>下一次选择的最少张数覆盖（FromChooseACardScreen 的 canSkip），用后即清。</summary>
    public static int? NextMinSelect { get; set; }

    /// <summary>下一次选择能否取消（选牌命令的 CardSelectorPrefs.Cancelable），用后即清。选牌器接口本身不带这个信息。</summary>
    public static bool NextCancelable { get; set; }

    /// <summary>下一次选择的用途（见 SelectInfo），用后即清。</summary>
    public static SelectInfo? NextInfo { get; set; }

    /// <summary>正在使用的药水（从开始使用到效果结算完；药水发起的选牌没有来源参数，用它当选牌说明的编号）。</summary>
    public static MegaCrit.Sts2.Core.Models.PotionModel? UsingPotion { get; set; }

    /// <summary>卡牌奖励是同步接口，M0 先用这个回调直接决定；M1 改为挂起式。</summary>
    public Func<IReadOnlyList<CardCreationResult>, IReadOnlyList<CardRewardAlternative>, CardRewardSelection>? OnCardReward;

    public async Task<IEnumerable<CardModel>> GetSelectedCards(IEnumerable<CardModel> options, int minSelect, int maxSelect)
    {
        var list = options.ToList();
        if (NextMinSelect is { } min) { minSelect = Math.Min(min, list.Count); NextMinSelect = null; }
        var p = new PendingCardChoice(list, minSelect, maxSelect, NextCancelable, NextInfo);
        NextCancelable = false; NextInfo = null;
        Pending = p;
        try { return await p.Tcs.Task; }
        finally { Pending = null; }
    }

    public CardRewardSelection GetSelectedCardReward(IReadOnlyList<CardCreationResult> options, IReadOnlyList<CardRewardAlternative> alternatives) =>
        OnCardReward?.Invoke(options, alternatives) ?? default;
}
