using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>奖励界面上的一次选择：领某个奖励，或离开（proceed）。</summary>
public sealed class PendingRewards(RewardsSet set, IReadOnlySet<Reward> attempted)
{
    public RewardsSet Set { get; } = set;
    /// <summary>
    /// 当前可领的奖励（已领、已放弃过的卡牌奖励、药水栏满时的药水不在其中）。每次按当前状态计算：
    /// 界面挂起期间可以丢药水腾出空位，之后药水奖励就能领了（NRewardButton 一直可点）。
    /// </summary>
    public List<Reward> Available => Set.Rewards
        .Where(r => !r.SuccessfullySelected && !attempted.Contains(r))
        .Where(r => r is not PotionReward || Set.Player.HasOpenPotionSlots)
        .ToList();
    /// <summary>战斗房间结束时的奖励是“终结”的：离开后进入地图或下一幕；其他（事件、遗物给的）离开后回到原处。</summary>
    public bool IsTerminal => Set.Room is CombatRoom;
    internal readonly TaskCompletionSource<Reward?> Tcs = new();
    public void Take(Reward r) => Tcs.SetResult(r);
    public void Proceed() => Tcs.SetResult(null);
}

/// <summary>
/// 接管 RewardsSet.Offer 的测试分支（RewardsSet.testSelector）：每次挂起一个 PendingRewards，
/// 按选择逐个调用 RewardsSetSynchronizer.SelectLocalReward，与界面上点奖励按钮等价。
/// 离开时的处理照搬 NRewardsScreen.OnProceedButtonPressed。
/// </summary>
public static class RewardsFlow
{
    public static PendingRewards? Pending { get; private set; }
    /// <summary>
    /// 正在领的那一组奖励与正在打开的那一个（版本 18，决策记录 210）：打开卡牌奖励时，其余没领的奖励（别的卡牌奖励的牌、药水、遗物）
    /// 照样在观测里。游戏在发出这组奖励时就生成好了每个卡牌奖励的牌（RewardsSet 发奖前对每个奖励 Populate），
    /// 打开一个卡牌奖励不选也能关掉、再开别的，界面上全都看得到，不涉及随机数。
    /// </summary>
    public static RewardsSet? ActiveSet { get; private set; }
    public static Reward? ActiveChoice { get; private set; }
    public static IReadOnlySet<Reward>? ActiveAttempted { get; private set; }

    /// <summary>这一组里还没领、也还能领的奖励（不含 except）：已领的、打开过又放弃的卡牌奖励不算。</summary>
    public static IEnumerable<Reward> Unclaimed(RewardsSet set, IReadOnlySet<Reward>? attempted, Reward? except) =>
        set.Rewards.Where(r => !r.SuccessfullySelected && r != except && (attempted == null || !attempted.Contains(r)));
    /// <summary>金币奖励是否自动领（决策记录 68 起默认自动）。整局 reset 带 "auto_gold":false 时改为作为“领奖励”候选交给策略（决策记录 160）。</summary>
    public static bool AutoGold = true;
    /// <summary>
    /// 无负面效果的遗物是否自动拿（决策记录 162 起默认自动）：奖励界面里不在“多选一”联动组中的、宝箱里只有一件的，
    /// 名单里的遗物（engine/catalog/no_downside_relics.txt）由引擎直接拿，不作为决策交给策略。reset 带 "auto_relics":false 时关闭。
    /// </summary>
    public static bool AutoRelics = true;
    static HashSet<string>? _noDownside;

    /// <summary>无负面遗物名单（遗物 ID，不带 RELIC. 前缀）；读不到时为空（等于不自动拿），并在标准错误里说一声。</summary>
    public static HashSet<string> NoDownside
    {
        get
        {
            if (_noDownside != null) return _noDownside;
            var path = AppContext.GetData("Sts2NoDownsideRelics") as string ?? "no_downside_relics.txt";
            try
            {
                _noDownside = File.ReadAllLines(path).Select(l => l.Trim()).Where(l => l.Length > 0 && !l.StartsWith('#')).ToHashSet();
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"[auto-relics] 读不到无负面遗物名单 {path}：{e.Message}，不自动拿遗物");
                _noDownside = new HashSet<string>();
            }
            return _noDownside;
        }
    }

    public static bool IsNoDownside(MegaCrit.Sts2.Core.Models.RelicModel? r) => r != null && NoDownside.Contains(r.Id.Entry);
    /// <summary>上一个终结奖励界面离开后要执行的“继续”动作，由 RunSession 在泵上执行。</summary>
    public static AbstractRoom? TerminalProceedRoom { get; set; }

    public static void Install() => RewardsSet.testSelector = Select;

    /// <summary>开新局时清掉上一局遗留的挂起状态（旧局里挂起的任务不再继续）。</summary>
    public static void Reset() { Pending = null; TerminalProceedRoom = null; ActiveSet = null; ActiveChoice = null; ActiveAttempted = null; }

    static async Task Select(RewardsSet set)
    {
        set.ThrowInTestIfRewardsNotTaken = false;
        var sync = RunManager.Instance.RewardsSetSynchronizer;
        var attempted = new HashSet<Reward>();
        while (true)
        {
            bool terminal = set.Room is CombatRoom;
            if (!terminal && sync.IsRewardsSetCompleted(set)) return;   // 非终结界面领完自动关闭
            // 金币奖励自动领取，不作为决策交给策略（2026-10-25 起）：领金币没有坏处，可策略几乎从不领（评测每局不到 0.05 次），
            // 局外探索又很少，学不到它的用处；没钱就买不了东西、删不了卡
            if (AutoGold && PendingGold(set, attempted) is { } gold)
            {
                attempted.Add(gold);
                await sync.SelectLocalReward(gold);
                continue;
            }
            // 无负面遗物自动拿（决策记录 162）：拿了不会比不拿差；“多选一”联动组里的不算（拿哪件仍是决策）
            if (AutoRelics && PendingFreeRelic(set, attempted) is { } relic)
            {
                attempted.Add(relic);
                await sync.SelectLocalReward(relic);
                continue;
            }
            var p = new PendingRewards(set, attempted);
            Pending = p;
            Reward? choice;
            try { choice = await p.Tcs.Task; }
            finally { Pending = null; }
            if (choice == null)
            {
                if (terminal) TerminalProceedRoom = set.Room;
                else if (!sync.IsRewardsSetCompleted(set)) sync.SkipLocalRewardsSet();
                return;
            }
            attempted.Add(choice);
            ActiveSet = set; ActiveChoice = choice; ActiveAttempted = attempted;
            try { await sync.SelectLocalReward(choice); }
            finally { ActiveSet = null; ActiveChoice = null; ActiveAttempted = null; }
            // 领到一半放弃的卡牌奖励不再提供，避免反复打开、关闭形成死循环；界面上它仍可再点，这是唯一的偏差
        }
    }

    static Reward? PendingFreeRelic(RewardsSet set, HashSet<Reward> attempted) =>
        set.Rewards.FirstOrDefault(r => r is RelicReward rr && rr.ParentRewardSet == null && IsNoDownside(rr.Relic) && !r.SuccessfullySelected && !attempted.Contains(r));

    static Reward? PendingGold(RewardsSet set, HashSet<Reward> attempted) =>
        set.Rewards.FirstOrDefault(r => r is GoldReward && !r.SuccessfullySelected && !attempted.Contains(r));
}
