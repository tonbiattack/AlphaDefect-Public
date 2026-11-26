using System.Reflection;
using Godot;
using HarmonyLib;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Models.Potions;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// 原版由界面节点完成、无头下缺失的游戏逻辑（2026-10-25 引擎审查发现）。每一处都写明对应的界面代码。
/// </summary>
public static class HeadlessFixes
{
    public static void Install(Harmony h)
    {
        // 可取消的选牌：选牌器接口只有 min/max，把 CardSelectorPrefs.Cancelable 带过去（NDeckCardSelectScreen 等的关闭按钮）。
        // 选牌器在命令的第一个 await 之前同步调用，所以前缀设、终结器清是准确的；之后才调用选牌器的命令退回到不可取消。
        foreach (var m in typeof(CardSelectCmd).GetMethods(BindingFlags.Public | BindingFlags.Static)
                     .Where(m => !m.ContainsGenericParameters && m.GetParameters().Any(p => p.ParameterType == typeof(CardSelectorPrefs))))
            h.Patch(m, prefix: Hook(nameof(SetCancelable)), finalizer: Hook(nameof(ClearCancelable)));
        // FromChooseACardScreen 的前缀设了 NextMinSelect，但候选为空、超过 3 张或选牌器不是我们的（如 WhisperingEarring 自动出牌）时不会被用掉，
        // 留到下一次选牌就会改错最少张数
        GameCompat.Patch(h, GameCompat.Method(typeof(CardSelectCmd), nameof(CardSelectCmd.FromChooseACardScreen), "三选一结束后清掉最少张数"), "CardSelectCmd.FromChooseACardScreen", finalizer: Hook(nameof(ClearMinSelect)));
        // 正在使用的药水：开始时记下，效果结算完（返回的任务结束）清掉；药水发起的选牌（发现类药水等）用它当来源
        GameCompat.Patch(h, GameCompat.Method(typeof(MegaCrit.Sts2.Core.Models.PotionModel), nameof(MegaCrit.Sts2.Core.Models.PotionModel.OnUseWrapper), "用药水的前后"), "PotionModel.OnUseWrapper",
            prefix: Hook(nameof(PotionStart)), postfix: Hook(nameof(PotionWrap)));
        // v0.111.0 时删牌成功后由界面标记删牌位已用（NMerchantCardRemoval → OnCardRemovalUsed），无头下补了一个后置补丁；
        // v0.113.0 起 MerchantCardRemovalEntry.OnTryPurchaseInternal 成功后自己调用 SetUsed()（LordsParasol 进店自动删牌同样），补丁去掉
        // Foul Potion 只能扔向商人：目标取自商店或假商人界面上的商人按钮，无头下界面为空，返回假按钮
        GameCompat.Patch(h, GameCompat.Method(typeof(FoulPotion), nameof(FoulPotion.GetFoulPotionMerchantTarget), "污秽药水在假商店里的目标"), "FoulPotion.GetFoulPotionMerchantTarget", prefix: Hook(nameof(FoulTarget)));
        // 扔向假商人：原版要求事件节点是 NFakeMerchant 才继续，节点之外的逻辑（开战、奖励）照原样执行
        GameCompat.Patch(h, GameCompat.Method(typeof(FoulPotion), "OnUse", "污秽药水在假商店里"), "FoulPotion.OnUse", prefix: Hook(nameof(FoulAtFakeMerchant)));
        // 毛皮大衣（FurCoat）开战前、敌人入场时按当前地图节点的坐标判断要不要把敌人血量砍到 1。只打一场战斗的练习没有走地图，
        // 当前节点为空，原版直接空引用，回合循环开战前就死掉，战斗卡在“不在出牌阶段”（租用机日志累计约 2.8 万次，2026-10-26 查到）。
        // 没有当前节点时这场战斗不是它标记的节点，按不触发处理
        foreach (var name in new[] { nameof(FurCoat.BeforeCombatStart), nameof(FurCoat.AfterCreatureAddedToCombat) })
            GameCompat.Patch(h, GameCompat.Method(typeof(FurCoat), name, "毛皮大衣要地图节点"), "FurCoat." + name, prefix: Hook(nameof(FurCoatNeedsMapPoint)));
    }

    static bool FurCoatNeedsMapPoint(FurCoat __instance, ref Task __result)
    {
        if (__instance.Owner?.RunState?.CurrentMapPoint != null) return true;
        __result = Task.CompletedTask;
        return false;
    }

    static HarmonyMethod Hook(string name) => GameCompat.Own(typeof(HeadlessFixes), name);

    static void SetCancelable(object[] __args, MethodBase __originalMethod)
    {
        var p = __args.FirstOrDefault(a => a is CardSelectorPrefs) is CardSelectorPrefs pr ? pr : default;
        AgentCardSelector.NextCancelable = p.Cancelable;
        // 界面上的提示语与选牌来源（NCardGridSelectionScreen 等显示 prefs.Prompt）；附魔选牌另外把要附的附魔显示在预览上
        // 来源：命令的 source 参数（从手牌选牌时卡牌、能力传自己）
        var ps = __originalMethod.GetParameters();
        int si = Array.FindIndex(ps, q => q.Name == "source");
        AgentCardSelector.NextInfo = new SelectInfo(p.Prompt?.LocTable, p.Prompt?.LocEntryKey, __originalMethod.Name,
            __args.OfType<MegaCrit.Sts2.Core.Models.EnchantmentModel>().FirstOrDefault(),
            si >= 0 && si < __args.Length ? __args[si] as MegaCrit.Sts2.Core.Models.AbstractModel : null);
    }

    static void PotionStart(MegaCrit.Sts2.Core.Models.PotionModel __instance) => AgentCardSelector.UsingPotion = __instance;

    static void PotionWrap(MegaCrit.Sts2.Core.Models.PotionModel __instance, ref Task __result) => __result = PotionDone(__instance, __result);

    static async Task PotionDone(MegaCrit.Sts2.Core.Models.PotionModel p, Task t)
    {
        try { await t; }
        finally { if (AgentCardSelector.UsingPotion == p) AgentCardSelector.UsingPotion = null; }
    }

    static Exception? ClearCancelable(Exception? __exception) { AgentCardSelector.NextCancelable = false; AgentCardSelector.NextInfo = null; return __exception; }

    static Exception? ClearMinSelect(Exception? __exception) { AgentCardSelector.NextMinSelect = null; AgentCardSelector.NextInfo = null; return __exception; }

    static readonly NMerchantButton FakeButton = FakeUi.Create<NMerchantButton>();
    static readonly Control FakeContext = FakeUi.Create<Control>();

    /// <summary>当前房间是商店，或还没开战的假商人事件。</summary>
    public static bool MerchantPresent(AbstractRoom? room) =>
        room?.RoomType == RoomType.Shop ||
        (room is EventRoom { CanonicalEvent: FakeMerchant } && RunManager.Instance.EventSynchronizer.GetLocalEvent() is FakeMerchant { StartedFight: false });

    static bool FoulTarget(AbstractRoom room, ref ValueTuple<NMerchantButton?, Control?> __result)
    {
        __result = MerchantPresent(room) ? (FakeButton, FakeContext) : (null, null);
        return false;
    }

    static bool FoulAtFakeMerchant(FoulPotion __instance, ref Task __result)
    {
        if (CombatManager.Instance.IsInProgress || __instance.Owner.RunState.CurrentRoom is not EventRoom { CanonicalEvent: FakeMerchant }) return true;
        __result = ThrowAtFakeMerchant(__instance);
        return false;
    }

    static async Task ThrowAtFakeMerchant(FoulPotion potion)
    {
        foreach (var player in potion.Owner.RunState.Players)
            await ((FakeMerchant)RunManager.Instance.EventSynchronizer.GetEventForPlayer(player)).FoulPotionThrown(potion);
    }
}
