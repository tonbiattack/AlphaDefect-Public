using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace Sts2Headless;

/// <summary>
/// 整局里的玩家快照与战斗信息（reset 带 "snapshots":true 时附在观测帧末尾，见 Observation.Extra）。
/// - 选路线时附玩家快照：游戏自己的存档格式 SerializablePlayer（牌组含升级与附魔、遗物含计数、药水、血量、金币）。
///   训练端存下来，战斗课程用 CombatSpec.PlayerSnapshot 还原，在真实牌组上练 Boss 和第二幕。
/// - 每场战斗开始时附房间类型、遭遇和牌组规模，评测用来统计死因。
/// </summary>
public static class Snapshot
{
    public static string? Extra(Decision d, ref bool prevInCombat)
    {
        var s = RunSession.State;
        var me = s.Players[0];
        bool inCombat = CombatManager.Instance.IsInProgress && me.PlayerCombatState != null;
        bool started = inCombat && !prevInCombat;
        prevInCombat = inCombat;
        // 换幕时导出的开幕快照（ActStart）：带在换幕后的第一帧（通常就是新一幕的选路线）
        if (ActStart.Pending is { } ast)
        {
            ActStart.Pending = null;
            var dict = new Dictionary<string, object> { ["act_start"] = JsonSerializer.Deserialize<JsonElement>(ast) };
            if (d.Kind == DecisionKind.Map) dict["snapshot"] = Player(me);
            return JsonSerializer.Serialize(dict);
        }
        if (d.Kind == DecisionKind.Map)
            return JsonSerializer.Serialize(new Dictionary<string, object> { ["snapshot"] = Player(me) });
        // 商店：整个货架（含买不起、药水栏满不能买、已售出的），只供回放显示，不进模型的观测（2026-10-26 用户要求回放能看全货架）
        if (d.Kind == DecisionKind.Shop && s.CurrentRoom is MerchantRoom mr)
            return JsonSerializer.Serialize(new Dictionary<string, object> { ["shop"] = Shelf(MegaCrit.Sts2.Core.Runs.RunManager.Instance.MerchantSynchronizer.GetLocalInventory(), me) });
        if (started && s.CurrentRoom is CombatRoom room)
            return JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["combat"] = new Dictionary<string, object>
                {
                    ["room"] = room.RoomType.ToString(), ["encounter"] = room.Encounter.Id.Entry, ["act"] = s.CurrentActIndex, ["floor"] = s.TotalFloor,
                    ["hp"] = me.Creature.CurrentHp, ["max_hp"] = me.Creature.MaxHp, ["deck"] = me.Deck.Cards.Count, ["relics"] = me.Relics.Count,
                    ["potions"] = me.PotionSlots.Count(p => p != null),
                    // 标定用（2026-10-26）：牌组里升级过的牌数
                    ["upgrades"] = me.Deck.Cards.Count(c => c.CurrentUpgradeLevel > 0),
                },
            });
        return null;
    }

    /// <summary>货架上的每件商品：类别、词表键、价格、有货、买得起、牌的升级次数；删牌服务另记是否已用。</summary>
    static Dictionary<string, object> Shelf(MerchantInventory inv, Player me)
    {
        var items = new List<object>();
        foreach (var e in inv.AllEntries)
        {
            (string k, string? id, int up) = e switch
            {
                // 卖掉以后条目里的牌、遗物、药水会变成空：照样列出，标“已售出”
                MerchantCardEntry c => ("CARD", c.CreationResult?.Card is { } cd ? "CARD." + cd.Id.Entry : null, c.CreationResult?.Card?.CurrentUpgradeLevel ?? 0),
                MerchantRelicEntry r => ("RELIC", r.Model == null ? null : "RELIC." + r.Model.Id.Entry, 0),
                MerchantPotionEntry p => ("POTION", p.Model == null ? null : "POTION." + p.Model.Id.Entry, 0),
                MerchantCardRemovalEntry => ("REMOVAL", null, 0),
                _ => (e.GetType().Name, null, 0),
            };
            var it = new Dictionary<string, object?> { ["k"] = k, ["id"] = id, ["cost"] = e.Cost, ["stock"] = e.IsStocked, ["gold"] = e.EnoughGold };
            if (up > 0) it["up"] = up;
            if (e is MerchantCardRemovalEntry rm) it["used"] = rm.Used;
            items.Add(it);
        }
        return new Dictionary<string, object> { ["items"] = items, ["gold"] = me.Gold, ["slots_full"] = !me.HasOpenPotionSlots };
    }

    /// <summary>玩家快照（JSON 文本）。随机数、遗物袋、解锁和图鉴不带：还原时用新开局自己的，也省体积。</summary>
    public static string Player(Player me)
    {
        var sp = me.ToSerializable();
        sp.Rng = null!; sp.Odds = null!; sp.RelicGrabBag = null!; sp.UnlockState = null!;
        sp.DiscoveredCards = new(); sp.DiscoveredEnemies = new(); sp.DiscoveredEpochs = new(); sp.DiscoveredPotions = new(); sp.DiscoveredRelics = new();
        return JsonSerializer.Serialize(sp, JsonSerializationUtility.Options);
    }

    /// <summary>
    /// 旧版本存下的快照里游戏改了名的模型 ID（v0.114.0：CARD.RAPID_SHOT → CARD.TWIN_SHOT），按游戏自己存档迁移用的改名表换掉
    /// （SharedMigrationHelper，同 SerializableRunV21ToV22）。不换时 SyncWithSerializedPlayer 经 SaveUtil.CardOrDeprecated 把找不到的牌
    /// 静默换成 DEPRECATED_CARD（0 费状态牌：抽 1、消耗、从牌组删掉）。
    /// </summary>
    internal static readonly IReadOnlyDictionary<string, string>[] Renames =
    {
        MegaCrit.Sts2.Core.Saves.Migrations.Shared.SharedMigrationHelper.v100Renames,
        MegaCrit.Sts2.Core.Saves.Migrations.Shared.SharedMigrationHelper.v110Renames,
        MegaCrit.Sts2.Core.Saves.Migrations.Shared.SharedMigrationHelper.v114Renames,
    };

    internal static string MigrateIds(string json)
    {
        var node = System.Text.Json.Nodes.JsonNode.Parse(json);
        foreach (var r in Renames) MegaCrit.Sts2.Core.Saves.Migrations.Shared.SharedMigrationHelper.ReplaceModelIds(node, r);
        return node?.ToJsonString() ?? json;
    }

    /// <summary>牌组规格里的卡牌 ID（不带 CARD. 前缀）按同一张改名表换成新名。</summary>
    internal static string RenamedCard(string id)
    {
        foreach (var r in Renames) if (r.TryGetValue("CARD." + id, out var n)) id = n["CARD.".Length..];
        return id;
    }

    /// <summary>
    /// 把快照还原到新开局的玩家身上：走多人联机同步玩家的原版流程（静默加入卡牌、遗物、药水，不触发获得时效果，
    /// 所以遗物加牌、加最大生命之类不会重复）。随机数和遗物袋保留新开局自己的，同一份快照每次练习的抽牌都不同。
    /// </summary>
    public static void Restore(Player me, string json)
    {
        var sp = JsonSerializer.Deserialize<SerializablePlayer>(MigrateIds(json), JsonSerializationUtility.Options) ?? throw new ArgumentException("玩家快照为空");
        sp.NetId = me.NetId;
        sp.Rng = me.PlayerRng.ToSerializable();
        sp.Odds = me.PlayerOdds.ToSerializable();
        sp.RelicGrabBag = me.RelicGrabBag.ToSerializable();
        me.SyncWithSerializedPlayer(sp);
    }
}
