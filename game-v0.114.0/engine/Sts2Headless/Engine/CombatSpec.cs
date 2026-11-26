using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// 直接构造一场战斗的开局（战斗课程用）：不生成地图、不走 Neow，按给定牌组、遗物、药水、血量进入指定遭遇。
/// 牌组、遗物等为 null 时用角色初始配置。遭遇可以是具体 ID（如 ENCOUNTER.XXX），也可以是“weak / regular / elite / boss”，
/// 此时从所在幕的遭遇池按种子随机抽取。
/// </summary>
public sealed class CombatSpec
{
    public string Character { get; set; } = "DEFECT";
    public int Ascension { get; set; } = 10;
    public string Seed { get; set; } = "COMBAT";
    public int Act { get; set; }
    public string Encounter { get; set; } = "regular";
    /// <summary>牌组：卡牌 ID（不带 CARD. 前缀，如 STRIKE_IRONCLAD）及升级次数。</summary>
    public List<CardSpec>? Deck { get; set; }
    public List<string>? Relics { get; set; }
    public List<string>? Potions { get; set; }
    public int? Hp { get; set; }
    public int? MaxHp { get; set; }
    public int? Gold { get; set; }
    /// <summary>开场血量占最大生命的比例（与 Hp 同时给出时以 Hp 为准）。</summary>
    public double? HpFraction { get; set; }
    /// <summary>整局里的玩家快照（Snapshot.Player 的 JSON 文本）：给出时牌组、遗物、药水、血量、金币都取自快照，忽略上面几项。</summary>
    public string? PlayerSnapshot { get; set; }
    /// <summary>连打（2026-10-29，决策记录 187）：这一场打赢后接着打的遭遇（具体 ID），带着打完时的血量、药水与牌组直接开打，
    /// 同 A10 最后一幕连打两个 Boss（两场之间游戏里没有任何事件）。为空时只打一场。</summary>
    public string? NextEncounter { get; set; }
    /// <summary>第二个 Boss 单打（版本 15）：本幕已经打赢的第一个 Boss（具体 ID），只用于观测头部（与整局对齐），不影响战斗本身。</summary>
    public string? PrevEncounter { get; set; }
}

public sealed class CardSpec
{
    public string Id { get; set; } = "";
    public int Upgrades { get; set; }
}

public static class CombatSetup
{
    public static async Task Start(CombatSpec spec)
    {
        await HeadlessRun.CreateRun(spec.Character, spec.Ascension, spec.Seed);
        var state = RunManager.Instance.DebugOnlyGetState()!;
        var me = state.Players[0];
        state.CurrentActIndex = spec.Act;

        if (spec.PlayerSnapshot != null)
            Snapshot.Restore(me, spec.PlayerSnapshot);
        else if (spec.Deck != null)
        {
            me.Deck.Clear(silent: true);
            foreach (var cs in spec.Deck)
            {
                var card = state.CreateCard(ModelDb.GetById<CardModel>(new ModelId("CARD", Snapshot.RenamedCard(cs.Id))), me);
                // 超过卡牌升级上限的部分忽略（如诅咒不能升级）
                for (int i = 0; i < Math.Min(cs.Upgrades, card.MaxUpgradeLevel); i++) { card.UpgradeInternal(); card.FinalizeUpgradeInternal(); }
                me.Deck.AddInternal(card, silent: true);
            }
        }
        // 遗物按正式的获得流程（会触发获得时效果，如加最大生命）
        foreach (string r in spec.PlayerSnapshot == null ? spec.Relics ?? new List<string>() : new List<string>())
            await RelicCmd.Obtain(ModelDb.GetById<RelicModel>(new ModelId("RELIC", r)).ToMutable(), me);
        foreach (string p in spec.PlayerSnapshot == null ? spec.Potions ?? new List<string>() : new List<string>())
            me.AddPotionInternal(ModelDb.GetById<PotionModel>(new ModelId("POTION", p)).ToMutable(), silent: true);
        if (spec.MaxHp is { } mhp) me.Creature.SetMaxHpInternal(mhp);
        if (spec.Hp is { } hp) me.Creature.SetCurrentHpInternal(Math.Min(hp, me.Creature.MaxHp));
        else if (spec.HpFraction is { } frac) me.Creature.SetCurrentHpInternal(Math.Max(1, (int)Math.Round(me.Creature.MaxHp * frac)));
        if (spec.Gold is { } gold) me.Gold = gold;

        var encounter = PickEncounter(state, spec);
        await RunManager.Instance.EnterRoomDebug(encounter.RoomType, model: encounter.ToMutable(), showTransition: false);
    }

    static EncounterModel PickEncounter(RunState state, CombatSpec spec)
    {
        var act = state.Acts[spec.Act];
        var rng = new Rng((uint)MegaCrit.Sts2.Core.Helpers.StringHelper.GetDeterministicHashCode(spec.Seed), "combat_spec_encounter");
        IEnumerable<EncounterModel> pool = spec.Encounter.ToLowerInvariant() switch
        {
            "weak" => act.AllWeakEncounters,
            "regular" => act.AllRegularEncounters,
            "elite" => act.AllEliteEncounters,
            "boss" => act.AllBossEncounters,
            _ => new[] { ModelDb.GetById<EncounterModel>(new ModelId("ENCOUNTER", spec.Encounter.Replace("ENCOUNTER.", ""))) },
        };
        var list = pool.ToList();
        if (list.Count == 0) throw new ArgumentException($"第 {spec.Act + 1} 幕没有 {spec.Encounter} 类遭遇");
        return rng.NextItem(list)!;
    }
}
