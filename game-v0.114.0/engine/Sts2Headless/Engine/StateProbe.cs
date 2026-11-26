using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;

namespace Sts2Headless;

/// <summary>
/// 对拍用的规范化战斗状态（{"op":"state"}，Rust引擎方案.md 的 M0）：只读，不走一步、不改局面。
/// 输出玩家（血量、格挡、能量、能力、充能球、药水、遗物）、四个牌堆的有序牌（抽牌堆按真实顺序，0 为顶部）、
/// 敌人（血量、格挡、能力、当前招式与意图），以及当前决策的候选（规范形式，Python 对拍脚本与 Rust 端的候选对应）。
/// 名字一律用游戏的条目 ID（如 STRIKE_DEFECT），由对拍脚本统一规范化。
/// </summary>
public static class StateProbe
{
    public static byte[] Json(RunSession run, Decision? d)
    {
        var s = RunSession.State;
        var me = s.Players[0];
        var o = new Dictionary<string, object?>
        {
            ["decision"] = d?.Kind.ToString(),
            ["over"] = d?.Kind == DecisionKind.GameOver,
            ["won"] = run.IsVictory,   // 战斗模式为赢下这场战斗，整局模式为通关
        };
        bool inCombat = CombatManager.Instance.IsInProgress && me.PlayerCombatState != null;
        o["in_combat"] = inCombat;
        // 对拍用：各随机数流已抽的次数（定位 Rust 哪一步多抽或少抽）
        o["rng_counters"] = Enum.GetValues<MegaCrit.Sts2.Core.Entities.Rngs.RunRngType>().ToDictionary(t => t.ToString(), t => (object?)s.Rng.GetRng(t).ToSerializable().counter);
        if (inCombat)
        {
            var cs = me.Creature.CombatState!;
            var pcs = me.PlayerCombatState!;
            o["round"] = (cs as CombatState)?.RoundNumber;
            o["player_turn"] = pcs.TurnNumber;   // 对拍用：玩家回合号（Ambergris 等额外回合会加）
            o["player"] = new Dictionary<string, object?>
            {
                ["hp"] = me.Creature.CurrentHp, ["max_hp"] = me.Creature.MaxHp, ["block"] = me.Creature.Block,
                ["energy"] = pcs.Energy, ["max_energy"] = pcs.MaxEnergy, ["stars"] = pcs.Stars,
                ["powers"] = Powers(me.Creature),
                ["orb_slots"] = pcs.OrbQueue.Capacity,
                ["orbs"] = pcs.OrbQueue.Orbs.Select(b => new object[] { b.Id.Entry, (int)b.PassiveVal, (int)b.EvokeVal }).ToList(),
                ["potions"] = me.PotionSlots.Select(p => p?.Id.Entry).ToList(),
                ["relics"] = me.Relics.Select(r => r.Id.Entry).ToList(),
                ["gold"] = me.Gold,
            };
            o["hand"] = pcs.Hand.Cards.Select(c => Card(c, true)).ToList();
            o["draw"] = pcs.DrawPile.Cards.Select(c => Card(c, false)).ToList();
            o["discard"] = pcs.DiscardPile.Cards.Select(c => Card(c, false)).ToList();
            o["exhaust"] = pcs.ExhaustPile.Cards.Select(c => Card(c, false)).ToList();
            // 格式 22 的核对用：玩家的宠物（亡灵契约师的奥斯提，活着、死了都在战斗里）。没有宠物时不出这一项，别的角色的输出与以前相同
            var pets = cs.Creatures.Where(c => !c.IsPlayer && c.Side == me.Creature.Side).ToList();
            if (pets.Count > 0)
                o["pets"] = pets.Select(c => new Dictionary<string, object?> { ["id"] = c.Monster?.Id.Entry, ["hp"] = c.CurrentHp, ["max_hp"] = c.MaxHp, ["block"] = c.Block, ["alive"] = c.IsAlive }).ToList();
            var enemies = Enemies(cs, me.Creature);
            o["enemies"] = enemies.Select(c => Enemy(c, cs)).ToList();
            if (d != null) o["choices"] = d.Choices.Select(ch => Choice(ch, pcs, enemies, me, run.Selecting?.Options)).ToList();
        }
        else
        {
            // 战斗外（整局对拍）：整局概况与候选
            o["run"] = new Dictionary<string, object?>
            {
                ["act"] = s.CurrentActIndex, ["floor"] = s.TotalFloor, ["room"] = s.CurrentRoom?.RoomType.ToString(),
                ["coord"] = s.CurrentMapCoord is { } mc ? new[] { mc.row, mc.col } : null,
                ["hp"] = me.Creature.CurrentHp, ["max_hp"] = me.Creature.MaxHp, ["gold"] = me.Gold,
                ["deck"] = me.Deck.Cards.Select(c => Card(c, false)).ToList(),
                ["relics"] = me.Relics.Select(r => r.Id.Entry).ToList(),
                ["potions"] = me.PotionSlots.Select(p => p?.Id.Entry).ToList(),
                ["potion_slots"] = me.PotionSlots.Count,
            };
            if (d != null) o["choices"] = d.Choices.Select(ch => RunChoice(ch, me, d.Kind == DecisionKind.CardSelect ? run.Selecting?.Options : null)).ToList();
        }
        return JsonSerializer.SerializeToUtf8Bytes(o);
    }

    /// <summary>战斗外候选的规范形式（整局对拍）：["map", 行, 列]、["event", 文本键]、["reward", 类别, 内容]、["leave"]、
    /// ["potion", 药水 ID, null]、["discard_potion", 药水 ID]、["card_reward", 卡牌 ID, 升级]、["card_alt", 键]、["bundle", 序号]、
    /// ["rest", 选项 ID]、["buy", 类别, 内容 ID, 价格]、["relic", 遗物 ID]、["skip_relic"]、["cell", x, y, 工具]，选牌同战斗内。</summary>
    static object?[] RunChoice(Choice ch, MegaCrit.Sts2.Core.Entities.Players.Player me, IReadOnlyList<CardModel>? options)
    {
        string? Id(object? x) => x is AbstractModel m ? m.Id.Entry : null;
        int Up(object? x) => x is CardModel c ? c.CurrentUpgradeLevel : 0;
        return ch.Kind switch
        {
            ActionKind.MapMove when ch.A is MegaCrit.Sts2.Core.Map.MapPoint p => new object?[] { "map", p.coord.row, p.coord.col },
            ActionKind.EventOption => new object?[] { "event", ch.Label },
            ActionKind.TakeReward => ch.A switch
            {
                MegaCrit.Sts2.Core.Rewards.GoldReward g => new object?[] { "reward", "GOLD", g.Amount },
                MegaCrit.Sts2.Core.Rewards.PotionReward pr => new object?[] { "reward", "POTION", pr.Potion?.Id.Entry },
                MegaCrit.Sts2.Core.Rewards.RelicReward rr => new object?[] { "reward", "RELIC", rr.Relic?.Id.Entry },
                MegaCrit.Sts2.Core.Rewards.CardReward cr => new object?[] { "reward", "CARDS", cr.Cards.Select(c => c.Id.Entry + (c.CurrentUpgradeLevel > 0 ? "+" : "")).ToList() },
                var r => new object?[] { "reward", r?.GetType().Name, null },
            },
            ActionKind.Leave when options != null => new object?[] { "cancel" },
            ActionKind.Leave => new object?[] { "leave" },
            ActionKind.UsePotion => new object?[] { "potion", Id(ch.A), null },
            ActionKind.DiscardPotion => new object?[] { "discard_potion", Id(ch.A) },
            ActionKind.PickCardReward => new object?[] { "card_reward", Id(ch.A), Up(ch.A) },
            ActionKind.CardRewardAlt => new object?[] { "card_alt", ch.Key },
            ActionKind.PickBundle => new object?[] { "bundle", (int)ch.F0 },
            ActionKind.RestOption => new object?[] { "rest", ch.Label },
            ActionKind.Buy => ch.A switch
            {
                CardModel c => new object?[] { "buy", "CARD", c.Id.Entry + (c.CurrentUpgradeLevel > 0 ? "+" : ""), (int)ch.F0 },
                RelicModel r => new object?[] { "buy", "RELIC", r.Id.Entry, (int)ch.F0 },
                PotionModel pm => new object?[] { "buy", "POTION", pm.Id.Entry, (int)ch.F0 },
                _ => new object?[] { "buy", "REMOVAL", null, (int)ch.F0 },
            },
            ActionKind.PickRelic => new object?[] { "relic", Id(ch.A) },
            ActionKind.SkipRelic => new object?[] { "skip_relic" },
            ActionKind.MinigameCell => new object?[] { "cell", (int)ch.F0, (int)ch.F1, (int)ch.F2 },
            ActionKind.SelectCard when ch.A is CardModel c => new object?[] { "select", c.Id.Entry, c.CurrentUpgradeLevel, Nth(c, options) },
            ActionKind.ConfirmSelection => new object?[] { "confirm" },
            _ => new object?[] { ch.Kind.ToString() },
        };
    }

    static int Nth(CardModel c, IReadOnlyList<CardModel>? options)
    {
        if (options == null) return 0;
        int k = 0;
        foreach (var x in options)
        {
            if (ReferenceEquals(x, c)) return k;
            if (x.Id == c.Id && x.CurrentUpgradeLevel == c.CurrentUpgradeLevel) k++;
        }
        return k;
    }

    /// <summary>
    /// 整局的随机数对拍（{"op":"runinfo"}）：各幕的房间清单（RoomSet：普通战、精英、事件、Boss、第二个 Boss、远古事件，
    /// 均为开局时用 UpFront 随机数预抽好的）、当前幕地图（每个节点的行、列、类型、子节点列号），以及玩家的血量、金币、牌组、遗物、药水。只读。
    /// </summary>
    public static byte[] RunInfo()
    {
        var s = RunSession.State;
        var me = s.Players[0];
        var roomsField = HarmonyLib.AccessTools.Field(typeof(ActModel), "_rooms");
        var acts = new List<object>();
        foreach (var act in s.Acts)
        {
            var rs = roomsField.GetValue(act) as MegaCrit.Sts2.Core.Rooms.RoomSet;
            acts.Add(new Dictionary<string, object?>
            {
                ["id"] = act.Id.Entry,
                ["boss"] = rs?.Boss?.Id.Entry, ["boss2"] = rs?.SecondBoss?.Id.Entry,
                ["ancient"] = rs != null && rs.HasAncient ? rs.Ancient.Id.Entry : null,
                ["normal"] = rs?.normalEncounters.Select(e => e.Id.Entry).ToList(),
                ["elite"] = rs?.eliteEncounters.Select(e => e.Id.Entry).ToList(),
                ["events"] = rs?.events.Select(e => e.Id.Entry).ToList(),
            });
        }
        var map = new List<object>();
        if (s.Map is { } m)
        {
            var pts = m.GetAllMapPoints().Append(m.StartingMapPoint).Append(m.BossMapPoint);
            if (m.SecondBossMapPoint != null) pts = pts.Append(m.SecondBossMapPoint);
            foreach (var p in pts.Distinct().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col))
                map.Add(new object[] { p.coord.row, p.coord.col, p.PointType.ToString(), p.Children.Select(c => c.coord.col).OrderBy(c => c).ToList() });
        }
        var o = new Dictionary<string, object?>
        {
            ["act"] = s.CurrentActIndex, ["floor"] = s.TotalFloor, ["acts"] = acts, ["map"] = map,
            ["hp"] = me.Creature.CurrentHp, ["max_hp"] = me.Creature.MaxHp, ["gold"] = me.Gold,
            ["deck"] = me.Deck.Cards.Select(c => c.Id.Entry + (c.CurrentUpgradeLevel > 0 ? "+" : "")).ToList(),
            ["relics"] = me.Relics.Select(r => r.Id.Entry).ToList(),
            ["potions"] = me.PotionSlots.Select(p => p?.Id.Entry).ToList(),
            // 各条随机流已读的次数（Rng._counter），与 Rust 的 runinfo 对照，找随机数用量的偏差
            ["rng"] = new Dictionary<string, int>
            {
                ["rewards"] = Ctr(me.PlayerRng.Rewards), ["shops"] = Ctr(me.PlayerRng.Shops), ["transformations"] = Ctr(me.PlayerRng.Transformations),
                ["up_front"] = Ctr(s.Rng.UpFront), ["shuffle"] = Ctr(s.Rng.Shuffle), ["unknown_map_point"] = Ctr(s.Rng.UnknownMapPoint),
                ["niche"] = Ctr(s.Rng.Niche), ["treasure_room"] = Ctr(s.Rng.TreasureRoomRelics),
                ["combat_card_generation"] = Ctr(s.Rng.CombatCardGeneration), ["combat_potion_generation"] = Ctr(s.Rng.CombatPotionGeneration),
            },
        };
        return JsonSerializer.SerializeToUtf8Bytes(o);
    }

    static readonly System.Reflection.FieldInfo RngCounter = HarmonyLib.AccessTools.Field(typeof(MegaCrit.Sts2.Core.Random.Rng), "_counter");
    static int Ctr(MegaCrit.Sts2.Core.Random.Rng r) => (int)RngCounter.GetValue(r)!;

    static List<Creature> Enemies(ICombatState cs, Creature player) => cs.Creatures.Where(c => c.Side != player.Side).ToList();

    static List<object[]> Powers(Creature c) => c.Powers.Select(p => new object[] { p.Id.Entry, p.Amount }).ToList();

    static Dictionary<string, object?> Card(CardModel c, bool inHand)
    {
        var d = new Dictionary<string, object?> { ["id"] = c.Id.Entry, ["up"] = c.CurrentUpgradeLevel };
        try { d["cost"] = c.EnergyCost.CostsX ? -1 : c.EnergyCost.GetWithModifiers(inHand ? CostModifiers.All : CostModifiers.Local); } catch { }
        if (c.Enchantment is { } e) d["ench"] = e.Id.Entry;
        // 对拍用：重放次数（Hidden Gem 等加的 BaseReplayCount），为 0 时不输出
        if (c.BaseReplayCount != 0) d["replay"] = c.BaseReplayCount;
        return d;
    }

    static Dictionary<string, object?> Enemy(Creature c, ICombatState cs)
    {
        var d = new Dictionary<string, object?>
        {
            ["id"] = c.Monster?.Id.Entry, ["hp"] = c.CurrentHp, ["max_hp"] = c.MaxHp, ["block"] = c.Block, ["alive"] = c.IsAlive,
            ["powers"] = Powers(c),
        };
        if (c.Monster is { } m && c.IsAlive && m.NextMove is { } mv)
        {
            d["move"] = mv.StateId;
            var intents = new List<object?[]>();
            foreach (AbstractIntent it in mv.Intents)
            {
                int dmg = 0, hits = 0;
                if (it is AttackIntent atk)
                    try { dmg = atk.GetSingleDamage(cs.PlayerCreatures, c); hits = Math.Max(1, atk.Repeats); } catch { }
                intents.Add(new object?[] { it.IntentType.ToString(), dmg, hits });
            }
            d["intents"] = intents;
        }
        return d;
    }

    /// <summary>候选的规范形式：["play", 手牌下标, 目标敌人下标或 null]、["potion", 栏位, 目标或 null]、["discard_potion", 栏位]、
    /// ["end_turn"]、["select", 卡牌 ID, 升级次数, 同名同升级的选项里排第几]、["confirm"]、["cancel"]，其余为 [动作类型名]。</summary>
    static object?[] Choice(Choice ch, MegaCrit.Sts2.Core.Entities.Players.PlayerCombatState pcs, List<Creature> enemies, MegaCrit.Sts2.Core.Entities.Players.Player me,
        IReadOnlyList<CardModel>? options)
    {
        int? Target(object? b) => b is Creature t && enemies.IndexOf(t) is var i and >= 0 ? i : null;
        int Nth(CardModel c)
        {
            if (options == null) return 0;
            int k = 0;
            foreach (var x in options)
            {
                if (ReferenceEquals(x, c)) return k;
                if (x.Id == c.Id && x.CurrentUpgradeLevel == c.CurrentUpgradeLevel) k++;
            }
            return k;
        }
        return ch.Kind switch
        {
            ActionKind.PlayCard when ch.A is CardModel c => new object?[] { "play", pcs.Hand.Cards.ToList().IndexOf(c), Target(ch.B) },
            ActionKind.UsePotion => new object?[] { "potion", me.PotionSlots.ToList().IndexOf(ch.A as PotionModel), Target(ch.B) },
            ActionKind.DiscardPotion => new object?[] { "discard_potion", me.PotionSlots.ToList().IndexOf(ch.A as PotionModel) },
            ActionKind.EndTurn => new object?[] { "end_turn" },
            ActionKind.SelectCard when ch.A is CardModel c => new object?[] { "select", c.Id.Entry, c.CurrentUpgradeLevel, Nth(c) },
            ActionKind.ConfirmSelection => new object?[] { "confirm" },
            ActionKind.Leave => new object?[] { "cancel" },
            _ => new object?[] { ch.Kind.ToString() },
        };
    }
}
