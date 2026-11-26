using System.Reflection;
using MegaCrit.Sts2.Core.Entities.Rngs;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Odds;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// 局外分枝的“公平重设”（2026-11-03，决策记录 258）：在任一决策点把玩家看不到的未来整个换掉，眼前看得到的一切不变。
/// 用于分枝打局——同一局面下各选项接着打，比较结果——避免分枝偷看这一局真实的未来（真实的卡牌奖励、遭遇、事件结果、后几幕地图）。
///
/// 游戏在开局那一刻就用 UpFront 随机数把三幕的普通战、精英、Boss、先古之民与事件顺序抽好存进各幕的 RoomSet；
/// 卡牌奖励、商店、变牌用玩家的随机数（PlayerRngSet）；事件、地图以及各卡牌遗物自带的随机数都由整局种子（RunRngSet.Seed）派生。
/// 这里依次换掉：
///   1. 整局种子（事件、地图、内容自带的随机数随之改变）；
///   2. 12 种整局随机数流（含问号房、宝箱遗物），问号房概率对象里持有的那份一并换；
///   3. 玩家的奖励、商店、变牌随机数；
///   4. 遗物袋（玩家的与共享的）里剩下的顺序；
///   5. 当前这一幕还没走到的事件、普通战、精英的顺序（Boss、先古之民玩家已看到，不动）；
///   6. 后面几幕的房间整幕重新生成（同 RunManager.GenerateRooms：事件、普通战、精英、Boss、先古之民；最后一幕原有第二个 Boss 的另抽一个）；
///   7. 正在进行的事件自己的随机数；战斗中另重洗抽牌堆（同 RunSession.Reseed）。
/// 不换：当前这一幕的地图、正在显示的卡牌奖励与商店货架、事件已显示的选项——都是玩家看得到的。
/// 同一个 salt 在同一局面下结果完全确定（分枝之间可以用共同随机数配对比较）。
/// </summary>
public static class Branch
{
    const BindingFlags Any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

    static ulong Mix(uint salt, ulong k) => unchecked((k + 1) * 0x9E3779B97F4A7C15UL ^ (salt + 0x632BE59BD9B4E019UL) * 0xBF58476D1CE4E5B9UL);

    static void SetBacking(object obj, string prop, object value)
    {
        var t = obj.GetType();
        FieldInfo? f = null;
        for (var c = t; c != null && f == null; c = c.BaseType) f = c.GetField($"<{prop}>k__BackingField", Any);
        if (f == null) throw new MissingFieldException(t.FullName, prop);
        f.SetValue(obj, value);
    }

    static FieldInfo Field(Type t, string name)
    {
        for (var c = t; c != null; c = c.BaseType)
            if (c.GetField(name, Any) is { } f) return f;
        throw new MissingFieldException(t.FullName, name);
    }

    /// <summary>list[from..] 原地打乱（Fisher–Yates，用游戏自己的 Rng）。</summary>
    static void ShuffleTail<T>(List<T> list, int from, Rng rng)
    {
        for (int i = list.Count - 1; i > from; i--)
        {
            int j = from + rng.NextInt(i - from + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    /// <summary>
    /// 遭遇列表 list[from..] 按游戏自己的规则重新抽（同 ActModel.GenerateRooms 与 AddWithoutRepeatingTags）：每个位置从它该用的池子里
    /// 用抽取袋抽（袋空了重装），尽量不与上一场同标签、不连着同一场。普通战前 NumberOfWeakEncounters 个位置是弱怪池、其余是普通池，
    /// 不能整段打乱——2026-11-03 第一版整段打乱，强怪排到了第一幕开头，分枝局通关率从 68% 掉到 49.5%。
    /// </summary>
    static void RegenTail(List<EncounterModel> list, int from, Func<int, IEnumerable<EncounterModel>> poolFor, Rng rng)
    {
        int n = list.Count;
        var keep = list.Take(from).ToList();
        list.Clear(); list.AddRange(keep);
        GrabBag<EncounterModel>? bag = null; IEnumerable<EncounterModel>? cur = null;
        for (int i = from; i < n; i++)
        {
            var pool = poolFor(i);
            if (!ReferenceEquals(pool, cur))
            {
                // 换到一个池子时，先扣掉这个池子本轮已经抽出的（游戏的抽取袋不放回，空了才重装）：
                // 前面用同一个池子的位置共 k 个、池子 m 个，本轮已抽的就是其中最后 k % m 个
                bag = new GrabBag<EncounterModel>(); cur = pool;
                var all = pool.ToList();
                var used = Enumerable.Range(0, from).Where(j => ReferenceEquals(poolFor(j), pool)).Select(j => keep[j]).ToList();
                var drawn = all.Count > 0 ? used.Skip(used.Count - used.Count % all.Count).ToList() : new List<EncounterModel>();
                foreach (var e in all) if (!drawn.Remove(e)) bag.Add(e, 1.0);
            }
            if (!bag!.Any()) foreach (var e in pool) bag.Add(e, 1.0);
            var last = list.LastOrDefault();
            var pick = bag.GrabAndRemove(rng, e => !e.SharesTagsWith(last) && e != last) ?? bag.GrabAndRemove(rng);
            if (pick != null) list.Add(pick);
        }
    }

    static void ShuffleBag(RelicGrabBag bag, Rng rng)
    {
        var deques = (System.Collections.IDictionary)Field(typeof(RelicGrabBag), "_deques").GetValue(bag)!;
        foreach (var v in deques.Values)
            if (v is List<RelicModel> l) ShuffleTail(l, 0, rng);
    }

    public static Dictionary<string, object> Apply(uint salt)
    {
        var s = RunSession.State;
        var info = new Dictionary<string, object>();
        // 1. 整局种子
        SetBacking(s.Rng, "Seed", unchecked(s.Rng.Seed ^ Mix(salt, 1)));
        // 2. 整局随机数流；问号房概率对象建立时就拿着那份 Rng 的引用，要一起换
        foreach (var t in Enum.GetValues<RunRngType>()) s.Rng.MockRng(t, Mix(salt, 100 + (ulong)t));
        Field(typeof(AbstractOdds), "_rng").SetValue(s.Odds.UnknownMapPoint, s.Rng.GetRng(RunRngType.UnknownMapPoint));
        // 3. 玩家随机数
        foreach (var p in s.Players)
        {
            var rngs = (System.Collections.IDictionary)Field(typeof(PlayerRngSet), "_rngs").GetValue(p.PlayerRng)!;
            foreach (var t in Enum.GetValues<PlayerRngType>()) rngs[t] = new Rng(Mix(salt, 200 + (ulong)t));
            // 4. 遗物袋
            ShuffleBag(p.RelicGrabBag, new Rng(Mix(salt, 300)));
        }
        ShuffleBag(s.SharedRelicGrabBag, new Rng(Mix(salt, 301)));
        // 5. 当前这一幕没走到的部分
        var roomsF = Field(typeof(ActModel), "_rooms");
        var cur = (RoomSet)roomsF.GetValue(s.Act)!;
        var r5 = new Rng(Mix(salt, 400));
        if (cur.eventsVisited < cur.events.Count) ShuffleTail(cur.events, cur.eventsVisited, r5);
        int weak = (int)typeof(ActModel).GetProperty("NumberOfWeakEncounters", Any)!.GetValue(s.Act)!;
        var act0 = s.Act;
        if (cur.normalEncountersVisited < cur.normalEncounters.Count)
            RegenTail(cur.normalEncounters, cur.normalEncountersVisited, i => i < weak ? act0.AllWeakEncounters : act0.AllRegularEncounters, r5);
        if (cur.eliteEncountersVisited < cur.eliteEncounters.Count)
            RegenTail(cur.eliteEncounters, cur.eliteEncountersVisited, _ => act0.AllEliteEncounters, r5);
        // 6. 后面几幕整幕重新生成
        int regen = 0;
        for (int i = s.CurrentActIndex + 1; i < s.Acts.Count; i++)
        {
            var act = s.Acts[i];
            var rs = (RoomSet)roomsF.GetValue(act)!;
            bool second = rs.HasSecondBoss;
            rs.events.Clear(); rs.normalEncounters.Clear(); rs.eliteEncounters.Clear();
            rs.eventsVisited = rs.normalEncountersVisited = rs.eliteEncountersVisited = rs.bossEncountersVisited = 0;
            var r6 = new Rng(Mix(salt, 500 + (ulong)i));
            act.GenerateRooms(r6, s.UnlockState, s.Players.Count > 1);
            if (second) act.SetSecondBossEncounter(r6.NextItem(act.AllBossEncounters.Where(e => e.Id != act.BossEncounter.Id)));
            regen++;
        }
        info["regenerated_acts"] = regen;
        // 7. 正在进行的事件；战斗中重洗抽牌堆
        if (MegaCrit.Sts2.Core.Runs.RunManager.Instance.EventSynchronizer?.GetLocalEvent() is { } ev && s.CurrentRoom is EventRoom)
        {
            SetBacking(ev, "Rng", new Rng(Mix(salt, 600)));
            info["event_rng"] = ev.Id.Entry;
        }
        var pile = s.Players[0].PlayerCombatState?.DrawPile;
        if (pile != null && pile.Cards.Count >= 2)
        {
            var cards = pile.Cards.ToList();
            new Rng(Mix(salt, 700)).Shuffle(cards);
            foreach (var c in cards) pile.RemoveInternal(c, silent: true);
            foreach (var c in cards) pile.AddInternal(c, -1, silent: true);
            info["draw_reshuffled"] = cards.Count;
        }
        return info;
    }
}
