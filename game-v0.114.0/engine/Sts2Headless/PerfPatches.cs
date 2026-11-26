using System.Reflection;
using System.Runtime.CompilerServices;
using HarmonyLib;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// 只影响速度、不改变结果的补丁。
/// CombatState.Creatures / PlayerCreatures / Players 每次读取都用 LINQ 拼一份新列表，而游戏遍历钩子时会反复读取，
/// 剖析显示占战斗模拟一半以上的时间。这里在生物列表（顺序和成员）不变时返回同一份快照，变化后重建。
/// 原实现返回的也是快照，调用方拿到后不会再随战斗状态变化，所以语义相同。
/// </summary>
public static class PerfPatches
{
    sealed class Cache
    {
        public Creature[] Allies = Array.Empty<Creature>(), Enemies = Array.Empty<Creature>();
        public IReadOnlyList<Creature>? Creatures, PlayerCreatures;
        public IReadOnlyList<Player>? Players;
    }

    static readonly ConditionalWeakTable<CombatState, Cache> Caches = new();
    static readonly AccessTools.FieldRef<CombatState, List<Creature>> AlliesRef = AccessTools.FieldRefAccess<CombatState, List<Creature>>("_allies");
    static readonly AccessTools.FieldRef<CombatState, List<Creature>> EnemiesRef = AccessTools.FieldRefAccess<CombatState, List<Creature>>("_enemies");

    public static void Install(Harmony h)
    {
        void P(string prop, string prefix) =>
            GameCompat.Patch(h, GameCompat.Getter(typeof(CombatState), prop, "战斗状态的生物列表提速"), $"CombatState.{prop}",
                prefix: GameCompat.Own(typeof(PerfPatches), prefix));
        P(nameof(CombatState.Creatures), nameof(CreaturesPrefix));
        P(nameof(CombatState.PlayerCreatures), nameof(PlayerCreaturesPrefix));
        P(nameof(CombatState.Players), nameof(PlayersPrefix));
        if (Environment.GetEnvironmentVariable("STS2_NO_PERF_HOOKS") == null) HookListenerPool.Install(h);
        SlugifyCache.Install(h);
    }

    static Cache Valid(CombatState cs)
    {
        var c = Caches.GetOrCreateValue(cs);
        var allies = AlliesRef(cs);
        var enemies = EnemiesRef(cs);
        if (c.Creatures == null || !Same(c.Allies, allies) || !Same(c.Enemies, enemies))
        {
            c.Allies = allies.ToArray();
            c.Enemies = enemies.ToArray();
            c.Creatures = c.Allies.Concat(c.Enemies).ToList();
            c.PlayerCreatures = c.Creatures.Where(x => x.IsPlayer).ToList();
            c.Players = c.PlayerCreatures.Select(x => x.Player!).ToList();
        }
        return c;
    }

    static bool Same(Creature[] snapshot, List<Creature> current)
    {
        if (snapshot.Length != current.Count) return false;
        for (int i = 0; i < snapshot.Length; i++)
            if (!ReferenceEquals(snapshot[i], current[i])) return false;
        return true;
    }

    static bool CreaturesPrefix(CombatState __instance, ref IReadOnlyList<Creature> __result) { __result = Valid(__instance).Creatures!; return false; }
    static bool PlayerCreaturesPrefix(CombatState __instance, ref IReadOnlyList<Creature> __result) { __result = Valid(__instance).PlayerCreatures!; return false; }
    static bool PlayersPrefix(CombatState __instance, ref IReadOnlyList<Player> __result) { __result = Valid(__instance).Players!; return false; }
}

/// <summary>
/// Hook-listener list pooling (2026-11-16, engine speed-up) is NOT included in the public release.
/// The working version restated two methods of the game's v0.113.1 code (CombatState.IterateHookListeners and
/// RunState.IterateHookListeners), written against the decompiled game, so that their lists could be pooled. That restated
/// game code is not ours to publish, so this stub keeps the original game methods: results are unchanged (the pooled version
/// was checked to give identical observations), only observation encoding is slower. See NOTICE.md.
/// </summary>
public static class HookListenerPool
{
    public static bool Enabled = false;
    public static bool FreezeEnabled = false;
    public static bool Installed => false;
    internal static void Install(Harmony h) { }
}

/// <summary>
/// StringHelper.Slugify 的结果缓存（2026-11-16，引擎提速）。ModelDb.Power&lt;T&gt;() / Card&lt;T&gt;() / Relic&lt;T&gt;() 每次都按类型名重算模型 ID
/// （ModelDb.GetId → GetCategory、GetEntry → Slugify：三遍正则替换），卡牌的能力数值预览（PowerVar.UpdateCardPreview）、施加能力等处处在调，
/// 剖析里观测编码有约 7% 花在这里。Slugify 只看输入的字符串（正则替换与 ToUpperInvariant，与文化、局面无关），同样的输入结果相同，
/// 所以按输入记下结果（线程静态，最多 4096 项；调用处传的都是类型名、枚举名这类有限的字符串）。唯一的差别是同样的输入返回同一个字符串实例，
/// 游戏按值比较字符串，不受影响。SlugifyCache.Enabled 运行时开关（bench-encode --ab slug 用）。
/// </summary>
public static class SlugifyCache
{
    [ThreadStatic] static Dictionary<string, string>? _cache;
    public static bool Enabled = true;

    internal static void Install(Harmony h)
    {
        var m = GameCompat.Method(typeof(MegaCrit.Sts2.Core.Helpers.StringHelper), nameof(MegaCrit.Sts2.Core.Helpers.StringHelper.Slugify), "模型 ID 的字符串转换缓存（提速）", new[] { typeof(string) });
        GameCompat.Patch(h, m, "StringHelper.Slugify", prefix: GameCompat.Own(typeof(SlugifyCache), nameof(Pre)), postfix: GameCompat.Own(typeof(SlugifyCache), nameof(Post)));
    }

    static bool Pre(string txt, ref string __result, out bool __state)
    {
        __state = false;
        if (!Enabled || txt == null) return true;
        if ((_cache ??= new Dictionary<string, string>()).TryGetValue(txt, out var r)) { __result = r; __state = true; return false; }
        return true;
    }

    static void Post(string txt, string __result, bool __state)
    {
        if (__state || !Enabled || txt == null || __result == null) return;
        var c = _cache ??= new Dictionary<string, string>();
        if (c.Count < 4096) c[txt] = __result;
    }
}
