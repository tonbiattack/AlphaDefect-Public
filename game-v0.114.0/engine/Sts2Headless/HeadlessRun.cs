using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Unlocks;

namespace Sts2Headless;

public static class HeadlessRun
{
    /// <summary>
    /// 游戏模式（reset 带 "game_mode"，每次 reset 重设）：标准模式，或恐惧模式（GameMode.Dread）。
    /// 从恐惧线 d36b03f 只搬了建局这一处（2026-11-20，铁甲战士决策记录第 16 条，给恐惧模式测试用）：统治、晋升者之灾、Boss 加强都由游戏的 DreadRunModel 完成；
    /// 恐惧线的统治观测（Obs/Dread.cs）、预演补项与战斗练习的 DreadSetup 没有搬，本线模型没见过它们。进阶固定为 10（同游戏 StartRunLobby）。
    /// </summary>
    public static GameMode Mode = GameMode.Standard;

    /// <summary>
    /// 训练加血（reset 带 "bonus_hp"，每次 reset 重设，不带为 0）：建局时最大生命与当前生命各加这么多（2026-11-20，铁甲战士决策记录第 24 条）。
    /// 加在初始遗物结算之后、Launch 之前；整局与战斗练习（CombatSetup 也经 CreateRun）都生效，战斗练习随后按 hp_fraction × 最大生命设血；
    /// 带玩家快照的练习与开幕快照开局由快照覆盖（SyncWithSerializedPlayer 连最大生命一起还原，快照取自加过血的训练局）。
    /// 评测、实机、搜索不带这一项，与原来完全相同。
    /// </summary>
    public static int BonusHp;

    /// <summary>建立一局（到 Launch 为止，不进入第一幕）：全解锁、无修改器，模式按 Mode（默认标准模式）。</summary>
    public static async Task CreateRun(string characterId, int ascension, string seed)
    {
        if (Mode == GameMode.Dread) ascension = 10;
        CharacterModel character = ModelDb.AllCharacters.First(c => c.Id.Entry == characterId);
        // 同游戏 StartRunLobby.BeginRunLocally：64 位哈希（XxHash64）原样传给 Rng(ulong, string)。
        // 原来先转 (uint) 截掉了高 32 位，等于换了个种子，第一幕约一半与实机不同（2026-10-31 实机对照，决策记录 227）
        var actRng = new Rng(StringHelper.GetDeterministicHashCode(seed), "act_selection");
        var acts = ActModel.GetRandomList(actRng, UnlockState.all, isMultiplayer: false).ToList();
        var player = Player.CreateForNewRun(character, UnlockState.all, 1uL);
        var state = RunState.CreateForNewRun(new[] { player }, acts.Select(a => a.ToMutable()).ToList(),
            Array.Empty<ModifierModel>(), Mode, isCustom: false, ascension, seed);   // v0.113.0 加了 isCustom（选角色界面的普通局传 false）
        RunManager.Instance.SetUpNewSingleplayer(state, shouldSave: false);
        await RunManager.Instance.FinalizeStartingRelics();
        if (BonusHp != 0)
        {
            var c = player.Creature;
            c.SetMaxHpInternal(c.MaxHp + BonusHp);
            c.SetCurrentHpInternal(c.CurrentHp + BonusHp);
        }
        RunManager.Instance.Launch();
    }

    /// <summary>按游戏的单人开局流程开一局：建立后进入第一幕（Neow）。</summary>
    public static async Task StartNew(string characterId, int ascension, string seed, Dictionary<string, double>? timing = null)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        await CreateRun(characterId, ascension, seed);
        if (timing != null) { timing["建立"] = timing.GetValueOrDefault("建立") + sw.Elapsed.TotalMilliseconds; sw.Restart(); }
        await RunManager.Instance.EnterAct(0, doTransition: false);
        if (timing != null) timing["进入第一幕"] = timing.GetValueOrDefault("进入第一幕") + sw.Elapsed.TotalMilliseconds;
    }

    public static void Dump()
    {
        var s = RunManager.Instance.DebugOnlyGetState();
        if (s == null) { Console.WriteLine("no run"); return; }
        var p = s.Players[0];
        Console.WriteLine($"acts={string.Join(",", s.Acts.Select(a => a.Id.Entry))} asc={s.AscensionLevel} room={s.CurrentRoom?.GetType().Name}");
        Console.WriteLine($"hp={p.Creature.CurrentHp}/{p.Creature.MaxHp} gold={p.Gold}");
        Console.WriteLine($"deck={string.Join(",", p.Deck.Cards.Select(c => c.Id.Entry))}");
        Console.WriteLine($"relics={string.Join(",", p.Relics.Select(r => r.Id.Entry))}");
    }
}
