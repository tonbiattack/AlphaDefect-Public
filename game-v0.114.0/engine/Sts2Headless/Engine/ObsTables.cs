using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// Rust 引擎的观测编码要查的静态表（obs-tables 命令，输出到 engine/rust/crates/sts2-serve/src/obs/tables.json）：
///   cards：每张卡未升级、升级一次后的 DynamicVars（按定义顺序的名字与基础值）、关键字（KeywordSources.Local）、费用、星费；
///   moves：每个怪物行动状态机里每个招式的意图类型列表（游戏一个招式可带多个意图，Rust 内核每招只记一个主意图）。
/// Rust 端的牌与怪物只有自己的数据结构，编码时按“名字 + 升级次数”“怪物 + 招式”到这里查。
/// </summary>
public static class ObsTables
{
    static readonly MethodInfo GenMoves = typeof(MonsterModel).GetMethod("GenerateMoveStateMachine", BindingFlags.NonPublic | BindingFlags.Instance)!;

    public static void Export(string path)
    {
        var task = Pump.Instance.Start(() => HeadlessRun.CreateRun("DEFECT", 10, "OBSTABLES"));
        if (task.IsFaulted) throw task.Exception!;
        var state = RunManager.Instance.DebugOnlyGetState()!;
        var me = state.Players[0];

        var cards = new Dictionary<string, object?>();
        foreach (var model in ModelDb.AllCards.OrderBy(m => m.Id.Entry, StringComparer.Ordinal))
        {
            var forms = new List<object?>();
            for (int up = 0; up <= 1; up++)
            {
                try
                {
                    var c = state.CreateCard(model, me);
                    if (up == 1)
                    {
                        if (!c.IsUpgradable) break;
                        c.UpgradeInternal(); c.FinalizeUpgradeInternal();
                    }
                    forms.Add(CardForm(c));
                }
                catch (Exception e) { forms.Add(new Dictionary<string, object?> { ["error"] = e.GetBaseException().Message }); break; }
            }
            cards[Vocab.Model(model)] = forms;
        }

        var moves = new Dictionary<string, object?>();
        var follows = new Dictionary<string, object?>();
        foreach (var m in ModelDb.Monsters.OrderBy(m => m.Id.Entry, StringComparer.Ordinal))
        {
            MonsterMoveStateMachine? sm = null;
            try { sm = (MonsterMoveStateMachine)GenMoves.Invoke(m, null)!; }
            catch
            {
                try { sm = (MonsterMoveStateMachine)GenMoves.Invoke(m.ToMutable(), null)!; }
                catch (Exception e) { Console.Error.WriteLine($"招式枚举失败 {Vocab.Model(m)}：{(e.InnerException ?? e).Message}"); }
            }
            if (sm == null) continue;
            var d = new Dictionary<string, object?>();
            var follow = new Dictionary<string, object?>();
            foreach (var (id, st) in sm.States)
                if (st is MoveState ms)
                {
                    d[id] = ms.Intents.Select(it => new object?[] { it.IntentType.ToString(), it is AttackIntent a ? Math.Max(1, a.Repeats) : 0 }).ToList();
                    // 后续状态的类型（招式、条件分支、随机分支）：Rust 内核有时把条件分支在建状态机时就解掉了，观测的“看条件”标记要按游戏的
                    string? next = null;
                    try { next = ms.GetNextState(null!, null!); } catch { }
                    next ??= ms.GetType().GetProperties().Where(pr => pr.PropertyType == typeof(string) && pr.Name.Contains("FollowUp")).Select(pr => pr.GetValue(ms) as string).FirstOrDefault(v => v != null);
                    if (next != null && sm.States.TryGetValue(next, out var ns)) follow[id] = ns.GetType().Name;
                }
            moves[m.Id.Entry] = d;
            follows[m.Id.Entry] = follow;
        }

        var doc = new Dictionary<string, object?> { ["cards"] = cards, ["moves"] = moves, ["follow"] = follows };
        File.WriteAllText(path, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = false }));
        Console.WriteLine($"卡牌 {cards.Count} 张，怪物 {moves.Count} 个，写入 {path}");
    }

    static Dictionary<string, object?> CardForm(CardModel c)
    {
        var kw = c.GetKeywordsWithSources(KeywordSources.Local);
        return new Dictionary<string, object?>
        {
            ["vars"] = c.DynamicVars.Select(kv => new object?[] { kv.Key, (double)kv.Value.BaseValue }).ToList(),
            ["keywords"] = kw.Select(k => k.ToString()).OrderBy(k => k, StringComparer.Ordinal).ToList(),
            ["cost"] = c.EnergyCost.CostsX ? 0 : c.EnergyCost.GetWithModifiers(CostModifiers.Local),
            ["x"] = c.EnergyCost.CostsX,
            ["star"] = c.CurrentStarCost,
            ["upgrades"] = c.CurrentUpgradeLevel,
            ["max_upgrades"] = c.MaxUpgradeLevel,
        };
    }
}
