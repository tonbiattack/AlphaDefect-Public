using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// sts2-sim 观测移植要查的静态表（sim-tables 命令，engine/Sts2SimServe/方案.md）：sts2-sim 没有游戏的 DynamicVars，
/// 观测里按名字取的数值（卡牌的伤害、格挡与其他数值，遗物、能力、药水、附魔的数值）从这里查名字与基础值。
///   cards：未升级、升级一次后的 DynamicVars（按定义顺序的名字与基础值）、关键字（KeywordSources.Local）；
///   relics、powers、potions、enchantments：DynamicVars（同上）；powers 另记是否减益，potions 另记使用场合与目标类型。
/// </summary>
public static class SimTables
{
    public static void Export(string path)
    {
        var task = Pump.Instance.Start(() => HeadlessRun.CreateRun("DEFECT", 10, "SIMTABLES"));
        if (task.IsFaulted) throw task.Exception!;
        var state = RunManager.Instance.DebugOnlyGetState()!;
        var me = state.Players[0];

        static List<object?[]> Vars(Func<IEnumerable<KeyValuePair<string, MegaCrit.Sts2.Core.Localization.DynamicVars.DynamicVar>>> get)
        {
            try { return get().Select(kv => new object?[] { kv.Key, (double)kv.Value.BaseValue }).ToList(); }
            catch { return new(); }
        }

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
                    forms.Add(new Dictionary<string, object?>
                    {
                        ["vars"] = Vars(() => c.DynamicVars),
                        ["keywords"] = c.GetKeywordsWithSources(KeywordSources.Local).Select(k => k.ToString()).OrderBy(k => k, StringComparer.Ordinal).ToList(),
                        ["upgrades"] = c.CurrentUpgradeLevel,
                    });
                }
                catch (Exception e) { forms.Add(new Dictionary<string, object?> { ["error"] = e.GetBaseException().Message }); break; }
            }
            cards[Vocab.Model(model)] = forms;
        }
        var relics = ModelDb.AllRelics.OrderBy(m => m.Id.Entry, StringComparer.Ordinal)
            .ToDictionary(m => Vocab.Model(m), m => (object?)Vars(() => m.ToMutable().DynamicVars));
        var powers = ModelDb.AllPowers.OrderBy(m => m.Id.Entry, StringComparer.Ordinal)
            .ToDictionary(m => Vocab.Model(m), m => (object?)new Dictionary<string, object?> { ["vars"] = Vars(() => m.ToMutable().DynamicVars), ["debuff"] = m.Type == PowerType.Debuff });
        var potions = ModelDb.AllPotions.OrderBy(m => m.Id.Entry, StringComparer.Ordinal)
            .ToDictionary(m => Vocab.Model(m), m => (object?)new Dictionary<string, object?> { ["vars"] = Vars(() => m.ToMutable().DynamicVars), ["usage"] = m.Usage.ToString(), ["target"] = m.TargetType.ToString() });
        var enchantments = ModelDb.DebugEnchantments.OrderBy(m => m.Id.Entry, StringComparer.Ordinal)
            .ToDictionary(m => Vocab.Model(m), m => (object?)Vars(() => m.ToMutable().DynamicVars));

        var doc = new Dictionary<string, object?> { ["cards"] = cards, ["relics"] = relics, ["powers"] = powers, ["potions"] = potions, ["enchantments"] = enchantments };
        File.WriteAllText(path, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = false }));
        Console.WriteLine($"卡牌 {cards.Count}、遗物 {relics.Count}、能力 {powers.Count}、药水 {potions.Count}、附魔 {enchantments.Count}，写入 {path}");
    }
}
