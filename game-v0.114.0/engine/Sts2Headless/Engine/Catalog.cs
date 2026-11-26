using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Rewards;
using HarmonyLib;

namespace Sts2Headless;

/// <summary>
/// 内容目录：生成词表（只追加），并导出每个实体的静态属性和“效果签名”。
/// 效果签名来自代码分析：该类重写了哪些游戏钩子、调用了哪些命令、施加/引用了哪些能力等模型。
/// 不依赖文本，新内容上线即可得到大致的“它做什么”。
/// </summary>
public static class Catalog
{
    /// <summary>把当前游戏版本的全部键追加进词表，返回新增数量。</summary>
    public static int ExtendVocab()
    {
        int before = Vocab.Count;
        foreach (string k in AllKeys().OrderBy(k => k, StringComparer.Ordinal)) Vocab.Add(k);
        foreach (string k in Vocab.Missing.Keys.OrderBy(k => k, StringComparer.Ordinal)) Vocab.Add(k);
        return Vocab.Count - before;
    }

    static IEnumerable<AbstractModel> AllModels()
    {
        var field = GameCompat.Field(typeof(ModelDb), "_contentById", "内容目录与词表：遍历全部模型")!;
        return ((System.Collections.IDictionary)field.GetValue(null)!).Values.Cast<AbstractModel>();
    }

    /// <summary>涅奥（Neow）开局的遗物选项，照 MegaCrit.Sts2.Core.Models.Events.Neow 的 RelicOption&lt;T&gt; 列出（typeof：游戏删掉某个遗物时编译报错）。</summary>
    static readonly Type[] NeowRelicOptions =
    {
            typeof(MegaCrit.Sts2.Core.Models.Relics.ArcaneScroll), typeof(MegaCrit.Sts2.Core.Models.Relics.BoomingConch), typeof(MegaCrit.Sts2.Core.Models.Relics.CursedPearl), typeof(MegaCrit.Sts2.Core.Models.Relics.DowsingRod),
            typeof(MegaCrit.Sts2.Core.Models.Relics.FishingRod), typeof(MegaCrit.Sts2.Core.Models.Relics.GoldenPearl), typeof(MegaCrit.Sts2.Core.Models.Relics.HeftyTablet), typeof(MegaCrit.Sts2.Core.Models.Relics.Kaleidoscope),
            typeof(MegaCrit.Sts2.Core.Models.Relics.LargeCapsule), typeof(MegaCrit.Sts2.Core.Models.Relics.LavaRock), typeof(MegaCrit.Sts2.Core.Models.Relics.LeadPaperweight), typeof(MegaCrit.Sts2.Core.Models.Relics.LeafyPoultice),
            typeof(MegaCrit.Sts2.Core.Models.Relics.LostCoffer), typeof(MegaCrit.Sts2.Core.Models.Relics.MassiveScroll), typeof(MegaCrit.Sts2.Core.Models.Relics.NeowsBones), typeof(MegaCrit.Sts2.Core.Models.Relics.NeowsSacrifice),
            typeof(MegaCrit.Sts2.Core.Models.Relics.NeowsTalisman), typeof(MegaCrit.Sts2.Core.Models.Relics.NeowsTorment), typeof(MegaCrit.Sts2.Core.Models.Relics.NewLeaf), typeof(MegaCrit.Sts2.Core.Models.Relics.NutritiousOyster),
            typeof(MegaCrit.Sts2.Core.Models.Relics.PhialHolster), typeof(MegaCrit.Sts2.Core.Models.Relics.Pomander), typeof(MegaCrit.Sts2.Core.Models.Relics.PrecariousShears), typeof(MegaCrit.Sts2.Core.Models.Relics.PreciseScissors),
            typeof(MegaCrit.Sts2.Core.Models.Relics.ScrollBoxes), typeof(MegaCrit.Sts2.Core.Models.Relics.SilkenTress), typeof(MegaCrit.Sts2.Core.Models.Relics.SilverCrucible), typeof(MegaCrit.Sts2.Core.Models.Relics.SmallCapsule),
            typeof(MegaCrit.Sts2.Core.Models.Relics.StoneHumidifier), typeof(MegaCrit.Sts2.Core.Models.Relics.WingedBoots),
    };

    /// <summary>当前游戏版本应有的全部词表键（模型、事件选项、休息处选项、奖励、地图节点、意图、招式等；selfcheck 用来查词表缺哪些）。</summary>
    internal static IEnumerable<string> AllKeys()
    {
        foreach (var m in AllModels()) yield return Vocab.Model(m);
        foreach (string k in LocKeys("events.json").Concat(LocKeys("ancients.json")))
            if (k.Contains(".options."))
                yield return Vocab.EventOption(StripLocSuffix(k));
        // v0.113.0 换版本时补上：涅奥开局的遗物选项（AncientEventModel.RelicOption<T>，文本键 NEOW.pages.INITIAL.options.<遗物>）本地化里没有逐项的键，
        // v0.111.0 时一直是词表外（selfcheck 冒烟每次报 6–8 个，引擎模块与游戏更新适配.md 第四节留到换版本时处理）。照 Neow.cs 的三组选项列出（两版相同，30 个）
        foreach (Type t in NeowRelicOptions) yield return Vocab.EventOption("NEOW.pages.INITIAL.options." + ModelDb.GetId(t).Entry);
        foreach (string k in LocKeys("rest_site_ui.json"))
            if (k.StartsWith("OPTION_")) yield return Vocab.Rest(StripLocSuffix(k)["OPTION_".Length..]);
        foreach (string k in LocKeys("card_reward_ui.json"))
            if (k.StartsWith("OPTION_")) yield return Vocab.Alt(StripLocSuffix(k)["OPTION_".Length..]);
        foreach (Type t in typeof(Reward).Assembly.GetTypes().Where(t => t.IsSubclassOf(typeof(Reward)) && !t.IsAbstract))
            yield return Vocab.Reward(t.Name);
        foreach (string n in Enum.GetNames<MapPointType>()) yield return Vocab.MapNode(n);
        foreach (string n in Enum.GetNames<IntentType>()) yield return Vocab.Intent(n);
        yield return Vocab.ShopRemoval;
        foreach (var m in AllModels().OfType<MonsterModel>())
            foreach (string k in MoveKeys(m)) yield return k;
    }

    static readonly MethodInfo GenMoves = GameCompat.Method(typeof(MonsterModel), "GenerateMoveStateMachine", BindingFlags.NonPublic | BindingFlags.Instance, "怪物的招式表（词表、招式预告）")!;

    /// <summary>怪物的全部招式键：生成一台行动状态机（游戏预载资源时也这样做），取其中的招式状态。
    /// 有的怪物生成时要读写实例状态，规范实例上会报错，改在它的可变副本上生成。</summary>
    internal static IEnumerable<string> MoveKeys(MonsterModel m)
    {
        MonsterMoveStateMachine? sm = null;
        try { sm = (MonsterMoveStateMachine)GenMoves.Invoke(m, null)!; }
        catch
        {
            try { sm = (MonsterMoveStateMachine)GenMoves.Invoke(m.ToMutable(), null)!; }
            catch (Exception e) { Console.Error.WriteLine($"招式枚举失败 {Vocab.Model(m)}：{(e.InnerException ?? e).Message}"); }
        }
        return sm == null ? Enumerable.Empty<string>() : sm.States.Where(kv => kv.Value is MoveState).Select(kv => Vocab.Move(m, kv.Key));
    }

    static string StripLocSuffix(string k)
    {
        int i = k.LastIndexOf('.');
        return i > 0 ? k[..i] : k;
    }

    static IEnumerable<string> LocKeys(string file)
    {
        string path = Path.Combine(VirtualFs.ResDir, "localization", "eng", file);
        if (!File.Exists(path)) return Array.Empty<string>();
        return JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(path))!.Keys;
    }

    // ---------------------------------------------------------------- 属性与效果签名

    public static void Export(string path, string version)
    {
        var entries = new List<Dictionary<string, object?>>();
        foreach (var m in AllModels().OrderBy(m => Vocab.Model(m), StringComparer.Ordinal))
        {
            var e = new Dictionary<string, object?> { ["key"] = Vocab.Model(m), ["id"] = Vocab.Id(Vocab.Model(m)), ["category"] = m.Id.Category };
            try { AddAttributes(m, e); } catch (Exception ex) { e["attr_error"] = ex.GetBaseException().Message; }
            e["signature"] = Signature(m.GetType());
            entries.Add(e);
        }
        var doc = new Dictionary<string, object?>
        {
            ["game_version"] = version,
            ["vocab_size"] = Vocab.Count,
            ["entities"] = entries,
        };
        File.WriteAllText(path, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = false }));
    }

    static void AddAttributes(AbstractModel m, Dictionary<string, object?> e)
    {
        switch (m)
        {
            case CardModel c:
                e["type"] = c.Type.ToString();
                e["rarity"] = c.Rarity.ToString();
                e["target"] = c.TargetType.ToString();
                e["cost"] = c.EnergyCost.Canonical;
                e["cost_x"] = c.EnergyCost.CostsX;
                e["star_cost"] = c.CanonicalStarCost;
                e["keywords"] = c.CanonicalKeywords.Select(k => k.ToString()).ToList();
                e["tags"] = c.Tags.Select(t => t.ToString()).ToList();
                e["vars"] = c.DynamicVars.ToDictionary(kv => kv.Key, kv => (double)kv.Value.BaseValue);
                try { e["pool"] = c.Pool?.Id.Entry; } catch { }
                // 升级后的样子（决策记录 264）：复制一份可变实例再升级一次（同升级预览），记下升级后的费用、星费、关键字、数值变量；
                // 不能升级或取不到时不写。玩家在游戏里随时能查看升级效果，是公开信息
                e["max_upgrade"] = c.MaxUpgradeLevel;
                try
                {
                    var mc = c.ToMutable();
                    if (mc.IsUpgradable)
                    {
                        mc.UpgradeInternal();
                        e["upgrade"] = new Dictionary<string, object?>
                        {
                            ["cost"] = mc.EnergyCost.CostsX ? -1 : mc.EnergyCost.GetWithModifiers(CostModifiers.None),
                            ["star_cost"] = mc.BaseStarCost,
                            ["keywords"] = mc.Keywords.Select(k => k.ToString()).OrderBy(k => k, StringComparer.Ordinal).ToList(),
                            ["vars"] = mc.DynamicVars.ToDictionary(kv => kv.Key, kv => (double)kv.Value.BaseValue),
                        };
                    }
                }
                catch (Exception ex) { e["upgrade_error"] = ex.GetBaseException().Message; }
                break;
            case RelicModel r:
                e["rarity"] = r.Rarity.ToString();
                e["vars"] = Vars(r.DynamicVars);
                try { e["pool"] = r.Pool?.Id.Entry; } catch { }
                break;
            case PotionModel p:
                e["rarity"] = p.Rarity.ToString();
                e["pools"] = ModelDb.AllPotionPools.Where(pp => pp.AllPotions.Any(x => x.Id == p.Id)).Select(pp => pp.Id.Entry).ToList();
                e["usage"] = p.Usage.ToString();
                e["target"] = p.TargetType.ToString();
                e["vars"] = Vars(p.DynamicVars);
                break;
            case PowerModel pw:
                e["type"] = pw.Type.ToString();
                e["stack"] = pw.StackType.ToString();
                e["vars"] = Vars(pw.DynamicVars);
                break;
            case MonsterModel mo:
                e["min_hp"] = mo.MinInitialHp;
                e["max_hp"] = mo.MaxInitialHp;
                break;
            case EncounterModel en:
                e["room"] = en.RoomType.ToString();
                e["weak"] = en.IsWeak;
                break;
        }
    }

    /// <summary>描述文字里的基础数值（遗物、药水、能力；回放播放器代入描述用）。有的模型在规范实例上取数会报错，取不到就留空。</summary>
    static Dictionary<string, double> Vars(MegaCrit.Sts2.Core.Localization.DynamicVars.DynamicVarSet? vs)
    {
        try { return vs == null ? new() : vs.ToDictionary(kv => kv.Key, kv => (double)kv.Value.BaseValue); }
        catch { return new(); }
    }

    static readonly string[] CommandNamespaces = { "MegaCrit.Sts2.Core.Commands" };

    /// <summary>
    /// 效果签名：类型自身（含编译器生成的嵌套类，如 async 状态机）的代码中
    ///   hook:X —— 重写的虚方法（游戏钩子，如 AfterCardPlayed）；
    ///   cmd:X.Y —— 调用的命令（DamageCmd、PowerCmd.Apply 等）；
    ///   ref:CATEGORY.ENTRY —— 作为泛型参数引用的其他模型（如 PowerCmd.Apply&lt;VulnerablePower&gt;、PowerVar&lt;VulnerablePower&gt;）。
    /// </summary>
    public static List<string> Signature(Type t)
    {
        var sig = new SortedSet<string>(StringComparer.Ordinal);
        foreach (MethodInfo m in t.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            if (m.IsVirtual && m.GetBaseDefinition().DeclaringType != t && !m.IsSpecialName)
                sig.Add("hook:" + m.Name);
        foreach (MethodInfo m in AllMethods(t))
        {
            if (m.GetMethodBody()?.GetILAsByteArray() is not { } il) continue;
            foreach (MethodBase callee in IlScan.Callees(il, m.Module, m))
            {
                var dt = callee.DeclaringType;
                if (dt?.Namespace != null && CommandNamespaces.Any(ns => dt.Namespace.StartsWith(ns)) && !dt.Name.StartsWith("<"))
                    sig.Add($"cmd:{dt.Name}.{callee.Name}");
                var typeArgs = (callee is MethodInfo { IsGenericMethod: true } gm ? gm.GetGenericArguments() : Type.EmptyTypes)
                    .Concat(dt is { IsGenericType: true } ? dt.GetGenericArguments() : Type.EmptyTypes);
                foreach (Type ga in typeArgs)
                    if (ga.IsSubclassOf(typeof(AbstractModel)) && !ga.IsAbstract && ga != t)
                        sig.Add($"ref:{ModelDb.GetCategory(ga).ToUpperInvariant()}.{ModelDb.GetEntry(ga)}");
            }
        }
        return sig.ToList();
    }

    static IEnumerable<MethodInfo> AllMethods(Type t) =>
        t.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly)
            .Where(m => !m.IsAbstract && !m.ContainsGenericParameters)
            .Concat(t.GetNestedTypes(BindingFlags.NonPublic | BindingFlags.Public).Where(n => !n.ContainsGenericParameters).SelectMany(AllMethods));
}
