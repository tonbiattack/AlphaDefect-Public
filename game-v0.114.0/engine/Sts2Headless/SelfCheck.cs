using System.Runtime.CompilerServices;
using MegaCrit.Sts2.Core.Models;

namespace Sts2Headless;

/// <summary>
/// 游戏更新后的自检（`dotnet Sts2Headless.dll selfcheck [每个角色的整局数] [--all]`，引擎模块与游戏更新适配.md 第四节）。
/// 一次查全引擎对游戏的依赖，有问题时退出码为 1：
///   1. 补丁目标、私有成员（GameCompat 登记的每一处）：启动时以自检模式装补丁（找不到的记下来继续），再把引擎里所有类型的静态初始化跑一遍（反射字段都在那里）；
///   2. 按名字写的内容表：OrbChannels 的卡牌类名、MoveGains 的（怪物，招式 ID）；格式 29 的 MoveGains29、MoveDebuffs29、睡眠类醒来的招式与分支（Wake29）；
///   3. 代码里直接写的词表键（STAT:STARS、CARDMOD:*、TINKER:* 等）在不在词表；当前游戏应有的键有哪些不在词表（新内容，要跑 catalog 追加）；
///   4. 冒烟：五个角色各打几局整局与第一、二、三幕的普通、精英、Boss 战（随机出牌，每步都编码观测），数异常、卡住与新出现的词表外键；
///      另按格式 29 再打一局整局与各幕的精英、Boss 战（各角色默认格式还不是 29 时，格式 29 的编码在默认格式下走不到）。
/// 只读：不写词表、不写文件。
/// </summary>
public static class SelfCheck
{
    static readonly string[] Chars = { "IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER", "FOOL" };   // v0.113.0 加了第六个角色 FOOL
    /// <summary>引擎代码里直接写的词表键（不经 ModelDb 的），新版词表里应该都有。</summary>
    static readonly string[] CodeKeys = { "STAT:STARS", "CARDMOD:REPLAY", "CARDMOD:COST_NEXT_TURN", "CARDMOD:COST_AFTER_PLAY", "CARDMOD:COPY", "CARDMOD:RETURN_NEXT_TURN", "CARDMOD:BRIDGE_POOL", Vocab.ShopRemoval };

    public static int Run(int runs, bool all)
    {
        int problems = 0;
        void Line(string s) => Console.WriteLine(s);

        // 1. 静态初始化：反射字段都在静态初始化里登记（GameCompat）
        var types = typeof(SelfCheck).Assembly.GetTypes().Where(t => t.Namespace == "Sts2Headless" && !t.IsGenericTypeDefinition && !t.Name.StartsWith('<')).ToList();
        foreach (var t in types)
        {
            try { RuntimeHelpers.RunClassConstructor(t.TypeHandle); }
            catch (Exception e) { problems++; Line($"缺  静态初始化 {t.FullName}：{e.GetBaseException().Message}"); }
        }

        // 2. 按名字写的内容表
        GameCompat.CardNames("故障机器人：哪些牌生成哪些充能球（Observation.OrbChannels）", Observation.OrbChannelCards);
        var monsters = ModelDb.All.OfType<MonsterModel>().ToList();
        foreach (var (mt, move) in Observation.MoveGainKeys)
        {
            var m = monsters.FirstOrDefault(x => x.GetType() == mt);
            bool ok = m != null && Catalog.MoveKeys(m).Contains(Vocab.Move(m, move));
            if (!ok) { problems++; Line($"缺  招式 {mt.Name}.{move}（招式给的力量、格挡：Observation.MoveGains）"); }
            else if (all) Line($"有  招式 {mt.Name}.{move}");
        }
        foreach (var (mt, move) in Observation.HandAddKeys)
        {
            var m = monsters.FirstOrDefault(x => x.GetType() == mt);
            bool ok = m != null && Catalog.MoveKeys(m).Contains(Vocab.Move(m, move));
            if (!ok) { problems++; Line($"缺  招式 {mt.Name}.{move}（往玩家手里塞牌：Observation.HandAdds）"); }
            else if (all) Line($"有  招式 {mt.Name}.{move}");
        }
        // 格式 29（G05）：招式推演的增益表、易伤表、睡眠类醒来的招式与分支（Creatures29.cs）
        foreach (var (what, keys) in new[] { ("招式推演的力量、仪式、活力：Observation.MoveGains29", Observation.MoveGain29Keys),
                                             ("招式推演给玩家上的易伤：Observation.MoveDebuffs29", Observation.MoveDebuff29Keys),
                                             ("睡眠类醒来的招式：Observation.Wake29", Observation.Wake29Moves) })
            foreach (var (mt, move) in keys)
            {
                var m = monsters.FirstOrDefault(x => mt.IsInstanceOfType(x));      // 千足虫的招式登记在基类 DecimillipedeSegment 上
                bool ok = m != null && Catalog.MoveKeys(m).Contains(Vocab.Move(m, move));
                if (!ok) { problems++; Line($"缺  招式 {mt.Name}.{move}（{what}）"); }
                else if (all) Line($"有  招式 {mt.Name}.{move}");
            }
        foreach (var (mt, branch) in Observation.Wake29Branches)
        {
            var m = monsters.FirstOrDefault(x => mt.IsInstanceOfType(x));
            bool ok = m != null && Observation.StateIds29(m).Contains(branch);
            if (!ok) { problems++; Line($"缺  分支 {mt.Name}.{branch}（睡眠类的条件分支：Observation.Wake29）"); }
            else if (all) Line($"有  分支 {mt.Name}.{branch}");
        }

        // 1、2 的汇总（补丁、反射、卡牌类名）
        foreach (var s in GameCompat.Report(all)) { if (s.StartsWith("缺")) problems++; Line(s); }
        var needs = GameCompat.Needs;
        Line($"兼容层登记 {needs.Count} 处（补丁 {needs.Count(n => n.Kind == "补丁")}），找不到 {needs.Count(n => !n.Found && !n.Optional)} 处（另有可选的 {needs.Count(n => !n.Found && n.Optional)} 处）");

        // 3. 词表
        var have = Vocab.AllKeys.ToHashSet();
        foreach (var k in CodeKeys.Where(k => !have.Contains(k))) { problems++; Line($"缺  词表键 {k}（引擎代码里直接写的）"); }
        var want = Catalog.AllKeys().Distinct().ToList();
        var newKeys = want.Where(k => !have.Contains(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        Line($"词表 {Vocab.Count} 项；当前游戏应有的键 {want.Count} 个，不在词表的 {newKeys.Count} 个" + (newKeys.Count > 0 ? "（新内容：跑 catalog 追加，旧 ID 不变）" : ""));
        foreach (var k in newKeys.Take(all ? int.MaxValue : 60)) Line($"新  {k}");
        if (!all && newKeys.Count > 60) Line($"新  ……另有 {newKeys.Count - 60} 个（--all 全列）");
        var gone = have.Where(k => k.Contains('.') && !k.StartsWith("EVENTOPT:") && !k.StartsWith("MOVE:") && !k.Contains(':')).Where(k => !want.Contains(k)).ToList();
        Line($"词表里有、当前游戏没有的模型键 {gone.Count} 个（游戏删掉或改名的内容；词表只追加不删，旧 ID 留着）");
        foreach (var k in gone.Take(all ? int.MaxValue : 40)) Line($"旧  {k}");
        // 别名（engine/catalog/aliases.txt，v0.113.0 起）：新键要是当前游戏会输出的键，指向的老 ID 要在词表里（老键不在词表的行载入时已忽略）
        var wantSet = want.ToHashSet();
        Line($"词表别名 {Vocab.AliasIds.Count} 条（改了名、意思没变的内容，观测用老键的 ID）");
        foreach (var (k, id) in Vocab.AliasIds)
        {
            // 招式键按各怪物的招式表核对，其余看当前游戏应有的键
            bool live = k.StartsWith("MOVE:") ? monsters.Any(m => Catalog.MoveKeys(m).Contains(k)) : wantSet.Contains(k);
            if (!live) { problems++; Line($"缺  别名的新键 {k}（当前游戏不输出它，别名失效）"); }
            else if (all) Line($"有  别名 {k} → {Vocab.AllKeys[id]}");
        }

        // 4. 冒烟
        int missingBefore = Vocab.Missing.Count;
        string[] encs = { "weak", "regular", "elite", "boss" };
        foreach (var c in Chars)
        {
            int frames = 0, errors = 0, stuck = 0, games = 0;
            var rng = new Random(c.GetHashCode() & 0xFFFF);
            void Play(Func<RunSession> start, int maxSteps)
            {
                games++;
                try
                {
                    using var run = start();
                    for (var d = run.GetDecision(); d.Kind != DecisionKind.GameOver && run.Steps < maxSteps; d = run.GetDecision())
                    {
                        if (d.Kind == DecisionKind.Stuck) { stuck++; Line($"卡  {c} 第 {run.Steps} 步：{d}"); break; }
                        Observation.Build(run, d, false).Serialize();
                        frames++;
                        int n = d.Choices.Count;
                        run.Choose(d, n > 1 && rng.NextDouble() > 0.2 ? rng.Next(n - 1) : n - 1);
                    }
                }
                catch (Exception e) { errors++; Line($"错  {c}：{e.GetBaseException().GetType().Name}：{e.GetBaseException().Message}"); }
            }
            for (int i = 0; i < runs; i++) Play(() => RunSession.Start(c, 10, $"SELFCHECK{i}"), 1500);
            for (int a = 0; a < 3; a++)
                foreach (var e in encs.Skip(a == 0 ? 0 : 1))
                    Play(() => RunSession.StartCombat(new CombatSpec { Character = c, Seed = $"SC{a}{e}", Act = a, Encounter = e }), 400);
            // 格式 29（R3-格式29规格.md）：再按格式 29 编一遍（各角色默认格式还不是 29 时，新加的编码在上面的冒烟里走不到）
            int f29 = frames;
            var prev = Observation.FormatOverride;
            Observation.FormatOverride = Observation.FormatAll;
            try
            {
                Play(() => RunSession.Start(c, 10, "SELFCHECK29"), 1500);
                for (int a = 0; a < 3; a++)
                    foreach (var e in new[] { "elite", "boss" })
                        Play(() => RunSession.StartCombat(new CombatSpec { Character = c, Seed = $"SC29{a}{e}", Act = a, Encounter = e }), 400);
            }
            finally { Observation.FormatOverride = prev; }
            problems += errors + stuck;
            Line($"冒烟 {c}：{games} 局 {frames} 帧（其中格式 29 {frames - f29} 帧），异常 {errors}，卡住 {stuck}");
        }
        int newMissing = Vocab.Missing.Count - missingBefore;
        Line($"冒烟中遇到的词表外键 {Vocab.Missing.Count} 个" + (Vocab.Missing.Count > 0 ? "：" + string.Join("，", Vocab.Missing.OrderByDescending(kv => kv.Value).Take(12).Select(kv => $"{kv.Key}×{kv.Value}")) : ""));
        Line(problems == 0 ? "自检通过" : $"自检发现 {problems} 处问题");
        return problems == 0 ? 0 : 1;
    }
}
