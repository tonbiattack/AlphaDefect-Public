namespace Sts2Headless;

/// <summary>随机合法动作驱动：在 RunSession 给出的候选中均匀随机选择，用于冒烟测试和测速。</summary>
public sealed class RandomDriver(int seed)
{
    readonly Random _rng = new(seed);
    public readonly List<string> Log = new();
    public bool Trace;
    public readonly Dictionary<DecisionKind, int> Counts = new();
    /// <summary>少见流程的经过次数（假商人、取消选牌、战斗外用药、丢药），冒烟测试用来确认这些分支被走到。</summary>
    public readonly Dictionary<string, int> Tags = new();
    /// <summary>流程测试用：每场战斗开始时给自己 999 层覆甲和再生（同官方 AutoSlay），不用于训练。</summary>
    public bool God;
    /// <summary>有其他动作时选“结束回合”的概率，避免随机策略每回合几乎不出牌。</summary>
    public double EndTurnProb = 0.15;
    object? _buffedCombat;

    /// <summary>走完一局或卡住。返回最后一个决策点（GameOver 或 Stuck）。</summary>
    public Decision Play(RunSession run, int maxSteps = 20000)
    {
        for (int i = 0; i < maxSteps; i++)
        {
            var d = run.GetDecision();
            Counts[d.Kind] = Counts.GetValueOrDefault(d.Kind) + 1;
            if (d.Kind is DecisionKind.GameOver or DecisionKind.Stuck) { Note($"[{d}]"); return d; }
            if (d.Kind == DecisionKind.Combat && God) Buff();
            int j = _rng.Next(d.Choices.Count);
            if (d.Kind == DecisionKind.Combat && d.Choices.Count > 1)
                j = _rng.NextDouble() < EndTurnProb ? d.Choices.Count - 1 : _rng.Next(d.Choices.Count - 1);
            // 逐张选牌时，随机决定是否就此确认（否则倾向于选满）
            if (d.Kind == DecisionKind.CardSelect && d.Choices[^1].Kind == ActionKind.ConfirmSelection && d.Choices.Count > 1)
                j = _rng.NextDouble() < 0.3 ? d.Choices.Count - 1 : _rng.Next(d.Choices.Count - 1);
            var c = d.Choices[j];
            string? tag = d.Context?.StartsWith("假商人") == true ? "假商人" : c.Label == "取消" ? "取消选牌"
                : c.Kind == ActionKind.DiscardPotion ? "丢药" : c.Kind == ActionKind.UsePotion && d.Kind != DecisionKind.Combat ? "战斗外用药" : null;
            if (tag != null) Tags[tag] = Tags.GetValueOrDefault(tag) + 1;
            Note($"[{d}] {c.Label}");
            run.Choose(d, j);
        }
        return new Decision(DecisionKind.Stuck) { Context = "超过步数上限" };
    }

    void Buff()
    {
        var cs = MegaCrit.Sts2.Core.Combat.CombatManager.Instance.DebugOnlyGetState();
        if (cs == null || ReferenceEquals(cs, _buffedCombat)) return;
        _buffedCombat = cs;
        var me = RunSession.State.Players[0].Creature;
        var ctx = new MegaCrit.Sts2.Core.GameActions.Multiplayer.ThrowingPlayerChoiceContext();
        Pump.Instance.Start(async () =>
        {
            await MegaCrit.Sts2.Core.Commands.PowerCmd.Apply<MegaCrit.Sts2.Core.Models.Powers.PlatingPower>(ctx, me, 999m, me, null);
            await MegaCrit.Sts2.Core.Commands.PowerCmd.Apply<MegaCrit.Sts2.Core.Models.Powers.RegenPower>(ctx, me, 999m, me, null);
        });
    }

    void Note(string s) { Log.Add(s); if (Trace) Console.WriteLine(s); }
}
