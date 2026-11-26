using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// 一局游戏的决策接口：GetDecision 给出当前决策点和全部合法候选，Choose / ChooseMany 执行后推进到下一个决策点。
/// 判断顺序：挂起的选择（选牌、卡牌奖励、卡包、奖励界面）优先，然后是战斗，最后按房间类型。
/// 房间里原本由界面节点完成的几步流程（开宝箱、离开商店、奖励界面的“继续”）在这里复刻，对应的界面代码见各方法注释。
/// </summary>
public sealed class RunSession : IDisposable
{
    public readonly AgentCardSelector Selector = new();
    readonly IDisposable _selectorScope;
    Task? _setup;      // 开局任务（卡住时留档用）
    string _origin = "";   // 这一局怎么开的（整局的种子，或单场战斗的完整开局参数），卡住时写进日志以便复现
    // 被放弃的会话里还挂着的选牌：等新局 CleanUp 把旧战斗取消后再取消它们（见 Begin）
    static readonly List<PendingCardChoice> _abandonedChoices = new();
    UiFlows? _uiFlows;

    // 当前房间内的进度。房间对象变了就重置。
    AbstractRoom? _room;
    bool _roomLeft;          // 商店、休息处：已选择离开
    bool _restChosen;        // 休息处：已成功选过一个选项
    bool _chestOpened;       // 宝箱：已开箱
    bool _relicDecided;      // 宝箱：已选遗物或跳过
    bool _shopInitialized;   // 商店：已做进店处理
    // 本房间内失败过的选项（休息处选项、商品）：失败不改变局面，重试结果相同，不再提供，否则策略可能一直重试
    readonly HashSet<object> _failed = new(ReferenceEqualityComparer.Instance);
    // 取消过选牌的房间：每个房间只给一次取消，之后的选牌必须选完，防止“点开选项—取消—再点开”来回打转
    AbstractRoom? _cancelRoom;

    // 执行出错的检查：动作返回的任务挂起时（等玩家选择），之后的续延出错没人看见；游戏的 RunSafely 等只记日志。
    // 两种情况都按卡住结束这一局，而不是让局面停在原地被反复重选
    readonly List<(Task Task, string Label)> _inflight = new();
    int _errorBase;

    public int Steps { get; private set; }

    /// <summary>
    /// 局面重复截断（2026-11-07，用户：“当第四次打出全局同样局面的时候结束回合……玩家的手牌、血量、防御状态都回到了原有的状态，而 BOSS 的血量、
    /// 防御状态也没有明显变化……我们的目的是为了防止像面对猎人杀手一样，被降到 0 伤害，导致永远打不死。或者面对瀑布巨兽最后那个无限血的情况……
    /// 其他的情况我们要允许无限的存在”）。reset 带 "loop_cut": true 时开（训练配置 loop_cut），不带时与原来完全相同。
    /// 同一个玩家回合里，每次给出出牌决策时记下局面：手牌（卡牌 ID 与升级次数的多重集合）、玩家生命、玩家格挡、活着的敌人数、药水栏；
    /// 同一局面再次出现、活着的敌人生命加格挡的合计与上一次出现时相比变化不到 max(1, 1%)、能量与星也都没有比上一次少时，这个局面的连续次数加 1，否则从 1 重新数。
    /// 连续到第 4 次时，这一步只给“结束回合”一个候选（引擎日志记一行 [loopcut]）。能把敌人打死的无限循环每圈都在掉敌人血，不受影响；
    /// 伤害被降到 0（猎人杀手的嫩弱、闪耀打击加不息陀螺）、对着瀑布巨兽爆发前 999,999,999 的血量打的循环会被截断。
    /// 第五版（2026-11-07，储君决策记录第 10 条补记；用户：“更改判断方式，这个地方4次以上，并且本回合预计不会受到伤害就结束”）：
    /// 格挡不再进局面，改为同能量、星一样比较——比上一次少了不算重复；只涨的照算（第 6,361 版评测一局同一回合连出近 2 万次粒子墙，
    /// 每次 +9 格挡、牌回到手里、能量与星都不花，格挡一直在涨，原来每次都算“局面变了”）。连续 4 次以上时：这一串里格挡一直没涨的照旧结束回合；
    /// 涨过的，只在现在结束回合预计不掉血（回合结束预演的最坏情况，Observation.EndTurnPreview 的 worstLoss）时才结束，还会掉血就让它接着加。
    /// </summary>
    public static bool LoopCut;
    readonly Dictionary<string, (double foes, int energy, int stars, int block, int run, bool grew)> _loopSeen = new();
    object? _loopCombat;
    int _loopTurn = -1, _loopStep = -1;
    bool _loopLast;
    readonly Queue<string> _lastLabels = new();     // 最近走过的几步（截断时写进日志，查是哪类动作在原地打转）

    bool LoopRepeated(PlayerCombatState pcs, ICombatState cs)
    {
        if (!ReferenceEquals(cs, _loopCombat) || pcs.TurnNumber != _loopTurn)
        {
            _loopSeen.Clear(); _loopCombat = cs; _loopTurn = pcs.TurnNumber; _loopStep = -1;
        }
        if (Steps == _loopStep) return _loopLast;       // 同一个局面重复取决策（没走新的一步）不重复计数
        _loopStep = Steps;
        var foes = cs.Creatures.Where(c => !c.IsPlayer && c.Side != Me.Creature.Side && c.IsAlive).ToList();
        double total = foes.Sum(c => (double)c.CurrentHp + c.Block);
        string sig = string.Join(",", pcs.Hand.Cards.Select(c => $"{c.Id.Entry}+{c.CurrentUpgradeLevel}").OrderBy(x => x, StringComparer.Ordinal))
                     + $"|{Me.Creature.CurrentHp}|{foes.Count}"
                     // 药水栏也算进局面：连用几瓶不加血不加格挡的药水时手牌、血量都没变，但药水用掉了，不是回到原状（第一次上线时误截断的多是这种）
                     + "|" + string.Join(",", Me.PotionSlots.Select(x => x?.Id.Entry ?? "-"));
        // 能量比上一次同局面时少了不算重复：花能量的循环会自己停（被塞满黏液时连出黏液，敌人往手里补新的，手牌看着没变；第二版上线时误截断的就是这种）。
        // 能量不变或变多照算（猎人杀手那局是 0 费的闪耀打击）。星同理：粒子墙（0 能量、2 星、回到手里）在敏捷被压到很低、格挡加不上时会连出，
        // 但每次花 2 星、星用完就停（第三版上线后剩下的 3 次截断都是它）；猎人杀手那局每圈 +2 星，照算
        int run = 1, energy = pcs.Energy, stars = pcs.Stars, block = Me.Creature.Block;
        bool grew = false;
        if (_loopSeen.TryGetValue(sig, out var prev) && Math.Abs(total - prev.foes) < Math.Max(1.0, 0.01 * prev.foes) && energy >= prev.energy && stars >= prev.stars
            && block >= prev.block)
        {
            run = prev.run + 1;
            grew = prev.grew || block > prev.block;
        }
        _loopSeen[sig] = (total, energy, stars, block, run, grew);
        float loss = 0;
        _loopLast = run >= 4 && (!grew || (loss = WorstLossIfEndTurn(cs)) <= 0);
        if (_loopLast) Console.Error.WriteLine($"[loopcut] 第 {pcs.TurnNumber} 回合局面第 {run} 次重复（敌人生命加格挡 {total:0}，玩家格挡 {block}{(grew ? "，格挡涨过、现在结束回合预计不掉血" : "")}），只给结束回合；"
                                               + $"手牌 {sig}；最近几步 {string.Join(" / ", _lastLabels)}");
        return _loopLast;
    }

    /// <summary>现在结束回合预计最多掉多少血（回合结束预演的最坏情况：随机目标的伤害不算击杀）；算不出时按会掉血处理。</summary>
    static float WorstLossIfEndTurn(ICombatState cs)
    {
        try { Observation.EndTurnPreview(cs, Me, out float worst, out _, out _, out _, out _); return worst; }
        catch { return float.MaxValue; }
    }

    /// <summary>
    /// 跨回合无进展判输（2026-11-07，储君决策记录第 13 条；用户：“加上”）。reset 带 "stall_turns": N（N &gt; 0，训练配置 stall_turns）时开，不带时与原来完全相同。
    /// 第 4,437 版评测有 4 局剩一只弱小史莱姆（1–7 血、伤害挡得住）时，模型手里有攻击牌也只出防御、结束回合，拖了上千回合到 30 分钟上限；
    /// 局面重复截断只看同一回合里的重复，管不到这种一回合接一回合的拖延。
    /// 每个玩家回合第一次给出牌决策时，看上一次检查以来战斗历史里有没有敌人受到的生命伤害（DamageReceivedEntry 的 UnblockedDamage &gt; 0，
    /// 毒、反伤等也算）：有就从 0 重新数，没有就加 1；连续 N 个回合都没有时，按这场战斗输处理——CreatureCmd.Kill(玩家, force: true)
    /// （不让仙女瓶等救回来），整局照常以死亡结束（战斗模式同样算输），引擎日志记一行 [stallcut]。能打掉敌人血的长局、无限循环都不受影响。
    /// 第二版（2026-11-08，储君决策记录第 20 条；用户：“改，先验证”）：“有进展”从“敌人受到过生命伤害”改为“活着的敌人生命合计创了新低”。
    /// 第 28,827 版评测一局（种子 3RSFC8YX2P）对瀑布巨兽打到没有伤害手段，靠 3 层荆棘每 5 回合反伤 12、巨兽每 5 回合吸血回 15，
    /// 几乎每回合都有生命伤害，原来的计数一直清零，玩家靠延续把格挡叠到十几万，拖了约 4,900 回合到 20,000 步上限。
    /// 现在：每回合检查时活着的敌人生命合计低于这一串以来的最低点，或有敌人死了，就从 0 重新数（最低点改为当前合计）；
    /// 有敌人新出现、复活或最大生命变了（召唤、换阶段，如瀑布巨兽爆发前设成 999,999,999）只把最低点改为当前合计，不算进展。
    /// </summary>
    public static int StallTurns;
    object? _stallCombat;
    int _stallTurn = -1, _stallRun;
    long _stallMin;
    Dictionary<Creature, int> _stallFoes = new(ReferenceEqualityComparer.Instance);    // 上次检查时活着的敌人 → 最大生命

    /// <summary>这一回合是不是该判输（每回合只在第一次给出牌决策时计数一次）。</summary>
    bool Stalled(PlayerCombatState pcs, ICombatState cs)
    {
        var foes = cs.Creatures.Where(c => !c.IsPlayer && c.Side != Me.Creature.Side && c.IsAlive).ToList();
        long total = foes.Sum(c => (long)c.CurrentHp);
        if (!ReferenceEquals(cs, _stallCombat))
        {
            _stallCombat = cs; _stallTurn = pcs.TurnNumber; _stallRun = 0; _stallMin = total;
            _stallFoes = MaxHpByFoe(foes);
            return false;
        }
        if (pcs.TurnNumber == _stallTurn) return false;
        _stallTurn = pcs.TurnNumber;
        var now = MaxHpByFoe(foes);
        bool died = _stallFoes.Keys.Any(c => !now.ContainsKey(c));
        bool reshaped = now.Any(kv => !_stallFoes.TryGetValue(kv.Key, out int m) || m != kv.Value);
        if (died || total < _stallMin) { _stallRun = 0; _stallMin = total; }
        else { _stallRun++; if (reshaped) _stallMin = total; }
        _stallFoes = now;
        return _stallRun >= StallTurns;
    }

    static Dictionary<Creature, int> MaxHpByFoe(IEnumerable<Creature> foes)
    {
        var d = new Dictionary<Creature, int>(ReferenceEqualityComparer.Instance);
        foreach (var c in foes) d[c] = c.MaxHp;
        return d;
    }

    // v0.113.0 起选牌器挂在每局的 RunManager.CardSelectorStack 上（每个玩家一个栈），原来的全局 CardSelectCmd.PushSelector 没了；
    // 照游戏 AutoSlayer 的做法用静态的 SetTestSelector：玩家的栈为空时用它，游戏自己压的（低语耳环自动打牌）照旧在上面
    RunSession() => _selectorScope = MegaCrit.Sts2.Core.CardSelection.CardSelectorStack.SetTestSelector(Selector);

    /// <summary>战斗模式：只打一场，战斗结束即结束。</summary>
    public bool CombatOnly { get; private set; }

    /// <summary>战斗模式的连打（CombatSpec.NextEncounter）：这一场打赢后还要接着打的遭遇，开打后清空；Stage 为已经开打的第几场（0 起）。</summary>
    string? _nextEncounter;
    public int Stage { get => Observation.Stage; private set => Observation.Stage = value; }

    /// <summary>遭遇的词表 ID（练习规格里的遭遇可带可不带 ENCOUNTER. 前缀）。</summary>
    static int EncounterId(string? e) => string.IsNullOrEmpty(e) ? 0 : Vocab.Id("ENCOUNTER." + e.Replace("ENCOUNTER.", ""));

    public static RunSession Start(string characterId, int ascension, string seed)
    {
        Observation.Stage = 0; Observation.BossPrev = Observation.BossNext = 0;
        var s = Begin(() => HeadlessRun.StartNew(characterId, ascension, seed), combatOnly: false);
        s._origin = $"整局 {characterId} 进阶 {ascension} 种子 {seed}";
        return s;
    }

    /// <summary>从开幕快照开整局（中途开局，见 ActStart）。</summary>
    public static RunSession StartFromAct(string characterId, int ascension, string seed, ActStartSnapshot a)
    {
        Observation.Stage = 0; Observation.BossPrev = Observation.BossNext = 0;
        var s = Begin(() => ActStart.Start(characterId, ascension, seed, a), combatOnly: false);
        s._origin = $"中途开局（第 {a.Act + 1} 幕）{characterId} 进阶 {ascension} 种子 {seed}";
        return s;
    }

    public static RunSession StartCombat(CombatSpec spec)
    {
        var s = Begin(() => CombatSetup.Start(spec), combatOnly: true);
        s._nextEncounter = string.IsNullOrEmpty(spec.NextEncounter) ? null : spec.NextEncounter;
        s.Stage = 0;
        Observation.BossPrev = EncounterId(spec.PrevEncounter); Observation.BossNext = EncounterId(s._nextEncounter);
        s._origin = "单场战斗 " + System.Text.Json.JsonSerializer.Serialize(spec, new System.Text.Json.JsonSerializerOptions { PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.SnakeCaseLower });
        return s;
    }

    static RunSession Begin(Func<Task> setup, bool combatOnly)
    {
        if (RunManager.Instance.IsInProgress)
        {
            RunManager.Instance.CleanUp();
            // 上一局的战斗如果打到一半就被重新开局，CleanUp 会取消它的令牌，被取消的续延还排在泵里；
            // 先执行掉再开新局，否则它们会在新战斗开始之后才跑（2026-10-26 核查卡住时加）
            Pump.Instance.RunUntilIdle();
        }
        // 旧战斗停在选牌上（例如赌博筹码在回合开始时让玩家弃牌）时，它的回合循环一直等这次选牌，被取消也不会结束，
        // 下一场开战要先等它 10 秒。旧战斗的令牌已经取消，这时再取消选牌，回合循环就按“战斗已拆除”正常退出（2026-10-26）
        if (_abandonedChoices.Count > 0)
        {
            foreach (var p in _abandonedChoices) p.Tcs.TrySetCanceled();
            _abandonedChoices.Clear();
            Pump.Instance.RunUntilIdle();
        }
        ResetActionExecutor();
        RewardsFlow.Install();
        RewardsFlow.Reset();
        FakeScreens.Reset();
        EventOptionTracker.Pending = null;
        AgentCardSelector.NextMinSelect = null;
        GameShim.LastRunVictory = null;
        var session = new RunSession { CombatOnly = combatOnly };
        try
        {
            int errors0 = GodotShim.ErrorTotal;
            var t = Pump.Instance.Start(setup);
            if (t.IsFaulted) throw t.Exception!;
            if (GodotShim.ErrorTotal != errors0)
                Console.Error.WriteLine($"[setup] 开局时游戏报错 {GodotShim.ErrorTotal - errors0} 条，最后一条：{GodotShim.LastError}");
            session._setup = t;
            if (!t.IsCompleted)
                Console.Error.WriteLine($"[setup] 开局任务没有完成（{t.Status}），战斗进行中 {CombatManager.Instance.IsInProgress}，正在开始 {CombatManager.Instance.IsStarting}");
            if (combatOnly && !CombatManager.Instance.IsInProgress)
            {
                var st = RunManager.Instance.DebugOnlyGetState();
                Console.Error.WriteLine($"[setup] 开局后战斗没有在进行：房间 {st?.CurrentRoom?.GetType().Name}，"
                    + $"房间里的敌人 {(st?.CurrentRoom as MegaCrit.Sts2.Core.Rooms.CombatRoom)?.CombatState?.Enemies?.Count() ?? -1}，玩家血量 {st?.Players[0].Creature.CurrentHp}，"
                    + $"正在开始 {CombatManager.Instance.IsStarting}，动作执行器 运行 {RunManager.Instance.ActionExecutor.IsRunning} 暂停 {RunManager.Instance.ActionExecutor.IsPaused} "
                    + $"当前动作 {RunManager.Instance.ActionExecutor.CurrentlyRunningAction?.GetType().Name}");
            }
            RunManager.Instance.RestSiteSynchronizer.AfterPlayerOptionChosen += session.OnRestOptionChosen;
            session._uiFlows = new UiFlows();
            session._errorBase = GodotShim.ErrorTotal;
            return session;
        }
        catch
        {
            session.Dispose();   // 构造时压入的选牌器要弹出，否则留在栈上
            throw;
        }
    }

    /// <summary>
    /// 动作执行器跨局复用，CleanUp 不重置它。战斗打到一半被重新开局时，它的执行循环被取消，游戏把“动作全部执行完”的任务标成出错
    /// （ActionExecutor.ExecuteActions：“should never be canceled”）并留在执行器里，暂停标志也可能留着；下一局开战时
    /// CombatManager.StartCombatInternal 第一步就等这个任务，拿到出错后整个开战流程被 TaskHelper.RunSafely 吞掉，
    /// 战斗停在“正在开始”。本机核对：单个引擎连续放弃约 19 场战斗后，之后每场都开不起来，被当成战斗结束、记为胜利。
    /// 新开局前把已结束的残留任务清掉、暂停解除（新建的执行器就是这个状态），并写日志留档（2026-10-26）。
    /// </summary>
    static void ResetActionExecutor()
    {
        var ex = RunManager.Instance.ActionExecutor;
        if (ex == null) return;
        var f = HarmonyLib.AccessTools.Field(typeof(MegaCrit.Sts2.Core.GameActions.ActionExecutor), "_queueTaskCompletionSource");
        string? note = null;
        if (f.GetValue(ex) is TaskCompletionSource<bool> tcs && tcs.Task.IsCompleted)
        {
            if (tcs.Task.IsFaulted) note = "清掉出错的“动作执行完”任务";
            f.SetValue(ex, null);
        }
        if (ex.IsPaused && !ex.IsRunning) { ex.Unpause(); note = (note == null ? "" : note + "，") + "解除暂停"; }
        if (note != null) Console.Error.WriteLine($"[reset] 动作执行器有上一局的残留：{note}");

        // 上一场战斗的回合循环被取消后偶尔停在某个不认取消令牌的等待上，永远不结束；下一场开战会先等它 10 秒，超时才继续。
        // 不等它：把记着它的字段清掉，并把它停在哪里写进日志留档
        var lf = HarmonyLib.AccessTools.Field(typeof(CombatManager), "_turnLoopTask");
        if (lf.GetValue(CombatManager.Instance) is Task loop && !loop.IsCompleted)
        {
            Console.Error.WriteLine($"[reset] 上一场战斗的回合循环没有结束，不再等它；停在：{AwaitChain(loop)}");
            lf.SetValue(CombatManager.Instance, null);
        }
    }

    /// <summary>沿 async 状态机的等待对象往下找，列出一个未完成任务当前停在哪些 async 方法里（只用于卡住日志）。</summary>
    static string AwaitChain(Task? t)
    {
        var parts = new List<string>();
        for (int depth = 0; t != null && !t.IsCompleted && depth < 20; depth++)
        {
            var sm = t.GetType().GetField("StateMachine")?.GetValue(t);
            if (sm == null)
            {
                // Task.WaitAsync、WhenAny 之类的包装：里面的原任务在 _task、_task1、_task2 字段
                // （WhenAny 的两个任务里优先跟 async 方法那个，另一个通常是“暂停”信号）
                var inners = new[] { "_task", "_task1", "_task2" }
                    .Select(n => t.GetType().GetField(n, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)?.GetValue(t) as Task)
                    .Where(it => it is { IsCompleted: false }).ToList();
                var inner = inners.FirstOrDefault(it => it!.GetType().GetField("StateMachine") != null) ?? inners.FirstOrDefault();
                parts.Add(t.GetType().Name.Split('`')[0]);
                t = inner;
                continue;
            }
            var smt = sm.GetType();
            var state = smt.GetField("<>1__state")?.GetValue(sm);
            parts.Add($"{smt.DeclaringType?.Name}.{smt.Name}#{state}");
            Task? next = null;
            foreach (var f in smt.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance))
            {
                if (!f.Name.StartsWith("<>u__")) continue;
                var aw = f.GetValue(sm);
                if (aw == null) continue;
                var tf = aw.GetType().GetField("m_task", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                      ?? aw.GetType().GetField("_task", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (tf?.GetValue(aw) is Task at && !at.IsCompleted) { next = at; break; }
                if (aw is not Task && tf == null && aw.GetType().Name is var n && !n.StartsWith("TaskAwaiter") && !n.StartsWith("Configured"))
                    parts.Add($"[{n}]");
            }
            t = next;
        }
        return string.Join(" → ", parts);
    }

    public void Dispose()
    {
        RunManager.Instance.RestSiteSynchronizer.AfterPlayerOptionChosen -= OnRestOptionChosen;
        _uiFlows?.Dispose();
        _selectorScope.Dispose();
        if (Selector.Pending is { } p) _abandonedChoices.Add(p);
    }

    void OnRestOptionChosen(MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption option, bool success, ulong __)
    {
        if (success) _restChosen = true;
        else _failed.Add(option);
    }

    public static RunState State => RunManager.Instance.DebugOnlyGetState()!;
    static Player Me => State.Players[0];

    public bool IsOver => State.IsGameOver || (CombatOnly && !CombatManager.Instance.IsInProgress);
    /// <summary>整局模式：通关；战斗模式：赢下这场战斗。战斗模式里限时假人（BattlewornDummyEventEncounter）到时间逃走不算赢：
    /// 战斗也结束了、玩家也活着，但事件的目标没达成，游戏走失败分支（第十三份外部评审，决策记录 206）。别的遭遇里敌人逃走不在此列</summary>
    public bool IsVictory => CombatOnly ? IsOver && !Me.Creature.IsDead && !DummyTimedOut() : State.IsGameOver && GameShim.LastRunVictory == true;

    static bool DummyTimedOut()
    {
        try { return State.CurrentRoom is CombatRoom { Encounter: MegaCrit.Sts2.Core.Models.Encounters.BattlewornDummyEventEncounter { RanOutOfTime: true } }; }
        catch { return false; }
    }

    // ---------------------------------------------------------------- 决策

    public Decision GetDecision()
    {
        var d = GetDecisionOnce();
        // 战斗中但还没到出牌阶段：游戏有续延是从别的线程（线程池上的异步加载等）投回来的，机器繁忙时会晚到，
        // 泵暂时是空的。原来立刻判卡住，开局就卡的局被采集端直接丢掉（租用机上每分钟约 21 次，并发开快照战斗时本机高达九成）。
        // 现在等它们到齐，最多 10 秒；等了 0.1 秒以上或者等到超时，都写到标准错误（引擎日志）里留档（2026-10-26）。
        if (d.Kind == DecisionKind.Stuck && d.Context.StartsWith("战斗中但不在出牌阶段"))
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            int posts0 = Pump.Instance.ForeignPosts;
            while (d.Kind == DecisionKind.Stuck && d.Context.StartsWith("战斗中但不在出牌阶段") && sw.ElapsedMilliseconds < 10_000)
            {
                Thread.Sleep(sw.ElapsedMilliseconds < 50 ? 0 : 2);
                d = GetDecisionOnce();
            }
            if (sw.ElapsedMilliseconds >= 100 || d.Kind == DecisionKind.Stuck)
                Console.Error.WriteLine($"[wait] 等出牌阶段 {sw.ElapsedMilliseconds} ms，外线程续延 {Pump.Instance.ForeignPosts - posts0} 条，结果 {d.Kind}（{d.Context}）；"
                    + $"开局任务 {_setup?.Status}{(_setup?.Exception is { } ex ? " " + ex.GetBaseException().Message : "")}，战斗进行中 {CombatManager.Instance.IsInProgress}，"
                    + $"正在开始 {CombatManager.Instance.IsStarting}，敌方回合已开始 {CombatManager.Instance.IsEnemyTurnStarted}，进行中的动作 {_inflight.Count}"
                    + (d.Kind == DecisionKind.Stuck ? StuckDetail() : ""));
        }
        return d;
    }

    /// <summary>卡住时补充的现场：回合循环停在哪里、遭遇与回合数、楼层、这一局怎么开的。</summary>
    string StuckDetail()
    {
        var loop = HarmonyLib.AccessTools.Field(typeof(CombatManager), "_turnLoopTask").GetValue(CombatManager.Instance) as Task;
        var cs = Me.Creature.CombatState;
        var loopState = loop == null ? "无" : loop.IsFaulted ? $"已出错（{loop.Exception!.GetBaseException()}）" : loop.Status.ToString();
        return $"；回合循环 {loopState}，停在：{AwaitChain(loop)}；遭遇 {cs?.Encounter?.Id.Entry}，第 {Me.PlayerCombatState?.TurnNumber} 回合，"
            + $"第 {State.CurrentActIndex + 1} 幕第 {State.TotalFloor} 层；{_origin}";
    }

    Decision GetDecisionOnce()
    {
        Pump.Instance.RunUntilIdle();
        var s = State;
        if (Fault() is { } why) return new Decision(DecisionKind.Stuck) { Context = why };
        // 只打一场战斗的模式：战斗没开起来（停在“正在开始”）时不能当成战斗结束、记为胜利
        if (CombatOnly && !CombatManager.Instance.IsInProgress && CombatManager.Instance.IsStarting)
            return new Decision(DecisionKind.Stuck) { Context = "战斗中但不在出牌阶段（战斗停在正在开始）" };
        // 连打：这一场打赢了、还有下一场，就带着当前的玩家状态进下一场（不领这场的奖励）
        // 血量为 0 但还没判死（同归于尽、结算还没跑完）时不算打赢
        if (CombatOnly && _nextEncounter != null && !CombatManager.Instance.IsInProgress && !Me.Creature.IsDead && Me.Creature.CurrentHp > 0 && !s.IsGameOver)
        {
            var enc = ModelDb.GetById<EncounterModel>(new ModelId("ENCOUNTER", _nextEncounter.Replace("ENCOUNTER.", "")));
            _nextEncounter = null; Stage++;
            // 版本 15：第二场的头部 16 为刚打赢的第一场，17 为这一场
            Observation.BossPrev = s.CurrentRoom is CombatRoom { RoomType: RoomType.Boss } br0 ? Vocab.Id(br0.Encounter) : 0; Observation.BossNext = 0;
            RewardsFlow.Reset(); FakeScreens.Reset();
            var t = Pump.Instance.Start(() => RunManager.Instance.EnterRoomDebug(enc.RoomType, model: enc.ToMutable(), showTransition: false));
            Pump.Instance.RunUntilIdle();
            if (t.IsFaulted) return new Decision(DecisionKind.Stuck) { Context = "连打的下一场开不起来：" + t.Exception!.GetBaseException().Message };
            // 下一场一开打就结束（带着很少的血进场、开场就被打死）照常按结束处理；只有既没在打、又没结束才算卡住
            if (!CombatManager.Instance.IsInProgress && !IsOver)
                return new Decision(DecisionKind.Stuck) { Context = $"连打的下一场开打后战斗没有在进行（任务 {t.Status}，正在开始 {CombatManager.Instance.IsStarting}，"
                    + $"房间 {State.CurrentRoom?.GetType().Name}，玩家血量 {Me.Creature.CurrentHp}，已死 {Me.Creature.IsDead}，结束 {State.IsGameOver}）" };
            return GetDecisionOnce();
        }
        if (IsOver) return new Decision(DecisionKind.GameOver) { Context = IsVictory ? "胜利" : "失败" };

        if (Selector.Pending is { } pc) return CardSelectDecision(pc);
        if (FakeScreens.CardReward is { } cr) return CardRewardDecision(cr);
        if (FakeScreens.Bundle is { } b) return BundleDecision(b);
        if (RewardsFlow.Pending is { } pr) return RewardsDecision(pr);
        if (FakeScreens.CrystalSphere is { IsFinished: false } cs) return CrystalSphereDecision(cs);
        if (CombatManager.Instance.IsInProgress) return CombatDecision();

        if (!ReferenceEquals(s.CurrentRoom, _room))
        {
            _room = s.CurrentRoom;
            _roomLeft = _restChosen = _chestOpened = _relicDecided = _shopInitialized = false;
            _failed.Clear();
        }
        return s.CurrentRoom switch
        {
            // 假商人的买卖全在界面节点里（NFakeMerchant），事件本身没有选项、一开始就是结束状态
            EventRoom { CanonicalEvent: FakeMerchant } when !_roomLeft && RunManager.Instance.EventSynchronizer.GetLocalEvent() is FakeMerchant { StartedFight: false } fm
                => FakeMerchantDecision(fm),
            EventRoom when RunManager.Instance.EventSynchronizer.GetLocalEvent() is { IsFinished: false } ev => EventDecision(ev),
            RestSiteRoom when RestDecision() is { } d => d,
            MerchantRoom m when !_roomLeft => ShopDecision(m),
            TreasureRoom t when TreasureDecision(t) is { } d => d,
            _ => MapDecision(),
        };
    }

    public void Choose(Decision d, int index)
    {
        Steps++;
        if (LoopCut || StallTurns > 0) { _lastLabels.Enqueue($"{d.Kind}:{d.Choices[index].Label}"); while (_lastLabels.Count > 8) _lastLabels.Dequeue(); }
        var t = Pump.Instance.Start(d.Choices[index].Run);
        if (t.IsFaulted) throw new InvalidOperationException($"执行“{d.Choices[index].Label}”出错", t.Exception);
        if (!t.IsCompleted) _inflight.Add((t, d.Choices[index].Label));
        AfterChoice();
    }

    /// <summary>
    /// 跳过稳赢的战斗（reports/跳过稳赢战斗方案.md）：整局里战斗的顶层出牌决策时，杀光敌人，让游戏照常走胜利流程
    /// （战斗结束时触发的效果、奖励界面都不变）；等这些跑完，再把血量设成给定值（截在 1 到最大生命之间），
    /// 按 ID 各去掉一瓶给定的药水（走游戏的丢弃动作）。血量放到最后设：训练标签量的就是战斗结束后第一帧的血量，已含战斗结束时的回血等效果。
    /// 杀敌时触发的死亡效果照常发生；死后分裂、召唤的敌人循环杀到没有为止（至多 8 轮）。只用于整局模式，不改其他请求的行为。
    /// </summary>
    public void Resolve(Decision d, int hp, List<string> potionIds)
    {
        if (CombatOnly) throw new InvalidOperationException("resolve 只用于整局模式");
        if (d.Kind != DecisionKind.Combat || !CombatManager.Instance.IsInProgress || Selector.Pending != null)
            throw new InvalidOperationException($"resolve 只能在战斗的顶层出牌决策时用（当前 {d.Kind}，战斗中 {CombatManager.Instance.IsInProgress}）");
        Steps++;
        for (int round = 0; round < 8 && CombatManager.Instance.IsInProgress; round++)
        {
            var cs = Me.Creature.CombatState;
            var foes = cs == null ? new List<Creature>() : cs.Creatures.Where(c => !c.IsPlayer && c.Side != Me.Creature.Side && c.IsAlive).ToList();
            foreach (var f in foes)
            {
                if (!f.IsAlive || !CombatManager.Instance.IsInProgress) continue;
                var kill = Pump.Instance.Start(() => CreatureCmd.Kill(f, force: true));
                Pump.Instance.RunUntilIdle();
                if (kill.IsFaulted) throw new InvalidOperationException("resolve 杀敌时出错：" + kill.Exception!.GetBaseException().Message);
            }
            // CreatureCmd.Kill(force: true) 不检查胜负（游戏里另有 KillWithoutCheckingWinCondition 一路），补一次胜负检查，同开发控制台的伤害命令
            var win = Pump.Instance.Start(() => CombatManager.Instance.CheckWinCondition());
            Pump.Instance.RunUntilIdle();
            if (win.IsFaulted) throw new InvalidOperationException("resolve 检查胜负时出错：" + win.Exception!.GetBaseException().Message);
        }
        if (CombatManager.Instance.IsInProgress) throw new InvalidOperationException("resolve 杀了 8 轮敌人，战斗仍未结束");
        AfterChoice();
        if (Me.Creature.IsDead || State.IsGameOver) return;     // 死亡效果把玩家打死了：照常按结束处理
        Me.Creature.SetCurrentHpInternal(Math.Clamp(hp, 1, Me.Creature.MaxHp));
        foreach (string raw in potionIds)
        {
            string id = raw.StartsWith("POTION.") ? raw["POTION.".Length..] : raw;
            var slots = Me.PotionSlots;
            int idx = -1;
            for (int i = 0; i < slots.Count; i++)
                if (slots[i] is { } p && p.Id.Entry == id) { idx = i; break; }
            if (idx < 0) continue;      // 战斗结束时的效果已经把这瓶用掉或换掉了
            uint slot = (uint)idx;
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new DiscardPotionGameAction(Me, slot, false));
            Pump.Instance.RunUntilIdle();
        }
    }

    string? Fault()
    {
        _inflight.RemoveAll(x => x.Task.IsCompletedSuccessfully);
        foreach (var (t, label) in _inflight)
            if (t.IsFaulted) return $"执行“{label}”出错：{t.Exception!.GetBaseException().Message}";
        if (GodotShim.ErrorTotal != _errorBase) return "游戏报错：" + GodotShim.LastError;
        return null;
    }

    /// <summary>终结奖励界面离开后，执行界面上“继续”按钮的逻辑（NRewardsScreen.OnProceedButtonPressed）。</summary>
    void AfterChoice()
    {
        Pump.Instance.RunUntilIdle();
        if (RewardsFlow.TerminalProceedRoom is not { } room) return;
        RewardsFlow.TerminalProceedRoom = null;
        var s = State;
        var t = Pump.Instance.Start(async () =>
        {
            if (room.RoomType == RoomType.Boss || room.IsVictoryRoom)
            {
                if (s.Map.SecondBossMapPoint != null && s.CurrentMapCoord == s.Map.BossMapPoint.coord)
                    await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync();
                else
                    RunManager.Instance.ActChangeSynchronizer.SetLocalPlayerReady();
            }
            else await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync();
        });
        if (t.IsFaulted) throw t.Exception!;
    }

    // ---------------------------------------------------------------- 各类决策

    // 选牌界面拆成逐张选：_selecting 是当前界面，_selected 是已选下标
    PendingCardChoice? _selecting;
    readonly List<int> _selected = new();
    /// <summary>观测用：当前选牌界面与已选的下标。</summary>
    internal PendingCardChoice? Selecting => _selecting;
    internal IReadOnlyList<int> SelectedIndices => _selected;

    Decision CardSelectDecision(PendingCardChoice pc)
    {
        if (!ReferenceEquals(pc, _selecting)) { _selecting = pc; _selected.Clear(); }
        int max = Math.Min(pc.Max, pc.Options.Count);
        int min = Math.Min(pc.Min, max);
        // 从牌组里选牌升级（休息处锻造、事件等，CardSelectCmd.FromDeckForUpgrade）：只要有能升级的牌，就不许取消、至少选一张
        // （2026-10-26 用户决定。原版界面可以关掉选牌、回到休息处改选别的，回放里 11 次升级选牌有 4 次这样空走一趟）
        bool upgrade = pc.Info?.Method == "FromDeckForUpgrade" && max > 0;
        if (upgrade) min = Math.Max(min, 1);
        bool cancel = pc.Cancelable && !upgrade && _selected.Count == 0 && !ReferenceEquals(_cancelRoom, State.CurrentRoom);
        // 没得选（上限为 0，或必须全选且不能取消）时直接提交，不占一个决策
        if (max == 0 || (min == max && max == pc.Options.Count && !cancel))
        {
            pc.Resolve(Enumerable.Range(0, max).ToList());
            return GetDecision();
        }
        var d = new Decision(DecisionKind.CardSelect) { Context = $"选 {min}..{max} 张，已选 {_selected.Count}", Selected = _selected.Count };
        // 升级类选牌（牌组里选牌升级、战斗中从手牌选牌升级等）：候选的对象 B 是这张牌升级后的样子（观测格式版本 9，同游戏界面的升级预览）
        bool upgradeAny = pc.Info?.Method is { } meth && meth.Contains("ForUpgrade");
        for (int i = 0; i < pc.Options.Count; i++)
        {
            if (_selected.Contains(i)) continue;
            int k = i;
            d.Add(ActionKind.SelectCard, "选 " + pc.Options[i].Id.Entry, () =>
            {
                _selected.Add(k);
                if (_selected.Count >= max) pc.Resolve(_selected.ToList());
            }, a: pc.Options[i], b: upgradeAny ? Observation.UpgradedCopy(pc.Options[i]) : null);
        }
        if (_selected.Count >= min) d.Add(ActionKind.ConfirmSelection, "确认", () => pc.Resolve(_selected.ToList()));
        // 关闭按钮（如锻造、删牌界面）：返回空集，调用方按“没选”处理
        if (cancel) d.Add(ActionKind.Leave, "取消", () => { _cancelRoom = State.CurrentRoom; pc.Resolve(Array.Empty<int>()); });
        return d;
    }

    /// <summary>卡牌奖励里的牌作为“看得到但不是候选”的记录（版本 18），区域为这个奖励所在的组。</summary>
    internal static void ShowRewardCards(Decision d, RewardsSet set, CardReward cr)
    {
        var z = Observation.RewardZone(set, cr);
        foreach (var c in cr.Cards) d.Shown.Add((c, z, 0));
    }

    static Decision CardRewardDecision(PendingCardReward cr)
    {
        var d = new Decision(DecisionKind.CardReward);
        // 版本 18（决策记录 210）：同一组里其余没领的奖励照样在观测里：别的卡牌奖励（选项记录与它的牌，分组区域）、药水、遗物、金币
        if (RewardsFlow.ActiveSet is { } set)
            foreach (var r in RewardsFlow.Unclaimed(set, RewardsFlow.ActiveAttempted, RewardsFlow.ActiveChoice))
            {
                d.Shown.Add((r, Zone.Offer, 0));
                if (r is CardReward other) ShowRewardCards(d, set, other);
            }
        for (int i = 0; i < cr.Cards.Count; i++) { int k = i; d.Add(ActionKind.PickCardReward, "拿 " + cr.Cards[i].Card.Id.Entry, () => cr.Resolve(k), a: cr.Cards[i].Card); }
        for (int i = 0; i < cr.Alternatives.Count; i++)
        {
            int k = cr.Cards.Count + i;
            d.Add(ActionKind.CardRewardAlt, cr.Alternatives[i].OptionId, () => cr.Resolve(k), key: Vocab.Alt(cr.Alternatives[i].OptionId));
        }
        return d;
    }

    static Decision BundleDecision(PendingBundle b)
    {
        var d = new Decision(DecisionKind.Bundle);
        for (int i = 0; i < b.Bundles.Count; i++)
        {
            int k = i;
            // 界面上两包的牌全部亮出（NChooseABundleSelectionScreen），按包分到区域“第 1 包”“第 2 包”
            foreach (var c in b.Bundles[i]) d.Shown.Add((c, i == 0 ? Zone.Pack1 : Zone.Pack2, 0));
            d.Add(ActionKind.PickBundle, "卡包 " + string.Join("+", b.Bundles[i].Select(c => c.Id.Entry)), () => b.Resolve(k), a: b.Bundles[i].FirstOrDefault(), f0: i);
        }
        return d;
    }

    /// <summary>Crystal Sphere：每次占卜选工具和格子（NCrystalSphereScreen 的 SetTool + CellClicked）。小工具揭开一格，大工具揭开 3×3。</summary>
    static Decision CrystalSphereDecision(MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame m)
    {
        var d = new Decision(DecisionKind.Minigame) { Context = $"Crystal Sphere 剩余 {m.DivinationCount} 次" };
        int w = m.GridSize.X, h = m.GridSize.Y;
        for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
        {
            var cell = m.cells[x, y];
            if (!cell.IsHidden) continue;   // NCrystalSphereCell：只有未揭开的格子能点，两种工具都一样
            d.Add(ActionKind.MinigameCell, $"小 ({x},{y})", () => { m.SetTool(MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType.Small); return m.CellClicked(cell); }, f0: x, f1: y, f2: 0);
            d.Add(ActionKind.MinigameCell, $"大 ({x},{y})", () => { m.SetTool(MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame.CrystalSphereToolType.Big); return m.CellClicked(cell); }, f0: x, f1: y, f2: 1);
        }
        return d;
    }

    /// <summary>选项悬浮提示里的第一张卡牌、第一件遗物或药水（选项本身附带的遗物除外）：界面上鼠标移到选项上就看得到，比如要失去的那张牌、换来的遗物。</summary>
    static object? HoverObject(MegaCrit.Sts2.Core.Events.EventOption opt)
    {
        if (opt.HoverTips == null) return null;
        foreach (var t in opt.HoverTips)
        {
            object? m = t is MegaCrit.Sts2.Core.HoverTips.CardHoverTip ct ? ct.Card
                : t is MegaCrit.Sts2.Core.HoverTips.HoverTip ht ? ht.CanonicalModel : null;
            if (m is CardModel or PotionModel || (m is RelicModel r && r.Id != opt.Relic?.Id)) return m;
        }
        return null;
    }

    static Decision RewardsDecision(PendingRewards pr)
    {
        var d = new Decision(DecisionKind.Rewards) { Context = pr.IsTerminal ? "战斗奖励" : "额外奖励" };
        var available = pr.Available;
        foreach (var r in available) d.Add(ActionKind.TakeReward, "领 " + RewardLabel(r), () => pr.Take(r), a: r);
        // 版本 18（决策记录 210）：每个没领的卡牌奖励的牌都亮出来，区域按它在这组奖励里的先后分到“第 1/2/3 组奖励”（同组的选项记录也在这个区域）
        foreach (var r in available) if (r is CardReward cr) ShowRewardCards(d, pr.Set, cr);
        // NeowsBones 等不许跳过（NRewardsScreen 禁用跳过按钮）：还有能领的就不能走
        if (!pr.Set.DisallowSkipping || available.Count == 0) d.Add(ActionKind.Leave, "离开", pr.Proceed);
        AddPotionActions(d);
        // 药水栏满、界面上还有没领的药水时，可以先丢一瓶腾位置（战斗结束和事件给的药水都走这个界面）
        if (!Me.HasOpenPotionSlots && pr.Set.Rewards.Any(r => r is PotionReward && !r.SuccessfullySelected))
        {
            AddPotionDiscards(d);
            // 栏满时这瓶药水领不了、不是候选，但界面上照样显示
            foreach (var r in pr.Set.Rewards) if (r is PotionReward { Potion: not null } p && !p.SuccessfullySelected) d.Shown.Add((p, Zone.Offer, 0));
        }
        return d;
    }

    static string RewardLabel(Reward r) => r switch
    {
        GoldReward => "金币",
        PotionReward p => "药水 " + p.Potion?.Id.Entry,
        RelicReward rr => "遗物 " + rr.Relic?.Id.Entry,
        CardReward => "卡牌奖励",
        _ => r.GetType().Name,
    };

    Decision CombatDecision()
    {
        var pcs = Me.PlayerCombatState;
        if (pcs == null || pcs.Phase != PlayerTurnPhase.Play) return new Decision(DecisionKind.Stuck) { Context = $"战斗中但不在出牌阶段（{pcs?.Phase}）" };
        var cs = Me.Creature.CombatState!;
        if (StallTurns > 0 && Stalled(pcs, cs))
        {
            var foes = cs.Creatures.Where(c => !c.IsPlayer && c.Side != Me.Creature.Side && c.IsAlive).Select(c => $"{c.Monster?.Id.Entry} {c.CurrentHp}/{c.MaxHp}+{c.Block}");
            Console.Error.WriteLine($"[stallcut] 第 {pcs.TurnNumber} 回合：连续 {_stallRun} 个回合敌人生命合计没有创新低（最低 {_stallMin}），判这场战斗输；遭遇 {cs.Encounter?.Id.Entry}，"
                + $"第 {State.CurrentActIndex + 1} 幕第 {State.TotalFloor} 层，玩家 {Me.Creature.CurrentHp}/{Me.Creature.MaxHp}，敌人 {string.Join("、", foes)}；手牌 "
                + string.Join(",", pcs.Hand.Cards.Select(c => c.Id.Entry)) + $"；最近几步 {string.Join(" / ", _lastLabels)}");
            var kill = Pump.Instance.Start(() => CreatureCmd.Kill(Me.Creature, force: true));
            Pump.Instance.RunUntilIdle();
            if (kill.IsFaulted) return new Decision(DecisionKind.Stuck) { Context = "跨回合无进展判输时出错：" + kill.Exception!.GetBaseException().Message };
            if (IsOver) return new Decision(DecisionKind.GameOver) { Context = "失败（跨回合无进展）" };
            Console.Error.WriteLine($"[stallcut] 判输后战斗仍未结束（玩家已死 {Me.Creature.IsDead}，结束 {State.IsGameOver}），照常继续");
        }
        var d = new Decision(DecisionKind.Combat) { Context = $"第 {pcs.TurnNumber} 回合" };
        var targets = new List<Creature?> { null };
        targets.AddRange(cs.Creatures.Where(c => c.IsAlive));
        foreach (CardModel c in PileType.Hand.GetPile(Me).Cards)
        {
            if (!c.CanPlay(out _, out _)) continue;
            foreach (var t in targets)
                if (c.CanPlayTargeting(t))
                {
                    var target = t;
                    d.Add(ActionKind.PlayCard, $"出 {c.Id.Entry}{TargetLabel(t)}", () => c.TryManualPlay(target), a: c, b: t);
                }
        }
        foreach (var p in UsablePotions(inCombat: true))
            foreach (var t in targets)
                if (p.IsValidTarget(t))
                {
                    var target = t;
                    d.Add(ActionKind.UsePotion, $"用 {p.Id.Entry}{TargetLabel(t)}", () => p.EnqueueManualUse(target), a: p, b: t);
                }
        int turn = pcs.TurnNumber;
        d.Add(ActionKind.EndTurn, "结束回合", () => RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(Me, turn)));
        if (LoopCut && LoopRepeated(pcs, cs))
        {
            var end = d.Choices[^1];
            d.Choices.Clear(); d.Choices.Add(end);
        }
        return d;
    }

    static string TargetLabel(Creature? t) => t == null ? "" : t.IsPlayer ? " → 自己" : $" → {t.Monster?.Id.Entry}";

    /// <summary>与 NPotionPopup 的可用条件一致。</summary>
    static IEnumerable<PotionModel> UsablePotions(bool inCombat) =>
        Me.CanUseOrRemovePotions && !Me.Creature.IsDead
            ? Me.Potions.Where(p => !p.IsQueued && p.PassesCustomUsabilityCheck &&
                                   (p.Usage == PotionUsage.AnyTime || (inCombat && p.Usage == PotionUsage.CombatOnly)))
            : Enumerable.Empty<PotionModel>();

    static void AddPotionDiscards(Decision d)
    {
        if (!Me.CanUseOrRemovePotions) return;
        var slots = Me.PotionSlots;
        for (int i = 0; i < slots.Count; i++)
            if (slots[i] is { IsQueued: false } p)
            {
                uint slot = (uint)i;
                d.Add(ActionKind.DiscardPotion, "丢弃 " + p.Id.Entry, () => RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                    new DiscardPotionGameAction(Me, slot, CombatManager.Instance.IsInProgress)), a: p);
            }
    }

    /// <summary>
    /// 战斗外使用“随时可用”的药水（NPotionPopup：顶栏药水随时可点；含在商店、假商人处扔 Foul Potion）。
    /// 加在奖励、事件、休息处、商店、宝箱、地图决策上；选牌、卡牌奖励等全屏选择界面上不加。
    /// 丢弃药水不随这里提供（2026-10-25 起，用户决定）：原版随时可丢，但模型几乎把领到的药水都在下一个地图决策丢掉，
    /// 现在只在需要腾位置领新药水时提供——奖励界面上有没领的药水（RewardsDecision）、商店里有买得起的药水（ShopDecision），且药水栏已满。
    /// </summary>
    static void AddPotionActions(Decision d)
    {
        foreach (var p in UsablePotions(inCombat: false))
            if (p.IsValidTarget(Me.Creature) || p.IsValidTarget(null))
                d.Add(ActionKind.UsePotion, "用 " + p.Id.Entry, () => p.EnqueueManualUse(null), a: p);
    }

    /// <summary>
    /// 不提供的事件选项（键为 TextKey），每一项写明原因。
    /// TRIAL 的 DOUBLE_DOWN：只弹出“放弃本局”确认框（NAbandonRunConfirmPopup），等于认输。
    /// </summary>
    static readonly HashSet<string> ExcludedEventOptions = new() { "TRIAL.pages.REJECT.options.DOUBLE_DOWN" };

    static Decision EventDecision(EventModel ev)
    {
        if (EventOptionTracker.Pending is { IsFaulted: true } f)
            return new Decision(DecisionKind.Stuck) { Context = $"事件 {ev.Id.Entry} 的选项执行出错：{f.Exception!.GetBaseException().Message}" };
        if (EventOptionTracker.Pending is { IsCompleted: false })
            return new Decision(DecisionKind.Stuck) { Context = $"事件 {ev.Id.Entry} 的选项仍在执行，却没有可做的决策" };
        var d = new Decision(DecisionKind.Event) { Context = ev.Id.Entry };
        var opts = ev.CurrentOptions;
        for (int i = 0; i < opts.Count; i++)
        {
            if (opts[i].IsLocked || ExcludedEventOptions.Contains(opts[i].TextKey)) continue;
            int k = i;
            var opt = opts[i];
            // NEventRoom.OptionButtonClicked：“继续”类选项直接调用 Chosen，不经过同步器
            var tip = HoverObject(opt);
            if (opt.IsProceed) d.Add(ActionKind.EventOption, opt.TextKey, () => opt.Chosen(), a: opt, b: tip);
            else d.Add(ActionKind.EventOption, opt.TextKey, () => RunManager.Instance.EventSynchronizer.ChooseLocalOption(k), a: opt, b: tip);
        }
        if (d.Choices.Count == 0) return new Decision(DecisionKind.Stuck) { Context = $"事件 {ev.Id.Entry} 没有可选项" };
        AddPotionActions(d);
        return d;
    }

    /// <summary>
    /// 确定化（搜索用，2026-10-27 起）：把战斗里以后要用到的随机数换成由 seed 派生的新随机数，并重新打乱抽牌堆。
    /// 眼前的局面（手牌、弃牌堆、敌人这一回合已定的意图等玩家看得到的东西）不变，只换玩家看不到的未来。
    /// 不换的：地图、宝箱、开局等与本场战斗无关的随机数，以及个别卡牌、怪物自带的随机数（为数很少）。
    /// 已知的局限：被效果明确放到牌堆顶的牌，玩家其实知道位置，这里也一并打乱了。
    /// </summary>
    /// <param name="keepTop">
    /// 保留已知的抽牌堆顶（2026-11-16，局面复制方案 E1，reseed 请求带 "keep_top":true 时）：被效果明确放到牌堆顶、玩家知道位置的那一段
    /// （Observation.DrawTop.Known，观测格式 20 起编进帧里的同一份）原地不动，只洗下面的部分。不带时与原来完全相同。
    /// </param>
    /// <summary>
    /// 战斗后的奖励也确定化（2026-11-17，直播树搜索不让推演看到真实的战后奖励）：玩家的随机数流（奖励、商店、变牌、局外生成）换成由 seed 派生的新随机数，
    /// 卡牌稀有度与药水掉落的几率对象改用新的奖励随机数，遗物袋各稀有度的顺序重新打乱。reseed 请求带 "rewards":true 时在 Reseed 之后调用；不带时不变。
    /// 原因：Reseed 只换整局随机数流，战斗打完时生成的金币、药水、精英遗物用的是玩家随机数与遗物袋，推演走到奖励帧时看到的是真实的那一份。
    /// </summary>
    internal static void ReseedRewards(uint seed)
    {
        const System.Reflection.BindingFlags F = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
        var prs = Me.PlayerRng;
        var dict = (System.Collections.Generic.Dictionary<MegaCrit.Sts2.Core.Entities.Rngs.PlayerRngType, MegaCrit.Sts2.Core.Random.Rng>)
            typeof(MegaCrit.Sts2.Core.Random.PlayerRngSet).GetField("_rngs", F)!.GetValue(prs)!;
        foreach (var k in dict.Keys.ToList())
            dict[k] = new MegaCrit.Sts2.Core.Random.Rng(unchecked((ulong)seed * 0x9E3779B97F4A7C15UL + (ulong)(int)k * 0x632BE59BD9B4E019UL + 0x5EEDUL));
        var rewards = dict[MegaCrit.Sts2.Core.Entities.Rngs.PlayerRngType.Rewards];
        var rngField = typeof(MegaCrit.Sts2.Core.Odds.AbstractOdds).GetField("_rng", F)!;
        rngField.SetValue(Me.PlayerOdds.CardRarity, rewards);
        rngField.SetValue(Me.PlayerOdds.PotionReward, rewards);
        var deques = (System.Collections.Generic.Dictionary<MegaCrit.Sts2.Core.Entities.Relics.RelicRarity, System.Collections.Generic.List<MegaCrit.Sts2.Core.Models.RelicModel>>)
            typeof(MegaCrit.Sts2.Core.Runs.RelicGrabBag).GetField("_deques", F)!.GetValue(Me.RelicGrabBag)!;
        uint i = 0;
        foreach (var kv in deques.OrderBy(kv => (int)kv.Key))
            new MegaCrit.Sts2.Core.Random.Rng(unchecked((ulong)seed ^ (0xA5A5A5A5UL + 7919UL * ++i))).Shuffle(kv.Value);
    }

    internal static void Reseed(uint seed, bool keepTop = false)
    {
        var rs = State.Rng;
        foreach (var t in Enum.GetValues<MegaCrit.Sts2.Core.Entities.Rngs.RunRngType>())
        {
            if (t is MegaCrit.Sts2.Core.Entities.Rngs.RunRngType.UpFront or MegaCrit.Sts2.Core.Entities.Rngs.RunRngType.UnknownMapPoint
                or MegaCrit.Sts2.Core.Entities.Rngs.RunRngType.TreasureRoomRelics) continue;
            rs.MockRng(t, unchecked(seed * 2654435761u + (uint)t * 40503u + 1u));
        }
        var pile = Me.PlayerCombatState?.DrawPile;
        if (pile == null || pile.Cards.Count < 2) return;
        var cards = pile.Cards.ToList();
        if (keepTop)
        {
            int known = Observation.DrawTop.Known(pile);
            cards = cards.Skip(known).ToList();
            if (cards.Count < 2) return;
        }
        new MegaCrit.Sts2.Core.Random.Rng(unchecked(seed ^ 0x9E3779B9u)).Shuffle(cards);
        foreach (var c in cards) pile.RemoveInternal(c, silent: true);
        foreach (var c in cards) pile.AddInternal(c, -1, silent: true);
    }

    /// <summary>
    /// 休息回血的预计（观测格式版本 8 起进休息选项的记录，也用于满血屏蔽）：实际能回的血（名义回血量与缺的血取小）、
    /// 名义回血量（HealRestSiteOption.GetHealAmount，含皇家枕头等加成）、休息附带的其他效果条数——给休息加了说明文字
    /// 又不改回血量的钩子（捕梦网、小邮箱、石头加湿器等），只改回血量的（皇家枕头）不算。
    /// </summary>
    internal static (int actual, int nominal, int extra) RestHealInfo(Player p)
    {
        var cr = p.Creature;
        int nominal = (int)HealRestSiteOption.GetHealAmount(p);
        int actual = Math.Clamp(nominal, 0, Math.Max(0, cr.MaxHp - cr.CurrentHp));
        int extra = 0;
        foreach (var m in p.RunState.IterateHookListeners(null))
        {
            int n = m.ModifyExtraRestSiteHealText(p, Array.Empty<MegaCrit.Sts2.Core.Localization.LocString>()).Count;
            if (n > 0 && m.ModifyRestSiteHealAmount(cr, 100m) == 100m) extra += n;
        }
        return (actual, nominal, extra);
    }

    /// <summary>
    /// 休息处：必须先成功选一项才能离开；没有可用选项时直接可走（NRestSiteRoom 的继续按钮规则）。
    /// 满血屏蔽（2026-10-27 起）：休息一点血都回不了、也没有附带效果、又能升级时，不给“休息”候选（严格劣于升级）。
    /// </summary>
    Decision? RestDecision()
    {
        if (_roomLeft) return null;
        var sync = RunManager.Instance.RestSiteSynchronizer;
        var opts = sync.GetLocalOptions();
        var d = new Decision(DecisionKind.RestSite);
        bool canSmith = opts.Any(o => o is SmithRestSiteOption && o.IsEnabled && !_failed.Contains(o));
        for (int i = 0; i < opts.Count; i++)
        {
            if (!opts[i].IsEnabled || _failed.Contains(opts[i])) continue;
            if (opts[i] is HealRestSiteOption && canSmith && RestHealInfo(Me) is { actual: 0, extra: 0 }) continue;
            int k = i;
            d.Add(ActionKind.RestOption, opts[i].OptionId, () => sync.ChooseLocalOption(k), a: opts[i]);
        }
        if (d.Choices.Count == 0) return null;
        if (_restChosen) d.Add(ActionKind.Leave, "离开", () => _roomLeft = true);   // 遗物允许多选时，选过一项后也可以直接走
        if (_restChosen && d.Choices.Count == 1) return null;
        AddPotionActions(d);
        return d;
    }

    Decision ShopDecision(MerchantRoom m)
    {
        var inv = RunManager.Instance.MerchantSynchronizer.GetLocalInventory();   // v0.113.0：货架从 MerchantRoom 挪到 MerchantSynchronizer（进店时 OnMerchantEntered 生成）
        // NMerchantCardRemoval：有遗物等禁止商店删牌时，进店即标记删牌已用
        if (!_shopInitialized)
        {
            _shopInitialized = true;
            if (inv.CardRemovalEntry is { } cre && !MegaCrit.Sts2.Core.Hooks.Hook.ShouldAllowMerchantCardRemoval(Me.RunState, Me)) cre.SetUsed();
        }
        var d = new Decision(DecisionKind.Shop) { Context = $"金币 {Me.Gold}" };
        AddShopEntries(d, inv);
        d.Add(ActionKind.Leave, "离开", () => _roomLeft = true);
        AddPotionActions(d);   // 含 Foul Potion 扔向商人（+100 金币）
        // 药水栏满、店里有买得起的药水时，可以先丢一瓶腾位置再买
        if (!Me.HasOpenPotionSlots && inv.AllEntries.Any(e => e is MerchantPotionEntry && e.IsStocked && e.EnoughGold && !_failed.Contains(e)))
            AddPotionDiscards(d);
        // 栏满时药水不能买、不是候选，但货架上照样摆着、标着价
        if (!Me.HasOpenPotionSlots)
            foreach (var e in inv.AllEntries) if (e is MerchantPotionEntry { Model: { } p } && e.IsStocked) d.Shown.Add((p, Zone.Offer, e.Cost));
        return d;
    }

    /// <summary>假商人（FakeMerchant 事件）：六个假遗物各 50 金币，扔 Foul Potion 开战。买卖照搬 NFakeMerchant 打开的商店界面，离开即走（HideScreen 打开地图）。</summary>
    Decision FakeMerchantDecision(FakeMerchant fm)
    {
        var d = new Decision(DecisionKind.Shop) { Context = $"假商人 金币 {Me.Gold}" };
        AddShopEntries(d, fm.Inventory);
        d.Add(ActionKind.Leave, "离开", () => _roomLeft = true);
        AddPotionActions(d);
        return d;
    }

    void AddShopEntries(Decision d, MerchantInventory inv)
    {
        foreach (MerchantEntry e in inv.AllEntries)
        {
            // 版本 19（决策记录 227）：买不起的商品不是候选，但记进观测（货架上看得到、标着价）
            if (e.IsStocked && !e.EnoughGold && !_failed.Contains(e) && e is not MerchantCardRemovalEntry { Used: true }) d.Locked.Add((EntryObject(e), e.Cost));
            if (!e.IsStocked || !e.EnoughGold || _failed.Contains(e)) continue;
            if (e is MerchantPotionEntry && !Me.HasOpenPotionSlots) continue;
            if (e is MerchantCardRemovalEntry removal)
            {
                // 删牌成功后游戏自己 SetUsed（v0.113.0 起；之前由 HeadlessFixes 补）；取消删牌也算失败，本次进店不再提供
                if (removal.Used) continue;
                d.Add(ActionKind.Buy, $"买 删牌 ({e.Cost})", async () => { if (await removal.OnTryPurchaseUnsynchronized(inv)) _failed.Clear(); else _failed.Add(removal); }, a: removal, f0: e.Cost);
                continue;
            }
            // 购买会失败但不扣钱（如有禁用药水的遗物、牌加不进牌组）；买成功后局面变了，之前失败的可以再试
            d.Add(ActionKind.Buy, $"买 {EntryLabel(e)} ({e.Cost})", async () => { if (await e.OnTryPurchaseUnsynchronized(inv)) _failed.Clear(); else _failed.Add(e); },
                  a: EntryObject(e), f0: e.Cost);
        }
    }

    /// <summary>商品对应的游戏对象：卡牌、遗物、药水本身，删牌为条目本身。</summary>
    static object EntryObject(MerchantEntry e) => e switch
    {
        MerchantCardEntry c => c.CreationResult.Card,
        MerchantRelicEntry { Model: { } r } => r,
        MerchantPotionEntry { Model: { } p } => p,
        _ => e,
    };

    static string EntryLabel(MerchantEntry e) => e switch
    {
        MerchantCardEntry c => c.CreationResult.Card.Id.Entry,
        MerchantRelicEntry r => r.Model?.Id.Entry ?? "遗物",
        MerchantPotionEntry p => p.Model?.Id.Entry ?? "药水",
        MerchantCardRemovalEntry => "删牌",
        _ => e.GetType().Name,
    };

    /// <summary>宝箱：开箱（NTreasureRoom.OpenChest 中的逻辑部分）后选一个遗物或跳过。</summary>
    Decision? TreasureDecision(TreasureRoom room)
    {
        if (!_chestOpened)
        {
            _chestOpened = true;
            var t = Pump.Instance.Start(async () =>
            {
                await room.DoNormalRewards();
                await room.DoExtraRewardsIfNeeded();
            });
            if (t.IsFaulted) throw t.Exception!;
            return GetDecision();   // 开箱可能弹出额外奖励
        }
        if (_relicDecided) return null;
        var sync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        var relics = sync.CurrentRelics;
        if (relics == null || relics.Count == 0)
        {
            // 空宝箱（如 SilverCrucible）：NTreasureRoomRelicCollection.AnimIn 中调用
            _relicDecided = true;
            if (relics != null) sync.CompleteWithNoRelics();
            return null;
        }
        // 宝箱里只有一件、且是无负面遗物：引擎直接拿（决策记录 162；多件可选时拿哪件仍是决策）
        if (RewardsFlow.AutoRelics && relics.Count == 1 && RewardsFlow.IsNoDownside(relics[0]))
        {
            _relicDecided = true;
            var pick = Pump.Instance.Start(async () => { sync.PickRelicLocally(0); await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync(); });
            if (pick.IsFaulted) throw pick.Exception!;
            if (!pick.IsCompleted) _inflight.Add((pick, "自动拿宝箱遗物 " + relics[0].Id.Entry));
            return GetDecision();
        }
        var d = new Decision(DecisionKind.Treasure);
        for (int i = 0; i < relics.Count; i++)
        {
            int k = i;
            d.Add(ActionKind.PickRelic, "拿 " + relics[i].Id.Entry, async () => { _relicDecided = true; sync.PickRelicLocally(k); await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync(); }, a: relics[i]);
        }
        d.Add(ActionKind.SkipRelic, "跳过", async () => { _relicDecided = true; sync.SkipRelicLocally(); await RunManager.Instance.ProceedFromTerminalRewardsScreenAsync(); });
        AddPotionActions(d);
        return d;
    }

    static IEnumerable<MapPoint> TravelablePoints(RunState s) => RunSessionMap.Travelable(s);

    Decision MapDecision()
    {
        var s = State;
        var cur = s.CurrentMapPoint;
        IEnumerable<MapPoint> next = TravelablePoints(s);
        var d = new Decision(DecisionKind.Map) { Context = $"第 {s.CurrentActIndex + 1} 幕 {(cur == null ? "起点" : $"({cur.coord.col},{cur.coord.row})")}" };
        foreach (var p in next.OrderBy(p => p.coord.col))
        {
            var dest = p;
            d.Add(ActionKind.MapMove, $"→ ({p.coord.col},{p.coord.row}) {p.PointType}", () =>
            {
                var vote = new MapVote { coord = dest.coord, mapGenerationCount = RunManager.Instance.MapSelectionSynchronizer.MapGenerationCount };
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new VoteForMapCoordAction(Me, s.MapLocation, vote));
            }, a: p);
        }
        if (d.Choices.Count == 0) return new Decision(DecisionKind.Stuck) { Context = $"地图上没有可去的节点（房间 {s.CurrentRoom?.GetType().Name}）" };
        AddPotionActions(d);
        return d;
    }
}

public static class RunSessionMap
{
    /// <summary>可前往的节点，规则照搬 NMapScreen.RecalculateTravelability。</summary>
    public static IEnumerable<MapPoint> Travelable(RunState s)
    {
        var map = s.Map;
        if (s.VisitedMapCoords.Count == 0) return new[] { map.StartingMapPoint };
        MapCoord last = s.VisitedMapCoords[^1];
        if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord) return new[] { map.SecondBossMapPoint };
        if (last.row == map.GetRowCount() - 1) return new[] { map.BossMapPoint };
        var point = map.GetPoint(last);
        return point == null ? new[] { map.StartingMapPoint } : MapTravel.GetTravelablePointsFrom(s, point);
    }
}
