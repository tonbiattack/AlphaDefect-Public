using System.Text.Json;

namespace Sts2Headless;

/// <summary>
/// 管道服务：从标准输入逐行读 JSON 请求，向标准输出写长度前缀的二进制响应（int32 字节数 + 内容）。日志只写标准错误。
/// 请求：
///   {"op":"reset","mode":"run","character":"IRONCLAD","ascension":10,"seed":"S1"}
///   {"op":"reset","mode":"combat","spec":{...CombatSpec...}}
///   {"op":"step","action":3}
///   {"op":"reseed","seed":7}  确定化：换掉以后要用的随机数、重洗抽牌堆（搜索用），返回当前决策的观测
///   {"op":"resolve","hp":40,"potions":["POTION.X"]}  跳过稳赢的战斗（reports/跳过稳赢战斗方案.md）：整局里战斗的顶层出牌决策时杀光敌人、走正常胜利流程，战后设血量、去药水
///   {"op":"branch","salt":7}  局外分枝的公平重设（Branch.Apply，决策记录 258）：换掉玩家看不到的整个未来（卡牌奖励、商店、遭遇、事件、后几幕），
///                             看得到的不变；返回当前决策的观测。reset 也可带 "branch"（在 replay 之后做），再带 "first" 就接着走那一步
///   {"op":"vocab"}            返回词表（UTF-8 文本，每行一个键）
///   {"op":"player"}           返回当前玩家快照与幕（UTF-8 JSON，牌组前瞻用）
///   {"op":"state"}            返回规范化的战斗状态与候选（UTF-8 JSON，只读；与 Rust 引擎对拍用，见 StateProbe）
///   {"op":"runinfo"}          返回各幕房间清单、当前幕地图与玩家概况（UTF-8 JSON，只读；整局随机数对拍用）
///   {"op":"save"}             返回游戏自己的整局存档（SerializableRun 的 JSON，只读；sts2-sim 移植快照用）
///   {"op":"snapshot","slot":0}  局面复制（Engine/Clone/局面复制.md）：把当前局面（上一个回复之后的样子）存进槽位，回复 UTF-8 JSON {"slot","objects","ms"}
///   {"op":"restore","slot":0,"reseed":7,"first":2}  恢复到槽位里的快照（快照不变，可反复恢复），再可带 replay / branch / reseed / first（语义同 reset；只带 first 时也走这一步），返回观测帧
///   {"op":"drop","slot":0}    丢掉一个槽位；不带 "slot" 时丢掉全部。回复 UTF-8 JSON {"dropped","slots"}
///   任何返回观测帧的请求都可带 "snapshot":槽位：回复之后把这时的局面存进槽位（推演进程跟随根局面时省一次往返）
///   任何返回观测帧的请求都可带 "auto_single":true 或整数 N：只有一个候选的决策由引擎自己走过（整数时最多 N 步），帧尾附加信息里 "auto_steps" 为走了几步（推演用）
///   reseed（单独的请求或 reset / restore 里的）可带 "keep_top":true：已知的抽牌堆顶（被效果放到顶部、玩家知道位置的牌）不洗
///   {"op":"close"}
/// reset / step 返回观测帧（Observation.Serialize）；"labels":true 时附带候选文字；整局 reset 带 "snapshots":true 时，
/// 选路线和战斗开始的观测帧附带玩家快照与战斗信息（Snapshot.Extra）；整局 reset 带 "route_summary":true 时，地图决策加路线摘要记录（Observation.RouteSummary）。
/// reset 带 "privileged":true 时输出特权特征（抽牌顺序、预抽的遭遇、怪物下一次抽签的推演，只给价值网络；不带时一律为 0，见 Observation.Privileged）。
/// 整局 reset 同时带 "privileged":true 与 "future_content":true 时，战斗之外的帧另有整局预生成内容的记录（类型 17，只给价值网络，见 Observation.EmitFuture）。
/// 整局 reset 同时带 "privileged":true 与 "oracle":true 时，战斗之外的帧另有预知记录（类型 20：近期卡牌奖励、药水、遗物袋、问号房的推算结果，只给价值网络，见 ObservationOracle.cs）。
/// reset 带 "loop_cut":true 时开局面重复截断（RunSession.LoopCut：同一回合里同一局面第 4 次出现、敌人没有明显掉血时只给“结束回合”）。
/// reset 带 "stall_turns":N 时开跨回合无进展判输（RunSession.StallTurns：连续 N 个玩家回合活着的敌人生命合计没有创新低、也没有敌人死亡，这场战斗判输；第二版见 RunSession.Stalled）。
/// reset 带 "obs_format":19–23 时按这个观测格式编码（不带时按角色查表：储君 20、静默猎手 23（2026-11-10 前为 21）、铁甲战士与亡灵契约师 22、其余角色 19，见 Observation.FormatFor）。
/// 无负面遗物（engine/catalog/no_downside_relics.txt）在奖励界面与宝箱里由引擎自动拿；reset 带 "auto_relics":false 时关闭（RewardsFlow.AutoRelics）。
/// 整局 reset 带 "auto_gold":false 时，金币奖励不再自动领，作为奖励界面的“领奖励”候选交给策略（RewardsFlow.AutoGold，不带时照旧自动领）。
/// 出错时返回 int32 -1 帧头 + UTF-8 错误信息，进程保持可用。
/// 现场记录（2026-10-27 起）：一局卡住或出错时，往标准错误（引擎日志）写一行 "[stuck-detail] " / "[error-detail] " + JSON：
/// 卡住原因、游戏报错全文（含调用栈）、这一局的 reset 请求原文、reset 之后依次走的动作（中途 reseed 记为 {"reseed": 种子}），
/// 照着重放就能复现（同一程序集下引擎是确定的）。每局最多写一次。
/// </summary>
public static class Server
{
    public static void Run()
    {
        var stdin = Console.OpenStandardInput();
        var stdout = Console.OpenStandardOutput();
        var reader = new StreamReader(stdin, System.Text.Encoding.UTF8);
        RunSession? run = null;
        Decision? decision = null;
        bool labels = false, snapshots = false, inCombat = false;
        string? line;
        var timing = new Dictionary<string, (double ms, int n)>();
        string? lastReset = null;               // 这一局的 reset 请求原文（现场记录用）
        var tape = new List<object>();          // reset 之后走过的动作
        bool detailLogged = false;
        void Detail(string tag, Dictionary<string, object?> what)
        {
            if (detailLogged) return;
            detailLogged = true;
            what["reset"] = lastReset; what["actions"] = tape;
            Console.Error.WriteLine($"[{tag}] " + JsonSerializer.Serialize(what, DetailJson));
        }
        void Time(string k, System.Diagnostics.Stopwatch sw) { var (t, n) = timing.GetValueOrDefault(k); timing[k] = (t + sw.Elapsed.TotalMilliseconds, n + 1); sw.Restart(); }
        // reset 与 restore 共用的后半段：replay、branch、reseed、first。record 为真时（restore）把它们记进现场记录的动作带，
        // 这样 reset 原文 + 动作带照样能复现恢复之后的局面（reset 的这几项本来就在 reset 原文里）
        void Tail(JsonElement req, RunSession run, bool record)
        {
            // 搜索用（tools/mcts_eval.py）：开局后按 "replay" 里的动作重放到当前局面（中间不编码观测），
            // 再按 "reseed" 换随机数、重洗抽牌堆，按 "first" 走一步，一次往返完成
            if (req.TryGetProperty("replay", out var rp) && rp.ValueKind == JsonValueKind.Array)
                foreach (var x in rp.EnumerateArray())
                {
                    var d = run.GetDecision();
                    if (d.Kind is DecisionKind.GameOver or DecisionKind.Stuck || d.Choices.Count == 0) break;
                    if (record) tape.Add(x.GetInt32());
                    run.Choose(d, x.GetInt32());
                }
            // 局外分枝（决策记录 258）：重放到局外决策点后公平重设未来，"first" 为这一分枝要走的选项
            if (req.TryGetProperty("branch", out var brs) && brs.ValueKind == JsonValueKind.Number)
            {
                var d = run.GetDecision();
                if (d.Kind is not (DecisionKind.GameOver or DecisionKind.Stuck) && d.Choices.Count > 0)
                {
                    Branch.Apply(brs.GetUInt32());
                    if (record) tape.Add(new Dictionary<string, uint> { ["branch"] = brs.GetUInt32() });
                    if (!req.TryGetProperty("reseed", out _) && req.TryGetProperty("first", out var bfs) && bfs.ValueKind == JsonValueKind.Number)
                    {
                        if (record) tape.Add(bfs.GetInt32());
                        run.Choose(run.GetDecision(), bfs.GetInt32());
                    }
                }
            }
            // restore 只带 first（不带 reseed、branch）时也走这一步（reset 照旧：first 只跟着 reseed 或 branch 生效）
            if (record && !req.TryGetProperty("reseed", out _) && !req.TryGetProperty("branch", out _) && req.TryGetProperty("first", out var onlyFirst)
                && onlyFirst.ValueKind == JsonValueKind.Number && run.GetDecision() is { Kind: not (DecisionKind.GameOver or DecisionKind.Stuck) } fd && fd.Choices.Count > 0)
            {
                tape.Add(onlyFirst.GetInt32());
                run.Choose(fd, onlyFirst.GetInt32());
            }
            if (req.TryGetProperty("reseed", out var rsd) && rsd.ValueKind == JsonValueKind.Number)
            {
                var d = run.GetDecision();
                if (d.Kind is not (DecisionKind.GameOver or DecisionKind.Stuck) && d.Choices.Count > 0)
                {
                    RunSession.Reseed(rsd.GetUInt32(), KeepTop(req));
                    if (ReseedRewardsOn(req)) RunSession.ReseedRewards(rsd.GetUInt32());
                    if (record) tape.Add(ReseedNote(rsd.GetUInt32(), req));
                    if (req.TryGetProperty("first", out var fst) && fst.ValueKind == JsonValueKind.Number)
                    {
                        if (record) tape.Add(fst.GetInt32());
                        run.Choose(run.GetDecision(), fst.GetInt32());
                    }
                }
            }
        }
        // reseed 是否连战后奖励也换（RunSession.ReseedRewards）：请求里 "rewards" 为 true/false 时照它；不写时看程序集旁有没有 reseed_rewards.on
        // （直播树搜索用的引擎目录放了这个文件，旧的策略端代码不用改；离线工具不放就和原来一样）
        static bool ReseedRewardsOn(JsonElement req) => req.TryGetProperty("rewards", out var r) && r.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? r.ValueKind == JsonValueKind.True : RewardsDefault;
        // reseed 是否保留已知的抽牌堆顶（"keep_top":true，见 RunSession.Reseed）；现场动作带里记成 {"reseed":种子,"keep_top":1}
        static bool KeepTop(JsonElement req) => req.TryGetProperty("keep_top", out var kt) && kt.ValueKind == JsonValueKind.True;
        static Dictionary<string, uint> ReseedNote(uint seed, JsonElement req) =>
            KeepTop(req) ? new() { ["reseed"] = seed, ["keep_top"] = 1 } : new() { ["reseed"] = seed };
        // 存快照：服务端的会话状态（当前局、当前决策、各开关、现场记录）随快照一起存，恢复时一起取回
        byte[] SnapshotReply(int slot)
        {
            var snap = HeapSnapshot.Capture(new ServeState(run, decision, labels, snapshots, inCombat, lastReset, tape, detailLogged));
            CloneSlots.Put(slot, snap);
            return System.Text.Encoding.UTF8.GetBytes($"{{\"slot\":{slot},\"objects\":{snap.Objects},\"ms\":{snap.CaptureMs:F3}}}");
        }
        while ((line = reader.ReadLine()) != null)
        {
            if (line.Length == 0) continue;
            byte[] reply;
            try
            {
                using var doc = JsonDocument.Parse(line);
                var req = doc.RootElement;
                string op = req.GetProperty("op").GetString()!;
                if (op == "close") break;
                if (op == "vocab")
                {
                    reply = System.Text.Encoding.UTF8.GetBytes(string.Join("\n", Vocab.AllKeys));
                }
                else if (op == "state")
                {
                    // 对拍用（Rust引擎方案.md）：规范化战斗状态与候选，UTF-8 JSON，只读
                    if (run == null) throw new InvalidOperationException("尚未 reset");
                    reply = StateProbe.Json(run, decision);
                }
                else if (op == "runinfo")
                {
                    if (run == null) throw new InvalidOperationException("尚未 reset");
                    reply = StateProbe.RunInfo();
                }
                else if (op == "save")
                {
                    // sts2-sim 对拍与搜索用（engine/Sts2SimServe/方案.md）：游戏自己的整局存档（RunManager.ToSave，同存档文件的内容），
                    // UTF-8 JSON，只读，不走一步；由 tools/sim_transplant.py 转成 sts2-sim 的移植快照
                    if (run == null) throw new InvalidOperationException("尚未 reset");
                    var save = MegaCrit.Sts2.Core.Runs.RunManager.Instance.ToSave(null);
                    string saveJson = JsonSerializer.Serialize(save, MegaCrit.Sts2.Core.Saves.JsonSerializationUtility.GetTypeInfo(save));
                    // 存档里没有、移植快照要的运行时状态：下一个房间编号（RunState.NextRoomId，每幕开头清零）
                    reply = System.Text.Encoding.UTF8.GetBytes($"{{\"save\":{saveJson},\"next_room_id\":{RunSession.State.NextRoomId}}}");
                }
                else if (op == "snapshot")
                {
                    // 局面复制：把当前局面（上一个回复之后的样子）存进槽位，回复 UTF-8 JSON {"slot","objects","ms"}
                    if (run == null || decision == null) throw new InvalidOperationException("尚未 reset");
                    reply = SnapshotReply(req.GetProperty("slot").GetInt32());
                }
                else if (op == "drop")
                {
                    // 局面复制：丢掉一个槽位（带 "slot"）或全部槽位（不带），释放内存
                    int dropped = req.TryGetProperty("slot", out var ds) ? (CloneSlots.Drop(ds.GetInt32()) ? 1 : 0) : CloneSlots.Count;
                    if (!req.TryGetProperty("slot", out _)) CloneSlots.Clear();
                    reply = System.Text.Encoding.UTF8.GetBytes($"{{\"dropped\":{dropped},\"slots\":{CloneSlots.Count}}}");
                }
                else if (op == "player")
                {
                    // 牌组前瞻用（tools/lookahead_eval.py）：当前玩家快照（同 Snapshot.Player）与当前幕，UTF-8 JSON，不走一步、不编码观测
                    if (run == null) throw new InvalidOperationException("尚未 reset");
                    var st = RunSession.State;
                    reply = System.Text.Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new Dictionary<string, object>
                    {
                        ["snapshot"] = Snapshot.Player(st.Players[0]), ["act"] = st.CurrentActIndex, ["floor"] = st.TotalFloor,
                    }));
                }
                else
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    if (op == "reset")
                    {
                        lastReset = line; tape.Clear(); detailLogged = false; GodotShim.LastErrorFull = "";   // 报错全文只留这一局的
                        run?.Dispose();
                        run = null; decision = null;   // 开局失败时不能留着已释放的旧会话
                        labels = req.TryGetProperty("labels", out var lb) && lb.GetBoolean();
                        string mode = req.TryGetProperty("mode", out var m) ? m.GetString()! : "run";
                        snapshots = mode != "combat" && req.TryGetProperty("snapshots", out var sn) && sn.GetBoolean();
                        inCombat = false;
                        Observation.RouteSummary = mode != "combat" && req.TryGetProperty("route_summary", out var rsum) && rsum.GetBoolean();
                        RewardsFlow.AutoGold = !(req.TryGetProperty("auto_gold", out var ag) && ag.ValueKind == JsonValueKind.False);
                        Observation.Privileged = req.TryGetProperty("privileged", out var pv) && pv.ValueKind == JsonValueKind.True;
                        Observation.FutureContent = req.TryGetProperty("future_content", out var fc) && fc.ValueKind == JsonValueKind.True;   // 整局预生成内容，只在 privileged 时有效（决策记录 259）
                        Observation.Oracle = req.TryGetProperty("oracle", out var orc) && orc.ValueKind == JsonValueKind.True;   // 预知记录（类型 20），只在 privileged 时有效（决策记录 280）
                        // 观测格式：不带时按角色查表（储君 20，静默猎手 23，铁甲战士 28，亡灵契约师 22，Fool 27，其余 19，见 Observation.FormatFor）；带 "obs_format" 时以它为准（核对用）
                        Observation.FormatOverride = req.TryGetProperty("obs_format", out var ofm) && ofm.ValueKind == JsonValueKind.Number ? ofm.GetInt32() : null;
                        // 局面重复截断（RunSession.LoopCut）：同一局面第 4 次出现、敌人没有明显掉血时只给结束回合
                        RunSession.LoopCut = req.TryGetProperty("loop_cut", out var lcut) && lcut.ValueKind == JsonValueKind.True;
                        // 跨回合无进展判输（RunSession.StallTurns）：连续这么多个玩家回合敌人生命合计没有创新低（也没有敌人死亡）就判这场战斗输
                        RunSession.StallTurns = req.TryGetProperty("stall_turns", out var stl) && stl.ValueKind == JsonValueKind.Number ? stl.GetInt32() : 0;
                        RewardsFlow.AutoRelics = !(req.TryGetProperty("auto_relics", out var ar) && ar.ValueKind == JsonValueKind.False);
                        ActStart.Export = snapshots; ActStart.Pending = null;
                        // 游戏模式："game_mode":"dread" 为恐惧模式（同恐惧线 d36b03f），不带或 "standard" 为标准模式
                        string gmode = req.TryGetProperty("game_mode", out var gmv) && gmv.ValueKind == JsonValueKind.String ? gmv.GetString()!.ToLowerInvariant() : "standard";
                        HeadlessRun.Mode = gmode switch
                        {
                            "standard" => MegaCrit.Sts2.Core.Runs.GameMode.Standard,
                            "dread" => MegaCrit.Sts2.Core.Runs.GameMode.Dread,
                            _ => throw new ArgumentException($"game_mode 只能是 standard 或 dread，收到 {gmode}"),
                        };
                        // 训练加血（HeadlessRun.BonusHp，铁甲战士决策记录第 24 条）："bonus_hp": N 时建局最大生命与当前生命各加 N；不带为 0
                        int bonusHp = req.TryGetProperty("bonus_hp", out var bhp) && bhp.ValueKind == JsonValueKind.Number ? bhp.GetInt32() : 0;
                        if (bonusHp is < 0 or > 500) throw new ArgumentException($"bonus_hp 须在 0–500，收到 {bonusHp}");
                        HeadlessRun.BonusHp = bonusHp;
                        // 整局可带 "act_start"（开幕快照，见 ActStart）：从那一幕开局
                        ActStartSnapshot? actStart = mode != "combat" && req.TryGetProperty("act_start", out var ast) && ast.ValueKind == JsonValueKind.Object
                            ? ast.Deserialize<ActStartSnapshot>(JsonOptions) : null;
                        run = mode == "combat"
                            ? RunSession.StartCombat(req.GetProperty("spec").Deserialize<CombatSpec>(JsonOptions)!)
                            : actStart != null
                                ? RunSession.StartFromAct(req.GetProperty("character").GetString()!, req.GetProperty("ascension").GetInt32(), req.GetProperty("seed").GetString()!, actStart)
                                : RunSession.Start(req.GetProperty("character").GetString()!, req.GetProperty("ascension").GetInt32(), req.GetProperty("seed").GetString()!);
                        Time("reset", sw);
                        Tail(req, run, record: false);
                        Time("replay", sw);
                    }
                    else if (op == "restore")
                    {
                        // 局面复制（Engine/Clone/局面复制.md）：恢复到槽位里的快照，之后可带 replay / branch / reseed / first，语义同 reset
                        var snap = CloneSlots.Get(req.GetProperty("slot").GetInt32());
                        snap.Restore();
                        ((ServeState)snap.Extra[0]!).Load(ref run, ref decision, ref labels, ref snapshots, ref inCombat, ref lastReset, ref tape, ref detailLogged);
                        Time("restore", sw);
                        Tail(req, run!, record: true);
                        Time("replay", sw);
                    }
                    else if (op == "step")
                    {
                        if (run == null || decision == null) throw new InvalidOperationException("尚未 reset");
                        int a = req.GetProperty("action").GetInt32();
                        if (a < 0 || a >= decision.Choices.Count) throw new ArgumentOutOfRangeException("action", $"共 {decision.Choices.Count} 个候选，收到 {a}");
                        tape.Add(a);
                        run.Choose(decision, a);
                        Time("step", sw);
                    }
                    else if (op == "reseed")
                    {
                        // 确定化（搜索用）：只换随机数、重洗抽牌堆，不走一步；返回的仍是当前决策的观测
                        if (run == null || decision == null) throw new InvalidOperationException("尚未 reset");
                        RunSession.Reseed(req.GetProperty("seed").GetUInt32(), KeepTop(req));
                        if (ReseedRewardsOn(req)) RunSession.ReseedRewards(req.GetProperty("seed").GetUInt32());
                        tape.Add(ReseedNote(req.GetProperty("seed").GetUInt32(), req));
                        Time("reseed", sw);
                    }
                    else if (op == "resolve")
                    {
                        // 跳过稳赢的战斗（reports/跳过稳赢战斗方案.md）：{"op":"resolve","hp":战后血量,"potions":["POTION.X",…]}，
                        // 只在整局里战斗的顶层出牌决策时可用；杀光敌人走正常胜利流程，之后设血量、去药水，返回下一个决策（奖励界面）
                        if (run == null || decision == null) throw new InvalidOperationException("尚未 reset");
                        int rhp = req.GetProperty("hp").GetInt32();
                        var rpots = new List<string>();
                        if (req.TryGetProperty("potions", out var rps) && rps.ValueKind == JsonValueKind.Array)
                            foreach (var x in rps.EnumerateArray()) rpots.Add(x.GetString()!);
                        run.Resolve(decision, rhp, rpots);
                        tape.Add(new Dictionary<string, object> { ["resolve"] = rhp, ["potions"] = rpots });
                        Time("resolve", sw);
                    }
                    else if (op == "branch")
                    {
                        if (run == null || decision == null) throw new InvalidOperationException("尚未 reset");
                        Branch.Apply(req.GetProperty("salt").GetUInt32());
                        tape.Add(new Dictionary<string, uint> { ["branch"] = req.GetProperty("salt").GetUInt32() });
                        Time("branch", sw);
                    }
                    else throw new ArgumentException("未知操作 " + op);
                    decision = run!.GetDecision();
                    // 只有一个候选的决策由引擎自己走过（局面复制方案 C2，请求带 "auto_single":true 时）：不编码、不回 Python，
                    // 走了几步写进帧尾附加信息的 "auto_steps"，也记进现场动作带（照样能复现）。不带时与原来完全相同
                    // 可以是 true（走到有多个候选为止），也可以是整数 N（最多自动走 N 步：推演传“离叶子估值还剩几步”，计步口径与逐步走完全相同）
                    int autoCap = !req.TryGetProperty("auto_single", out var aus) ? 0
                        : aus.ValueKind == JsonValueKind.True ? 100_000 : aus.ValueKind == JsonValueKind.Number ? Math.Max(0, aus.GetInt32()) : 0;
                    bool autoSingle = aus.ValueKind is JsonValueKind.True or JsonValueKind.Number;
                    int autoSteps = 0;
                    if (autoSingle)
                    {
                        while (decision.Kind is not (DecisionKind.GameOver or DecisionKind.Stuck) && decision.Choices.Count == 1 && autoSteps < autoCap)
                        {
                            tape.Add(0);
                            run.Choose(decision, 0);
                            decision = run.GetDecision();
                            autoSteps++;
                        }
                        Time("auto", sw);
                    }
                    if (decision.Kind == DecisionKind.Stuck)
                    {
                        Console.Error.WriteLine("[stuck] " + decision.Context);
                        Detail("stuck-detail", new() { ["context"] = decision.Context, ["error"] = Clip(GodotShim.LastErrorFull, 6000) });
                    }
                    Time("decision", sw);
                    var obs = Observation.Build(run, decision, labels);
                    if (snapshots) obs.Extra = Snapshot.Extra(decision, ref inCombat);
                    if (autoSingle) obs.Extra = obs.Extra is { Length: > 1 } ex ? $"{{\"auto_steps\":{autoSteps}," + ex[1..] : $"{{\"auto_steps\":{autoSteps}}}";
                    reply = obs.Serialize();
                    Time(decision.Kind == DecisionKind.Map ? "encode_map" : "encode", sw);
                    if (req.TryGetProperty("snapshot", out var snapSlot) && snapSlot.ValueKind == JsonValueKind.Number)
                    {
                        SnapshotReply(snapSlot.GetInt32());
                        Time("snapshot", sw);
                    }
                }
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e);
                if (lastReset != null) Detail("error-detail", new() { ["context"] = e.GetBaseException().Message, ["error"] = Clip(e.ToString(), 6000) });
                byte[] msg = System.Text.Encoding.UTF8.GetBytes(e.GetBaseException().Message);
                reply = new byte[4 + msg.Length];
                BitConverter.TryWriteBytes(reply, -1);
                msg.CopyTo(reply, 4);
            }
            stdout.Write(BitConverter.GetBytes(reply.Length));
            stdout.Write(reply);
            stdout.Flush();
        }
        if (System.Environment.GetEnvironmentVariable("STS2_TIMING") != null)
            foreach (var (k, (t, n)) in timing) Console.Error.WriteLine($"[timing] {k,-9} {n,7} 次  平均 {t / Math.Max(1, n):F3} ms  共 {t / 1000:F1} s");
    }

    // 程序集旁放了 reseed_rewards.on：reseed 默认连战后奖励也换（见 ReseedRewardsOn）
    static readonly bool RewardsDefault = File.Exists(Path.Combine(AppContext.BaseDirectory, "reseed_rewards.on"));
    static readonly JsonSerializerOptions JsonOptions = new() { PropertyNameCaseInsensitive = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    static readonly JsonSerializerOptions DetailJson = new() { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    static string Clip(string s, int n) => s.Length > n ? s[..n] + "…" : s;
}
