using System.Reflection;
using System.Text.Json;
using HarmonyLib;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Runs.History;
using MegaCrit.Sts2.Core.Saves;

namespace Sts2Headless;

/// <summary>
/// 从中途局面开整局（技术路线第十四节，2026-10-26）。
/// 导出：整局（reset 带 "snapshots":true）换幕时，在游戏调用 RunManager.EnterAct(下一幕) 的那一刻记下“开幕快照”——
///   玩家快照（Snapshot.Player）、之前每幕走过的地图历史、已遇到的事件——下一帧观测里以 {"act_start": ...} 带给训练端。
///   取在调用之前而不是进幕之后：还原时再走同一个 EnterAct，进幕时的效果（AfterActEntered 等）不会触发两次。
/// 还原（reset 带 "act_start"）：用新种子建局，还原玩家，补地图历史（总层数由它推出）和已遇事件，
///   把已有遗物从遗物袋里去掉，再调用同一个 EnterAct(N) 接着打完整局（游戏自带的跳幕调试命令也走这条路）。
///   随机数、卡牌稀有度保底计数、幕的选择都用新种子自己的。
/// </summary>
public sealed class ActStartSnapshot
{
    /// <summary>要进入的幕（0 起，1 = 第二幕）。</summary>
    public int Act { get; set; }
    /// <summary>玩家快照（Snapshot.Player 的 JSON 文本）。</summary>
    public string Player { get; set; } = "";
    /// <summary>之前每幕的地图历史（游戏存档格式的 JSON 文本，List&lt;List&lt;MapPointHistoryEntry&gt;&gt;）。</summary>
    public string History { get; set; } = "[]";
    /// <summary>已遇到的事件（游戏存档格式的 JSON 文本，List&lt;ModelId&gt;）。</summary>
    public string Events { get; set; } = "[]";
}

public static class ActStart
{
    /// <summary>当前这局要不要导出开幕快照（Server 按 reset 的 "snapshots" 设置）。</summary>
    public static bool Export;
    /// <summary>导出后还没带出去的开幕快照（JSON 文本），Snapshot.Extra 取走后清空。</summary>
    public static string? Pending;
    static bool _restoring;

    static readonly FieldInfo HistoryField = GameCompat.Field(typeof(RunState), "_mapPointHistory", "开幕快照的路线历史")!;
    static readonly FieldInfo EventsField = GameCompat.Field(typeof(RunState), "_visitedEventIds", "开幕快照的已见事件")!;

    public static void Install(Harmony h) =>
        GameCompat.Patch(h, GameCompat.Method(typeof(RunManager), nameof(RunManager.EnterAct), "开幕快照"), "RunManager.EnterAct", prefix: new HarmonyMethod(typeof(ActStart), nameof(BeforeEnterAct)));

    static void BeforeEnterAct(int currentActIndex)
    {
        if (!Export || _restoring || currentActIndex < 1) return;
        var s = RunManager.Instance.DebugOnlyGetState();
        if (s == null) return;
        Pending = JsonSerializer.Serialize(Take(s, currentActIndex));
    }

    /// <summary>当前整局的开幕快照（act 为将要进入的幕）。</summary>
    public static ActStartSnapshot Take(RunState s, int act) => new()
    {
        Act = act,
        Player = Snapshot.Player(s.Players[0]),
        History = JsonSerializer.Serialize(s.MapPointHistory.Take(act).Select(a => a.ToList()).ToList(), JsonSerializationUtility.Options),
        Events = JsonSerializer.Serialize(s.VisitedEventIds.ToList(), JsonSerializationUtility.Options),
    };

    /// <summary>按开幕快照开一局：建局、还原玩家与整局记录、进入那一幕。</summary>
    public static async Task Start(string characterId, int ascension, string seed, ActStartSnapshot a)
    {
        await HeadlessRun.CreateRun(characterId, ascension, seed);
        var s = RunManager.Instance.DebugOnlyGetState()!;
        if (a.Act < 1 || a.Act >= s.Acts.Count) throw new ArgumentException($"开幕快照的幕号 {a.Act} 不在 1–{s.Acts.Count - 1}");
        var me = s.Players[0];
        Snapshot.Restore(me, a.Player);
        var hist = JsonSerializer.Deserialize<List<List<MapPointHistoryEntry>>>(a.History, JsonSerializationUtility.Options) ?? new();
        var list = (List<List<MapPointHistoryEntry>>)HistoryField.GetValue(s)!;
        list.Clear();
        list.AddRange(hist.Take(a.Act));
        while (list.Count < a.Act) list.Add(new());
        var events = (HashSet<ModelId>)EventsField.GetValue(s)!;
        foreach (var id in JsonSerializer.Deserialize<List<ModelId>>(a.Events, JsonSerializationUtility.Options) ?? new())
            events.Add(id);
        // 已有的遗物不能再从遗物袋里抽到（正常整局里获得时就从袋里拿掉了）
        foreach (var r in me.Relics)
        {
            me.RelicGrabBag.Remove(r);
            s.SharedRelicGrabBag.Remove(r);
        }
        _restoring = true;
        try { await RunManager.Instance.EnterAct(a.Act, doTransition: false); }
        finally { _restoring = false; }
    }
}
