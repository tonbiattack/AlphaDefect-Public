namespace Sts2Headless;

/// <summary>
/// 服务端（Server.Run）的会话状态：当前局、当前决策、reset 时定下的开关、现场记录（reset 原文与之后的动作带）。
/// 拍快照时把它当额外的根一并存（Decision 里候选的闭包、动作带列表都随之进快照），恢复时整组取回，
/// 所以恢复之后的 step 用的是拍快照时那个决策对象，卡住日志的“reset 原文 + 动作带”照样能复现恢复后的局面。
/// </summary>
public sealed class ServeState(RunSession? run, Decision? decision, bool labels, bool snapshots, bool inCombat, string? lastReset, List<object> tape, bool detailLogged)
{
    readonly RunSession? _run = run;
    readonly Decision? _decision = decision;
    readonly bool _labels = labels, _snapshots = snapshots, _inCombat = inCombat, _detailLogged = detailLogged;
    readonly string? _lastReset = lastReset;
    readonly List<object> _tape = tape;

    public void Load(ref RunSession? run, ref Decision? decision, ref bool labels, ref bool snapshots, ref bool inCombat, ref string? lastReset, ref List<object> tape, ref bool detailLogged)
    {
        run = _run; decision = _decision; labels = _labels; snapshots = _snapshots; inCombat = _inCombat;
        lastReset = _lastReset; tape = _tape; detailLogged = _detailLogged;
    }
}
