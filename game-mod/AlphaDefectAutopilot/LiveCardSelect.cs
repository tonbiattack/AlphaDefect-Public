using System.Reflection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// カード選択（捨てる・強化する・山札から選ぶ等）を AI に選ばせる。
///
/// ゲームには「自動で選ぶ選牌器」の口が用意されている（CardSelectCmd.Selector、
/// AutoSlay などが使う）。無頭エンジンと同じ AgentCardSelector をその口に差し込み、
/// 選牌待ちを観測の選牌プロンプトとしてエンコードし、選んだ添字で解決する。
///
/// 選牌のフレームは DecisionKind.CardSelect。1 枚ずつ選び、最後に確定する
/// （エンジンの RunSession.CardSelectDecision と同じ形）。
/// </summary>
internal static class LiveCardSelect
{
    static readonly List<int> _selected = new();
    static PendingCardChoice? _selecting;
    static RefusalRoom? _refused;          // 同じ部屋でキャンセルを何度も出さないため
    static readonly HashSet<object> _logged = new();
    static string _lastSig = "";
    static int _sameSig;

    record RefusalRoom(object Room);

    public static AgentCardSelector Selector { get; } = new();

    /// <summary>CardSelectCmd の選牌器スタックへ差し込む（ゲームの自動選牌の口を使う）。</summary>
    public static void Install()
    {
        LiveSelector.Instance = Selector;
        try
        {
            var field = typeof(CardSelectCmd).GetField("_selectorStack", BindingFlags.Static | BindingFlags.NonPublic);
            if (field?.GetValue(null) is Stack<ICardSelector> stack)
            {
                stack.Push(Selector);
                Bridge.LogLine("card selector installed");
            }
            else
            {
                Bridge.LogLine("card selector stack not found; in-combat card choices will wait for the player");
            }
        }
        catch (Exception e)
        {
            Bridge.LogLine("card selector install failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    public static void Reset()
    {
        _selected.Clear();
        _selecting = null;
        _refused = null;
        LiveSelector.Selected = Array.Empty<int>();
    }

    public static bool Active => Selector.Pending != null;

    /// <summary>選牌の決定。無ければ null。</summary>
    public static Decision? Build(RunState run)
    {
        var pc = Selector.Pending;
        if (pc == null) return null;

        // 同じ選牌が何度も来る（選んでも解決しない）なら、待ち続けずに確定して先へ進める。
        // 選牌の解決が画面依存で返らないことがあり、そのままだと 1 ランが止まる。
        string sig = pc.Min + "/" + pc.Max + "/" + string.Join(",", pc.Options.Select(o => o.Id.Entry));
        if (sig == _lastSig) _sameSig++; else { _lastSig = sig; _sameSig = 0; }
        if (_sameSig >= 6)
        {
            var pick = _selected.Count > 0 ? _selected.ToList() : new List<int> { 0 };
            _selected.Clear(); _selecting = null; LiveSelector.Selected = Arr();
            try
            {
                pc.Resolve(pick);
                Bridge.LogLine("card select force-resolved after repeated prompts: " + string.Join(",", pick));
            }
            catch (Exception e)
            {
                Bridge.LogLine("card select force-resolve failed: " + e.GetType().Name);
            }
            _sameSig = 0; _lastSig = "";
            return null;
        }
        if (!ReferenceEquals(pc, _selecting)) { _selecting = pc; _selected.Clear(); LiveSelector.Selected = Arr(); }
        int max = Math.Min(pc.Max, pc.Options.Count);
        int min = Math.Min(pc.Min, max);
        // デッキから選んで強化（休息所の鍛造など）は、選べるなら 1 枚は選ぶ
        bool upgrade = pc.Info?.Method == "FromDeckForUpgrade" && max > 0;
        if (upgrade) min = Math.Max(min, 1);
        bool cancel = pc.Cancelable && !upgrade && _selected.Count == 0 && !ReferenceEquals(_refused?.Room, run.CurrentRoom);
        // 選ぶ余地が無い（0 枚、または全部選ぶしかない）ならその場で解決して次の判断へ
        if (max == 0 || (min == max && max == pc.Options.Count && !cancel))
        {
            pc.Resolve(Enumerable.Range(0, max).ToList());
            Bridge.LogLine($"card select auto-resolved ({max} cards)");
            _selected.Clear(); _selecting = null; LiveSelector.Selected = Arr();
            return null;
        }
        var d = new Decision(DecisionKind.CardSelect)
        {
            Context = $"选 {min}..{max} 张，已选 {_selected.Count}",
            Selected = _selected.Count,
        };
        if (!_logged.Contains(pc))
        {
            _logged.Add(pc);
            // 誰が選牌を求めているか（繰り返しの原因を追う）。スタックの上位数フレームだけ。
            string where = "";
            try
            {
                var st = new System.Diagnostics.StackTrace(true);
                where = string.Join(" < ", st.GetFrames()
                    .Where(f => f.GetMethod()?.DeclaringType?.Assembly == typeof(LiveCardSelect).Assembly
                             || f.GetMethod()?.DeclaringType?.Namespace?.StartsWith("MegaCrit") == true)
                    .Take(4)
                    .Select(f => $"{f.GetMethod()?.DeclaringType?.Name}.{f.GetMethod()?.Name}"));
            }
            catch { }
            Bridge.LogLine($"card select prompt: options={pc.Options.Count} min={pc.Min} max={pc.Max} cancelable={pc.Cancelable} method={pc.Info?.Method} caller={where}");
        }
        bool upgradeAny = pc.Info?.Method is { } meth && meth.Contains("ForUpgrade");
        for (int i = 0; i < pc.Options.Count; i++)
        {
            if (_selected.Contains(i)) continue;
            int k = i;
            var card = pc.Options[i];
            if (upgradeAny)
                d.Add(ActionKind.SelectCard, "选 " + card.Id.Entry, () => Toggle(k, false, pc), a: card, b: (object?)null);
            else
                d.Add(ActionKind.SelectCard, "选 " + card.Id.Entry, () => Toggle(k, false, pc), a: card);
        }
        if (_selected.Count >= min)
            d.Add(ActionKind.ConfirmSelection, "确认", () => Confirm(pc));
        if (cancel)
            d.Add(ActionKind.CardRewardAlt, "取消", () => Cancel(run, pc), key: "CANCEL");
        return d;
    }

    static IReadOnlyList<int> Arr() => _selected.ToArray();

    static void Toggle(int index, bool _, PendingCardChoice pc)
    {
        if (_selected.Contains(index)) _selected.Remove(index);
        else
        {
            int max = Math.Min(pc.Max, pc.Options.Count);
            if (_selected.Count >= max) return;
            _selected.Add(index);
        }
        LiveSelector.Selected = Arr();
    }

    static void Confirm(PendingCardChoice pc)
    {
        var pick = _selected.ToList();
        _selected.Clear();
        _selecting = null;
        LiveSelector.Selected = Arr();
        try
        {
            pc.Resolve(pick);
            Bridge.LogLine($"card select resolved: {string.Join(",", pick)}");
        }
        catch (Exception e)
        {
            Bridge.LogLine("card select resolve failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    static void Cancel(RunState run, PendingCardChoice pc)
    {
        _refused = new RefusalRoom(run.CurrentRoom!);
        _selected.Clear(); _selecting = null; LiveSelector.Selected = Arr();
        try
        {
            pc.Resolve(Array.Empty<int>());
            Bridge.LogLine("card select cancelled");
        }
        catch (Exception e)
        {
            Bridge.LogLine("card select cancel failed: " + e.GetType().Name + ": " + e.Message);
        }
    }
}
