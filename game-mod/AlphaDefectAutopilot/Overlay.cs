using Godot;

namespace AlphaDefectAutopilot;

/// <summary>
/// 画面に「いま AI が何を選んだか」を出す小さな表示（観戦用）。
///
/// ゲームの状態には触らない（読むだけ・描くだけ）。文字は受信スレッドから来るので、
/// いったん変数に置いてメインスレッド（Tick）で反映する。
///
/// 取り付けは「ラン中に 1 回だけ」。ゲーム起動の途中で Godot のノードを作ると
/// 起動処理とぶつかることがあるため、NRun がある（＝ラン中）フレームで作る。
/// 失敗しても 1 回で諦め、ゲームには影響させない。
/// </summary>
internal static class Overlay
{
    static Label? _status;
    static Label? _action;
    static Label? _candidates;
    static bool _failed;
    static int _attachTries;

    static volatile string _statusText = "";
    static volatile string _actionText = "";
    static volatile string _candidatesText = "";
    static volatile bool _dirty;

    /// <summary>メインスレッドで毎フレーム呼ぶ。</summary>
    public static void Tick()
    {
        if (_failed) return;
        try
        {
            if (_status == null)
            {
                if (!InRun()) return;               // ラン中だけ取り付ける
                if (_attachTries++ > 600) { _failed = true; return; }
                Attach();
                if (_status == null) return;
            }
            if (!GodotObject.IsInstanceValid(_status)) { _failed = true; return; }
            if (!_dirty) return;
            _dirty = false;
            _status!.Text = _statusText;
            _action!.Text = _actionText;
            _candidates!.Text = _candidatesText;
        }
        catch (Exception e)
        {
            _failed = true;
            Bridge.LogLine("overlay stopped: " + e.GetType().Name + ": " + e.Message);
        }
    }

    static bool InRun()
    {
        try { return MegaCrit.Sts2.Core.Nodes.NRun.Instance != null; }
        catch { return false; }
    }

    static void Attach()
    {
        if (Engine.GetMainLoop() is not SceneTree tree || tree.Root == null || !GodotObject.IsInstanceValid(tree.Root)) return;
        var layer = new CanvasLayer { Name = "AlphaDefectOverlay", Layer = 120 };
        var box = new VBoxContainer
        {
            // パネル（左上、y=178）の下に置いて重ならないようにする
            Position = new Vector2(16, 396),
            MouseFilter = Control.MouseFilterEnum.Ignore,
        };
        _status = MakeLabel(17, new Color(0.8f, 0.9f, 1f));
        _action = MakeLabel(21, new Color(1f, 0.92f, 0.5f));
        _candidates = MakeLabel(15, new Color(0.88f, 0.95f, 0.88f));
        box.AddChild(_status);
        box.AddChild(_action);
        box.AddChild(_candidates);
        layer.AddChild(box);
        tree.Root.AddChild(layer);
        _dirty = true;
        Bridge.LogLine("overlay attached");
        SetStatus("観戦中");
    }

    static Label MakeLabel(int size, Color color)
    {
        var label = new Label { Text = "", MouseFilter = Control.MouseFilterEnum.Ignore };
        label.AddThemeFontSizeOverride("font_size", size);
        label.AddThemeColorOverride("font_color", color);
        label.AddThemeColorOverride("font_outline_color", new Color(0, 0, 0, 0.9f));
        label.AddThemeConstantOverride("outline_size", 6);
        return label;
    }

    // ---------------------------------------------------------------- 受信側（どのスレッドからでも）

    public static void SetStatus(string text)
    {
        _statusText = "[AlphaDefect] " + text;
        _dirty = true;
    }

    public static void SetAction(int index, string label, double prob, string candidates)
    {
        _actionText = $"AI: [{index}] {label}  p={prob:0.000}";
        _candidatesText = candidates.Length == 0 ? "" : "候補: " + candidates;
        _dirty = true;
    }

    public static void Clear()
    {
        _actionText = "";
        _candidatesText = "";
        _dirty = true;
    }
}
