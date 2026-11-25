using Godot;

namespace AlphaDefectAutopilot;

/// <summary>
/// ゲーム内の操作パネル（速度・一時停止・1 手実行・手動切り替え・確認モード）。
///
/// 置き場所は画面の左上、HUD のすぐ下。カードや報酬の操作を邪魔しないよう小さくまとめ、
/// 折りたたむと「⚙」1 つになる。状態は PanelState が持ち、ここは表示と入力だけ。
/// ノードはラン中に 1 回だけ作る（ゲーム起動の途中で作ると起動処理とぶつかるため）。
/// </summary>
internal static class ControlPanel
{
    static CanvasLayer? _layer;
    static PanelContainer? _panel;
    static Label? _head;
    static Label? _reco;
    static Button? _pause;
    static Button? _mode;
    static Button? _collapse;
    static readonly List<Control> ExtraRows = new();
    static bool _collapsed;
    static bool _visible = true;
    static bool _failed;
    static int _tries;

    static readonly double[] SpeedPresets = { 0.25, 0.5, 1, 2, 5, 10 };
    static readonly int[] PacingPresets = { 0, 500, 1000, 2000, 5000 };

    static readonly List<(Button button, string label, Func<string> text)> Dynamic = new();

    public static void ToggleVisible()
    {
        _visible = !_visible;
        ApplyVisibility();
        Bridge.LogLine("[panel] visible=" + _visible);
    }

    public static void SetVisible(bool visible)
    {
        _visible = visible;
        ApplyVisibility();
    }

    static void ApplyVisibility()
    {
        if (_panel != null && GodotObject.IsInstanceValid(_panel)) _panel.Visible = _visible;
        if (_collapse != null && GodotObject.IsInstanceValid(_collapse))
            _collapse.Text = _visible ? "−" : "⚙";
    }

    /// <summary>メインスレッドから毎フレーム。</summary>
    public static void Tick()
    {
        if (_failed) return;
        try
        {
            if (_panel == null)
            {
                if (!InRun()) return;                 // ラン中だけ出す
                if (_tries++ > 900) { _failed = true; return; }
                Attach();
                if (_panel == null) return;
            }
            if (!GodotObject.IsInstanceValid(_panel)) { _failed = true; return; }
            if (!_visible) return;
            Refresh();
        }
        catch (Exception e)
        {
            _failed = true;
            Bridge.LogLine("[panel] stopped: " + e.GetType().Name + ": " + e.Message);
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
        var layer = new CanvasLayer { Name = "AlphaDefectPanelLayer", Layer = 121 };
        var panel = new PanelContainer
        {
            Name = "AlphaDefectPanel",
            // 上部バー（HP・ゴールド・遺物）のすぐ下の左に置く。カードや報酬の操作は中央・下部なので重ならない
            Position = new Vector2(16, 178),
            MouseFilter = Control.MouseFilterEnum.Stop,
        };
        var bg = new StyleBoxFlat
        {
            BgColor = new Color(0.06f, 0.07f, 0.1f, 0.88f),
            CornerRadiusTopLeft = 6, CornerRadiusTopRight = 6,
            CornerRadiusBottomLeft = 6, CornerRadiusBottomRight = 6,
            ContentMarginLeft = 10, ContentMarginRight = 10,
            ContentMarginTop = 8, ContentMarginBottom = 8,
        };
        panel.AddThemeStyleboxOverride("panel", bg);

        var root = new VBoxContainer();
        panel.AddChild(root);

        var top = new HBoxContainer();
        _head = MakeLabel(14, new Color(0.8f, 0.9f, 1f));
        _head.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
        top.AddChild(_head);
        _collapse = MakeButton("−", () => SetCollapsed(!_collapsed));
        top.AddChild(_collapse);
        top.AddChild(MakeButton("×", ToggleVisible));
        root.AddChild(top);

        _reco = MakeLabel(14, new Color(1f, 0.92f, 0.5f));
        _reco.AutowrapMode = TextServer.AutowrapMode.Off;
        _reco.Text = "推奨: —";
        root.AddChild(_reco);

        var controls = new HBoxContainer();
        _pause = MakeButton("一時停止", () => PanelState.Apply("toggle-pause"));
        _mode = MakeButton("手動にする", () => PanelState.Apply("toggle-mode"));
        controls.AddChild(_pause);
        controls.AddChild(_mode);
        controls.AddChild(MakeButton("1手", () => PanelState.Apply("step")));
        controls.AddChild(MakeButton("確認", () => PanelState.Apply("confirm")));
        controls.AddChild(MakeButton("戦闘終了で停止", () => PanelState.Apply("stopcombat")));
        root.AddChild(controls);
        ExtraRows.Add(controls);

        var speeds = new HBoxContainer();
        speeds.AddChild(MakeLabel(13, new Color(0.75f, 0.8f, 0.9f), "速度"));
        foreach (var s in SpeedPresets)
        {
            double v = s;
            speeds.AddChild(MakeButton($"{s:0.##}x", () => PanelState.Apply("speed", v.ToString(System.Globalization.CultureInfo.InvariantCulture))));
        }
        root.AddChild(speeds);
        ExtraRows.Add(speeds);

        var pacing = new HBoxContainer();
        pacing.AddChild(MakeLabel(13, new Color(0.75f, 0.8f, 0.9f), "間隔"));
        foreach (var ms in PacingPresets)
        {
            int v = ms;
            string text = ms == 0 ? "既定" : $"{ms / 1000.0:0.#}s";
            pacing.AddChild(MakeButton(text, () => PanelState.Apply("pacing", v.ToString())));
        }
        root.AddChild(pacing);
        ExtraRows.Add(pacing);

        layer.AddChild(panel);
        tree.Root.AddChild(layer);
        _layer = layer;
        _panel = panel;
        ApplyVisibility();
        PanelState.Changed += Refresh;
        Bridge.LogLine("[panel] attached");
        Refresh();
    }

    static Label MakeLabel(int size, Color color, string text = "")
    {
        var label = new Label { Text = text, MouseFilter = Control.MouseFilterEnum.Ignore };
        label.AddThemeFontSizeOverride("font_size", size);
        label.AddThemeColorOverride("font_color", color);
        label.AddThemeColorOverride("font_outline_color", new Color(0, 0, 0, 0.9f));
        label.AddThemeConstantOverride("outline_size", 4);
        return label;
    }

    static Button MakeButton(string text, Action pressed)
    {
        var button = new Button
        {
            Text = text,
            FocusMode = Control.FocusModeEnum.None,
            MouseFilter = Control.MouseFilterEnum.Stop,
        };
        button.AddThemeFontSizeOverride("font_size", 13);
        button.Pressed += () =>
        {
            try { pressed(); }
            catch (Exception e) { Bridge.LogLine("[panel] button failed: " + e.GetType().Name + ": " + e.Message); }
        };
        return button;
    }

    static void SetCollapsed(bool collapsed)
    {
        _collapsed = collapsed;
        if (_reco != null && GodotObject.IsInstanceValid(_reco)) _reco.Visible = !collapsed;
        foreach (var row in ExtraRows)
            if (GodotObject.IsInstanceValid(row)) row.Visible = !collapsed;
        if (_collapse != null && GodotObject.IsInstanceValid(_collapse))
            _collapse.Text = collapsed ? "+" : "−";
        Bridge.LogLine("[panel] collapsed=" + collapsed);
    }

    static void Refresh()
    {
        if (_head == null || !GodotObject.IsInstanceValid(_head)) return;
        string mode = PanelState.Mode == PanelState.RunMode.Auto ? "AI自動" : "手動";
        string state = PanelState.Mode == PanelState.RunMode.Manual ? "（AI は動きません）"
            : PanelState.Paused ? "一時停止" : "実行中";
        string conn = PanelState.Connected ? "接続" : "未接続";
        string confirm = PanelState.Confirm ? " 確認ON" : "";
        string stops = PanelState.StopOnCombatEnd ? " 戦闘終了停止" : "";
        string pacing = PanelState.PacingMs > 0 ? $" 間隔{PanelState.PacingMs}ms" : "";
        _head.Text = $"{mode} {state} {PanelState.Speed:0.##}x {conn}{confirm}{stops}{pacing}";
        if (_reco != null && GodotObject.IsInstanceValid(_reco))
            _reco.Text = "推奨: " + (PanelState.Status.Length > 0 ? PanelState.Status : "—");
        if (_pause != null && GodotObject.IsInstanceValid(_pause))
            _pause.Text = PanelState.Paused || PanelState.Mode == PanelState.RunMode.Manual ? "再開" : "一時停止";
        if (_mode != null && GodotObject.IsInstanceValid(_mode))
            _mode.Text = PanelState.Mode == PanelState.RunMode.Auto ? "手動にする" : "AIにする";
    }
}
