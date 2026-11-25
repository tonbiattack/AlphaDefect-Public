using System.IO;
using System.Text.Json;
using Godot;

namespace AlphaDefectAutopilot;

/// <summary>
/// 観戦中の操作状態（自動/手動・一時停止・1 手実行・速度・行動間隔・確認モード）。
///
/// 操作の入口は 3 つあり、どれもここを通す:
///   * ゲーム内パネルのボタン（ControlPanel）
///   * キーボードショートカット（keys.json で変更・無効化できる）
///   * 推論サーバからの command メッセージ（HTTP / ランチャーの --pause など）
///
/// 状態が変わったら推論サーバへ panel メッセージで知らせ、/status と表示を一致させる。
/// 手動に切り替えたときは epoch を進め、それ以前に送った判断の応答を実行しない
/// （古い応答で勝手に動かないようにする）。
/// </summary>
internal static class PanelState
{
    public enum RunMode { Auto, Manual }

    static bool _paused;
    static bool _stepOnce;
    static int _epoch;
    static bool _connected;
    static string _recommendation = "";
    static string _candidates = "";
    static int _pendingIndex = -1;
    static int _pendingEpoch = -1;
    static bool _approved;
    static long _lastTakeLog;
    static long _stateSeq;

    public static RunMode Mode { get; private set; } = RunMode.Auto;
    public static bool Paused => _paused;
    public static bool StepOnce => _stepOnce;
    public static bool Confirm { get; private set; }
    public static double Speed { get; private set; } = 1.0;
    public static int PacingMs { get; private set; } = 0;
    public static bool StopOnCombatEnd { get; private set; }
    public static bool StopOnCardReward { get; private set; }
    public static int Epoch => _epoch;
    public static bool Connected => _connected;
    public static string Status => _recommendation;
    public static string Candidates => _candidates;

    /// <summary>手動・停止・切断のどれでもない（AI が自分で打ってよい）。</summary>
    public static bool CanAct => Mode == RunMode.Auto && !_paused && _connected;

    /// <summary>確認モード: 応答が来たら実行せず、ユーザーの許可を待つ。</summary>
    public static bool AwaitingApproval => Confirm && _pendingIndex >= 0 && !_approved && Mode == RunMode.Auto;

    public static bool KeysEnabled { get; private set; } = true;
    static readonly Dictionary<string, string> Keys = new();

    static readonly HashSet<Key> Down = new();

    // ------------------------------------------------------------ 操作

    /// <summary>操作を 1 つ適用する。fromUser はパネル/キーボード（true）かサーバ（false）。</summary>
    public static void Apply(string command, string value = "", bool fromUser = true)
    {
        string cmd = (command ?? "").Trim().ToLowerInvariant();
        switch (cmd)
        {
            case "pause":
                SetPaused(true, "operator");
                break;
            case "restore":
                RestoreTimeScale();
                break;
            case "resume":
                SetPaused(false, "operator");
                break;
            case "toggle-pause":
                SetPaused(!_paused, "operator");
                break;
            case "step":
                Step();
                break;
            case "manual":
                SetMode(RunMode.Manual, "operator");
                break;
            case "auto":
                SetMode(RunMode.Auto, "operator");
                break;
            case "toggle-mode":
                SetMode(Mode == RunMode.Auto ? RunMode.Manual : RunMode.Auto, "operator");
                break;
            case "confirm":
                _ = bool.TryParse(value, out bool on);
                Confirm = string.IsNullOrEmpty(value) ? !Confirm : on;
                Log("confirm=" + Confirm);
                break;
            case "confirm-on":
                Confirm = true;
                Log("confirm=on");
                break;
            case "confirm-off":
                Confirm = false;
                Log("confirm=off");
                break;
            case "speed":
                if (double.TryParse(value, System.Globalization.NumberStyles.Float,
                                    System.Globalization.CultureInfo.InvariantCulture, out double sp))
                    SetSpeed(sp, fromUser);
                break;
            case "speed-up":
                SetSpeed(NextSpeed(Speed), fromUser);
                break;
            case "speed-down":
                SetSpeed(PrevSpeed(Speed), fromUser);
                break;
            case "pacing":
                if (int.TryParse(value, out int ms)) SetPacing(ms);
                break;
            case "stopcombat":
                StopOnCombatEnd = string.IsNullOrEmpty(value) ? !StopOnCombatEnd : value is "on" or "true" or "1";
                Log("stop_on_combat_end=" + StopOnCombatEnd);
                break;
            case "stopreward":
                StopOnCardReward = string.IsNullOrEmpty(value) ? !StopOnCardReward : value is "on" or "true" or "1";
                Log("stop_on_card_reward=" + StopOnCardReward);
                break;
            case "hide":
            case "show":
                ControlPanel.SetVisible(cmd == "show");
                break;
            case "toggle-panel":
                ControlPanel.ToggleVisible();
                break;
            default:
                Log("unknown panel command: " + cmd);
                return;
        }
        Report();
    }

    public static void SetPaused(bool paused, string why)
    {
        if (_paused == paused) return;
        _paused = paused;
        if (paused) _stepOnce = false;
        Log($"paused={paused} ({why})");
        if (!paused) Bump("resume");       // 再開したら古い応答は捨てる
        Changed?.Invoke();
    }

    public static void SetMode(RunMode mode, string why)
    {
        if (Mode == mode) return;
        Mode = mode;
        _stepOnce = false;
        if (mode == RunMode.Manual)
        {
            _paused = false;               // 手動では AI が止まるだけで、ゲームは止めない
            Bump("manual");
            Log("mode=manual (" + why + ")");
        }
        else
        {
            Bump("auto");
            Log("mode=auto (" + why + ")");
        }
        Changed?.Invoke();
    }

    /// <summary>1 手だけ実行してまた止める。応答待ちがあればそれを実行する。</summary>
    public static void Step()
    {
        if (Mode != RunMode.Auto) return;
        if (_pendingIndex >= 0 && !_approved)
        {
            _stepOnce = true;              // この 1 手のあとまた止める
            Log("step: approved pending action");
            ApprovePending();
            return;
        }
        _paused = false;
        _stepOnce = true;
        Bump("step");
        Log("step: one decision");
        Changed?.Invoke();
    }

    static void SetSpeed(double speed, bool fromUser)
    {
        Speed = Math.Clamp(speed, 0.1, 20.0);
        Godot.Engine.TimeScale = Speed;      // ゲームの演出速度
        Log($"speed={Speed:0.##} ({(fromUser ? "panel" : "server")})");
        Changed?.Invoke();
    }

    static void SetPacing(int ms)
    {
        PacingMs = Math.Clamp(ms, 0, 10000);
        Log("pacing=" + PacingMs + "ms");
        Changed?.Invoke();
    }

    static readonly double[] Speeds = { 0.25, 0.5, 1.0, 2.0, 5.0, 10.0 };

    static double NextSpeed(double current)
    {
        foreach (var s in Speeds) if (s > current + 1e-9) return s;
        return Speeds[^1];
    }

    static double PrevSpeed(double current)
    {
        for (int i = Speeds.Length - 1; i >= 0; i--) if (Speeds[i] < current - 1e-9) return Speeds[i];
        return Speeds[0];
    }

    // ------------------------------------------------------------ 応答と実行の境界

    public static event Action? Changed;

    /// <summary>判断を送った時点の epoch。応答が返ってきたときの照合用。</summary>
    public static int MarkSent() => _epoch;

    /// <summary>推論サーバから行動が届いた。確認モードならここで止めて表示だけする。</summary>
    public static void OnAction(int index, int sentEpoch, string label, double prob, string candidates)
    {
        _recommendation = $"[{index}] {label} p={prob:0.000}";
        _candidates = candidates;
        if (sentEpoch != _epoch)
        {
            Log($"stale action dropped: index={index} sent_epoch={sentEpoch} now={_epoch}");
            Changed?.Invoke();
            return;
        }
        _pendingIndex = index;
        _pendingEpoch = sentEpoch;
        _approved = !Confirm;              // 確認モードでは許可が要る
        if (Confirm)
        {
            _paused = true;                // ユーザーが許可するまで実行しない
            Log("confirm: waiting for approval: " + _recommendation);
        }
        Changed?.Invoke();
    }

    /// <summary>確認待ちの行動を取り出す（実行してよい場合のみ）。</summary>
    public static bool TryTakePending(out int index)
    {
        index = -1;
        if (_pendingIndex < 0) return false;
        if (_pendingEpoch != _epoch)        // 古い応答は捨てる
        {
            _pendingIndex = -1;
            _approved = false;
            return false;
        }
        if (!_approved)
        {
            if (System.Environment.TickCount64 - _lastTakeLog > 1000)
            {
                _lastTakeLog = System.Environment.TickCount64;
                Log($"take: waiting for approval (pending={_pendingIndex} epoch={_pendingEpoch}/{_epoch})");
            }
            return false;                   // ユーザーの許可待ち
        }
        index = _pendingIndex;
        _pendingIndex = -1;
        _approved = false;
        return true;
    }

    public static void ApprovePending()
    {
        if (_pendingIndex < 0) return;
        _approved = true;
        _paused = false;
        Log("approved: " + _recommendation);
        Changed?.Invoke();
    }

    /// <summary>手動・停止・再開のたびに epoch を進める（古い応答を捨てる）。</summary>
    static void Bump(string why)
    {
        _epoch++;
        _pendingIndex = -1;
        _pendingEpoch = -1;
        _approved = false;
        Log($"epoch={_epoch} ({why})");
    }

    /// <summary>演出速度を等倍に戻す（停止・切断・終了時にゲームを変な速さで残さない）。</summary>
    public static void RestoreTimeScale()
    {
        try
        {
            if (Math.Abs(Godot.Engine.TimeScale - 1.0) > 1e-6)
            {
                Godot.Engine.TimeScale = 1.0;
                Log("TimeScale restored to 1");
            }
            Speed = 1.0;
        }
        catch (Exception e)
        {
            Log("TimeScale restore failed: " + e.GetType().Name);
        }
        Changed?.Invoke();
    }

    public static void SetConnected(bool connected)
    {
        if (_connected == connected) return;
        _connected = connected;
        if (!connected)
        {
            _pendingIndex = -1;            // 切れたら保留の行動は捨てる（手動と同じ扱い）
            RestoreTimeScale();            // 切断時は等倍に戻す（次の接続で speed が来る）
            Log("connection lost: holding (no actions)");
        }
        else
        {
            Log("connection established");
        }
        Changed?.Invoke();
    }

    /// <summary>自動停止の設定に当てはまったら止める。呼び出し側は判断の直前。</summary>
    public static void AutoStop(string reason)
    {
        if (_paused) return;
        _paused = true;
        _stepOnce = false;
        Log("auto stop: " + reason);
        Changed?.Invoke();
        Report();
    }

    /// <summary>1 手実行したので、また止める（ステップ実行）。</summary>
    public static void AfterStep()
    {
        if (!_stepOnce) return;
        _stepOnce = false;
        _paused = true;
        Log("step done: paused again");
        Changed?.Invoke();
        Report();
    }

    // ------------------------------------------------------------ キーボード

    public static void LoadKeys()
    {
        string path = Path.Combine(Path.GetDirectoryName(typeof(PanelState).Assembly.Location) ?? ".", "keys.json");
        try
        {
            if (!File.Exists(path)) return;
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.TryGetProperty("enabled", out var en))
                KeysEnabled = en.ValueKind != JsonValueKind.False;
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Value.ValueKind == JsonValueKind.String)
                    Keys[prop.Name.ToLowerInvariant()] = prop.Value.GetString() ?? "";
            }
        }
        catch (Exception e)
        {
            Log("keys.json を読めませんでした: " + e.GetType().Name);
        }
    }

    static (bool alt, bool ctrl, bool meta, Key key) ParseKey(string spec)
    {
        bool alt = false, ctrl = false, meta = false;
        string rest = spec;
        foreach (var part in spec.Split('+', StringSplitOptions.RemoveEmptyEntries))
        {
            switch (part.Trim().ToLowerInvariant())
            {
                case "alt": alt = true; break;
                case "ctrl": case "control": ctrl = true; break;
                case "meta": case "cmd": case "win": meta = true; break;
                default: rest = part.Trim(); break;
            }
        }
        return (alt, ctrl, meta, Enum.TryParse<Key>(rest, true, out var k) ? k : Key.None);
    }

    static string KeyFor(string action, string fallback) =>
        Keys.TryGetValue(action, out var v) && !string.IsNullOrWhiteSpace(v) ? v : fallback;

    /// <summary>毎フレーム呼ぶ。押した瞬間だけ反応する。</summary>
    public static void PollKeys()
    {
        if (!KeysEnabled) return;
        var seen = new HashSet<Key>();
        void Check(string action, string fallback, Action run)
        {
            var (alt, ctrl, meta, key) = ParseKey(KeyFor(action, fallback));
            if (key == Key.None) return;
            bool down = Godot.Input.IsPhysicalKeyPressed(key)
                        && (!alt || Godot.Input.IsPhysicalKeyPressed(Key.Alt))
                        && (!ctrl || Godot.Input.IsPhysicalKeyPressed(Key.Ctrl))
                        && (!meta || Godot.Input.IsPhysicalKeyPressed(Key.Meta));
            seen.Add(key);
            if (down && Down.Add(key))
            {
                Log($"key {KeyFor(action, fallback)} -> {action}");
                run();
            }
            else if (!down) Down.Remove(key);
        }
        Check("hide", "F6", () => ControlPanel.ToggleVisible());
        Check("pause", "F7", () => Apply("toggle-pause"));
        Check("step", "F8", () => Apply("step"));
        Check("mode", "F9", () => Apply("toggle-mode"));
        Check("confirm", "F10", () => Apply("confirm"));
        Check("speed_down", "Alt+BracketLeft", () => Apply("speed-down"));
        Check("speed_up", "Alt+BracketRight", () => Apply("speed-up"));
        if (seen.Count == 0) Down.Clear();
    }

    // ------------------------------------------------------------ 報告

    public static void Report() => Bridge.SendPanel(ToJson());

    public static string ToJson()
    {
        _stateSeq++;
        return "{"
               + $"\"mode\":\"{(Mode == RunMode.Auto ? "auto" : "manual")}\","
               + $"\"paused\":{(_paused ? "true" : "false")},"
               + $"\"step\":{(_stepOnce ? "true" : "false")},"
               + $"\"confirm\":{(Confirm ? "true" : "false")},"
               + $"\"speed\":{Speed.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture)},"
               + $"\"pacing_ms\":{PacingMs},"
               + $"\"stop_combat_end\":{(StopOnCombatEnd ? "true" : "false")},"
               + $"\"stop_card_reward\":{(StopOnCardReward ? "true" : "false")},"
               + $"\"connected\":{(_connected ? "true" : "false")},"
               + $"\"epoch\":{_epoch},"
               + $"\"seq\":{_stateSeq}"
               + "}";
    }

    static void Log(string message) => Bridge.LogLine("[panel] " + message);
}
