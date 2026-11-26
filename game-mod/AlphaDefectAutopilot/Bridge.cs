using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text;

namespace AlphaDefectAutopilot;

/// <summary>
/// 推論サーバ（WSL 側の local/bridge）と 127.0.0.1 でつなぐ。
///
/// 送る: hello（版とクライアント種別）、probe（公開状態の要約）、log
/// 受け取る: welcome（観測形式）、control（paused / running / stopped）、error
///
/// 接続が切れてもゲームは止めず、2 秒おきに張り直す。停止指示（control: stopped）が来たら
/// 以後は何も送らない。読み取りはゲームのメインスレッド（NRun._Process の後処理）で行う。
/// </summary>
internal static class Bridge
{
    const int ProtocolVersion = 1;
    const string ClientKind = "autopilot";   // 観測フレームを送り、行動を受け取る段階

    static readonly string Host = Environment.GetEnvironmentVariable("ALPHADEFECT_AUTOPILOT_HOST") ?? "127.0.0.1";
    static readonly int Port = ParsePort(Environment.GetEnvironmentVariable("ALPHADEFECT_AUTOPILOT_PORT"));
    static readonly ConcurrentQueue<string> Outbox = new();

    static volatile bool _stopped;
    static volatile bool _paused;
    static volatile int _pendingAction = -1;
    static int _sentEpoch;
    static int _obsFormat;
    static bool _formatApplied;
    public static bool RunReported;
    static NetworkStream? _stream;
    static long _lastProbeTicks;
    static bool _started;

    public static void Start()
    {
        if (_started) return;
        _started = true;
        LogLine($"bridge starting -> {Host}:{Port}");
        new Thread(Supervise) { IsBackground = true, Name = "AlphaDefectBridge" }.Start();
    }

    /// <summary>ゲームのメインスレッドから呼ばれる（NRun._Process の後処理）。</summary>
    public static bool Stopped => _stopped;
    public static bool Paused => _paused;

    /// <summary>ゲームのメインスレッドから毎フレーム。状態を読める局面なら戦闘ドライバに渡す。</summary>
    public static void Tick()
    {
        if (_stopped) return;
        long now = Environment.TickCount64;
        if (now - Interlocked.Read(ref _lastProbeTicks) >= 500)
        {
            Interlocked.Exchange(ref _lastProbeTicks, now);
            Outbox.Enqueue("{\"t\":\"probe\",\"data\":" + Probe.Snapshot() + "}");
        }
        try
        {
            // パネルとキーは、停止中・手動中でも操作できるように先に動かす
            PanelState.PollKeys();
            ControlPanel.Tick();
        }
        catch (Exception e)
        {
            LogLine("panel tick failed: " + e.GetType().Name + ": " + e.Message);
        }
        if (_stream == null) return;
        try
        {
            ApplyFormat();
            Overlay.Tick();
            Launcher.Tick();
            if (MentorClient.Enabled)
            {
                MentorClient.Tick();
                return;
            }
            if (PanelState.CanAct || PanelState.AwaitingApproval || PanelState.Advising) CombatDriver.Tick();
        }
        catch (Exception e)
        {
            LogLine("driver tick failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    /// <summary>実行してよい行動の添字（確認モードで許可待ちなら -1）。</summary>
    public static int TakeAction() => PanelState.TryTakePending(out int idx) ? idx : -1;

    /// <summary>パネルの状態をサーバへ知らせる（/status と表示を一致させる）。</summary>
    public static void SendPanel(string json)
    {
        Outbox.Enqueue("{\"t\":\"panel\",\"state\":" + json + "}");
    }

    /// <summary>観測フレームを送る（base64）。</summary>
    public static void SendState(byte[] frame, int seq, string metaJson, string labels = "")
    {
        if (_stream == null || _stopped) return;
        _sentEpoch = PanelState.MarkSent();
        string b64 = Convert.ToBase64String(frame);
        string labelsField = string.IsNullOrEmpty(labels) ? "" : ",\"labels\":\"" + labels.Replace("\"", "'") + "\"";
        Outbox.Enqueue("{\"t\":\"state\",\"seq\":" + seq + ",\"frame\":\"" + b64
                       + "\",\"meta\":{" + metaJson + labelsField + "}}");
    }

    /// <summary>ラン開始設定（キャラ・アセンション・シード）をサーバへ知らせる。</summary>
    public static void SendRunConfig(string character, int ascension, int? seed)
    {
        string seedText = seed is int s ? s.ToString(System.Globalization.CultureInfo.InvariantCulture) : "null";
        Outbox.Enqueue("{\"t\":\"runconfig\",\"character\":\"" + character + "\",\"ascension\":" + ascension
                       + ",\"seed\":" + seedText + "}");
    }

    public static void SendDone(bool win, int floor, int act)
    {
        if (_stream == null) return;
        RunReported = true;
        Outbox.Enqueue("{\"t\":\"done\",\"win\":" + (win ? "true" : "false") + ",\"floor\":" + floor + ",\"act\":" + act + "}");
    }

    /// <summary>候補（candidates）を「添字:名前 確率」の並びにする。</summary>
    static List<string> ParseCandidates(string line, int max = 4)
    {
        var top = new List<string>();
        int at = line.IndexOf("\"candidates\":[[", StringComparison.Ordinal);
        if (at >= 0)
        {
            string body = line[(at + "\"candidates\":[[".Length)..];
            foreach (var part in body.Split("],["))
            {
                var bits = part.Split(',');
                if (bits.Length < 3) continue;
                string num = bits[0].Trim(' ', '[', ']', '{', '}');
                string name = bits[1].Trim().Trim('"');
                string p = bits[2].Trim(' ', ']', '}', '"');
                top.Add($"{num}:{name} {p}");
                if (top.Count >= max) break;
            }
        }
        return top;
    }

    static string ShowActionCandidates(string line) => string.Join("  ", ParseCandidates(line));

    /// <summary>返ってきた行動を画面に出す（候補は candidates から素朴に読む）。</summary>
    static void ShowAction(int index, string label, string prob, string line)
    {
        double.TryParse(prob, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double pv);
        Overlay.SetAction(index, label, pv, ShowActionCandidates(line));
    }

    static void ApplyFormat()
    {
        if (_formatApplied || _obsFormat <= 0) return;
        _formatApplied = true;
        try
        {
            Sts2Headless.Observation.FormatOverride = _obsFormat;
            LogLine("observation format set to " + _obsFormat);
        }
        catch (Exception e)
        {
            LogLine("format override failed: " + e.GetType().Name);
        }
    }

    public static void Stop(string reason)
    {
        _stopped = true;
        LogLine($"bridge stopped: {reason}");
    }

    // ---------------------------------------------------------------- 接続

    static void Supervise()
    {
        while (!_stopped)
        {
            TcpClient? client = null;
            try
            {
                client = new TcpClient();
                client.Connect(Host, Port);
                var stream = client.GetStream();
                _stream = stream;
                PanelState.SetConnected(true);
                _paused = false;                  // 新しいセッションは動いている状態から
                SendRaw(stream, Hello());
                LogLine($"connected to {Host}:{Port}");
                var reader = new Thread(() => ReadLoop(stream)) { IsBackground = true, Name = "AlphaDefectBridgeReader" };
                reader.Start();
                WriteLoop(stream);
            }
            catch (Exception e)
            {
                LogLine($"connection failed: {e.GetType().Name} {e.Message}");
            }
            finally
            {
                _stream = null;
                PanelState.SetConnected(false);
                try { client?.Close(); } catch { /* 閉じるだけ */ }
            }
            if (!_stopped) Thread.Sleep(2000);
        }
    }

    static void WriteLoop(NetworkStream stream)
    {
        while (!_stopped && ReferenceEquals(_stream, stream))
        {
            if (Outbox.TryDequeue(out var line))
            {
                SendRaw(stream, line);
            }
            else
            {
                Thread.Sleep(50);
            }
        }
    }

    static void ReadLoop(NetworkStream stream)
    {
        var reader = new StreamReader(stream, Encoding.UTF8);
        try
        {
            string? line;
            while (!_stopped && (line = reader.ReadLine()) != null)
            {
                Handle(line);
            }
        }
        catch (Exception e)
        {
            LogLine($"read failed: {e.GetType().Name} {e.Message}");
        }
    }

    static void Handle(string line)
    {
        // 返事は小さな JSON だけ。壊れていてもゲームは止めない。
        try
        {
            string type = Field(line, "t");
            switch (type)
            {
                case "welcome":
                    LogLine($"welcome: run={Field(line, "run")} obs_format={Field(line, "obs_format")}");
                    Overlay.SetStatus($"観戦中（観測形式 {Field(line, "obs_format")}）");
                    if (int.TryParse(Field(line, "obs_format"), out int fmt)) _obsFormat = fmt;
                    PanelState.Report();
                    break;
                case "control":
                    string state = Field(line, "state");
                    if (state == "stopped")
                    {
                        // サーバ側の都合（待ち時間切れ・操作者の停止）で切れただけかもしれない。
                        // ゲームは止めず、行動を止めて再接続を待つ。
                        PanelState.SetPaused(true, "server stopped");
                        PanelState.RestoreTimeScale();
                        LogLine("session stopped by server: " + Field(line, "reason"));
                    }
                    else
                    {
                        PanelState.SetPaused(state == "paused", "server");
                        LogLine($"control: {state}");
                    }
                    break;
                case "command":
                    // HTTP / ランチャーからの操作（パネルと同じ入口）
                    PanelState.Apply(Field(line, "name"), Field(line, "value"), fromUser: false);
                    LogLine($"command: {Field(line, "name")} {Field(line, "value")}");
                    break;
                case "error":
                    LogLine("server error: " + Field(line, "msg"));
                    break;
                case "speed":
                    PanelState.Apply("speed", Field(line, "scale"), fromUser: false);
                    Overlay.SetStatus($"速度 {Godot.Engine.TimeScale:0.##}x");
                    break;
                case "start":
                    Launcher.Request(Field(line, "character") + "|" + Field(line, "ascension") + "|"
                                     + Field(line, "seed") + "|" + Field(line, "encounter"));
                    LogLine("start requested: " + Field(line, "character") + " seed=" + Field(line, "seed"));
                    break;
                case "action":
                    if (int.TryParse(Field(line, "index"), out int idx))
                    {
                        double prob = 0;
                        double.TryParse(Field(line, "prob"), System.Globalization.NumberStyles.Float,
                                        System.Globalization.CultureInfo.InvariantCulture, out prob);
                        PanelState.OnAction(idx, _sentEpoch, Field(line, "label"), prob, ShowActionCandidates(line));
                        LogLine($"action: index={idx} p={Field(line, "prob")} {Field(line, "label")}");
                        ShowAction(idx, Field(line, "label"), Field(line, "prob"), line);
                    }
                    break;
            }
        }
        catch (Exception e)
        {
            LogLine("bad message: " + e.GetType().Name);
        }
    }

    // ---------------------------------------------------------------- 小物

    static string Hello()
    {
        return "{\"t\":\"hello\",\"v\":" + ProtocolVersion
             + ",\"client\":\"" + ClientKind + "\""
             + ",\"game\":{\"version\":\"" + Escape(GameVersion()) + "\"}"
             + ",\"character\":\"\""
             + ",\"mod\":{\"id\":\"AlphaDefectAutopilot\",\"version\":\"0.1.0\"}}";
    }

    static string GameVersion()
    {
        // ゲームが同梱している release_info.json を読む（local/steam.py と同じ正）。
        // ゲームの型に依存しないので、更新で API が変わっても壊れない。
        try
        {
            string? exe = Environment.ProcessPath;
            string? dir = exe == null ? null : Path.GetDirectoryName(exe);
            if (dir == null) return "?";
            string path = Path.Combine(dir, "release_info.json");
            if (!File.Exists(path)) return "?";
            string text = File.ReadAllText(path);
            int i = text.IndexOf("\"version\"", StringComparison.Ordinal);
            if (i < 0) return "?";
            int colon = text.IndexOf(':', i);
            int quote = colon < 0 ? -1 : text.IndexOf('"', colon + 1);
            int end = quote < 0 ? -1 : text.IndexOf('"', quote + 1);
            return quote < 0 || end < 0 ? "?" : text.Substring(quote + 1, end - quote - 1);
        }
        catch
        {
            return "?";
        }
    }

    static void SendRaw(NetworkStream stream, string line)
    {
        var bytes = Encoding.UTF8.GetBytes(line + "\n");
        stream.Write(bytes, 0, bytes.Length);
        stream.Flush();
    }

    static int ParsePort(string? raw)
    {
        return int.TryParse(raw, out int p) && p > 0 && p < 65536 ? p : 18765;
    }

    static string Field(string json, string key)
    {
        // 値は文字列か数値だけを想定した素朴な取り出し（完全な JSON パーサは要らない）
        string needle = "\"" + key + "\":";
        int i = json.IndexOf(needle, StringComparison.Ordinal);
        if (i < 0) return "";
        i += needle.Length;
        if (i >= json.Length) return "";
        if (json[i] == '"')
        {
            int j = json.IndexOf('"', i + 1);
            return j < 0 ? "" : json.Substring(i + 1, j - i - 1);
        }
        int k = i;
        while (k < json.Length && (char.IsDigit(json[k]) || json[k] == '-' || json[k] == '.')) k++;
        return json.Substring(i, k - i);
    }

    static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");

    /// <summary>大事な出来事はゲームのログとサーバの両方に送る（計測の停止理由に使う）。</summary>
    internal static void LogEvent(string message)
    {
        LogLine(message);
        try
        {
            var escaped = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
            Outbox.Enqueue("{\"t\":\"log\",\"level\":\"info\",\"msg\":\"" + escaped + "\"}");
        }
        catch
        {
            // 送れなくても止めない
        }
    }

    internal static void LogLine(string message)
    {
        try
        {
            MegaCrit.Sts2.Core.Logging.Log.Info("[AlphaDefect] " + message);
        }
        catch
        {
            // ログが出せなくても止めない
        }
    }
}
