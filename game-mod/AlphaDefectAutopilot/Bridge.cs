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
    const string ClientKind = "probe";     // 観測フレームではなく状態の要約を送る段階

    static readonly string Host = Environment.GetEnvironmentVariable("ALPHADEFECT_AUTOPILOT_HOST") ?? "127.0.0.1";
    static readonly int Port = ParsePort(Environment.GetEnvironmentVariable("ALPHADEFECT_AUTOPILOT_PORT"));
    static readonly ConcurrentQueue<string> Outbox = new();

    static volatile bool _stopped;
    static volatile bool _paused;
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
    public static void Tick()
    {
        if (_stopped || _paused || _stream == null) return;
        long now = Environment.TickCount64;
        if (now - Interlocked.Read(ref _lastProbeTicks) < 500) return;
        Interlocked.Exchange(ref _lastProbeTicks, now);
        Outbox.Enqueue("{\"t\":\"probe\",\"data\":" + Probe.Snapshot() + "}");
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
                    break;
                case "control":
                    string state = Field(line, "state");
                    _paused = state == "paused";
                    if (state == "stopped") Stop("control:" + Field(line, "reason"));
                    else LogLine($"control: {state}");
                    break;
                case "error":
                    LogLine("server error: " + Field(line, "msg"));
                    break;
                case "action":
                    // この段階では観測フレームを送っていないので、行動は来ない
                    LogLine("action ignored (probe mode): index=" + Field(line, "index"));
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
        return int.TryParse(raw, out int p) && p > 0 && p < 65536 ? p : 8765;
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

    static void LogLine(string message)
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
