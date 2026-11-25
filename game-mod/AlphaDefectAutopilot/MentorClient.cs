using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// 実験: spire-mentor（MIT）の操作基盤を借り、判断だけをうちのモデルに任せるクライアント。
///
/// ねらい（#29）: 画面の準備・非同期処理・選牌・鍛冶・店の駆動は spire-mentor に任せ、
/// 意思決定は AlphaDefect の観測エンコーダと学習済みモデルで行う。モデルは置換しない。
///
/// 仕組み: この MOD はゲーム内で動いているので、ゲームのオブジェクトからそのまま
/// うちのフレーム（Observation.Build）を作れる。spire-mentor の TCP（9002）にクライアントとして
/// 繋ぎ、判断メッセージが来たらフレームを推論サーバへ送り、返ってきた候補 index を
/// spire-mentor の行動（play / end_turn / choose）に翻訳して返す。
///
/// 有効化: 環境変数 ALPHADEFECT_MENTOR=1（既定は無効。通常の自動プレイは今まで通り）。
/// 有効時はこちらの選牌器とドライバは動かさない（spire-mentor と衝突するため）。
/// </summary>
internal static class MentorClient
{
    public static bool Enabled { get; private set; }

    const int Port = 9002;
    static TcpClient? _client;
    static NetworkStream? _stream;
    static readonly ConcurrentQueue<string> Inbox = new();
    static readonly object SendLock = new();

    static string _pendingKind = "";
    static int _pendingRequestId = -1;
    static int _pendingStateId = -1;
    static Decision? _pendingDecision;
    static int _pendingSeq;
    static bool _awaiting;
    static long _deadline;
    static int _replies;
    static int _unmapped;
    static string? _lastMessage;

    public static void Start()
    {
        string? flag = Environment.GetEnvironmentVariable("ALPHADEFECT_MENTOR");
        string modDir = Path.GetDirectoryName(typeof(MentorClient).Assembly.Location) ?? ".";
        Enabled = flag is "1" or "true" or "yes" || File.Exists(Path.Combine(modDir, "mentor.txt"));
        if (!Enabled) return;
        Bridge.LogEvent("[mentor] experimental mode: spire-mentor drives, our model decides");
        new Thread(Supervise) { IsBackground = true, Name = "MentorClient" }.Start();
    }

    static void Supervise()
    {
        while (true)
        {
            try
            {
                if (_client == null || !_client.Connected)
                {
                    _client = new TcpClient();
                    _client.Connect("127.0.0.1", Port);
                    _stream = _client.GetStream();
                    Bridge.LogEvent("[mentor] connected to spire-mentor on 127.0.0.1:" + Port);
                    new Thread(ReadLoop) { IsBackground = true, Name = "MentorReader" }.Start();
                }
            }
            catch (Exception)
            {
                // ゲームがまだ立ち上がっていない等。少し待ってやり直す
            }
            Thread.Sleep(2000);
        }
    }

    static void ReadLoop()
    {
        try
        {
            var stream = _stream;
            if (stream == null) return;
            var reader = new StreamReader(stream, Encoding.UTF8);
            while (true)
            {
                string? line = reader.ReadLine();
                if (line == null) break;
                if (line.Length > 0) Inbox.Enqueue(line);
            }
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[mentor] reader stopped: " + e.GetType().Name);
        }
    }

    /// <summary>メインスレッドから毎フレーム。</summary>
    public static void Tick()
    {
        if (!Enabled) return;
        // 1) 届いた判断メッセージを処理（フレームを送る）
        while (Inbox.TryDequeue(out string? line))
        {
            if (line is not null) HandleMessage(line);
        }
        // 2) 推論の返答が来ていたら spire-mentor に返す
        if (_awaiting && _pendingDecision != null)
        {
            int idx = Bridge.TakeAction();
            if (idx >= 0)
            {
                Reply(idx);
                _awaiting = false;
                _pendingDecision = null;
                return;
            }
            if (Environment.TickCount64 > _deadline)
            {
                Bridge.LogEvent("[mentor] inference timeout; answering with a legal default");
                Reply(-1);
                _awaiting = false;
                _pendingDecision = null;
            }
        }
    }

    static void HandleMessage(string line)
    {
        try
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            string kind = root.TryGetProperty("type", out var t) ? t.GetString() ?? "" : "";
            int requestId = ReadInt(root, "request_id");
            int stateId = ReadInt(root, "state_id");
            _lastMessage = line;
            if (kind == "run_config")
            {
                // ランの開始もこちらで決める（Defect / A0 / 乱数 seed）
                SendReply(WithIds("{\"action\":\"start_run\",\"character\":\"DEFECT\",\"ascension\":0,\"seed\":null}", requestId, stateId));
                Bridge.LogEvent("[mentor] answered run_config with start_run (DEFECT, A0)");
                return;
            }
            if (kind != "combat_action")
            {
                // 最初の PoC は戦闘のみ。他は合法な既定（先頭を選ぶ）で返す。
                _unmapped++;
                if (_unmapped <= 5 || _unmapped % 50 == 0)
                    Bridge.LogEvent($"[mentor] unmapped decision '{kind}' ({_unmapped}); replying with the first option");
                SendReply(DefaultReply(kind, root, requestId));
                return;
            }
            var run = RunManager.Instance.DebugOnlyGetState();
            if (run == null || run.Players.Count == 0) return;
            var me = run.Players[0];
            var pcs = me.PlayerCombatState;
            if (pcs == null) return;
            var d = CombatDriver.BuildCombatDecision(me, pcs);
            var frame = Observation.Build(new RunSession { Steps = _replies }, d, labels: true).Serialize();
            _pendingKind = kind;
            _pendingRequestId = requestId;
            _pendingStateId = stateId;
            _pendingDecision = d;
            _pendingSeq = ++_replies;
            _awaiting = true;
            _deadline = Environment.TickCount64 + 15000;
            Bridge.SendState(frame, _pendingSeq,
                $"\"character\":\"{me.Character?.Id?.Entry ?? ""}\",\"act\":{run.CurrentActIndex},"
                + $"\"floor\":{run.TotalFloor},\"hp\":{me.Creature.CurrentHp},\"mentor\":true");
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[mentor] message handling failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    static string DefaultReply(string kind, JsonElement root, int requestId)
    {
        string reply = kind switch
        {
            "reward_screen" or "shop" or "rest_site" or "event" or "treasure" or "boss_relic"
                or "card_bundle" or "card_select" or "card_reward" or "map_select"
                => "{\"action\":\"choose\",\"index\":0}",
            _ => "{\"action\":\"end_turn\"}",
        };
        if (kind == "map_select") reply = "{\"index\":0}";
        return WithIds(reply, requestId, -1);
    }

    static int ReadInt(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var v)) return -1;
        return v.ValueKind switch
        {
            JsonValueKind.Number => v.TryGetInt32(out int n) ? n : -1,
            JsonValueKind.String => int.TryParse(v.GetString(), out int m) ? m : -1,
            _ => -1,
        };
    }

    static string WithIds(string json, int requestId, int stateId)
    {
        var dict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json)
                   ?? new Dictionary<string, JsonElement>();
        var outDict = new Dictionary<string, object?>();
        foreach (var kv in dict) outDict[kv.Key] = kv.Value.ValueKind == JsonValueKind.Number
            ? kv.Value.GetInt32() : kv.Value.GetString();
        // spire-mentor は request_id を「JSON 文字列」で返すことを要求する（数値だと捨てられる）
        if (requestId >= 0) outDict["request_id"] = requestId.ToString();
        if (stateId >= 0) outDict["state_id"] = stateId;
        return JsonSerializer.Serialize(outDict);
    }

    static void Reply(int index)
    {
        var d = _pendingDecision;
        if (d == null) return;
        string reply = index >= 0 && index < d.Choices.Count
            ? Translate(d.Choices[index])
            : "{\"action\":\"end_turn\"}";
        SendReply(WithIds(reply, _pendingRequestId, _pendingStateId));
    }

    /// <summary>うちの候補を spire-mentor の行動へ翻訳する。</summary>
    static string Translate(Choice choice)
    {
        try
        {
            if (choice.Kind == ActionKind.EndTurn) return "{\"action\":\"end_turn\"}";
            if (choice.Kind == ActionKind.PlayCard && choice.A is CardModel card)
            {
                var run = RunManager.Instance.DebugOnlyGetState();
                var me = run?.Players[0];
                if (me == null) return "{\"action\":\"end_turn\"}";
                int handIndex = -1;
                var hand = PileType.Hand.GetPile(me).Cards;
                for (int i = 0; i < hand.Count; i++)
                    if (ReferenceEquals(hand[i], card)) { handIndex = i; break; }
                if (handIndex < 0) return "{\"action\":\"end_turn\"}";
                int targetIndex = -1;
                if (choice.B is Creature target)
                {
                    var cs = me.Creature.CombatState;
                    if (cs != null)
                    {
                        int k = 0;
                        foreach (var c in cs.Creatures)
                        {
                            if (c.IsPlayer || !c.IsAlive) continue;
                            if (ReferenceEquals(c, target)) { targetIndex = k; break; }
                            k++;
                        }
                    }
                }
                return targetIndex >= 0
                    ? $"{{\"action\":\"play\",\"card_index\":{handIndex},\"target_index\":{targetIndex}}}"
                    : $"{{\"action\":\"play\",\"card_index\":{handIndex}}}";
            }
            if (choice.Kind == ActionKind.UsePotion && choice.A is PotionModel potion)
            {
                var run = RunManager.Instance.DebugOnlyGetState();
                var me = run?.Players[0];
                if (me != null)
                {
                    int slot = -1;
                    int slotCount = 0;
                    foreach (var p in me.Potions)
                    {
                        if (ReferenceEquals(p, potion)) { slot = slotCount; break; }
                        slotCount++;
                    }
                    if (slot >= 0) return $"{{\"action\":\"potion\",\"slot\":{slot}}}";
                }
                return "{\"action\":\"end_turn\"}";
            }
            return "{\"action\":\"end_turn\"}";
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[mentor] translate failed: " + e.GetType().Name);
            return "{\"action\":\"end_turn\"}";
        }
    }

    static int _sent;

    static void SendReply(string json)
    {
        var stream = _stream;
        if (stream == null) return;
        if (_sent++ < 40) Bridge.LogEvent("[mentor] reply: " + json);
        try
        {
            lock (SendLock)
            {
                byte[] bytes = Encoding.UTF8.GetBytes(json + "\n");
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush();
            }
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[mentor] reply failed: " + e.GetType().Name);
        }
    }
}
