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
    static int _ignored;
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
        // 推論の返答待ちの間は、次のメッセージを処理しない（pending を上書きしない）
        if (!_awaiting)
        {
            while (Inbox.TryDequeue(out string? line))
            {
                if (line is not null) HandleMessage(line);
                if (_awaiting) break;      // 1 件処理したら返答待ちへ
            }
        }
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
            // 返事が要らないメッセージ（エラー通知・完了通知など）には返さない。
            // 返すと相手が「不正な返答」として再送し、ループになる。
            if (kind is "error" or "pong" or "snapshot_saved" or "run_complete" or "game_over"
                or "combat_ended" or "annotation" or "advice_request" or "human_action")
            {
                if (_ignored++ < 5) Bridge.LogEvent($"[mentor] ignoring '{kind}' (no reply needed)");
                return;
            }
            if (kind == "run_config")
            {
                // ランの開始もこちらで決める（Defect / A0 / 乱数 seed）
                SendReply(WithIds("{\"action\":\"start_run\",\"character\":\"DEFECT\",\"ascension\":0,\"seed\":null}", requestId, stateId));
                Bridge.LogEvent("[mentor] answered run_config with start_run (DEFECT, A0)");
                return;
            }
            var run = RunManager.Instance.DebugOnlyGetState();
            if (run == null || run.Players.Count == 0) return;
            var me = run.Players[0];
            Decision? d = kind switch
            {
                "combat_action" => me.PlayerCombatState == null ? null : CombatDriver.BuildCombatDecision(me, me.PlayerCombatState),
                "card_reward" => LiveCardReward.Active ? LiveCardReward.Build() : null,
                "rest_site" => BuildRestDecision(),
                "event" => OutOfCombat.BuildEventDecisionForMentor(run, me),
                "map_select" => BuildMapDecision(run),
                _ => null,
            };
            if (d == null || d.Choices.Count == 0)
            {
                _unmapped++;
                if (_unmapped <= 5 || _unmapped % 50 == 0)
                    Bridge.LogEvent($"[mentor] '{kind}' has no model decision yet ({_unmapped}); replying with a legal default");
                SendReply(DefaultReply(kind, root, requestId));
                return;
            }
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

    /// <summary>休息所の選択肢を読むだけ（押さない）。</summary>
    static Decision? BuildRestDecision()
    {
        try
        {
            var sync = RunManager.Instance.RestSiteSynchronizer;
            var opts = sync.GetLocalOptions();
            var d = new Decision(DecisionKind.RestSite);
            foreach (var opt in opts)
            {
                if (!opt.IsEnabled) continue;
                d.Add(ActionKind.RestOption, opt.OptionId, () => { }, a: opt);
            }
            return d.Choices.Count == 0 ? null : d;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>次のノードの候補を読むだけ（動かさない）。</summary>
    static Decision? BuildMapDecision(RunState run)
    {
        try
        {
            var d = new Decision(DecisionKind.Map);
            foreach (var p in RunSessionMap.Travelable(run).OrderBy(p => p.coord.col))
                d.Add(ActionKind.MapMove, $"→ ({p.coord.col},{p.coord.row}) {p.PointType}", () => { }, a: p);
            return d.Choices.Count == 0 ? null : d;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>直前のメッセージから、id が一致する要素の並び位置を返す。</summary>
    static int IndexOf(string? message, string arrayName, string idField, string id)
    {
        try
        {
            if (message == null) return 0;
            using var doc = JsonDocument.Parse(message);
            if (!doc.RootElement.TryGetProperty(arrayName, out var arr) || arr.ValueKind != JsonValueKind.Array)
                return 0;
            int i = 0;
            foreach (var el in arr.EnumerateArray())
            {
                if (el.TryGetProperty(idField, out var idEl) && idEl.ValueKind == JsonValueKind.String
                    && string.Equals(idEl.GetString(), id, StringComparison.OrdinalIgnoreCase))
                    return i;
                i++;
            }
            return 0;
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>直前に受け取った map_select の nodes[] から、同じ座標の位置を返す。</summary>
    static int MapNodeIndex(MegaCrit.Sts2.Core.Map.MapCoord coord)
    {
        try
        {
            if (_lastMessage == null) return 0;
            using var doc = JsonDocument.Parse(_lastMessage);
            if (!doc.RootElement.TryGetProperty("nodes", out var nodes) || nodes.ValueKind != JsonValueKind.Array)
                return 0;
            int i = 0;
            foreach (var el in nodes.EnumerateArray())
            {
                var c = el.TryGetProperty("coord", out var coordEl) ? coordEl : el;
                int col = c.TryGetProperty("col", out var colEl) ? colEl.GetInt32() : -1;
                int row = c.TryGetProperty("row", out var rowEl) ? rowEl.GetInt32() : -1;
                if (col == coord.col && row == coord.row) return i;
                i++;
            }
            return 0;
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>直前に受け取った card_reward の cards[] から、同じカードの位置を返す。</summary>
    static int CardIndexInMessage(CardModel card)
    {
        try
        {
            if (_lastMessage == null) return 0;
            using var doc = JsonDocument.Parse(_lastMessage);
            if (!doc.RootElement.TryGetProperty("cards", out var cards) || cards.ValueKind != JsonValueKind.Array)
                return 0;
            int i = 0;
            foreach (var el in cards.EnumerateArray())
            {
                string? id = el.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
                    ? idEl.GetString() : null;
                if (string.Equals(id, card.Id.Entry, StringComparison.OrdinalIgnoreCase)) return i;
                i++;
            }
            return 0;
        }
        catch
        {
            return 0;
        }
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
        if (index >= 0 && index < d.Choices.Count && _replies < 60 && reply.Contains("end_turn"))
            Bridge.LogEvent($"[mentor] unmapped choice kind={d.Choices[index].Kind} label='{d.Choices[index].Label}' for {_pendingKind}");
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
            if (choice.Kind == ActionKind.EventOption && choice.Label.Length > 0)
                return "{\"action\":\"choose\",\"index\":" + IndexOf(_lastMessage, "options", "text_key", choice.Label) + "}";
            if (choice.Kind == ActionKind.RestOption && choice.A is MegaCrit.Sts2.Core.Entities.RestSite.RestSiteOption rest)
                return "{\"action\":\"choose\",\"index\":" + IndexOf(_lastMessage, "options", "id", rest.OptionId) + "}";
            if (choice.Kind == ActionKind.MapMove && choice.A is MegaCrit.Sts2.Core.Map.MapPoint point)
                return "{\"index\":" + MapNodeIndex(point.coord) + "}";
            if (choice.Kind == ActionKind.PickCardReward && choice.A is CardModel rewardCard)
                return "{\"action\":\"choose\",\"index\":" + CardIndexInMessage(rewardCard) + "}";
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
        if (_sent++ < 60) Bridge.LogEvent($"[mentor] reply({_pendingKind}): " + json);
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
