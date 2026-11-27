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

    /// <summary>いま開始したランの設定（HUD・計測用）。</summary>
    public static string RunCharacter { get; private set; } = "";
    public static int RunAscension { get; private set; }
    public static int? RunSeed { get; private set; }

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
    static bool _discardNext;
    static bool _reportedEnd;
    static int _cardRewardSkips;
    static int _crystalReplies;
    static int _multiSelectReplies;
    static int _pendingMinSelect;
    static int _crystalStateId = -1;
    static int _crystalSameState;
    static int _crystalCells = 1;
    static int _unmapped;
    static string? _lastMessage;

    public static void Start()
    {
        string? flag = Environment.GetEnvironmentVariable("ALPHADEFECT_MENTOR");
        string modDir = Path.GetDirectoryName(typeof(MentorClient).Assembly.Location) ?? ".";
        Enabled = flag is "1" or "true" or "yes" || File.Exists(Path.Combine(modDir, "mentor.txt"));
        _lastMessageAt = Enabled ? Environment.TickCount64 : 0;
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
        if (PanelState.HandoverPending)
        {
            // 保留中の返答は捨てる（上流 7f0fa3b1 で StopAutopilot が確実に戻るようになった）
            _readyReply = null;
            PanelState.CompleteHandover();
            Bridge.LogEvent("[mentor] handed the game over to the player");
        }
        Watchdog(Environment.TickCount64);
        PumpReady();
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
                if (_discardNext)
                {
                    // 前の要求をタイムアウトで打ち切ったあとに届いた古い返答。使わない。
                    _discardNext = false;
                    Bridge.LogEvent("[mentor] discarded a late inference answer");
                }
                else
                {
                    HoldReply(idx);
                }
                return;
            }
            if (Environment.TickCount64 > _deadline)
            {
                _timeouts++;
                Bridge.LogEvent("[mentor] inference timeout; answering with a legal default for " + _pendingKind);
                Reply(-1);
                _awaiting = false;
                _pendingDecision = null;
                _discardNext = true;
            }
        }
    }

    // 観戦用の保留（実時間の待機・一時停止・1手）
    static string? _readyReply;
    static long _readyAt;
    static int _readyStateId = -1;
    static string _readyLabel = "";
    static bool _heldLogged;

    /// <summary>推論結果をすぐ実行せず、HUD に出してから設定した実時間だけ待つ。</summary>
    static void HoldReply(int index)
    {
        var d = _pendingDecision;
        if (d == null) return;
        string label = index >= 0 && index < d.Choices.Count ? d.Choices[index].Label : "（判断なし）";
        string reply = index >= 0 && index < d.Choices.Count ? Translate(d.Choices[index]) : FallbackForKind(_pendingKind);
        reply = WithIds(reply, _pendingRequestId, _pendingStateId);   // 保留中に送るので、ここで ID を付ける
        int wait = Math.Max(0, PanelState.PacingMs);
        _readyReply = reply;
        _readyLabel = label;
        _readyStateId = _pendingStateId;
        _readyAt = System.Environment.TickCount64 + wait;
        _heldLogged = false;
        PanelState.SetPreview($"[{index}] {label}" + (wait > 0 ? $"（{wait} ms 待って実行）" : ""));
        try
        {
            if (index >= 0 && index < d.Choices.Count)
                MentorHighlight.Show(d.Choices[index].A as MegaCrit.Sts2.Core.Models.CardModel,
                                    d.Choices[index].B as MegaCrit.Sts2.Core.Entities.Creatures.Creature);
        }
        catch (Exception)
        {
            // 強調できなくても実行は続ける
        }
        ModelCounts[_pendingKind] = ModelCounts.TryGetValue(_pendingKind, out int mc) ? mc + 1 : 1;
        Bridge.LogEvent($"[mentor] chosen: {label} (wait {wait} ms, {_pendingKind})");
        _awaiting = false;
        _pendingDecision = null;
    }

    /// <summary>保留中の返答を、停止・1手・待機に従って送る。</summary>
    static void PumpReady()
    {
        if (_readyReply == null) return;
        if (PanelState.Paused)
        {
            if (!_heldLogged) { Bridge.LogEvent("[mentor] held the action (paused)"); _heldLogged = true; }
            return;
        }
        bool step = PanelState.TakeStepOnce();
        if (!step && System.Environment.TickCount64 < _readyAt) return;
        if (_readyStateId >= 0 && _readyStateId != _pendingStateId)
        {
            _stale++;
            Bridge.LogEvent($"[mentor] dropped a stale action (state {_readyStateId} -> {_pendingStateId}): {_readyLabel}");
            _readyReply = null;
            MentorHighlight.Clear();
            return;
        }
        SendReply(_readyReply);
        MentorHighlight.Clear();
        if (step)
        {
            Bridge.LogEvent("[mentor] step: executed one action, pausing");
            PanelState.SetPaused(true, "step");
        }
        _readyReply = null;
        _heldLogged = false;
    }

    // 忠実性の集計（#68）: モデル判断・ルール代替・拒否・タイムアウトを種別ごとに数える
    static readonly Dictionary<string, int> ModelCounts = new();
    static readonly Dictionary<string, int> RuleCounts = new();
    static readonly Dictionary<string, int> NoopCounts = new();
    static int _rejects;
    static int _errorLogged;
    static int _timeouts;
    static int _stale;
    static long _lastMessageAt;
    static int _stallWarned;

    /// <summary>止まったときの保険: 判断が来ないまま時間が過ぎたら記録し、長引けば安全停止する。</summary>
    static void Watchdog(long now)
    {
        if (_lastMessageAt == 0) return;
        long idle = now - _lastMessageAt;
        if (idle > 30000 && _stallWarned == 0)
        {
            _stallWarned = 1;
            Bridge.LogEvent($"[mentor] no decision for {idle / 1000}s; the game may be waiting on a screen we do not answer");
        }
        if (idle > 90000 && _stallWarned == 1 && !PanelState.Paused)
        {
            _stallWarned = 2;
            PanelState.AutoStop("mentor: no decision from the game for 90 s");
        }
    }

    static void HandleMessage(string line)
    {
        _lastMessageAt = Environment.TickCount64;
        _stallWarned = 0;
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
            if (kind is "game_over" or "run_complete")
            {
                if (_reportedEnd)
                {
                    Bridge.LogEvent($"[mentor] '{kind}' after the run already ended; ignoring");
                    return;
                }
                _reportedEnd = true;
                // 勝敗は `victory`（bool）か `result`（"victory"）で来る。`win` は無い。
                bool win = false;
                if (root.TryGetProperty("victory", out var v))
                    win = v.ValueKind == JsonValueKind.True;
                if (!win && root.TryGetProperty("result", out var res) && res.ValueKind == JsonValueKind.String)
                    win = string.Equals(res.GetString(), "victory", StringComparison.OrdinalIgnoreCase);
                if (!win && root.TryGetProperty("win", out var w) && w.ValueKind == JsonValueKind.True)
                    win = true;
                int floor = ReadInt(root, "floor");
                int act = ReadInt(root, "act");
                // `run_complete` は幕の区切りでも送られてくることがある（floor 0）。
                // 本当の終了は `game_over`（勝敗つき）だけを採用する。
                if (kind != "game_over" || floor <= 0)
                {
                    Bridge.LogEvent($"[mentor] '{kind}' without a usable end (win={win} floor={floor}); not recording it");
                    _reportedEnd = false;
                    return;
                }
                Bridge.LogEvent($"[mentor] run finished: win={win} floor={floor} act={act}");
                ReportFidelity();
                Bridge.SendDone(win, floor, act);
                return;
            }
            if (kind is "error" or "pong" or "snapshot_saved"
                or "combat_ended" or "annotation" or "advice_request" or "human_action")
            {
                if (kind == "error" && (_errorLogged < 10 || _errorLogged % 25 == 0))
                {
                    string code = root.TryGetProperty("error", out var c) && c.ValueKind == JsonValueKind.String ? c.GetString() ?? "" : "";
                    string detail = root.TryGetProperty("detail", out var m) && m.ValueKind == JsonValueKind.String ? m.GetString() ?? "" : "";
                    _rejects++;
                    _errorLogged++;
                    Bridge.LogEvent($"[mentor] error from spire-mentor: {code} / {detail} (total {_rejects})");
                }
                if (_ignored++ < 5) Bridge.LogEvent($"[mentor] ignoring '{kind}' (no reply needed)");
                return;
            }
            if (kind == "card_select")
            {
                // 複数枚えらぶ選択（イベントの融合など）は、モデルが 1 枚しか返せないため
                // ルールで先頭から min_select 枚を選ぶ（止まらないことを優先。ログに残す）。
                int minSel = ReadInt(root, "min_select");
                int cardCount = 0;
                try
                {
                    if (root.TryGetProperty("cards", out var cardsEl) && cardsEl.ValueKind == JsonValueKind.Array)
                        cardCount = cardsEl.GetArrayLength();
                }
                catch { }
                if (_multiSelectReplies++ < 5)
                    Bridge.LogEvent($"[mentor] card_select: cards={cardCount} min={minSel} max={ReadInt(root, "max_select")} cancelable={ReadInt(root, "cancelable")}");
                if (minSel > 1 && cardCount > 0)
                {
                    // モデルに 1 枚目を選ばせ、残りは候補の並び順で埋める（先頭固定よりはモデルの意図を残す）
                    _pendingMinSelect = Math.Min(minSel, cardCount);
                    Bridge.LogEvent($"[mentor] card_select needs {minSel} cards: asking the model for the first pick");
                }
            }
            if (kind == "crystal_sphere")
            {
                // ミニゲーム（水晶球）。index だけを返す形式で、進める（proceed）か先頭の未公開セル。
                // これはモデルではなくルール（ログに残す）。
                int index = 0;
                try
                {
                    if (root.TryGetProperty("options", out var opts) && opts.ValueKind == JsonValueKind.Array)
                    {
                        int last = -1, firstCell = -1;
                        foreach (var el in opts.EnumerateArray())
                        {
                            int i = el.TryGetProperty("index", out var iEl) && iEl.ValueKind == JsonValueKind.Number ? iEl.GetInt32() : -1;
                            string action = el.TryGetProperty("action", out var a) && a.ValueKind == JsonValueKind.String ? a.GetString() ?? "" : "";
                            if (action == "proceed") { last = i; continue; }
                            if (firstCell < 0) firstCell = i;
                        }
                        // 未公開セルがあればセルを選ぶ（proceed は「セルが無いとき」だけ）。
                        // proceed を選ぶと、返答が届く前にボタンが無効化されて拒否されることがある。
                        index = firstCell >= 0 ? firstCell : Math.Max(0, last);
                        _crystalCells = Math.Max(1, last >= 0 ? last : firstCell + 1);
                    }
                }
                catch { }
                // 同じ state_id で何度も聞かれるなら、こちらの index が受け付けられていない。
                // セルを順に試して、必ず前へ進むようにする（無限ループの保険）。
                if (stateId >= 0 && stateId == _crystalStateId) _crystalSameState++;
                else { _crystalStateId = stateId; _crystalSameState = 0; }
                if (_crystalSameState >= 2)
                {
                    int cells = Math.Max(1, _crystalCells);
                    index = (_crystalSameState - 2) % cells;
                    Bridge.LogEvent($"[mentor] crystal_sphere: same state asked {_crystalSameState} times; trying index {index}");
                }
                _crystalCells = Math.Max(1, _crystalCells);
                if (_crystalReplies++ < 3)
                    Bridge.LogEvent($"[mentor] crystal_sphere: rule-based answer (index {index}); the minigame is not modelled");
                SendReply(WithIds("{\"index\":" + index + "}", requestId, stateId));
                return;
            }
            if (kind == "run_config")
            {
                _reportedEnd = false;      // 次のラン
                StartRun(root, requestId, stateId);
                return;
            }
            var run = RunManager.Instance.DebugOnlyGetState();
            if (run == null || run.Players.Count == 0) return;
            var me = run.Players[0];
            if (!CheckCharacter(me))
            {
                // 要求と違うキャラで始まった（spire-mentor が黙って別キャラを選ぶことがある）。
                // 観測もモデルも変わるので、そのまま続けない。
                return;
            }
            if (kind == "rest_site")
            {
                // 診断: 休憩所の候補と HP を残す（休憩を選ばない理由の切り分け）
                try
                {
                    var opts = RunManager.Instance.RestSiteSynchronizer?.GetLocalOptions();
                    string ids = opts == null ? "" : string.Join(",", opts.Select(o => o.OptionId + (o.IsEnabled ? "" : "(off)")));
                    Bridge.LogEvent($"[mentor] rest_site at hp={me.Creature.CurrentHp}/{me.Creature.MaxHp} options={ids}");
                }
                catch (Exception)
                {
                }
            }
            Decision? d = kind switch
            {
                "combat_action" => me.PlayerCombatState == null ? null : CombatDriver.BuildCombatDecision(me, me.PlayerCombatState),
                "card_reward" => BuildCardRewardDecision(me, root),
                "rest_site" => BuildRestDecision(),
                "event" => OutOfCombat.BuildEventDecisionForMentor(run, me),
                "map_select" => BuildMapDecision(run),
                "shop" => OutOfCombat.BuildShopDecisionForMentor(run, me),
                "card_select" => BuildCardSelectDecision(me, root),
                "treasure" => BuildTreasureDecisionFromRequest(root, run, me),
                "card_bundle" => BuildCardBundleDecision(root),
                "reward_screen" => RewardScreenDecision(me),
                _ => null,
            };
            if (d == null || d.Choices.Count == 0)
            {
                _unmapped++;
                // 判断の余地が無い画面（終わったイベントの「進む」など）は、モデルの上書きではなく noop として数える
                bool nothingToDecide = kind == "event" && NothingToDecideForEvent();
                var counts = nothingToDecide ? NoopCounts : RuleCounts;
                counts[kind] = counts.TryGetValue(kind, out int rc) ? rc + 1 : 1;
                if (_unmapped <= 5 || _unmapped % 50 == 0)
                    Bridge.LogEvent(nothingToDecide
                        ? $"[mentor] '{kind}': nothing to decide (event already finished); replying with a legal default"
                        : $"[mentor] '{kind}' has no model decision yet ({_unmapped}); replying with a legal default");
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
            _deadline = Environment.TickCount64 + 60000;   // 初回ロード等を見込んで長めに待つ
            Bridge.SendState(frame, _pendingSeq,
                $"\"character\":\"{me.Character?.Id?.Entry ?? ""}\",\"act\":{run.CurrentActIndex},"
                + $"\"floor\":{run.TotalFloor},\"hp\":{me.Creature.CurrentHp},\"mentor\":true",
                CombatDriver.LabelsOf(d));
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
    static int _rewardMissLogged;

    /// <summary>イベントが終わっていて選ぶものが無い（＝モデルの判断を上書きしているわけではない）。</summary>
    static bool NothingToDecideForEvent()
    {
        try
        {
            var ev = RunManager.Instance.EventSynchronizer.GetLocalEvent();
            return ev == null || ev.IsFinished || ev.CurrentOptions.Count == 0;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>報酬画面の判断（こちらの捕捉が無いときは理由を残す。#68）。</summary>
    static Decision? RewardScreenDecision(Player me)
    {
        if (LiveRewards.Active)
        {
            var d = LiveRewards.Build(me);
            if (d != null && d.Choices.Count > 0) return d;
            if (_rewardMissLogged++ < 3) Bridge.LogEvent("[mentor] reward_screen: capture active but no candidates");
            return null;
        }
        if (_rewardMissLogged++ < 3) Bridge.LogEvent("[mentor] reward_screen: no live capture (patch not active)");
        return null;
    }

    static Decision? BuildRestDecision()
    {
        try
        {
            var sync = RunManager.Instance.RestSiteSynchronizer;
            var opts = sync.GetLocalOptions();
            var d = new Decision(DecisionKind.RestSite);
            bool canSmith = opts.Any(o => o is MegaCrit.Sts2.Core.Entities.RestSite.SmithRestSiteOption && o.IsEnabled);
            var run = RunManager.Instance.DebugOnlyGetState();
            var me = run?.Players[0];
            foreach (var opt in opts)
            {
                if (!opt.IsEnabled) continue;
                // ヘッドレス版と同じ: 満血で回復 0・追加効果なし・鍛冶可なら「休む」は出さない
                if (canSmith && me != null && opt is MegaCrit.Sts2.Core.Entities.RestSite.HealRestSiteOption)
                {
                    var (actual, _, extra) = RunSession.RestHealInfo(me);
                    if (actual == 0 && extra == 0) continue;
                }
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
    static int _cardAmbiguous;
    static int _targetRetargeted;
    static int _cardNotFound;

    /// <summary>
    /// 対象を、spire-mentor と同じ「combat.Enemies の並び」の index にする（#68）。
    /// 見つからない・狙えない敵なら、狙える敵の先頭に切り替える（拒否されるより進める）。
    /// </summary>
    static int TargetIndexOf(Creature target)
    {
        var cs = target.CombatState;
        if (cs == null) return -1;
        var enemies = cs.Enemies;
        for (int i = 0; i < enemies.Count; i++)
        {
            if (!ReferenceEquals(enemies[i], target)) continue;
            if (enemies[i].IsHittable) return i;
            break;
        }
        for (int i = 0; i < enemies.Count; i++)
        {
            if (!enemies[i].IsHittable) continue;
            if (_targetRetargeted++ < 5)
                Bridge.LogEvent($"[mentor] target not hittable; retargeting to enemies[{i}]");
            return i;
        }
        return -1;
    }


    /// <summary>
    /// カードをメッセージの並びの index に写す。id だけでなく強化・コストも見て、
    /// 同名カードの取り違えを避ける（#68）。見つからなければ -1。
    /// </summary>
    static int CardIndexInMessage(CardModel card)
    {
        try
        {
            if (_lastMessage == null) return -1;
            using var doc = JsonDocument.Parse(_lastMessage);
            if (!doc.RootElement.TryGetProperty("cards", out var cards) || cards.ValueKind != JsonValueKind.Array)
                return -1;
            int best = -1, bestScore = -1, ties = 0;
            int i = 0;
            foreach (var el in cards.EnumerateArray())
            {
                string? id = el.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
                    ? idEl.GetString() : null;
                if (!string.Equals(id, card.Id.Entry, StringComparison.OrdinalIgnoreCase))
                {
                    i++;
                    continue;
                }
                int score = 1;
                bool upgraded = el.TryGetProperty("upgraded", out var upEl) && upEl.ValueKind == JsonValueKind.True;
                if (upgraded == card.IsUpgraded) score += 2;
                int? cost = el.TryGetProperty("cost", out var cEl) && cEl.ValueKind == JsonValueKind.Number
                    && cEl.TryGetInt32(out int cv) ? cv : null;
                if (cost != null && cost == card.EnergyCost?.GetResolved() ) score += 1;
                if (score > bestScore) { bestScore = score; best = i; ties = 1; }
                else if (score == bestScore) ties++;
                i++;
            }
            if (best < 0)
            {
                if (_cardNotFound++ < 5)
                    Bridge.LogEvent($"[mentor] card {card.Id.Entry} was not in the selection message; not guessing an index");
                return -1;
            }
            if (ties > 1 && _cardAmbiguous++ < 5)
                Bridge.LogEvent($"[mentor] card {card.Id.Entry} matched {ties} entries equally well; using index {best}");
            return best;
        }
        catch
        {
            return -1;
        }
    }

    static int ShopOptionIndex(Choice choice)
    {
        try
        {
            string? id = choice.A switch
            {
                CardModel c => c.Id.Entry,
                RelicModel r => r.Id.Entry,
                PotionModel p => p.Id.Entry,
                _ => null,
            };
            string label = choice.Label;
            if (_lastMessage == null) return 0;
            using var doc = JsonDocument.Parse(_lastMessage);
            if (!doc.RootElement.TryGetProperty("options", out var opts) || opts.ValueKind != JsonValueKind.Array)
                return 0;
            int i = 0;
            foreach (var el in opts.EnumerateArray())
            {
                string? itemId = el.TryGetProperty("item", out var item) && item.ValueKind == JsonValueKind.Object
                    && item.TryGetProperty("id", out var itemIdEl) && itemIdEl.ValueKind == JsonValueKind.String
                    ? itemIdEl.GetString() : null;
                string? optId = el.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
                    ? idEl.GetString() : null;
                if (id != null && string.Equals(itemId, id, StringComparison.OrdinalIgnoreCase)) return i;
                if (id == null && optId != null && label.Contains("删牌") && optId.Contains("remove")) return i;
                i++;
            }
            return 0;
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>報酬画面の「進む」の index。</summary>
    static int ProceedIndex() => IndexOfAction(_lastMessage, "proceed");

    /// <summary>報酬画面で、うちの候補に対応する報酬の index。</summary>
    static int RewardOptionIndex(Choice choice)
    {
        try
        {
            if (_lastMessage == null) return 0;
            // うちの「受け取る」候補のうち何番目か（0 始まり）を数え、相手の pick_reward の同じ順に当てる
            int wanted = 0;
            if (_pendingDecision != null)
            {
                foreach (var c in _pendingDecision.Choices)
                {
                    if (c.Kind != ActionKind.TakeReward) continue;
                    if (ReferenceEquals(c, choice)) break;
                    wanted++;
                }
            }
            using var doc = JsonDocument.Parse(_lastMessage);
            if (!doc.RootElement.TryGetProperty("options", out var opts) || opts.ValueKind != JsonValueKind.Array)
                return 0;
            int i = 0, seen = 0;
            foreach (var el in opts.EnumerateArray())
            {
                string? action = el.TryGetProperty("action", out var a) && a.ValueKind == JsonValueKind.String ? a.GetString() : null;
                if (string.Equals(action, "pick_reward", StringComparison.OrdinalIgnoreCase))
                {
                    if (seen == wanted) return i;
                    seen++;
                }
                i++;
            }
            return 0;
        }
        catch
        {
            return 0;
        }
    }

    static int IndexOfAction(string? message, string action)
    {
        try
        {
            if (message == null) return 0;
            using var doc = JsonDocument.Parse(message);
            if (!doc.RootElement.TryGetProperty("options", out var opts) || opts.ValueKind != JsonValueKind.Array)
                return 0;
            int i = 0;
            foreach (var el in opts.EnumerateArray())
            {
                string? a = el.TryGetProperty("action", out var actionEl) && actionEl.ValueKind == JsonValueKind.String
                    ? actionEl.GetString() : null;
                if (string.Equals(a, action, StringComparison.OrdinalIgnoreCase)) return i;
                i++;
            }
            return 0;
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>忠実性の集計を 1 行で出す（モデル判断とルール代替の比率）。</summary>
    static void ReportFidelity()
    {
        try
        {
            string model = string.Join(",", ModelCounts.Select(kv => $"{kv.Key}={kv.Value}"));
            string rule = string.Join(",", RuleCounts.Select(kv => $"{kv.Key}={kv.Value}"));
            string noop = string.Join(",", NoopCounts.Select(kv => $"{kv.Key}={kv.Value}"));
            Bridge.LogEvent($"[mentor] fidelity: model[{model}] rule[{rule}] noop[{noop}]"
                            + $" rejects={_rejects} timeouts={_timeouts} stale={_stale}");
        }
        catch (Exception)
        {
            // 集計できなくても進行は止めない
        }
    }

    /// <summary>run_config に答えてランを開始する（キャラ・アセンション・シードは mentor.json から）。</summary>
    static void StartRun(JsonElement root, int requestId, int stateId)
    {
        var cfg = ReadChoiceConfig();
        var available = new List<string>();
        int maxAsc = 10;
        try
        {
            if (root.TryGetProperty("characters", out var chars) && chars.ValueKind == JsonValueKind.Array)
            {
                foreach (var c in chars.EnumerateArray())
                {
                    string? id = c.ValueKind == JsonValueKind.String ? c.GetString()
                        : c.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String ? idEl.GetString() : null;
                    if (!string.IsNullOrEmpty(id)) available.Add(id!);
                }
            }
            if (root.TryGetProperty("max_ascension", out var ma) && ma.ValueKind == JsonValueKind.Number) maxAsc = ma.GetInt32();
        }
        catch { }

        string want = (cfg.Character ?? "auto").Trim();
        string character;
        if (string.Equals(want, "select", StringComparison.OrdinalIgnoreCase))
        {
            Bridge.LogEvent("[mentor] character=select: not starting a run; pick the character in game and start it yourself");
            return;
        }
        if (string.Equals(want, "auto", StringComparison.OrdinalIgnoreCase))
        {
            // その版で動くキャラ（auto_order）の順に見て、ゲームが提示する最初のものを選ぶ。
            // ゲームが提示してもこちらにモデルが無いキャラ（例: v0.111.1 の IRONCLAD）は選ばない。
            string? picked = null;
            foreach (var c in cfg.AutoOrder)
            {
                if (available.Count == 0 || available.Any(a => string.Equals(a, c, StringComparison.OrdinalIgnoreCase)))
                {
                    picked = c;
                    break;
                }
            }
            if (picked == null)
            {
                Bridge.LogEvent($"[mentor] cannot start: none of [{string.Join(",", cfg.AutoOrder)}] is offered by the game (available: {string.Join(",", available)})");
                return;
            }
            character = picked.ToUpperInvariant();
            Bridge.LogEvent($"[mentor] character=auto -> {character} (preferred: {string.Join(",", cfg.AutoOrder)}; available: {string.Join(",", available)})");
        }
        else
        {
            character = want.ToUpperInvariant();
            if (available.Count > 0 && !available.Any(a => string.Equals(a, character, StringComparison.OrdinalIgnoreCase)))
            {
                Bridge.LogEvent($"[mentor] cannot start: '{character}' is not offered by the game (available: {string.Join(",", available)})");
                return;
            }
        }
        int ascension = Math.Clamp(cfg.Ascension, 0, Math.Max(0, maxAsc));
        if (cfg.Ascension > maxAsc)
            Bridge.LogEvent($"[mentor] ascension {cfg.Ascension} is above the game maximum {maxAsc}; using {ascension}");
        string seedText = cfg.Seed is int si ? si.ToString(System.Globalization.CultureInfo.InvariantCulture) : "null";
        string reply = "{\"action\":\"start_run\",\"character\":\"" + character + "\",\"ascension\":" + ascension
                       + ",\"seed\":" + (cfg.Seed is int ? seedText : "null") + "}";
        _charChecked = false;
        RunCharacter = character;
        RunAscension = ascension;
        RunSeed = cfg.Seed;
        Bridge.LogEvent($"[mentor] answered run_config: {character} / A{ascension} / seed={(cfg.Seed is int ? seedText : "random")}");
        Bridge.SendRunConfig(character, ascension, cfg.Seed);
        SendReply(WithIds(reply, requestId, stateId));
    }

    sealed class ChoiceConfig
    {
        public string? Character { get; set; }
        public int Ascension { get; set; }
        public int? Seed { get; set; }
        public List<string> AutoOrder { get; } = new();
    }

    /// <summary>mentor.json（キャラ・アセンション・シード）。無ければ既定。</summary>
    static ChoiceConfig ReadChoiceConfig()
    {
        var cfg = new ChoiceConfig { Character = "auto", Ascension = 0, Seed = null };
        try
        {
            string modDir = Path.GetDirectoryName(typeof(MentorClient).Assembly.Location) ?? ".";
            var path = Path.Combine(modDir, "mentor.json");
            if (!File.Exists(path)) return cfg;
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            if (root.TryGetProperty("character", out var c) && c.ValueKind == JsonValueKind.String)
                cfg.Character = c.GetString();
            if (root.TryGetProperty("ascension", out var a) && a.ValueKind == JsonValueKind.Number)
                cfg.Ascension = a.GetInt32();
            if (root.TryGetProperty("auto_order", out var ao) && ao.ValueKind == JsonValueKind.Array)
            {
                foreach (var entry in ao.EnumerateArray())
                    if (entry.ValueKind == JsonValueKind.String && entry.GetString() is { } id)
                        cfg.AutoOrder.Add(id.ToUpperInvariant());
            }
            if (root.TryGetProperty("seed", out var sd))
            {
                if (sd.ValueKind == JsonValueKind.Number && sd.TryGetInt32(out int n)) cfg.Seed = n;
                else if (sd.ValueKind == JsonValueKind.String && int.TryParse(sd.GetString(), out int m)) cfg.Seed = m;
            }
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[mentor] mentor.json read failed: " + e.GetType().Name);
        }
        return cfg;
    }

    /// <summary>宝箱の候補を、相手の options[]（遺物 ID）から作る。開ける前は同期が空なので要求から作る。</summary>
    static Decision? BuildTreasureDecisionFromRequest(JsonElement root, RunState run, Player me)
    {
        try
        {
            if (!root.TryGetProperty("options", out var opts) || opts.ValueKind != JsonValueKind.Array)
                return null;
            var d = new Decision(DecisionKind.Treasure);
            int i = 0;
            foreach (var el in opts.EnumerateArray())
            {
                string? action = el.TryGetProperty("action", out var a) && a.ValueKind == JsonValueKind.String ? a.GetString() : null;
                if (!string.Equals(action, "collect", StringComparison.OrdinalIgnoreCase)) { i++; continue; }
                string? id = el.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String ? idEl.GetString() : null;
                RelicModel? relic = null;
                if (id != null)
                {
                    try { relic = ModelDb.GetById<RelicModel>(new ModelId("RELIC", id)); }
                    catch { relic = null; }
                }
                string label = id != null ? "拿 " + id : "拿 " + (i + 1);
                d.Add(ActionKind.PickRelic, label, () => { }, a: relic, key: id ?? "");
                i++;
            }
            if (d.Choices.Count == 0) return null;
            d.Add(ActionKind.SkipRelic, "跳过", () => { }, key: "SKIP");
            return d;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>カード報酬の候補を、相手の cards[] から作る（mentor モードではこちらの捕捉が無い）。</summary>
    static Decision? BuildCardRewardDecision(Player me, JsonElement root)
    {
        try
        {
            if (!root.TryGetProperty("cards", out var cardsEl) || cardsEl.ValueKind != JsonValueKind.Array)
                return null;
            var d = new Decision(DecisionKind.CardReward);
            foreach (var el in cardsEl.EnumerateArray())
            {
                string? id = el.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
                    ? idEl.GetString() : null;
                if (id == null) continue;
                var card = ResolveCard(me, id);
                if (card == null) continue;
                var c = card;
                d.Add(ActionKind.PickCardReward, "选 " + c.Id.Entry, () => { }, a: c);
            }
            d.Add(ActionKind.CardRewardAlt, "跳过", () => { }, key: "SKIP");
            return d;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>カード ID から CardModel を引く。持っていないカード（報酬・変化）はテンプレートから取る。</summary>
    static CardModel? ResolveCard(Player me, string id)
    {
        var owned = FindOwnedCard(me, id);
        if (owned != null) return owned;
        try
        {
            return ModelDb.GetById<CardModel>(new ModelId("CARD", id));
        }
        catch
        {
            return null;
        }
    }

    static CardModel? FindOwnedCard(Player me, string id)
    {
        foreach (var pile in new[] { PileType.Draw, PileType.Hand, PileType.Discard, PileType.Exhaust })
        {
            try
            {
                var found = pile.GetPile(me).Cards.FirstOrDefault(c => string.Equals(c.Id.Entry, id, StringComparison.OrdinalIgnoreCase));
                if (found != null) return found;
            }
            catch { }
        }
        try { return me.Deck.Cards.FirstOrDefault(c => string.Equals(c.Id.Entry, id, StringComparison.OrdinalIgnoreCase)); }
        catch { return null; }
    }

    /// <summary>カードパック（bundles[]）の候補を、その並びのまま作る。</summary>
    static Decision? BuildCardBundleDecision(JsonElement root)
    {
        try
        {
            if (!root.TryGetProperty("bundles", out var bundles) || bundles.ValueKind != JsonValueKind.Array)
                return null;
            var d = new Decision(DecisionKind.Bundle);
            int i = 0;
            foreach (var bundle in bundles.EnumerateArray())
            {
                int index = i;
                string label = "パック " + (i + 1);
                if (bundle.ValueKind == JsonValueKind.Object && bundle.TryGetProperty("cards", out var cards)
                    && cards.ValueKind == JsonValueKind.Array && cards.GetArrayLength() > 0)
                {
                    label = "パック " + (i + 1) + " (" + cards.GetArrayLength() + "枚)";
                }
                d.Add(ActionKind.PickBundle, label, () => { }, f0: index);
                i++;
            }
            return d.Choices.Count == 0 ? null : d;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>選牌（カード選択）の候補を、相手の cards[] とデッキから作る。</summary>
    static Decision? BuildCardSelectDecision(Player me, JsonElement root)
    {
        try
        {
            if (!root.TryGetProperty("cards", out var cardsEl) || cardsEl.ValueKind != JsonValueKind.Array)
                return null;
            var owned = new List<CardModel>();
            foreach (var pile in new[] { PileType.Draw, PileType.Hand, PileType.Discard, PileType.Exhaust })
            {
                try { owned.AddRange(pile.GetPile(me).Cards); } catch { }
            }
            try { owned.AddRange(me.Deck.Cards); } catch { }   // デッキから選ぶ系（鍛冶など）
            var d = new Decision(DecisionKind.CardSelect);
            foreach (var el in cardsEl.EnumerateArray())
            {
                string? id = el.TryGetProperty("id", out var idEl) && idEl.ValueKind == JsonValueKind.String
                    ? idEl.GetString() : null;
                if (id == null) continue;
                var card = owned.FirstOrDefault(c => string.Equals(c.Id.Entry, id, StringComparison.OrdinalIgnoreCase));
                if (card == null) continue;
                d.Add(ActionKind.SelectCard, "选 " + card.Id.Entry, () => { }, a: card);
            }
            return d.Choices.Count == 0 ? null : d;
        }
        catch
        {
            return null;
        }
    }

    static bool _charChecked;

    /// <summary>要求したキャラと実際に始まったキャラが一致するか。違えば安全停止する（#72）。</summary>
    static bool CheckCharacter(Player me)
    {
        if (_charChecked || RunCharacter.Length == 0) return true;
        _charChecked = true;
        string actual = me.Character?.Id?.Entry ?? "";
        if (actual.Length == 0 || string.Equals(actual, RunCharacter, StringComparison.OrdinalIgnoreCase))
            return true;
        Bridge.LogEvent($"[mentor] character mismatch: requested {RunCharacter}, the game started {actual}; stopping");
        PanelState.AutoStop($"mentor: started character {actual} differs from the request {RunCharacter}");
        return false;
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
        foreach (var kv in dict)
        {
            object? value = kv.Value.ValueKind switch
            {
                JsonValueKind.Number => kv.Value.TryGetInt32(out int n) ? n : kv.Value.GetDouble(),
                JsonValueKind.String => kv.Value.GetString(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                _ => JsonSerializer.Deserialize<object>(kv.Value.GetRawText()),
            };
            outDict[kv.Key] = value;
        }
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
            : FallbackForKind(_pendingKind);
        if (index >= 0 && index < d.Choices.Count && _replies < 60 && reply.Contains("end_turn"))
            Bridge.LogEvent($"[mentor] unmapped choice kind={d.Choices[index].Kind} label='{d.Choices[index].Label}' for {_pendingKind}");
        SendReply(WithIds(reply, _pendingRequestId, _pendingStateId));
    }

    /// <summary>判断できないときの、種類に合った安全な返答。</summary>
    static string FallbackForKind(string kind) => kind switch
    {
        "card_reward" => "{\"action\":\"skip\"}",
        "map_select" => "{\"index\":0}",
        "combat_action" => "{\"action\":\"end_turn\"}",
        _ => "{\"action\":\"choose\",\"index\":0}",
    };

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
                int targetIndex = choice.B is Creature target ? TargetIndexOf(target) : -1;
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
            if (choice.Kind == ActionKind.PickRelic)
            {
                string id = choice.A is RelicModel relic ? relic.Id.Entry : choice.Key;
                return "{\"action\":\"choose\",\"index\":" + IndexOf(_lastMessage, "options", "id", id) + "}";
            }
            if (choice.Kind == ActionKind.SkipRelic)
                return "{\"action\":\"skip\"}";
            if (choice.Kind == ActionKind.PickBundle)
                return "{\"action\":\"choose\",\"index\":" + ((int)choice.F0) + "}";
            if (choice.Kind == ActionKind.Buy)
                return "{\"action\":\"choose\",\"index\":" + ShopOptionIndex(choice) + "}";
            if (choice.Kind == ActionKind.Leave && _pendingKind == "shop")
                return "{\"action\":\"choose\",\"index\":0}";   // 店の 0 は leave_shop
            if (choice.Kind == ActionKind.Leave && _pendingKind == "reward_screen")
                return "{\"action\":\"choose\",\"index\":" + ProceedIndex() + "}";
            if (choice.Kind == ActionKind.TakeReward)
                return "{\"action\":\"choose\",\"index\":" + RewardOptionIndex(choice) + "}";
            if (choice.Kind == ActionKind.SelectCard && choice.A is CardModel picked)
            {
                if (_pendingMinSelect > 1 && _pendingDecision != null)
                {
                    // モデルの 1 枚目 + 残りは候補順（同じカードを重複させない）
                    var pickedIndices = new List<int>();
                    void Take(int i)
                    {
                        if (pickedIndices.Count >= _pendingMinSelect) return;
                        if (!pickedIndices.Contains(i)) pickedIndices.Add(i);
                    }
                    int firstIndex = CardIndexInMessage(picked);
                    if (firstIndex >= 0) Take(firstIndex);
                    foreach (var c in _pendingDecision.Choices)
                    {
                        if (c.Kind != ActionKind.SelectCard || c.A is not CardModel other) continue;
                        Take(CardIndexInMessage(other));
                        if (pickedIndices.Count >= _pendingMinSelect) break;
                    }
                    // 念のため: 枚数が足りないときは有効な index を順に補う（拒否されるより進める）
                    int need = _pendingMinSelect;
                    for (int i = 0; pickedIndices.Count < need; i++)
                    {
                        if (!pickedIndices.Contains(i)) pickedIndices.Add(i);
                        if (i > 64) break;
                    }
                    if (_multiSelectReplies++ < 10)
                        Bridge.LogEvent($"[mentor] card_select multi reply: indices=[{string.Join(",", pickedIndices)}] need={need}");
                    _pendingMinSelect = 0;
                    return "{\"action\":\"choose\",\"indices\":[" + string.Join(",", pickedIndices) + "]}";
                }
                int cardIndex = CardIndexInMessage(picked);
                if (cardIndex < 0)
                {
                    Bridge.LogEvent($"[mentor] no index for {picked.Id.Entry}; answering with a legal default");
                    return "{\"action\":\"choose\",\"indices\":[0]}";
                }
                return "{\"action\":\"choose\",\"indices\":[" + cardIndex + "]}";
            }
            if (choice.Kind == ActionKind.CardRewardAlt)
            {
                // スキップを繰り返すと報酬画面がカード報酬を開き直して止まることがある。
                // 2 回目からは先頭のカードを取って画面を閉じる（ルールベースの代替であることはログに残す）。
                _cardRewardSkips++;
                if (_cardRewardSkips <= 1) return "{\"action\":\"skip\"}";
                Bridge.LogEvent("[mentor] card reward re-opened after a skip; taking the first card instead");
                return "{\"action\":\"choose\",\"index\":0}";
            }
            if (choice.Kind == ActionKind.PickCardReward && choice.A is CardModel rewardCard)
            {
                _cardRewardSkips = 0;
                return "{\"action\":\"choose\",\"index\":" + CardIndexInMessage(rewardCard) + "}";
            }
            if (choice.Kind == ActionKind.UsePotion && choice.A is PotionModel potion)
            {
                var run = RunManager.Instance.DebugOnlyGetState();
                var me = run?.Players[0];
                if (me != null)
                {
                    // スロットは「空きを含む」並びで数える（me.Potions は詰められているので index がずれる）
                    int slot = -1;
                    var slots = me.PotionSlots;
                    for (int i = 0; i < slots.Count; i++)
                    {
                        if (ReferenceEquals(slots[i], potion)) { slot = i; break; }
                    }
                    if (slot >= 0)
                    {
                        // 敵を対象にするポーションは target_index も送る（#67）
                        if (choice.B is Creature target)
                        {
                            int targetIndex = TargetIndexOf(target);
                            if (targetIndex >= 0)
                                return $"{{\"action\":\"potion\",\"slot\":{slot},\"target_index\":{targetIndex}}}";
                        }
                        return $"{{\"action\":\"potion\",\"slot\":{slot}}}";
                    }
                }
                return "{\"action\":\"end_turn\"}";
            }
            // 未対応の種別は、その判断に合った安全な既定にする（戦闘以外に end_turn を返さない）
            return FallbackForKind(_pendingKind);
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[mentor] translate failed: " + e.GetType().Name);
            return FallbackForKind(_pendingKind);
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
