using System.Security.Cryptography;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// 戦闘の出牌を AI に決めさせる部分（実機）。
///
/// 流れはヘッドレスの評価と同じ:
///   1. 出牌できる局面（PlayerTurnPhase.Play）で、無頭エンジンと同じ候補を作る
///      （手札 × 対象、薬水、ターン終了）
///   2. エンジンと同じ観測エンコーダでフレームを作り、推論サーバへ送る
///   3. 返ってきた添字の候補を実行する（出牌は CardModel.TryManualPlay、終了は EndPlayerTurnAction）
///
/// ゲームの描画は本体が持っているので、こちらは状態を読んで手を選び、モデルの呼び出しだけ行う。
/// 実行はメインスレッド（NRun._Process の後処理）で、返ってきた Task は待たずに走らせる。
/// </summary>
internal static class CombatDriver
{
    const int ActionTimeoutMs = 15000;   // 行動が返らないときの諦め
    const int SettleMs = 350;            // 実行後、ゲームが落ち着くまで次の判断を待つ
    /// <summary>行動の間の待ち時間（パネルの「間隔」が指定されていればそちら）。</summary>
    static int EffectiveSettle() => Math.Max(SettleMs, PanelState.PacingMs);
    const int MaxSameFrame = 6;          // 同じフレームが続いたら止める（手が効いていない）

    static Decision? _decision;
    static int _seq;
    static bool _awaiting;
    static long _deadline;
    static long _nextAt;
    static string _lastDigest = "";
    static int _sameFrame;
    static bool _wasInCombat;
    static string _lastWaitingRoom = "";
    /// <summary>前回と同じ状態（＝直前に実行した行動が効いていない）なら、その候補を外して送り直す。</summary>
    static bool SendDecision(Decision d, RunSession session, string meta)
    {
        byte[] frame = Observation.Build(session, d, labels: true).Serialize();
        string digest = Digest(frame);
        string signature = StateSignature();
        if (_lastExecutedLabel != null && _lastSignature == signature)
        {
            if (_failedLabels.Add(_lastExecutedLabel))
                Bridge.LogEvent($"[driver] {_lastExecutedLabel} had no effect; excluding it");
        }
        if (_failedLabels.Count > 0)
        {
            int removed = d.Choices.RemoveAll(c => _failedLabels.Contains(c.Label));
            if (removed > 0)
            {
                frame = Observation.Build(session, d, labels: true).Serialize();
                digest = Digest(frame);
            }
            if (d.Choices.Count == 0)
            {
                PanelState.AutoStop("no candidate changes the state (all had no effect)");
                return false;
            }
        }
        _lastFrameDigest = digest;
        _lastSignature = signature;
        _sentContext = ContextKey();
        Bridge.SendState(frame, _seq++, meta);
        return true;
    }

    static readonly HashSet<string> _failedLabels = new();

    /// <summary>局面が変わったら「効かなかった候補」の記憶を捨てる。</summary>
    static void NoteStateKey(RunState run, Player me)
    {
        int hp = me.Creature.CurrentHp;
        int turn = 0;
        try
        {
            if (MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress)
                turn = me.PlayerCombatState?.TurnNumber ?? 0;
        }
        catch { }
        string key = $"{run.TotalFloor}|{run.CurrentActIndex}|{turn}|{hp}|{run.CurrentRoom?.GetType().Name}";
        if (key != _stateKey)
        {
            _stateKey = key;
            if (_failedLabels.Count > 0) _failedLabels.Clear();
        }
    }

    static string _stateKey = "";
    static string _lastSignature = "";

    /// <summary>ゲームの状態そのものの指紋（フレームの Steps は毎回変わるので使わない）。</summary>
    static string StateSignature()
    {
        try
        {
            var run = RunManager.Instance.DebugOnlyGetState();
            if (run == null || run.Players.Count == 0) return "none";
            var me = run.Players[0];
            var pcs = me.PlayerCombatState;
            string hand = string.Join(",", PileType.Hand.GetPile(me).Cards.Select(c => c.Id.Entry));
            string enemies = "";
            if (SafeInCombat())
            {
                enemies = string.Join(",", MegaCrit.Sts2.Core.Combat.CombatManager.Instance.DebugOnlyGetState()
                    ?.Creatures.Where(c => !c.IsPlayer).Select(c => $"{c.Monster?.Id.Entry}:{c.CurrentHp}:{c.Block}") ?? Array.Empty<string>());
            }
            return $"{run.TotalFloor}|{run.CurrentActIndex}|{run.CurrentRoom?.GetType().Name}|{run.VisitedMapCoords.Count}"
                   + $"|{pcs?.TurnNumber}|{pcs?.Energy}|{me.Creature.CurrentHp}|{me.Creature.Block}|{hand}|{enemies}";
        }
        catch (Exception e)
        {
            return "sig-error:" + e.GetType().Name;
        }
    }
    static string _sentContext = "";

    /// <summary>いまの局面（戦闘中か・部屋の種類）。判断を送ったときと違えば、その判断は捨てる。</summary>
    static string ContextKey()
    {
        try
        {
            var run = RunManager.Instance.DebugOnlyGetState();
            return (SafeInCombat() ? "combat" : "out") + "|" + (run?.CurrentRoom?.GetType().Name ?? "null");
        }
        catch
        {
            return "unknown";
        }
    }

    static string Digest(byte[] frame)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(frame));
    }

    static string? _lastFrameDigest;
    static string? _lastExecutedLabel;

    static bool _prevInCombat;
    static bool _cardRewardStop;

    /// <summary>戦闘終了・カード報酬で自動停止する（設定されているときだけ）。</summary>
    static void AutoStopPoints()
    {
        bool inCombat = SafeInCombat();
        if (_prevInCombat && !inCombat && PanelState.StopOnCombatEnd)
            PanelState.AutoStop("combat finished");
        _prevInCombat = inCombat;

        bool cardReward = LiveCardReward.Active;
        if (cardReward && !_cardRewardStop && PanelState.StopOnCardReward)
        {
            _cardRewardStop = true;
            PanelState.AutoStop("card reward screen");
        }
        if (!cardReward) _cardRewardStop = false;
    }
    static long _lastDiag;
    static int _steps;
    static int _awaitEpoch;

    public static string LastNote { get; private set; } = "";
    public static int Seq => _seq;

    public static void Tick()
    {
        if (Bridge.Stopped) return;
        AutoStopPoints();
        // 戦闘に入った／部屋が変わったのに前の判断を待っているなら、その判断は捨てる
        if ((_awaiting || _decision != null) && _sentContext.Length > 0 && _sentContext != ContextKey())
        {
            Bridge.LogEvent($"[driver] situation changed ({_sentContext} -> {ContextKey()}); dropping the pending decision");
            _awaiting = false;
            _decision = null;
            _sentContext = "";
        }
        try
        {
            var st = RunManager.Instance.DebugOnlyGetState();
            var p0 = st?.Players != null && st.Players.Count > 0 ? st.Players[0] : null;
            if (st != null && p0 != null) NoteStateKey(st, p0);
        }
        catch { }
        if (AsyncGate.Busy)
        {
            if (nowDiag(3000)) Bridge.LogLine($"[wait] game is busy with {AsyncGate.LastOp}; not deciding yet");
            return;
        }
        if (_awaiting && _awaitEpoch != PanelState.Epoch)
        {
            // 手動切り替え・1 手実行・再開で古い判断は無効。応答は捨てて次の状態から作り直す。
            _awaiting = false;
            Bridge.LogLine("epoch changed: dropped the waiting decision");
        }
        RunState? run;
        try
        {
            run = RunManager.Instance.DebugOnlyGetState();
        }
        catch (Exception e)
        {
            Bridge.LogLine("state read failed: " + e.GetType().Name);
            return;
        }
        if (run == null)
        {
            _wasInCombat = false;
            return;
        }

        var me = run.Players[0];
        var pcs = me.PlayerCombatState;
        long now = Environment.TickCount64;
        if (PanelState.AwaitingApproval) return;      // ユーザーの許可待ち: 新しい状態は送らない
        if (!PanelState.CanAct) return;
        // 選牌待ち（捨てる・強化する等）は戦闘の内外を問わず先に片付ける
        if (LiveCardSelect.Active)
        {
            DriveCardSelect(run, me, now);
            return;
        }
        // 戦闘中かどうかは CombatManager で決める。報酬画面の間も PlayerCombatState は
        // 残っていることがあるので、それで判断すると戦闘外の判断に降りられない。
        bool inCombat = SafeInCombat();

        // 死んだ／終わった: それ以上フレームを送らない（送り続けるとゲーム側の表示と噛み合わない）
        if (run.IsGameOver || me.Creature.IsDead)
        {
            _awaiting = false;
            _decision = null;
            if (_wasInCombat || run.IsGameOver) ReportRunEnd(run, me);
            _wasInCombat = false;
            return;
        }

        // 戦闘が終わったら数える（AI が打った戦闘の勝利を数える。ヘッダ 12 用）
        if (!inCombat)
        {
            if (_wasInCombat && !me.Creature.IsDead) UiFlows.CombatsWon++;
            _wasInCombat = false;
            if (run.IsGameOver) { ReportRunEnd(run, me); return; }
            // 戦闘の外（報酬・経路）。判断できる局面だけ扱い、それ以外はプレイヤーに任せる
            DriveOutOfCombat(run, me, now);
            return;
        }
        _wasInCombat = true;

        if (pcs == null || pcs.Phase != PlayerTurnPhase.Play)
        {
            _awaiting = false;                  // 敵のターン中は何もしない
            if (now - _lastDiag > 2000)
            {
                _lastDiag = now;
                Bridge.LogLine($"in a fight but not in the play phase: phase={pcs?.Phase} inCombat={inCombat} room={run.CurrentRoom?.GetType().Name}");
            }
            return;
        }
        if (!_awaiting && now < _nextAt) return;

        if (_awaiting)
        {
            int idx = Bridge.TakeAction();
            if (idx >= 0)
            {
                Execute(idx);
                return;
            }
            if (now > _deadline)
            {
                Bridge.LogLine("action timeout; resend state");
                _awaiting = false;
            }
            return;
        }

        // 出牌できる局面: 候補を作ってフレームを送る
        try
        {
            var d = BuildDecision(me, pcs);
            if (d.Choices.Count == 0) return;
            var session = new RunSession { Steps = _steps };
            string character = me.Character?.Id?.Entry ?? "";
            if (!SendDecision(d, session, $"\"character\":\"{character}\",\"act\":{run.CurrentActIndex},\"floor\":{run.TotalFloor},\"hp\":{me.Creature.CurrentHp},\"turn\":{pcs.TurnNumber}"))
                return;
            _decision = d;
            _awaiting = true;
            _deadline = now + ActionTimeoutMs;
        }
        catch (Exception e)
        {
            Bridge.LogLine("build/encode failed: " + e.GetType().Name + ": " + e.Message);
            _nextAt = now + 1000;
        }
    }

    /// <summary>報酬・経路の決定を送って実行する（戦闘外）。</summary>
    static void DriveOutOfCombat(RunState run, Player me, long now)
    {
        if (_awaiting)
        {
            int idx = Bridge.TakeAction();
            if (idx >= 0) { Execute(idx); return; }
            if (PanelState.AwaitingApproval && now < _deadline) return;   // 許可待ち
            if (now > _deadline) { _awaiting = false; }
            return;
        }
        if (now < _nextAt) return;
        try
        {
            var d = OutOfCombat.Build(run, me);
            if (d == null || d.Choices.Count == 0)
            {
                _decision = null;
                if (d == null)
                {
                    _nextAt = now + 300;                   // 準備中・未対応: すぐ再確認（実時間の待ちは入場待ちで行う）
                    string room = run.CurrentRoom?.GetType().Name ?? "null";
                    if (room != _lastWaitingRoom)
                    {
                        _lastWaitingRoom = room;
                        Bridge.LogLine($"waiting for the player here: {room} rewards={LiveRewards.Active} cards={LiveCardReward.Active} inCombat={SafeInCombat()}");
                    }
                }
                return;
            }
            var session = new RunSession { Steps = _steps };
            if (!SendDecision(d, session, $"\"character\":\"{me.Character?.Id?.Entry ?? ""}\",\"act\":{run.CurrentActIndex},\"floor\":{run.TotalFloor},\"hp\":{me.Creature.CurrentHp}"))
                return;
            _awaitEpoch = PanelState.Epoch;
            _decision = d;
            _awaiting = true;
            _deadline = now + ActionTimeoutMs;
        }
        catch (Exception e)
        {
            Bridge.LogLine("out-of-combat build failed: " + e.GetType().Name + ": " + e.Message);
            _nextAt = now + 1000;
        }
    }

    /// <summary>診断ログを間引く（同じメッセージを毎フレーム出さない）。</summary>
    static bool nowDiag(int intervalMs)
    {
        long now = System.Environment.TickCount64;
        if (now - _lastDiagAt < intervalMs) return false;
        _lastDiagAt = now;
        return true;
    }

    static long _lastDiagAt;

    static bool SafeInCombat()
    {
        try { return MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress; }
        catch { return false; }
    }

    /// <summary>選牌の決定を送って実行する（1 枚ずつ選び、最後に確定）。</summary>
    static void DriveCardSelect(RunState run, Player me, long now)
    {
        if (_awaiting)
        {
            int idx = Bridge.TakeAction();
            if (idx >= 0) { Execute(idx); return; }
            if (PanelState.AwaitingApproval) return;      // 確認モード: 許可待ちのまま待つ
            if (now > _deadline) _awaiting = false;
            return;
        }
        if (now < _nextAt) return;
        try
        {
            var d = LiveCardSelect.Build(run);
            if (d == null) { _nextAt = now + 200; return; }     // 自動解決した等
            if (!SendDecision(d, new RunSession { Steps = _steps }, $"\"character\":\"{me.Character?.Id?.Entry ?? ""}\",\"act\":{run.CurrentActIndex},\"floor\":{run.TotalFloor},\"hp\":{me.Creature.CurrentHp}"))
                return;
            _awaitEpoch = PanelState.Epoch;
            _decision = d;
            _awaiting = true;
            _deadline = now + ActionTimeoutMs;
        }
        catch (Exception e)
        {
            Bridge.LogLine("card select build failed: " + e.GetType().Name + ": " + e.Message);
            _nextAt = now + 1000;
        }
    }

    static void Execute(int idx)
    {
        var d = _decision;
        _decision = null;
        _awaiting = false;
        _nextAt = Environment.TickCount64 + EffectiveSettle();
        PanelState.AfterStep();
        if (d == null || idx < 0 || idx >= d.Choices.Count)
        {
            Bridge.LogLine($"action index out of range: {idx}");
            return;
        }
        var choice = d.Choices[idx];
        LastNote = choice.Label;
        _lastExecutedLabel = choice.Label;
        _steps++;                    // 意思決定の数（ヘッダ 13）。局面が同じ間は増やさない（重複検出のため）
        try
        {
            var task = choice.Run();
            task.ContinueWith(t =>
            {
                if (t.IsFaulted) Bridge.LogLine("action failed: " + t.Exception?.GetBaseException().Message);
            }, TaskScheduler.Default);
        }
        catch (Exception e)
        {
            Bridge.LogLine("action threw: " + e.GetType().Name + ": " + e.Message);
        }
    }

    /// <summary>ヘッドレスの CombatDecision と同じ候補（手札×対象、薬水、ターン終了）。</summary>
    static Decision BuildDecision(Player me, PlayerCombatState pcs)
    {
        var d = new Decision(DecisionKind.Combat) { Context = $"第 {pcs.TurnNumber} 回合" };
        var cs = me.Creature.CombatState;
        if (cs == null) return d;
        var targets = new List<Creature?> { null };
        targets.AddRange(cs.Creatures.Where(c => c.IsAlive));
        foreach (CardModel c in PileType.Hand.GetPile(me).Cards)
        {
            CardModel card = c;
            if (!card.CanPlay(out _, out _)) continue;
            foreach (var t in targets)
            {
                if (!card.CanPlayTargeting(t)) continue;
                var target = t;
                string label = $"出 {card.Id.Entry}{TargetLabel(t)}";
                d.Add(ActionKind.PlayCard, label, () => PlayChecked(card, target, label), a: card, b: t);
            }
        }
        foreach (var p in UsablePotions(me))
        {
            PotionModel potion = p;
            foreach (var t in targets)
            {
                if (!potion.IsValidTarget(t)) continue;
                var target = t;
                d.Add(ActionKind.UsePotion, $"用 {potion.Id.Entry}{TargetLabel(t)}", () => potion.EnqueueManualUse(target), a: potion, b: t);
            }
        }
        int turn = pcs.TurnNumber;
        d.Add(ActionKind.EndTurn, "结束回合", () =>
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new EndPlayerTurnAction(me, turn)));
        return d;
    }

    static string TargetLabel(Creature? t) => t == null ? "" : t.IsPlayer ? " → 自己" : $" → {t.Monster?.Id.Entry}";

    /// <summary>
    /// 出牌を実行する前に、いま出せるかを確かめる。
    /// 判断に時間がかかる間に状況（エネルギー・対象・フェーズ）が変わると、ゲーム側は出牌を
    /// 受け付けず、何も起きないまま同じ判断を繰り返す（画面ではカードが何度も出てくるように見える）。
    /// その場合はその候補を当面の候補から外す。
    /// </summary>
    static Task PlayChecked(CardModel card, Creature target, string label)
    {
        try
        {
            if (!card.CanPlayTargeting(target))
            {
                Bridge.LogEvent($"[driver] play refused (the state changed): {label}");
                _failedLabels.Add(label);
                return Task.CompletedTask;
            }
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[driver] play check failed: " + e.GetType().Name);
            return Task.CompletedTask;
        }
        if (!card.TryManualPlay(target))
        {
            Bridge.LogEvent($"[driver] play refused by the game: {label}");
            _failedLabels.Add(label);
        }
        return Task.CompletedTask;
    }

    static IEnumerable<PotionModel> UsablePotions(Player me) =>
        me.CanUseOrRemovePotions && !me.Creature.IsDead
            ? me.Potions.Where(p => !p.IsQueued && p.PassesCustomUsabilityCheck &&
                                    (p.Usage == PotionUsage.AnyTime || p.Usage == PotionUsage.CombatOnly))
            : Enumerable.Empty<PotionModel>();

    static void ReportRunEnd(RunState run, Player me)
    {
        if (Bridge.RunReported) return;
        // プレイヤーが生きていれば勝利とみなす（ゲーム内の正確な判定は今後の課題）
        Bridge.SendDone(!me.Creature.IsDead, run.TotalFloor, run.CurrentActIndex);
        Bridge.LogLine($"run finished (win={!me.Creature.IsDead}) floor={run.TotalFloor}");
    }
}
