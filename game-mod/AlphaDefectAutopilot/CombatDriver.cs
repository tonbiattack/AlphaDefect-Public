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
    const int MaxSameFrame = 6;          // 同じフレームが続いたら止める（手が効いていない）

    static Decision? _decision;
    static int _seq;
    static bool _awaiting;
    static long _deadline;
    static long _nextAt;
    static string _lastDigest = "";
    static int _sameFrame;
    static bool _wasInCombat;
    static int _steps;

    public static string LastNote { get; private set; } = "";
    public static int Seq => _seq;

    public static void Tick()
    {
        if (Bridge.Stopped || Bridge.Paused) return;
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

        // 戦闘が終わったら数える（AI が打った戦闘の勝利を数える。ヘッダ 12 用）
        if (pcs == null)
        {
            if (_wasInCombat && !me.Creature.IsDead) UiFlows.CombatsWon++;
            _wasInCombat = false;
            _awaiting = false;
            _decision = null;
            if (run.IsGameOver) ReportRunEnd(run, me);
            return;
        }
        _wasInCombat = true;

        if (pcs.Phase != PlayerTurnPhase.Play)
        {
            _awaiting = false;                  // 敵のターン中は何もしない
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
            var session = new RunSession { Steps = ++_steps };
            var frame = Observation.Build(session, d, labels: true).Serialize();
            string digest = Convert.ToHexString(SHA256.HashData(frame)).Substring(0, 16);
            if (digest == _lastDigest) _sameFrame++; else { _sameFrame = 0; _lastDigest = digest; }
            if (_sameFrame >= MaxSameFrame)
            {
                Bridge.LogLine($"same frame {_sameFrame} times; stopping (no progress)");
                Bridge.Stop("no-progress");
                return;
            }
            Bridge.SendState(frame, _seq++, $"\"act\":{run.CurrentActIndex},\"floor\":{run.TotalFloor},\"hp\":{me.Creature.CurrentHp},\"turn\":{pcs.TurnNumber}");
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

    static void Execute(int idx)
    {
        var d = _decision;
        _decision = null;
        _awaiting = false;
        _nextAt = Environment.TickCount64 + SettleMs;
        if (d == null || idx < 0 || idx >= d.Choices.Count)
        {
            Bridge.LogLine($"action index out of range: {idx}");
            return;
        }
        var choice = d.Choices[idx];
        LastNote = choice.Label;
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
                d.Add(ActionKind.PlayCard, $"出 {card.Id.Entry}{TargetLabel(t)}", () => card.TryManualPlay(target), a: card, b: t);
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
