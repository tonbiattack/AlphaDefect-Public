using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Game;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// 戦闘の外の意思決定（報酬・経路）。無頭エンジンの RunSession にある判断と実行を、
/// 実機のゲームでもそのまま使えるように移したもの。
///
/// 報酬はエンジンの RewardsFlow をそのまま使う（RewardsSet の testSelector を差し替える
/// 仕組みで、エンジンでも同じものを使っている）。こちらが選んだら
/// RunManager.RewardsSetSynchronizer.SelectLocalReward でゲーム側に反映される。
///
/// まだ無いもの: イベント・ショップ・休息・宝箱・選牌。そこはゲームが待つので、
/// その局面ではフレームを送らずに止まる（プレイヤーが操作すれば続く）。
/// </summary>
internal static class OutOfCombat
{
    static bool _installed;
    static bool _chestOpened;
    static bool _relicDecided;
    static bool _shopLeft;
    static int _restWaits;
    const long RestReadyMs = 2500;      // 画面が出てから選択肢が揃うまでの猶予
    static bool _roomLeft;
    static bool _eventLeft;
    static object? _lastRoom;

    public static void Install()
    {
        if (_installed) return;
        RewardsFlow.Install();          // RewardsSet.testSelector = RewardsFlow.Select
        _installed = true;
        Bridge.LogLine("reward flow installed (test selector)");
    }

    public static void ResetForNewRun() => RewardsFlow.Reset();

    /// <summary>いま判断すべき戦闘外の決定。無ければ null。</summary>
    public static Decision? Build(RunState run, Player me)
    {
        // 部屋が変わったら「この部屋ではもう用は済んだ」系のフラグを戻す（エンジンと同じ）
        long nowMs = System.Environment.TickCount64;
        if (!ReferenceEquals(run.CurrentRoom, _lastRoom))
        {
            _lastRoom = run.CurrentRoom;
            _currentRoomName = run.CurrentRoom?.GetType().Name ?? "null";
            _roomEnteredAt = nowMs;
            _entryLogged = false;
            _shopLeft = false;
            _roomLeft = false;
            _eventLeft = false;
            _chestOpened = false;
            _relicDecided = false;
            _restWaits = 0;
            _restChosen = false;
            _failedLeaves = 0;
            _leaveStuckLogged = false;
            _lastLeaveAttempt = 0;
            _firstFailedAt = 0;
            _sawRoomScreen = false;
            _screenClosedLogged = false;
            _mapLogged = false;
            _restLogged = false;
            _shopLogged = false;
            Bridge.LogLine($"[room] now in {_currentRoomName}");
        }
        // 部屋の「進む」を押したあとは、その部屋では地図の判断に切り替える
        // （ボタンは地図画面を開くだけで、部屋のオブジェクトはすぐには変わらない）
        if (_roomLeft) return MapDecision(run, me);
        // 部屋の画面が閉じた（地図が開いた等）なら退出済み。部屋のモデルが残っていても
        // その部屋の操作はしない（「進む」が押せないまま繰り返すのを防ぐ）。
        if (_currentRoomName is not ("null" or "MapRoom"))
        {
            bool screen = FindRoomNode(_currentRoomName) != null;
            if (screen) _sawRoomScreen = true;
            else if (_sawRoomScreen && nowMs - _roomEnteredAt > 500)
            {
                if (!_screenClosedLogged)
                {
                    _screenClosedLogged = true;
                    Bridge.LogLine($"[room] {_currentRoomName} screen closed; going back to the map");
                }
                return MapDecision(run, me);
            }
        }
        if (LiveBundle.Active && LiveBundle.Build() is { } bundles) return bundles;
        if (LiveCardReward.Active && LiveCardReward.Build() is { } cardReward) return cardReward;
        if (LiveRewards.Active && LiveRewards.Build(me) is { } rewards) return rewards;
        if (RewardsFlow.Pending is { } pr) return RewardsDecision(me, pr);
        if (!LiveBundle.Active && !LiveCardReward.Active && !LiveRewards.Active)
        {
            // どの画面も出ていない: 古い捕捉は捨てる
        }
        if (run.CurrentRoom is EventRoom)
        {
            if (EventDecision(run, me) is { } ev) return ev;
            // イベントが終わったのに部屋が残っている: 画面の「続ける」を押して抜ける
            // （NEventRoom は終了時に PROCEED の 1 択を作り、押すと Proceed() が走る）
            if (!_eventLeft)
            {
                _eventLeft = true;
                try
                {
                    _ = MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom.Proceed();
                    Bridge.LogLine("event finished: pressed continue");
                }
                catch (Exception e)
                {
                    Bridge.LogLine("event continue failed: " + e.GetType().Name + ": " + e.Message);
                }
                return null;
            }
            return MapDecision(run, me);
        }
        // 戦闘が終わって報酬も閉じたあとは、部屋のオブジェクトがまだ CombatRoom のことがある。
        // エンジンと同じく、他にすることが無ければ地図の選択に落とす。
        if (run.CurrentRoom is RestSiteRoom) return RestDecision(me);
        if (run.CurrentRoom is TreasureRoom tr) return TreasureDecision(tr, me);
        if (run.CurrentRoom is MerchantRoom)
            // 店を出たあとは地図（NMerchantRoom.HideScreen が地図を開く）を選ぶ
            return _shopLeft ? MapDecision(run, me) : ShopDecision(run, me);
        if (run.CurrentRoom is MapRoom or CombatRoom || run.CurrentRoom is null) return MapDecision(run, me);
        return null;
    }

    /// <summary>エンジンの RewardsDecision と同じ並び（領得できる報酬、薬水、離開）。</summary>
    static Decision RewardsDecision(Player me, PendingRewards pr)
    {
        var d = new Decision(DecisionKind.Rewards) { Context = pr.IsTerminal ? "战斗奖励" : "额外奖励" };
        // カード報酬は画面（NCardRewardSelectionScreen）の選択待ちになる。画面の操作はまだ
        // 実装していないので、いまは候補から外して「取らない」で進む（デッキは育たない）。
        var available = pr.Available.Where(r => r is not CardReward).ToList();
        bool cardOnly = pr.Available.Count > 0 && available.Count == 0;
        foreach (var r in available)
            d.Add(ActionKind.TakeReward, "领 " + RewardLabel(r), () => pr.Take(r), a: r);
        if (!pr.Set.DisallowSkipping || available.Count == 0)
            d.Add(ActionKind.Leave, "离开", pr.Proceed);
        if (cardOnly && pr.Set.DisallowSkipping)
            Bridge.LogLine("card reward cannot be skipped here; the game will wait for the player");
        return d;
        AddPotionActions(me, d);
        if (!me.HasOpenPotionSlots && pr.Set.Rewards.Any(r => r is PotionReward && !r.SuccessfullySelected))
            AddPotionDiscards(me, d);
    }

    /// <summary>イベントの選択（エンジンの EventDecision と同じ。Neow もここで進む）。</summary>
    static Decision? EventDecision(RunState run, Player me)
    {
        var ev = RunManager.Instance.EventSynchronizer.GetLocalEvent();
        if (ev == null || ev.IsFinished) return null;
        var d = new Decision(DecisionKind.Event) { Context = ev.Id.Entry };
        var opts = ev.CurrentOptions;
        for (int i = 0; i < opts.Count; i++)
        {
            var opt = opts[i];
            if (opt.IsLocked || ExcludedEventOptions.Contains(opt.TextKey)) continue;
            int k = i;
            // NEventRoom.OptionButtonClicked: 「進む」系は Chosen を直接、それ以外は同期器経由
            if (opt.IsProceed)
                d.Add(ActionKind.EventOption, opt.TextKey, () => opt.Chosen(), a: opt);
            else
                d.Add(ActionKind.EventOption, opt.TextKey, () => RunManager.Instance.EventSynchronizer.ChooseLocalOption(k), a: opt);
        }
        if (d.Choices.Count == 0) return null;
        AddPotionActions(me, d);
        return d;
    }

    /// <summary>選ばない選択肢（エンジンと同じ。TRIAL の DOUBLE_DOWN は放棄確認だけなので出さない）。</summary>
    static readonly HashSet<string> ExcludedEventOptions = new() { "TRIAL.pages.REJECT.options.DOUBLE_DOWN" };

    /// <summary>休息所: 使える選択肢を候補に（回復・鍛造など）。</summary>
    static Decision? RestDecision(Player me)
    {
        var sync = RunManager.Instance.RestSiteSynchronizer;
        var opts = sync.GetLocalOptions();
        var d = new Decision(DecisionKind.RestSite);
        for (int i = 0; i < opts.Count; i++)
        {
            if (!opts[i].IsEnabled) continue;
            int k = i;
            d.Add(ActionKind.RestOption, opts[i].OptionId, () => sync.ChooseLocalOption(k), a: opts[i]);
        }
        if (_restChosen)
        {
            // もう選んだ: 選牌（鍛冶）が終わったら退出するだけ。同じ選択を押し続けない。
            if (LeaveIsStuck()) return null;
            d.Add(ActionKind.Leave, "离开", () => LeaveRestSite());
            return d;
        }
        if (d.Choices.Count == 0)
        {
            // 同期器に選択肢が無くても、画面にボタンが出ていることがある（休息所の 休憩/鍛冶）。
            // 画面のボタンから候補を作り、押して選ぶ（プレイヤーのクリックと同じ）。
            if (AddRestButtons(d)) return d;
            // 画面が出ていないだけなのか、もう選び終えて退出待ちなのかを分ける。
            long now = System.Environment.TickCount64;
            long waited = now - _roomEnteredAt;
            bool screen = FindRoomNode("NRestSiteRoom") != null;
            if (!screen)
            {
                if (!_restLogged) { _restLogged = true; Bridge.LogLine("[rest] waiting for the room screen"); }
                return null;                                  // 表示待ち: すぐ再確認する
            }
            if (WaitEntry(now)) return null;                   // 観戦用の入場待ち
            if (waited < RestReadyMs)
            {
                if (!_restLogged)
                {
                    _restLogged = true;
                    Bridge.LogLine($"[rest] screen is up, options are not ready yet (waited {waited}ms)");
                }
                return null;
            }
            // ここまで来たら「もう選び終えた」か「この部屋に選択肢が無い」。
            // 押せる「進む」があるときだけ退出する（押せないなら待つ）。
            if (LeaveIsStuck()) return null;
            d.Add(ActionKind.Leave, "离开", () => LeaveRestSite());
            Bridge.LogLine($"[rest] no options after {waited}ms; leaving with the proceed button");
            return d;
        }
        _restLogged = true;
        if (!_restOptionsLogged)
        {
            _restOptionsLogged = true;
            Bridge.LogLine($"[rest] options={d.Choices.Count}: " + string.Join(", ", d.Choices.Select(c => c.Label)));
        }
        AddPotionActions(me, d);
        return d;
    }

    /// <summary>いまの画面に、その型名の部屋ノードがあるか（＝画面が初期化されたか）。</summary>
    static Godot.Node? FindRoomNode(string roomTypeName)
    {
        try
        {
            var scene = MegaCrit.Sts2.Core.Nodes.NGame.Instance?.RootSceneContainer?.CurrentScene;
            if (scene == null) return null;
            foreach (var node in scene.FindChildren("*", "", true, false))
                if (node.GetType().Name == roomTypeName) return node;
        }
        catch
        {
            // 画面の読み取り失敗は「まだ無い」と同じ扱い
        }
        return null;
    }

    static int _failedLeaves;
    static bool _leaveStuckLogged;
    static long _lastLeaveAttempt;
    static long _firstFailedAt;
    const long LeaveRetryMs = 1200;      // 退出の再試行は 1.2 秒以上あける（演出待ちを潰さない）
    const int LeaveGiveUpCount = 4;      // 4 回失敗し、かつ
    const long LeaveGiveUpMs = 6000;     // 最初の失敗から 6 秒経っても駄目なら安全停止
    static bool _screenClosedLogged;

    /// <summary>部屋を出る。押せなかったら数える（押せないまま何度も選ばないため）。</summary>
    static void LeaveVia(string roomTypeName)
    {
        long now = System.Environment.TickCount64;
        if (now - _lastLeaveAttempt < LeaveRetryMs)
        {
            Bridge.LogLine($"[room] leave attempt spacing ({now - _lastLeaveAttempt}ms); waiting");
            return;                                  // 連打しない（演出待ちを潰さない）
        }
        _lastLeaveAttempt = now;
        if (PressRoomProceed(roomTypeName))
        {
            _roomLeft = true;
            _failedLeaves = 0;
            _firstFailedAt = 0;
            return;
        }
        if (_failedLeaves == 0) _firstFailedAt = now;
        _failedLeaves++;
        Bridge.LogLine($"[room] cannot leave {roomTypeName} yet (attempt {_failedLeaves}, "
                       + $"{now - _firstFailedAt}ms since the first attempt)");
    }

    /// <summary>退出できないまま繰り返していないか。十分待っても駄目なら安全停止する。</summary>
    static bool LeaveIsStuck()
    {
        long now = System.Environment.TickCount64;
        if (_failedLeaves < LeaveGiveUpCount) return false;
        if (_firstFailedAt > 0 && now - _firstFailedAt < LeaveGiveUpMs) return false;
        // 「進む」が無効のまま: 地図の移動を直接積んで部屋を抜ける（休息所の鍛冶後など）。
        if (EscapeViaMap())
        {
            _failedLeaves = 0;
            _roomLeft = true;
            Bridge.LogLine("[room] left via a map move instead of the proceed button");
            return false;
        }
        if (!_leaveStuckLogged)
        {
            _leaveStuckLogged = true;
            Bridge.LogLine($"[room] stuck in {_currentRoomName}: the proceed button stays disabled");
        }
        PanelState.AutoStop("stuck in " + _currentRoomName + ": the proceed button is disabled");
        return true;
    }

    /// <summary>休息所の画面ボタン（休憩・鍛冶など）を候補にする。押せたら true。</summary>
    static bool AddRestButtons(Decision d)
    {
        try
        {
            var node = FindRoomNode("NRestSiteRoom");
            if (node == null) return false;
            foreach (var child in node.FindChildren("*", "", true, false))
            {
                if (child.GetType().Name != "NRestSiteButton") continue;
                if (child is not MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl clickable) continue;
                if (!clickable.IsEnabled) continue;
                var option = child.GetType().GetProperty("Option")?.GetValue(child);
                if (option is not RestSiteOption rest) continue;
                string label = rest.OptionId;
                d.Add(ActionKind.RestOption, label, () => PressRestButton(clickable, label), a: rest);
            }
        }
        catch (Exception e)
        {
            Bridge.LogLine("[rest] reading the screen buttons failed: " + e.GetType().Name);
        }
        return d.Choices.Count > 0;
    }

    static void PressRestButton(Godot.Node button, string label)
    {
        if (button is not MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl clickable)
        {
            Bridge.LogLine("[rest] button is not clickable: " + label);
            return;
        }
        AsyncGate.RunSync("rest choose " + label, () => clickable.EmitSignal(
            MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl.SignalName.Released, clickable));
        _restChosen = true;
        Bridge.LogLine("[rest] chose " + label);
    }

    /// <summary>宝箱の「開ける」ボタンを押す（UI 自身の流れを進める）。</summary>
    static bool PressChestButton()
    {
        try
        {
            var node = FindRoomNode("NTreasureRoom");
            if (node == null) return false;
            foreach (var child in node.FindChildren("*", "", true, false))
            {
                if (child.GetType().Name != "NTreasureButton") continue;
                if (child is not MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl clickable) continue;
                if (!clickable.IsEnabled) continue;
                AsyncGate.RunSync("open chest button", () => clickable.EmitSignal(
                    MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl.SignalName.Released, clickable));
                Bridge.LogLine("[treasure] pressed the chest button");
                return true;
            }
            return false;
        }
        catch (Exception e)
        {
            Bridge.LogLine("[treasure] chest press failed: " + e.GetType().Name + ": " + e.Message);
            return false;
        }
    }

    /// <summary>部屋から出る代わりに、次のノードへ直接移動する（出口ボタンが無効のとき）。</summary>
    static bool EscapeViaMap()
    {
        try
        {
            var run = RunManager.Instance.DebugOnlyGetState();
            if (run == null) return false;
            var me = MegaCrit.Sts2.Core.Runs.RunManager.Instance.DebugOnlyGetState().Players[0];
            var next = RunSessionMap.Travelable(run).OrderBy(p => p.coord.col).FirstOrDefault();
            if (next == null) return false;
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                new MegaCrit.Sts2.Core.GameActions.MoveToMapCoordAction(me, next.coord));
            return true;
        }
        catch (Exception e)
        {
            Bridge.LogLine("[room] escape move failed: " + e.GetType().Name + ": " + e.Message);
            return false;
        }
    }

    /// <summary>休息所を出る（押せたら出た扱いにする）。</summary>
    static void LeaveRestSite() => LeaveVia("NRestSiteRoom");

    /// <summary>部屋の「進む」ボタンを押す（部屋を出る）。押せたら true。</summary>
    static bool PressRoomProceed(string roomTypeName)
    {
        try
        {
            var scene = MegaCrit.Sts2.Core.Nodes.NGame.Instance?.RootSceneContainer?.CurrentScene;
            if (scene == null) return false;
            foreach (var node in scene.FindChildren("*", "", true, false))
            {
                if (node.GetType().Name != roomTypeName) continue;
                if (node is not MegaCrit.Sts2.Core.Nodes.Rooms.IRoomWithProceedButton holder) continue;
                var button = holder.ProceedButton;
                if (button == null || !Godot.GodotObject.IsInstanceValid(button)) continue;
                if (!button.IsEnabled)
                {
                    Bridge.LogLine($"proceed in {roomTypeName} is disabled; waiting");
                    return false;
                }
                AsyncGate.RunSync("proceed " + roomTypeName, () =>
                    button.EmitSignal(MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl.SignalName.Released, button));
                _restWaits = 0;
                return true;
            }
            Bridge.LogLine($"proceed button not found in {roomTypeName}");
            return false;
        }
        catch (Exception e)
        {
            Bridge.LogLine($"proceed in {roomTypeName} failed: {e.GetType().Name}: {e.Message}");
            return false;
        }
    }

    /// <summary>部屋に入ってから、操作できるようになるまで少し待つ（観戦しやすいように）。</summary>
    static bool WaitEntry(long now)
    {
        long waited = now - _roomEnteredAt;
        if (waited >= PanelState.EntryDelayMs) return false;
        if (!_entryLogged)
        {
            _entryLogged = true;
            Bridge.LogLine($"[room] entered {_currentRoomName}: waiting {PanelState.EntryDelayMs}ms before acting");
        }
        return true;
    }

    static bool _restLogged;
    static bool _restChosen;
    static bool _restOptionsLogged;
    static bool _shopLogged;
    static bool _shopOptionsLogged;
    static bool _entryLogged;
    static bool _sawRoomScreen;
    static bool _mapLogged;
    static bool _mapTreeLogged;

    /// <summary>地図画面のノード構成を 1 回だけ記録する（点ボタンが見つからないときの手がかり）。</summary>
    static void DumpMapTree()
    {
        try
        {
            var screen = MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance;
            if (screen == null)
            {
                Bridge.LogLine("[map] tree: NMapScreen.Instance is null");
                return;
            }
            var types = screen.FindChildren("*", "", true, false).Take(40)
                .Select(n => n.GetType().Name).ToList();
            Bridge.LogLine("[map] tree: " + string.Join(", ", types));
        }
        catch (Exception e)
        {
            Bridge.LogLine("[map] tree dump failed: " + e.GetType().Name);
        }
    }



    /// <summary>
    /// 経路投票の出所を、いまの RunState の現在地に合わせる。
    /// 部屋を出た直後は MapSelectionSynchronizer が古い出所（開始前の座標）を持ったままで、
    /// 投票が「出所が違う」として捨てられることがあるため。
    /// </summary>
    static void SyncAcceptingSource(RunState run)
    {
        try
        {
            var sync = RunManager.Instance.MapSelectionSynchronizer;
            var field = sync.GetType().GetField("_acceptingVotesFromSource",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            if (field == null) return;
            if (field.GetValue(sync) is not MapLocation current) return;
            var want = run.MapLocation;
            if (current.actIndex == want.actIndex && Nullable.Equals(current.coord, want.coord)) return;
            sync.OnLocationChanged(want);
            Bridge.LogLine($"[map] accepting source synced to act {want.actIndex} coord "
                           + (want.coord.HasValue ? $"({want.coord.Value.col},{want.coord.Value.row})" : "(null)"));
        }
        catch (Exception e)
        {
            Bridge.LogLine("[map] source sync failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    static bool _mapWaitLogged;
    static string _lastMoveDest = "";
    static long _lastMoveAt;
    static int _moveSkips;
    static long _roomEnteredAt;
    static string _currentRoomName = "";

    /// <summary>宝箱: 開けて、遺物が 1 件なら取り、複数なら選ぶ。</summary>
    static Decision? TreasureDecision(TreasureRoom room, Player me)
    {
        if (!_chestOpened)
        {
            _chestOpened = true;
            // 画面の宝箱ボタンを押す（UI 側が開封・演出・遺物選択を進める）。
            // 押せないときだけモデル側で開ける。
            if (!PressChestButton())
            {
                Bridge.LogLine("[treasure] chest button not found; opening from the model");
                try
                {
                    AsyncGate.Run("open chest", async () =>
                    {
                        await room.DoNormalRewards();
                        await room.DoExtraRewardsIfNeeded();
                    });
                }
                catch (Exception e)
                {
                    Bridge.LogLine("open chest failed: " + e.GetType().Name);
                }
            }
            return null;                       // 開けた次のフレームで判断
        }
        if (_relicDecided)
        {
            // 遺物は決めたのに部屋が残っている: 部屋の「進む」で出る（押せなければ止める）
            if (LeaveIsStuck()) return null;
            var leave = new Decision(DecisionKind.Treasure);
            leave.Add(ActionKind.Leave, "离开", () => LeaveVia("NTreasureRoom"));
            return leave;
        }
        var sync = RunManager.Instance.TreasureRoomRelicSynchronizer;
        var relics = sync.CurrentRelics;
        if (relics == null || relics.Count == 0)
        {
            _relicDecided = true;
            return null;
        }
        var d = new Decision(DecisionKind.Treasure);
        for (int i = 0; i < relics.Count; i++)
        {
            int k = i;
            d.Add(ActionKind.PickRelic, "拿 " + relics[i].Id.Entry, () => PickRelic(k), a: relics[i]);
        }
        d.Add(ActionKind.SkipRelic, "跳过", () => SkipRelic(), key: "SKIP");
        return d;
    }

    static async Task OpenChest(TreasureRoom room)
    {
        await room.DoNormalRewards();
        await room.DoExtraRewardsIfNeeded();
        Bridge.LogLine("chest opened");
    }

    static void PickRelic(int index)
    {
        _relicDecided = true;
        _ = PickRelicAsync(index);
    }

    static async Task PickRelicAsync(int index)
    {
        try
        {
            var sync = RunManager.Instance.TreasureRoomRelicSynchronizer;
            sync.PickRelicLocally(index);
            await RunManager.Instance.ProceedFromTerminalRewardsScreen();
            Bridge.LogLine($"picked treasure relic {index}");
        }
        catch (Exception e)
        {
            Bridge.LogLine("pick relic failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    static void SkipRelic()
    {
        _relicDecided = true;
        try
        {
            RunManager.Instance.TreasureRoomRelicSynchronizer.CompleteWithNoRelics();
            _ = RunManager.Instance.ProceedFromTerminalRewardsScreen();
            Bridge.LogLine("skipped treasure relic");
        }
        catch (Exception e)
        {
            Bridge.LogLine("skip relic failed: " + e.GetType().Name);
        }
    }

    /// <summary>店: 買える商品を候補にし、「出る」も出す（エンジンの ShopDecision と同じ並び）。</summary>
    static Decision? ShopDecision(RunState run, Player me)
    {
        if (run.CurrentRoom is not MerchantRoom room) return null;
        long now = System.Environment.TickCount64;
        if (FindRoomNode("NMerchantRoom") == null)
        {
            if (!_shopLogged) { _shopLogged = true; Bridge.LogLine("[shop] waiting for the room screen"); }
            return null;                                  // 表示待ち
        }
        if (WaitEntry(now)) return null;                   // 観戦用の入場待ち（既定 1.5 秒）
        if (!_shopOptionsLogged)
        {
            _shopOptionsLogged = true;
            Bridge.LogLine("[shop] gold=" + me.Gold);
        }
        var inv = room.GetLocalInventory();
        var d = new Decision(DecisionKind.Shop) { Context = $"金币 {me.Gold}" };
        foreach (var entry in inv.AllEntries)
        {
            if (!entry.IsStocked) continue;
            if (!entry.EnoughGold) { d.Locked.Add((EntryObject(entry), entry.Cost)); continue; }
            if (entry is MerchantPotionEntry && !me.HasOpenPotionSlots) continue;
            if (entry is MerchantCardRemovalEntry { Used: true }) continue;
            var e = entry;
            d.Add(ActionKind.Buy, $"买 {EntryLabel(entry)} ({entry.Cost})", () => Buy(e, inv), a: EntryObject(entry), f0: entry.Cost);
        }
        d.Add(ActionKind.Leave, "离开", () => LeaveShop());
        AddPotionActions(me, d);
        return d;
    }

    static void Buy(MerchantEntry entry, MerchantInventory inv)
    {
        string label = EntryLabel(entry);
        int cost = entry.Cost;
        int goldBefore = RunManager.Instance.DebugOnlyGetState()?.Players[0]?.Gold ?? 0;
        Bridge.LogLine($"[shop] buy {label} for {cost}g (gold {goldBefore})");
        AsyncGate.Run("buy " + label, async () =>
        {
            bool ok = await entry.OnTryPurchaseWrapper(inv);
            int goldAfter = RunManager.Instance.DebugOnlyGetState()?.Players[0]?.Gold ?? 0;
            Bridge.LogLine($"[shop] buy {label}: {(ok ? "ok" : "failed")}, gold {goldBefore} -> {goldAfter}"
                           + $", stocked={entry.IsStocked}");
        });
    }

    static object EntryObject(MerchantEntry e) => e switch
    {
        MerchantCardEntry c => c.CreationResult.Card,
        MerchantRelicEntry { Model: { } r } => r,
        MerchantPotionEntry { Model: { } p } => p,
        _ => e,
    };

    static string EntryLabel(MerchantEntry e) => e switch
    {
        MerchantCardEntry c => c.CreationResult.Card.Id.Entry,
        MerchantRelicEntry r => r.Model?.Id.Entry ?? "遗物",
        MerchantPotionEntry p => p.Model?.Id.Entry ?? "药水",
        MerchantCardRemovalEntry => "删牌",
        _ => e.GetType().Name,
    };

    static void LeaveShop()
    {
        try
        {
            // 部屋のノードは「進むボタンを持つ部屋」として取れる（IRoomWithProceedButton）。
            // そのボタンにクリックと同じ Released 信号を送る。
            var scene = MegaCrit.Sts2.Core.Nodes.NGame.Instance?.RootSceneContainer?.CurrentScene;
            if (scene != null)
            {
                foreach (var node in scene.FindChildren("*", "", true, false))
                {
                    if (node is not MegaCrit.Sts2.Core.Nodes.Rooms.IRoomWithProceedButton holder) continue;
                    var button = holder.ProceedButton;
                    if (button == null || !Godot.GodotObject.IsInstanceValid(button)) continue;
                    button.EmitSignal(MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl.SignalName.Released, button);
                    Bridge.LogLine("left the shop");
                    _shopLeft = true;
                    _roomLeft = true;
                    return;
                }
            }
            Bridge.LogLine("shop leave button not found; the game will wait for the player");
        }
        catch (Exception e)
        {
            Bridge.LogLine("leave shop failed: " + e.GetType().Name + ": " + e.Message);
        }
    }

    static Decision MapDecision(RunState run, Player me)
    {
        var cur = run.CurrentMapPoint;
        var d = new Decision(DecisionKind.Map)
        {
            Context = $"第 {run.CurrentActIndex + 1} 幕 {(cur == null ? "起点" : $"({cur.coord.col},{cur.coord.row})")}",
        };
        if (MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance == null)
        {
            if (!_mapWaitLogged)
            {
                _mapWaitLogged = true;
                Bridge.LogLine("[map] waiting for the map screen");
            }
            return null;                        // 画面が開いてから動かす
        }
        _mapWaitLogged = false;
        var travel = RunSessionMap.Travelable(run).OrderBy(p => p.coord.col).ToList();
        if (!_mapLogged)
        {
            _mapLogged = true;
            Bridge.LogLine($"[map] visited={string.Join(" ", run.VisitedMapCoords.Select(c => $"({c.col},{c.row})"))}"
                           + $" travelable={string.Join(" ", travel.Select(p => $"({p.coord.col},{p.coord.row}){p.PointType}"))}");
        }
        foreach (var p in travel)
        {
            var dest = p;
            d.Add(ActionKind.MapMove, $"→ ({p.coord.col},{p.coord.row}) {p.PointType}", () => MoveTo(dest, run, me), a: p);
        }
        AddPotionActions(me, d);
        return d;
    }

    /// <summary>
    /// 次のノードへ移る。シングルプレイなのでホストと同じ「移動」を直接積む（投票は出所の
    /// 判定で捨てられることがある）。画面の点があればプレイヤーと同じクリックを優先する。
    /// </summary>
    static void MoveTo(MapPoint dest, RunState run, Player me)
    {
        // 同じ目的地へ連続で積まない（移動の演出中に何度も積むと二重操作になる）。
        long now = System.Environment.TickCount64;
        string key = $"{dest.coord.col},{dest.coord.row}";
        if (key == _lastMoveDest)
        {
            // 同じ目的地への再試行は 3 秒以上あける。何度も駄目なら安全停止する
            // （毎フレーム積むと二重操作になり、ログも埋まる）。
            if (now - _lastMoveAt < 3000)
            {
                _moveSkips++;
                if (_moveSkips == 4)
                    Bridge.LogLine($"[map] move to {key} does not seem to take effect");
                if (_moveSkips >= 8) PanelState.AutoStop("map move had no effect: " + key);
                return;
            }
            _moveSkips++;
            if (_moveSkips >= 8)
            {
                PanelState.AutoStop("map move had no effect: " + key);
                return;
            }
        }
        else
        {
            _moveSkips = 0;
        }
        _lastMoveDest = key;
        _lastMoveAt = now;
        // シングルプレイなので、ホストと同じ「移動」を直接積むのが最も確実
        // （投票は出所の判定で捨てられ、点のクリックは画面外のノードで無視されることがある）。
        try
        {
            RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                new MegaCrit.Sts2.Core.GameActions.MoveToMapCoordAction(me, dest.coord));
            Bridge.LogLine($"[map] move → ({dest.coord.col},{dest.coord.row})");
            return;
        }
        catch (Exception e)
        {
            Bridge.LogLine("[map] move enqueue failed: " + e.GetType().Name + ": " + e.Message);
        }
        if (ClickMapPoint(dest.coord))
        {
            Bridge.LogLine($"[map] clicked point ({dest.coord.col},{dest.coord.row})");
            return;
        }
        var vote = new MapVote
        {
            coord = dest.coord,
            mapGenerationCount = RunManager.Instance.MapSelectionSynchronizer.MapGenerationCount,
        };
        var source = new MapLocation(run.CurrentMapCoord, run.CurrentActIndex);
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new VoteForMapCoordAction(me, source, vote));
        Bridge.LogLine($"[map] vote → ({dest.coord.col},{dest.coord.row})");
    }

    /// <summary>地図の点をクリックする（NMapPoint.OnRelease → 画面が正しい出所で移動する）。</summary>
    static bool ClickMapPoint(MapCoord coord)
    {
        try
        {
            var screen = MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen.Instance;
            if (screen == null || !Godot.GodotObject.IsInstanceValid(screen)) return false;
            foreach (var node in screen.FindChildren("*", "", true, false))
            {
                if (node is not MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapPoint point) continue;
                if (point.Point.coord != coord) continue;
                if (point is not MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl clickable) return false;
                if (!clickable.IsEnabled) return false;
                AsyncGate.RunSync("map click", () => clickable.EmitSignal(
                    MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl.SignalName.Released, clickable));
                return true;
            }
        }
        catch (Exception e)
        {
            Bridge.LogLine("[map] point click failed: " + e.GetType().Name + ": " + e.Message);
        }
        return false;
    }

    static void AddPotionActions(Player me, Decision d)
    {
        foreach (var p in UsablePotions(me))
        {
            PotionModel potion = p;
            if (potion.IsValidTarget(me.Creature) || potion.IsValidTarget(null))
                d.Add(ActionKind.UsePotion, "用 " + potion.Id.Entry, () => potion.EnqueueManualUse(null), a: potion);
        }
    }

    static void AddPotionDiscards(Player me, Decision d)
    {
        if (!me.CanUseOrRemovePotions) return;
        var slots = me.PotionSlots;
        for (int i = 0; i < slots.Count; i++)
        {
            if (slots[i] is not { IsQueued: false } potion) continue;
            uint slot = (uint)i;
            d.Add(ActionKind.DiscardPotion, "丢弃 " + potion.Id.Entry, () =>
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                    new DiscardPotionGameAction(me, slot, CombatInProgress)), a: potion);
        }
    }

    static bool CombatInProgress
    {
        get
        {
            try { return MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress; }
            catch { return false; }
        }
    }

    static IEnumerable<PotionModel> UsablePotions(Player me) =>
        me.CanUseOrRemovePotions && !me.Creature.IsDead
            ? me.Potions.Where(p => !p.IsQueued && p.PassesCustomUsabilityCheck &&
                                    p.Usage == PotionUsage.AnyTime)
            : Enumerable.Empty<PotionModel>();

    static string RewardLabel(Reward r) => r switch
    {
        GoldReward => "金币",
        PotionReward p => "药水 " + p.Potion?.Id.Entry,
        RelicReward rr => "遗物 " + rr.Relic?.Id.Entry,
        CardReward => "卡牌奖励",
        _ => r.GetType().Name,
    };
}
