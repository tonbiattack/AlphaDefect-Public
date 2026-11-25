using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Entities.Merchant;
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
        if (!ReferenceEquals(run.CurrentRoom, _lastRoom))
        {
            _lastRoom = run.CurrentRoom;
            _shopLeft = false;
            _roomLeft = false;
            _eventLeft = false;
            _chestOpened = false;
            _relicDecided = false;
        }
        // 部屋の「進む」を押したあとは、その部屋では地図の判断に切り替える
        // （ボタンは地図画面を開くだけで、部屋のオブジェクトはすぐには変わらない）
        if (_roomLeft) return MapDecision(run, me);
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
        if (d.Choices.Count == 0)
        {
            // まだ選択肢が用意されていない（部屋の画面が開く前）: 少し待つ。
            // 待っても出ないなら「何もせず出る」でランを止めない。
            if (_restWaits++ < 20) return null;
            d.Add(ActionKind.Leave, "离开", () => PressRoomProceed("NRestSiteRoom"));
            Bridge.LogLine("rest site has no options; leaving");
            return d;
        }
        AddPotionActions(me, d);
        return d;
    }

    /// <summary>部屋の「進む」ボタンを押す（部屋を出る）。</summary>
    static void PressRoomProceed(string roomTypeName)
    {
        try
        {
            var scene = MegaCrit.Sts2.Core.Nodes.NGame.Instance?.RootSceneContainer?.CurrentScene;
            if (scene == null) return;
            foreach (var node in scene.FindChildren("*", "", true, false))
            {
                if (node.GetType().Name != roomTypeName) continue;
                if (node is not MegaCrit.Sts2.Core.Nodes.Rooms.IRoomWithProceedButton holder) continue;
                var button = holder.ProceedButton;
                if (button == null || !Godot.GodotObject.IsInstanceValid(button)) continue;
                button.EmitSignal(MegaCrit.Sts2.Core.Nodes.GodotExtensions.NClickableControl.SignalName.Released, button);
                Bridge.LogLine($"pressed proceed in {roomTypeName}");
                _restWaits = 0;
                _roomLeft = true;
                return;
            }
            Bridge.LogLine($"proceed button not found in {roomTypeName}");
        }
        catch (Exception e)
        {
            Bridge.LogLine($"proceed in {roomTypeName} failed: {e.GetType().Name}");
        }
    }

    /// <summary>宝箱: 開けて、遺物が 1 件なら取り、複数なら選ぶ。</summary>
    static Decision? TreasureDecision(TreasureRoom room, Player me)
    {
        if (!_chestOpened)
        {
            _chestOpened = true;
            try
            {
                _ = OpenChest(room);
            }
            catch (Exception e)
            {
                Bridge.LogLine("open chest failed: " + e.GetType().Name);
            }
            return null;                       // 開けた次のフレームで判断
        }
        if (_relicDecided)
        {
            // 遺物は決めたのに部屋が残っている: 部屋の「進む」で出る
            var leave = new Decision(DecisionKind.Treasure);
            leave.Add(ActionKind.Leave, "离开", () => PressRoomProceed("NTreasureRoom"));
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
        _ = BuyAsync(entry, inv);
    }

    static async Task BuyAsync(MerchantEntry entry, MerchantInventory inv)
    {
        try
        {
            bool ok = await entry.OnTryPurchaseWrapper(inv);
            Bridge.LogLine($"buy {EntryLabel(entry)}: {(ok ? "ok" : "failed")}");
        }
        catch (Exception e)
        {
            Bridge.LogLine("buy failed: " + e.GetType().Name + ": " + e.Message);
        }
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
        _shopLeft = true;
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
        foreach (var p in RunSessionMap.Travelable(run).OrderBy(p => p.coord.col))
        {
            var dest = p;
            d.Add(ActionKind.MapMove, $"→ ({p.coord.col},{p.coord.row}) {p.PointType}", () =>
            {
                var vote = new MapVote
                {
                    coord = dest.coord,
                    mapGenerationCount = RunManager.Instance.MapSelectionSynchronizer.MapGenerationCount,
                };
                RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(
                    new VoteForMapCoordAction(me, run.MapLocation, vote));
            }, a: p);
        }
        AddPotionActions(me, d);
        return d;
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
