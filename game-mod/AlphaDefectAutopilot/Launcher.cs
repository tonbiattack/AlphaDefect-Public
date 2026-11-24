using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Unlocks;

namespace AlphaDefectAutopilot;

/// <summary>
/// 推論サーバからの指示で、ゲーム側から 1 局を始める（観戦の起動を簡単にするため）。
///
/// ゲーム本来の開始経路 `NGame.StartNewSingleplayerRun` を使うので、画面も普通に
/// ランへ切り替わる。そのあと最初の戦闘へ直接入る（報酬・経路・イベントの判断は
/// まだ実装していないため、まず戦闘を観戦できるようにする）。
///
/// 指示の形: "DEFECT|0|ABC123|regular"（キャラ|アセンション|シード|遭遇の種類）
/// 遭遇の種類は weak / regular / elite / boss、または ENCOUNTER.XXX。
/// </summary>
internal static class Launcher
{
    static volatile string _request = "";
    static volatile bool _busy;

    public static void Request(string spec)
    {
        _request = spec ?? "";
    }

    public static bool Busy => _busy;

    /// <summary>メインスレッドから呼ぶ。開始要求があれば開始して true。</summary>
    public static bool Tick()
    {
        string spec = Interlocked.Exchange(ref _request, "");
        if (spec.Length == 0 || _busy) return false;
        _busy = true;
        _ = StartAsync(spec);
        return true;
    }

    static async Task StartAsync(string spec)
    {
        try
        {
            string[] parts = spec.Split('|');
            string characterId = parts.ElementAtOrDefault(0) is { Length: > 0 } c ? c : "DEFECT";
            int ascension = int.TryParse(parts.ElementAtOrDefault(1), out int a) ? a : 0;
            string seed = parts.ElementAtOrDefault(2) is { Length: > 0 } s ? s : "ALPHADEFECT";
            string encounter = parts.ElementAtOrDefault(3) is { Length: > 0 } e ? e : "regular";

            var character = ModelDb.AllCharacters.FirstOrDefault(m => m.Id.Entry == characterId)
                            ?? throw new InvalidOperationException($"unknown character {characterId}");
            var actRng = new Rng(StringHelper.GetDeterministicHashCode(seed), "act_selection");
            var acts = ActModel.GetRandomList(actRng, UnlockState.all, isMultiplayer: false).ToList();

            Bridge.LogLine($"starting run {characterId} A{ascension} seed={seed} encounter={encounter}");
            await NGame.Instance!.StartNewSingleplayerRun(character, shouldSave: false, acts,
                Array.Empty<ModifierModel>(), seed, GameMode.Standard, ascension);

            var state = RunManager.Instance.DebugOnlyGetState();
            if (state == null) throw new InvalidOperationException("run did not start");
            var enc = PickEncounter(state, seed, encounter);
            Bridge.LogLine($"entering combat: {enc.Id.Entry} ({enc.RoomType})");
            await RunManager.Instance.EnterRoomDebug(enc.RoomType, model: enc.ToMutable(), showTransition: false);
        }
        catch (Exception e)
        {
            Bridge.LogLine("start failed: " + e.GetType().Name + ": " + e.Message);
        }
        finally
        {
            _busy = false;
        }
    }

    /// <summary>無頭エンジンの CombatSetup.PickEncounter と同じ選び方（種子で決まる）。</summary>
    static EncounterModel PickEncounter(RunState state, string seed, string kind)
    {
        var act = state.Acts[0];
        IEnumerable<EncounterModel> pool = kind.ToLowerInvariant() switch
        {
            "weak" => act.AllWeakEncounters,
            "regular" => act.AllRegularEncounters,
            "elite" => act.AllEliteEncounters,
            "boss" => act.AllBossEncounters,
            _ => new[] { ModelDb.GetById<EncounterModel>(new ModelId("ENCOUNTER", kind.Replace("ENCOUNTER.", ""))) },
        };
        var list = pool.ToList();
        if (list.Count == 0) throw new InvalidOperationException($"act 1 has no {kind} encounters");
        var rng = new Rng((uint)StringHelper.GetDeterministicHashCode(seed), "combat_spec_encounter");
        return rng.NextItem(list)!;
    }
}
