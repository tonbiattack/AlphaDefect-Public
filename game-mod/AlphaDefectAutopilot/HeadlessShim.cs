using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace Sts2Headless;

/// <summary>
/// ゲーム内で、無頭エンジンの観測エンコーダ（Engine/Obs/**）をそのまま使うための最小の代替物。
///
/// エンコーダは `RunSession` の静的メンバー（現在の RunState）といくつかのインスタンス
/// メンバーだけを無頭ホストから借りている。実機では RunState は RunManager が持っているので、
/// それを返すだけでよい。使わないメンバー（選牌・勝利判定など）は戦闘のフレーム用に固定値を返す。
/// </summary>
public sealed class RunSession
{
    /// <summary>いま進んでいる一局の状態（無頭エンジンの RunSession.State と同じ）。</summary>
    public static RunState State => RunManager.Instance.DebugOnlyGetState()!;

    /// <summary>実機はいまの一局の途中。戦闘だけの会話ではない。</summary>
    public bool CombatOnly => false;

    /// <summary>勝利フレームは出さない（ゲーム終了は done で伝える）。</summary>
    public bool IsVictory => false;

    /// <summary>この層で動かした意思決定の数（ヘッダ 13）。</summary>
    public int Steps { get; set; }

    /// <summary>選牌の待ちは無い（いまは戦闘の出牌だけを扱う）。</summary>
    public PendingCardChoice? Selecting => null;

    public IReadOnlyList<int> SelectedIndices => Array.Empty<int>();

    /// <summary>休息所の回復量（イベントのフレームだけが使う。いまは 0 を返す）。</summary>
    internal static (int actual, int nominal, int extra) RestHealInfo(Player p) => (0, 0, 0);
}

/// <summary>勝利した戦闘の数（ヘッダ 12）。</summary>
public static class UiFlows
{
    public static int CombatsWon { get; set; }
}

/// <summary>選牌の用途（エンコーダの選牌プロンプトだけが参照する）。</summary>
public sealed record SelectInfo(string? PromptTable, string? PromptKey, string Method, EnchantmentModel? Enchantment, AbstractModel? Source = null);

/// <summary>選牌の待ち（いまは作らないが、エンコーダのコンパイルに要る）。</summary>
public sealed class PendingCardChoice(IReadOnlyList<CardModel> options, int min, int max, bool cancelable, SelectInfo? info = null)
{
    public SelectInfo? Info { get; } = info;
    public IReadOnlyList<CardModel> Options { get; } = options;
    public int Min { get; } = min;
    public int Max { get; } = max;
    public bool Cancelable { get; } = cancelable;
}

/// <summary>エンコーダが参照する選牌器の状態（いまは常に無し）。</summary>
public static class AgentCardSelector
{
    public static PotionModel? UsingPotion => null;
}

/// <summary>無頭ホストの偽画面（実機では本物の画面があるので、状態は持たない）。</summary>
public static class FakeScreens
{
    public static MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent.CrystalSphereMinigame? CrystalSphere => null;
}

/// <summary>地図の移動可能性（エンジンの RunSessionMap と同じ規則）。</summary>
public static class RunSessionMap
{
    public static IEnumerable<MapPoint> Travelable(RunState s)
    {
        var map = s.Map;
        if (s.VisitedMapCoords.Count == 0) return new[] { map.StartingMapPoint };
        MapCoord last = s.VisitedMapCoords[^1];
        if (map.SecondBossMapPoint != null && last == map.BossMapPoint.coord) return new[] { map.SecondBossMapPoint };
        if (last.row == map.GetRowCount() - 1) return new[] { map.BossMapPoint };
        var point = map.GetPoint(last);
        return point == null ? new[] { map.StartingMapPoint } : MapTravel.GetTravelablePointsFrom(s, point);
    }
}

/// <summary>エンコーダの「预知」部分（ObservationOracle.cs）は実機では使わないので、無効化だけ用意する。</summary>
public sealed partial class Observation
{
    public static bool Oracle;
    void EmitOracle(RunState s, Player me) { }
}
