using System.Text;

namespace AlphaDefectAutopilot;

/// <summary>
/// MOD から見た「いまの公開状態」を 1 つの JSON オブジェクトにする。
///
/// 実機の MOD は、無頭エンジン（engine/Sts2Headless）と同じ観測エンコーダを積んで
/// 観測フレームを送るのが最終形。ここではその前段として、画面に出ている公開情報だけを
/// 読めることを確かめる（イシュー #5 の「状態同期」の最小形）。
///
/// 読むのは RunManager が持つ現在の RunState（無頭エンジンの RunSession.State と同じもの）と
/// CombatManager の進行フラグだけで、山札の中身や乱数など隠し情報には触れない。
/// ゲームが更新されて API が変わっても MOD 全体が落ちないよう、読み取りは全部 try で包む。
/// </summary>
internal static class Probe
{
    public static string Snapshot()
    {
        var sb = new StringBuilder(256);
        sb.Append('{');
        sb.Append("\"kind\":\"run\"");
        try
        {
            var run = MegaCrit.Sts2.Core.Runs.RunManager.Instance;
            var state = run?.DebugOnlyGetState();
            if (state == null)
            {
                sb.Append(",\"in_run\":false");
                sb.Append('}');
                return sb.ToString();
            }

            sb.Append(",\"in_run\":true");
            sb.Append(",\"act\":").Append(state.CurrentActIndex);
            sb.Append(",\"floor\":").Append(state.TotalFloor);
            sb.Append(",\"ascension\":").Append(state.AscensionLevel);
            sb.Append(",\"combat\":").Append(MegaCrit.Sts2.Core.Combat.CombatManager.Instance.IsInProgress ? "true" : "false");

            var me = state.Players.Count > 0 ? state.Players[0] : null;
            if (me != null)
            {
                sb.Append(",\"character\":\"").Append(Escape(me.Character?.GetType().Name ?? "?")).Append('"');
                sb.Append(",\"hp\":").Append(me.Creature?.CurrentHp ?? 0);
                sb.Append(",\"max_hp\":").Append(me.Creature?.MaxHp ?? 0);
                sb.Append(",\"gold\":").Append(me.Gold);
                sb.Append(",\"deck\":").Append(me.Deck?.Cards?.Count ?? 0);
                if (MentorClient.Enabled && MentorClient.RunCharacter.Length > 0)
                {
                    sb.Append(",\"mentor_character\":\"").Append(Escape(MentorClient.RunCharacter)).Append('"');
                    sb.Append(",\"mentor_ascension\":").Append(MentorClient.RunAscension);
                    sb.Append(",\"mentor_seed\":").Append(MentorClient.RunSeed?.ToString() ?? "null");
                }
            }
        }
        catch (Exception e)
        {
            sb.Append(",\"error\":\"").Append(Escape(e.GetType().Name + ": " + e.Message)).Append('"');
        }
        sb.Append('}');
        return sb.ToString();
    }

    static string Escape(string s) =>
        s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", " ").Replace("\r", " ");
}
