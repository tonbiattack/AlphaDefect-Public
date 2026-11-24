using System.IO;
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;

namespace AlphaDefectAutopilot;

/// <summary>
/// MOD の入口。ゲームの ModManager が起動時に呼ぶ（mods/AlphaDefectAutopilot/ に置いた DLL）。
///
/// やること（0.2.0）:
///   * Harmony で NRun._Process の後処理を差し込み、ゲームのメインスレッドで状態を読む
///   * 推論サーバ（WSL 側 local/bridge）へ接続し、公開状態（probe）を定期的に送る
///   * 戦闘の出牌局面では、無頭エンジンと同じ観測フレームを作って送り、返ってきた手を実行する
///   * pause / resume / stop の指示を受ける
///
/// まだ扱わない意思決定: 報酬・経路・ショップ・イベント・選牌（戦闘の外は状態を送るだけ）。
/// </summary>
[ModInitializer(nameof(Start))]
public static class ModEntry
{
    public static void Start()
    {
        try
        {
            MegaCrit.Sts2.Core.Logging.Log.Info("[AlphaDefect] mod initializer called");
            // 観測エンコーダが使う語彙（MOD のフォルダに置いてある vocab.txt、行番号が ID）
            string modDir = Path.GetDirectoryName(typeof(ModEntry).Assembly.Location) ?? ".";
            string vocab = Path.Combine(modDir, "vocab.txt");
            Sts2Headless.Vocab.Load(vocab);
            MegaCrit.Sts2.Core.Logging.Log.Info($"[AlphaDefect] vocab loaded ({Sts2Headless.Vocab.Count} keys) from {vocab}");
            var harmony = new Harmony("alphadefect.autopilot");
            harmony.PatchAll(typeof(ModEntry).Assembly);
            MegaCrit.Sts2.Core.Logging.Log.Info("[AlphaDefect] harmony patches applied");
            Bridge.Start();
        }
        catch (Exception e)
        {
            try
            {
                MegaCrit.Sts2.Core.Logging.Log.Error("[AlphaDefect] mod failed to start: " + e);
            }
            catch
            {
                // ログも出せないなら何もできることがない
            }
        }
    }
}

/// <summary>毎フレームの後処理で状態を 1 枚読む（メインスレッドなので Godot の状態を触ってよい）。</summary>
[HarmonyPatch(typeof(MegaCrit.Sts2.Core.Nodes.NRun), "_Process")]
internal static class NRunProcessPatch
{
    [HarmonyPostfix]
    static void Postfix()
    {
        try
        {
            Bridge.Tick();
        }
        catch (Exception e)
        {
            try
            {
                MegaCrit.Sts2.Core.Logging.Log.Error("[AlphaDefect] tick failed: " + e.GetType().Name);
            }
            catch
            {
                // 同上
            }
        }
    }
}
