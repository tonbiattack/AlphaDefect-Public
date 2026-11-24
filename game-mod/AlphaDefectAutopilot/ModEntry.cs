using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;

namespace AlphaDefectAutopilot;

/// <summary>
/// MOD の入口。ゲームの ModManager が起動時に呼ぶ（mods/AlphaDefectAutopilot/ に置いた DLL）。
///
/// いまの版（0.1.0）がやること:
///   * Harmony で NRun._Process の後処理を差し込み、ゲームのメインスレッドで状態を読む
///   * 推論サーバ（WSL 側 local/bridge）へ接続し、公開状態を定期的に送る
///   * pause / resume / stop の指示を受ける
///
/// まだやっていないこと: 観測フレームの生成と、選ばれた行動の実行。それは
/// エンジンの観測エンコーダをこの MOD に載せてからになる（game-mod/README.md）。
/// </summary>
[ModInitializer(nameof(Start))]
public static class ModEntry
{
    public static void Start()
    {
        try
        {
            MegaCrit.Sts2.Core.Logging.Log.Info("[AlphaDefect] mod initializer called");
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
