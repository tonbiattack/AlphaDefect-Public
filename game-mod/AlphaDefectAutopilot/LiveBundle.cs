using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// カードパックの選択（NChooseABundleSelectionScreen）を AI に選ばせる。
/// Neow の「箱を選ぶ」などがこれで、選ばないとイベントが終わらない。
///
/// 画面は選んだパックを TaskCompletionSource に流すだけなので、こちらも同じ所に
/// 「このパックを選んだ」値を入れる（画面の ConfirmSelection と同じ形）。
/// </summary>
internal static class LiveBundle
{
    public static NChooseABundleSelectionScreen? Screen { get; private set; }
    public static IReadOnlyList<IReadOnlyList<CardModel>> Bundles { get; private set; } = Array.Empty<IReadOnlyList<CardModel>>();

    static FieldInfo? _completion;

    public static void Capture(NChooseABundleSelectionScreen screen, IReadOnlyList<IReadOnlyList<CardModel>> bundles)
    {
        Screen = screen;
        Bundles = bundles;
        _completion ??= typeof(NChooseABundleSelectionScreen)
            .GetField("_completionSource", BindingFlags.Instance | BindingFlags.NonPublic);
        Bridge.LogLine($"bundle screen: {bundles.Count} packs");
    }

    public static void Forget()
    {
        Screen = null;
        Bundles = Array.Empty<IReadOnlyList<CardModel>>();
    }

    public static bool Active => Screen != null && Godot.GodotObject.IsInstanceValid(Screen) && _completion != null;

    public static Decision? Build()
    {
        if (!Active) return null;
        var d = new Decision(DecisionKind.Bundle);
        for (int i = 0; i < Bundles.Count; i++)
        {
            int k = i;
            foreach (var c in Bundles[i]) d.Shown.Add((c, k == 0 ? Zone.Pack1 : Zone.Pack2, 0));
            d.Add(ActionKind.PickBundle, "卡包 " + string.Join("+", Bundles[i].Select(c => c.Id.Entry)),
                  () => Resolve(k), a: Bundles[i].FirstOrDefault(), f0: i);
        }
        return d;
    }

    static void Resolve(int index)
    {
        if (!Active) return;
        try
        {
            object source = _completion!.GetValue(Screen)!;
            var pick = new[] { Bundles[index] };
            source.GetType().GetMethod("SetResult")!.Invoke(source, new object?[] { pick });
            Bridge.LogLine($"picked bundle {index}");
            Forget();
        }
        catch (Exception e)
        {
            Bridge.LogLine("bundle select failed: " + e.GetType().Name + ": " + e.Message);
        }
    }
}

/// <summary>パック選択画面が出た瞬間を捕まえる。</summary>
[HarmonyPatch(typeof(NChooseABundleSelectionScreen), nameof(NChooseABundleSelectionScreen.ShowScreen))]
internal static class BundleScreenPatch
{
    [HarmonyPostfix]
    static void Postfix(IReadOnlyList<IReadOnlyList<CardModel>> __0, NChooseABundleSelectionScreen __result)
    {
        try
        {
            LiveBundle.Capture(__result, __0);
        }
        catch (Exception e)
        {
            Bridge.LogLine("bundle capture failed: " + e.GetType().Name);
        }
    }
}
