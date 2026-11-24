using System.Reflection;
using HarmonyLib;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using Sts2Headless;

namespace AlphaDefectAutopilot;

/// <summary>
/// カード報酬の選牌画面（NCardRewardSelectionScreen）を AI から選ぶ。
///
/// ゲームは画面の OptionSelected() が返す添字で報酬を受け取る。画面の内部では
/// カードを選ぶと _completionSource に添字（代替項目は 枚数 + 添字）を入れるだけなので、
/// ここでも同じところに値を入れて「選んだ」ことにする（画面の処理はそのまま進む）。
///
/// 候補はカード（PickCardReward）と代替項目（CardRewardAlt: スキップ、生贄など）。
/// </summary>
internal static class LiveCardReward
{
    public static NCardRewardSelectionScreen? Screen { get; private set; }
    public static IReadOnlyList<CardCreationResult> Options { get; private set; } = Array.Empty<CardCreationResult>();
    public static IReadOnlyList<CardRewardAlternative> Alternatives { get; private set; } = Array.Empty<CardRewardAlternative>();

    static FieldInfo? _completion;

    public static void Capture(NCardRewardSelectionScreen screen, IReadOnlyList<CardCreationResult> options,
                               IReadOnlyList<CardRewardAlternative> alternatives)
    {
        Screen = screen;
        Options = options;
        Alternatives = alternatives;
        _completion ??= typeof(NCardRewardSelectionScreen)
            .GetField("_completionSource", BindingFlags.Instance | BindingFlags.NonPublic);
        Bridge.LogLine($"card reward screen: {options.Count} cards, {alternatives.Count} alternatives");
    }

    public static void Forget()
    {
        Screen = null;
        Options = Array.Empty<CardCreationResult>();
        Alternatives = Array.Empty<CardRewardAlternative>();
    }

    public static bool Active => Screen != null && Godot.GodotObject.IsInstanceValid(Screen);

    /// <summary>選択待ちの受け皿がまだ用意されていない（OptionSelected 前）なら false。</summary>
    static bool Ready(out object? source)
    {
        source = null;
        if (!Active || _completion == null) return false;
        source = _completion.GetValue(Screen);
        return source != null;
    }

    public static Decision? Build()
    {
        if (!Active) return null;
        if (!Ready(out _)) return null;                    // まだ受け皿が無い: 次のフレームで
        var d = new Decision(DecisionKind.CardReward);
        for (int i = 0; i < Options.Count; i++)
        {
            int k = i;
            var card = Options[i].Card;
            d.Add(ActionKind.PickCardReward, "拿 " + card.Id.Entry, () => Resolve(k), a: card);
        }
        for (int i = 0; i < Alternatives.Count; i++)
        {
            int k = Options.Count + i;
            string id = Alternatives[i].OptionId;
            d.Add(ActionKind.CardRewardAlt, id, () => Resolve(k), key: id);
        }
        return d;
    }

    static void Resolve(int index)
    {
        if (!Ready(out var source)) { Bridge.LogLine("card reward not ready"); return; }
        try
        {
            // TCS<int?>: カードは添字、代替項目は 枚数 + 添字
            source!.GetType().GetMethod("SetResult")!.Invoke(source, new object?[] { index });
            Bridge.LogLine($"picked card reward index {index}");
            Forget();
        }
        catch (Exception e)
        {
            Bridge.LogLine("card reward select failed: " + e.GetType().Name + ": " + e.Message);
        }
    }
}

/// <summary>カード報酬画面が出た瞬間を捕まえる。</summary>
[HarmonyPatch(typeof(NCardRewardSelectionScreen), nameof(NCardRewardSelectionScreen.ShowScreen))]
internal static class CardRewardScreenPatch
{
    [HarmonyPostfix]
    static void Postfix(IReadOnlyList<CardCreationResult> __0, IReadOnlyList<CardRewardAlternative> __1,
                        NCardRewardSelectionScreen __result)
    {
        try
        {
            LiveCardReward.Capture(__result, __0, __1);
        }
        catch (Exception e)
        {
            Bridge.LogLine("card reward capture failed: " + e.GetType().Name);
        }
    }
}
