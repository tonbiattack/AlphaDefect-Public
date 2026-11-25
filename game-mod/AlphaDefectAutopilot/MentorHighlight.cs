using Godot;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Combat;

namespace AlphaDefectAutopilot;

/// <summary>
/// 観戦用: これから使うカードと対象を、実ゲームの画面で強調する（#52）。
///
/// 操作は奪わない（色を変えるだけ）。手動切替・画面遷移・停止で必ず解除する。
/// </summary>
internal static class MentorHighlight
{
    static CanvasItem? _card;
    static Color _cardColor = Colors.White;
    static bool _cardHasColor;
    static CanvasItem? _enemy;
    static Color _enemyColor = Colors.White;
    static bool _enemyHasColor;

    static readonly Color CardTint = new(1.0f, 0.92f, 0.45f);   // 金
    static readonly Color EnemyTint = new(1.0f, 0.45f, 0.40f);  // 赤

    /// <summary>選んだカードと対象を強調する（null なら何もしない）。</summary>
    public static void Show(CardModel? card, Creature? target)
    {
        Clear();
        try
        {
            if (card != null)
            {
                var node = NCard.FindOnTable(card);
                if (node != null) Tint(node, CardTint, ref _card, ref _cardColor, ref _cardHasColor);
            }
            if (target != null)
            {
                var node = FindCreatureNode(target);
                if (node != null) Tint(node, EnemyTint, ref _enemy, ref _enemyColor, ref _enemyHasColor);
            }
        }
        catch (Exception e)
        {
            Bridge.LogEvent("[highlight] failed: " + e.GetType().Name);
        }
    }

    /// <summary>強調を解除する。</summary>
    public static void Clear()
    {
        Restore(ref _card, ref _cardColor, ref _cardHasColor);
        Restore(ref _enemy, ref _enemyColor, ref _enemyHasColor);
    }

    static void Tint(CanvasItem node, Color color, ref CanvasItem? slot, ref Color original, ref bool had)
    {
        slot = node;
        original = node.Modulate;
        had = true;
        node.Modulate = color;
    }

    static void Restore(ref CanvasItem? slot, ref Color original, ref bool had)
    {
        try
        {
            if (slot != null && GodotObject.IsInstanceValid(slot) && had) slot.Modulate = original;
        }
        catch (Exception)
        {
            // ノードが消えていても構わない
        }
        slot = null;
        had = false;
    }

    static NCreature? FindCreatureNode(Creature entity)
    {
        var root = (Engine.GetMainLoop() as SceneTree)?.Root;
        return root == null ? null : Walk(root, entity, 0);
    }

    static NCreature? Walk(Node node, Creature entity, int depth)
    {
        if (depth > 40) return null;
        if (node is NCreature creature && ReferenceEquals(creature.Entity, entity)) return creature;
        int n = node.GetChildCount();
        for (int i = 0; i < n; i++)
        {
            var found = Walk(node.GetChild(i), entity, depth + 1);
            if (found != null) return found;
        }
        return null;
    }
}
