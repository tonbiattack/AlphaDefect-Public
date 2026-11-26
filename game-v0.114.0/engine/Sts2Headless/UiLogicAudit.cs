using System.Reflection;

namespace Sts2Headless;

/// <summary>
/// 列出界面层（Nodes 命名空间）调用的逻辑层方法。无头引擎不运行界面，凡是界面里驱动游戏状态的调用，
/// 都必须在 RunSession / UiFlows 中复刻。游戏更新后重新生成此清单并与上一版本比较，新增项逐一复核。
/// </summary>
public static class UiLogicAudit
{
    static readonly string[] LogicNamespaces =
    {
        "MegaCrit.Sts2.Core.Commands", "MegaCrit.Sts2.Core.Runs", "MegaCrit.Sts2.Core.Rooms", "MegaCrit.Sts2.Core.Multiplayer.Game",
        "MegaCrit.Sts2.Core.Rewards", "MegaCrit.Sts2.Core.Combat", "MegaCrit.Sts2.Core.Events", "MegaCrit.Sts2.Core.GameActions",
        "MegaCrit.Sts2.Core.Entities", "MegaCrit.Sts2.Core.Map", "MegaCrit.Sts2.Core.Models", "MegaCrit.Sts2.Core.Hooks",
    };

    static bool IsLogic(Type? t) => t?.Namespace != null && LogicNamespaces.Any(ns => t.Namespace.StartsWith(ns)) && !UiCallStubber.IsUi(t);

    /// <summary>不改变状态的常见调用（查询、事件订阅、日志），不列出。</summary>
    static bool IsBoring(MethodBase m) =>
        m.Name.StartsWith("get_") || m.Name.StartsWith("add_") || m.Name.StartsWith("remove_") || m.IsConstructor
        || m.Name is "ToString" or "Equals" or "GetHashCode" or "GetType" or "AssertMutable" or "IsMe" or "GetMe" or "IsMine"
        || m.Name.StartsWith("Get") || m.Name.StartsWith("Is") || m.Name.StartsWith("Can") || m.Name.StartsWith("Has") || m.Name.StartsWith("Should");

    public static IEnumerable<string> Run(Assembly game)
    {
        var lines = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Type t in game.GetTypes().Where(UiCallStubber.IsUi))
        foreach (MethodBase m in t.GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
        {
            if (m.GetMethodBody()?.GetILAsByteArray() is not { } il) continue;
            foreach (MethodBase callee in IlScan.Callees(il, m.Module, m))
                if (IsLogic(callee.DeclaringType) && !IsBoring(callee))
                    lines.Add($"{Outer(t).Name}\t→ {callee.DeclaringType!.Name}.{callee.Name}");
        }
        return lines;
    }

    static Type Outer(Type t) { while (t.DeclaringType != null) t = t.DeclaringType; return t; }
}
