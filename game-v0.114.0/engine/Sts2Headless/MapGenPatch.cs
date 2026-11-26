using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using HarmonyLib;
using MegaCrit.Sts2.Core.Map;

namespace Sts2Headless;

/// <summary>
/// 地图生成提速：替换 MapPathPruning.FindMatchingSegments，结果（列表内容与顺序）与原实现完全相同。
///
/// 原实现先展开从起点到 Boss 的全部路径（每幕上万条），再对每条路径的每个合格子段生成字符串键，放进有序字典去重，
/// 每幕要做十几万次。关键性质：同一个子段（同一串对象引用）第二次出现时，必然与列表里的自己“重叠”而被拒绝，
/// 不改变任何状态。所以先用引用序列的哈希跳过重复子段，只对第一次出现的子段生成键和做重叠检查。
/// 路径按与原实现相同的深度优先顺序产生，因此第一次出现的顺序、字典内列表的顺序都不变，后续的随机数消耗也不变。
/// 验证：`mapgen-check` 命令对大量种子比较替换前后的地图（见 M1 报告补充）。
/// </summary>
public static class MapGenPatch
{
    public static void Install(Harmony h) =>
        GameCompat.Patch(h, GameCompat.Method(typeof(MapPathPruning), nameof(MapPathPruning.FindMatchingSegments), "地图生成提速"), "MapPathPruning.FindMatchingSegments",
            prefix: GameCompat.Own(typeof(MapGenPatch), nameof(Prefix)));

    static bool Prefix(MapPoint startingMapPoint, ref List<List<MapPoint[]>> __result)
    {
        __result = FindMatchingSegments(startingMapPoint);
        return false;
    }

    /// <summary>子段视图：数组中的一段，按元素引用比较。</summary>
    readonly struct Seg(MapPoint[] arr, int start, int len) : IEquatable<Seg>
    {
        public readonly MapPoint[] Arr = arr;
        public readonly int Start = start, Len = len;

        public bool Equals(Seg o)
        {
            if (Len != o.Len) return false;
            for (int k = 0; k < Len; k++)
                if (!ReferenceEquals(Arr[Start + k], o.Arr[o.Start + k])) return false;
            return true;
        }

        public override bool Equals(object? obj) => obj is Seg s && Equals(s);

        public override int GetHashCode()
        {
            var hc = new HashCode();
            for (int k = 0; k < Len; k++) hc.Add(RuntimeHelpers.GetHashCode(Arr[Start + k]));
            return hc.ToHashCode();
        }
    }

    public static List<List<MapPoint[]>> FindMatchingSegments(MapPoint startingMapPoint)
    {
        var segments = new SortedDictionary<string, List<MapPoint[]>>(StringComparer.Ordinal);
        var seen = new HashSet<Seg>();
        var path = new MapPoint[64];
        int depth = 0;

        void Visit(MapPoint p)
        {
            if (depth == path.Length) Array.Resize(ref path, path.Length * 2);
            path[depth++] = p;
            if (p.PointType == MapPointType.Boss) AddSegments(path, depth, segments, seen);
            else foreach (MapPoint child in p.Children) Visit(child);
            depth--;
        }

        Visit(startingMapPoint);
        return segments.Values.Where(l => l.Count > 1).ToList();
    }

    /// <summary>与原 AddSegmentsToDictionary 相同的遍历顺序与判定，只是跳过已见过的子段。</summary>
    static void AddSegments(MapPoint[] path, int count, SortedDictionary<string, List<MapPoint[]>> segments, HashSet<Seg> seen)
    {
        for (int i = 0; i < count - 1; i++)
        {
            var start = path[i];
            if (!(start.Children.Count > 1 || start.coord.row == 0)) continue;
            for (int j = 2; j < count - i; j++)
            {
                if (path[i + j].parents.Count < 2) continue;
                var view = new Seg(path, i, j + 1);
                if (seen.Contains(view)) continue;
                var array = new MapPoint[j + 1];
                Array.Copy(path, i, array, 0, j + 1);
                seen.Add(new Seg(array, 0, array.Length));
                string key = Key(array);
                if (!segments.TryGetValue(key, out var list)) segments[key] = new List<MapPoint[]> { array };
                else if (!list.Any(e => Overlapping(e, array))) list.Add(array);
            }
        }
    }

    static string Key(MapPoint[] seg)
    {
        var sb = new StringBuilder();
        MapPoint a = seg[0], b = seg[^1];
        if (a.coord.row == 0) sb.Append(a.coord.row).Append('-').Append(b.coord.col).Append(',').Append(b.coord.row).Append('-');
        else sb.Append(a.coord.col).Append(',').Append(a.coord.row).Append('-').Append(b.coord.col).Append(',').Append(b.coord.row).Append('-');
        for (int k = 0; k < seg.Length; k++)
        {
            if (k > 0) sb.Append(',');
            sb.Append((int)seg[k].PointType);
        }
        return sb.ToString();
    }

    static bool Overlapping(MapPoint[] a, MapPoint[] b)
    {
        if (a.Length < 3 || b.Length < 3) return false;
        for (int i = 1; i <= a.Length - 2; i++)
            if (Equals(a[i], b[i])) return true;
        return false;
    }

    /// <summary>地图的完整摘要（每个点的坐标、类型、子节点），用于比较替换前后的结果。</summary>
    public static string Digest(ActMap map)
    {
        var sb = new StringBuilder();
        var pts = map.GetAllMapPoints().Append(map.StartingMapPoint).Append(map.BossMapPoint);
        if (map.SecondBossMapPoint != null) pts = pts.Append(map.SecondBossMapPoint);
        foreach (var p in pts.Distinct().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col))
        {
            sb.Append(p.coord.col).Append(',').Append(p.coord.row).Append(':').Append((int)p.PointType).Append('>');
            foreach (var c in p.Children.OrderBy(c => c.coord.row).ThenBy(c => c.coord.col)) sb.Append(c.coord.col).Append(',').Append(c.coord.row).Append(';');
            sb.Append('|');
        }
        return sb.ToString();
    }
}
