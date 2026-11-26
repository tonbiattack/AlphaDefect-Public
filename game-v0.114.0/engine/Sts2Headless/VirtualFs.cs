namespace Sts2Headless;

/// <summary>
/// Godot 路径到本地磁盘的映射：res:// 指向从 pck 提取出的资源目录，user:// 指向训练专用的数据目录（不碰玩家存档）。
/// </summary>
public static class VirtualFs
{
    public static string ResDir => Path.Combine(GameAssemblies.Dir, "res");
    public static string UserDir { get; } = Path.Combine(
        System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile), ".local/share/sls2/user");

    public static string Map(string path)
    {
        if (path.StartsWith("res://")) return Path.Combine(ResDir, path["res://".Length..]);
        if (path.StartsWith("user://")) return Path.Combine(UserDir, path["user://".Length..]);
        return path;
    }
}
