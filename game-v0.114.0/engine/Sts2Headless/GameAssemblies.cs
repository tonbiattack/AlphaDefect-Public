using System.Reflection;
using System.Runtime.Loader;

namespace Sts2Headless;

/// <summary>
/// 从游戏目录加载 sts2.dll 及其依赖。必须在第一次触碰任何游戏类型之前调用。
/// </summary>
public static class GameAssemblies
{
    public static string Dir { get; private set; } = "";

    public static void Register()
    {
        Dir = Environment.GetEnvironmentVariable("STS2_GAME_DIR")
              ?? AppContext.GetData("Sts2GameDir") as string
              ?? throw new InvalidOperationException("未设置游戏程序集目录");
        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            string path = Path.Combine(Dir, name.Name + ".dll");
            return File.Exists(path) ? ctx.LoadFromAssemblyPath(path) : null;
        };
    }
}
