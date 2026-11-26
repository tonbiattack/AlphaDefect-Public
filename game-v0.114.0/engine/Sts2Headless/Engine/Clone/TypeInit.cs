namespace Sts2Headless;

/// <summary>
/// 不触发静态初始化地判断一个类型的静态初始化有没有跑过（.NET 没有公开接口；读取静态字段本身就会触发初始化）。
/// 读 CoreCLR 的 MethodTable：偏移 32 是 MethodTableAuxiliaryData 指针（System.Private.CoreLib 的 MethodTable 结构里也是这个偏移），
/// 它的标志位 0x1 是 enum_flag_Initialized、0x100 是 enum_flag_IsInitError（dotnet/runtime 9.0 的 methodtable.h）。
/// 这是运行时的内部布局：第一次用前拿一个专用的哨兵类型核对（初始化前读到“未初始化”、初始化后读到“已初始化”），对不上就拒绝拍快照，
/// 免得换了运行时版本后静默地判错。
/// </summary>
public static unsafe class TypeInit
{
    const int AuxOffset = 32;
    const uint FlagInitialized = 0x1, FlagInitError = 0x100;
    static bool _validated;

    static uint Flags(Type t)
    {
        nint mt = t.TypeHandle.Value;
        nint aux = *(nint*)(mt + AuxOffset);
        return aux == 0 ? 0 : *(uint*)aux;
    }

    public static bool IsInitialized(Type t) => (Flags(t) & (FlagInitialized | FlagInitError)) == FlagInitialized;
    public static bool Failed(Type t) => (Flags(t) & FlagInitError) != 0;

    /// <summary>哨兵：只在这里被初始化一次。</summary>
    static class Sentinel
    {
        public static readonly int Value = Environment.TickCount | 1;
    }

    public static void Validate()
    {
        if (_validated) return;
        bool before = IsInitialized(typeof(Sentinel));
        System.Runtime.CompilerServices.RuntimeHelpers.RunClassConstructor(typeof(Sentinel).TypeHandle);
        bool after = IsInitialized(typeof(Sentinel));
        bool core = IsInitialized(typeof(Pump));      // 启动时就用到的类型，早就初始化了
        if (before || !after || !core)
            throw new NotSupportedException($"局面复制：判断类型是否已初始化的办法在当前运行时（{System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}）上对不上"
                + $"（哨兵初始化前 {before}、后 {after}，Pump {core}），见 Engine/Clone/TypeInit.cs");
        _validated = true;
    }
}
