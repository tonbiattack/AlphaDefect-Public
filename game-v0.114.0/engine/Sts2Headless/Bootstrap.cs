using MegaCrit.Sts2.Core.GameActions.Multiplayer;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Modding;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.TestSupport;

namespace Sts2Headless;

/// <summary>
/// 无头启动顺序，对应游戏的 OneTimeInitialization.ExecuteVeryEarly / ExecuteEssential，去掉图集与界面部分。
/// </summary>
public static class Bootstrap
{
    static bool _done;

    /// <summary>界面语言只影响文本，不影响规则；调试时用英文，便于对照 ID。</summary>
    public static string Language { get; set; } = "eng";

    public static void Init()
    {
        if (_done) return;
        _done = true;
        Pump.Instance.Attach();
        GodotShim.Install();
        GameShim.Install();
        TestMode.TurnOnInternal();
        SaveManager.Instance.InitSettingsDataForTest();
        SaveManager.Instance.InitPrefsDataForTest();
        SaveManager.Instance.SettingsSave.Language = Language;
        ModManager.Initialize(null!, null, null).GetAwaiter().GetResult();   // 测试模式下直接跳过，不加载任何模组
        LocManager.Initialize();
        AssemblyInfo.Init();        // v0.111.0 起：内容排序（ContentSorter）要先知道哪些类型属于本体
        ModelDb.Init();
        ModelIdSerializationCache.Init();
        ModelDb.InitIds();
        MessageTypes.Initialize();
        ActionTypes.Initialize();
    }
}
