using System.Threading;

namespace AlphaDefectAutopilot;

/// <summary>
/// 非同期のゲーム操作（購入・宝箱・部屋の退出など）が終わるまで、次の判断を待たせる門。
///
/// これまでは購入などを投げっぱなしにしていたため、画面が更新される前に次の判断を作って
/// しまい、二重購入や「買ったのに在庫が減っていない」ように見えることがあった。
/// 操作は必ずここを通し、終わる（または失敗する）までドライバは次へ進まない。
/// </summary>
internal static class AsyncGate
{
    static int _busy;
    static volatile string _last = "";

    public static bool Busy => Volatile.Read(ref _busy) > 0;
    public static string LastOp => _last;

    public static void Run(string name, Func<Task> op)
    {
        Interlocked.Increment(ref _busy);
        _last = name;
        Bridge.LogLine($"[op] {name} start");
        _ = RunAsync(name, op);
    }

    /// <summary>同期の操作（クリック相当）を実行し、完了を待つ。例外は握ってログに残す。</summary>
    public static bool RunSync(string name, Action action, int timeoutMs = 5000)
    {
        long t0 = System.Environment.TickCount64;
        try
        {
            action();
            Bridge.LogLine($"[op] {name} done in {System.Environment.TickCount64 - t0}ms");
            return true;
        }
        catch (Exception e)
        {
            Bridge.LogLine($"[op] {name} failed: {e.GetType().Name}: {e.Message}");
            return false;
        }
    }

    static async Task RunAsync(string name, Func<Task> op)
    {
        long t0 = System.Environment.TickCount64;
        try
        {
            await op().ConfigureAwait(false);
            Bridge.LogLine($"[op] {name} done in {System.Environment.TickCount64 - t0}ms");
        }
        catch (Exception e)
        {
            Bridge.LogLine($"[op] {name} failed: {e.GetType().Name}: {e.Message}");
        }
        finally
        {
            Interlocked.Decrement(ref _busy);
            if (!Busy) _last = "";
        }
    }
}
