using System.Collections.Concurrent;

namespace Sts2Headless;

/// <summary>
/// 单线程同步上下文：游戏逻辑里的 await 续延全部排到这里，在调用线程上按顺序执行。
/// 队列清空而任务未完成，说明游戏在等外部输入（玩家选择），此时 RunUntilIdle 返回。
/// </summary>
public sealed class Pump : SynchronizationContext
{
    readonly ConcurrentQueue<(SendOrPostCallback, object?)> _queue = new();
    public static Pump Instance { get; } = new();

    public int OwnerThread { get; private set; } = -1;
    /// <summary>已执行的续延条数（DelayPatch 用来识别同一条续延里的忙等）。</summary>
    public long Executed;
    public int ForeignPosts;

    /// <summary>把本泵设为当前线程的永久同步上下文。所有 await 都会捕获它，续延回到这个线程。</summary>
    public void Attach()
    {
        OwnerThread = Environment.CurrentManagedThreadId;
        SetSynchronizationContext(this);
    }

    public override void Post(SendOrPostCallback d, object? state)
    {
        if (Environment.CurrentManagedThreadId != OwnerThread) Interlocked.Increment(ref ForeignPosts);
        _queue.Enqueue((d, state));
    }

    public override void Send(SendOrPostCallback d, object? state) => d(state);

    public override SynchronizationContext CreateCopy() => this;

    /// <summary>执行排队的续延，直到队列为空。返回执行的条数。</summary>
    public int RunUntilIdle(int limit = 10_000_000)
    {
        var prev = Current;
        SetSynchronizationContext(this);
        int n = 0;
        try
        {
            while (_queue.TryDequeue(out var item))
            {
                Executed++;
                item.Item1(item.Item2);
                if (++n >= limit) throw new InvalidOperationException("消息泵超过上限，可能死循环");
            }
        }
        finally { SetSynchronizationContext(prev); }
        return n;
    }

    /// <summary>在泵上启动一个异步操作，并推进到空闲。</summary>
    public Task Start(Func<Task> action)
    {
        var prev = Current;
        SetSynchronizationContext(this);
        Executed++;
        Task t;
        try { t = action(); }
        finally { SetSynchronizationContext(prev); }
        RunUntilIdle();
        return t;
    }
}
