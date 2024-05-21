using System.Collections.Concurrent;

namespace Connect4;

sealed class MonteCarloMoveGenerator : IGenerateMoves
{
    const int TimeToThink = 1000;
    const int NumberOfThreads = 8;

    sealed class TreeNode(ulong key)
    {
        public ulong Key { get; } = key;
        public double[] Wins { get; } = new double[7];
        public int[] Simulations { get; } = new int[7];
        public int OwnSimulations { get; set; }
    }

    static readonly double C = Math.Sqrt(2);
    
    readonly ConcurrentDictionary<ulong, TreeNode> nodes = new();

    public string Name => "MCTS";
    public async Task<int> GetMoveAsync(Connect4Board env)
    {
        var key = env.State;
        var node = nodes.GetOrAdd(key, k => new TreeNode(k));

        using CancellationTokenSource cts = new (TimeToThink);

        await Task.WhenAll(Enumerable.Range(0, NumberOfThreads).Select(_ => Task.Run(() => MonteCarloTreeSearch(env.Clone(), cts.Token))));

        return node.Simulations.Select((simulations, action) => (simulations, action)).MaxBy(x => x.simulations).action;
    }

    void MonteCarloTreeSearch(Connect4Board env, CancellationToken cancellation)
    {
        while (!cancellation.IsCancellationRequested)
            MonteCarloTreeSearch(env, 1);
    }
    double MonteCarloTreeSearch(Connect4Board env, int parentSimulations)
    {
        var key = env.State;
        var node = nodes.GetOrAdd(key, k => new TreeNode(k));
        var ln = Math.Log(parentSimulations);
        int action, ownSimulations;
        lock (node)
        {
            ownSimulations = node.OwnSimulations += 1;
            action = node.Simulations.Select((simulations, a) => (simulations, a)).Where(x => env.Height(x.a) < 6).MaxBy(x => x.simulations == 0 ? double.MaxValue : (node.Wins[x.a]/x.simulations + C * Math.Sqrt(ln / x.simulations))).a;
            node.Simulations[action] += 1;
        }

        double result;
        env.Move(action);
        if (env.Finished)
        {
            lock (node)
            {
                result = env.Winner != 0 ? 1 : 0.5;
                node.Wins[action] += result;
                env.Undo();
                return result;
            }
        }

        result = -MonteCarloTreeSearch(env, ownSimulations);
        env.Undo();
        if (result > 0)
            lock (node) node.Wins[action] += result;

        return result;
    }
}
