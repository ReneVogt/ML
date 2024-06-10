namespace Connect4;

sealed class HumanMoveGenerator: IGenerateMoves
{
    TaskCompletionSource<int>? _tcs;

    public string Name => "Human";

    public void MoveClicked(int move)
    {
        _tcs?.SetResult(move);
    }
    public Task<int> GetMoveAsync(Connect4Board env)
    {
        _tcs = new();
        return _tcs.Task;
    }
}
