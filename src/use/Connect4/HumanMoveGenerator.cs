namespace Connect4;

sealed class HumanMoveGenerator: IGenerateMoves
{
    TaskCompletionSource<int> _tcs = new();

    public string Name => "Human";

    public void MoveClicked(int move)
    {
        _tcs.SetResult(move);
        _tcs = new();
    }
    public Task<int> GetMoveAsync(Connect4Board env) => _tcs.Task;
}
