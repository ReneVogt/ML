namespace Connect4;

sealed class MiniMaxMoveGenerator : IGenerateMoves
{
    public bool IsHuman => false;
    public int GetMove(Connect4Board env) => throw new NotImplementedException();
}
