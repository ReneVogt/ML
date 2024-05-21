namespace Connect4;

interface IGenerateMoves
{
    string Name { get; }
    Task<int> GetMoveAsync(Connect4Board env);
}
