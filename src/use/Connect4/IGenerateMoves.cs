namespace Connect4;

interface IGenerateMoves
{
    bool IsHuman { get; }
    int GetMove(Connect4Board env);
}
