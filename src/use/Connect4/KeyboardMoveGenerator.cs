using static Connect4.ConsoleHelper;
namespace Connect4;

sealed class KeyboardMoveGenerator : IGenerateMoves
{
    static readonly ConsoleKey[] allowedKeys = [ConsoleKey.D1, ConsoleKey.D2, ConsoleKey.D3, ConsoleKey.D4, ConsoleKey.D5, ConsoleKey.D6, ConsoleKey.D7, ConsoleKey.Backspace, ConsoleKey.Escape];

    public bool IsHuman => true;

    public int GetMove(Connect4Board env) => GetKey([.. allowedKeys]);
}
