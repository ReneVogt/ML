using static Connect4.ConsoleHelper;

namespace Connect4;


sealed class GameRunner(bool redHuman, bool yellowHuman)
{
    const string Frame = @"
╔═══╤═══╤═══╤═══╤═══╤═══╤═══╗
║   │   │   │   │   │   │   ║
╟───┼───┼───┼───┼───┼───┼───╢
║   │   │   │   │   │   │   ║
╟───┼───┼───┼───┼───┼───┼───╢
║   │   │   │   │   │   │   ║
╟───┼───┼───┼───┼───┼───┼───╢
║   │   │   │   │   │   │   ║
╟───┼───┼───┼───┼───┼───┼───╢
║   │   │   │   │   │   │   ║
╟───┼───┼───┼───┼───┼───┼───╢
║   │   │   │   │   │   │   ║
╚═══╧═══╧═══╧═══╧═══╧═══╧═══╝
";
    static readonly List<ConsoleKey> allowedKeys = new List<ConsoleKey> { ConsoleKey.D1, ConsoleKey.D2, ConsoleKey.D3, ConsoleKey.D4, ConsoleKey.D5, ConsoleKey.D6, ConsoleKey.D7, ConsoleKey.Backspace, ConsoleKey.Escape };

    readonly Connect4Board env = new();

    public void Run()
    {
        Console.Clear();
        Write(0, 0, ConsoleColor.Blue, Frame.Trim());
        Write(0, 13, ConsoleColor.Gray, "  1   2   3   4   5   6   7");

        for(; ; )
        {
            int action;
            if (env.Player == 1 && redHuman || env.Player == 2 && yellowHuman)
            {
                var key = GetKey([.. allowedKeys]);
                action = allowedKeys.FindIndex(k => k == key);
                if (action == 7)
                {
                    if (!env.CanUndo) return;
                    Undo();
                    continue;
                }
                if (action > 7) return;
            }
            else
                action = MoveGenerator.GetMove(env);

            if (env.Height(action) == 6) continue;
            env.Move(action);

            WriteStone(action, env.Height(action) - 1, env.Player == 2 ? ConsoleColor.Red : ConsoleColor.Yellow);

            if (!env.Finished) continue;

            if (env.Winner != 0)
            {
                if (env.Winner == 1)
                    Write(10, 15, ConsoleColor.Red, "RED WINS!");
                else
                    Write(9, 15, ConsoleColor.Yellow, "YELLOW WINS!");
            }
            else if (env.Full)
                Write(12, 15, ConsoleColor.Green, "DRAW!");

            var k = GetKey([ConsoleKey.Escape, ConsoleKey.Backspace]);
            if (k == ConsoleKey.Escape) return;
            Undo();
            Write(9, 15, ConsoleColor.Black, "            ");
        }
    }

    void Undo()
    {
        var action = env.Undo();
        WriteStone(action, env.Height(action), ConsoleColor.Black);
        if (redHuman && yellowHuman) return;
        action = env.Undo();
        WriteStone(action, env.Height(action), ConsoleColor.Black);
    }

    static void WriteStone(int column, int row, ConsoleColor color) => Write(1 + 4 * column, 2 * (6 - row) - 1, color, "███");
}
