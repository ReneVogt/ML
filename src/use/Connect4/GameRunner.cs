using static Connect4.ConsoleHelper;

namespace Connect4;

sealed class GameRunner(IGenerateMoves redMoveGenerator, IGenerateMoves yellowMoveGenerator)
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

    readonly Connect4Board env = new();

    public void Run()
    {
        Console.Clear();
        Write(0, 0, ConsoleColor.Blue, Frame.Trim());
        Write(0, 13, ConsoleColor.Gray, "  1   2   3   4   5   6   7");

        for(; ; )
        {
            int action = env.Player == 1 ? redMoveGenerator.GetMove(env) : yellowMoveGenerator.GetMove(env);
            if (action == 7)
            {
                if (!env.CanUndo) return;
                Undo();
                continue;
            }
            if (action > 7) return;

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
            if (k == 0) return;
            Undo();
            Write(9, 15, ConsoleColor.Black, "            ");
        }
    }

    void Undo()
    {
        var action = env.Undo();
        WriteStone(action, env.Height(action), ConsoleColor.Black);
        if (redMoveGenerator.IsHuman && yellowMoveGenerator.IsHuman || !env.CanUndo) return;       
        action = env.Undo();
        WriteStone(action, env.Height(action), ConsoleColor.Black);
    }

    static void WriteStone(int column, int row, ConsoleColor color) => Write(1 + 4 * column, 2 * (6 - row) - 1, color, "███");
}
