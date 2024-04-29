using System.Data;
using Connect4;

Console.CursorVisible = false;

mainMenu:
Console.Clear();
Write(0, 0, ConsoleColor.Gray, "[ 1 ]:");
Write(7, 0, ConsoleColor.Red, "USER");
Write(12, 0, ConsoleColor.Gray, "-");
Write(15, 0, ConsoleColor.Yellow, "USER");
Write(0, 1, ConsoleColor.Gray, "[ 2 ]:");
Write(7, 1, ConsoleColor.Red, "USER");
Write(12, 1, ConsoleColor.Gray, "-");
Write(15, 1, ConsoleColor.Yellow, "AI");
Write(0, 2, ConsoleColor.Gray, "[ 3 ]:");
Write(8, 2, ConsoleColor.Red, "AI");
Write(12, 2, ConsoleColor.Gray, "-");
Write(14, 2, ConsoleColor.Yellow, "USER");
Write(0, 3, ConsoleColor.Gray, "[ 4 ]:");
Write(8, 3, ConsoleColor.Red, "AI");
Write(12, 3, ConsoleColor.Gray, "-");
Write(15, 3, ConsoleColor.Yellow, "AI");

bool redHuman = false, yellowHuman = false;
switch(GetKey(ConsoleKey.D1, ConsoleKey.D2, ConsoleKey.D3, ConsoleKey.D4, ConsoleKey.Escape))
{
    case ConsoleKey.D1:
        redHuman = true;
        yellowHuman = true;
        break;
    case ConsoleKey.D2:
        redHuman = true;
        break;
    case ConsoleKey.D3:
        redHuman = true;
        break;
    case ConsoleKey.D4:
        yellowHuman = true;
        break;
    case ConsoleKey.Escape:
        return;
}

var frame = @"
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
".Trim();
Console.Clear();
Write(0, 0, ConsoleColor.Blue, frame);
Write(0, 13, ConsoleColor.Gray, "  1   2   3   4   5   6   7");

var keys = new List<ConsoleKey>{ ConsoleKey.D1, ConsoleKey.D2, ConsoleKey.D3, ConsoleKey.D4, ConsoleKey.D5, ConsoleKey.D6, ConsoleKey.D7, ConsoleKey.Backspace, ConsoleKey.Escape };

var done = false;
var env = new Connect4Board();
do
{
    int action;
    if (env.Player == 1 && redHuman || env.Player == 2 && yellowHuman)
    {
        var key = GetKey([.. keys]);
        action = keys.FindIndex(k => k == key);
        if (action == 7)
        {
            if (!env.CanUndo) goto mainMenu;
            action = env.Undo();
            Write(1 + 4 * action, 1 + 2 * (6 - env.Height(action)+1), ConsoleColor.Black, "   ");            
            action = env.Undo();
            Write(1 + 4 * action, 1 + 2 * (6 - env.Height(action)+1), ConsoleColor.Black, "   ");
            continue;
        }
        if (action > 7) goto mainMenu;
    }
    else
        action = MoveGenerator.GetMove(env);

    if (env.Height(action) == 6) continue;
    env.Move(action);

    Write(1 + 4 * action, 1 + 2 * (6 - env.Height(action)), env.Player == 2 ? ConsoleColor.Red : ConsoleColor.Yellow, "███");

    if (env.Winner != 0)
    {
        done = true;
        if (env.Winner == 1)
            Write(10, 15, ConsoleColor.Red, "RED WINS!");
        else
            Write(9, 15, ConsoleColor.Yellow, "YELLOW WINS!");
    }
    else if (env.Full)
    {
        done = true;
        Write(12, 15, ConsoleColor.Green, "DRAW!");
    }

} while (!done);

Console.ReadKey(true);
goto mainMenu;

static ConsoleKey GetKey(params ConsoleKey[] allowed)
{
    ConsoleKey key;
    do
    {
        key = Console.ReadKey(true).Key;
    } while (!allowed.Contains(key));
    return key;
}

static void Write(int x, int y, ConsoleColor color, string text)
{
    Console.SetCursorPosition(x, y);
    Console.ForegroundColor = color;
    Console.Write(text);
    Console.ResetColor();
}

