using System.Data;
using Connect4;

Console.CursorVisible = false;

mainMenu:
Console.Clear();
Write(0, 0, ConsoleColor.Gray, "[ 1 ]:");
Write(7, 0, ConsoleColor.Red, "USER");
Write(12, 0, ConsoleColor.Gray, "-");
Write(15, 0, ConsoleColor.Yellow, "AI");
Write(0, 1, ConsoleColor.Gray, "[ 2 ]:");
Write(8, 1, ConsoleColor.Red, "AI");
Write(12, 1, ConsoleColor.Gray, "-");
Write(14, 1, ConsoleColor.Yellow, "USER");
Write(0, 2, ConsoleColor.Gray, "[ 3 ]:");
Write(8, 2, ConsoleColor.Red, "AI");
Write(12, 2, ConsoleColor.Gray, "-");
Write(15, 2, ConsoleColor.Yellow, "AI");

bool redHuman = false, yellowHuman = false;
switch(GetKey(ConsoleKey.D1, ConsoleKey.D2, ConsoleKey.D3, ConsoleKey.Escape))
{
    case ConsoleKey.D1:
        redHuman = true;
        break;
    case ConsoleKey.D2:
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
var state = Enumerable.Range(0, 7).Select(_ => new List<int>()).ToArray();

var done = false;
var player = 1;
var history = new Stack<int>();
do
{
    int action;
    if (player == 1 && redHuman || player == 2 && yellowHuman)
    {
        var key = GetKey(keys.ToArray());
        action = keys.FindIndex(k => k == key);
        if (action == 7)
        {
            if (history.Count == 0) goto mainMenu;
            action = history.Pop();
            Write(1 + 4 * action, 1 + 2 * (6 - state[action].Count), ConsoleColor.Black, "   ");
            state[action].RemoveAt(state[action].Count-1);
            action = history.Pop();
            Write(1 + 4 * action, 1 + 2 * (6 - state[action].Count), ConsoleColor.Black, "   ");
            state[action].RemoveAt(state[action].Count-1);
            continue;
        }
        if (action > 7) goto mainMenu;
    }
    else
        action = MoveGenerator.GetMove(state, player);

    if (state[action].Count == 6) continue;
    state[action].Add(player);
    history.Push(action);

    Write(1 + 4 * action, 1 + 2 * (6 - state[action].Count), player == 1 ? ConsoleColor.Red : ConsoleColor.Yellow, "███");

    var won = false;

    // horizontal check
    var row = state[action].Count - 1;
    for (int x = Math.Max(0, action - 3); x <= Math.Min(3, action); x++)
        if (state[x].Count > row && state[x][row] == player &&
            state[x+1].Count > row && state[x+1][row] == player &&
            state[x+2].Count > row && state[x+2][row] == player &&
            state[x+3].Count > row && state[x+3][row] == player)
        {
            won = true;
            break;
        }

    // vertical check
    if (!won && state[action].Count > 3 && state[action][2] == player && state[action][3] == player)
    {
        if (state[action][1] == player)
        {
            if (state[action][0] == player || state[action].Count > 4 && state[action][4] == player)
                won = true;
        }
        else if (state[action].Count > 5 && state[action][4] == player && state[action][5] == player)
            won = true;
    }

    // diagonal
    if (!won)
    {
        var consec1 = 0;
        var consec2 = 0;
        for (var i = -3; i < 4; i++)
        {
            var x = action + i;
            var y = row + i;
            if (x >= 0 && x < 7 && y >= 0 && y < state[x].Count && state[x][y] == player)
                consec1++;
            else
                consec1 = 0;
            if (consec1 == 4)
            {
                won = true;
                break;
            }

            x = action + i;
            y = row - i;
            if (x >= 0 && x < 7 && y >= 0 && y < state[x].Count && state[x][y] == player)
                consec2++;
            else
                consec2 = 0;
            if (consec2 == 4)
            {
                won = true;
                break;
            }
        }
    }

    if (won)
    {
        done = true;
        if (player == 1)
            Write(10, 15, ConsoleColor.Red, "RED WINS!");
        else
            Write(9, 15, ConsoleColor.Yellow, "YELLOW WINS!");
    }
    else if (state.All(l => l.Count == 6))
    {
        done = true;
        Write(12, 15, ConsoleColor.Green, "DRAW!");
    }

    player = 3 - player;

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

