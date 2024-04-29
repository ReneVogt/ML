using Connect4;
using static Connect4.ConsoleHelper;

Console.CursorVisible = false;

for (; ; )
{
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
    switch (GetKey(ConsoleKey.D1, ConsoleKey.D2, ConsoleKey.D3, ConsoleKey.D4, ConsoleKey.Escape))
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
        case ConsoleKey.Escape:
            return;
    }

    new GameRunner(redHuman, yellowHuman).Run();
}

