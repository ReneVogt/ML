using Connect4;
using static Connect4.ConsoleHelper;

Console.CursorVisible = false;

for (; ; )
{
    Console.Clear();
    Write(0, 0, ConsoleColor.Gray, "[ 1 ]:");
    Write(9, 0, ConsoleColor.Red, "USER");
    Write(14, 0, ConsoleColor.Gray, "-");
    Write(16, 0, ConsoleColor.Yellow, "USER");

    Write(0, 1, ConsoleColor.Gray, "[ 2 ]:");
    Write(9, 1, ConsoleColor.Red, "USER");
    Write(14, 1, ConsoleColor.Gray, "-");
    Write(16, 1, ConsoleColor.Yellow, "NN");

    Write(0, 2, ConsoleColor.Gray, "[ 3 ]:");
    Write(9, 2, ConsoleColor.Red, "USER");
    Write(14, 2, ConsoleColor.Gray, "-");
    Write(16, 2, ConsoleColor.Yellow, "MC");


    Write(0, 4, ConsoleColor.Gray, "[ 4 ]:");
    Write(11, 4, ConsoleColor.Red, "NN");
    Write(14, 4, ConsoleColor.Gray, "-");
    Write(16, 4, ConsoleColor.Yellow, "USER");

    Write(0, 5, ConsoleColor.Gray, "[ 5 ]:");
    Write(11, 5, ConsoleColor.Red, "NN");
    Write(14, 5, ConsoleColor.Gray, "-");
    Write(16, 5, ConsoleColor.Yellow, "NN");

    Write(0, 6, ConsoleColor.Gray, "[ 6 ]:");
    Write(11, 6, ConsoleColor.Red, "NN");
    Write(14, 6, ConsoleColor.Gray, "-");
    Write(16, 6, ConsoleColor.Yellow, "MC");


    Write(0, 8, ConsoleColor.Gray, "[ 7 ]:");
    Write(11, 8, ConsoleColor.Red, "MC");
    Write(14, 8, ConsoleColor.Gray, "-");
    Write(16, 8, ConsoleColor.Yellow, "USER");

    Write(0, 9, ConsoleColor.Gray, "[ 8 ]:");
    Write(11, 9, ConsoleColor.Red, "MC");
    Write(14, 9, ConsoleColor.Gray, "-");
    Write(16, 9, ConsoleColor.Yellow, "NN");

    Write(0, 10, ConsoleColor.Gray, "[ 9 ]:");
    Write(11, 10, ConsoleColor.Red, "MC");
    Write(14, 10, ConsoleColor.Gray, "-");
    Write(16, 10, ConsoleColor.Yellow, "MC");

    
    var key = GetKey(ConsoleKey.D1, ConsoleKey.D2, ConsoleKey.D3, ConsoleKey.D4, ConsoleKey.D5, ConsoleKey.D6, ConsoleKey.D7, ConsoleKey.D8, ConsoleKey.D9, ConsoleKey.Escape);
    if (key > 8) return;
    new GameRunner(
        key switch
        {
            < 3 => new KeyboardMoveGenerator(),
            < 6 => new NeuralNetworkMoveGenerator(),
            _ => new MonteCarloMoveGenerator()
        },
        (key % 3) switch
        {
            0 => new KeyboardMoveGenerator(),
            1 => new NeuralNetworkMoveGenerator(),
            _ => new MonteCarloMoveGenerator()
        }).Run();
}

