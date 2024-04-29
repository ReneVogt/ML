namespace Connect4;

static class ConsoleHelper
{   
    public static ConsoleKey GetKey(params ConsoleKey[] allowed)
    {
        ConsoleKey key;
        do
        {
            key = Console.ReadKey(true).Key;
        } while (!allowed.Contains(key));
        return key;
    }
    public static void Write(int x, int y, ConsoleColor color, string text)
    {
        Console.SetCursorPosition(x, y);
        Console.ForegroundColor = color;
        Console.Write(text);
        Console.ResetColor();
    }
}
