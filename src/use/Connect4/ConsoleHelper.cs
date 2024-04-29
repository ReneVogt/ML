namespace Connect4;

static class ConsoleHelper
{   
    public static int GetKey(params ConsoleKey[] allowed)
    {
        var allowedKeys = allowed.ToList();
        ConsoleKey key;
        do
        {
            key = Console.ReadKey(true).Key;
        } while (!allowedKeys.Contains(key));
        return allowedKeys.IndexOf(key);
    }
    public static void Write(int x, int y, ConsoleColor color, string text)
    {
        Console.SetCursorPosition(x, y);
        Console.ForegroundColor = color;
        Console.Write(text);
        Console.ResetColor();
    }
}
