using NameGeneration;

do
{
    Console.WriteLine(NameGenerator.GetName());
} while (Console.ReadKey(true).Key != ConsoleKey.Escape);