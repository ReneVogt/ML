using InfiniteShakespeare;

foreach (var sample in Shakespeare.Generate())
{
    Console.Write(sample);
    if (Console.KeyAvailable)
    {
        Console.ReadKey(true);
        Console.ReadKey(true);
    }
}
