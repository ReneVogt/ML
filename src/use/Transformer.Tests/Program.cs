using CommitMessageGenerator;
using InfiniteShakespeare;

#pragma warning disable CS8321

//TestCommitMessageGeneartor();
TestShakespeare();


static void TestShakespeare()
{
    const int outputLength = 200;
    while (true) 
    {
        Console.Write("Ask William: ");
        var context = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(context)) return;
        var slashN = false;
        var length = 0;
        foreach (var sample in Shakespeare.Continue(context))
        {
            var index = sample.IndexOf("\n\n");
            if (index >= 0 && length + index >= outputLength)
            {
                Console.Write(sample[0..(index + 2)]);
                break;
            }

            if (slashN && sample.StartsWith("\n") && length > outputLength)
            {
                Console.Write('\n');
                break;
            }

            Console.Write(sample);
            length += sample.Length;
            slashN = sample.EndsWith("\n");
        }
    }
}
static void TestCommitMessageGeneartor()
{
    foreach (var sample in MessageGenerator.Generate())
    {
        Console.WriteLine(sample);
        if (Console.KeyAvailable)
        {
            Console.ReadKey(true);
            Console.ReadKey(true);
        }
    }
}