var transformer = Transformer.Transformer.Load("vocabulary.json", "shakespeare.onnx");
foreach(var sample in transformer.Generate())
{
    Console.Write(sample);
    if (Console.KeyAvailable)
    {
        Console.ReadKey(true);
        Console.ReadKey(true);
    }
}
