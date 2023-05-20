using System.Reflection;

namespace NameGeneration;

public static class NameGenerator
{
    static NameGenerator()
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream("NameGeneration.names.dat")!;
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        var array = memoryStream.ToArray();
        Console.WriteLine(array.Length);
    }

    public static IEnumerable<string> GetNames()
    {
        while (true) yield return GetName();
    }
    public static string GetName() => "not yet implemented!";
}