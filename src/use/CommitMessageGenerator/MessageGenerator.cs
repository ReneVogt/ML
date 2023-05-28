using Newtonsoft.Json;
using System.Reflection;
using Transformer;

namespace CommitMessageGenerator;

public static class MessageGenerator
{
    static readonly Lazy<TransformerModel> _transformerInit = new Lazy<TransformerModel>(CreateTransformer);
    static TransformerModel Model => _transformerInit.Value;

    static TransformerModel CreateTransformer()
    {
        var assembly = Assembly.GetExecutingAssembly();

        using var vocabularyStream = assembly.GetManifestResourceStream($"{nameof(CommitMessageGenerator)}.vocabulary.json")!;
        using var streamReader = new StreamReader(vocabularyStream);
        var decoder = JsonConvert.DeserializeObject<Dictionary<string, string>>(streamReader.ReadToEnd())!.ToDictionary(x => int.Parse(x.Key), x => x.Value);

        using var modelStream = assembly.GetManifestResourceStream($"{nameof(CommitMessageGenerator)}.commits.onnx")!;
        var model = new byte[modelStream.Length];
        modelStream.Read(model, 0, model.Length);
        return TransformerModel.Load(decoder, model);
    }
    public static string Next() => string.Concat(Model.Generate().TakeWhile(s => s != "\n"));
    public static IEnumerable<string> Generate()
    {
        while (true) yield return Next();
    }
}