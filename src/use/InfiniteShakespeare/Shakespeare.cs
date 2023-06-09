﻿using Newtonsoft.Json;
using System.Reflection;
using Transformer;

namespace InfiniteShakespeare;

public static class Shakespeare
{
    static readonly Lazy<TransformerModel> _transformerInit = new Lazy<TransformerModel>(CreateTransformer);
    public static TransformerModel Model => _transformerInit.Value;

    static TransformerModel CreateTransformer()
    {
        var assembly = Assembly.GetExecutingAssembly();

        using var vocabularyStream = assembly.GetManifestResourceStream($"{nameof(InfiniteShakespeare)}.vocabulary.json")!;
        using var streamReader = new StreamReader(vocabularyStream);
        var decoder = JsonConvert.DeserializeObject<Dictionary<string, string>>(streamReader.ReadToEnd())!.ToDictionary(x => int.Parse(x.Key), x => x.Value);

        using var modelStream = assembly.GetManifestResourceStream($"{nameof(InfiniteShakespeare)}.shakespeare.onnx")!;
        var model = new byte[modelStream.Length];
        modelStream.Read(model, 0, model.Length);
        var transformer = TransformerModel.Load(decoder, model);
        transformer.EmptyIndex = decoder.First(k => k.Value == " ").Key;
        return transformer;
    }
    public static IEnumerable<string> Continue(string context) => Model.Continue(context);
    public static IEnumerable<string> Generate() => Continue(string.Empty);

}