using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;

namespace Transformer;

public sealed class Transformer
{
    readonly Random _random = new Random();
    readonly InferenceSession _model;

    public IReadOnlyDictionary<string, int> Encoder { get; }
    public IReadOnlyDictionary<int, string> Decoder { get; }
    public int SampleSize { get; }

    Transformer(InferenceSession model, Dictionary<int, string> decoder)
    {
        _model = model;
        Decoder = decoder.AsReadOnly();
        Encoder = Decoder.ToDictionary(x => x.Value, x => x.Key).AsReadOnly();
        SampleSize = _model.InputMetadata[_model.InputNames[0]].Dimensions[1];
    }

    public IEnumerable<string> Generate() => Continue(Array.Empty<string>());
    public IEnumerable<string> Continue(string[] previousTokens)
    {
        var context = Enumerable.Repeat(0, Math.Max(SampleSize - previousTokens.Length, 0))
            .Concat(previousTokens.TakeLast(SampleSize).Select(s => Encoder[s])).ToArray();

        var contextTensor = new DenseTensor<long>(new[] { 1, SampleSize });
        for (var i = 0; i < SampleSize; i++)
            contextTensor[0, i] = context[i];

        for (; ;)
        {
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(_model.InputNames[0], contextTensor) };
            using var results = _model.Run(inputs);
            var result = results.First().AsTensor<float>();
            var distribution = new double[Decoder.Count];
            for (var i = 0; i < Decoder.Count; i++)
                distribution[i] = Math.Exp(result[0,SampleSize-1, i]);
            var sum = distribution.Sum();
            for (var i = 0; i < Decoder.Count; i++)
                distribution[i] /= sum;

            var rand = _random.NextSingle();
            var index = 0;
            var acc = distribution[0];
            while(rand > acc)
            {
                index++;
                acc += distribution[index];
            }
            yield return Decoder[index];

            for (var i = 0; i < SampleSize-1; i++)
                contextTensor[0, i] = contextTensor[0, i + 1];
            contextTensor[0, SampleSize - 1] = index;            
        }
    }

    public static Transformer Load(string vocublaryPath, string modelPath)
    {
        var model = new InferenceSession(modelPath);
        var decoder = JsonConvert.DeserializeObject<Dictionary<string, string>>(File.ReadAllText(vocublaryPath))!.ToDictionary(x => int.Parse(x.Key), x => x.Value);
       
        return new Transformer(model, decoder);
    }
}

/*

var session = new InferenceSession("model.onnx");

var inputMeta = session.InputMetadata;
var container = new NamedOnnxValue[inputMeta.Count];
var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 }); // Again, adjust the dimensions accordingly

// Assuming you have preprocessed your input and filled the tensor
container[0] = NamedOnnxValue.CreateFromTensor<float>(inputMeta.Keys.First(), tensor);

using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(container);

// Process the results
foreach (var r in results)
{
    Console.WriteLine($"{r.Name} with data type {r.AsTensor<float>().ElementType}");
}

 */