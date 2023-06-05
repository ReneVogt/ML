using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using System.Text;

namespace Transformer;

public sealed class TransformerModel
{
    readonly Random _random = new Random();
    readonly InferenceSession _model;

    public IReadOnlyDictionary<string, int> Encoder { get; }
    public IReadOnlyDictionary<int, string> Decoder { get; }
    public int SampleSize { get; }
    public int VocabularySize => Encoder.Count;
    public int EmptyIndex { get; set; }

    TransformerModel(InferenceSession model, IDictionary<int, string> decoder)
    {
        _model = model;
        Decoder = decoder.AsReadOnly();
        Encoder = Decoder.ToDictionary(x => x.Value, x => x.Key).AsReadOnly();
        SampleSize = _model.InputMetadata[_model.InputNames[0]].Dimensions[1];
    }

    public IEnumerable<string> Generate() => Continue();
    public IEnumerable<string> Continue(string previous = "")
    {
        var context = Encode(previous ?? string.Empty);

        var contextTensor = new DenseTensor<long>(new[] { 1, SampleSize });
        for (var i = 0; i < SampleSize; i++)
            contextTensor[0, i] = context[i];

        for (; ;)
        {
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(_model.InputNames[0], contextTensor) };
            using var results = _model.Run(inputs);
            var result = results.First().AsTensor<float>();
            var distribution = new double[VocabularySize];
            for (var i = 0; i < VocabularySize; i++)
                distribution[i] = Math.Exp(result[0,SampleSize-1, i]);
            var sum = distribution.Sum();
            for (var i = 0; i < VocabularySize; i++)
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

    public int[] Encode(string context)
    {
        var tokens = new List<int>();
        var index = 0;
        var maxLength = Encoder.Keys.Select(k => k.Length).Max();
        while (index < context.Length)
        {
            var length = Math.Min(maxLength, context.Length - index);
            while (length > 0)
            {
                if (Encoder.ContainsKey(context[index..(index + length)]))
                    break;
                length--;
            }
            if (length == 0)
            {
                tokens.Add(EmptyIndex);
                index++;
            }
            else
            {
                tokens.Add(Encoder[context[index..(index + length)]]);
                index += length;
            }
        }

        return Enumerable.Repeat(EmptyIndex, Math.Max(0, SampleSize - tokens.Count)).Concat(tokens).ToArray();
    }

    public static TransformerModel Load(string vocublaryPath, string modelPath)
    {
        var model = new InferenceSession(modelPath);
        var decoder = JsonConvert.DeserializeObject<Dictionary<string, string>>(File.ReadAllText(vocublaryPath))!.ToDictionary(x => int.Parse(x.Key), x => x.Value);
       
        return new TransformerModel(model, decoder);
    }
    public static TransformerModel Load(IDictionary<int, string> decoder, byte[] model) => new TransformerModel(new InferenceSession(model), decoder);
}