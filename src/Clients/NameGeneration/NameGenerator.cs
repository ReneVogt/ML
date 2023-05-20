using System.Reflection;
using System.Text;

namespace NameGeneration;

public static class NameGenerator
{
    const int vocabulary = 27;

    static readonly Random _random = new();

    static readonly int _featureSize;
    static readonly int _blockSize;
    static readonly int _hiddenSize;

    static readonly float[] _embedding;
    static readonly float[] _w1;
    static readonly float[] _batchNormGain;
    static readonly float[] _batchNormBias;
    static readonly float[] _batchNormMean;
    static readonly float[] _batchNormStd;
    static readonly float[] _w2;
    static readonly float[] _b2;

    static NameGenerator()
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream("NameGeneration.names.dat")!;
        using var reader = new BinaryReader(stream);

        _featureSize = reader.ReadInt32();
        _blockSize = reader.ReadInt32();
        _hiddenSize = reader.ReadInt32();

        _embedding = Enumerable.Range(0, vocabulary * _featureSize).Select(_ => reader.ReadSingle()).ToArray();
        _w1 = Enumerable.Range(0, _featureSize * _blockSize * _hiddenSize).Select(_ => reader.ReadSingle()).ToArray();
        _batchNormGain = Enumerable.Range(0, _hiddenSize).Select(_ => reader.ReadSingle()).ToArray();
        _batchNormBias = Enumerable.Range(0, _hiddenSize).Select(_ => reader.ReadSingle()).ToArray();
        _batchNormMean = Enumerable.Range(0, _hiddenSize).Select(_ => reader.ReadSingle()).ToArray();
        _batchNormStd = Enumerable.Range(0, _hiddenSize).Select(_ => reader.ReadSingle()).ToArray();
        _w2 = Enumerable.Range(0, vocabulary * _hiddenSize).Select(_ => reader.ReadSingle()).ToArray();
        _b2 = Enumerable.Range(0, vocabulary).Select(_ => reader.ReadSingle()).ToArray();
    }

    public static IEnumerable<string> GetNames()
    {
        while (true) yield return GetName();
    }
    public static string GetName()
    {
        var context = Enumerable.Repeat(0, _blockSize).ToList();
        int next;
        var builder = new StringBuilder();

        do
        {
            var distribution = Forward(context);
            next = Multinomial(distribution);
            if (next != 0)
            {
                context.RemoveAt(0);
                context.Add(next);
                builder.Append((char)('a' + next-1));
            }

        } while (next != 0);

        return builder.ToString();
    }

    static float[] Forward(List<int> context)
    {
        // embedding
        var input = context.SelectMany(i => _embedding.Skip(i * _featureSize).Take(_featureSize)).ToArray();

        // first layer weights
        var o1 = Enumerable.Range(0, _hiddenSize).Select(i => Enumerable.Range(0, _featureSize * _blockSize).Select(j => input[j] * _w1[j * _hiddenSize + i]).Sum()).ToArray();

        // batchnorm and tanh
        for (var i = 0; i < _hiddenSize; i++)
            o1[i] = (float)Math.Tanh(_batchNormGain[i] * (o1[i] - _batchNormMean[i]) / _batchNormStd[i] + _batchNormBias[i]);

        var logits = Enumerable.Range(0, vocabulary).Select(i => Enumerable.Range(0, _hiddenSize).Select(j => o1[j] * _w2[j * vocabulary + i]).Sum() + _b2[i]).ToArray();
        var exps = logits.Select(l => (float)Math.Exp(l)).ToArray();
        var sum = exps.Sum();
        return exps.Select(e => e / sum).ToArray();
    }
    static int Multinomial(float[] distribution)
    {
        var rand = _random.NextSingle();
        var acc = 0f;
        for (var i = 0; i<distribution.Length; i++)
        {
            acc += distribution[i];
            if (rand < acc) return i;
        }
        return distribution.Length-1;
    }
}