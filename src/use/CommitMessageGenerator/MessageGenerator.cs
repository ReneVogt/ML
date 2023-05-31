using Newtonsoft.Json;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using Transformer;

namespace CommitMessageGenerator;

public static class MessageGenerator
{
    static readonly Lazy<IEnumerator<string>> _transformerInit = new Lazy<IEnumerator<string>>(CreateTransformer);
    static IEnumerator<string> Generator => _transformerInit.Value;

    static readonly StringBuilder messageBuilder = new();

    static IEnumerator<string> CreateTransformer()
    {
        var assembly = Assembly.GetExecutingAssembly();

        using var vocabularyStream = assembly.GetManifestResourceStream($"{nameof(CommitMessageGenerator)}.vocabulary.json")!;
        using var streamReader = new StreamReader(vocabularyStream);
        var decoder = JsonConvert.DeserializeObject<Dictionary<string, string>>(streamReader.ReadToEnd())!.ToDictionary(x => int.Parse(x.Key), x => x.Value);

        using var modelStream = assembly.GetManifestResourceStream($"{nameof(CommitMessageGenerator)}.commits.onnx")!;
        var model = new byte[modelStream.Length];
        modelStream.Read(model, 0, model.Length);
        var transformer = TransformerModel.Load(decoder, model);
        transformer.EmptyIndex = decoder.First(x => x.Value.EndsWith("\n")).Key;
        return transformer.Generate().GetEnumerator();
    }
    [MethodImpl(MethodImplOptions.Synchronized)]
    public static string Next()
    {
        while(Generator.MoveNext())
        {
            var s = Generator.Current;
            var index = s.IndexOf('\n');
            if (index > -1)
            {
                messageBuilder.Append(s[0..index]);
                var message = messageBuilder.ToString();
                messageBuilder.Clear();
                if (index < s.Length - 1)
                    messageBuilder.Append(s[index + 1]);
                return message;
            }
            messageBuilder.Append(s);
        }
        return messageBuilder.ToString();
    }
    public static IEnumerable<string> Generate()
    {
        while (true) yield return Next();
    }
}