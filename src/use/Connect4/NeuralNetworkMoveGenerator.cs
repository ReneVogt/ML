using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace Connect4;

sealed class NeuralNetworkMoveGenerator : IGenerateMoves
{
    static readonly InferenceSession inferenceSession;

    public string Name => "NeuralNet";

    static NeuralNetworkMoveGenerator()
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var modelStream = assembly.GetManifestResourceStream($"{nameof(Connect4)}.connect4.onnx")!;
        var model = new byte[modelStream.Length];
        modelStream.Read(model, 0, model.Length);
        inferenceSession = new InferenceSession(model);
    }

    public Task<int> GetMoveAsync(Connect4Board env)
    {
        var inputTensor = new DenseTensor<float>([1, 3, 6, 7]);
        for (var col = 0; col < 7; col++)
            for (var row = 0; row < 6; row++)
            {
                inputTensor[0, 0, row, col] = env[col,row] == 0 ? 1 : 0;
                inputTensor[0, 1, row, col] = env[col, row] == env.Player ? 1 : 0;
                inputTensor[0, 2, row, col] = env[col, row] == env.Opponent ? 1 : 0;
            }

        lock (inferenceSession)
        {
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inferenceSession.InputNames[0], inputTensor) };
            using var results = inferenceSession.Run(inputs);
            var result = results.First().AsTensor<float>();
            return Task.FromResult(Enumerable.Range(0, 7).Where(a => env.Height(a) < 6).MaxBy(a => result[0, a]));
        }
    }
}