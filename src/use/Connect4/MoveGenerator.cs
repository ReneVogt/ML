using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace Connect4;

static class MoveGenerator
{
    static readonly InferenceSession _model;

    static MoveGenerator()
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var modelStream = assembly.GetManifestResourceStream($"{nameof(Connect4)}.connect4.onnx")!;
        var model = new byte[modelStream.Length];
        modelStream.Read(model, 0, model.Length);
        _model = new InferenceSession(model);
    }
    public static int GetMove(List<int>[] state, int currentPlayer)
    {
        var opponent = 3- currentPlayer;
        var inputTensor = new DenseTensor<float>([3, 6, 7]);
        for (var col = 0; col < 7; col++)
            for (var row = 0; row < 6; row++)
            {
                inputTensor[0, row, col] = state[col].Count <= row || state[col][row] == 0 ? 1 : 0;
                inputTensor[1, row, col] = state[col].Count > row && state[col][row] == currentPlayer ? 1 : 0;
                inputTensor[2, row, col] = state[col].Count > row && state[col][row] == opponent ? 1 : 0;
            }

        var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(_model.InputNames[0], inputTensor) };
        using var results = _model.Run(inputs);
        var result = results.First().AsTensor<float>();
        return Enumerable.Range(0, 6).Where(a => state[a].Count < 6).MaxBy(a => result[a]);
    }
}