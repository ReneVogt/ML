using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
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
    public static int GetMove(List<int>[] state)
    {
        var inputTensor = new DenseTensor<float>(new[] { 3*6*7 });
        for (var col = 0; col < 7; col++)
            for (var row = 0; row < 6; row++)
            {
                var r = 5 - row;
                var player = state[col].Count > r ? state[col][r] : 0;
                inputTensor[3*(col * 6 + row) + player] = 1;
            }

        var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(_model.InputNames[0], inputTensor) };
        using var results = _model.Run(inputs);
        var result = results.First().AsTensor<float>();
        return Enumerable.Range(0, 6).Where(a => state[a].Count < 6).MaxBy(a => result[a]);
    }
}