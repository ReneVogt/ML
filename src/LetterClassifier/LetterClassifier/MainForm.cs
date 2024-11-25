using TorchSharp;

namespace LetterClassifier;

public partial class MainForm : Form
{
    static long _inferenceCounter;

    readonly ToolTip _labelToolTip = new() { InitialDelay=0 };
    readonly Label[] _targetLabels;
    readonly torch.jit.ScriptModule _module;

    public MainForm()
    {
        try
        {
            _module = torch.jit.load("model.pt");
        }
        catch (Exception exception)
        {
            MessageBox.Show($"Failed to load model: {exception}", "Letter classification", MessageBoxButtons.OK, MessageBoxIcon.Error);
            Environment.Exit(0);
            return;
        }

        InitializeComponent();

        var spacePerLabel = pnImage.Width / 26;
        var labelTop = pnImage.Top + pnImage.Height + 5;

        _targetLabels = Enumerable.Range(0, 26).Select(CreateTargetLabel).ToArray();
        Controls.AddRange(_targetLabels);

        Label CreateTargetLabel(int i) => new()
        {
            Anchor = AnchorStyles.Bottom | AnchorStyles.Left,
            Text = ((char)('a' + i)).ToString(),
            AutoSize = false,
            Height = spacePerLabel,
            Width = spacePerLabel,
            Top = labelTop,
            Left = pnImage.Left + i * spacePerLabel,
            TextAlign = ContentAlignment.MiddleCenter,
        };

        _ = UpdateInferenceAsync();
    }

    private async void OnSketchMouseUp(object sender, MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left) return;
        await UpdateInferenceAsync();
    }

    private async void OnClearClicked(object sender, EventArgs e)
    {
        pnImage.Clear();
        await UpdateInferenceAsync();
    }

    async Task UpdateInferenceAsync()
    {
        long counter = ++_inferenceCounter;

        var scaledBitmap = new Bitmap(pnImage.BackgroundImage!, new Size(28, 28));
        pbPreview.Image?.Dispose();

        float[] normalized = new float[28*28];
        for (int x = 0; x<28; x++)
            for (int y = 0; y<28; y++)
                normalized[x*28 + y] = -((float)scaledBitmap.GetPixel(x, y).R - 128) / 128f;
        pbPreview.Image = scaledBitmap;
        var tensor = torch.tensor(normalized, [1, 1, 28, 28]);
        var inference = await Task.Run(() => Infere(tensor));
        if (counter == _inferenceCounter) UpdateLabels(inference);
    }

    float[] Infere(torch.Tensor input)
    {
        var output = ((torch.Tensor)_module.forward(input)).squeeze(0).data<float>();
        return [.. output];
    }

    void UpdateLabels(float[] inference)
    {
        var softmax = Softmax(inference);

        for (int i = 0; i<26; i++)
        {
            _targetLabels[i].BorderStyle = BorderStyle.None;
            _targetLabels[i].BackColor = Color.FromArgb((int)(255*softmax[i]), Color.Gold);
            _labelToolTip.SetToolTip(_targetLabels[i], $"{100*softmax[i]:F2}%");
        }

        var winner = _targetLabels[Enumerable.Range(0, 26).MaxBy(i => softmax[i])];
        winner.BackColor = Color.Gold;
        winner.BorderStyle = BorderStyle.FixedSingle;
    }

    static float[] Softmax(float[] values)
    {
        var e = values.Select(f => (float)Math.Exp(f)).ToArray();
        var sum = e.Sum();
        return e.Select(f => f/sum).ToArray();
    }
}
