using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch.nn;

namespace LetterClassifier;

public partial class MainForm : Form
{
    readonly Bitmap _bitmap;
    readonly Bitmap _penBitmap;
    readonly Graphics _graphics, _penGraphics;
    readonly List<Point> _points = new();
    readonly ToolTip _labelToolTip = new() { InitialDelay=0 };

    readonly Label[] _targetLabels;

    readonly torch.jit.ScriptModule _module;

    float PenSize => tbPenSize.Value / 10f;
    static long _inferenceCounter;

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
        _bitmap = new(pnImage.Width, pnImage.Height);
        _graphics = Graphics.FromImage(_bitmap);
        _graphics.Clear(Color.White);
        pnImage.BackgroundImage = _bitmap;

        _penBitmap = new(pnPen.Width, pnPen.Height);
        _penGraphics = Graphics.FromImage(_penBitmap);
        pnPen.BackgroundImage = _penBitmap;

        typeof(Panel).InvokeMember("DoubleBuffered",
            System.Reflection.BindingFlags.SetProperty |
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic,
            null, pnImage, [true]);
        typeof(Panel).InvokeMember("DoubleBuffered",
            System.Reflection.BindingFlags.SetProperty |
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic,
            null, pnPen, [true]);

        UpdatePenSizeImage();

        var spacePerLabel = pnImage.Width / 26;
        var labelTop = pnImage.Top + pnImage.Height + 5;

        _targetLabels = Enumerable.Range(0, 26).Select(CreateTargetLabel).ToArray();
        Controls.AddRange(_targetLabels);

        Label CreateTargetLabel(int i) => new Label
        {
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

    private void pnImage_MouseDown(object sender, MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left) return;
        _points.Clear();
        _points.Add(e.Location);
    }
    private void pnImage_MouseMove(object sender, MouseEventArgs e)
    {
        if (_points.Count == 0) return;
        _points.Add(e.Location);
        using var pen = new Pen(Color.Black, PenSize);
        _graphics.DrawCurve(pen, _points.ToArray());
        pnImage.Invalidate();
    }
    private async void pnImage_MouseUp(object sender, MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left) return;
        _points.Clear();
        await UpdateInferenceAsync();
    }

    private void OnPenSizeChanged(object sender, EventArgs e)
    {
        UpdatePenSizeImage();
    }

    private async void OnClearClicked(object sender, EventArgs e)
    {
        _graphics.Clear(Color.White);
        pnImage.Invalidate();
        await UpdateInferenceAsync();
    }

    void UpdatePenSizeImage()
    {
        _penGraphics.Clear(SystemColors.Control);
        using var pen = new Pen(Color.Black, PenSize);
        _penGraphics.DrawLine(pen, 2, pnPen.Height / 2, pnPen.Height-2, pnPen.Height / 2);
        pnPen.Invalidate();
    }

    async Task UpdateInferenceAsync()
    {
        long counter = ++_inferenceCounter;

        using var scaledBitmap = new Bitmap(_bitmap, new Size(28, 28));
        var inference = await Task.Run(() => Infere(scaledBitmap));
        if (counter == _inferenceCounter) UpdateLabels(inference);
    }

    float[] Infere(Bitmap scaledBitmap)
    {
        float[] normalized = new float[28*28];
        for (int x = 0; x<28; x++)
            for (int y = 0; y<28; y++)
                normalized[x*28 + y] = -((float)scaledBitmap.GetPixel(x, y).R - 128) / 128f;

        var tensor = torch.tensor(normalized, [1, 1, 28, 28]);
        //for (int x = 0; x<28; x++)
        //    Debug.WriteLine(string.Join(", ", Enumerable.Range(0, 28).Select(y => tensor[0][0][x][y].item<float>())));
        var output = ((torch.Tensor)_module.forward(tensor)).squeeze(0).data<float>();
        return output.ToArray();
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
