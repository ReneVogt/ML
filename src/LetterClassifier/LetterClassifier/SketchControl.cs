
using System.ComponentModel;

namespace LetterClassifier;

public class SketchControl : UserControl
{
    readonly List<Point> _points = [];

    Bitmap _bitmap = new Bitmap(1, 1);
    Graphics _graphics;
    Pen _pen = new(Color.Black);

    public override Image? BackgroundImage
    {
        get => base.BackgroundImage;
        set { }
    }
    [Browsable(false)]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public override ImageLayout BackgroundImageLayout
    {
        get => base.BackgroundImageLayout;
        set { }
    }
    [Browsable(true)]
    [EditorBrowsable(EditorBrowsableState.Always)]
    public float PenSize 
    { 
        get => _pen.Width; 
        set => _pen.Width = value;
    }

    public Image Image => _bitmap;

    public SketchControl()
    {
        base.BackgroundImageLayout = ImageLayout.None;
        _graphics = Graphics.FromImage(_bitmap);
        _graphics.Clear(Color.White);
        DoubleBuffered = true;
    }
    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (!disposing) return;
        _graphics.Dispose();
        _bitmap.Dispose();
        _pen.Dispose();
    }

    protected override void OnCreateControl()
    {
        base.OnCreateControl();
        RecreateBitmap();
        _graphics.Clear(Color.White);
    }
    protected override void OnClientSizeChanged(EventArgs e)
    {
        base.OnClientSizeChanged(e);
        RecreateBitmap();
    }
    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.Button != MouseButtons.Left) return;
        _points.Clear();
        _points.Add(e.Location);
    }
    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (_points.Count == 0) return;
        _points.Add(e.Location);
        _graphics.DrawLines(_pen, _points.ToArray());
        Invalidate();
    }
    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        if (e.Button != MouseButtons.Left) return;
        _points.Clear();
    }

    public void Clear()
    {
        _points.Clear();
        _graphics.Clear(Color.White);
        Invalidate();
    }    

    void RecreateBitmap()
    {
        var oldBitmap = _bitmap;
        _graphics.Dispose();

        _bitmap = new Bitmap(oldBitmap, Math.Max(1, ClientSize.Width), Math.Max(1, ClientSize.Height));
        _graphics = Graphics.FromImage(_bitmap);
        oldBitmap.Dispose();

        base.BackgroundImage = _bitmap;
    }
}
