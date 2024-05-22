namespace Connect4
{
    public partial class MainForm : Form
    {
        readonly Connect4Board _environment = new();
        IGenerateMoves _redGenerator = new HumanMoveGenerator(), _yellowGenerator = new HumanMoveGenerator();
        CancellationTokenSource cts = new();

        public MainForm()
        {
            InitializeComponent();            
        }
        async void OnMainFormLoad(object sender, EventArgs e)
        {
            await Task.Yield();
            await RunGameAsync(cts.Token);
        }

        private void OnPaintBoard(object sender, PaintEventArgs e)
        {
            var width = (float)pnBoard.ClientRectangle.Width;
            var height = (float)pnBoard.ClientRectangle.Height;

            var lineWidth = 0.05f * width / 7;
            var boxWidth = (width - 8 * lineWidth)/7;
            using var verticalLinePen = new Pen(Color.Blue, lineWidth);
            for (var i = 0; i<=7; i++)
            {
                var x = i * (lineWidth+boxWidth);
                e.Graphics.DrawLine(verticalLinePen, x, 0, x, height);
            }

            var lineHeight = 0.05f * width / 6;
            var boxHeight = (height - 7 * lineHeight)/6;
            using var horizontalLinePen = new Pen(Color.Blue, lineHeight);
            for (var i = 0; i<=7; i++)
            {
                var y = i * (lineHeight+boxHeight);
                e.Graphics.DrawLine(horizontalLinePen, 0, y, width, y);
            }

            for (var column = 0; column < 7; column++)
                for (var row = 0; row < 6; row++)
                {
                    var player = _environment[column, row];
                    var brush = player switch
                    {
                        1 => Brushes.Red,
                        2 => Brushes.Yellow,
                        _ => Brushes.Black
                    };
                    e.Graphics.FillRectangle(brush, column * (lineWidth + boxWidth) + lineWidth, (5-row) * (lineHeight + boxHeight) + lineHeight, boxWidth, boxHeight);
                }
        }

        private void OnBoardClicked(object sender, MouseEventArgs e)
        {
            if (_environment.Finished) return;
            if ((_environment.Player == 1 ? _redGenerator : _yellowGenerator) is not HumanMoveGenerator generator) return;
            var width = (float)pnBoard.ClientRectangle.Width;
            var move = (int)(7 * e.Location.X / width);
            if (_environment.Height(move) >= 6) return;
            generator.MoveClicked(move);
        }
        async void OnRunClicked(object sender, EventArgs e)
        {
            var dlg = new DlgChoosePlayer();
            if (dlg.ShowDialog(this) != DialogResult.OK) return;

            cts.Cancel();
            cts = new();
            _redGenerator = dlg.RedPlayer;
            _yellowGenerator = dlg.YellowPlayer;
            _environment.Clear();
            pnBoard.Invalidate();
            await RunGameAsync(cts.Token);
        }

        async Task RunGameAsync(CancellationToken cancellationToken)
        {
            lbRed.Text = _redGenerator.Name;
            lbYellow.Text = _yellowGenerator.Name;
            while (!(_environment.Finished || cancellationToken.IsCancellationRequested))
            {
                var generator = _environment.Player == 1 ? _redGenerator : _yellowGenerator;
                var move = await generator.GetMoveAsync(_environment);
                if (cancellationToken.IsCancellationRequested) return;
                _environment.Move(move);
                pnBoard.Invalidate();
            }

            if (cancellationToken.IsCancellationRequested) return;

            switch (_environment.Winner)
            {
                case 1:
                    MessageBox.Show(this, $"{_redGenerator.Name} (red) wins.", "Connect Four");
                    return;
                case 2:
                    MessageBox.Show(this, $"{_yellowGenerator.Name} (yellow) wins.", "Connect Four");
                    return;
                default:
                    MessageBox.Show(this, "Draw.", "Connect Four");
                    return;

            }
        }
    }
}
