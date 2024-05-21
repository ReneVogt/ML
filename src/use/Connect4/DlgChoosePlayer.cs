namespace Connect4
{
    public partial class DlgChoosePlayer : Form
    {
        internal IGenerateMoves RedPlayer { get; private set; } = new HumanMoveGenerator();
        internal IGenerateMoves YellowPlayer { get; private set; } = new HumanMoveGenerator();

        public DlgChoosePlayer()
        {
            InitializeComponent();

            foreach(var type in typeof(IGenerateMoves).Assembly.GetTypes().Where(t => t.IsClass && typeof(IGenerateMoves).IsAssignableFrom(t)))
            {
                cmbRed.Items.Add(Activator.CreateInstance(type)!);
                cmbYellow.Items.Add(Activator.CreateInstance(type)!);
            }

            cmbRed.SelectedIndex = 0;
            cmbYellow.SelectedIndex = 0;
        }

        private void OnGeneratorSelectionChanged(object sender, EventArgs e)
        {
            RedPlayer = (IGenerateMoves)cmbRed.SelectedItem!;
            YellowPlayer = (IGenerateMoves)cmbYellow.SelectedItem!;
        }
    }
}
