namespace Connect4
{
    partial class MainForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            btRun = new Button();
            pnBoard = new PictureBox();
            lbRed = new Label();
            lbYellow = new Label();
            ((System.ComponentModel.ISupportInitialize)pnBoard).BeginInit();
            SuspendLayout();
            // 
            // btRun
            // 
            btRun.BackColor = Color.Green;
            btRun.Location = new Point(240, 21);
            btRun.Name = "btRun";
            btRun.Size = new Size(113, 23);
            btRun.TabIndex = 4;
            btRun.Text = "Go";
            btRun.UseVisualStyleBackColor = false;
            btRun.Click += OnRunClicked;
            // 
            // pnBoard
            // 
            pnBoard.Location = new Point(12, 74);
            pnBoard.Name = "pnBoard";
            pnBoard.Size = new Size(571, 430);
            pnBoard.TabIndex = 5;
            pnBoard.TabStop = false;
            pnBoard.Paint += OnPaintBoard;
            pnBoard.MouseClick += OnBoardClicked;
            // 
            // lbRed
            // 
            lbRed.AutoEllipsis = true;
            lbRed.Font = new Font("Segoe UI", 14.25F, FontStyle.Bold, GraphicsUnit.Point, 0);
            lbRed.ForeColor = Color.Red;
            lbRed.Location = new Point(12, 17);
            lbRed.Name = "lbRed";
            lbRed.Size = new Size(222, 31);
            lbRed.TabIndex = 6;
            lbRed.Text = "Red";
            lbRed.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lbYellow
            // 
            lbYellow.AutoEllipsis = true;
            lbYellow.Font = new Font("Segoe UI", 14.25F, FontStyle.Bold, GraphicsUnit.Point, 0);
            lbYellow.ForeColor = Color.Yellow;
            lbYellow.Location = new Point(361, 17);
            lbYellow.Name = "lbYellow";
            lbYellow.Size = new Size(222, 31);
            lbYellow.TabIndex = 7;
            lbYellow.Text = "Red";
            lbYellow.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.Black;
            ClientSize = new Size(593, 516);
            Controls.Add(lbYellow);
            Controls.Add(lbRed);
            Controls.Add(pnBoard);
            Controls.Add(btRun);
            DoubleBuffered = true;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            Name = "MainForm";
            SizeGripStyle = SizeGripStyle.Hide;
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Connect Four";
            Load += OnMainFormLoad;
            ((System.ComponentModel.ISupportInitialize)pnBoard).EndInit();
            ResumeLayout(false);
        }

        #endregion
        private Button btRun;
        private PictureBox pnBoard;
        private Label lbRed;
        private Label lbYellow;
    }
}
