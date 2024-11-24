namespace LetterClassifier
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
            pnImage = new Panel();
            btClear = new Button();
            pnPen = new Panel();
            tbPenSize = new TrackBar();
            ((System.ComponentModel.ISupportInitialize)tbPenSize).BeginInit();
            SuspendLayout();
            // 
            // pnImage
            // 
            pnImage.BackColor = Color.White;
            pnImage.BackgroundImageLayout = ImageLayout.None;
            pnImage.BorderStyle = BorderStyle.Fixed3D;
            pnImage.Location = new Point(12, 12);
            pnImage.Name = "pnImage";
            pnImage.Size = new Size(390, 390);
            pnImage.TabIndex = 0;
            pnImage.MouseDown += pnImage_MouseDown;
            pnImage.MouseMove += pnImage_MouseMove;
            pnImage.MouseUp += pnImage_MouseUp;
            // 
            // btClear
            // 
            btClear.AutoSize = true;
            btClear.Image = Properties.Resources.trash;
            btClear.Location = new Point(408, 12);
            btClear.Name = "btClear";
            btClear.Size = new Size(38, 36);
            btClear.TabIndex = 1;
            btClear.UseVisualStyleBackColor = true;
            btClear.Click += OnClearClicked;
            // 
            // pnPen
            // 
            pnPen.Location = new Point(408, 54);
            pnPen.Name = "pnPen";
            pnPen.Size = new Size(38, 36);
            pnPen.TabIndex = 2;
            // 
            // tbPenSize
            // 
            tbPenSize.AutoSize = false;
            tbPenSize.Location = new Point(410, 96);
            tbPenSize.Maximum = 300;
            tbPenSize.Minimum = 1;
            tbPenSize.Name = "tbPenSize";
            tbPenSize.Orientation = Orientation.Vertical;
            tbPenSize.Size = new Size(36, 204);
            tbPenSize.TabIndex = 3;
            tbPenSize.TickStyle = TickStyle.None;
            tbPenSize.Value = 1;
            tbPenSize.ValueChanged += OnPenSizeChanged;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(452, 436);
            Controls.Add(tbPenSize);
            Controls.Add(pnPen);
            Controls.Add(btClear);
            Controls.Add(pnImage);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            Name = "MainForm";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Letter classification";
            ((System.ComponentModel.ISupportInitialize)tbPenSize).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Panel pnImage;
        private Button btClear;
        private Panel pnPen;
        private TrackBar tbPenSize;
    }
}
