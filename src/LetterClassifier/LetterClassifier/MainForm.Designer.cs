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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            pnImage = new SketchControl();
            btClear = new Button();
            pbPreview = new PictureBox();
            ((System.ComponentModel.ISupportInitialize)pbPreview).BeginInit();
            SuspendLayout();
            // 
            // pnImage
            // 
            pnImage.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            pnImage.BackgroundImage = (Image)resources.GetObject("pnImage.BackgroundImage");
            pnImage.BackgroundImageLayout = ImageLayout.None;
            pnImage.BorderStyle = BorderStyle.Fixed3D;
            pnImage.Location = new Point(12, 12);
            pnImage.Name = "pnImage";
            pnImage.PenSize = 20F;
            pnImage.Size = new Size(390, 390);
            pnImage.TabIndex = 0;
            pnImage.MouseUp += OnSketchMouseUp;
            // 
            // btClear
            // 
            btClear.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            btClear.AutoSize = true;
            btClear.Image = Properties.Resources.trash;
            btClear.Location = new Point(408, 12);
            btClear.Name = "btClear";
            btClear.Size = new Size(38, 36);
            btClear.TabIndex = 1;
            btClear.UseVisualStyleBackColor = true;
            btClear.Click += OnClearClicked;
            // 
            // pbPreview
            // 
            pbPreview.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;
            pbPreview.BorderStyle = BorderStyle.FixedSingle;
            pbPreview.Location = new Point(408, 370);
            pbPreview.Name = "pbPreview";
            pbPreview.Size = new Size(32, 32);
            pbPreview.SizeMode = PictureBoxSizeMode.CenterImage;
            pbPreview.TabIndex = 4;
            pbPreview.TabStop = false;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(452, 436);
            Controls.Add(pbPreview);
            Controls.Add(btClear);
            Controls.Add(pnImage);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            Name = "MainForm";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Letter classification";
            ((System.ComponentModel.ISupportInitialize)pbPreview).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private SketchControl pnImage;
        private Button btClear;
        private PictureBox pbPreview;
    }
}
