namespace Connect4
{
    partial class DlgChoosePlayer
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
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
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            gbRed = new GroupBox();
            cmbRed = new ComboBox();
            gbYellow = new GroupBox();
            cmbYellow = new ComboBox();
            btOK = new Button();
            btCancel = new Button();
            gbRed.SuspendLayout();
            gbYellow.SuspendLayout();
            SuspendLayout();
            // 
            // gbRed
            // 
            gbRed.Controls.Add(cmbRed);
            gbRed.Location = new Point(12, 12);
            gbRed.Name = "gbRed";
            gbRed.Size = new Size(200, 60);
            gbRed.TabIndex = 0;
            gbRed.TabStop = false;
            gbRed.Text = "Red";
            // 
            // cmbRed
            // 
            cmbRed.DisplayMember = "Name";
            cmbRed.DropDownStyle = ComboBoxStyle.DropDownList;
            cmbRed.FormattingEnabled = true;
            cmbRed.Location = new Point(6, 22);
            cmbRed.Name = "cmbRed";
            cmbRed.Size = new Size(188, 23);
            cmbRed.TabIndex = 0;
            cmbRed.SelectedIndexChanged += OnGeneratorSelectionChanged;
            // 
            // gbYellow
            // 
            gbYellow.Controls.Add(cmbYellow);
            gbYellow.Location = new Point(218, 12);
            gbYellow.Name = "gbYellow";
            gbYellow.Size = new Size(200, 60);
            gbYellow.TabIndex = 1;
            gbYellow.TabStop = false;
            gbYellow.Text = "Yellow";
            // 
            // cmbYellow
            // 
            cmbYellow.DisplayMember = "Name";
            cmbYellow.DropDownStyle = ComboBoxStyle.DropDownList;
            cmbYellow.FormattingEnabled = true;
            cmbYellow.Location = new Point(6, 22);
            cmbYellow.Name = "cmbYellow";
            cmbYellow.Size = new Size(188, 23);
            cmbYellow.TabIndex = 0;
            cmbYellow.SelectedIndexChanged += OnGeneratorSelectionChanged;
            // 
            // btOK
            // 
            btOK.DialogResult = DialogResult.OK;
            btOK.Location = new Point(262, 78);
            btOK.Name = "btOK";
            btOK.Size = new Size(75, 23);
            btOK.TabIndex = 2;
            btOK.Text = "OK";
            btOK.UseVisualStyleBackColor = true;
            // 
            // btCancel
            // 
            btCancel.DialogResult = DialogResult.Cancel;
            btCancel.Location = new Point(343, 78);
            btCancel.Name = "btCancel";
            btCancel.Size = new Size(75, 23);
            btCancel.TabIndex = 3;
            btCancel.Text = "Cancel";
            btCancel.UseVisualStyleBackColor = true;
            // 
            // DlgChoosePlayer
            // 
            AcceptButton = btOK;
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            CancelButton = btCancel;
            ClientSize = new Size(430, 105);
            Controls.Add(btCancel);
            Controls.Add(btOK);
            Controls.Add(gbYellow);
            Controls.Add(gbRed);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            Name = "DlgChoosePlayer";
            ShowIcon = false;
            ShowInTaskbar = false;
            StartPosition = FormStartPosition.CenterParent;
            Text = "Choose players";
            gbRed.ResumeLayout(false);
            gbYellow.ResumeLayout(false);
            ResumeLayout(false);
        }

        #endregion

        private GroupBox gbRed;
        private ComboBox cmbRed;
        private GroupBox gbYellow;
        private ComboBox cmbYellow;
        private Button btOK;
        private Button btCancel;
    }
}