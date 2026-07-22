"""
comparison_widget.py
Modul untuk halaman pembanding model di Sawit Vision.
"""

import os
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QFileDialog, QMessageBox
)

class ComparisonPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Pembanding Performa Model (Model Comparison)")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        desc = QLabel("Bandingkan metrik evaluasi dari beberapa model YOLO (.pt) untuk deteksi sawit multispektral.")
        desc.setStyleSheet("color: gray;")
        layout.addWidget(desc)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Nama Model", "Path File", "Precision", "Recall", "mAP50"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Tambah Model...")
        self.btn_add.clicked.connect(self.add_model_row)
        self.btn_clear = QPushButton("Bersihkan Tabel")
        self.btn_clear.clicked.connect(self.clear_table)
        
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

    def add_model_row(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Model YOLO (.pt)", "", "PyTorch Model (*.pt)")
        if path:
            row = self.table.rowCount()
            self.table.insertRow(row)
            model_name = os.path.basename(path)
            
            self.table.setItem(row, 0, QTableWidgetItem(model_name))
            self.table.setItem(row, 1, QTableWidgetItem(path))
            self.table.setItem(row, 2, QTableWidgetItem("-"))
            self.table.setItem(row, 3, QTableWidgetItem("-"))
            self.table.setItem(row, 4, QTableWidgetItem("-"))

    def clear_table(self):
        self.table.setRowCount(0)

    def apply_theme_icons(self, text_color: str, accent_color: str, accent_text: str):
        pass