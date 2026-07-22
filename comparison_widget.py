"""
comparison_widget.py
Halaman Pembanding Performa Model (Model Comparison) untuk Sawit Vision.
Membandingkan beberapa hasil deteksi model terhadap Ground Truth.
"""

import os
from datetime import datetime
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QFileDialog, QMessageBox,
    QLineEdit, QDoubleSpinBox, QFormLayout, QPlainTextEdit, QSplitter
)

import model_comparison


class ComparisonPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.comparison_results = []
        self._build_ui()

    def _build_ui(self):
        # Layout utama menggunakan horizontal splitter agar sidebar dan tabel bisa di-resize
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ----------------------------------------------------
        # SIDEBAR (Control Panel) - Kiri
        # ----------------------------------------------------
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setMinimumWidth(320)
        self.sidebar.setMaximumWidth(380)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(16)

        # Judul & Deskripsi Sidebar
        lbl_section = QLabel("PEMBANDING MODEL")
        lbl_section.setStyleSheet("font-weight: bold; font-size: 14px; letter-spacing: 0.5px;")
        sidebar_layout.addWidget(lbl_section)

        desc = QLabel("Bandingkan beberapa file hasil deteksi (.shp, .gpkg, .geojson) dengan data Ground Truth manual.")
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #8a8d92;")
        sidebar_layout.addWidget(desc)

        # Card 1: Ground Truth
        card_gt = QFrame()
        card_gt.setObjectName("sidebarCard")
        card_gt.setStyleSheet("border: 1px solid rgba(128,128,128,0.2); border-radius: 6px; background-color: rgba(128,128,128,0.05);")
        lay_gt = QVBoxLayout(card_gt)
        lay_gt.setContentsMargins(12, 12, 12, 12)
        lay_gt.setSpacing(8)

        lbl_gt_title = QLabel("DATA GROUND TRUTH (GT)")
        lbl_gt_title.setStyleSheet("font-weight: bold; font-size: 10px; color: #10b981;")
        lay_gt.addWidget(lbl_gt_title)

        row_gt = QHBoxLayout()
        self.gt_edit = QLineEdit()
        self.gt_edit.setReadOnly(True)
        self.gt_edit.setPlaceholderText("Pilih file GT...")
        self.btn_pick_gt = QPushButton("...")
        self.btn_pick_gt.setFixedSize(32, 28)
        self.btn_pick_gt.clicked.connect(self.pick_gt_file)
        row_gt.addWidget(self.gt_edit, 1)
        row_gt.addWidget(self.btn_pick_gt)
        lay_gt.addLayout(row_gt)

        self.gt_info_label = QLabel("Belum ada data.")
        self.gt_info_label.setStyleSheet("font-size: 11px; color: #8a8d92;")
        lay_gt.addWidget(self.gt_info_label)

        sidebar_layout.addWidget(card_gt)

        # Card 2: Parameter
        card_param = QFrame()
        card_param.setObjectName("sidebarCard")
        card_param.setStyleSheet("border: 1px solid rgba(128,128,128,0.2); border-radius: 6px; background-color: rgba(128,128,128,0.05);")
        lay_param = QVBoxLayout(card_param)
        lay_param.setContentsMargins(12, 12, 12, 12)
        lay_param.setSpacing(8)

        lbl_param_title = QLabel("PARAMETER EVALUASI")
        lbl_param_title.setStyleSheet("font-weight: bold; font-size: 10px; color: #10b981;")
        lay_param.addWidget(lbl_param_title)

        form_layout = QFormLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 20.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(1.5)
        form_layout.addRow("Threshold Jarak (m):", self.threshold_spin)
        lay_param.addLayout(form_layout)

        self.btn_auto_threshold = QPushButton("Hitung Otomatis dari GT")
        self.btn_auto_threshold.clicked.connect(self.auto_compute_threshold)
        lay_param.addWidget(self.btn_auto_threshold)

        sidebar_layout.addWidget(card_param)

        # Card 3: Aksi & Status
        card_action = QFrame()
        card_action.setObjectName("sidebarCard")
        card_action.setStyleSheet("border: 1px solid rgba(128,128,128,0.2); border-radius: 6px; background-color: rgba(128,128,128,0.05);")
        lay_action = QVBoxLayout(card_action)
        lay_action.setContentsMargins(12, 12, 12, 12)
        lay_action.setSpacing(8)

        lbl_action_title = QLabel("AKSI & LAPORAN")
        lbl_action_title.setStyleSheet("font-weight: bold; font-size: 10px; color: #10b981;")
        lay_action.addWidget(lbl_action_title)

        self.btn_run = QPushButton("Jalankan Perbandingan")
        self.btn_run.setObjectName("runButton")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self.run_comparison)
        lay_action.addWidget(self.btn_run)

        self.btn_export = QPushButton("Ekspor Laporan Lengkap...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results)
        lay_action.addWidget(self.btn_export)

        sidebar_layout.addWidget(card_action)
        sidebar_layout.addStretch(1)

        # ----------------------------------------------------
        # MAIN AREA (Content) - Kanan
        # ----------------------------------------------------
        self.main_content = QWidget()
        right_layout = QVBoxLayout(self.main_content)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        # Tabel hasil
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Nama Model", "Path Hasil", "Deteksi", "TP", "FP", "FN",
            "Precision", "Recall", "F1-Score", "Mean Dist", "RMSE"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.SelectedClicked)
        right_layout.addWidget(self.table, 3)

        # Tombol pengaturan baris tabel
        btn_row_layout = QHBoxLayout()
        self.btn_add = QPushButton("Tambah Hasil Model...")
        self.btn_add.clicked.connect(self.add_model_row)
        self.btn_remove = QPushButton("Hapus Terpilih")
        self.btn_remove.clicked.connect(self.remove_selected_rows)
        self.btn_clear = QPushButton("Bersihkan Tabel")
        self.btn_clear.clicked.connect(self.clear_table)

        btn_row_layout.addWidget(self.btn_add)
        btn_row_layout.addWidget(self.btn_remove)
        btn_row_layout.addWidget(self.btn_clear)
        btn_row_layout.addStretch(1)
        right_layout.addLayout(btn_row_layout)

        # Log Console
        lbl_console = QLabel("LOG PERBANDINGAN")
        lbl_console.setStyleSheet("font-weight: bold; font-size: 11px;")
        right_layout.addWidget(lbl_console)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        right_layout.addWidget(self.console, 1)

        # Tambahkan ke splitter
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.main_content)
        splitter.setSizes([340, 760])

        main_layout.addWidget(splitter)

    def log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.console.appendPlainText(f"[{ts}] {text}")

    def pick_gt_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih File Ground Truth (Point)", "",
            "Vector Files (*.shp *.gpkg *.geojson *.json)"
        )
        if path:
            self.gt_edit.setText(path)
            try:
                xy, _, _ = model_comparison.read_points_any_full(path)
                self.gt_info_label.setText(f"{len(xy)} titik centroid terdeteksi.")
                self.log(f"Ground Truth berhasil diatur: {os.path.basename(path)} ({len(xy)} titik)")
            except Exception as e:
                self.gt_info_label.setText("Error membaca file.")
                self.log(f"[ERROR] Gagal membaca GT: {e}")

    def add_model_row(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih File Hasil Deteksi Model", "",
            "Vector Files (*.shp *.gpkg *.geojson *.json)"
        )
        if path:
            row = self.table.rowCount()
            self.table.insertRow(row)
            model_name = os.path.splitext(os.path.basename(path))[0]

            item_name = QTableWidgetItem(model_name)
            item_path = QTableWidgetItem(path)
            item_path.setToolTip(path)
            item_path.setFlags(item_path.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.table.setItem(row, 0, item_name)
            self.table.setItem(row, 1, item_path)

            for col in range(2, 11):
                item = QTableWidgetItem("-")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)

            self.log(f"Hasil model ditambahkan: {model_name}")

    def remove_selected_rows(self):
        indices = sorted([index.row() for index in self.table.selectedIndexes()], reverse=True)
        # Unique rows only
        unique_indices = []
        for idx in indices:
            if idx not in unique_indices:
                unique_indices.append(idx)

        for row in unique_indices:
            model_name = self.table.item(row, 0).text()
            self.table.removeRow(row)
            self.log(f"Menghapus baris hasil model: {model_name}")

    def clear_table(self):
        self.table.setRowCount(0)
        self.btn_export.setEnabled(False)
        self.comparison_results = []
        self.log("Tabel dibersihkan.")

    def auto_compute_threshold(self):
        gt_path = self.gt_edit.text().strip()
        if not gt_path:
            QMessageBox.warning(self, "Peringatan", "Pilih file Ground Truth terlebih dahulu.")
            return

        try:
            res = model_comparison.auto_compute_threshold(gt_path)
            threshold = res["threshold"]
            explanation = res["explanation"]

            self.threshold_spin.setValue(threshold)
            self.log(explanation)
            if res.get("warning"):
                self.log(f"[WARNING] {res['warning']}")

            QMessageBox.information(
                self, "Auto-Threshold Berhasil",
                f"Threshold diatur ke {threshold} m berdasarkan analisis spasial GT.\n\n{explanation}"
            )
        except Exception as e:
            self.log(f"[ERROR] Gagal menghitung threshold otomatis: {e}")
            QMessageBox.critical(self, "Error", f"Gagal menghitung threshold otomatis:\n{e}")

    def run_comparison(self):
        gt_path = self.gt_edit.text().strip()
        if not gt_path:
            QMessageBox.warning(self, "Peringatan", "Silakan pilih file Ground Truth terlebih dahulu.")
            return

        row_count = self.table.rowCount()
        if row_count == 0:
            QMessageBox.warning(self, "Peringatan", "Silakan tambah minimal satu file hasil model untuk dibandingkan.")
            return

        try:
            self.log("Memulai perbandingan performa model...")
            self.log(f"Ground Truth: {os.path.basename(gt_path)}")

            manual_xy, manual_attrs, _ = model_comparison.read_points_any_full(gt_path)
            self.log(f"Berhasil membaca {len(manual_xy)} titik Ground Truth.")

            threshold = self.threshold_spin.value()
            self.log(f"Threshold jarak pencocokan: {threshold} meter")

            self.comparison_results = []

            for row in range(row_count):
                model_name = self.table.item(row, 0).text().strip()
                path = self.table.item(row, 1).text().strip()

                self.log(f"Mengevaluasi model: {model_name}...")

                infer_xy, infer_attrs, infer_bboxes = model_comparison.read_points_any_full(path)

                metrics, matches, fp, fn = model_comparison.evaluate_model(
                    manual_xy, infer_xy, threshold, infer_bboxes, infer_attrs
                )

                self.comparison_results.append({
                    "name": model_name,
                    "path": path,
                    "xy": infer_xy,
                    "attrs": infer_attrs,
                    "bboxes": infer_bboxes,
                    "metrics": metrics,
                    "matches": matches,
                    "fp": fp,
                    "fn": fn
                })

                self.table.item(row, 2).setText(str(len(infer_xy)))
                self.table.item(row, 3).setText(str(metrics["tp"]))
                self.table.item(row, 4).setText(str(metrics["fp"]))
                self.table.item(row, 5).setText(str(metrics["fn"]))
                self.table.item(row, 6).setText(f"{metrics['precision']:.1%}")
                self.table.item(row, 7).setText(f"{metrics['recall']:.1%}")
                self.table.item(row, 8).setText(f"{metrics['f1']:.1%}")
                self.table.item(row, 9).setText(f"{metrics['mean_dist']:.3f} m")
                self.table.item(row, 10).setText(f"{metrics['rmse_dist']:.3f} m")

                self.log(f"  -> TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
                self.log(f"  -> Precision: {metrics['precision']:.1%}, Recall: {metrics['recall']:.1%}, F1: {metrics['f1']:.1%}")

            self.btn_export.setEnabled(True)
            self.log("Semua perbandingan model selesai dengan sukses!")
            QMessageBox.information(self, "Sukses", "Perbandingan performa model berhasil dijalankan.")
        except Exception as e:
            self.log(f"[ERROR] Gagal menjalankan perbandingan: {e}")
            import traceback
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Gagal menjalankan perbandingan:\n{e}")

    def export_results(self):
        if not self.comparison_results:
            QMessageBox.warning(self, "Peringatan", "Silakan jalankan perbandingan terlebih dahulu.")
            return

        gt_path = self.gt_edit.text().strip()
        threshold = self.threshold_spin.value()

        out_dir = QFileDialog.getExistingDirectory(self, "Pilih Folder untuk Menyimpan Laporan")
        if not out_dir:
            return

        try:
            self.log("Memulai ekspor laporan komparasi...")

            manual_xy, manual_attrs, _ = model_comparison.read_points_any_full(gt_path)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subfolder_name = f"Perbandingan_Model_{timestamp}"
            export_out_dir = os.path.join(out_dir, export_subfolder_name)
            os.makedirs(export_out_dir, exist_ok=True)

            excel_path = os.path.join(export_out_dir, "perbandingan_model.xlsx")
            model_comparison.export_comparison_excel(
                excel_path, manual_xy, self.comparison_results, threshold, manual_attrs
            )
            self.log(f"Tabel Ringkasan Excel berhasil disimpan ke: {excel_path}")

            model_comparison.export_comparison_points(
                export_out_dir, manual_xy, self.comparison_results, gt_path, manual_attrs
            )
            self.log(f"Detail titik spasial (TP/FP/FN) diekspor ke: {export_out_dir}")

            self.log("Ekspor laporan selesai dengan sukses!")
            QMessageBox.information(
                self, "Ekspor Sukses",
                f"Laporan berhasil diekspor di folder:\n{export_out_dir}\n\n"
                f"Isi folder:\n"
                f"1. perbandingan_model.xlsx (Tabel Ringkasan + Detail)\n"
                f"2. Subfolder per-model berisi Shapefile, GeoJSON, dan CSV (TP, FP, FN, Gabungan)"
            )
        except Exception as e:
            self.log(f"[ERROR] Gagal mengekspor laporan: {e}")
            QMessageBox.critical(self, "Error", f"Gagal mengekspor laporan:\n{e}")

    def apply_theme_icons(self, text_color: str, accent_color: str, accent_text: str):
        # Di sini kita bisa mewarnai tombol aksi utama agar konsisten dengan tema
        self.btn_run.setStyleSheet(f"""
            QPushButton#runButton {{
                background-color: {accent_color};
                color: {accent_text};
                border: 1px solid {accent_color};
                border-radius: 7px;
                padding: 6px 12px;
                font-weight: 700;
            }}
            QPushButton#runButton:hover {{
                opacity: 0.9;
            }}
        """)