"""
comparison_widget.py
Halaman "Pembanding Model" untuk Sawit Vision.

User pilih 1 shapefile centroid manual (ground truth) + N shapefile hasil
inference (masing-masing diberi nama model bebas), lalu jalankan evaluasi
(Precision / Recall / F1 / error jarak) dan export hasilnya ke Excel sebagai
acuan penilaian model AI.
"""
import os
import traceback

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QLineEdit, QPushButton,
    QFileDialog, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QMessageBox,
    QListWidget, QListWidgetItem, QInputDialog, QHeaderView, QSplitter,
    QPlainTextEdit, QStyle,
)

from model_comparison import read_points_any, evaluate_model, export_comparison_excel

VECTOR_FILTER = "Vector Files (*.shp *.gpkg *.geojson *.json);;Shapefile (*.shp);;GeoPackage (*.gpkg);;GeoJSON (*.geojson *.json)"


class ComparisonWorker(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, manual_path, model_entries, threshold):
        super().__init__()
        self.manual_path = manual_path
        self.model_entries = model_entries  # list of (name, path)
        self.threshold = threshold
        self.manual_xy = None
        self.manual_attrs = []

    def run(self):
        try:
            self.log.emit(f"Membaca centroid manual: {self.manual_path}")
            self.manual_xy, self.manual_attrs = read_points_any(self.manual_path)
            self.log.emit(f"  -> {len(self.manual_xy)} titik manual.")

            results = []
            for name, path in self.model_entries:
                self.log.emit(f"Memproses model '{name}': {path}")
                infer_xy, infer_attrs = read_points_any(path)
                metrics, matches, fp, fn = evaluate_model(self.manual_xy, infer_xy, self.threshold)
                self.log.emit(
                    f"  -> N={metrics['n_infer']} TP={metrics['tp']} FP={metrics['fp']} "
                    f"FN={metrics['fn']} | P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                    f"F1={metrics['f1']:.3f} | mean_dist={metrics['mean_dist']:.3f}m"
                )
                results.append({
                    "name": name, "path": path, "xy": infer_xy, "attrs": infer_attrs,
                    "metrics": metrics, "matches": matches, "fp": fp, "fn": fn,
                })
            self.finished.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ComparisonPage(QWidget):
    """Halaman penuh: sidebar kiri (input) + tabel hasil & log di kanan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manual_path = None
        self.model_entries = []  # list of (name, path)
        self.manual_xy = None
        self.manual_attrs = []
        self.results = None
        self.threshold = 1.0
        self.worker = None
        self.thread = None
        self._build_ui()

    # ------------------------------------------------------------
    def _std_icon(self, sp):
        return self.style().standardIcon(sp)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        split = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- Sidebar kiri ----------------
        side = QWidget()
        side.setObjectName("sidebar")
        sv = QVBoxLayout(side)
        sv.setContentsMargins(10, 10, 10, 10)
        sv.setSpacing(12)

        card_manual = QFrame()
        card_manual.setObjectName("sidebarCard")
        lm = QVBoxLayout(card_manual)
        lm.setContentsMargins(12, 12, 12, 12)
        lm.setSpacing(8)
        title_manual = QLabel("Centroid Manual (Ground Truth)")
        title_manual.setObjectName("sidebarCardTitle")
        lm.addWidget(title_manual)
        row_m = QHBoxLayout()
        self.txt_manual = QLineEdit()
        self.txt_manual.setReadOnly(True)
        self.txt_manual.setPlaceholderText("Belum dipilih (.shp / .gpkg / .geojson)")
        btn_manual = QPushButton()
        btn_manual.setIcon(self._std_icon(QStyle.StandardPixmap.SP_DirOpenIcon))
        btn_manual.setToolTip("Pilih file centroid manual")
        btn_manual.setFixedSize(34, 30)
        btn_manual.clicked.connect(self.pick_manual)
        row_m.addWidget(self.txt_manual, 1)
        row_m.addWidget(btn_manual)
        lm.addLayout(row_m)
        sv.addWidget(card_manual)

        card_models = QFrame()
        card_models.setObjectName("sidebarCard")
        lmo = QVBoxLayout(card_models)
        lmo.setContentsMargins(12, 12, 12, 12)
        lmo.setSpacing(8)
        title_models = QLabel("Model / Hasil Inference Dibandingkan")
        title_models.setObjectName("sidebarCardTitle")
        lmo.addWidget(title_models)

        self.list_models = QListWidget()
        self.list_models.setMinimumHeight(90)
        self.list_models.setMaximumHeight(160)
        self.list_models.setSpacing(3)
        self.list_models.setAlternatingRowColors(True)
        lmo.addWidget(self.list_models)

        self.lbl_empty_hint = QLabel("Belum ada model. Klik \u201c+ Tambah Model\u201d untuk mulai.")
        self.lbl_empty_hint.setWordWrap(True)
        self.lbl_empty_hint.setStyleSheet("color: #8a8d92; font-size: 11px; padding: 2px 0;")
        lmo.addWidget(self.lbl_empty_hint)

        row_btn = QHBoxLayout()
        btn_add = QPushButton(" Tambah Model")
        btn_add.setIcon(self._std_icon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        btn_add.clicked.connect(self.add_model)
        btn_remove = QPushButton()
        btn_remove.setIcon(self._std_icon(QStyle.StandardPixmap.SP_TrashIcon))
        btn_remove.setToolTip("Hapus model terpilih")
        btn_remove.setFixedWidth(40)
        btn_remove.clicked.connect(self.remove_model)
        row_btn.addWidget(btn_add, 1)
        row_btn.addWidget(btn_remove)
        lmo.addLayout(row_btn)
        sv.addWidget(card_models)

        card_param = QFrame()
        card_param.setObjectName("sidebarCard")
        lp = QVBoxLayout(card_param)
        lp.setContentsMargins(12, 12, 12, 12)
        lp.setSpacing(8)
        title_param = QLabel("Threshold Toleransi Jarak")
        title_param.setObjectName("sidebarCardTitle")
        lp.addWidget(title_param)
        row_th = QHBoxLayout()
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.05, 100.0)
        self.spin_threshold.setSingleStep(0.1)
        self.spin_threshold.setSuffix(" m")
        self.spin_threshold.setValue(1.0)
        row_th.addWidget(self.spin_threshold, 1)
        lp.addLayout(row_th)
        hint = QLabel(
            "Deteksi dianggap benar (TP) jika \u2264 jarak ini dari titik\n"
            "manual terdekat. Acuan: \u00bd radius kanopi sawit (\u2248 1\u20131.5 m)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a8d92; font-size: 11px;")
        lp.addWidget(hint)
        sv.addWidget(card_param)

        self.btn_run = QPushButton("Jalankan Perbandingan")
        self.btn_run.setObjectName("runButton")
        self.btn_run.setIcon(self._std_icon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self.run_comparison)
        sv.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export ke Excel...")
        self.btn_export.setIcon(self._std_icon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_excel)
        sv.addWidget(self.btn_export)

        sv.addStretch(1)

        # ---------------- Panel kanan ----------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 8, 8, 8)
        rv.setSpacing(8)

        rv.addWidget(QLabel("Ringkasan Perbandingan Model"))
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "Model", "N Deteksi", "TP", "FP", "FN",
            "Precision", "Recall", "F1", "Mean Dist (m)", "RMSE (m)",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rv.addWidget(self.table, 1)

        rv.addWidget(QLabel("Log Proses"))
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(160)
        rv.addWidget(self.log_box)

        split.addWidget(side)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 1100])

        root.addWidget(split)

    # ------------------------------------------------------------
    def pick_manual(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Centroid Manual", "", VECTOR_FILTER)
        if path:
            self.manual_path = path
            self.txt_manual.setText(path)

    def add_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Hasil Inference Model", "", VECTOR_FILTER)
        if not path:
            return
        default_name = os.path.splitext(os.path.basename(path))[0]
        name, ok = QInputDialog.getText(self, "Nama Model", "Beri nama model ini:", text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        self.model_entries.append((name, path))
        item = QListWidgetItem(f"{name}  \u2014  {os.path.basename(path)}")
        item.setIcon(self._std_icon(QStyle.StandardPixmap.SP_FileIcon))
        self.list_models.addItem(item)
        self._update_empty_hint()

    def remove_model(self):
        row = self.list_models.currentRow()
        if row < 0:
            return
        self.list_models.takeItem(row)
        del self.model_entries[row]
        self._update_empty_hint()

    def _update_empty_hint(self):
        self.lbl_empty_hint.setVisible(self.list_models.count() == 0)

    def run_comparison(self):
        if not self.manual_path:
            QMessageBox.warning(self, "Peringatan", "Pilih dulu centroid manual (ground truth).")
            return
        if not self.model_entries:
            QMessageBox.warning(self, "Peringatan", "Tambahkan minimal 1 model untuk dibandingkan.")
            return

        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.log_box.clear()
        self.table.setRowCount(0)

        self.threshold = self.spin_threshold.value()
        self.thread = QThread()
        self.worker = ComparisonWorker(self.manual_path, list(self.model_entries), self.threshold)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log_box.appendPlainText)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def _on_finished(self, results):
        self.results = results
        self.manual_xy = self.worker.manual_xy
        self.manual_attrs = self.worker.manual_attrs
        self._fill_table(results)
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.log_box.appendPlainText("Selesai.")

    def _on_failed(self, msg):
        self.btn_run.setEnabled(True)
        self.log_box.appendPlainText("ERROR:\n" + msg)
        QMessageBox.critical(self, "Gagal", "Perbandingan gagal dijalankan. Lihat log untuk detail.")

    def _fill_table(self, results):
        self.table.setRowCount(len(results))
        for row, r in enumerate(results):
            m = r["metrics"]
            values = [
                r["name"], m["n_infer"], m["tp"], m["fp"], m["fn"],
                f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}",
                f"{m['mean_dist']:.3f}", f"{m['rmse_dist']:.3f}",
            ]
            for col, val in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(str(val)))

    def export_excel(self):
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Excel", "perbandingan_model.xlsx", "Excel Files (*.xlsx)")
        if not path:
            return
        try:
            export_comparison_excel(
                path, self.manual_xy, self.results, self.threshold,
                manual_attrs=getattr(self, "manual_attrs", None),
            )
            QMessageBox.information(self, "Berhasil", f"Excel tersimpan:\n{path}")
        except Exception:
            QMessageBox.critical(self, "Gagal", "Gagal menyimpan Excel:\n" + traceback.format_exc())
