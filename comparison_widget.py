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
    QPlainTextEdit,
)

from model_comparison import (
    read_points_any, read_points_any_full, evaluate_model,
    export_comparison_excel, export_comparison_points,
    auto_compute_threshold,
)

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
            # GT selalu POINT, jadi cukup pakai read_points_any (return 2-tuple)
            self.manual_xy, self.manual_attrs = read_points_any(self.manual_path)
            self.log.emit(f"  -> {len(self.manual_xy)} titik manual.")

            results = []
            for name, path in self.model_entries:
                self.log.emit(f"Memproses model '{name}': {path}")
                # Inference SELALU dibaca dengan bbox -- kalau inputnya POLYGON
                # (hasil Sawit Vision), bbox otomatis terisi dan matching pakai
                # containment. Kalau inputnya POINT, bbox = list of None dan
                # evaluate_model auto-fallback ke distance-based (backward compat).
                infer_xy, infer_attrs, infer_bboxes = read_points_any_full(path)
                metrics, matches, fp, fn = evaluate_model(
                    self.manual_xy, infer_xy, self.threshold,
                    infer_bboxes=infer_bboxes, infer_attrs=infer_attrs,
                )
                mode = metrics.get("match_mode", "distance")
                self.log.emit(
                    f"  -> mode={mode.upper()} | N={metrics['n_infer']} "
                    f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} | "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                    f"F1={metrics['f1']:.3f} | mean_dist={metrics['mean_dist']:.3f}m"
                )
                results.append({
                    "name": name, "path": path, "xy": infer_xy, "attrs": infer_attrs,
                    "bboxes": infer_bboxes,
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
    def _icon(self, name: str, color: str = None, size: int = 16):
        """Ambil ikon custom flat satu-warna yang sama seperti menu utama
        (bukan ikon bawaan OS), supaya tampilan halaman ini konsisten dengan
        menu utama. Import ditunda (deferred) untuk menghindari circular
        import dengan main_window.py yang meng-import ComparisonPage ini."""
        from main_window import Icons
        return Icons.icon(name, color or self._icon_color, size)

    def apply_theme_icons(self, text_color: str, accent_color: str, accent_text_color: str = "#ffffff"):
        """Dipanggil dari MainWindow._apply_theme() supaya ikon di halaman ini
        ikut berganti warna saat tema dark/light di-toggle, sama seperti ikon
        di menu utama."""
        self._icon_color = text_color
        self._accent_color = accent_color
        self._accent_text_color = accent_text_color
        self.btn_manual.setIcon(self._icon("folder", text_color, 16))
        self.btn_add.setIcon(self._icon("add", text_color, 16))
        self.btn_remove.setIcon(self._icon("trash", text_color, 16))
        self.btn_run.setIcon(self._icon("run", accent_text_color, 16))
        self.btn_export.setIcon(self._icon("export", text_color, 16))
        for i in range(self.list_models.count()):
            self.list_models.item(i).setIcon(self._icon("file", text_color, 14))

    def _build_ui(self):
        # Warna default (tema gelap) dipakai saat widget pertama kali dibuat;
        # akan diperbarui lagi lewat apply_theme_icons() saat tema berganti.
        self._icon_color = "#eceef0"
        self._accent_color = "#2fbf71"
        self._accent_text_color = "#ffffff"

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
        btn_manual = self.btn_manual = QPushButton()
        btn_manual.setIcon(self._icon("folder"))
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
        btn_add = self.btn_add = QPushButton(" Tambah Model")
        btn_add.setIcon(self._icon("add"))
        btn_add.clicked.connect(self.add_model)
        btn_remove = self.btn_remove = QPushButton()
        btn_remove.setIcon(self._icon("trash"))
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

        # Row 1: spin box (input threshold yang bisa di-edit user) + tombol auto
        row_th = QHBoxLayout()
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.05, 100.0)
        self.spin_threshold.setSingleStep(0.1)
        self.spin_threshold.setSuffix(" m")
        self.spin_threshold.setValue(1.0)
        self.spin_threshold.setToolTip(
            "Threshold matching (dalam meter).\n"
            "Deteksi dianggap benar (True Positive) apabila berjarak \u2264 nilai ini\n"
            "terhadap titik ground truth terdekat.\n\n"
            "Klik 'Hitung Otomatis' untuk mengisi nilai rekomendasi\n"
            "berdasarkan struktur spasial data ground truth yang dipilih."
        )
        # Kalau user manual edit angka, tandai bahwa ini sudah "override"
        # (bukan lagi auto), supaya kita bisa update label info-nya.
        self.spin_threshold.valueChanged.connect(self._on_threshold_edited)

        self.btn_auto_threshold = QPushButton("Hitung Otomatis")
        self.btn_auto_threshold.setToolTip(
            "Menghitung threshold matching secara otomatis dari jarak\n"
            "antar-tetangga titik ground truth (median nearest-neighbor / 2 x faktor keamanan 0.7).\n\n"
            "Memerlukan file Ground Truth yang sudah dipilih pada kolom Centroid Manual."
        )
        self.btn_auto_threshold.clicked.connect(self.auto_threshold_from_gt)
        row_th.addWidget(self.spin_threshold, 1)
        row_th.addWidget(self.btn_auto_threshold)
        lp.addLayout(row_th)

        # Label info: menjelaskan sumber angka threshold (auto vs manual)
        # supaya user tahu dari mana angka datang, dan gampang trace kalau
        # nanti hasilnya aneh.
        self.lbl_threshold_info = QLabel(
            "Sumber nilai: default aplikasi (1.0 m).\n"
            "Klik 'Hitung Otomatis' untuk mendapatkan rekomendasi berdasarkan data ground truth.\n\n"
            "Catatan: threshold jarak hanya berlaku untuk data inference berformat POINT.\n"
            "Untuk data POLYGON (hasil Sawit Vision), pencocokan menggunakan metode\n"
            "containment (titik ground truth berada di dalam bounding box deteksi),\n"
            "dengan confidence sebagai penentu apabila terjadi ambiguitas."
        )
        self.lbl_threshold_info.setWordWrap(True)
        self.lbl_threshold_info.setStyleSheet("color: #8a8d92; font-size: 11px;")
        lp.addWidget(self.lbl_threshold_info)

        # Flag internal: apakah angka spin_threshold saat ini adalah hasil
        # auto-compute (True) atau sudah di-override manual (False).
        # Dipakai untuk logika update lbl_threshold_info tanpa nyala-mati loop.
        self._threshold_is_auto = False
        # Cache hasil terakhir auto_compute_threshold, untuk info tambahan.
        self._auto_threshold_result = None

        sv.addWidget(card_param)

        self.btn_run = QPushButton("Jalankan Perbandingan")
        self.btn_run.setObjectName("runButton")
        self.btn_run.setIcon(self._icon("run", self._accent_text_color))
        self.btn_run.setMinimumHeight(36)
        self.btn_run.clicked.connect(self.run_comparison)
        sv.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export Hasil...")
        self.btn_export.setIcon(self._icon("export"))
        self.btn_export.setToolTip(
            "Simpan semua hasil ke dalam satu folder:\n"
            "Excel ringkasan + Shapefile/GeoJSON/CSV titik TP-FP-FN per model."
        )
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results)
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
    def _set_threshold_programmatically(self, value):
        """Set nilai spin_threshold TANPA memicu tanda 'override manual'.
        Dipakai saat hasil auto-compute di-inject ke spin box -- kita tidak
        mau valueChanged signal-nya menganggap ini sebagai edit user.
        """
        # blockSignals memastikan valueChanged tidak trigger _on_threshold_edited
        # saat kita set nilai secara program (bukan dari user).
        self.spin_threshold.blockSignals(True)
        self.spin_threshold.setValue(value)
        self.spin_threshold.blockSignals(False)

    def _on_threshold_edited(self, value):
        """Dipicu saat user MANUAL edit angka di spin box (bukan program).
        Ubah status jadi 'override manual' dan update label info-nya."""
        if self._threshold_is_auto:
            # Transisi dari auto -> manual override
            self._threshold_is_auto = False
        if self._auto_threshold_result:
            auto_val = self._auto_threshold_result.get("threshold")
            delta = value - auto_val
            info = (
                f"Sumber nilai: override manual (rekomendasi otomatis = {auto_val:.2f} m, "
                f"selisih {delta:+.2f} m).\n"
                f"Klik 'Hitung Otomatis' untuk kembali ke nilai rekomendasi."
            )
        else:
            info = (
                "Sumber nilai: input manual.\n"
                "Klik 'Hitung Otomatis' untuk mendapatkan rekomendasi berdasarkan data ground truth."
            )
        self.lbl_threshold_info.setText(info)

    def auto_threshold_from_gt(self):
        """Panggil auto_compute_threshold() dari logic layer, isi ke spin box,
        dan update label info supaya user tahu dari mana angkanya datang."""
        if not self.manual_path:
            QMessageBox.warning(
                self, "Peringatan",
                "Pilih dulu file Centroid Manual (GT) di atas sebelum menghitung "
                "threshold otomatis. Threshold otomatis diturunkan dari struktur "
                "spasial titik-titik GT."
            )
            return

        try:
            result = auto_compute_threshold(self.manual_path)
        except ValueError as e:
            QMessageBox.warning(self, "Gagal Hitung Otomatis", str(e))
            return
        except Exception:
            QMessageBox.critical(
                self, "Error",
                "Gagal membaca file GT:\n" + traceback.format_exc()
            )
            return

        # Cache hasil untuk referensi label
        self._auto_threshold_result = result
        self._threshold_is_auto = True

        # Set nilai ke spin box tanpa memicu tanda 'override'
        self._set_threshold_programmatically(result["threshold"])

        # Update label info dengan penjelasan lengkap
        stats = result.get("nn_stats") or {}
        info_lines = [
            f"Sumber nilai: otomatis, dihitung dari {result['n_points']} titik ground truth.",
            f"Median jarak antar-tetangga: {stats.get('nn_median', 0):.2f} m",
            f"Threshold = (median / 2) x 0.7 = {result['threshold']:.2f} m",
            f"Rentang: konservatif {result['conservative']:.2f} m \u2014 longgar {result['liberal']:.2f} m",
        ]
        if result.get("warning"):
            info_lines.append(f"WARNING: {result['warning']}")
        self.lbl_threshold_info.setText("\n".join(info_lines))

        # Log ke box log juga supaya user gampang lihat historisnya
        self.log_box.appendPlainText(result["explanation"])
        if result.get("warning"):
            self.log_box.appendPlainText("!! " + result["warning"])

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
        item.setIcon(self._icon("file", size=14))
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

    def export_results(self):
        if not self.results:
            return

        parent_dir = QFileDialog.getExistingDirectory(self, "Pilih folder untuk menyimpan hasil perbandingan")
        if not parent_dir:
            return

        # Semua output (Excel + shapefile + geojson/csv per model) disimpan dalam
        # SATU folder tersendiri per run (bukan file lepas), supaya rapi dan tidak
        # ketimpa/kecampur kalau user export beberapa kali dengan parameter beda.
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(parent_dir, f"Perbandingan_Model_{stamp}")
        try:
            os.makedirs(out_dir, exist_ok=False)
        except FileExistsError:
            QMessageBox.critical(self, "Gagal", "Folder output sudah ada, coba lagi.")
            return

        excel_path = os.path.join(out_dir, "perbandingan_model.xlsx")
        try:
            self.log_box.appendPlainText(f"Menyimpan folder hasil: {out_dir}")
            export_comparison_excel(
                excel_path, self.manual_xy, self.results, self.threshold,
                manual_attrs=getattr(self, "manual_attrs", None),
            )
            self.log_box.appendPlainText(f"  -> Excel: {os.path.basename(excel_path)}")

            point_outputs = export_comparison_points(
                out_dir, self.manual_xy, self.results,
                manual_path=self.manual_path, manual_attrs=getattr(self, "manual_attrs", None),
            )
            for po in point_outputs:
                self.log_box.appendPlainText(f"  -> {po['name']}:")
                for status in ("TP", "FP", "FN"):
                    fileset = po["by_status"][status]
                    if fileset:
                        self.log_box.appendPlainText(f"       {status}: {os.path.basename(fileset['shp'])}")
                    else:
                        self.log_box.appendPlainText(f"       {status}: (0 titik, dilewati)")
                self.log_box.appendPlainText(f"       Gabungan: {os.path.basename(po['combined']['shp'])}")

            QMessageBox.information(
                self, "Berhasil",
                "Hasil perbandingan tersimpan di folder:\n" + out_dir +
                "\n\nIsi folder:\n"
                "\u2022 perbandingan_model.xlsx (ringkasan)\n"
                "\u2022 <model>_TP.shp / _FP.shp / _FN.shp (+ .geojson/.csv) -- titik per status, "
                "siap diberi simbol beda-beda di QGIS\n"
                "\u2022 <model>_hasil.shp (+ .geojson/.csv) -- gabungan semua titik dengan kolom \"status\""
            )
        except Exception:
            QMessageBox.critical(self, "Gagal", "Gagal menyimpan hasil:\n" + traceback.format_exc())
