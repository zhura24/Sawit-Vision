"""
inference_core.py
Engine deteksi sawit dari raster multispektral, direfactor dari
inference_multispectral_v2.py menjadi class yang bisa dipanggil dari GUI
(dengan callback log/progress dan dukungan cancel), tanpa mengubah logic asli.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
import rasterio
import rasterio.windows
import rasterio.transform
import torch
from ultralytics import YOLO


# ============================================================
# HASIL
# ============================================================
@dataclass
class InferenceResult:
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    classes: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    shp_path: Path = None
    preview_path: Path = None
    preview_bgr: np.ndarray = None  # composite RGB + kotak, siap ditampilkan di canvas


class CancelledError(Exception):
    pass


# ============================================================
# FUNGSI MURNI (identik dengan v2 script, dipisah biar mudah ditest)
# ============================================================
def load_band_stats(stats_path: Path) -> dict:
    with open(stats_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def is_multiref_schema(band_stats: dict) -> bool:
    return any("sources" in v for v in band_stats.values())


def stretch_band(band: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    band = band.astype(np.float32)  # float32 cukup (raster & hasil akhir uint8), hemat RAM 2x vs float64
    if p_high - p_low == 0:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, p_low, p_high)
    scaled = (clipped - p_low) / (p_high - p_low) * 255.0
    return scaled.astype(np.uint8)


def generate_tile_windows(width, height, tile_size=640, overlap=64):
    stride = tile_size - overlap
    windows = []
    y = 0
    while y < height:
        x = 0
        h = min(tile_size, height - y)
        while x < width:
            w = min(tile_size, width - x)
            windows.append((x, y, w, h))
            if x + tile_size >= width:
                break
            x += stride
        if y + tile_size >= height:
            break
        y += stride
    return windows


def pad_tile_for_inference(tile_hwc: np.ndarray, target_size: int = 640) -> np.ndarray:
    height, width = tile_hwc.shape[:2]
    if height >= target_size and width >= target_size:
        return tile_hwc
    pad_h = max(0, target_size - height)
    pad_w = max(0, target_size - width)
    if pad_h == 0 and pad_w == 0:
        return tile_hwc
    return np.pad(tile_hwc, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)


def nms_global(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5,
               centroid_dist_threshold: float = None,
               centroid_dist_factor: float = None,
               max_radius_px: float = None,
               tile_ids: np.ndarray = None):
    """
    NMS untuk menghapus deteksi duplikat di area overlap antar tile.

    Root cause duplikat di tepi tile: satu pohon yang sama bisa terdeteksi dua kali
    dari dua tile yang overlap, dengan bounding box yang SEDIKIT bergeser (bukan
    persis sama posisinya) -- misalnya kanopi sedikit terpotong di salah satu tile --
    sehingga IoU-nya kadang jatuh DI BAWAH iou_threshold biasa meski itu pohon yang sama.

    Untuk menutup celah itu, ditambahkan fallback: box dianggap duplikat juga kalau
    jarak centroid-nya cukup dekat, meskipun IoU rendah. Ada dua mode fallback:

    1. ADAPTIF (centroid_dist_factor, DIREKOMENDASIKAN): radius merge dihitung
       PER-PASANGAN kandidat, dari rata-rata diagonal box pasangan itu sendiri
       (radius = centroid_dist_factor * rata-rata diagonal kedua box). Ini penting
       kalau dalam satu raster ada campuran kanopi kecil & besar -- radius jadi
       otomatis kecil untuk pasangan pohon kecil dan besar untuk pasangan pohon
       besar, tidak memakai satu angka radius global yang sama untuk semua ukuran.

    2. GLOBAL (centroid_dist_threshold, LEGACY): satu radius piksel tetap untuk
       semua pasangan, dihitung sebelumnya dari median diagonal SELURUH box di
       gambar (lihat estimate_centroid_dist_threshold). Dipertahankan untuk
       backward-compat.

    max_radius_px (PENYEIMBANG untuk mode adaptif): batas ATAS pair_threshold,
       berapa pun hasil "centroid_dist_factor * rata-rata diagonal" pasangan itu.
       Alasannya: radius adaptif di atas ikut membesar linear terhadap ukuran box,
       padahal JARAK TANAM ASLI di lapangan (jarak antar batang pohon) TIDAK ikut
       membesar cuma karena kanopinya sudah dewasa/lebar -- jarak tanam ditentukan
       saat penanaman, bukan oleh ukuran kanopi saat ini. Akibatnya, kalau raster
       berisi campuran kanopi kecil & besar dan dipakai satu factor yang sama,
       pasangan kanopi BESAR bisa dapat radius merge yang lebih besar dari jarak
       tanam sesungguhnya -> dua pohon dewasa yang benar-benar berbeda individu
       (tapi berdekatan sesuai jarak tanam normal) berisiko salah dianggap
       duplikat dan ke-gabung jadi satu (turunin recall). Nilainya dihitung
       otomatis oleh pemanggil (InferenceEngine.run) dari persentil-90 diagonal
       SELURUH box hasil deteksi di raster ini -- tidak perlu diisi manual.
       Radius pasangan kanopi kecil tetap proporsional kecil (di bawah cap,
       tidak kepotong), sementara radius pasangan kanopi besar dibatasi supaya
       tidak pernah jauh melebihi ukuran kanopi "wajar" di raster tsb.
       None = tidak ada batas (perilaku lama, radius adaptif murni).

    tile_ids: array (sepanjang boxes) berisi indeks tile asal tiap box. Kalau
        diisi, fallback centroid-distance HANYA diterapkan ke pasangan box yang
        berasal dari tile BERBEDA. Ini penting: duplikat akibat overlap tile
        cuma mungkin terjadi antar box dari tile berbeda -- dua pohon ASLI yang
        kebetulan berdekatan tapi terdeteksi dari tile YANG SAMA bukan kasus
        duplikat tile-boundary, dan tidak boleh ikut ke-merge hanya karena
        jaraknya kebetulan di bawah radius. Kalau tile_ids=None, fallback
        diterapkan ke semua pasangan seperti perilaku lama (kurang presisi,
        berisiko menghapus pohon berdekatan yang sebenarnya beda individu).

    Kalau centroid_dist_factor dan centroid_dist_threshold dua-duanya None,
    perilaku identik dengan NMS lama (murni IoU). Kalau dua-duanya diisi,
    centroid_dist_factor (adaptif) yang dipakai.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    diag = None
    if centroid_dist_factor is not None:
        box_w = x2 - x1
        box_h = y2 - y1
        diag = np.sqrt(box_w ** 2 + box_h ** 2)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter)

        is_duplicate = iou > iou_threshold

        if diag is not None or centroid_dist_threshold is not None:
            dist = np.sqrt((cx[i] - cx[rest]) ** 2 + (cy[i] - cy[rest]) ** 2)

            if diag is not None:
                # Radius per-pasangan: rata-rata diagonal box i & box pasangannya,
                # jadi kanopi kecil dan besar masing-masing dapat radius yang
                # proporsional dengan ukurannya sendiri, bukan radius global.
                pair_threshold = centroid_dist_factor * (diag[i] + diag[rest]) / 2.0
            else:
                pair_threshold = centroid_dist_threshold

            if max_radius_px is not None:
                # Cap supaya radius pasangan kanopi BESAR tidak pernah melebihi
                # jarak tanam minimum riil -- lihat penjelasan max_radius_px di
                # docstring. Pasangan kanopi kecil tidak terpengaruh (nilainya
                # sudah di bawah cap ini).
                pair_threshold = np.minimum(pair_threshold, max_radius_px)

            centroid_match = dist <= pair_threshold

            if tile_ids is not None:
                # Fallback jarak cuma valid untuk pasangan LINTAS TILE -- duplikat
                # tile-boundary memang cuma bisa terjadi antar tile berbeda.
                cross_tile = tile_ids[i] != tile_ids[rest]
                centroid_match = centroid_match & cross_tile

            is_duplicate = is_duplicate | centroid_match

        order = rest[~is_duplicate]
    return keep


def estimate_centroid_dist_threshold(boxes: np.ndarray, factor: float = 0.5,
                                      min_detections: int = 20,
                                      fallback: float = 15.0, log=print) -> float:
    """
    Estimasi otomatis centroid_dist_threshold dari ukuran box hasil deteksi itu
    sendiri (sebelum NMS) -- bukan angka piksel hardcoded.

    Dimensi box yang dihasilkan model adalah proxy langsung untuk ukuran kanopi
    sawit DI RESOLUSI RASTER INI, jadi threshold otomatis menyesuaikan kalau raster
    beda GSD (resolusi piksel), tanpa perlu tuning ulang manual tiap raster.

    Pakai MEDIAN (bukan mean) supaya tidak gampang terpengaruh outlier (mis. box
    salah deteksi yang ukurannya tidak wajar).

    factor: pengali terhadap diagonal box (0.5 = radius kanopi, dari diameter).
    min_detections: kalau jumlah deteksi kurang dari ini, statistik dianggap belum
        cukup stabil untuk dipercaya -> pakai fallback.
    """
    if len(boxes) < min_detections:
        log(f"[NMS] Deteksi terlalu sedikit ({len(boxes)}) untuk auto-estimate, "
            f"pakai fallback: {fallback:.1f}px")
        return fallback

    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    diag = np.sqrt(box_w ** 2 + box_h ** 2)
    median_diag = float(np.median(diag))
    threshold = median_diag * factor

    log(f"[NMS] Auto-estimate centroid_dist_threshold dari {len(boxes)} box: "
        f"median diagonal={median_diag:.2f}px, faktor={factor} -> threshold={threshold:.2f}px")
    return threshold


def auto_detect_band_mapping(src, band_stats: dict, log=print) -> dict:
    n_bands_input = src.count
    input_means = {}
    for b in range(1, n_bands_input + 1):
        # Downsample read to avoid OOM on large images (e.g. 15423 x 20056)
        h_new = max(1, src.height // 10)
        w_new = max(1, src.width // 10)
        data = src.read(b, out_shape=(h_new, w_new)).astype(np.float32)
        valid = data[data > 0]
        input_means[b] = float(valid.mean()) if len(valid) > 0 else 0.0

    training_means = {band_idx: stats["mean"] for band_idx, stats in band_stats.items()}

    available_train = set(training_means.keys())
    available_input = set(input_means.keys())
    mapping = {}

    while available_train and available_input:
        best_pair, best_diff = None, None
        for tb in available_train:
            for ib in available_input:
                diff = abs(training_means[tb] - input_means[ib])
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_pair = (tb, ib)
        tb, ib = best_pair
        mapping[tb] = ib
        available_train.remove(tb)
        available_input.remove(ib)
        log(f"  -> Band training {tb} <- band input {ib} (diff={best_diff:.6f})")

    return mapping


def auto_detect_band_mapping_multiref(src, band_stats: dict, log=print) -> dict:
    n_bands_input = src.count
    input_means = {}
    for b in range(1, n_bands_input + 1):
        # Downsample read to avoid OOM on large images (e.g. 15423 x 20056)
        h_new = max(1, src.height // 10)
        w_new = max(1, src.width // 10)
        data = src.read(b, out_shape=(h_new, w_new)).astype(np.float32)
        valid = data[data > 0]
        input_means[b] = float(valid.mean()) if len(valid) > 0 else 0.0

    candidates = []
    for slot, entry in band_stats.items():
        for source_name, stats in entry["sources"].items():
            candidates.append((slot, source_name, stats["mean"], stats))

    available_slots = set(band_stats.keys())
    available_input = set(input_means.keys())
    mapping = {}

    while available_slots and available_input:
        best = None
        for slot, source_name, mean_val, stats in candidates:
            if slot not in available_slots:
                continue
            for ib in available_input:
                diff = abs(mean_val - input_means[ib])
                if best is None or diff < best[0]:
                    best = (diff, slot, ib, source_name, stats)
        diff, slot, ib, source_name, stats = best
        mapping[slot] = {
            "input_band": ib,
            "source": source_name,
            "p_low": stats.get("p_low"),
            "p_high": stats.get("p_high"),
        }
        available_slots.remove(slot)
        available_input.remove(ib)
        flag = "  <-- selisih besar, VERIFIKASI MANUAL" if diff > 0.05 else ""
        log(f"  -> Slot {slot} <- band input {ib} (sumber: {source_name}, diff={diff:.6f}){flag}")

    return mapping


def build_preview_bgr(raster_path: Path, boxes: np.ndarray, scores: np.ndarray,
                       stretch_lower_pct: float, stretch_upper_pct: float,
                       max_dim: int = 2000) -> np.ndarray:
    """Composite RGB (band 1-3) dari raster asli + kotak deteksi. Return array BGR (bukan simpan file)."""
    with rasterio.open(raster_path) as src:
        h_orig, w_orig = src.height, src.width
        scale = 1.0
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
        h_new = int(h_orig * scale)
        w_new = int(w_orig * scale)

        n_bands = src.count
        idx_r = min(3, n_bands)
        idx_g = min(2, n_bands)
        idx_b = min(1, n_bands)
        
        # Read downsampled version to avoid huge memory footprint
        r = src.read(idx_r, out_shape=(h_new, w_new)).astype(np.float32)
        g = src.read(idx_g, out_shape=(h_new, w_new)).astype(np.float32)
        b = src.read(idx_b, out_shape=(h_new, w_new)).astype(np.float32)

    def stretch_for_display(band):
        p_low, p_high = np.percentile(band, (stretch_lower_pct, stretch_upper_pct))
        if p_high - p_low == 0:
            return np.zeros_like(band, dtype=np.uint8)
        band = np.clip(band, p_low, p_high)
        return ((band - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    rgb = np.stack([stretch_for_display(r), stretch_for_display(g), stretch_for_display(b)], axis=-1)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for box, score in zip(boxes, scores):
        # Scale bounding box coordinates to match downsampled preview image
        x1, y1, x2, y2 = (box * scale).astype(int)
        cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_bgr, f"{score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return rgb_bgr


def save_shapefile(raster_path: Path, boxes, scores, classes, out_shp: Path, model_name: str = "model_gabungan"):
    import shapefile  # pyshp

    with rasterio.open(raster_path) as src:
        raster_transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs else None

    with shapefile.Writer(str(out_shp), shapeType=shapefile.POLYGON) as shp:
        shp.field("id", "N", size=10)
        shp.field("kelas", "C", size=20)
        shp.field("confidence", "N", size=10, decimal=4)
        shp.field("model", "C", size=30)
        shp.field("x1_px", "N", size=10, decimal=1)
        shp.field("y1_px", "N", size=10, decimal=1)
        shp.field("x2_px", "N", size=10, decimal=1)
        shp.field("y2_px", "N", size=10, decimal=1)

        for i, (cls, score, box) in enumerate(zip(classes, scores, boxes), start=1):
            x1_px, y1_px, x2_px, y2_px = box
            x1_geo, y1_geo = rasterio.transform.xy(raster_transform, y1_px, x1_px)
            x2_geo, y2_geo = rasterio.transform.xy(raster_transform, y2_px, x2_px)
            polygon = [[x1_geo, y1_geo], [x2_geo, y1_geo], [x2_geo, y2_geo], [x1_geo, y2_geo], [x1_geo, y1_geo]]
            shp.poly([polygon])
            shp.record(i, "sawit", round(float(score), 4), model_name,
                       round(float(x1_px), 1), round(float(y1_px), 1),
                       round(float(x2_px), 1), round(float(y2_px), 1))

    if crs_wkt:
        with open(out_shp.with_suffix(".prj"), "w") as prj:
            prj.write(crs_wkt)


def load_detection_from_shapefile(shp_path: Path):
    """Baca hasil deteksi lama dari shapefile yang dihasilkan aplikasi."""
    import shapefile

    shp_path = Path(shp_path)
    if not shp_path.is_file():
        raise FileNotFoundError(f"Shapefile tidak ditemukan: {shp_path}")

    with shapefile.Reader(str(shp_path)) as shp:
        fields = [f[0] for f in shp.fields[1:]]
        boxes = []
        scores = []
        classes = []

        for record in shp.iterRecords():
            values = dict(zip(fields, record))
            x1 = float(values.get("x1_px", 0.0))
            y1 = float(values.get("y1_px", 0.0))
            x2 = float(values.get("x2_px", 0.0))
            y2 = float(values.get("y2_px", 0.0))
            boxes.append([x1, y1, x2, y2])
            scores.append(float(values.get("confidence", 0.0)))
            classes.append(0)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(classes, dtype=np.float32),
    )


# ============================================================
# ENGINE
# ============================================================
class InferenceEngine:
    """
    Bungkus semua logic v2 script jadi satu class.
    log_fn(str) dan progress_fn(current, total) dipanggil dari worker thread GUI.
    should_cancel() dipanggil berkala; kalau True, proses dihentikan lewat CancelledError.
    """

    STRETCH_LOWER_PCT = 1.0
    STRETCH_UPPER_PCT = 99.0

    def __init__(self, model_path: str, band_stats_path: str,
                 log_fn=print, progress_fn=None, should_cancel=None):
        self.model_path = Path(model_path)
        self.band_stats_path = Path(band_stats_path)
        self.log_fn = log_fn
        self.progress_fn = progress_fn or (lambda cur, total: None)
        self.should_cancel = should_cancel or (lambda: False)

        self.model = None
        self.band_stats = None
        self.device = None

    def load(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model tidak ditemukan: {self.model_path}")
        if not self.band_stats_path.is_file():
            raise FileNotFoundError(f"band_stats tidak ditemukan: {self.band_stats_path}")

        # Deteksi & pakai GPU secara EKSPLISIT -- kalau tidak diset manual,
        # ultralytics kadang jatuh ke CPU tanpa terlihat jelas di log.
        if torch.cuda.is_available():
            self.device = 0
            gpu_name = torch.cuda.get_device_name(0)
            self.log_fn(f"GPU terdeteksi: {gpu_name}. Inference akan pakai GPU.")
        else:
            self.device = "cpu"
            self.log_fn("[PERINGATAN] GPU/CUDA TIDAK terdeteksi oleh torch. "
                         "Inference akan jalan di CPU dan JAUH lebih lambat. "
                         "Cek instalasi torch+CUDA kamu.")

        self.log_fn(f"Memuat model: {self.model_path.name} ...")
        self.model = YOLO(str(self.model_path))
        self.band_stats = load_band_stats(self.band_stats_path)
        self.log_fn(f"Model & band stats siap ({len(self.band_stats)} slot).")

    def run(self, raster_path: str, conf: float = 0.25, tile_size: int = 640,
            overlap: int = 64, iou_threshold: float = 0.5,
            centroid_dist_factor: float = 0.5,
            output_dir: str = None, batch_size: int = 8, out_name: str = None) -> InferenceResult:
        """
        centroid_dist_factor: faktor pengali terhadap diagonal box untuk fallback
            centroid-distance di NMS (menutup duplikat tepi tile yang lolos dari
            IoU biasa). Set None untuk menonaktifkan (NMS murni IoU, perilaku lama).

        Radius merge dihitung ADAPTIF per-pasangan (dari diagonal box pasangan
        itu sendiri, lihat nms_global), lalu otomatis DIBATASI ke persentil-90
        diagonal seluruh box di raster ini -- supaya pasangan kanopi yang jauh
        lebih besar dari kanopi "wajar" di raster tsb tidak dapat radius merge
        yang kebablasan (bisa salah menggabung dua pohon dewasa berbeda yang
        cuma kebetulan berdekatan sesuai jarak tanam normal). Ini dihitung
        otomatis dari statistik deteksi itu sendiri -- tidak perlu input
        tambahan dari pengguna, dan otomatis menyesuaikan resolusi/skala
        raster apa pun.
        """
        if self.model is None:
            self.load()

        raster_path = Path(raster_path)
        if not raster_path.is_file():
            raise FileNotFoundError(f"Raster tidak ditemukan: {raster_path}")

        out_dir_base = Path(output_dir) if output_dir else raster_path.parent
        out_dir_base.mkdir(parents=True, exist_ok=True)

        self.log_fn(f"Membuka raster: {raster_path.name}")
        with rasterio.open(raster_path) as src:
            width, height, n_bands = src.width, src.height, src.count
            self.log_fn(f"Ukuran raster: {width} x {height} px, {n_bands} band")

            raster_dtype = str(src.dtypes[0])
            is_uint8_input = raster_dtype == "uint8"

            expected_n_bands = len(self.band_stats)
            multiref = is_multiref_schema(self.band_stats)

            if n_bands == expected_n_bands and not multiref:
                band_mapping = {b: b for b in range(1, expected_n_bands + 1)}
                self.log_fn(f"Band lengkap ({n_bands}). Mapping 1-to-1.")
            elif multiref:
                self.log_fn(f"Model gabungan terdeteksi. Mencocokkan {n_bands} band input -> {expected_n_bands} slot...")
                band_mapping = auto_detect_band_mapping_multiref(src, self.band_stats, log=self.log_fn)
            else:
                self.log_fn(f"Jumlah band beda ({n_bands} vs {expected_n_bands}). Mode adaptif...")
                band_mapping = auto_detect_band_mapping(src, self.band_stats, log=self.log_fn)

            windows = generate_tile_windows(width, height, tile_size, overlap)
            total = len(windows)
            self.log_fn(f"Akan diproses {total} tile ({tile_size}x{tile_size}, overlap {overlap}px)")

            all_boxes, all_scores, all_classes, all_tile_ids = [], [], [], []

            def _prepare_tile(x_off, y_off, w, h):
                """Baca & stretch satu tile dari raster (I/O + CPU, TIDAK menyentuh GPU)."""
                tile_chw = np.zeros((expected_n_bands, h, w), dtype=np.uint8)
                window = rasterio.windows.Window(x_off, y_off, w, h)

                for target_b in range(1, expected_n_bands + 1):
                    entry = band_mapping.get(target_b)
                    if entry is None:
                        continue

                    if isinstance(entry, dict):
                        input_b_idx = entry["input_band"]
                        fallback_p_low = entry.get("p_low")
                        fallback_p_high = entry.get("p_high")
                    else:
                        input_b_idx = entry
                        stats = self.band_stats.get(target_b, {})
                        fallback_p_low = stats.get("p_low")
                        fallback_p_high = stats.get("p_high")

                    data = src.read(input_b_idx, window=window)

                    if is_uint8_input:
                        stretched = data.astype(np.uint8)
                    else:
                        valid_pixels = data[data > 0]
                        if len(valid_pixels) > (w * h * 0.05):
                            p_low, p_high = np.percentile(valid_pixels, (self.STRETCH_LOWER_PCT, self.STRETCH_UPPER_PCT))
                        elif fallback_p_low is not None and fallback_p_high is not None:
                            p_low, p_high = fallback_p_low, fallback_p_high
                        else:
                            p_low, p_high = 0, 255
                        stretched = stretch_band(data, p_low, p_high)

                    tile_chw[target_b - 1] = stretched

                tile_hwc = tile_chw.transpose(1, 2, 0)
                return pad_tile_for_inference(tile_hwc, target_size=tile_size)

            def _flush_batch(tile_batch, offset_batch):
                """Kirim satu batch tile sekaligus ke model -- GPU jauh lebih efisien
                diberi banyak gambar sekaligus daripada satu-satu."""
                results = self.model.predict(source=tile_batch, device=self.device,
                                              conf=conf, save=False, verbose=False)
                n_det_total = 0
                for r, (x_off, y_off, tile_idx) in zip(results, offset_batch):
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy()
                        boxes_xyxy[:, [0, 2]] += x_off
                        boxes_xyxy[:, [1, 3]] += y_off
                        all_boxes.append(boxes_xyxy)
                        all_scores.append(scores)
                        all_classes.append(classes)
                        all_tile_ids.append(np.full(len(scores), tile_idx, dtype=np.int32))
                        n_det_total += len(scores)
                return n_det_total

            tile_batch, offset_batch = [], []

            for idx, (x_off, y_off, w, h) in enumerate(windows, start=1):
                if self.should_cancel():
                    raise CancelledError("Dibatalkan oleh pengguna.")

                tile_batch.append(_prepare_tile(x_off, y_off, w, h))
                offset_batch.append((x_off, y_off, idx))

                is_last = (idx == total)
                if len(tile_batch) >= batch_size or is_last:
                    n_det = _flush_batch(tile_batch, offset_batch)
                    self.log_fn(f"[{idx}/{total}] batch selesai ({len(tile_batch)} tile), "
                                f"{n_det} objek di batch ini")
                    tile_batch, offset_batch = [], []

                self.progress_fn(idx, total)

        result = InferenceResult()
        if not all_boxes:
            self.log_fn("Tidak ada objek terdeteksi di seluruh raster.")
            return result

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)
        all_tile_ids = np.concatenate(all_tile_ids, axis=0)

        self.log_fn(f"Total deteksi sebelum NMS: {len(all_boxes)}")

        max_radius_px = None
        if centroid_dist_factor is not None:
            _box_w = all_boxes[:, 2] - all_boxes[:, 0]
            _box_h = all_boxes[:, 3] - all_boxes[:, 1]
            _diag = np.sqrt(_box_w ** 2 + _box_h ** 2)
            _radius = centroid_dist_factor * _diag
            self.log_fn(
                f"[NMS] Radius gabung duplikat: mode adaptif per-pasangan "
                f"(faktor={centroid_dist_factor}), HANYA berlaku untuk pasangan "
                f"box lintas-tile. Estimasi rentang radius dari {len(all_boxes)} "
                f"box -- min={_radius.min():.1f}px, "
                f"median={float(np.median(_radius)):.1f}px, max={_radius.max():.1f}px."
            )

            if len(_diag) >= 20:
                # Batas otomatis: persentil-90 diagonal SELURUH box di raster ini
                # dipakai sebagai acuan "kanopi wajar terbesar". Radius pasangan
                # yang lebih besar dari itu (mis. dua box raksasa/outlier) dipangkas
                # ke batas ini, supaya tidak salah menggabung dua pohon dewasa
                # berbeda yang cuma kebetulan berdekatan sesuai jarak tanam normal.
                # Dihitung otomatis dari statistik deteksi itu sendiri -- tidak
                # butuh input tambahan dari pengguna.
                ref_diag = float(np.percentile(_diag, 90))
                max_radius_px = centroid_dist_factor * ref_diag
                self.log_fn(
                    f"[NMS] Batas radius otomatis: persentil-90 diagonal "
                    f"={ref_diag:.1f}px -> radius pasangan dipangkas maks "
                    f"{max_radius_px:.1f}px."
                )
            else:
                self.log_fn("[NMS] Deteksi terlalu sedikit untuk batas radius "
                             "otomatis, dilewati (radius adaptif tidak dibatasi).")

        keep_idx = nms_global(all_boxes, all_scores, iou_threshold=iou_threshold,
                               centroid_dist_factor=centroid_dist_factor,
                               max_radius_px=max_radius_px,
                               tile_ids=all_tile_ids)
        final_boxes = all_boxes[keep_idx]
        final_scores = all_scores[keep_idx]
        final_classes = all_classes[keep_idx]
        self.log_fn(f"Total deteksi setelah NMS: {len(final_boxes)}")

        # Nama output menyertakan nama model agar tidak tertimpa jika memakai model berbeda
        model_stem = self.model_path.stem
        if out_name and out_name.strip():
            out_stem = out_name.strip()
            for ext in (".shp", ".shx", ".dbf", ".prj"):
                if out_stem.lower().endswith(ext):
                    out_stem = out_stem[: -len(ext)]
        else:
            out_stem = f"deteksi_{raster_path.stem}__{model_stem}"

        # Setiap run disimpan dalam FOLDER TERSENDIRI (bukan file lepas di
        # folder output utama) -- shapefile, jpg preview, dan centroid
        # geojson/csv (yang otomatis dieksport ke folder shapefile ini juga)
        # jadi berkumpul rapi per run, tidak bercampur antar run/model lain.
        out_dir = out_dir_base / out_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        out_shp = out_dir / f"{out_stem}.shp"
        self.log_fn("Menyimpan shapefile...")
        save_shapefile(raster_path, final_boxes, final_scores, final_classes, out_shp,
                       model_name=model_stem)
        self.log_fn(f"Shapefile: {out_shp}")

        self.log_fn("Membuat preview visual...")
        preview_bgr = build_preview_bgr(raster_path, final_boxes, final_scores,
                                         self.STRETCH_LOWER_PCT, self.STRETCH_UPPER_PCT)
        out_img = out_dir / f"{out_stem}.jpg"
        cv2.imwrite(str(out_img), preview_bgr)

        result.boxes = final_boxes
        result.scores = final_scores
        result.classes = final_classes
        result.shp_path = out_shp
        result.preview_path = out_img
        result.preview_bgr = preview_bgr
        return result
