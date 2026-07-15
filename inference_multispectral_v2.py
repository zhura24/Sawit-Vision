import sys
import json
from pathlib import Path

try:
    import cv2
    import numpy as np
    import rasterio
    from ultralytics import YOLO
except ImportError as e:
    print(f"[ERROR] Modul belum terinstall: {e}")
    print("Jalankan: pip install ultralytics rasterio numpy opencv-python")
    sys.exit(1)


# ============================================================
# KONFIGURASI MODEL -- edit path ini kalau lokasi model/band_stats berubah
# ============================================================
MODEL_PATH = r"D:\multispectral.v2\dataset_final\runs\detect\combined_stage2_channel_dropout\weights\best.pt"
BAND_STATS_PATH = r"D:\multispectral.v2\dataset_final\band_stats_combined.json"
EXPECTED_N_BANDS = 7

# HARUS SAMA PERSIS dengan lower_pct/upper_pct yang dipakai hitung_band_stats()
# waktu generate band_stats.json sensor1 & sensor2 (Tab 1 GUI pipeline_core.py).
STRETCH_LOWER_PCT = 1.0
STRETCH_UPPER_PCT = 99.0


def load_band_stats(stats_path: Path):
    with open(stats_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def is_multiref_schema(band_stats: dict) -> bool:
    """band_stats hasil prepare_combined_dataset.py punya key 'sources' per slot."""
    return any("sources" in v for v in band_stats.values())


def stretch_band(band: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    band = band.astype(np.float32)  # float32 cukup akurat, hemat RAM vs float64
    if p_high - p_low == 0:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, p_low, p_high)
    scaled = (clipped - p_low) / (p_high - p_low) * 255.0
    return scaled.astype(np.uint8)


def generate_tile_windows(width: int, height: int, tile_size: int = 640, overlap: int = 64):
    """
    Hasilkan daftar window (x_off, y_off, w, h) untuk sliding window
    sepanjang raster, dengan overlap antar tile.
    """
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
    """
    Tambahkan padding kanan/bawah bila tile lebih kecil dari target_size.
    """
    height, width = tile_hwc.shape[:2]
    if height >= target_size and width >= target_size:
        return tile_hwc

    pad_h = max(0, target_size - height)
    pad_w = max(0, target_size - width)
    if pad_h == 0 and pad_w == 0:
        return tile_hwc

    return np.pad(
        tile_hwc,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def nms_global(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5):
    """
    NMS sederhana untuk menghapus deteksi duplikat di area overlap antar tile.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_threshold]

    return keep


def auto_detect_band_mapping(src, band_stats: dict) -> dict:
    """
    Deteksi otomatis pasangan band input <-> band training berdasarkan
    kemiripan mean, MENDUKUNG kedua arah:
      - Band input LEBIH SEDIKIT dari training (band training sisa diisi 0)
      - Band input LEBIH BANYAK dari training (band input sisa diabaikan)
    """
    n_bands_input = src.count

    print("[AUTO-DETECT] Menghitung statistik tiap band input untuk mencocokkan band training...")
    input_means = {}
    for b in range(1, n_bands_input + 1):
        # Downsampling 10x untuk hemat RAM ~100x, cukup untuk menghitung mean
        h_new = max(1, src.height // 10)
        w_new = max(1, src.width // 10)
        data = src.read(b, out_shape=(h_new, w_new)).astype(np.float32)
        valid = data[data > 0]
        mean_val = float(valid.mean()) if len(valid) > 0 else 0.0
        input_means[b] = mean_val
        print(f"  Band input {b}: mean = {mean_val:.6f}")

    training_means = {band_idx: stats["mean"] for band_idx, stats in band_stats.items()}
    print(f"[AUTO-DETECT] Mean band training: { {k: round(v,6) for k,v in training_means.items()} }")

    # Greedy global matching
    available_train = set(training_means.keys())
    available_input = set(input_means.keys())
    mapping = {}

    while available_train and available_input:
        best_pair = None
        best_diff = None
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
        print(f"  -> Band training {tb} (mean={training_means[tb]:.6f}) dipetakan dari "
              f"band input {ib} (mean={input_means[ib]:.6f})")

    if available_train:
        print(f"[INFO] Band training {sorted(available_train)} tidak dapat pasangan "
              f"-> akan diisi 0 (hitam).")
    if available_input:
        print(f"[INFO] Band input {sorted(available_input)} tidak dipakai/diabaikan "
              f"(model cuma butuh {len(training_means)} band).")

    return mapping


def auto_detect_band_mapping_multiref(src, band_stats: dict) -> dict:
    """
    Sama seperti auto_detect_band_mapping, tapi tiap slot canonical boleh
    punya LEBIH DARI SATU referensi (satu per sensor sumber).
    """
    n_bands_input = src.count

    print("[AUTO-DETECT] Menghitung statistik tiap band input untuk mencocokkan slot canonical...")
    input_means = {}
    for b in range(1, n_bands_input + 1):
        # Downsampling 10x untuk hemat RAM ~100x, cukup untuk menghitung mean
        h_new = max(1, src.height // 10)
        w_new = max(1, src.width // 10)
        data = src.read(b, out_shape=(h_new, w_new)).astype(np.float32)
        valid = data[data > 0]
        mean_val = float(valid.mean()) if len(valid) > 0 else 0.0
        input_means[b] = mean_val
        print(f"  Band input {b}: mean = {mean_val:.6f}")

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
        print(f"  -> Slot {slot} (referensi paling dekat: {source_name}, mean={stats['mean']:.6f}) "
              f"dipetakan dari band input {ib} (mean={input_means[ib]:.6f}), diff={diff:.6f}{flag}")

    if available_slots:
        print(f"[INFO] Slot canonical {sorted(available_slots)} tidak dapat pasangan "
              f"-> akan diisi 0 (hitam).")
    if available_input:
        print(f"[INFO] Band input {sorted(available_input)} tidak dipakai/diabaikan.")

    return mapping


def draw_detections_on_raster(raster_path: Path, boxes: np.ndarray, scores: np.ndarray,
                                out_path: Path, max_dim: int = 4000):
    """
    Buat gambar visual: composite RGB (band 1-3) dari raster asli + bounding box.
    """
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

        # Baca langsung dalam ukuran kecil agar hemat RAM
        r = src.read(idx_r, out_shape=(h_new, w_new)).astype(np.float32)
        g = src.read(idx_g, out_shape=(h_new, w_new)).astype(np.float32)
        b = src.read(idx_b, out_shape=(h_new, w_new)).astype(np.float32)

    def stretch_for_display(band):
        p_low, p_high = np.percentile(band, (STRETCH_LOWER_PCT, STRETCH_UPPER_PCT))
        if p_high - p_low == 0:
            return np.zeros_like(band, dtype=np.uint8)
        band = np.clip(band, p_low, p_high)
        return ((band - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    rgb = np.stack([stretch_for_display(r), stretch_for_display(g), stretch_for_display(b)], axis=-1)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for box, score in zip(boxes, scores):
        # Skalakan koordinat bounding box agar pas di gambar yang sudah diperkecil
        x1, y1, x2, y2 = (box * scale).astype(int)
        cv2.rectangle(rgb_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{score:.2f}"
        cv2.putText(rgb_bgr, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), rgb_bgr)
    return out_path


def main():
    print("=== Inference Raster Besar Mentah (Sliding Window) - Deteksi Sawit ===\n")

    model_path = Path(MODEL_PATH)
    stats_path = Path(BAND_STATS_PATH)

    raster_path = Path(input("Path ke raster MENTAH UTUH (.tif): ").strip().strip('"'))

    if not model_path.is_file():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)
    if not stats_path.is_file():
        print(f"[ERROR] band_stats.json tidak ditemukan: {stats_path}")
        sys.exit(1)
    if not raster_path.is_file():
        print(f"[ERROR] Raster tidak ditemukan: {raster_path}")
        sys.exit(1)

    tile_size = 640
    overlap = 64

    conf_raw = input("Confidence threshold (Enter untuk default 0.25): ").strip()
    conf = float(conf_raw) if conf_raw else 0.25
    print("\nMemuat model...")
    model = YOLO(str(model_path))
    band_stats = load_band_stats(stats_path)

    print(f"\nMembuka raster: {raster_path.name}")
    with rasterio.open(raster_path) as src:
        width, height, n_bands = src.width, src.height, src.count
        print(f"Ukuran raster: {width} x {height} piksel, {n_bands} band")

        raster_dtype = str(src.dtypes[0])
        is_uint8_input = raster_dtype == "uint8"
        if is_uint8_input:
            print("Tipe data: uint8 -- SUDAH 8-bit, stretch/normalisasi kontras di-SKIP, "
                  "nilai piksel dipakai apa adanya.")
        else:
            print(f"Tipe data: {raster_dtype} -- akan di-stretch ke uint8 per-tile "
                  f"(percentile {STRETCH_LOWER_PCT}-{STRETCH_UPPER_PCT}).")

        expected_n_bands = len(band_stats)
        multiref = is_multiref_schema(band_stats)

        if n_bands == expected_n_bands and not multiref:
            band_mapping = {b: b for b in range(1, expected_n_bands + 1)}
            print(f"Band lengkap ({n_bands} band). Mapping langsung 1-to-1.")
        elif multiref:
            print(f"\n[INFO] Model gabungan (multi-referensi) terdeteksi. "
                  f"Mencocokkan {n_bands} band input ke {expected_n_bands} slot canonical...\n")
            band_mapping = auto_detect_band_mapping_multiref(src, band_stats)
            print(f"\n[AUTO-DETECT] Hasil pemetaan final: {band_mapping}")
            print("Slot yang tidak dapat pasangan akan diisi 0 (hitam).\n")
        else:
            print(f"\n[WARNING] Jumlah band raster ({n_bands}) BEDA dari model terpilih ({expected_n_bands}).")
            print("Mode adaptif otomatis diaktifkan. Mencocokkan band input ke band training...\n")
            band_mapping = auto_detect_band_mapping(src, band_stats)
            print(f"\n[AUTO-DETECT] Hasil pemetaan final (band training -> band input): {band_mapping}")
            print("Band training yang tidak dapat pasangan akan diisi 0 (hitam).\n")

        windows = generate_tile_windows(width, height, tile_size, overlap)
        print(f"Akan diproses sebagai {len(windows)} tile ({tile_size}x{tile_size}, overlap {overlap}px)\n")

        all_boxes = []
        all_scores = []
        all_classes = []

        for idx, (x_off, y_off, w, h) in enumerate(windows, start=1):
            print(f"[{idx}/{len(windows)}] Tile di posisi ({x_off}, {y_off}), ukuran {w}x{h} ...", end=" ")

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
                    stats = band_stats.get(target_b, {})
                    fallback_p_low = stats.get("p_low")
                    fallback_p_high = stats.get("p_high")

                data = src.read(input_b_idx, window=window)

                if is_uint8_input:
                    stretched = data.astype(np.uint8)
                else:
                    valid_pixels = data[data > 0]
                    if len(valid_pixels) > (w * h * 0.05):
                        p_low, p_high = np.percentile(valid_pixels, (STRETCH_LOWER_PCT, STRETCH_UPPER_PCT))
                    elif fallback_p_low is not None and fallback_p_high is not None:
                        p_low, p_high = fallback_p_low, fallback_p_high
                    else:
                        p_low, p_high = 0, 255
                    stretched = stretch_band(data, p_low, p_high)

                tile_chw[target_b - 1] = stretched

            tile_hwc = tile_chw.transpose(1, 2, 0)
            tile_hwc = pad_tile_for_inference(tile_hwc, target_size=640)

            results = model.predict(source=tile_hwc, conf=conf, save=False, verbose=False)

            n_det = 0
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()

                    boxes_xyxy[:, [0, 2]] += x_off
                    boxes_xyxy[:, [1, 3]] += y_off

                    all_boxes.append(boxes_xyxy)
                    all_scores.append(scores)
                    all_classes.append(classes)
                    n_det = len(scores)

            print(f"{n_det} objek")

    if not all_boxes:
        print("\n[INFO] Tidak ada objek terdeteksi di seluruh raster.")
        return

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    print(f"\nTotal deteksi sebelum NMS antar-tile: {len(all_boxes)}")

    keep_idx = nms_global(all_boxes, all_scores, iou_threshold=0.5)
    final_boxes = all_boxes[keep_idx]
    final_scores = all_scores[keep_idx]
    final_classes = all_classes[keep_idx]

    print(f"Total deteksi setelah NMS antar-tile: {len(final_boxes)}")
    print(f"Confidence rata-rata: {final_scores.mean():.3f}, "
          f"min: {final_scores.min():.3f}, max: {final_scores.max():.3f}")

    print("\nMenyimpan hasil sebagai Shapefile (.shp)...")
    try:
        import shapefile
    except ImportError:
        print("[ERROR] Library pyshp belum terinstall.")
        print("Jalankan: pip install pyshp")
        sys.exit(1)

    with rasterio.open(raster_path) as src:
        raster_transform = src.transform
        crs_wkt = src.crs.to_wkt() if src.crs else None

    # ---------------------------------------------------------------
    # Nama output menyertakan nama model agar tidak tertimpa saat
    # menjalankan model yang berbeda pada raster yang sama.
    # Contoh: deteksi_kebun__best.shp, deteksi_kebun__stage2.shp
    # ---------------------------------------------------------------
    model_stem = model_path.stem
    out_stem = f"deteksi_{raster_path.stem}__{model_stem}"
    out_shp = Path(f"{out_stem}.shp")

    with shapefile.Writer(str(out_shp), shapeType=shapefile.POLYGON) as shp:
        shp.field("id",         "N", size=10)
        shp.field("kelas",      "C", size=20)
        shp.field("confidence", "N", size=10, decimal=4)
        shp.field("model",      "C", size=30)
        shp.field("x1_px",     "N", size=10, decimal=1)
        shp.field("y1_px",     "N", size=10, decimal=1)
        shp.field("x2_px",     "N", size=10, decimal=1)
        shp.field("y2_px",     "N", size=10, decimal=1)

        for i, (cls, score, box) in enumerate(
                zip(final_classes, final_scores, final_boxes), start=1):

            x1_px, y1_px, x2_px, y2_px = box

            x1_geo, y1_geo = rasterio.transform.xy(raster_transform, y1_px, x1_px)
            x2_geo, y2_geo = rasterio.transform.xy(raster_transform, y2_px, x2_px)

            polygon = [
                [x1_geo, y1_geo],
                [x2_geo, y1_geo],
                [x2_geo, y2_geo],
                [x1_geo, y2_geo],
                [x1_geo, y1_geo],
            ]

            shp.poly([polygon])
            shp.record(
                i,
                "sawit",
                round(float(score), 4),
                model_stem,          # nama model disimpan juga dalam atribut Shapefile
                round(float(x1_px), 1),
                round(float(y1_px), 1),
                round(float(x2_px), 1),
                round(float(y2_px), 1),
            )

    if crs_wkt:
        prj_path = out_shp.with_suffix(".prj")
        with open(prj_path, "w") as prj:
            prj.write(crs_wkt)

    print(f"Shapefile disimpan di: {out_shp.resolve()}")
    print(f"Total {len(final_boxes)} pohon sawit terdeteksi di seluruh raster.")
    print("\nCara buka di QGIS:")
    print("  Layer → Add Layer → Add Vector Layer → pilih file .shp")
    print("  Layer akan otomatis ter-overlay di posisi yang benar di peta.")

    print("\nMembuat visualisasi (gambar dengan bounding box)...")
    out_img = Path(f"{out_stem}.jpg")
    draw_detections_on_raster(raster_path, final_boxes, final_scores, out_img)
    print(f"Gambar hasil visual disimpan di: {out_img.resolve()}")


if __name__ == "__main__":
    main()
