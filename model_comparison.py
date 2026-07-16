"""
model_comparison.py
Modul evaluasi model AI untuk Sawit Vision.

Membandingkan centroid hasil inference (satu atau beberapa model) terhadap
centroid manual (ground truth) memakai greedy nearest-neighbor matching
(mirip evaluasi objek deteksi: 1 titik manual hanya boleh dipasangkan
dengan 1 titik inference terdekat dalam radius threshold).

Tidak bergantung pada QGIS. Hanya butuh: pyshp, scipy, numpy, openpyxl
(semua ringan, tidak ada dependensi GDAL/geopandas).
"""
import numpy as np
import shapefile  # pyshp
from scipy.spatial import cKDTree
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


def read_points_shp(path):
    """
    Baca semua titik (Point) dari shapefile.
    Return: (xy, attrs)
      xy    -> np.ndarray shape (N, 2)
      attrs -> list of dict atribut per titik (boleh kosong/tidak dipakai)
    """
    sf = shapefile.Reader(path)
    field_names = [f[0] for f in sf.fields[1:]]  # field[0] = DeletionFlag, skip
    pts = []
    attrs = []
    for sr in sf.iterShapeRecords():
        geom = sr.shape
        if not geom.points:
            continue
        x, y = geom.points[0]
        pts.append((x, y))
        rec = list(sr.record)
        attrs.append(dict(zip(field_names, rec)))
    if not pts:
        return np.empty((0, 2), dtype=float), []
    return np.array(pts, dtype=float), attrs


def match_greedy(manual_xy: np.ndarray, infer_xy: np.ndarray, threshold: float):
    """
    Greedy nearest-neighbor, one-to-one matching.
    Setiap titik manual maksimal dipasangkan 1x (mencegah 1 manual dihitung TP berkali-kali
    kalau ada beberapa deteksi tumpang tindih di sekitarnya).

    Return:
      matches -> list of (infer_idx, manual_idx, distance)
      fp_idx  -> list infer_idx yang TIDAK dapat pasangan (False Positive)
      fn_idx  -> list manual_idx yang TIDAK dapat pasangan (False Negative)
    """
    n_manual, n_infer = len(manual_xy), len(infer_xy)
    if n_infer == 0:
        return [], [], list(range(n_manual))
    if n_manual == 0:
        return [], list(range(n_infer)), []

    tree = cKDTree(manual_xy)
    dist, idx = tree.query(infer_xy, k=1)
    order = np.argsort(dist)  # proses dari yang paling dekat dulu

    used_manual = set()
    matches = []
    fp = []
    for i in order:
        d = float(dist[i])
        m = int(idx[i])
        if d <= threshold and m not in used_manual:
            used_manual.add(m)
            matches.append((int(i), m, d))
        else:
            fp.append(int(i))
    fn = [m for m in range(n_manual) if m not in used_manual]
    return matches, fp, fn


def evaluate_model(manual_xy, infer_xy, threshold):
    """Hitung metrik lengkap untuk satu model. Return (metrics_dict, matches, fp, fn)."""
    matches, fp, fn = match_greedy(manual_xy, infer_xy, threshold)
    tp = len(matches)
    n_fp = len(fp)
    n_fn = len(fn)
    precision = tp / (tp + n_fp) if (tp + n_fp) > 0 else 0.0
    recall = tp / (tp + n_fn) if (tp + n_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    dists = np.array([d for _, _, d in matches]) if matches else np.array([])
    metrics = {
        "n_manual": len(manual_xy),
        "n_infer": len(infer_xy),
        "tp": tp, "fp": n_fp, "fn": n_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_dist": float(dists.mean()) if dists.size else 0.0,
        "median_dist": float(np.median(dists)) if dists.size else 0.0,
        "rmse_dist": float(np.sqrt((dists ** 2).mean())) if dists.size else 0.0,
        "max_dist": float(dists.max()) if dists.size else 0.0,
    }
    return metrics, matches, fp, fn


def _safe_sheet_name(name: str, used: set) -> str:
    base = "".join(c for c in name if c not in '[]:*?/\\')[:26] or "Model"
    candidate = f"Detail_{base}"[:31]
    n = 1
    while candidate in used:
        n += 1
        candidate = f"Detail_{base}_{n}"[:31]
    used.add(candidate)
    return candidate


def export_comparison_excel(output_path, manual_xy, model_results, threshold):
    """
    model_results: list of dict:
        {"name": str, "xy": np.ndarray, "metrics": dict, "matches": list, "fp": list, "fn": list}
    Menulis 1 sheet ringkasan (buat acuan penilaian antar model) + 1 sheet detail per model.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Ringkasan"

    headers = ["Model", "N Manual", "N Deteksi", "TP", "FP", "FN",
               "Precision", "Recall", "F1", "Mean Dist (m)", "Median Dist (m)",
               "RMSE Dist (m)", "Max Dist (m)"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2FBF71")
        cell.alignment = Alignment(horizontal="center")

    for r in model_results:
        m = r["metrics"]
        ws.append([
            r["name"], m["n_manual"], m["n_infer"], m["tp"], m["fp"], m["fn"],
            round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4),
            round(m["mean_dist"], 4), round(m["median_dist"], 4),
            round(m["rmse_dist"], 4), round(m["max_dist"], 4),
        ])
    for col_idx in range(1, len(headers) + 1):
        ws.column_dimensions[chr(64 + col_idx) if col_idx <= 26 else "A"].width = 14

    ws_info = wb.create_sheet("Info")
    ws_info.append(["Threshold jarak (meter)", threshold])
    ws_info.append(["Jumlah model dibandingkan", len(model_results)])
    ws_info.append(["Dibuat oleh", "Sawit Vision - Pembanding Model"])

    used_names = set()
    for r in model_results:
        sheet_name = _safe_sheet_name(r["name"], used_names)
        sh = wb.create_sheet(sheet_name)
        sh.append(["status", "manual_idx", "infer_idx", "distance_m",
                   "infer_x", "infer_y", "manual_x", "manual_y"])
        for cell in sh[1]:
            cell.font = Font(bold=True)

        infer_xy = r["xy"]
        for i_idx, m_idx, d in r["matches"]:
            sh.append(["TP", m_idx, i_idx, round(d, 4),
                       float(infer_xy[i_idx][0]), float(infer_xy[i_idx][1]),
                       float(manual_xy[m_idx][0]), float(manual_xy[m_idx][1])])
        for i_idx in r["fp"]:
            sh.append(["FP", "", i_idx, "",
                       float(infer_xy[i_idx][0]), float(infer_xy[i_idx][1]), "", ""])
        for m_idx in r["fn"]:
            sh.append(["FN", m_idx, "", "", "", "",
                       float(manual_xy[m_idx][0]), float(manual_xy[m_idx][1])])

    wb.save(output_path)
