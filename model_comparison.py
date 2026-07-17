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
import os
import json
import sqlite3
import struct

import numpy as np
import shapefile  # pyshp
from scipy.spatial import cKDTree
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

SUPPORTED_EXTS = (".shp", ".gpkg", ".geojson", ".json")


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


def _wkb_point_xy(wkb: bytes):
    """Ambil x,y dari WKB Point/PointZ/PointM/PointZM (2 titik pertama cukup buat centroid)."""
    order = "<" if wkb[0] == 1 else ">"
    x, y = struct.unpack(order + "dd", wkb[5:21])
    return x, y


def read_points_gpkg(path):
    """
    Baca titik dari GeoPackage (.gpkg) pakai sqlite3 bawaan Python (tanpa GDAL).
    Ambil layer geometri pertama yang terdaftar di gpkg_geometry_columns.
    Sekalian ambil kolom atribut lain di tabel yang sama (di luar kolom geometri)
    supaya bisa di-join dengan data model lain saat export perbandingan.
    """
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT table_name, column_name FROM gpkg_geometry_columns")
        rows = cur.fetchall()
        if not rows:
            raise ValueError("Tidak ada layer geometri ditemukan di GeoPackage ini.")
        table, geom_col = rows[0]  # layer pertama; kalau ada beberapa layer, ambil yang pertama
        cur.execute(f'SELECT * FROM "{table}"')
        col_names = [d[0] for d in cur.description]
        attr_cols = [c for c in col_names if c != geom_col]
        geom_idx = col_names.index(geom_col)

        pts = []
        attrs = []
        for row in cur.fetchall():
            blob = row[geom_idx]
            if blob is None:
                continue
            # header GeoPackage: 'GP'(2) + version(1) + flags(1) + srs_id(4) [+ envelope]
            flags = blob[3]
            envelope_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get((flags >> 1) & 0x07, 0)
            wkb = blob[8 + envelope_len:]
            pts.append(_wkb_point_xy(wkb))
            attrs.append({c: row[col_names.index(c)] for c in attr_cols})
    finally:
        conn.close()
    if not pts:
        return np.empty((0, 2), dtype=float), []
    return np.array(pts, dtype=float), attrs


def read_points_geojson(path):
    """Baca titik dari .geojson / .json (FeatureCollection Point/MultiPoint/Polygon->centroid).
    Kolom "properties" tiap feature ikut diambil sebagai atribut supaya bisa
    di-join dengan data model lain saat export perbandingan."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = []
    attrs = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        props = feat.get("properties") or {}
        if not coords:
            continue
        if gtype == "Point":
            pts.append((coords[0], coords[1]))
            attrs.append(dict(props))
        elif gtype == "MultiPoint":
            for c in coords:
                pts.append((c[0], c[1]))
                attrs.append(dict(props))
        elif gtype == "Polygon":
            ring = coords[0]
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            pts.append((sum(xs) / len(xs), sum(ys) / len(ys)))
            attrs.append(dict(props))
    if not pts:
        return np.empty((0, 2), dtype=float), []
    return np.array(pts, dtype=float), attrs


def read_points_any(path):
    """Dispatcher: baca titik dari .shp / .gpkg / .geojson / .json berdasarkan ekstensi file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".shp":
        return read_points_shp(path)
    elif ext == ".gpkg":
        return read_points_gpkg(path)
    elif ext in (".geojson", ".json"):
        return read_points_geojson(path)
    raise ValueError(f"Format file tidak didukung: {ext} (pakai .shp / .gpkg / .geojson)")


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


def _numeric_avg(values):
    """Rata-rata dari sekumpulan nilai, lewati yang bukan angka / kosong."""
    nums = []
    for v in values:
        if v is None or v == "":
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    return float(np.mean(nums)) if nums else None


def _numeric_attr_keys(attrs_list):
    """Cari nama kolom atribut yang isinya angka (untuk dihitung rata-ratanya)."""
    keys = []
    seen = set()
    for a in attrs_list:
        for k, v in a.items():
            if k in seen:
                continue
            seen.add(k)
            try:
                if v is not None and v != "":
                    float(v)
                    keys.append(k)
            except (TypeError, ValueError):
                continue
    return keys


def _safe_sheet_name(name: str, used: set) -> str:
    base = "".join(c for c in name if c not in '[]:*?/\\')[:26] or "Model"
    candidate = f"Detail_{base}"[:31]
    n = 1
    while candidate in used:
        n += 1
        candidate = f"Detail_{base}_{n}"[:31]
    used.add(candidate)
    return candidate


# ============================================================
# STYLING HELPERS -- dipakai bersama oleh semua sheet biar tampilannya
# konsisten dan lebih enak dibaca (rapi, tidak berantakan).
# ============================================================
_COLOR_HEADER_FILL = "2FBF71"     # hijau brand Sawit Vision
_COLOR_HEADER_TEXT = "FFFFFF"
_COLOR_BORDER = "D9D9D9"
_COLOR_BAND = "F2F9F6"            # selang-seling baris (banding) supaya mudah diikuti mata
_COLOR_BEST = "C6EFCE"            # highlight model terbaik (F1 tertinggi)
_COLOR_TP = "E2F0D9"              # hijau muda -> deteksi benar
_COLOR_FP = "FCE4E4"              # merah muda -> deteksi salah/duplikat
_COLOR_FN = "FFF2CC"              # kuning muda -> pohon manual yang tidak ketemu

_THIN = Side(style="thin", color=_COLOR_BORDER)
_BORDER_ALL = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _style_header_row(ws, row_idx: int = 1, n_cols: int = None):
    n_cols = n_cols or ws.max_column
    ws.row_dimensions[row_idx].height = 22
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=col_idx)
        cell.font = Font(bold=True, color=_COLOR_HEADER_TEXT)
        cell.fill = PatternFill("solid", fgColor=_COLOR_HEADER_FILL)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = _BORDER_ALL


def _apply_borders(ws, min_row=1, max_row=None, min_col=1, max_col=None):
    max_row = max_row or ws.max_row
    max_col = max_col or ws.max_column
    for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
        for cell in row:
            cell.border = _BORDER_ALL


def _apply_row_banding(ws, header_row: int, last_row: int, n_cols: int, skip_fill=None):
    """Kasih warna selang-seling ke baris data (bukan header) supaya baris demi baris
    lebih gampang diikuti mata saat scroll data yang panjang. skip_fill: set baris
    (nomor absolut) yang TIDAK boleh ditimpa banding (mis. baris yang sudah di-highlight)."""
    skip_fill = skip_fill or set()
    for r in range(header_row + 1, last_row + 1):
        if r in skip_fill:
            continue
        if (r - header_row) % 2 == 0:
            fill = PatternFill("solid", fgColor=_COLOR_BAND)
            for c in range(1, n_cols + 1):
                ws.cell(row=r, column=c).fill = fill


def _autofit_columns(ws, min_width=10, max_width=42, padding=2, start_row=1):
    widths = {}
    for row in ws.iter_rows(min_row=start_row):
        for cell in row:
            if cell.value is None:
                continue
            length = len(str(cell.value))
            col = cell.column_letter
            widths[col] = max(widths.get(col, 0), length)
    for col, length in widths.items():
        ws.column_dimensions[col].width = max(min_width, min(max_width, length + padding))


def _apply_number_formats(ws, headers, start_row, start_col=1):
    """Format kolom berdasarkan nama header-nya: persen untuk Precision/Recall/F1,
    3 desimal untuk kolom jarak (m), integer untuk kolom hitungan (N/TP/FP/FN),
    4 desimal untuk kolom rata-rata atribut (Avg ...)."""
    for offset, header in enumerate(headers):
        col_idx = start_col + offset
        if header in ("Precision", "Recall", "F1"):
            fmt = "0.0%"
        elif header.endswith("(m)"):
            fmt = "0.000"
        elif header in ("N Manual", "N Deteksi", "TP", "FP", "FN"):
            fmt = "0"
        elif header.startswith("Avg "):
            fmt = "0.0000"
        elif header == "distance_m":
            fmt = "0.000"
        elif header.endswith("_x") or header.endswith("_y") or header == "confidence":
            fmt = "0.0000"
        else:
            continue
        for row in ws.iter_rows(min_row=start_row, min_col=col_idx, max_col=col_idx):
            for cell in row:
                if cell.value != "" and cell.value is not None:
                    cell.number_format = fmt


def _write_glossary_sheet(wb, threshold: float):
    """Sheet 'Keterangan': penjelasan istilah & rumus, supaya orang yang baru lihat
    Excel ini (mis. dosen pembimbing / pihak yang menilai) tidak perlu tanya-tanya
    arti tiap kolom."""
    ws = wb.create_sheet("Keterangan")
    ws.append(["Istilah / Kolom", "Penjelasan"])
    _style_header_row(ws, n_cols=2)

    rows = [
        ("N Manual", "Jumlah titik centroid manual (ground truth) yang jadi acuan penilaian."),
        ("N Deteksi", "Jumlah total titik hasil deteksi model AI (sebelum dicocokkan ke titik manual)."),
        ("TP (True Positive)",
         "Deteksi model yang berhasil dipasangkan dengan 1 titik manual dalam radius "
         f"threshold ({threshold} m). Dianggap deteksi yang BENAR."),
        ("FP (False Positive)",
         "Deteksi model yang TIDAK berhasil dipasangkan dengan titik manual manapun "
         "(dalam radius threshold). Bisa berarti deteksi palsu, objek bukan sawit, "
         "atau duplikat yang belum ter-suppress."),
        ("FN (False Negative)",
         "Titik manual yang TIDAK berhasil ditemukan oleh model manapun. Artinya pohon "
         "yang seharusnya ada tapi terlewat/tidak terdeteksi."),
        ("Precision", "Rumus: TP / (TP + FP). Dari SEMUA yang dideteksi model, berapa persen "
                       "yang benar-benar sawit (bukan deteksi palsu/duplikat)."),
        ("Recall", "Rumus: TP / (TP + FN). Dari SEMUA pohon yang sebenarnya ada (manual), "
                    "berapa persen yang berhasil ditemukan model."),
        ("F1", "Rumus: 2 x (Precision x Recall) / (Precision + Recall). Skor tunggal yang "
               "menyeimbangkan Precision & Recall -- dipakai sebagai acuan utama membandingkan model."),
        ("Mean Dist (m)", "Rata-rata jarak (meter) antara titik TP dan pasangan manual-nya. "
                           "Makin kecil, makin presisi posisi deteksi model."),
        ("Median Dist (m)", "Nilai tengah dari semua jarak TP. Lebih tahan terhadap outlier "
                             "dibanding Mean Dist."),
        ("RMSE Dist (m)", "Root Mean Square Error jarak TP. Lebih sensitif terhadap jarak yang "
                           "besar (outlier) dibanding Mean Dist -- kalau RMSE jauh lebih besar "
                           "dari Mean, artinya ada beberapa pasangan yang jaraknya jauh meleset."),
        ("Max Dist (m)", "Jarak TP terjauh -- kasus pencocokan terburuk pada model ini."),
        ("Threshold jarak", f"Radius maksimal ({threshold} m) supaya sebuah deteksi dianggap "
                             "\"cocok\" (match) dengan titik manual terdekatnya."),
        ("Avg Infer.<kolom>", "Rata-rata nilai atribut numerik (mis. confidence) dari SEMUA "
                               "deteksi model ini, bukan cuma yang TP."),
        ("Avg Manual.<kolom> (TP)", "Rata-rata nilai atribut numerik dari titik manual, dihitung "
                                     "HANYA dari pasangan yang berhasil match (TP)."),
        ("Catatan \u2605 F1 Tertinggi", "Model dengan skor F1 paling tinggi di antara yang "
                                         "dibandingkan -- kandidat model terbaik berdasarkan "
                                         "keseimbangan Precision & Recall."),
        ("status (sheet Detail)", "TP = deteksi benar (hijau), FP = deteksi salah/duplikat "
                                   "(merah), FN = pohon manual terlewat (kuning)."),
    ]
    for term, desc in rows:
        ws.append([term, desc])

    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=1).font = Font(bold=True)
        ws.cell(row=r, column=1).alignment = Alignment(vertical="top", wrap_text=True)
        ws.cell(row=r, column=2).alignment = Alignment(vertical="top", wrap_text=True)
        ws.row_dimensions[r].height = 30

    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 95
    ws.freeze_panes = "A2"
    _apply_borders(ws)
    return ws


def export_comparison_excel(output_path, manual_xy, model_results, threshold, manual_attrs=None):
    """
    model_results: list of dict:
        {"name": str, "xy": np.ndarray, "metrics": dict, "matches": list, "fp": list, "fn": list,
         "attrs": list of dict (atribut per titik inference, boleh kosong)}
    manual_attrs: list of dict, atribut per titik centroid manual (boleh None/kosong).

    Struktur workbook:
      1. Ringkasan   -- tabel utama perbandingan model (dengan highlight F1 tertinggi)
      2. Keterangan  -- penjelasan istilah & rumus tiap kolom (TP, Precision, dst)
      3. Info        -- metadata run (threshold, jumlah model, dsb)
      4. Detail_<model> -- baris per titik (TP/FP/FN) dengan warna status
    """
    manual_attrs = manual_attrs or []

    wb = Workbook()
    ws = wb.active
    ws.title = "Ringkasan"

    base_headers = ["Model", "N Manual", "N Deteksi", "TP", "FP", "FN",
                    "Precision", "Recall", "F1", "Mean Dist (m)", "Median Dist (m)",
                    "RMSE Dist (m)", "Max Dist (m)"]

    # Kumpulkan kolom atribut numerik (union dari semua model) supaya bisa dirata-ratakan
    # di sheet Ringkasan berdasarkan pasangan TP yang sudah di-join dengan atribut manual.
    infer_numeric_keys = []
    for r in model_results:
        for k in _numeric_attr_keys(r.get("attrs", [])):
            if k not in infer_numeric_keys:
                infer_numeric_keys.append(k)
    manual_numeric_keys = _numeric_attr_keys(manual_attrs)

    avg_headers = [f"Avg Infer.{k}" for k in infer_numeric_keys] + \
                  [f"Avg Manual.{k} (TP)" for k in manual_numeric_keys]
    headers = base_headers + avg_headers + ["Catatan"]
    ws.append(headers)

    # Model dengan F1 tertinggi di-highlight -- acuan cepat "mana yang terbaik".
    f1_list = [r["metrics"]["f1"] for r in model_results]
    best_idx = int(np.argmax(f1_list)) if f1_list else -1

    for i, r in enumerate(model_results):
        m = r["metrics"]
        infer_attrs = r.get("attrs", [])
        row = [
            r["name"], m["n_manual"], m["n_infer"], m["tp"], m["fp"], m["fn"],
            round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4),
            round(m["mean_dist"], 4), round(m["median_dist"], 4),
            round(m["rmse_dist"], 4), round(m["max_dist"], 4),
        ]
        # Rata-rata atribut infer dihitung dari SEMUA deteksi model ini (bukan cuma TP),
        # supaya tetap terisi walau jumlah TP sedikit/nol.
        for k in infer_numeric_keys:
            avg = _numeric_avg(a.get(k) for a in infer_attrs)
            row.append(round(avg, 4) if avg is not None else "")
        # Rata-rata atribut manual dihitung dari titik manual yang berhasil di-join (TP) saja,
        # karena atribut manual cuma bermakna untuk pasangan yang benar-benar match.
        for k in manual_numeric_keys:
            joined_vals = [manual_attrs[m_idx].get(k) for _, m_idx, _ in r["matches"]
                           if m_idx < len(manual_attrs)]
            avg = _numeric_avg(joined_vals)
            row.append(round(avg, 4) if avg is not None else "")
        row.append("\u2605 F1 Tertinggi" if i == best_idx else "")
        ws.append(row)

    last_row = ws.max_row
    _style_header_row(ws, n_cols=len(headers))
    _apply_number_formats(ws, headers, start_row=2)
    _apply_borders(ws, max_row=last_row, max_col=len(headers))

    best_row_abs = 2 + best_idx if best_idx >= 0 else None
    if best_row_abs:
        fill = PatternFill("solid", fgColor=_COLOR_BEST)
        for c in range(1, len(headers) + 1):
            ws.cell(row=best_row_abs, column=c).fill = fill

    _apply_row_banding(ws, header_row=1, last_row=last_row, n_cols=len(headers),
                       skip_fill={best_row_abs} if best_row_abs else None)
    _autofit_columns(ws)
    ws.freeze_panes = "A2"

    _write_glossary_sheet(wb, threshold)

    ws_info = wb.create_sheet("Info")
    ws_info.append(["Keterangan", "Nilai"])
    _style_header_row(ws_info, n_cols=2)
    ws_info.append(["Threshold jarak (meter)", threshold])
    ws_info.append(["Jumlah model dibandingkan", len(model_results)])
    ws_info.append(["Model dengan F1 tertinggi", model_results[best_idx]["name"] if best_idx >= 0 else "-"])
    ws_info.append(["Dibuat oleh", "Sawit Vision - Pembanding Model"])
    _apply_borders(ws_info)
    _autofit_columns(ws_info)
    ws_info.column_dimensions["A"].width = 28

    used_names = set()
    for r in model_results:
        sheet_name = _safe_sheet_name(r["name"], used_names)
        sh = wb.create_sheet(sheet_name)

        infer_attrs = r.get("attrs", [])
        # Nama kolom atribut infer & manual (union), di-prefix supaya jelas asalnya
        # dan tidak bentrok kalau ada nama kolom yang sama di kedua sumber data.
        infer_attr_keys = []
        for a in infer_attrs:
            for k in a.keys():
                if k not in infer_attr_keys:
                    infer_attr_keys.append(k)
        manual_attr_keys = []
        for a in manual_attrs:
            for k in a.keys():
                if k not in manual_attr_keys:
                    manual_attr_keys.append(k)

        header_row = (["status", "manual_idx", "infer_idx", "distance_m",
                        "infer_x", "infer_y", "manual_x", "manual_y"]
                      + [f"infer.{k}" for k in infer_attr_keys]
                      + [f"manual.{k}" for k in manual_attr_keys])
        sh.append(header_row)

        infer_xy = r["xy"]

        def _infer_attr_row(i_idx):
            a = infer_attrs[i_idx] if i_idx < len(infer_attrs) else {}
            return [a.get(k, "") for k in infer_attr_keys]

        def _manual_attr_row(m_idx):
            a = manual_attrs[m_idx] if m_idx < len(manual_attrs) else {}
            return [a.get(k, "") for k in manual_attr_keys]

        # TP: join langsung atribut manual + infer dalam satu baris, ini yang tadinya kosong.
        for i_idx, m_idx, d in r["matches"]:
            sh.append(["TP", m_idx, i_idx, round(d, 4),
                       float(infer_xy[i_idx][0]), float(infer_xy[i_idx][1]),
                       float(manual_xy[m_idx][0]), float(manual_xy[m_idx][1])]
                      + _infer_attr_row(i_idx)
                      + _manual_attr_row(m_idx))
        for i_idx in r["fp"]:
            sh.append(["FP", "", i_idx, "",
                       float(infer_xy[i_idx][0]), float(infer_xy[i_idx][1]), "", ""]
                      + _infer_attr_row(i_idx)
                      + [""] * len(manual_attr_keys))
        for m_idx in r["fn"]:
            sh.append(["FN", m_idx, "", "", "", "",
                       float(manual_xy[m_idx][0]), float(manual_xy[m_idx][1])]
                      + [""] * len(infer_attr_keys)
                      + _manual_attr_row(m_idx))

        n_cols = len(header_row)
        sh_last_row = sh.max_row
        _style_header_row(sh, n_cols=n_cols)
        _apply_number_formats(sh, header_row, start_row=2)
        _apply_borders(sh, max_row=sh_last_row, max_col=n_cols)

        # Warnai tiap baris sesuai status: TP hijau, FP merah, FN kuning --
        # supaya bisa langsung "dipindai mata" tanpa baca satu-satu.
        status_fill = {"TP": _COLOR_TP, "FP": _COLOR_FP, "FN": _COLOR_FN}
        for row_idx in range(2, sh_last_row + 1):
            status = sh.cell(row=row_idx, column=1).value
            color = status_fill.get(status)
            if color:
                fill = PatternFill("solid", fgColor=color)
                for c in range(1, n_cols + 1):
                    ws_cell = sh.cell(row=row_idx, column=c)
                    if ws_cell.fill.fgColor.rgb in (None, "00000000"):
                        ws_cell.fill = fill

        _autofit_columns(sh)
        sh.freeze_panes = "A2"

    wb.save(output_path)
