# build.spec
# Build: pyinstaller build.spec
#
# CATATAN PENTING sebelum build:
# 1. Jalankan di venv yang SAMA dengan yang dipakai training/inference (torch versi CUDA,
#    ultralytics, rasterio, dst). PyInstaller membundel apa yang terinstall di environment
#    aktif saat build dijalankan.
# 2. best.pt TIDAK dibundel ke dalam exe. Taruh di folder yang sama dengan exe hasil build
#    (folder dist/SawitVision/), user pilih lewat GUI ("Pilih model...").
# 3. Build di Windows untuk hasil .exe Windows (PyInstaller tidak cross-compile).

import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# --- sawit-chan.png: aset gambar, bukan python module, harus didaftarkan manual ---
# taruh di root folder exe (dist/SawitVision/sawit-chan.png), match sama pengecekan
# sys._MEIPASS di load_and_process_sprites().
import os as _os
_sawit_chan_png = _os.path.join(_os.path.dirname(_os.path.abspath(SPEC)), "sawit-chan.png")
if _os.path.exists(_sawit_chan_png):
    datas += [(_sawit_chan_png, ".")]

# --- rasterio: butuh GDAL data files + banyak submodule dinamis ---
for pkg in ["rasterio"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# --- ultralytics: banyak modul di-import dinamis berdasarkan konfigurasi model ---
for pkg in ["ultralytics"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# --- pyproj: dipakai model_comparison.py buat resolve CRS/EPSG apapun (UTM,
# datum lokal, dll) pas export hasil Pembanding Model. WAJIB collect_all,
# bukan cuma hiddenimports -- pyproj butuh proj.db (database EPSG) yang
# didaftarkan sebagai data file, bukan modul python biasa. Tanpa ini exe
# akan crash "PROJ: proj_create_from_database: Cannot find proj.db" begitu
# fitur Pembanding Model coba baca/tulis CRS.
for pkg in ["pyproj"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# --- torch/torchvision: JANGAN pakai collect_all -- itu narik SEMUA submodule
# termasuk testing internals, distributed training, quantization, dll yang gak
# kepake buat inference single-GPU biasa dan bikin ukuran bengkak parah (GB-an).
# Cukup andalkan hook resmi PyInstaller buat torch (sudah otomatis ke-load dari
# pyinstaller-hooks-contrib), model .pt di-load lewat ultralytics jadi kebutuhan
# hidden-imports torch dasar sudah ke-cover dari situ.

# --- lain-lain yang sering luput terdeteksi otomatis ---
hiddenimports += [
    "cv2",
    "shapefile",  # pyshp
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib.tests",
        "numpy.tests",
        # --- paket yang numpang di venv (share site-packages sama Python global)
        # tapi SAMA SEKALI gak dipakai app ini. Aman dikecualikan buat mempercepat
        # build & mengecilkan ukuran exe. Kalau nanti exe error "ModuleNotFoundError"
        # menyebut salah satu nama di bawah, hapus baris itu dari excludes.
        "django",
        "tensorflow",
        "pygame",
        "boto3",
        "botocore",
        "sentry_sdk",
        "psycopg_binary",
        "psycopg2",
        "anyio",
        "win32com",
        "IPython",
        "notebook",
        "jupyter",
        "jupyterlab",
        "sphinx",
        "pytest",
        "dns",
        # --- CATATAN: sempat nambahin exclude buat beberapa submodule torch
        # (torch.distributed, torch.onnx, dll) dengan asumsi gak kepake buat
        # inference single-GPU. TERNYATA SALAH -- torch/__init__.py sendiri
        # import torch.distributed secara internal buat inisialisasi dasar,
        # jadi exclude itu bikin exe crash "No module named 'torch.distributed'".
        # Semua exclude torch.* SUDAH DICABUT. Ukuran gede karena CUDA DLL
        # (cuDNN/cuBLAS/dll) itu memang gak bisa dihindari kalau mau pakai GPU --
        # lihat penjelasan soal NCCL sebelumnya kalau mau coba potong manual.
    ],
    # Catatan: pandas, pyarrow, lxml, matplotlib SENGAJA TIDAK dikecualikan --
    # walau kemungkinan gak dipakai langsung, beberapa dependency ultralytics/scipy
    # kadang diam-diam butuh salah satu dari itu. Kalau mau coba kecilin lagi,
    # tambahkan satu-satu ke excludes di atas, rebuild, TES aplikasinya jalan
    # dulu sebelum nambah yang lain (biar gampang lacak mana yang bikin rusak).
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SawitVision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX sering bikin torch/cv2 dll gagal load, biarkan off
    console=False,        # ganti True dulu saat debugging biar keliatan traceback
    icon=None,             # taruh path .ico di sini kalau punya
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="SawitVision",
)
