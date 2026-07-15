# Sawit Vision — Deteksi Sawit Multispektral

GUI desktop (PyQt6) untuk inference model YOLO combined (7-slot canonical)
di raster multispektral mentah, dibangun dari `inference_multispectral_v2.py`.

## Struktur

- `inference_core.py` — engine inference (logic dari v2 script, dijadikan class)
- `main_window.py` — GUI (PyQt6, dark theme, canvas zoomable, log console)
- `app.py` — entry point
- `build.spec` — PyInstaller spec untuk build .exe
- `requirements.txt` — dependency Python

## Jalankan langsung (tanpa build exe dulu)

Di venv yang sama dengan environment training/inference kamu (Python 3.12, CUDA 12.4):

```
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python app.py
```

GUI akan terbuka. Pilih model `.pt`, `band_stats_combined.json`, dan raster `.tif`,
lalu klik **Jalankan Deteksi**. Progress dan log muncul real-time, hasil (bounding box)
ditampilkan langsung di canvas, dan file `.shp` + `.jpg` disimpan di folder yang sama
dengan raster input.

## Build jadi .exe

Setelah `python app.py` jalan lancar di venv-mu, baru build:

```
pyinstaller build.spec
```

Hasil ada di `dist/SawitVision/SawitVision.exe`. **best.pt tidak dibundel** —
taruh file model & band_stats di folder yang mudah diakses, user tinggal pilih
lewat dialog "Pilih model..." saat GUI dibuka. Semua isi `dist/SawitVision/`
harus ikut didistribusikan (bukan cuma .exe-nya), karena situ tempat semua
DLL/dependency numpy, torch, rasterio dll disimpan.

## Hal-hal yang perlu dicek manual sebelum build final

1. **Ukuran hasil build besar** (500MB–1.5GB+) karena torch+CUDA dibundel. Normal.
2. **Set `console=True`** dulu di `build.spec` waktu debugging pertama kali, biar
   kelihatan traceback kalau exe crash. Baru set `False` lagi setelah stabil.
3. **Test build di mesin lain** (atau minimal fresh venv) sebelum distribusi —
   PyInstaller kadang lupa bundle DLL yang "nempel" di environment dev.
4. Kalau exe error `DLL load failed` terkait GDAL/rasterio saat start, biasanya
   perlu tambahan `rasterio._shim` atau `fiona` ke `hiddenimports` — cek pesan
   error persisnya lalu tambahkan modul yang disebut.
5. Icon aplikasi: siapkan file `.ico`, isi path-nya di parameter `icon=` pada
   `build.spec`.

## Belum termasuk (bisa ditambah kalau perlu)

- Preview raster SEBELUM inference dijalankan (saat ini canvas cuma nampilin
  hasil setelah selesai)
- Load ulang hasil `.shp` lama tanpa run ulang
- Export hasil ke format lain (GeoJSON, KML)
- Multi-raster batch processing
