# 🌴 Sawit Vision

Desktop application for **Oil Palm Tree Detection** using a **YOLO-based multispectral object detection model**. Sawit Vision provides an intuitive graphical interface for performing inference on raw multispectral drone imagery and exporting detection results into GIS-compatible formats.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)
![YOLO](https://img.shields.io/badge/Model-YOLO-orange)
![Platform](https://img.shields.io/badge/Platform-Windows-success)

---

# 📌 Features

- 🌴 Oil palm tree detection using a trained YOLO model.
- 🛰️ Supports **7-channel multispectral imagery**.
- 🖥️ Modern desktop interface built with **PyQt6**.
- 📍 Automatic export of detection results to **Shapefile (.shp)**.
- 🖼️ Bounding box visualization after inference.
- 📊 Confidence score for each detected object.
- ⚡ GPU acceleration with CUDA (optional).
- 📜 Real-time inference log.
- 🔍 Zoomable image viewer.

---

# 📷 Application Screenshot



<p align="center">
<img src="docs/gui_screenshot.png" width="900">
</p>

> Save your screenshot as:

```
docs/gui_screenshot.png
```

or change the filename in the README.

---

# 📂 Project Structure

```
Sawit-Vision
│
├── app.py
├── inference_core.py
├── inference_multispectral_v2.py
├── main_window.py
├── build.spec
├── requirements.txt
├── README.md
│
├── docs/
│   └── gui_screenshot.png
│
└── dist/
```

---

# 🚀 Installation

## Option 1 — Installer (Recommended)

Download the latest installer from the **Releases** section.

```
SawitVisionSetup.exe
```

Run the installer and follow the installation wizard.

---

## Option 2 — Run from Source

Clone this repository

```bash
git clone https://github.com/zhura24/Sawit-Vision.git
cd Sawit-Vision
```

Install dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Run the application

```bash
python app.py
```

---

# 🛠️ How to Use

1. Launch **Sawit Vision**.
2. Select the trained YOLO model (`.pt`).
3. Select `band_stats_combined.json`.
4. Select the multispectral raster (`.tif`).
5. Click **Jalankan Deteksi**.
6. Wait until inference is completed.

The application will automatically generate:

- Detection preview
- Bounding boxes
- Confidence scores
- JPG visualization
- Shapefile (.shp)

---

# 📁 Output Example

After inference, the application produces:

```
Output Folder
│
├── detection_result.jpg
├── detection_result.shp
├── detection_result.dbf
├── detection_result.shx
└── detection_result.prj
```

---

# 🏗️ Build Executable

Build using PyInstaller

```bash
pyinstaller build.spec
```

The executable will be generated inside:

```
dist/SawitVision/
```

For distribution, it is recommended to use the installer generated with **Inno Setup**.

---

# 💻 Technology Stack

- Python 3.12
- PyQt6
- Ultralytics YOLO
- PyTorch
- Rasterio
- NumPy
- OpenCV
- GeoPandas
- Shapely

---

# ⚠️ Notes

- The application installer may exceed **1 GB** because PyTorch and CUDA libraries are included.
- The trained model (`.pt`) is **not embedded** inside the application and should be selected through the GUI.
- CUDA is optional but recommended for faster inference.

---

# 👨‍💻 Authors

Developed by:

**zhura24**

Computer Engineering  
Universitas Diponegoro

---

# 📄 License

This project is intended for academic and research purposes.
