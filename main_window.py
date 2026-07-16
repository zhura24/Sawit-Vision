"""
main_window.py
GUI utama aplikasi Sawit Vision - Deteksi Sawit Multispektral (PyQt6).

Redesign TOTAL tampilan (dark/light theme, header modern, toolbar, sidebar
kolaps, dashboard card, canvas profesional, progress bertahap, log console
berwarna, settings dialog, about dialog, export multi-format, recent files)
di atas engine `inference_core.py` yang TIDAK diubah logikanya sama sekali:
CUDA/GPU detection, worker thread, batch inference, tile generation, band
mapping, band stretch, preview generation, NMS, shapefile export, progress
callback dan cancel process semuanya tetap dari inference_core.py apa adanya.
"""

import csv
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.transform
import torch

from PyQt6.QtCore import (
    QEasingCurve, QObject, QPointF, QPropertyAnimation, QRectF, QSettings,
    QSize, Qt, QThread, QUrl, pyqtSignal, QTimer,
)
from PyQt6.QtGui import (
    QAction, QBrush, QColor, QDesktopServices, QFont, QIcon, QImage,
    QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
)
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget, QStatusBar,
    QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from inference_core import (
    CancelledError, InferenceEngine, build_preview_bgr, is_multiref_schema,
    load_band_stats, load_detection_from_shapefile,
)
from comparison_widget import ComparisonPage

# ============================================================
# IDENTITAS APLIKASI
# ============================================================
APP_ORG = "UniversitasDiponegoro"
APP_NAME = "SawitVision"
APP_TITLE = "Sawit Vision"
APP_SUBTITLE = "Deteksi Sawit Multispektral"
APP_VERSION = "2.1.0"
APP_TITLE_FULL = f"{APP_TITLE} \u2014 {APP_SUBTITLE}"

# ============================================================
# IKON -- digambar inline (vector, satu warna), tanpa aset eksternal
# ============================================================
class Icons:
    """Pabrik ikon minimalis satu-warna, digambar lewat QPainter/QPainterPath."""

    _cache = {}

    @staticmethod
    def icon(name: str, color: str = "#e6e9ed", size: int = 20) -> QIcon:
        key = (name, color, size)
        if key in Icons._cache:
            return Icons._cache[key]
        drawer = _ICON_DRAWERS.get(name)
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color))
        pen.setWidthF(max(1.4, size * 0.09))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if drawer is not None:
            drawer(p, size, QColor(color))
        else:
            p.drawRect(QRectF(size * 0.2, size * 0.2, size * 0.6, size * 0.6))
        p.end()
        icon = QIcon(pm)
        Icons._cache[key] = icon
        return icon

    @staticmethod
    def pixmap(name: str, color: str = "#e6e9ed", size: int = 20) -> QPixmap:
        return Icons.icon(name, color, size).pixmap(size, size)


def _poly(points):
    return QPolygonF([QPointF(x, y) for x, y in points])


def _draw_raster(p, s, c):
    p.drawRect(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76))
    p.drawPolyline(_poly([
        (s * 0.16, s * 0.68), (s * 0.36, s * 0.44), (s * 0.52, s * 0.60),
        (s * 0.68, s * 0.34), (s * 0.84, s * 0.58),
    ]))
    p.setBrush(QBrush(c))
    p.drawEllipse(QPointF(s * 0.68, s * 0.28), s * 0.06, s * 0.06)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_model(p, s, c):
    p.drawRoundedRect(QRectF(s * 0.26, s * 0.26, s * 0.48, s * 0.48), s * 0.06, s * 0.06)
    for f in (0.36, 0.5, 0.64):
        p.drawLine(QPointF(s * f, s * 0.10), QPointF(s * f, s * 0.26))
        p.drawLine(QPointF(s * f, s * 0.74), QPointF(s * f, s * 0.90))
        p.drawLine(QPointF(s * 0.10, s * f), QPointF(s * 0.26, s * f))
        p.drawLine(QPointF(s * 0.74, s * f), QPointF(s * 0.90, s * f))


def _draw_bandstats(p, s, c):
    p.setBrush(QBrush(c))
    p.drawRect(QRectF(s * 0.18, s * 0.52, s * 0.16, s * 0.34))
    p.drawRect(QRectF(s * 0.42, s * 0.30, s * 0.16, s * 0.56))
    p.drawRect(QRectF(s * 0.66, s * 0.42, s * 0.16, s * 0.44))
    p.setBrush(Qt.BrushStyle.NoBrush)


def _draw_run(p, s, c):
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(_poly([(s * 0.28, s * 0.18), (s * 0.28, s * 0.82), (s * 0.82, s * 0.5)]))


def _draw_stop(p, s, c):
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(s * 0.24, s * 0.24, s * 0.52, s * 0.52), s * 0.08, s * 0.08)


def _draw_export(p, s, c):
    p.drawLine(QPointF(s * 0.5, s * 0.12), QPointF(s * 0.5, s * 0.56))
    p.drawPolyline(_poly([(s * 0.32, s * 0.4), (s * 0.5, s * 0.58), (s * 0.68, s * 0.4)]))
    p.drawPolyline(_poly([
        (s * 0.16, s * 0.68), (s * 0.16, s * 0.86), (s * 0.84, s * 0.86), (s * 0.84, s * 0.68),
    ]))


def _draw_clear(p, s, c):
    p.drawPolyline(_poly([
        (s * 0.24, s * 0.28), (s * 0.28, s * 0.86), (s * 0.72, s * 0.86), (s * 0.76, s * 0.28),
    ]))
    p.drawLine(QPointF(s * 0.16, s * 0.28), QPointF(s * 0.84, s * 0.28))
    p.drawLine(QPointF(s * 0.38, s * 0.16), QPointF(s * 0.62, s * 0.16))
    p.drawLine(QPointF(s * 0.62, s * 0.16), QPointF(s * 0.68, s * 0.28))
    p.drawLine(QPointF(s * 0.38, s * 0.16), QPointF(s * 0.32, s * 0.28))
    p.drawLine(QPointF(s * 0.4, s * 0.42), QPointF(s * 0.42, s * 0.74))
    p.drawLine(QPointF(s * 0.6, s * 0.42), QPointF(s * 0.58, s * 0.74))


def _draw_settings(p, s, c):
    """Ikon gear (roda gigi) sesungguhnya: cincin gigi persegi + lubang tengah."""
    import math
    cx, cy = s * 0.5, s * 0.5
    r_outer = s * 0.40
    r_inner = s * 0.27
    tooth_half_angle = math.radians(16)
    n_teeth = 8
    path = QPainterPath()
    started = False
    for i in range(n_teeth):
        base_angle = 2 * math.pi * i / n_teeth
        angles = [
            base_angle - tooth_half_angle * 1.6,
            base_angle - tooth_half_angle,
            base_angle + tooth_half_angle,
            base_angle + tooth_half_angle * 1.6,
        ]
        radii = [r_inner, r_outer, r_outer, r_inner]
        for a, r in zip(angles, radii):
            x, y = cx + r * math.sin(a), cy - r * math.cos(a)
            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)
    p.drawEllipse(QPointF(cx, cy), s * 0.13, s * 0.13)


def _draw_about(p, s, c):
    p.drawEllipse(QRectF(s * 0.14, s * 0.14, s * 0.72, s * 0.72))
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(s * 0.5, s * 0.32), s * 0.045, s * 0.045)
    p.setBrush(Qt.BrushStyle.NoBrush)
    pen = p.pen()
    p.setPen(pen)
    p.drawLine(QPointF(s * 0.5, s * 0.44), QPointF(s * 0.5, s * 0.72))


def _draw_theme_sun(p, s, c):
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.2, s * 0.2)
    for i in range(8):
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(i * 45)
        p.drawLine(QPointF(0, -s * 0.32), QPointF(0, -s * 0.42))
        p.restore()


def _draw_theme_moon(p, s, c):
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68))
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    p.drawEllipse(QRectF(s * 0.32, s * 0.10, s * 0.68, s * 0.68))
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)


def _draw_zoom_in(p, s, c):
    p.drawEllipse(QPointF(s * 0.42, s * 0.42), s * 0.26, s * 0.26)
    p.drawLine(QPointF(s * 0.62, s * 0.62), QPointF(s * 0.86, s * 0.86))
    p.drawLine(QPointF(s * 0.3, s * 0.42), QPointF(s * 0.54, s * 0.42))
    p.drawLine(QPointF(s * 0.42, s * 0.3), QPointF(s * 0.42, s * 0.54))


def _draw_zoom_out(p, s, c):
    p.drawEllipse(QPointF(s * 0.42, s * 0.42), s * 0.26, s * 0.26)
    p.drawLine(QPointF(s * 0.62, s * 0.62), QPointF(s * 0.86, s * 0.86))
    p.drawLine(QPointF(s * 0.3, s * 0.42), QPointF(s * 0.54, s * 0.42))


def _draw_fit(p, s, c):
    for (x1, y1, x2, y2, x3, y3) in [
        (s * 0.14, s * 0.32, s * 0.14, s * 0.14, s * 0.32, s * 0.14),
        (s * 0.68, s * 0.14, s * 0.86, s * 0.14, s * 0.86, s * 0.32),
        (s * 0.86, s * 0.68, s * 0.86, s * 0.86, s * 0.68, s * 0.86),
        (s * 0.32, s * 0.86, s * 0.14, s * 0.86, s * 0.14, s * 0.68),
    ]:
        p.drawPolyline(_poly([(x1, y1), (x2, y2), (x3, y3)]))


def _draw_reset(p, s, c):
    rect = QRectF(s * 0.18, s * 0.18, s * 0.64, s * 0.64)
    p.drawArc(rect, 30 * 16, 300 * 16)
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(_poly([(s * 0.78, s * 0.16), (s * 0.9, s * 0.32), (s * 0.66, s * 0.34)]))


def _draw_actual_size(p, s, c):
    p.drawRect(QRectF(s * 0.18, s * 0.18, s * 0.64, s * 0.64))
    f = QFont("Segoe UI", int(s * 0.28))
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(s * 0.14, s * 0.14, s * 0.72, s * 0.72), Qt.AlignmentFlag.AlignCenter, "1:1")


def _draw_save_image(p, s, c):
    p.drawRoundedRect(QRectF(s * 0.16, s * 0.14, s * 0.68, s * 0.72), s * 0.05, s * 0.05)
    p.drawRect(QRectF(s * 0.3, s * 0.14, s * 0.4, s * 0.2))
    p.drawRect(QRectF(s * 0.28, s * 0.5, s * 0.44, s * 0.28))


def _draw_folder(p, s, c):
    p.drawPolyline(_poly([
        (s * 0.14, s * 0.3), (s * 0.14, s * 0.22), (s * 0.4, s * 0.22), (s * 0.46, s * 0.3),
    ]))
    p.drawRoundedRect(QRectF(s * 0.14, s * 0.3, s * 0.72, s * 0.5), s * 0.04, s * 0.04)


def _draw_gpu(p, s, c):
    p.drawRoundedRect(QRectF(s * 0.26, s * 0.26, s * 0.48, s * 0.48), s * 0.06, s * 0.06)
    for f in (0.4, 0.5, 0.6):
        p.drawLine(QPointF(s * f, s * 0.1), QPointF(s * f, s * 0.26))
        p.drawLine(QPointF(s * f, s * 0.74), QPointF(s * f, s * 0.9))
    p.drawLine(QPointF(s * 0.1, s * 0.5), QPointF(s * 0.26, s * 0.5))
    p.drawLine(QPointF(s * 0.74, s * 0.5), QPointF(s * 0.9, s * 0.5))


def _draw_target(p, s, c):
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.34, s * 0.34)
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.16, s * 0.16)
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.05, s * 0.05)


def _draw_percent(p, s, c):
    p.drawEllipse(QPointF(s * 0.32, s * 0.32), s * 0.12, s * 0.12)
    p.drawEllipse(QPointF(s * 0.68, s * 0.68), s * 0.12, s * 0.12)
    p.drawLine(QPointF(s * 0.22, s * 0.78), QPointF(s * 0.78, s * 0.22))


def _draw_clock(p, s, c):
    p.drawEllipse(QRectF(s * 0.14, s * 0.14, s * 0.72, s * 0.72))
    p.drawLine(QPointF(s * 0.5, s * 0.5), QPointF(s * 0.5, s * 0.28))
    p.drawLine(QPointF(s * 0.5, s * 0.5), QPointF(s * 0.66, s * 0.58))


def _draw_grid(p, s, c):
    for gx in (0.18, 0.54):
        for gy in (0.18, 0.54):
            p.drawRect(QRectF(s * gx, s * gy, s * 0.28, s * 0.28))


def _draw_chevron_down(p, s, c):
    p.drawPolyline(_poly([(s * 0.26, s * 0.38), (s * 0.5, s * 0.62), (s * 0.74, s * 0.38)]))


def _draw_chevron_right(p, s, c):
    p.drawPolyline(_poly([(s * 0.38, s * 0.24), (s * 0.62, s * 0.5), (s * 0.38, s * 0.76)]))


def _draw_close(p, s, c):
    p.drawLine(QPointF(s * 0.26, s * 0.26), QPointF(s * 0.74, s * 0.74))
    p.drawLine(QPointF(s * 0.74, s * 0.26), QPointF(s * 0.26, s * 0.74))


def _draw_compare(p, s, c):
    """Ikon 'pembanding model': dua batang chart berdampingan + panah dua arah."""
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRect(QRectF(s * 0.16, s * 0.44, s * 0.16, s * 0.40))
    p.drawRect(QRectF(s * 0.42, s * 0.24, s * 0.16, s * 0.60))
    p.drawRect(QRectF(s * 0.68, s * 0.56, s * 0.16, s * 0.28))
    p.setBrush(Qt.BrushStyle.NoBrush)
    pen = p.pen()
    p.setPen(pen)
    p.drawLine(QPointF(s * 0.14, s * 0.14), QPointF(s * 0.86, s * 0.14))
    p.drawLine(QPointF(s * 0.14, s * 0.14), QPointF(s * 0.22, s * 0.08))
    p.drawLine(QPointF(s * 0.14, s * 0.14), QPointF(s * 0.22, s * 0.20))


_ICON_DRAWERS = {
    "raster": _draw_raster, "model": _draw_model, "bandstats": _draw_bandstats,
    "run": _draw_run, "stop": _draw_stop, "export": _draw_export, "clear": _draw_clear,
    "settings": _draw_settings, "about": _draw_about, "theme_sun": _draw_theme_sun,
    "theme_moon": _draw_theme_moon, "zoom_in": _draw_zoom_in, "zoom_out": _draw_zoom_out,
    "fit": _draw_fit, "reset": _draw_reset, "actual_size": _draw_actual_size,
    "save_image": _draw_save_image, "folder": _draw_folder, "gpu": _draw_gpu,
    "target": _draw_target, "percent": _draw_percent, "clock": _draw_clock,
    "grid": _draw_grid, "chevron_down": _draw_chevron_down,
    "chevron_right": _draw_chevron_right, "close": _draw_close,
    "compare": _draw_compare,
}


def draw_logo_pixmap(size: int = 64, color: str = "#35c96b") -> QPixmap:
    """Mark abstrak bertema daun sawit, digambar vektor (bukan file gambar)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    base = QColor(color)
    cx, cy = size / 2, size / 2
    leaf_len = size * 0.42
    for i in range(6):
        p.save()
        p.translate(cx, cy)
        p.rotate(i * 60)
        shade = base.lighter(100 + i * 6)
        p.setBrush(QBrush(shade))
        path = QPainterPath()
        path.moveTo(0, 0)
        path.quadTo(size * 0.13, -leaf_len * 0.55, 0, -leaf_len)
        path.quadTo(-size * 0.13, -leaf_len * 0.55, 0, 0)
        p.drawPath(path)
        p.restore()
    p.setBrush(QBrush(base.darker(140)))
    p.drawEllipse(QPointF(cx, cy), size * 0.1, size * 0.1)
    p.end()
    return pm


# ============================================================
# TEMA (Dark / Light) -- token warna + generator QSS
# ============================================================
DARK_TOKENS = {
    "bg": "#1a1b1e", "bg_elevated": "#232428", "bg_panel": "#202124",
    "bg_card": "#26272b", "bg_input": "#161719", "bg_hover": "#323438",
    "border": "#323438", "border_light": "#3f4146",
    "text": "#eceef0", "text_dim": "#b7bac0", "text_faint": "#75787e",
    "accent": "#2fbf71", "accent_hover": "#25a35f", "accent_text": "#ffffff",
    "danger": "#f0524a", "danger_hover": "#d43c34",
    "warning": "#e0a030", "info": "#2ea6ff",
    "canvas_bg": "#121214", "console_text": "#7fd18c",
}

LIGHT_TOKENS = {
    "bg": "#eef0f2", "bg_elevated": "#ffffff", "bg_panel": "#f7f8f9",
    "bg_card": "#ffffff", "bg_input": "#eef0f2", "bg_hover": "#dfe2e6",
    "border": "#d3d6da", "border_light": "#e4e6e9",
    "text": "#1c1d20", "text_dim": "#4b4e53", "text_faint": "#8a8d92",
    "accent": "#178a4c", "accent_hover": "#12703d", "accent_text": "#ffffff",
    "danger": "#d43c34", "danger_hover": "#b6291f",
    "warning": "#b57a12", "info": "#0d7bd6",
    "canvas_bg": "#2b2c30", "console_text": "#137a2f",
}

QSS_TEMPLATE = """
QMainWindow, QWidget {
    background-color: %(bg)s;
    color: %(text)s;
    font-family: 'Segoe UI', 'Sans Serif';
    font-size: 13px;
}
QWidget#headerBar {
    background-color: %(bg_elevated)s;
    border-bottom: 1px solid %(border)s;
}
QLabel#appTitle { font-size: 16px; font-weight: 700; color: %(text)s; }
QLabel#appSubtitle { font-size: 11px; color: %(text_faint)s; }
QToolBar#mainToolbar {
    background-color: %(bg_panel)s;
    border-bottom: 1px solid %(border)s;
    padding: 6px 12px;
    spacing: 8px;
}
QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px;
    color: %(text)s;
}
QToolButton:hover { background-color: %(bg_hover)s; border: 1px solid %(border_light)s; }
QToolButton:pressed { background-color: %(border)s; }
QToolButton:checked { background-color: %(bg_hover)s; border: 1px solid %(accent)s; }
QWidget#sidebar, QScrollArea#sidebarScroll { background-color: %(bg_panel)s; border: none; }
QScrollArea#sidebarScroll QWidget#sidebarInner { background-color: %(bg_panel)s; }
QWidget#dashboardRow { background-color: %(bg)s; border-bottom: 1px solid %(border)s; }

QFrame#dashboardCard {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
}
QFrame#dashboardCard:hover {
    border: 1px solid %(accent)s;
    background-color: %(bg_hover)s;
}
QLabel#cardValue { font-size: 18px; font-weight: 700; color: %(text)s; }
QLabel#cardTitle { font-size: 10px; color: %(text_faint)s; font-weight: 600; }

QFrame#sidebarCard {
    background-color: %(bg_card)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
}
QLabel#sidebarCardTitle {
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
    color: %(accent)s;
    letter-spacing: 0.5px;
}

QPushButton {
    background-color: %(bg_card)s;
    border: 1px solid %(border_light)s;
    border-radius: 7px;
    padding: 6px 12px;
    color: %(text)s;
    font-weight: 600;
}
QPushButton:hover { background-color: %(bg_hover)s; border: 1px solid %(accent)s; }
QPushButton:pressed { background-color: %(border)s; }
QPushButton:disabled { color: %(text_faint)s; border: 1px solid %(border)s; background-color: transparent; }

QPushButton#runButton {
    background-color: %(accent)s;
    border: 1px solid %(accent_hover)s;
    color: %(accent_text)s;
    font-weight: 700;
}
QPushButton#runButton:hover { background-color: %(accent_hover)s; }
QPushButton#runButton:disabled { background-color: %(bg_input)s; border: 1px solid %(border)s; color: %(text_faint)s; }

QPushButton#cancelButton {
    background-color: %(danger)s;
    border: 1px solid %(danger_hover)s;
    color: white;
}
QPushButton#cancelButton:hover { background-color: %(danger_hover)s; }
QPushButton#cancelButton:disabled { background-color: %(bg_input)s; border: 1px solid %(border)s; color: %(text_faint)s; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: %(bg_input)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 5px 8px;
    color: %(text)s;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid %(accent)s; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background-color: %(bg_elevated)s;
    border: 1px solid %(border)s;
    selection-background-color: %(accent)s;
    selection-color: %(accent_text)s;
    color: %(text)s;
}

QPlainTextEdit#logConsole {
    background-color: %(bg_input)s;
    border: none;
    border-top: 1px solid %(border)s;
    color: %(console_text)s;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QSlider::groove:horizontal { height: 4px; background: %(border_light)s; border-radius: 2px; }
QSlider::handle:horizontal {
    background: %(accent)s; width: 14px; margin: -5px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QProgressBar {
    border: 1px solid %(border)s;
    border-radius: 6px;
    text-align: center;
    background-color: %(bg_input)s;
    color: %(text)s;
    height: 18px;
    font-weight: bold;
}
QProgressBar::chunk { background-color: %(accent)s; border-radius: 5px; }
QGraphicsView#canvasView { background-color: %(canvas_bg)s; border: none; }
QWidget#canvasToolbar {
    background-color: %(bg_elevated)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
}
QStatusBar { background-color: %(bg_panel)s; color: %(text_faint)s; border-top: 1px solid %(border)s; }

QFrame#stepperDotPending {
    background-color: %(border)s;
    border-radius: 5px;
}
QFrame#stepperDotActive {
    background-color: %(info)s;
    border: 1px solid %(text)s;
    border-radius: 5px;
}
QFrame#stepperDotDone {
    background-color: %(accent)s;
    border-radius: 5px;
}

QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: %(border_light)s; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: %(text_faint)s; }
QScrollBar:horizontal { background: transparent; height: 8px; }
QScrollBar::handle:horizontal { background: %(border_light)s; border-radius: 4px; min-width: 24px; }
QMenu { background-color: %(bg_elevated)s; border: 1px solid %(border)s; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 22px 6px 12px; border-radius: 5px; color: %(text)s; }
QMenu::item:selected { background-color: %(accent)s; color: %(accent_text)s; }
QToolTip {
    background-color: %(bg_elevated)s; color: %(text)s;
    border: 1px solid %(border)s; padding: 4px 6px; border-radius: 4px;
}
"""


def build_qss(tokens: dict) -> str:
    return QSS_TEMPLATE % tokens


DARK_QSS = build_qss(DARK_TOKENS)
LIGHT_QSS = build_qss(LIGHT_TOKENS)


# ============================================================
# TAHAPAN PROSES (untuk stage-stepper progress) -- deteksi lewat
# potongan kata pada log yang SUDAH dipancarkan inference_core.py.
# ============================================================
STAGES = [
    "Model", "Raster", "Tile", "YOLO", "NMS", "Shapefile", "Preview", "Selesai"
]

_STAGE_TRIGGERS = [
    ("memuat model", 0),
    ("membuka raster", 1),
    ("akan diproses", 2),
    ("batch selesai", 3),
    ("total deteksi sebelum nms", 4),
    ("menyimpan shapefile", 5),
    ("membuat preview visual", 6),
]


def detect_stage(log_line: str, current: int) -> int:
    low = log_line.lower()
    for keyword, idx in _STAGE_TRIGGERS:
        if keyword in low and idx >= current:
            return idx
    return current


def classify_log(msg: str) -> str:
    low = msg.lower()
    if "[error]" in low or "traceback" in low:
        return "ERROR"
    if "peringatan" in low or "[warning]" in low:
        return "WARNING"
    if "selesai" in low or "siap" in low or "shapefile:" in low or "gpu terdeteksi" in low:
        return "SUCCESS"
    return "INFO"


# ============================================================
# WORKER THREAD
# ============================================================
class InferenceWorker(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)   # InferenceResult
    failed = pyqtSignal(str)

    def __init__(self, model_path, stats_path, raster_path, conf, tile_size,
                 overlap, batch_size=8, output_dir=None, out_name=None, force_cpu=False):
        super().__init__()
        self.model_path = model_path
        self.stats_path = stats_path
        self.raster_path = raster_path
        self.conf = conf
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.out_name = out_name
        self.force_cpu = force_cpu
        self._cancelled = False
        self._prev_cuda_visible = None

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.force_cpu:
                self._prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self.log.emit("Mode 'Paksa CPU' aktif dari Pengaturan -- GPU disembunyikan dari proses ini.")

            engine = InferenceEngine(
                self.model_path, self.stats_path,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
                should_cancel=lambda: self._cancelled,
            )
            result = engine.run(
                self.raster_path, conf=self.conf,
                tile_size=self.tile_size, overlap=self.overlap,
                batch_size=self.batch_size, output_dir=self.output_dir,
                out_name=self.out_name,
            )
            self.finished.emit(result)
        except CancelledError:
            self.log.emit("Proses dibatalkan.")
            self.failed.emit("__cancelled__")
        except Exception as e:
            self.log.emit(f"[ERROR] {e}")
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))
        finally:
            if self.force_cpu:
                if self._prev_cuda_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self._prev_cuda_visible


class QuickPreviewWorker(QObject):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, raster_path):
        super().__init__()
        self.raster_path = raster_path

    def run(self):
        try:
            empty_boxes = np.zeros((0, 4))
            empty_scores = np.zeros((0,))
            preview = build_preview_bgr(
                Path(self.raster_path), empty_boxes, empty_scores,
                stretch_lower_pct=1.0, stretch_upper_pct=99.0,
            )
            self.ready.emit(preview)
        except Exception as e:
            self.failed.emit(str(e))


# ============================================================
# CANVAS (zoomable/pannable)
# ============================================================
class CanvasView(QGraphicsView):
    zoomChanged = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setObjectName("canvasView")
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.pixmap_item = None
        self._zoom = 1.0
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._placeholder()

    def _placeholder(self):
        self.scene_.clear()
        self.pixmap_item = None
        text = self.scene_.addText("Belum ada raster dimuat.\nBuka raster (.tif) untuk mulai.")
        text.setDefaultTextColor(Qt.GlobalColor.gray)

    def show_bgr_image(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.scene_.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene_.addItem(self.pixmap_item)
        self.scene_.setSceneRect(self.pixmap_item.boundingRect())
        self.fit_to_view()

    def has_image(self) -> bool:
        return self.pixmap_item is not None

    def current_pixmap(self):
        return self.pixmap_item.pixmap() if self.pixmap_item else None

    def fit_to_view(self):
        if self.pixmap_item is not None:
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = self.transform().m11()
            self.zoomChanged.emit(self._zoom)

    def actual_size(self):
        if self.pixmap_item is not None:
            self.resetTransform()
            self._zoom = 1.0
            self.zoomChanged.emit(self._zoom)

    def zoom_in(self):
        self.scale(1.2, 1.2)
        self._zoom *= 1.2
        self.zoomChanged.emit(self._zoom)

    def zoom_out(self):
        self.scale(1 / 1.2, 1 / 1.2)
        self._zoom /= 1.2
        self.zoomChanged.emit(self._zoom)

    def wheelEvent(self, event):
        if self.pixmap_item is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom *= factor
        self.zoomChanged.emit(self._zoom)

    def save_current_image(self, path: str) -> bool:
        if self.pixmap_item is None:
            return False
        return self.pixmap_item.pixmap().save(path)


# ============================================================
# LOG CONSOLE
# ============================================================
class LogConsole(QPlainTextEdit):
    LEVEL_COLORS = {
        "INFO": "#06b6d4", "SUCCESS": "#10b981",
        "WARNING": "#f59e0b", "ERROR": "#ef4444",
    }

    def __init__(self):
        super().__init__()
        self.setObjectName("logConsole")
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)

    def log(self, message: str, level: str = None):
        level = level or classify_log(message)
        ts = datetime.now().strftime("%H:%M:%S")
        color = self.LEVEL_COLORS.get(level, "#cbd5e1")
        safe = (message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        html = (
            f'<span style="color:#64748b;">[{ts}]</span> '
            f'<span style="color:{color}; font-weight:600;">{level:<7}</span> '
            f'<span>{safe}</span>'
        )
        self.appendHtml(html)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


# ============================================================
# STAGE STEPPER (Dot bar horizontal hemat ruang)
# ============================================================
class StageStepper(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        self.status_label = QLabel("Siap memulai.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(self.status_label)

        # Baris bulatan dot
        dots_row = QWidget()
        dots_layout = QHBoxLayout(dots_row)
        dots_layout.setContentsMargins(0, 0, 0, 0)
        dots_layout.setSpacing(8)
        dots_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.dots = []
        for i in range(len(STAGES)):
            dot = QFrame()
            dot.setFixedSize(10, 10)
            dot.setObjectName("stepperDotPending")
            dots_layout.addWidget(dot)
            self.dots.append(dot)

        layout.addWidget(dots_row)
        self._current = -1

    def set_stage(self, idx: int):
        if idx == self._current:
            return
        self._current = idx
        if 0 <= idx < len(STAGES):
            self.status_label.setText(f"Langkah: {STAGES[idx]}")
        else:
            self.status_label.setText("Siap memulai.")

        for i, dot in enumerate(self.dots):
            if i < idx:
                dot.setObjectName("stepperDotDone")
            elif i == idx:
                dot.setObjectName("stepperDotActive")
            else:
                dot.setObjectName("stepperDotPending")
            dot.style().unpolish(dot)
            dot.style().polish(dot)

    def reset(self):
        self._current = -1
        self.status_label.setText("Siap memulai.")
        for dot in self.dots:
            dot.setObjectName("stepperDotPending")
            dot.style().unpolish(dot)
            dot.style().polish(dot)


# ============================================================
# DASHBOARD CARD
# ============================================================
class DashboardCard(QFrame):
    def __init__(self, icon_name: str, title: str, accent: str = "#10b981"):
        super().__init__()
        self.setObjectName("dashboardCard")
        self._accent = accent
        self.setMinimumWidth(150)
        self.setMinimumHeight(64)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.icon_label = QLabel()
        self.icon_label.setPixmap(Icons.pixmap(icon_name, accent, 26))
        layout.addWidget(self.icon_label)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        self.value_label = QLabel("-")
        self.value_label.setObjectName("cardValue")
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("cardTitle")
        text_col.addWidget(self.value_label)
        text_col.addWidget(self.title_label)
        layout.addLayout(text_col, 1)

    def set_value(self, text: str):
        self.value_label.setText(text)

    def restyle_icon(self, icon_name: str, color: str):
        self.icon_label.setPixmap(Icons.pixmap(icon_name, color, 26))


# ============================================================
# SETTINGS DIALOG
# ============================================================
class SettingsDialog(QDialog):
    def __init__(self, parent, current: dict):
        super().__init__(parent)
        self.setWindowTitle("Pengaturan")
        self.setMinimumWidth(420)
        self.result_values = dict(current)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Gelap (Dark)", "Terang (Light)"])
        self.theme_combo.setCurrentIndex(0 if current.get("theme", "dark") == "dark" else 1)
        form.addRow("Tema:", self.theme_combo)

        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["Otomatis (deteksi CUDA)", "Paksa CPU"])
        self.gpu_combo.setCurrentIndex(1 if current.get("force_cpu", False) else 0)
        form.addRow("Mode GPU:", self.gpu_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(current.get("conf", 0.25))
        form.addRow("Confidence:", self.conf_spin)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(320, 1280)
        self.tile_spin.setSingleStep(64)
        self.tile_spin.setValue(current.get("tile_size", 640))
        form.addRow("Tile size:", self.tile_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 256)
        self.overlap_spin.setSingleStep(16)
        self.overlap_spin.setValue(current.get("overlap", 64))
        form.addRow("Overlap:", self.overlap_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(current.get("batch_size", 8))
        form.addRow("Batch size:", self.batch_spin)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit(current.get("output_dir", ""))
        self.output_edit.setPlaceholderText("(default: folder yang sama dengan raster)")
        btn_browse = QPushButton("Pilih...")
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(btn_browse)
        form.addRow("Folder output:", out_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Pilih folder output")
        if path:
            self.output_edit.setText(path)

    def values(self) -> dict:
        return {
            "theme": "dark" if self.theme_combo.currentIndex() == 0 else "light",
            "force_cpu": self.gpu_combo.currentIndex() == 1,
            "conf": self.conf_spin.value(),
            "tile_size": self.tile_spin.value(),
            "overlap": self.overlap_spin.value(),
            "batch_size": self.batch_spin.value(),
            "output_dir": self.output_edit.text().strip(),
        }


# ============================================================
# ABOUT DIALOG
# ============================================================
class AboutDialog(QDialog):
    def __init__(self, parent, accent="#10b981"):
        super().__init__(parent)
        self.setWindowTitle("Tentang Sawit Vision")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        logo_label = QLabel()
        logo_label.setPixmap(draw_logo_pixmap(72, accent))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        title = QLabel(APP_TITLE)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = title.font()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)
        layout.addWidget(title)

        subtitle = QLabel(f"{APP_SUBTITLE} \u2014 v{APP_VERSION}")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        try:
            cuda_available = torch.cuda.is_available()
            cuda_txt = torch.cuda.get_device_name(0) if cuda_available else "Tidak tersedia"
            torch_cuda_version = torch.version.cuda or "-"
        except Exception:
            cuda_txt, torch_cuda_version = "Tidak diketahui", "-"

        try:
            import ultralytics
            yolo_version = getattr(ultralytics, "__version__", "?")
        except Exception:
            yolo_version = "?"

        try:
            rasterio_version = rasterio.__version__
        except Exception:
            rasterio_version = "?"

        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

        info_lines = [
            ("Python", platform.python_version()),
            ("PyQt6", f"{PYQT_VERSION_STR} (Qt {QT_VERSION_STR})"),
            ("Ultralytics YOLO", yolo_version),
            ("Rasterio", rasterio_version),
            ("Torch", torch.__version__),
            ("CUDA runtime", torch_cuda_version),
            ("GPU terdeteksi", cuda_txt),
            ("Institusi", "Universitas Diponegoro"),
        ]

        info_box = QFrame()
        info_layout = QFormLayout(info_box)
        info_layout.setSpacing(4)
        for label, value in info_lines:
            info_layout.addRow(f"{label}:", QLabel(str(value)))
        layout.addWidget(info_box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


# ============================================================
# SAWIT-CHAN PIXEL DECORATION WIDGET
# ============================================================
class SawitChanWidget(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(32, 40)
        self.setStyleSheet("background: transparent; border: none;")
        self.sprites = {'idle': [], 'walk': [], 'interact': []}
        self.state = 'walk'
        self.direction = 'right'
        self.current_frame = 0
        self.x_pos = 450.0  # Start on the empty part of the toolbar
        self.y_pos = 63     # Centered vertically in the 46px toolbar (y: 60-106)
        self.speed = 0.5    # Pixels per frame step
        
        self.load_and_process_sprites()
        
        # Create speech bubble
        self.bubble = QLabel(parent)
        self.bubble.setStyleSheet("""
            QLabel {
                background-color: #10b981;
                color: white;
                border: 1px solid #059669;
                border-radius: 6px;
                padding: 4px 8px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        self.bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bubble.hide()
        
        # Timers for animation and movement
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.next_frame)
        self.anim_timer.start(150) # 150ms per animation frame
        
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.update_position)
        self.move_timer.start(30) # 30ms for smooth position update
        
        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self.randomize_state)
        self.state_timer.start(8000) # Re-evaluate state every 8 seconds
        
        self.update_display()
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.trigger_interaction()
            
    def is_gpu_active(self):
        parent = self.parent()
        if not parent:
            return False
        force_cpu = False
        if hasattr(parent, "settings"):
            val = parent.settings.value("force_cpu", False)
            force_cpu = (str(val).lower() == 'true') if not isinstance(val, bool) else val
        import torch
        try:
            return torch.cuda.is_available() and not force_cpu
        except Exception:
            return False

    def trigger_interaction(self):
        import random
        
        gpu_active = self.is_gpu_active()
        
        # Count clicks for special reactions
        self._click_count = getattr(self, '_click_count', 0) + 1
        
        if self._click_count >= 5:
            # Annoyed mode - natural, no anime
            phrases = [
                "Aduh diklik mulu sih! 😤",
                "Iya iya aku tau, stop klik aku! 😒",
                "Kerjain sawitnya dulu sana! 😑",
                "Hei, aku lagi jalan nih! 😠",
            ]
        elif gpu_active:
            phrases = [
                "Halo! ✨",
                "Sawit Vision siap! 🌴",
                "Jangan diklik mulu dong 😅",
                "Semangat ya! 😊",
                "Lagi patroli nih... 🌿",
                "Ada apa? 👀",
                "GPU nyala, gas! ⚡",
                "Cuaca bagus hari ini! ☀️",
                "Deteksi sawit jalan! 🚀",
                "Jangan lupa istirahat ya! 💧",
            ]
        else:
            phrases = [
                "Halo! ✨",
                "GPU-nya off... sabar ya! 🐢",
                "Lagi agak lambat nih 😅",
                "Kalau GPU nyala lebih cepet! ⚡",
                "CPU mode, tapi tetep jalan! ⚙️",
                "Sabar ya, lagi proses! ⌛",
                "Nyalain GPU-nya dulu dong! 😬",
                "Tetep semangat! 💪",
            ]
        
        self.state = 'interact'
        self.current_frame = 0
        self.update_display()
        
        # Show speech bubble
        text = random.choice(phrases)
        self.bubble.setText(text)
        self.bubble.adjustSize()
        
        # Position bubble - always above the character
        bx = int(self.x_pos + (self.width() - self.bubble.width()) / 2)
        bx = max(5, bx)  # Don't go offscreen left
        by = self.y_pos - self.bubble.height() - 4
        by = max(2, by)
        self.bubble.move(bx, by)
        self.bubble.show()
        self.bubble.raise_()
        
        # Reset click count after a while
        if self._click_count >= 7:
            QTimer.singleShot(5000, lambda: setattr(self, '_click_count', 0))
        
        # Single shot to end interaction
        QTimer.singleShot(2200, self.stop_interaction)

    def stop_interaction(self):
        self.state = 'walk'
        self.current_frame = 0
        self.bubble.hide()
        self.update_display()

    def load_and_process_sprites(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "sawit-chan.png")
        if not os.path.exists(path):
            path = r"C:\Users\user\Downloads\Savvision\sawit-chan.png"
            
        if not os.path.exists(path):
            self.hide()
            return
            
        img = cv2.imread(path)
        if img is None:
            self.hide()
            return
            
        cell_w = 172
        cell_h = 180
        col_start = 32
        row_starts = [20, 200, 395]  # Row 0: idle, Row 1: walk, Row 2: interact
        
        target_w, target_h = 32, 40
        
        # Load idle and walk
        for row_idx, row_name in [(0, 'idle'), (1, 'walk')]:
            for col in range(4):
                rx = col_start + col * cell_w
                ry = row_starts[row_idx]
                
                crop = img[ry:ry+155, rx:rx+120]
                
                # Transparentize using HSV
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h_val, s_val, v_val = cv2.split(hsv)
                bg_mask = (s_val < 35) & (v_val > 60)
                
                alpha = np.ones(crop.shape[:2], dtype=np.uint8) * 255
                alpha[bg_mask] = 0
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < 15:
                        alpha[labels == i] = 0
                        
                rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = alpha
                
                h_crop, w_crop, _ = rgba.shape
                rgba = np.ascontiguousarray(rgba)
                qimg = QImage(rgba.data, w_crop, h_crop, w_crop * 4, QImage.Format.Format_ARGB32)
                pixmap = QPixmap.fromImage(qimg.copy())
                
                scaled_pixmap = pixmap.scaled(
                    target_w, target_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )
                self.sprites[row_name].append(scaled_pixmap)
                
        # Load interact (Row 2, Cols 3 and 4)
        for col in [3, 4]:
            rx = col_start + col * cell_w
            ry = row_starts[2]
            
            crop = img[ry:ry+155, rx:rx+120]
            
            # Transparentize using HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_val, s_val, v_val = cv2.split(hsv)
            bg_mask = (s_val < 35) & (v_val > 60)
            
            alpha = np.ones(crop.shape[:2], dtype=np.uint8) * 255
            alpha[bg_mask] = 0
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 15:
                    alpha[labels == i] = 0
                    
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha
            
            h_crop, w_crop, _ = rgba.shape
            rgba = np.ascontiguousarray(rgba)
            qimg = QImage(rgba.data, w_crop, h_crop, w_crop * 4, QImage.Format.Format_ARGB32)
            pixmap = QPixmap.fromImage(qimg.copy())
            
            scaled_pixmap = pixmap.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            self.sprites['interact'].append(scaled_pixmap)

    def next_frame(self):
        frames = self.sprites.get(self.state, [])
        if not frames:
            return
        self.current_frame = (self.current_frame + 1) % len(frames)
        self.update_display()

    def update_display(self):
        frames = self.sprites.get(self.state, [])
        if not frames or self.current_frame >= len(frames):
            return
            
        pixmap = frames[self.current_frame]
        
        # Mirror if walking left
        if self.direction == 'left':
            img = pixmap.toImage()
            mirrored = img.mirrored(True, False)
            pixmap = QPixmap.fromImage(mirrored)
            
        self.setPixmap(pixmap)

    def update_position(self):
        if not self.parent():
            return
            
        parent_w = self.parent().width()
        min_x = 420  # Avoid toolbar buttons on the left
        max_x = parent_w - self.width() - 10
        
        if max_x <= min_x:
            return
            
        # Ensure position bounds on window resize
        if self.x_pos > max_x:
            self.x_pos = max_x
        if self.x_pos < min_x:
            self.x_pos = min_x
            
        if self.state == 'walk':
            if self.direction == 'right':
                self.x_pos += self.speed
                if self.x_pos >= max_x:
                    self.x_pos = max_x
                    self.direction = 'left'
                    self.state = 'idle'
                    self.current_frame = 0
            else:
                self.x_pos -= self.speed
                if self.x_pos <= min_x:
                    self.x_pos = min_x
                    self.direction = 'right'
                    self.state = 'idle'
                    self.current_frame = 0
                    
        self.move(int(self.x_pos), self.y_pos)

    def randomize_state(self):
        import random
        if self.state == 'interact':
            return
        # 75% walking, 25% idle
        if random.random() < 0.75:
            self.state = 'walk'
        else:
            self.state = 'idle'
        self.current_frame = 0


# ============================================================
# HEADER BAR
# ============================================================
class HeaderBar(QWidget):
    themeToggled = pyqtSignal()
    settingsRequested = pyqtSignal()
    aboutRequested = pyqtSignal()

    def __init__(self, accent="#10b981"):
        super().__init__()
        self.setObjectName("headerBar")
        self.setFixedHeight(60)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 12, 6)
        layout.setSpacing(10)

        self.logo_label = QLabel()
        self.logo_label.setPixmap(draw_logo_pixmap(36, accent))
        layout.addWidget(self.logo_label)

        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        self.title_label = QLabel(APP_TITLE)
        self.title_label.setObjectName("appTitle")
        self.subtitle_label = QLabel(APP_SUBTITLE)
        self.subtitle_label.setObjectName("appSubtitle")
        title_col.addWidget(self.title_label)
        title_col.addWidget(self.subtitle_label)
        layout.addLayout(title_col)

        layout.addStretch(1)

        self.theme_btn = QToolButton()
        self.theme_btn.setToolTip("Ganti tema terang/gelap")
        self.theme_btn.clicked.connect(self.themeToggled.emit)

        self.settings_btn = QToolButton()
        self.settings_btn.setToolTip("Pengaturan")
        self.settings_btn.clicked.connect(self.settingsRequested.emit)

        self.about_btn = QToolButton()
        self.about_btn.setToolTip("Tentang aplikasi")
        self.about_btn.clicked.connect(self.aboutRequested.emit)

        for b in (self.theme_btn, self.settings_btn, self.about_btn):
            b.setIconSize(QSize(20, 20))
            b.setFixedSize(36, 36)
            layout.addWidget(b)

    def set_theme_icon(self, is_dark: bool, color: str):
        self.theme_btn.setIcon(Icons.icon("theme_moon" if is_dark else "theme_sun", color, 20))

    def set_icons(self, color: str):
        self.settings_btn.setIcon(Icons.icon("settings", color, 20))
        self.about_btn.setIcon(Icons.icon("about", color, 20))

    def set_logo(self, accent: str):
        self.logo_label.setPixmap(draw_logo_pixmap(36, accent))


# ============================================================
# MAIN WINDOW
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE_FULL)
        self.resize(1440, 900)
        self.setWindowIcon(QIcon(draw_logo_pixmap(64, "#10b981")))

        self.settings = QSettings(APP_ORG, APP_NAME)

        self.model_path = None
        self.stats_path = None
        self.raster_path = None
        self.worker = None
        self.thread = None
        self.preview_worker = None
        self.preview_thread = None
        self.last_result = None
        self._run_start_time = None
        self._last_tile_total = 0
        self._current_stage = -1
        self._is_dark = self.settings.value("theme", "dark") == "dark"

        self._build_ui()
        self._apply_theme(initial=True)
        self._load_settings_into_ui()
        self._update_run_enabled()
        self._update_gpu_card()

        # Instantiate Sawit-chan (pixel decoration) on MainWindow
        self.sawit_chan = SawitChanWidget(self)

    # ------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Header ----
        self.header = HeaderBar()
        self.header.themeToggled.connect(self.toggle_theme)
        self.header.settingsRequested.connect(self.open_settings)
        self.header.aboutRequested.connect(self.open_about)
        outer.addWidget(self.header)

        # ---- Toolbar ----
        self.toolbar = QToolBar()
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.setMovable(False)

        self.act_open_raster = QAction("Buka Raster", self)
        self.act_open_raster.setToolTip("Buka Raster (TIF)")
        self.act_open_raster.setStatusTip("Membuka berkas raster multispektral")
        self.act_open_raster.triggered.connect(self.pick_raster)

        self.act_load_model = QAction("Muat Model", self)
        self.act_load_model.setToolTip("Muat Model YOLO (.pt)")
        self.act_load_model.setStatusTip("Memilih file model bobot YOLO")
        self.act_load_model.triggered.connect(self.pick_model)

        self.act_band_stats = QAction("Band Stats", self)
        self.act_band_stats.setToolTip("Band Stats (JSON)")
        self.act_band_stats.setStatusTip("Memilih file statistik band raster")
        self.act_band_stats.triggered.connect(self.pick_stats)

        self.act_run = QAction("Jalankan", self)
        self.act_run.setToolTip("Jalankan Deteksi")
        self.act_run.setStatusTip("Memulai proses inference deteksi")
        self.act_run.triggered.connect(self.start_inference)

        self.act_stop = QAction("Stop", self)
        self.act_stop.setToolTip("Batalkan Deteksi")
        self.act_stop.setStatusTip("Membatalkan proses inference yang berjalan")
        self.act_stop.triggered.connect(self.cancel_inference)
        self.act_stop.setEnabled(False)

        self.act_clear = QAction("Bersihkan", self)
        self.act_clear.setToolTip("Bersihkan Layout")
        self.act_clear.setStatusTip("Reset seluruh input dan canvas")
        self.act_clear.triggered.connect(self.clear_all)

        self.act_load_result = QAction("Muat Hasil", self)
        self.act_load_result.setToolTip("Muat Hasil Shapefile")
        self.act_load_result.setStatusTip("Memuat hasil deteksi lama dari file .shp")
        self.act_load_result.triggered.connect(self.load_existing_result)

        self.act_model_comparison = QAction("Pembanding Model", self)
        self.act_model_comparison.setToolTip("Pembanding Model")
        self.act_model_comparison.setStatusTip(
            "Bandingkan centroid manual vs beberapa hasil inference model AI"
        )
        self.act_model_comparison.setCheckable(True)
        self.act_model_comparison.toggled.connect(self.toggle_comparison_page)

        for a in (self.act_open_raster, self.act_load_model, self.act_band_stats):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()
        for a in (self.act_run, self.act_stop):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()

        self.export_btn = QToolButton()
        self.export_btn.setToolTip("Export Hasil")
        self.export_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu = QMenu(self.export_btn)
        self.export_menu.addAction("Simpan sebagai PNG...", lambda: self.export_image("png"))
        self.export_menu.addAction("Simpan sebagai JPEG...", lambda: self.export_image("jpg"))
        self.export_menu.addAction("Salin Shapefile ke...", self.export_shapefile_copy)
        self.export_menu.addAction("Export GeoJSON...", self.export_geojson)
        self.export_menu.addAction("Export CSV...", self.export_csv)
        self.export_menu.addSeparator()
        self.export_menu.addAction("Export Centroid GeoJSON...", self.export_centroid_geojson)
        self.export_menu.addAction("Export Centroid CSV...", self.export_centroid_csv)
        self.export_btn.setMenu(self.export_menu)
        self.toolbar.addWidget(self.export_btn)

        self.toolbar.addSeparator()
        for a in (self.act_clear, self.act_load_result):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_model_comparison)

        outer.addWidget(self.toolbar)

        # ---- Content: sidebar | (dashboard + canvas + log) ----
        content_split = QSplitter(Qt.Orientation.Horizontal)

        # Panel Kontrol Kiri (Clean, Non-accordion, Non-cluttered)
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setMinimumWidth(320)
        self.sidebar_scroll.setMaximumWidth(380)
        
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(10)
        
        self._build_sidebar()
        self.sidebar_scroll.setWidget(self.sidebar)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._build_dashboard()
        right_layout.addWidget(self.dashboard_row)

        right_split = QSplitter(Qt.Orientation.Vertical)
        self.canvas_container = self._build_canvas_area()
        self.log_console = LogConsole()
        right_split.addWidget(self.canvas_container)
        right_split.addWidget(self.log_console)
        right_split.setSizes([620, 180])
        right_layout.addWidget(right_split, 1)

        content_split.addWidget(self.sidebar_scroll)
        content_split.addWidget(right_widget)
        content_split.setStretchFactor(0, 0)
        content_split.setStretchFactor(1, 1)
        content_split.setSizes([340, 1100])

        # ---- Stack halaman: 0 = Deteksi, 1 = Pembanding Model ----
        self.detection_page = content_split
        self.comparison_page = ComparisonPage()

        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self.detection_page)
        self.content_stack.addWidget(self.comparison_page)

        outer.addWidget(self.content_stack, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Siap.")

    def _build_canvas_area(self):
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(8, 8, 8, 0)
        v.setSpacing(6)

        self.canvas = CanvasView()

        toolbar_row = QWidget()
        toolbar_row.setObjectName("canvasToolbar")
        h = QHBoxLayout(toolbar_row)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(4)

        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setToolTip("Zoom out")
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setToolTip("Zoom in")
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_fit = QToolButton()
        self.btn_fit.setToolTip("Fit ke layar")
        self.btn_fit.clicked.connect(self.canvas.fit_to_view)
        self.btn_actual = QToolButton()
        self.btn_actual.setToolTip("Ukuran asli (100%)")
        self.btn_actual.clicked.connect(self.canvas.actual_size)
        self.btn_reset = QToolButton()
        self.btn_reset.setToolTip("Reset tampilan")
        self.btn_reset.clicked.connect(self.canvas.fit_to_view)
        self.btn_save_img = QToolButton()
        self.btn_save_img.setToolTip("Simpan gambar canvas")
        self.btn_save_img.clicked.connect(lambda: self.export_image("png"))

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(46)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{int(z * 100)}%"))

        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_fit, self.btn_actual,
                  self.btn_reset, self.btn_save_img):
            b.setFixedSize(30, 30)
            h.addWidget(b)
        h.addWidget(self.zoom_label)
        h.addStretch(1)

        v.addWidget(toolbar_row)
        v.addWidget(self.canvas, 1)
        return container

    def _build_dashboard(self):
        self.dashboard_row = QWidget()
        self.dashboard_row.setObjectName("dashboardRow")
        h = QHBoxLayout(self.dashboard_row)
        h.setContentsMargins(12, 10, 12, 10)
        h.setSpacing(10)

        self.card_gpu = DashboardCard("gpu", "GPU")
        self.card_detections = DashboardCard("target", "Jumlah Deteksi")
        self.card_confidence = DashboardCard("percent", "Confidence Rata-rata")
        self.card_time = DashboardCard("clock", "Waktu Inference")
        self.card_tiles = DashboardCard("grid", "Jumlah Tile")

        for c in (self.card_gpu, self.card_detections, self.card_confidence,
                  self.card_time, self.card_tiles):
            h.addWidget(c, 1)

    def _build_sidebar(self):
        # 1. Card Output (nama file + folder tujuan output, ditentukan SEBELUM run)
        card_recent = QFrame()
        card_recent.setObjectName("sidebarCard")
        lay_recent = QVBoxLayout(card_recent)
        lay_recent.setContentsMargins(12, 12, 12, 12)
        lay_recent.setSpacing(6)

        lbl_recent_title = QLabel("OUTPUT")
        lbl_recent_title.setObjectName("sidebarCardTitle")
        lay_recent.addWidget(lbl_recent_title)

        lbl_out_name = QLabel("Nama Output:")
        lay_recent.addWidget(lbl_out_name)
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Otomatis (deteksi_<raster>__<model>)")
        lay_recent.addWidget(self.output_name_edit)

        lbl_out_dir = QLabel("Folder Output:")
        lay_recent.addWidget(lbl_out_dir)
        out_dir_row = QHBoxLayout()
        out_dir_row.setSpacing(6)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Otomatis (folder sama dengan raster)")
        self.btn_pick_out_dir = QToolButton()
        self.btn_pick_out_dir.setFixedSize(32, 28)
        self.btn_pick_out_dir.setToolTip("Pilih folder output")
        self.btn_pick_out_dir.clicked.connect(self.pick_output_dir)
        out_dir_row.addWidget(self.output_dir_edit, 1)
        out_dir_row.addWidget(self.btn_pick_out_dir)
        lay_recent.addLayout(out_dir_row)

        self.btn_open_out = QPushButton("Buka Folder Output")
        self.btn_open_out.clicked.connect(self.open_output_folder)
        lay_recent.addWidget(self.btn_open_out)

        self.sidebar_layout.addWidget(card_recent)

        # 2. Card Data Masukan (Konfigurasi Input)
        card_input = QFrame()
        card_input.setObjectName("sidebarCard")
        lay_input = QVBoxLayout(card_input)
        lay_input.setContentsMargins(12, 12, 12, 12)
        lay_input.setSpacing(8)

        lbl_input_title = QLabel("DATA MASUKAN")
        lbl_input_title.setObjectName("sidebarCardTitle")
        lay_input.addWidget(lbl_input_title)

        # Row Raster
        lay_input.addWidget(QLabel("Raster Input (.tif):"))
        row_raster = QHBoxLayout()
        self.raster_edit = QLineEdit()
        self.raster_edit.setReadOnly(True)
        self.raster_edit.setPlaceholderText("Pilih raster multispektral...")
        self.btn_raster = QPushButton()
        self.btn_raster.setFixedSize(32, 28)
        self.btn_raster.clicked.connect(self.pick_raster)
        row_raster.addWidget(self.raster_edit, 1)
        row_raster.addWidget(self.btn_raster)
        lay_input.addLayout(row_raster)
        self.raster_info_label = QLabel("-")
        self.raster_info_label.setWordWrap(True)
        self.raster_info_label.setObjectName("cardTitle")
        lay_input.addWidget(self.raster_info_label)

        # Row Model
        lay_input.addWidget(QLabel("Model YOLO (.pt):"))
        row_model = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        self.model_edit.setPlaceholderText("Pilih model yolo...")
        self.btn_model = QPushButton()
        self.btn_model.setFixedSize(32, 28)
        self.btn_model.clicked.connect(self.pick_model)
        row_model.addWidget(self.model_edit, 1)
        row_model.addWidget(self.btn_model)
        lay_input.addLayout(row_model)
        self.model_info_label = QLabel("-")
        self.model_info_label.setObjectName("cardTitle")
        lay_input.addWidget(self.model_info_label)

        # Row Band Stats
        lay_input.addWidget(QLabel("Band Stats (.json):"))
        row_stats = QHBoxLayout()
        self.stats_edit = QLineEdit()
        self.stats_edit.setReadOnly(True)
        self.stats_edit.setPlaceholderText("Pilih stats gabungan...")
        self.btn_stats = QPushButton()
        self.btn_stats.setFixedSize(32, 28)
        self.btn_stats.clicked.connect(self.pick_stats)
        row_stats.addWidget(self.stats_edit, 1)
        row_stats.addWidget(self.btn_stats)
        lay_input.addLayout(row_stats)
        self.stats_info_label = QLabel("-")
        self.stats_info_label.setObjectName("cardTitle")
        lay_input.addWidget(self.stats_info_label)

        self.sidebar_layout.addWidget(card_input)

        # 3. Card Parameter Inference
        card_param = QFrame()
        card_param.setObjectName("sidebarCard")
        lay_param = QVBoxLayout(card_param)
        lay_param.setContentsMargins(12, 12, 12, 12)
        lay_param.setSpacing(8)

        lbl_param_title = QLabel("PARAMETER DETEKSI")
        lbl_param_title.setObjectName("sidebarCardTitle")
        lay_param.addWidget(lbl_param_title)

        param_form = QFormLayout()
        param_form.setSpacing(6)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")
        self.conf_slider.valueChanged.connect(lambda val: self.conf_label.setText(f"{val/100:.2f}"))
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_label)
        param_form.addRow("Confidence:", conf_row)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(320, 1280)
        self.tile_spin.setSingleStep(64)
        self.tile_spin.setValue(640)
        param_form.addRow("Tile size:", self.tile_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 256)
        self.overlap_spin.setSingleStep(16)
        self.overlap_spin.setValue(64)
        param_form.addRow("Overlap:", self.overlap_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        self.batch_spin.setToolTip(
            "Jumlah tile yang diproses sekaligus di GPU.\n"
            "Meningkatkan performa namun memerlukan VRAM GPU besar."
        )
        param_form.addRow("Batch size:", self.batch_spin)

        from PyQt6.QtWidgets import QCheckBox
        self.chk_export_centroid = QCheckBox("Ekspor titik tengah (centroid) otomatis")
        self.chk_export_centroid.setToolTip(
            "Jika dicentang, setelah deteksi selesai akan otomatis menyimpan\n"
            "file centroid GeoJSON (.geojson) dan CSV di folder output.\n"
            "Titik tengah dihitung dari pusat bounding box setiap deteksi."
        )
        lay_param.addLayout(param_form)
        lay_param.addWidget(self.chk_export_centroid)
        self.sidebar_layout.addWidget(card_param)

        # 4. Card Proses dan Hasil (Selalu Terbuka / Tampak)
        card_run = QFrame()
        card_run.setObjectName("sidebarCard")
        lay_run = QVBoxLayout(card_run)
        lay_run.setContentsMargins(12, 12, 12, 12)
        lay_run.setSpacing(8)

        lbl_run_title = QLabel("AKSI & STATUS PROSES")
        lbl_run_title.setObjectName("sidebarCardTitle")
        lay_run.addWidget(lbl_run_title)

        lay_btns = QHBoxLayout()
        self.run_btn = QPushButton("Jalankan")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self.start_inference)

        self.cancel_btn = QPushButton("Batal")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.clicked.connect(self.cancel_inference)
        self.cancel_btn.setEnabled(False)

        lay_btns.addWidget(self.run_btn, 2)
        lay_btns.addWidget(self.cancel_btn, 1)
        lay_run.addLayout(lay_btns)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        lay_run.addWidget(self.progress_bar)

        self.stage_stepper = StageStepper()
        lay_run.addWidget(self.stage_stepper)

        self.result_label = QLabel("Belum ada hasil.")
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("cardTitle")
        lay_run.addWidget(self.result_label)

        self.sidebar_layout.addWidget(card_run)
        self.sidebar_layout.addStretch(1)

    # ------------------------------------------------------
    # THEME
    # ------------------------------------------------------
    def _apply_theme(self, initial: bool = False):
        tokens = DARK_TOKENS if self._is_dark else LIGHT_TOKENS
        qss = build_qss(tokens)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qss)

        accent = tokens["accent"]
        text_color = tokens["text"]

        self.header.set_theme_icon(self._is_dark, text_color)
        self.header.set_icons(text_color)
        self.header.set_logo(accent)
        self.setWindowIcon(QIcon(draw_logo_pixmap(64, accent)))

        self.act_open_raster.setIcon(Icons.icon("raster", text_color, 20))
        self.act_load_model.setIcon(Icons.icon("model", text_color, 20))
        self.act_band_stats.setIcon(Icons.icon("bandstats", text_color, 20))
        self.act_run.setIcon(Icons.icon("run", accent, 20))
        self.act_stop.setIcon(Icons.icon("stop", tokens["danger"], 20))
        self.act_clear.setIcon(Icons.icon("clear", text_color, 20))
        self.act_load_result.setIcon(Icons.icon("folder", text_color, 20))
        self.export_btn.setIcon(Icons.icon("export", text_color, 20))
        self.act_model_comparison.setIcon(
            Icons.icon("compare", accent if self.act_model_comparison.isChecked() else text_color, 20)
        )

        self.btn_zoom_in.setIcon(Icons.icon("zoom_in", text_color, 16))
        self.btn_zoom_out.setIcon(Icons.icon("zoom_out", text_color, 16))
        self.btn_fit.setIcon(Icons.icon("fit", text_color, 16))
        self.btn_actual.setIcon(Icons.icon("actual_size", text_color, 16))
        self.btn_reset.setIcon(Icons.icon("reset", text_color, 16))
        self.btn_save_img.setIcon(Icons.icon("save_image", text_color, 16))

        self.btn_raster.setIcon(Icons.icon("folder", text_color, 16))
        self.btn_model.setIcon(Icons.icon("folder", text_color, 16))
        self.btn_stats.setIcon(Icons.icon("folder", text_color, 16))
        self.btn_pick_out_dir.setIcon(Icons.icon("folder", text_color, 16))

        self.card_gpu.restyle_icon("gpu", accent)
        self.card_detections.restyle_icon("target", accent)
        self.card_confidence.restyle_icon("percent", accent)
        self.card_time.restyle_icon("clock", accent)
        self.card_tiles.restyle_icon("grid", accent)

        if not initial:
            self.settings.setValue("theme", "dark" if self._is_dark else "light")

    def toggle_theme(self):
        self._is_dark = not self._is_dark
        self._apply_theme()

    # ------------------------------------------------------
    # SETTINGS / ABOUT DIALOGS
    # ------------------------------------------------------
    def open_settings(self):
        current = {
            "theme": "dark" if self._is_dark else "light",
            "force_cpu": self.settings.value("force_cpu", False, type=bool),
            "conf": self.conf_slider.value() / 100.0,
            "tile_size": self.tile_spin.value(),
            "overlap": self.overlap_spin.value(),
            "batch_size": self.batch_spin.value(),
            "output_dir": self.settings.value("output_dir", "", type=str),
        }
        dlg = SettingsDialog(self, current)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vals = dlg.values()
            new_dark = vals["theme"] == "dark"
            if new_dark != self._is_dark:
                self._is_dark = new_dark
                self._apply_theme()
            self.conf_slider.setValue(int(round(vals["conf"] * 100)))
            self.tile_spin.setValue(vals["tile_size"])
            self.overlap_spin.setValue(vals["overlap"])
            self.batch_spin.setValue(vals["batch_size"])
            self.settings.setValue("force_cpu", vals["force_cpu"])
            self.settings.setValue("output_dir", vals["output_dir"])
            self.output_dir_edit.setText(vals["output_dir"])
            self.statusBar().showMessage("Pengaturan disimpan.")

    def open_about(self):
        accent = DARK_TOKENS["accent"] if self._is_dark else LIGHT_TOKENS["accent"]
        dlg = AboutDialog(self, accent=accent)
        dlg.exec()

    def _load_settings_into_ui(self):
        tile = self.settings.value("tile_size", 640, type=int)
        overlap = self.settings.value("overlap", 64, type=int)
        conf = self.settings.value("conf", 0.25, type=float)
        batch = self.settings.value("batch_size", 8, type=int)
        output_dir = self.settings.value("output_dir", "", type=str)
        self.tile_spin.setValue(tile)
        self.overlap_spin.setValue(overlap)
        self.conf_slider.setValue(int(round(conf * 100)))
        self.batch_spin.setValue(batch)
        self.output_dir_edit.setText(output_dir)

    def _persist_current_settings(self):
        self.settings.setValue("tile_size", self.tile_spin.value())
        self.settings.setValue("overlap", self.overlap_spin.value())
        self.settings.setValue("conf", self.conf_slider.value() / 100.0)
        self.settings.setValue("batch_size", self.batch_spin.value())
        self.settings.setValue("output_dir", self.output_dir_edit.text().strip())

    # ------------------------------------------------------
    # OUTPUT (nama & folder tujuan, ditentukan sebelum run)
    # ------------------------------------------------------
    def pick_output_dir(self):
        start_dir = self.output_dir_edit.text().strip() or (
            str(Path(self.raster_path).parent) if self.raster_path else str(Path.home())
        )
        out_dir = QFileDialog.getExistingDirectory(self, "Pilih folder output", start_dir)
        if out_dir:
            self.output_dir_edit.setText(out_dir)
            self.settings.setValue("output_dir", out_dir)

    def open_output_folder(self):
        out_dir = self.output_dir_edit.text().strip()
        target = out_dir if out_dir else (str(Path(self.raster_path).parent) if self.raster_path else str(Path.home()))
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    # ------------------------------------------------------
    # FILE PICKERS
    # ------------------------------------------------------
    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih model YOLO (.pt)", "", "PyTorch Model (*.pt)")
        if path:
            self.model_path = path
            self.model_edit.setText(path)
            try:
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                self.model_info_label.setText(f"{Path(path).name} ({size_mb:.1f} MB)")
            except OSError:
                self.model_info_label.setText(Path(path).name)
            self._update_run_enabled()
            self.log_console.log(f"Model dipilih: {path}", "INFO")

    def pick_stats(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih band_stats.json", "", "JSON (*.json)")
        if path:
            self.stats_path = path
            self.stats_edit.setText(path)
            try:
                stats = load_band_stats(Path(path))
                multiref = is_multiref_schema(stats)
                mode = "gabungan (multi-sumber)" if multiref else "tunggal"
                self.stats_info_label.setText(f"{len(stats)} slot band, mode {mode}")
            except Exception as e:
                self.stats_info_label.setText(f"Gagal membaca: {e}")
            self._update_run_enabled()
            self.log_console.log(f"Band stats dipilih: {path}", "INFO")

    def pick_raster(self):
        raster_filter = (
            "Raster Files (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.img *.jp2);;"
            "All Files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Buka raster", "", raster_filter)
        if path:
            self._set_raster(path)

    def _set_raster(self, path: str):
        self.raster_path = path
        self.raster_edit.setText(path)
        self._update_run_enabled()
        self.statusBar().showMessage(f"Raster dimuat: {Path(path).name}")
        self.log_console.log(f"Raster dibuka: {path}", "INFO")

        try:
            with rasterio.open(path) as src:
                info = f"{src.width} x {src.height} px, {src.count} band, dtype {src.dtypes[0]}"
                self.raster_info_label.setText(info)
        except Exception as e:
            self.raster_info_label.setText(f"Gagal membaca header: {e}")
            return

        self.result_label.setText("Memuat preview raster...")
        self._start_quick_preview(path)

    def _start_quick_preview(self, path: str):
        self.preview_thread = QThread()
        self.preview_worker = QuickPreviewWorker(path)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.ready.connect(self._on_quick_preview_ready)
        self.preview_worker.failed.connect(self._on_quick_preview_failed)
        self.preview_worker.ready.connect(self.preview_thread.quit)
        self.preview_worker.failed.connect(self.preview_thread.quit)
        self.preview_thread.start()

    def _on_quick_preview_ready(self, preview_bgr):
        self.canvas.show_bgr_image(preview_bgr)
        self.result_label.setText("Preview raster ditampilkan. Siap untuk deteksi.")

    def _on_quick_preview_failed(self, msg):
        self.log_console.log(f"Gagal membuat preview raster: {msg}", "WARNING")
        self.result_label.setText("Preview raster gagal dibuat (raster tetap bisa dideteksi).")

    def _update_run_enabled(self):
        ready = all([self.model_path, self.stats_path, self.raster_path])
        self.run_btn.setEnabled(ready)
        self.act_run.setEnabled(ready)

    def toggle_comparison_page(self, checked: bool):
        """Pindah antara halaman Deteksi (0) dan halaman Pembanding Model (1)."""
        if checked:
            self.content_stack.setCurrentWidget(self.comparison_page)
            self.statusBar().showMessage(
                "Mode Pembanding Model — bandingkan centroid manual vs hasil inference beberapa model."
            )
        else:
            self.content_stack.setCurrentWidget(self.detection_page)
            self.statusBar().showMessage("Siap.")
        self._apply_theme()  # refresh warna ikon compare (aktif/tidak)

    def load_existing_result(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih hasil deteksi (.shp)",
            "",
            "Shapefile (*.shp)",
        )
        if not path:
            return
        try:
            boxes, scores, classes = load_detection_from_shapefile(Path(path))
            result = type("LoadedResult", (), {})()
            result.boxes = boxes
            result.scores = scores
            result.classes = classes
            result.shp_path = Path(path)
            result.preview_path = None
            result.preview_bgr = None
            self.last_result = result
            self.result_label.setText(f"Hasil lama dimuat dari {Path(path).name}.")
            self.statusBar().showMessage(f"Hasil lama dimuat: {Path(path).name}")
            self.log_console.log(f"Hasil lama dimuat: {path}", "SUCCESS")
            if self.raster_path:
                preview = build_preview_bgr(Path(self.raster_path), boxes, scores, 1.0, 99.0)
                self.canvas.show_bgr_image(preview)
            self.card_detections.set_value(str(len(boxes)))
            self.card_confidence.set_value(f"{float(np.mean(scores)) * 100:.1f}%" if len(scores) > 0 else "-")
            self.card_time.set_value("-")
            self.card_tiles.set_value("-")
        except Exception as exc:
            QMessageBox.warning(self, "Gagal memuat hasil", f"Tidak dapat membaca hasil shapefile:\n{exc}")

    # ------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------
    def start_inference(self):
        self._persist_current_settings()
        self.run_btn.setEnabled(False)
        self.act_run.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.act_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.stage_stepper.reset()
        self.log_console.clear()
        self.result_label.setText("Memproses...")
        self.statusBar().showMessage("Menjalankan inference...")
        self._current_stage = -1
        self._run_start_time = None
        self._last_tile_total = 0
        self.card_detections.set_value("-")
        self.card_confidence.set_value("-")
        self.card_time.set_value("-")
        self.card_tiles.set_value("-")

        conf = self.conf_slider.value() / 100.0
        tile_size = self.tile_spin.value()
        overlap = self.overlap_spin.value()
        batch_size = self.batch_spin.value()
        output_dir = self.output_dir_edit.text().strip() or None
        out_name = self.output_name_edit.text().strip() or None
        force_cpu = self.settings.value("force_cpu", False, type=bool)

        import time as _time
        self._run_start_time = _time.time()

        self.thread = QThread()
        self.worker = InferenceWorker(
            self.model_path, self.stats_path, self.raster_path,
            conf, tile_size, overlap, batch_size,
            output_dir=output_dir, out_name=out_name, force_cpu=force_cpu,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)

        self.thread.start()

    def cancel_inference(self):
        if self.worker:
            self.worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.act_stop.setEnabled(False)
        self.statusBar().showMessage("Membatalkan...")

    def append_log(self, msg: str):
        self.log_console.log(msg)
        self._current_stage = detect_stage(msg, self._current_stage)
        self.stage_stepper.set_stage(self._current_stage)

    def update_progress(self, current: int, total: int):
        pct = int(current / total * 100) if total else 0
        self.progress_bar.setValue(pct)
        self._last_tile_total = total
        self.card_tiles.set_value(f"{current}/{total}")
        if self._current_stage < 3:
            self._current_stage = 3
            self.stage_stepper.set_stage(3)

    def _auto_export_centroid(self, result):
        """Dipanggil otomatis setelah inference jika checkbox centroid dicentang."""
        if self.last_result is None or self.raster_path is None:
            return
        try:
            boxes = result.boxes
            scores = result.scores
            with rasterio.open(self.raster_path) as src:
                transform = src.transform
                crs = src.crs

            stem = Path(self.raster_path).stem
            model_stem = Path(self.model_edit.text()).stem if hasattr(self, 'model_edit') else "model"
            out_stem = f"centroid_{stem}__{model_stem}"

            out_dir = Path(self.settings.value("output_dir", "")) if self.settings.value("output_dir", "") else Path(self.raster_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # --- GeoJSON Point ---
            geojson_path = out_dir / f"{out_stem}.geojson"
            geo_features = []
            for i, (box, score) in enumerate(zip(boxes, scores), start=1):
                x1_px, y1_px, x2_px, y2_px = box
                cx_px = (x1_px + x2_px) / 2.0
                cy_px = (y1_px + y2_px) / 2.0
                cx_geo, cy_geo = rasterio.transform.xy(transform, cy_px, cx_px)
                geo_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [cx_geo, cy_geo]},
                    "properties": {"id": i, "kelas": "sawit", "confidence": round(float(score), 4),
                                   "cx_px": round(float(cx_px), 2), "cy_px": round(float(cy_px), 2)},
                })
            fc = {"type": "FeatureCollection", "features": geo_features}
            if crs and crs.to_epsg():
                fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs.to_epsg()}"}}
            with open(geojson_path, "w") as fh:
                json.dump(fc, fh, indent=2)

            # --- CSV Centroid ---
            csv_path = out_dir / f"{out_stem}.csv"
            import csv as _csv
            with open(csv_path, "w", newline="") as fh:
                writer = _csv.DictWriter(fh, fieldnames=["id", "kelas", "confidence", "cx_px", "cy_px", "cx_geo", "cy_geo"])
                writer.writeheader()
                for feat in geo_features:
                    p = feat["properties"]
                    cx_geo, cy_geo = feat["geometry"]["coordinates"]
                    writer.writerow({"id": p["id"], "kelas": p["kelas"], "confidence": p["confidence"],
                                     "cx_px": p["cx_px"], "cy_px": p["cy_px"],
                                     "cx_geo": cx_geo, "cy_geo": cy_geo})

            self.log_console.log(f"Centroid GeoJSON: {geojson_path}", "SUCCESS")
            self.log_console.log(f"Centroid CSV   : {csv_path}", "SUCCESS")
        except Exception as e:
            self.log_console.log(f"[PERINGATAN] Gagal ekspor centroid otomatis: {e}", "WARN")

    def on_finished(self, result):
        import time as _time
        elapsed = _time.time() - self._run_start_time if self._run_start_time else 0.0

        self.run_btn.setEnabled(True)
        self.act_run.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.act_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.stage_stepper.set_stage(len(STAGES) - 1)

        self.last_result = result
        n = len(result.boxes)
        avg_conf = float(np.mean(result.scores)) if n > 0 else 0.0

        self.result_label.setText(
            f"Selesai. {n} objek terdeteksi.\n"
            f"Shapefile: {result.shp_path}\n"
            f"Preview: {result.preview_path}"
        )
        self.statusBar().showMessage(f"Selesai \u2014 {n} objek terdeteksi dalam {elapsed:.1f} detik.")

        self.card_detections.set_value(str(n))
        self.card_confidence.set_value(f"{avg_conf*100:.1f}%" if n > 0 else "-")
        self.card_time.set_value(f"{elapsed:.1f}s")
        self.card_tiles.set_value(str(self._last_tile_total))

        if result.preview_bgr is not None:
            self.canvas.show_bgr_image(result.preview_bgr)

        # Ekspor centroid otomatis jika checkbox dicentang
        if self.chk_export_centroid.isChecked():
            self._auto_export_centroid(result)

    def on_failed(self, error_msg: str):
        self.run_btn.setEnabled(True)
        self.act_run.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.act_stop.setEnabled(False)
        if error_msg == "__cancelled__":
            self.result_label.setText("Dibatalkan.")
            self.statusBar().showMessage("Dibatalkan.")
            self.stage_stepper.reset()
        else:
            self.result_label.setText(f"Gagal: {error_msg}")
            self.statusBar().showMessage("Terjadi error.")
            QMessageBox.critical(self, "Error", f"Inference gagal:\n\n{error_msg}")

    def _update_gpu_card(self):
        try:
            if torch.cuda.is_available():
                self.card_gpu.set_value(torch.cuda.get_device_name(0))
            else:
                self.card_gpu.set_value("CPU sahaja")
        except Exception:
            self.card_gpu.set_value("Tidak diketahui")

    # ------------------------------------------------------
    # CLEAR
    # ------------------------------------------------------
    def clear_all(self):
        self.model_path = None
        self.stats_path = None
        self.raster_path = None
        self.last_result = None
        self.model_edit.clear()
        self.stats_edit.clear()
        self.raster_edit.clear()
        self.model_info_label.setText("-")
        self.stats_info_label.setText("-")
        self.raster_info_label.setText("-")
        self.result_label.setText("Belum ada hasil.")
        self.log_console.clear()
        self.progress_bar.setValue(0)
        self.stage_stepper.reset()
        self.canvas._placeholder()
        for c in (self.card_detections, self.card_confidence, self.card_time, self.card_tiles):
            c.set_value("-")
        self._update_run_enabled()
        self.statusBar().showMessage("Dibersihkan.")

    # ------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------
    def export_image(self, fmt: str):
        if not self.canvas.has_image():
            QMessageBox.information(self, "Export", "Belum ada gambar di canvas untuk disimpan.")
            return
        stem = Path(self.raster_path).stem if self.raster_path else "sawit_vision"
        default_name = f"deteksi_{stem}.{fmt}"
        filt = "PNG (*.png)" if fmt == "png" else "JPEG (*.jpg *.jpeg)"
        path, _ = QFileDialog.getSaveFileName(self, "Simpan gambar", default_name, filt)
        if not path:
            return
        if not path.lower().endswith(f".{fmt}"):
            path += f".{fmt}"
        ok = self.canvas.save_current_image(path)
        if ok:
            self.log_console.log(f"Gambar disimpan: {path}", "SUCCESS")
            self.statusBar().showMessage(f"Gambar disimpan: {path}")
        else:
            QMessageBox.warning(self, "Export", "Gagal menyimpan gambar.")

    def export_shapefile_copy(self):
        if self.last_result is None or self.last_result.shp_path is None:
            QMessageBox.information(self, "Export", "Belum ada hasil shapefile. Jalankan deteksi terlebih dahulu.")
            return
        dest_dir = QFileDialog.getExistingDirectory(self, "Pilih folder tujuan")
        if not dest_dir:
            return
        src_shp = Path(self.last_result.shp_path)
        copied, skipped = [], []
        for suffix in (".shp", ".shx", ".dbf", ".prj"):
            src_file = src_shp.with_suffix(suffix)
            if not src_file.is_file():
                continue
            dst_file = Path(dest_dir) / src_file.name
            if src_file.resolve() == dst_file.resolve():
                skipped.append(dst_file.name)
                continue
            shutil.copy2(src_file, dst_file)
            copied.append(dst_file.name)
        if skipped and not copied:
            self.log_console.log(
                f"Shapefile sudah berada di folder tujuan: {', '.join(skipped)}", "INFO")
            self.statusBar().showMessage("Shapefile sudah ada di folder tujuan.")
            return
        if copied:
            self.log_console.log(f"Shapefile disalin ke {dest_dir}: {', '.join(copied)}", "SUCCESS")
            self.statusBar().showMessage(f"Shapefile disalin ke {dest_dir}")
        else:
            QMessageBox.warning(self, "Export", "Tidak ada berkas shapefile ditemukan untuk disalin.")

    def _geo_features(self):
        if self.last_result is None or self.raster_path is None:
            return None, None
        boxes = self.last_result.boxes
        scores = self.last_result.scores
        classes = self.last_result.classes
        with rasterio.open(self.raster_path) as src:
            transform = src.transform
            crs = src.crs
        features = []
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes), start=1):
            x1_px, y1_px, x2_px, y2_px = box
            x1_geo, y1_geo = rasterio.transform.xy(transform, y1_px, x1_px)
            x2_geo, y2_geo = rasterio.transform.xy(transform, y2_px, x2_px)
            features.append({
                "id": i, "kelas": "sawit", "confidence": round(float(score), 4),
                "x1_px": round(float(x1_px), 1), "y1_px": round(float(y1_px), 1),
                "x2_px": round(float(x2_px), 1), "y2_px": round(float(y2_px), 1),
                "x1_geo": x1_geo, "y1_geo": y1_geo, "x2_geo": x2_geo, "y2_geo": y2_geo,
            })
        return features, crs

    def export_geojson(self):
        features, crs = self._geo_features()
        if features is None:
            QMessageBox.information(self, "Export", "Belum ada hasil deteksi. Jalankan deteksi terlebih dahulu.")
            return
        stem = Path(self.raster_path).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export GeoJSON", f"deteksi_{stem}.geojson", "GeoJSON (*.geojson)")
        if not path:
            return
        geo_features = []
        for f in features:
            x1, y1, x2, y2 = f["x1_geo"], f["y1_geo"], f["x2_geo"], f["y2_geo"]
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
            geo_features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": {
                    "id": f["id"], "kelas": f["kelas"], "confidence": f["confidence"],
                },
            })
        fc = {"type": "FeatureCollection", "features": geo_features}
        try:
            if crs and crs.to_epsg():
                fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs.to_epsg()}"}}
        except Exception:
            pass
        with open(path, "w") as fh:
            json.dump(fc, fh, indent=2)
        self.log_console.log(f"GeoJSON disimpan: {path}", "SUCCESS")
        self.statusBar().showMessage(f"GeoJSON disimpan: {path}")

    def export_csv(self):
        features, _ = self._geo_features()
        if features is None:
            QMessageBox.information(self, "Export", "Belum ada hasil deteksi. Jalankan deteksi terlebih dahulu.")
            return
        stem = Path(self.raster_path).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"deteksi_{stem}.csv", "CSV (*.csv)")
        if not path:
            return
        fieldnames = ["id", "kelas", "confidence", "x1_px", "y1_px", "x2_px", "y2_px",
                      "x1_geo", "y1_geo", "x2_geo", "y2_geo"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for f in features:
                writer.writerow(f)
        self.log_console.log(f"CSV disimpan: {path}", "SUCCESS")
        self.statusBar().showMessage(f"CSV disimpan: {path}")

    # ------------------------------------------------------
    def export_centroid_geojson(self):
        """Export titik tengah (centroid) setiap bounding box sebagai GeoJSON Point."""
        features, crs = self._geo_features()
        if features is None:
            QMessageBox.information(self, "Export", "Belum ada hasil deteksi. Jalankan deteksi terlebih dahulu.")
            return
        stem = Path(self.raster_path).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export Centroid GeoJSON", f"centroid_{stem}.geojson", "GeoJSON (*.geojson)")
        if not path:
            return
        with rasterio.open(self.raster_path) as src:
            transform = src.transform
            crs_obj = src.crs
        geo_features = []
        for f in features:
            cx_px = (f["x1_px"] + f["x2_px"]) / 2.0
            cy_px = (f["y1_px"] + f["y2_px"]) / 2.0
            cx_geo, cy_geo = rasterio.transform.xy(transform, cy_px, cx_px)
            geo_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [cx_geo, cy_geo]},
                "properties": {
                    "id": f["id"], "kelas": f["kelas"], "confidence": f["confidence"],
                    "cx_px": round(cx_px, 2), "cy_px": round(cy_px, 2),
                },
            })
        fc = {"type": "FeatureCollection", "features": geo_features}
        try:
            if crs_obj and crs_obj.to_epsg():
                fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_obj.to_epsg()}"}}
        except Exception:
            pass
        with open(path, "w") as fh:
            json.dump(fc, fh, indent=2)
        self.log_console.log(f"Centroid GeoJSON disimpan: {path}", "SUCCESS")
        self.statusBar().showMessage(f"Centroid GeoJSON disimpan: {path}")

    def export_centroid_csv(self):
        """Export titik tengah (centroid) setiap bounding box sebagai CSV."""
        features, _ = self._geo_features()
        if features is None:
            QMessageBox.information(self, "Export", "Belum ada hasil deteksi. Jalankan deteksi terlebih dahulu.")
            return
        stem = Path(self.raster_path).stem
        path, _ = QFileDialog.getSaveFileName(self, "Export Centroid CSV", f"centroid_{stem}.csv", "CSV (*.csv)")
        if not path:
            return
        with rasterio.open(self.raster_path) as src:
            transform = src.transform
        fieldnames = ["id", "kelas", "confidence", "cx_px", "cy_px", "cx_geo", "cy_geo"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for f in features:
                cx_px = (f["x1_px"] + f["x2_px"]) / 2.0
                cy_px = (f["y1_px"] + f["y2_px"]) / 2.0
                cx_geo, cy_geo = rasterio.transform.xy(transform, cy_px, cx_px)
                writer.writerow({
                    "id": f["id"], "kelas": f["kelas"], "confidence": f["confidence"],
                    "cx_px": round(cx_px, 2), "cy_px": round(cy_px, 2),
                    "cx_geo": cx_geo, "cy_geo": cy_geo,
                })
        self.log_console.log(f"Centroid CSV disimpan: {path}", "SUCCESS")
        self.statusBar().showMessage(f"Centroid CSV disimpan: {path}")

    # ------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas.fit_to_view()

    def closeEvent(self, event):
        self._persist_current_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyleSheet(DARK_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
