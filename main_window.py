"""
main_window.py
GUI utama aplikasi Sawit Vision - Deteksi Sawit Multispektral (PyQt6).
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
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMenu,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget, QStatusBar,
    QToolBar, QToolButton, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem
)

from inference_core import (
    CancelledError, InferenceEngine, build_preview_bgr, is_multiref_schema,
    load_band_stats, load_detection_from_shapefile, resolve_class_name
)
from comparison_widget import ComparisonPage

# ============================================================
# IDENTITAS APLIKASI
# ============================================================
APP_ORG = "UniversitasDiponegoro"
APP_NAME = "SawitVision"
APP_TITLE = "Sawit Vision"
APP_SUBTITLE = "Deteksi Sawit Multispektral"
APP_VERSION = "1.3.0"
APP_TITLE_FULL = f"{APP_TITLE} \u2014 {APP_SUBTITLE}"

# ============================================================
# IKON -- digambar inline (vector, satu warna), tanpa aset eksternal
# ============================================================


class Icons:
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
    p.drawRoundedRect(
        QRectF(
            s * 0.26,
            s * 0.26,
            s * 0.48,
            s * 0.48),
        s * 0.06,
        s * 0.06)
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
    p.drawPolygon(
        _poly([(s * 0.28, s * 0.18), (s * 0.28, s * 0.82), (s * 0.82, s * 0.5)]))


def _draw_stop(p, s, c):
    p.setBrush(QBrush(c))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(
        QRectF(
            s * 0.24,
            s * 0.24,
            s * 0.52,
            s * 0.52),
        s * 0.08,
        s * 0.08)


def _draw_export(p, s, c):
    p.drawLine(QPointF(s * 0.5, s * 0.12), QPointF(s * 0.5, s * 0.56))
    p.drawPolyline(
        _poly([(s * 0.32, s * 0.4), (s * 0.5, s * 0.58), (s * 0.68, s * 0.4)]))
    p.drawPolyline(_poly([(s *
                           0.16, s *
                           0.68), (s *
                                   0.16, s *
                                   0.86), (s *
                                           0.84, s *
                                           0.86), (s *
                                                   0.84, s *
                                                   0.68), ]))


def _draw_clear(p, s, c):
    p.drawPolyline(_poly([(s *
                           0.24, s *
                           0.28), (s *
                                   0.28, s *
                                   0.86), (s *
                                           0.72, s *
                                           0.86), (s *
                                                   0.76, s *
                                                   0.28), ]))
    p.drawLine(QPointF(s * 0.16, s * 0.28), QPointF(s * 0.84, s * 0.28))
    p.drawLine(QPointF(s * 0.38, s * 0.16), QPointF(s * 0.62, s * 0.16))
    p.drawLine(QPointF(s * 0.62, s * 0.16), QPointF(s * 0.68, s * 0.28))
    p.drawLine(QPointF(s * 0.38, s * 0.16), QPointF(s * 0.32, s * 0.28))
    p.drawLine(QPointF(s * 0.4, s * 0.42), QPointF(s * 0.42, s * 0.74))
    p.drawLine(QPointF(s * 0.6, s * 0.42), QPointF(s * 0.58, s * 0.74))


def _draw_settings(p, s, c):
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
    p.drawPolygon(
        _poly([(s * 0.78, s * 0.16), (s * 0.9, s * 0.32), (s * 0.66, s * 0.34)]))


def _draw_actual_size(p, s, c):
    p.drawRect(QRectF(s * 0.18, s * 0.18, s * 0.64, s * 0.64))
    f = QFont("Segoe UI", int(s * 0.28))
    f.setBold(True)
    p.setFont(f)
    p.drawText(
        QRectF(
            s * 0.14,
            s * 0.14,
            s * 0.72,
            s * 0.72),
        Qt.AlignmentFlag.AlignCenter,
        "1:1")


def _draw_save_image(p, s, c):
    p.drawRoundedRect(
        QRectF(
            s * 0.16,
            s * 0.14,
            s * 0.68,
            s * 0.72),
        s * 0.05,
        s * 0.05)
    p.drawRect(QRectF(s * 0.3, s * 0.14, s * 0.4, s * 0.2))
    p.drawRect(QRectF(s * 0.28, s * 0.5, s * 0.44, s * 0.28))


def _draw_folder(p, s, c):
    p.drawPolyline(_poly([(s *
                           0.14, s *
                           0.3), (s *
                                  0.14, s *
                                  0.22), (s *
                   0.4, s *
                   0.22), (s *
                           0.46, s *
                           0.3), ]))
    p.drawRoundedRect(
        QRectF(
            s * 0.14,
            s * 0.3,
            s * 0.72,
            s * 0.5),
        s * 0.04,
        s * 0.04)


def _draw_gpu(p, s, c):
    p.drawRoundedRect(
        QRectF(
            s * 0.26,
            s * 0.26,
            s * 0.48,
            s * 0.48),
        s * 0.06,
        s * 0.06)
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
    p.drawPolyline(
        _poly([(s * 0.26, s * 0.38), (s * 0.5, s * 0.62), (s * 0.74, s * 0.38)]))


def _draw_chevron_right(p, s, c):
    p.drawPolyline(
        _poly([(s * 0.38, s * 0.24), (s * 0.62, s * 0.5), (s * 0.38, s * 0.76)]))


def _draw_close(p, s, c):
    p.drawLine(QPointF(s * 0.26, s * 0.26), QPointF(s * 0.74, s * 0.74))
    p.drawLine(QPointF(s * 0.74, s * 0.26), QPointF(s * 0.26, s * 0.74))


def _draw_file(p, s, c):
    p.drawPolyline(_poly([(s *
                           0.28, s *
                           0.12), (s *
                                   0.6, s *
                                   0.12), (s *
                                           0.76, s *
                                           0.28), (s *
                                                   0.76, s *
                                                   0.88), (s *
                                                           0.28, s *
                                                           0.88), (s *
                                                                   0.28, s *
                                                                   0.12), ]))
    p.drawPolyline(
        _poly([(s * 0.6, s * 0.12), (s * 0.6, s * 0.28), (s * 0.76, s * 0.28)]))
    p.drawLine(QPointF(s * 0.38, s * 0.5), QPointF(s * 0.66, s * 0.5))
    p.drawLine(QPointF(s * 0.38, s * 0.64), QPointF(s * 0.66, s * 0.64))


def _draw_add(p, s, c):
    p.drawLine(QPointF(s * 0.5, s * 0.18), QPointF(s * 0.5, s * 0.82))
    p.drawLine(QPointF(s * 0.18, s * 0.5), QPointF(s * 0.82, s * 0.5))


def _draw_trash(p, s, c):
    p.drawLine(QPointF(s * 0.2, s * 0.28), QPointF(s * 0.8, s * 0.28))
    p.drawLine(QPointF(s * 0.38, s * 0.28), QPointF(s * 0.4, s * 0.16))
    p.drawLine(QPointF(s * 0.4, s * 0.16), QPointF(s * 0.6, s * 0.16))
    p.drawLine(QPointF(s * 0.6, s * 0.16), QPointF(s * 0.62, s * 0.28))
    p.drawPolyline(_poly([(s *
                           0.28, s *
                           0.32), (s *
                                   0.32, s *
                                   0.86), (s *
                                           0.68, s *
                                           0.86), (s *
                                                   0.72, s *
                                                   0.32), ]))
    p.drawLine(QPointF(s * 0.42, s * 0.42), QPointF(s * 0.44, s * 0.76))
    p.drawLine(QPointF(s * 0.58, s * 0.42), QPointF(s * 0.56, s * 0.76))


def _draw_compare(p, s, c):
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
    "raster": _draw_raster,
    "model": _draw_model,
    "bandstats": _draw_bandstats,
    "run": _draw_run,
    "stop": _draw_stop,
    "export": _draw_export,
    "clear": _draw_clear,
    "settings": _draw_settings,
    "about": _draw_about,
    "theme_sun": _draw_theme_sun,
    "theme_moon": _draw_theme_moon,
    "zoom_in": _draw_zoom_in,
    "zoom_out": _draw_zoom_out,
    "fit": _draw_fit,
    "reset": _draw_reset,
    "actual_size": _draw_actual_size,
    "save_image": _draw_save_image,
    "folder": _draw_folder,
    "gpu": _draw_gpu,
    "target": _draw_target,
    "percent": _draw_percent,
    "clock": _draw_clock,
    "grid": _draw_grid,
    "chevron_down": _draw_chevron_down,
    "chevron_right": _draw_chevron_right,
    "close": _draw_close,
    "compare": _draw_compare,
    "add": _draw_add,
    "trash": _draw_trash,
    "file": _draw_file,
}


def draw_logo_pixmap(size: int = 64, color: str = "#35c96b") -> QPixmap:
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
    p.end()
    return pm


# ============================================================
# TEMA (Dark / Light)
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
    border-radius: 8px;
}
QFrame#dashboardCard:hover {
    border: 1px solid %(accent)s;
    background-color: %(bg_hover)s;
}
QLabel#cardValue {
    font-size: 18px;
    font-weight: 700;
    color: %(text)s;
}
QLabel#cardTitle { font-size: 10px; color: %(text_faint)s; font-weight: 600; }
QFrame#dashboardCard QLabel { background-color: transparent; border: none; }
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
QPushButton#cancelButton {
    background-color: %(danger)s;
    border: 1px solid %(danger_hover)s;
    color: white;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: %(bg_input)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 5px 8px;
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
QGraphicsView#canvasView { background-color: %(canvas_bg)s; border: none; }
QWidget#canvasToolbar {
    background-color: %(bg_elevated)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
}
QStatusBar { background-color: %(bg_panel)s; color: %(text_faint)s; border-top: 1px solid %(border)s; }
QFrame#stepperDotPending { background-color: %(border)s; border-radius: 5px; }
QFrame#stepperDotActive { background-color: %(info)s; border: 1px solid %(text)s; border-radius: 5px; }
QFrame#stepperDotDone { background-color: %(accent)s; border-radius: 5px; }
QTableWidget, QListWidget {
    background-color: %(bg_card)s;
    alternate-background-color: %(bg_panel)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    color: %(text)s;
}
"""


def build_qss(tokens: dict) -> str:
    return QSS_TEMPLATE % tokens


DARK_QSS = build_qss(DARK_TOKENS)
LIGHT_QSS = build_qss(LIGHT_TOKENS)

STAGES = [
    "Model",
    "Raster",
    "Tile",
    "YOLO",
    "NMS",
    "Shapefile",
    "Preview",
    "Selesai"]
_STAGE_TRIGGERS = [
    ("memuat model", 0), ("membuka raster", 1), ("akan diproses", 2),
    ("batch selesai", 3), ("total deteksi sebelum nms", 4),
    ("menyimpan shapefile", 5), ("membuat preview visual", 6),
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


class InferenceWorker(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
            self,
            model_path,
            stats_path,
            raster_path,
            conf,
            tile_size,
            overlap,
            batch_size=8,
            output_dir=None,
            out_name=None,
            force_cpu=False,
            centroid_dist_factor=0.5):
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
        self.centroid_dist_factor = centroid_dist_factor
        self._cancelled = False
        self._prev_cuda_visible = None

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.force_cpu:
                self._prev_cuda_visible = os.environ.get(
                    "CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self.log.emit("Mode 'Paksa CPU' aktif.")
            engine = InferenceEngine(
                self.model_path, self.stats_path,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
                should_cancel=lambda: self._cancelled,
            )
            result = engine.run(
                self.raster_path, conf=self.conf,
                tile_size=self.tile_size, overlap=self.overlap,
                centroid_dist_factor=self.centroid_dist_factor,
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


class DetectionDetailsDialog(QDialog):
    def __init__(self, parent, title: str, details: list):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 280)
        layout = QVBoxLayout(self)
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Keterangan", "Nilai"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setColumnWidth(0, 140)
        table.setRowCount(len(details))
        for row, (label, value) in enumerate(details):
            table.setItem(row, 0, QTableWidgetItem(str(label)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
        layout.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class CanvasView(QGraphicsView):
    zoomChanged = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setObjectName("canvasView")
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.pixmap_item = None
        self._zoom = 1.0
        self._overlay_items = []
        self._current_result = None
        self.setRenderHints(self.renderHints(
        ) | QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._placeholder()

    def _placeholder(self):
        self.scene_.clear()
        self.pixmap_item = None
        text = self.scene_.addText(
            "Belum ada raster dimuat.\nBuka raster (.tif) untuk mulai.")
        text.setDefaultTextColor(Qt.GlobalColor.gray)

    def show_bgr_image(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch *
                      w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.scene_.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene_.addItem(self.pixmap_item)
        self.scene_.setSceneRect(self.pixmap_item.boundingRect())
        self._overlay_items = []
        self._current_result = None
        self.fit_to_view()

    def show_result(self, result, class_names=None):
        self._current_result = result
        self._overlay_items = []
        if self.pixmap_item is None:
            return
        if result is None or len(result.boxes) == 0:
            return
        for idx, (box, score, cls_id) in enumerate(
                zip(result.boxes, result.scores, result.classes)):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            rect = QGraphicsRectItem(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            rect.setPen(QPen(QColor(0, 255, 0), 2))
            rect.setBrush(QBrush(QColor(0, 255, 0, 0)))
            rect.setFlag(QGraphicsItem.ItemIsSelectable, True)
            rect.setData(0, {"index": idx, "box": box, "score": float(score), "class_id": int(cls_id), "class_name": str(
                class_names[int(cls_id)] if class_names and 0 <= int(cls_id) < len(class_names) else int(cls_id)), })
            self.scene_.addItem(rect)
            self._overlay_items.append(rect)

    def _open_detail_for_item(self, item):
        if item is None:
            return False
        data = item.data(0)
        if not data:
            return False
        details = [
            ("Index", data["index"] + 1), ("Kelas", data["class_name"]),
            ("Confidence", f"{data['score']:.3f}"),
            ("x1", round(float(data["box"][0]), 2)), ("y1", round(float(data["box"][1]), 2)),
            ("x2", round(float(data["box"][2]), 2)), ("y2", round(float(data["box"][3]), 2)),
        ]
        DetectionDetailsDialog(self, "Detail Deteksi", details).exec()
        return True

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if self._open_detail_for_item(item):
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._current_result is None or not self._overlay_items:
            super().mouseDoubleClickEvent(event)
            return
        item = self.itemAt(event.position().toPoint())
        if self._open_detail_for_item(item):
            return
        super().mouseDoubleClickEvent(event)

    def has_image(self) -> bool:
        return self.pixmap_item is not None

    def fit_to_view(self):
        if self.pixmap_item is not None:
            self.resetTransform()
            self.fitInView(
                self.pixmap_item,
                Qt.AspectRatioMode.KeepAspectRatio)
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
        safe = (
            message.replace(
                "&",
                "&amp;").replace(
                "<",
                "&lt;").replace(
                ">",
                "&gt;"))
        html = f'<span style="color:#64748b;">[{ts}]</span> <span style="color:{color}; font-weight:600;">{
            level:<7}</span> <span>{safe}</span>'
        self.appendHtml(html)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


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


class DashboardCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, icon_name: str, title: str, accent: str = "#10b981"):
        super().__init__()
        self.setObjectName("dashboardCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


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
        self.theme_combo.setCurrentIndex(
            0 if current.get(
                "theme", "dark") == "dark" else 1)
        form.addRow("Tema:", self.theme_combo)

        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["Otomatis (deteksi CUDA)", "Paksa CPU"])
        self.gpu_combo.setCurrentIndex(
            1 if current.get(
                "force_cpu", False) else 0)
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
        self.output_edit.setPlaceholderText(
            "(default: folder yang sama dengan raster)")
        btn_browse = QPushButton("Pilih...")
        btn_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(btn_browse)
        form.addRow("Folder output:", out_row)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
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
            cuda_txt = torch.cuda.get_device_name(
                0) if cuda_available else "Tidak tersedia"
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
# SAWIT-CHAN PIXEL DECORATION WIDGET (BEBAS ROAMING & DRAG STABLE)
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

        # POSISI AWAL AMAN DI ATAS LANTAI CANVAS PREVIEW (DALAM KOORDINAT
        # WINDOW UTAMA)
        self.x_pos = 380.0
        self.y_pos = 650.0
        self.speed = 0.5

        self.load_and_process_sprites()

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

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.next_frame)
        self.anim_timer.start(150)

        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.update_position)
        self.move_timer.start(30)

        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self.randomize_state)
        self.state_timer.start(8000)

        # Smooth Dragging (Lerp)
        self._target_x = float(self.x_pos)
        self._target_y = float(self.y_pos)
        self.drag_timer = QTimer(self)
        self.drag_timer.timeout.connect(self._smooth_drag_update)
        self.drag_timer.setInterval(16)
        self._drag_active = False
        self._drag_start_global = None
        self._drag_start_pos = None

        self.update_display()
        if any(self.sprites.values()):
            self.show()
            self.raise_()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._drag_start_global = event.globalPosition().toPoint()
            self._drag_start_pos = self.pos()
            self._target_x = float(self.pos().x())
            self._target_y = float(self.pos().y())

            self.move_timer.stop()
            self.state = 'interact'
            self.current_frame = 0
            self.update_display()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.drag_timer.start()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_active and self.parent():
            delta = event.globalPosition().toPoint() - self._drag_start_global
            new_x = self._drag_start_pos.x() + delta.x()
            new_y = self._drag_start_pos.y() + delta.y()

            parent_w = self.parent().width()
            parent_h = self.parent().height()

            self._target_x = float(max(0, min(new_x, parent_w - self.width())))
            self._target_y = float(
                max(0, min(new_y, parent_h - self.height())))
            event.accept()

    def _smooth_drag_update(self):
        factor = 0.15
        self.x_pos += (self._target_x - self.x_pos) * factor
        self.y_pos += (self._target_y - self.y_pos) * factor

        self.move(int(self.x_pos), int(self.y_pos))

        if self.bubble.isVisible():
            bx = int(self.x_pos + (self.width() - self.bubble.width()) / 2)
            bx = max(5, bx)
            by = int(self.y_pos) - self.bubble.height() - 4
            by = max(2, by)
            self.bubble.move(bx, by)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_active:
                self._drag_active = False
                self.drag_timer.stop()
                self.setCursor(Qt.CursorShape.ArrowCursor)

                delta_x = abs(self.x_pos - self._drag_start_pos.x())
                delta_y = abs(self.y_pos - self._drag_start_pos.y())

                if delta_x < 5 and delta_y < 5:
                    self.trigger_interaction()
                else:
                    # Kunci posisi Y saat ini sebagai lantai jalan baru agar
                    # tidak reset
                    self.y_pos = float(self.pos().y())
                    self._target_y = self.y_pos
                    QTimer.singleShot(200, self._resume_after_drag)

                event.accept()

    def _resume_after_drag(self):
        self.state = 'walk'
        self.current_frame = 0
        self.bubble.hide()
        self.move_timer.start(30)
        self.update_display()

    def is_gpu_active(self):
        parent = self.parent()
        if not parent:
            return False
        force_cpu = False
        if hasattr(parent, "settings"):
            val = parent.settings.value("force_cpu", False)
            force_cpu = (
                str(val).lower() == 'true') if not isinstance(
                val, bool) else val
        import torch
        try:
            return torch.cuda.is_available() and not force_cpu
        except Exception:
            return False

    def trigger_interaction(self):
        import random
        gpu_active = self.is_gpu_active()
        self._click_count = getattr(self, '_click_count', 0) + 1

        if self._click_count >= 5:
            phrases = [
                "Aduh diklik mulu sih! 😤",
                "Iya iya aku tau, stop klik aku! 😒",
                "Kerjain sawitnya dulu sana! 😑",
                "Hei, aku lagi jalan nih! 😠"]
        elif gpu_active:
            phrases = [
                "Halo! ✨",
                "Sawit Vision siap! 🌴",
                "Jangan diklik mulu dong 😅",
                "Semangat ya! 😊",
                "Lagi patroli nih... 🌿",
                "Ada apa? 👀",
                "GPU nyala, gas! ⚡",
                "Cuaca bagus hari ini! ☀️"]
        else:
            phrases = [
                "Halo! ✨",
                "GPU-nya off... sabar ya! 🐢",
                "Lagi agak lambat nih 😅",
                "Kalau GPU nyala lebih cepet! ⚡",
                "CPU mode, tapi tetep jalan! ⚙️"]

        self.state = 'interact'
        self.current_frame = 0
        self.update_display()

        text = random.choice(phrases)
        self.bubble.setText(text)
        self.bubble.adjustSize()

        bx = int(self.x_pos + (self.width() - self.bubble.width()) / 2)
        bx = max(5, bx)
        by = int(self.y_pos) - self.bubble.height() - 4
        by = max(2, by)
        self.bubble.move(bx, by)
        self.bubble.show()
        self.bubble.raise_()

        if self._click_count >= 7:
            QTimer.singleShot(5000, lambda: setattr(self, '_click_count', 0))

        QTimer.singleShot(2200, self.stop_interaction)

    def stop_interaction(self):
        self.state = 'walk'
        self.current_frame = 0
        self.bubble.hide()
        self.update_display()

    def load_and_process_sprites(self):
        import sys as _sys
        candidates = [
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)),
                "sawit-chan.png"),
            os.path.join(
                os.getcwd(),
                "sawit-chan.png"),
            r"C:\Users\user\Downloads\Savvision\sawit-chan.png",
        ]
        if getattr(_sys, "frozen", False):
            candidates.insert(
                0,
                os.path.join(
                    os.path.dirname(
                        _sys.executable),
                    "sawit-chan.png"))
            if hasattr(_sys, "_MEIPASS"):
                candidates.insert(
                    0, os.path.join(
                        _sys._MEIPASS, "sawit-chan.png"))

        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            self.hide()
            return

        try:
            img = cv2.imread(path)
            if img is None:
                self.hide()
                return

            cell_w, cell_h = 172, 180
            col_start = 32
            row_starts = [20, 200, 395]
            target_w, target_h = 32, 40

            def _process_cell(rx, ry):
                crop = img[ry:ry + 155, rx:rx + 120]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    raise ValueError("Crop error")
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h_val, s_val, v_val = cv2.split(hsv)
                bg_mask = (s_val < 35) & (v_val > 60)
                alpha = np.ones(crop.shape[:2], dtype=np.uint8) * 255
                alpha[bg_mask] = 0
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    alpha)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < 15:
                        alpha[labels == i] = 0
                rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = alpha
                h_crop, w_crop, _ = rgba.shape
                rgba = np.ascontiguousarray(rgba)
                qimg = QImage(
                    rgba.data,
                    w_crop,
                    h_crop,
                    w_crop * 4,
                    QImage.Format.Format_ARGB32)
                pixmap = QPixmap.fromImage(qimg.copy())
                return pixmap.scaled(
                    target_w,
                    target_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation)

            for row_idx, row_name in [(0, 'idle'), (1, 'walk')]:
                for col in range(4):
                    rx = col_start + col * cell_w
                    ry = row_starts[row_idx]
                    self.sprites[row_name].append(_process_cell(rx, ry))

            for col in [3, 4]:
                rx = col_start + col * cell_w
                ry = row_starts[2]
                self.sprites['interact'].append(_process_cell(rx, ry))

        except Exception:
            self.hide()
            return

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
        if self.direction == 'left':
            img = pixmap.toImage()
            mirrored = img.mirrored(True, False)
            pixmap = QPixmap.fromImage(mirrored)
        self.setPixmap(pixmap)

    def update_position(self):
        if not self.parent():
            return

        parent_w = self.parent().width()
        min_x = 330
        max_x = parent_w - self.width() - 10

        if max_x <= min_x:
            return

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

        self.move(int(self.x_pos), int(self.y_pos))

    def randomize_state(self):
        import random
        if self.state == 'interact':
            return
        if random.random() < 0.75:
            self.state = 'walk'
        else:
            self.state = 'idle'
        self.current_frame = 0


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
        self.logo_label.setStyleSheet("background: transparent; padding: 0px;")
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
        self.theme_btn.clicked.connect(self.themeToggled.emit)
        self.settings_btn = QToolButton()
        self.settings_btn.clicked.connect(self.settingsRequested.emit)
        self.about_btn = QToolButton()
        self.about_btn.clicked.connect(self.aboutRequested.emit)

        for b in (self.theme_btn, self.settings_btn, self.about_btn):
            b.setIconSize(QSize(20, 20))
            b.setFixedSize(36, 36)
            layout.addWidget(b)

    def set_theme_icon(self, is_dark: bool, color: str):
        self.theme_btn.setIcon(
            Icons.icon(
                "theme_moon" if is_dark else "theme_sun",
                color,
                20))

    def set_icons(self, color: str):
        self.settings_btn.setIcon(Icons.icon("settings", color, 20))
        self.about_btn.setIcon(Icons.icon("about", color, 20))

    def set_logo(self, accent: str):
        self.logo_label.setPixmap(draw_logo_pixmap(36, accent))


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

        # Sawit-chan kembali diparentkan ke MainWindow supaya bebas roam di
        # seluruh layar aplikasi
        self.sawit_chan = SawitChanWidget(self)

    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = HeaderBar()
        self.header.themeToggled.connect(self.toggle_theme)
        self.header.settingsRequested.connect(self.open_settings)
        self.header.aboutRequested.connect(self.open_about)
        outer.addWidget(self.header)

        self.toolbar = QToolBar()
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.setMovable(False)

        self.act_open_raster = QAction("Buka Raster", self)
        self.act_open_raster.triggered.connect(self.pick_raster)
        self.act_load_model = QAction("Muat Model", self)
        self.act_load_model.triggered.connect(self.pick_model)
        self.act_band_stats = QAction("Band Stats", self)
        self.act_band_stats.triggered.connect(self.pick_stats)
        self.act_run = QAction("Jalankan", self)
        self.act_run.triggered.connect(self.start_inference)
        self.act_stop = QAction("Stop", self)
        self.act_stop.triggered.connect(self.cancel_inference)
        self.act_stop.setEnabled(False)
        self.act_clear = QAction("Bersihkan", self)
        self.act_clear.triggered.connect(self.clear_all)
        self.act_load_result = QAction("Muat Hasil", self)
        self.act_load_result.triggered.connect(self.load_existing_result)

        self.act_model_comparison = QAction("Pembanding Model", self)
        self.act_model_comparison.setCheckable(True)
        self.act_model_comparison.toggled.connect(self.toggle_comparison_page)

        for a in (
                self.act_open_raster,
                self.act_load_model,
                self.act_band_stats):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()
        for a in (self.act_run, self.act_stop):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()

        self.export_btn = QToolButton()
        self.export_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.export_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu = QMenu(self.export_btn)
        self.export_menu.addAction(
            "Simpan sebagai PNG...",
            lambda: self.export_image("png"))
        self.export_menu.addAction(
            "Simpan sebagai JPEG...",
            lambda: self.export_image("jpg"))
        self.export_menu.addAction(
            "Salin Shapefile ke...",
            self.export_shapefile_copy)
        self.export_menu.addAction("Export GeoJSON...", self.export_geojson)
        self.export_menu.addAction("Export CSV...", self.export_csv)
        self.export_menu.addSeparator()
        self.export_menu.addAction(
            "Export Centroid GeoJSON...",
            self.export_centroid_geojson)
        self.export_menu.addAction(
            "Export Centroid CSV...",
            self.export_centroid_csv)
        self.export_btn.setMenu(self.export_menu)
        self.toolbar.addWidget(self.export_btn)

        self.toolbar.addSeparator()
        for a in (self.act_clear, self.act_load_result):
            self.toolbar.addAction(a)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_model_comparison)

        outer.addWidget(self.toolbar)

        content_split = QSplitter(Qt.Orientation.Horizontal)

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

        self.detection_page = content_split
        self.comparison_page = ComparisonPage()
        _init_tokens = DARK_TOKENS if self._is_dark else LIGHT_TOKENS
        self.comparison_page.apply_theme_icons(
            _init_tokens["text"],
            _init_tokens["accent"],
            _init_tokens["accent_text"])

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
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_fit = QToolButton()
        self.btn_fit.clicked.connect(self.canvas.fit_to_view)
        self.btn_actual = QToolButton()
        self.btn_actual.clicked.connect(self.canvas.actual_size)
        self.btn_reset = QToolButton()
        self.btn_reset.clicked.connect(self.canvas.fit_to_view)
        self.btn_save_img = QToolButton()
        self.btn_save_img.clicked.connect(lambda: self.export_image("png"))

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(46)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.zoomChanged.connect(
            lambda z: self.zoom_label.setText(f"{int(z * 100)}%"))

        for b in (
            self.btn_zoom_out,
            self.btn_zoom_in,
            self.btn_fit,
            self.btn_actual,
            self.btn_reset,
                self.btn_save_img):
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
        card_recent = QFrame()
        card_recent.setObjectName("sidebarCard")
        lay_recent = QVBoxLayout(card_recent)
        lay_recent.setContentsMargins(12, 12, 12, 12)
        lay_recent.setSpacing(6)

        lbl_recent_title = QLabel("OUTPUT")
        lbl_recent_title.setObjectName("sidebarCardTitle")
        lay_recent.addWidget(lbl_recent_title)

        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("Otomatis")
        lay_recent.addWidget(self.output_name_edit)

        out_dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Otomatis")
        self.btn_pick_out_dir = QToolButton()
        self.btn_pick_out_dir.setFixedSize(32, 28)
        self.btn_pick_out_dir.clicked.connect(self.pick_output_dir)
        out_dir_row.addWidget(self.output_dir_edit, 1)
        out_dir_row.addWidget(self.btn_pick_out_dir)
        lay_recent.addLayout(out_dir_row)

        self.btn_open_out = QPushButton("Buka Folder Output")
        self.btn_open_out.clicked.connect(self.open_output_folder)
        lay_recent.addWidget(self.btn_open_out)

        self.sidebar_layout.addWidget(card_recent)

        card_input = QFrame()
        card_input.setObjectName("sidebarCard")
        lay_input = QVBoxLayout(card_input)
        lay_input.setContentsMargins(12, 12, 12, 12)
        lay_input.setSpacing(8)

        lbl_input_title = QLabel("DATA MASUKAN")
        lbl_input_title.setObjectName("sidebarCardTitle")
        lay_input.addWidget(lbl_input_title)

        row_raster = QHBoxLayout()
        self.raster_edit = QLineEdit()
        self.raster_edit.setReadOnly(True)
        self.btn_raster = QPushButton()
        self.btn_raster.setFixedSize(32, 28)
        self.btn_raster.clicked.connect(self.pick_raster)
        row_raster.addWidget(self.raster_edit, 1)
        row_raster.addWidget(self.btn_raster)
        lay_input.addLayout(row_raster)
        self.raster_info_label = QLabel("-")
        lay_input.addWidget(self.raster_info_label)

        row_model = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        self.btn_model = QPushButton()
        self.btn_model.setFixedSize(32, 28)
        self.btn_model.clicked.connect(self.pick_model)
        row_model.addWidget(self.model_edit, 1)
        row_model.addWidget(self.btn_model)
        lay_input.addLayout(row_model)
        self.model_info_label = QLabel("-")
        lay_input.addWidget(self.model_info_label)

        row_stats = QHBoxLayout()
        self.stats_edit = QLineEdit()
        self.stats_edit.setReadOnly(True)
        self.btn_stats = QPushButton()
        self.btn_stats.setFixedSize(32, 28)
        self.btn_stats.clicked.connect(self.pick_stats)
        row_stats.addWidget(self.stats_edit, 1)
        row_stats.addWidget(self.btn_stats)
        lay_input.addLayout(row_stats)
        self.stats_info_label = QLabel("-")
        lay_input.addWidget(self.stats_info_label)

        self.sidebar_layout.addWidget(card_input)

        card_param = QFrame()
        card_param.setObjectName("sidebarCard")
        lay_param = QVBoxLayout(card_param)
        lay_param.setContentsMargins(12, 12, 12, 12)
        lay_param.setSpacing(8)

        lbl_param_title = QLabel("PARAMETER")
        lbl_param_title.setObjectName("sidebarCardTitle")
        lay_param.addWidget(lbl_param_title)

        param_form = QFormLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")
        self.conf_slider.valueChanged.connect(
            lambda val: self.conf_label.setText(f"{val / 100:.2f}"))
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_label)
        param_form.addRow("Confidence:", conf_row)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(320, 1280)
        self.tile_spin.setValue(640)
        param_form.addRow("Tile size:", self.tile_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 256)
        self.overlap_spin.setValue(64)
        param_form.addRow("Overlap:", self.overlap_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        param_form.addRow("Batch size:", self.batch_spin)

        from PyQt6.QtWidgets import QCheckBox
        self.chk_export_centroid = QCheckBox("Ekspor titik tengah otomatis")
        lay_param.addLayout(param_form)
        lay_param.addWidget(self.chk_export_centroid)
        self.sidebar_layout.addWidget(card_param)

        card_run = QFrame()
        card_run.setObjectName("sidebarCard")
        lay_run = QVBoxLayout(card_run)
        lay_run.setContentsMargins(12, 12, 12, 12)
        lay_run.setSpacing(8)

        lbl_run_title = QLabel("AKSI & STATUS")
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
        lay_run.addWidget(self.result_label)

        self.sidebar_layout.addWidget(card_run)
        self.sidebar_layout.addStretch(1)

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
            Icons.icon(
                "compare",
                accent if self.act_model_comparison.isChecked() else text_color,
                20))
        if hasattr(self, "comparison_page"):
            self.comparison_page.apply_theme_icons(
                text_color, accent, tokens["accent_text"])

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
            self.settings.setValue(
                "theme", "dark" if self._is_dark else "light")

    def toggle_theme(self):
        self._is_dark = not self._is_dark
        self._apply_theme()

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

    def open_about(self):
        accent = DARK_TOKENS["accent"] if self._is_dark else LIGHT_TOKENS["accent"]
        dlg = AboutDialog(self, accent=accent)
        dlg.exec()

    def _load_settings_into_ui(self):
        self.tile_spin.setValue(
            self.settings.value(
                "tile_size", 640, type=int))
        self.overlap_spin.setValue(
            self.settings.value(
                "overlap", 64, type=int))
        self.conf_slider.setValue(
            int(round(self.settings.value("conf", 0.25, type=float) * 100)))
        self.batch_spin.setValue(
            self.settings.value(
                "batch_size", 8, type=int))
        self.output_dir_edit.setText(
            self.settings.value(
                "output_dir", "", type=str))

    def _persist_current_settings(self):
        self.settings.setValue("tile_size", self.tile_spin.value())
        self.settings.setValue("overlap", self.overlap_spin.value())
        self.settings.setValue("conf", self.conf_slider.value() / 100.0)
        self.settings.setValue("batch_size", self.batch_spin.value())
        self.settings.setValue(
            "output_dir",
            self.output_dir_edit.text().strip())

    def pick_output_dir(self):
        start_dir = self.output_dir_edit.text().strip() or (
            str(Path(self.raster_path).parent) if self.raster_path else str(Path.home()))
        out_dir = QFileDialog.getExistingDirectory(
            self, "Pilih folder output", start_dir)
        if out_dir:
            self.output_dir_edit.setText(out_dir)
            self.settings.setValue("output_dir", out_dir)

    def open_output_folder(self):
        out_dir = self.output_dir_edit.text().strip()
        target = out_dir if out_dir else (
            str(Path(self.raster_path).parent) if self.raster_path else str(Path.home()))
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih model YOLO (.pt)", "", "PyTorch Model (*.pt)")
        if path:
            self.model_path = path
            self.model_edit.setText(path)
            try:
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                self.model_info_label.setText(
                    f"{Path(path).name} ({size_mb:.1f} MB)")
            except OSError:
                self.model_info_label.setText(Path(path).name)
            self._update_run_enabled()

    def pick_stats(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih band_stats.json", "", "JSON (*.json)")
        if path:
            self.stats_path = path
            self.stats_edit.setText(path)
            try:
                stats = load_band_stats(Path(path))
                multiref = is_multiref_schema(stats)
                self.stats_info_label.setText(f"{len(stats)} slot band")
            except Exception as e:
                self.stats_info_label.setText(f"Gagal: {e}")
            self._update_run_enabled()

    def pick_raster(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Buka raster", "", "Raster Files (*.tif *.tiff *.png *.jpg *.jpeg)")
        if path:
            self._set_raster(path)

    def _set_raster(self, path: str):
        self.raster_path = path
        self.raster_edit.setText(path)
        self._update_run_enabled()
        try:
            with rasterio.open(path) as src:
                self.raster_info_label.setText(
                    f"{src.width} x {src.height} px")
        except Exception as e:
            self.raster_info_label.setText(f"Error: {e}")
            return
        self._start_quick_preview(path)

    def _start_quick_preview(self, path: str):
        self.preview_thread = QThread()
        self.preview_worker = QuickPreviewWorker(path)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.ready.connect(self.canvas.show_bgr_image)
        self.preview_worker.ready.connect(self.preview_thread.quit)
        self.preview_thread.start()

    def _update_run_enabled(self):
        ready = all([self.model_path, self.stats_path, self.raster_path])
        self.run_btn.setEnabled(ready)
        self.act_run.setEnabled(ready)

    def toggle_comparison_page(self, checked: bool):
        if checked:
            self.content_stack.setCurrentWidget(self.comparison_page)
        else:
            self.content_stack.setCurrentWidget(self.detection_page)
        self._apply_theme()

    def load_existing_result(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih hasil (.shp)", "", "Shapefile (*.shp)")
        if not path:
            return
        try:
            boxes, scores, classes, class_names = load_detection_from_shapefile(
                Path(path))
            result = type("LoadedResult", (), {})()
            result.boxes = boxes
            result.scores = scores
            result.classes = classes
            result.class_names = class_names
            result.shp_path = Path(path)
            self.last_result = result
            if self.raster_path:
                preview = build_preview_bgr(
                    Path(
                        self.raster_path),
                    boxes,
                    scores,
                    1.0,
                    99.0,
                    classes=classes)
                self.canvas.show_bgr_image(preview)
                self.canvas.show_result(result, class_names=class_names)
            self.card_detections.set_value(str(len(boxes)))
        except Exception as exc:
            QMessageBox.warning(self, "Gagal", f"Tidak dapat membaca:\n{exc}")

    def start_inference(self):
        self._persist_current_settings()
        self.run_btn.setEnabled(False)
        self.act_run.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.act_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.stage_stepper.reset()
        self.log_console.clear()

        conf = self.conf_slider.value() / 100.0
        tile_size = self.tile_spin.value()
        overlap = self.overlap_spin.value()
        batch_size = self.batch_spin.value()
        output_dir = self.output_dir_edit.text().strip() or None
        out_name = self.output_name_edit.text().strip() or None
        force_cpu = self.settings.value("force_cpu", False, type=bool)

        self.thread = QThread()
        self.worker = InferenceWorker(
            self.model_path, self.stats_path, self.raster_path,
            conf, tile_size, overlap, batch_size,
            output_dir=output_dir, out_name=out_name, force_cpu=force_cpu,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.log_console.log)
        self.worker.progress.connect(lambda cur, tot: self.progress_bar.setValue(
            int(cur / tot * 100) if tot else 0))
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

    def on_finished(self, result):
        self.run_btn.setEnabled(True)
        self.act_run.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.act_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.last_result = result
        self.card_detections.set_value(str(len(result.boxes)))
        if result.preview_bgr is not None:
            self.canvas.show_bgr_image(result.preview_bgr)
            self.canvas.show_result(
                result, getattr(
                    result, "class_names", None))

    def on_failed(self, error_msg: str):
        self.run_btn.setEnabled(True)
        self.act_run.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.act_stop.setEnabled(False)
        if error_msg != "__cancelled__":
            QMessageBox.critical(self, "Error", f"Gagal:\n\n{error_msg}")

    def _update_gpu_card(self):
        try:
            if torch.cuda.is_available():
                self.card_gpu.set_value(torch.cuda.get_device_name(0))
            else:
                self.card_gpu.set_value("CPU sahaja")
        except Exception:
            self.card_gpu.set_value("-")

    def clear_all(self):
        self.model_path = None
        self.stats_path = None
        self.raster_path = None
        self.last_result = None
        self.model_edit.clear()
        self.stats_edit.clear()
        self.raster_edit.clear()
        self.canvas._placeholder()
        self.card_detections.set_value("-")
        self._update_run_enabled()

    def export_image(self, fmt: str):
        if not self.canvas.has_image():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan gambar", f"hasil.{fmt}", f"Image (*.{fmt})")
        if path:
            self.canvas.save_current_image(path)

    def export_shapefile_copy(self):
        if not self.last_result or not getattr(self.last_result, "shp_path", None):
            QMessageBox.warning(self, "Peringatan", "Belum ada hasil shapefile untuk disalin.")
            return
        
        src_shp = Path(self.last_result.shp_path)
        if not src_shp.is_file():
            QMessageBox.warning(self, "Peringatan", f"File Shapefile asal tidak ditemukan: {src_shp}")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Salin Shapefile ke", src_shp.name, "Shapefile (*.shp)"
        )
        if not path:
            return
            
        try:
            dest_path = Path(path)
            dest_dir = dest_path.parent
            dest_stem = dest_path.stem
            
            src_dir = src_shp.parent
            src_stem = src_shp.stem
            
            # Copy all related shapefile components
            copied = []
            for ext in (".shp", ".shx", ".dbf", ".prj"):
                s_file = src_dir / f"{src_stem}{ext}"
                d_file = dest_dir / f"{dest_stem}{ext}"
                if s_file.is_file():
                    shutil.copy2(s_file, d_file)
                    copied.append(ext)
                    
            self.statusBar().showMessage(f"Berhasil menyalin Shapefile ({', '.join(copied)}) ke {dest_path}")
            QMessageBox.information(self, "Sukses", f"Shapefile berhasil disalin ke:\n{dest_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menyalin Shapefile:\n{e}")

    def export_geojson(self):
        if not self.last_result or len(self.last_result.boxes) == 0:
            QMessageBox.warning(self, "Peringatan", "Belum ada hasil deteksi untuk diekspor.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Ekspor GeoJSON", "deteksi.geojson", "GeoJSON (*.geojson)"
        )
        if not path:
            return
            
        try:
            with rasterio.open(self.raster_path) as src:
                raster_transform = src.transform
                epsg = src.crs.to_epsg() if src.crs else None

            features = []
            class_names = getattr(self.last_result, "class_names", None)
            for i, (box, score, cls) in enumerate(zip(self.last_result.boxes, self.last_result.scores, self.last_result.classes), start=1):
                x1_px, y1_px, x2_px, y2_px = box
                x1_geo, y1_geo = rasterio.transform.xy(raster_transform, y1_px, x1_px)
                x2_geo, y2_geo = rasterio.transform.xy(raster_transform, y2_px, x2_px)
                
                poly_coords = [[
                    [x1_geo, y1_geo],
                    [x2_geo, y1_geo],
                    [x2_geo, y2_geo],
                    [x1_geo, y2_geo],
                    [x1_geo, y1_geo]
                ]]
                
                class_name = resolve_class_name(int(cls), class_names)
                
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords
                    },
                    "properties": {
                        "id": i,
                        "kelas": class_name,
                        "confidence": round(float(score), 4),
                        "x1_px": round(float(x1_px), 1),
                        "y1_px": round(float(y1_px), 1),
                        "x2_px": round(float(x2_px), 1),
                        "y2_px": round(float(y2_px), 1)
                    }
                }
                features.append(feat)

            fc = {
                "type": "FeatureCollection",
                "features": features
            }
            if epsg:
                fc["crs"] = {
                    "type": "name",
                    "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}
                }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(fc, f, indent=2)
                
            self.statusBar().showMessage(f"Berhasil mengekspor GeoJSON ke {path}")
            QMessageBox.information(self, "Sukses", f"GeoJSON berhasil disimpan di:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal mengekspor GeoJSON:\n{e}")

    def export_csv(self):
        if not self.last_result or len(self.last_result.boxes) == 0:
            QMessageBox.warning(self, "Peringatan", "Belum ada hasil deteksi untuk diekspor.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Ekspor CSV", "deteksi.csv", "CSV (*.csv)"
        )
        if not path:
            return
            
        try:
            with rasterio.open(self.raster_path) as src:
                raster_transform = src.transform

            class_names = getattr(self.last_result, "class_names", None)
            
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "kelas", "confidence", "x1_px", "y1_px", "x2_px", "y2_px", "x1_geo", "y1_geo", "x2_geo", "y2_geo"])
                
                for i, (box, score, cls) in enumerate(zip(self.last_result.boxes, self.last_result.scores, self.last_result.classes), start=1):
                    x1_px, y1_px, x2_px, y2_px = box
                    x1_geo, y1_geo = rasterio.transform.xy(raster_transform, y1_px, x1_px)
                    x2_geo, y2_geo = rasterio.transform.xy(raster_transform, y2_px, x2_px)
                    class_name = resolve_class_name(int(cls), class_names)
                    
                    writer.writerow([
                        i, class_name, round(float(score), 4),
                        round(float(x1_px), 1), round(float(y1_px), 1),
                        round(float(x2_px), 1), round(float(y2_px), 1),
                        round(x1_geo, 6), round(y1_geo, 6),
                        round(x2_geo, 6), round(y2_geo, 6)
                    ])
                    
            self.statusBar().showMessage(f"Berhasil mengekspor CSV ke {path}")
            QMessageBox.information(self, "Sukses", f"CSV berhasil disimpan di:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal mengekspor CSV:\n{e}")

    def export_centroid_geojson(self):
        if not self.last_result or len(self.last_result.boxes) == 0:
            QMessageBox.warning(self, "Peringatan", "Belum ada hasil deteksi untuk diekspor.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Ekspor Centroid GeoJSON", "centroid_deteksi.geojson", "GeoJSON (*.geojson)"
        )
        if not path:
            return
            
        try:
            with rasterio.open(self.raster_path) as src:
                raster_transform = src.transform
                epsg = src.crs.to_epsg() if src.crs else None

            features = []
            class_names = getattr(self.last_result, "class_names", None)
            for i, (box, score, cls) in enumerate(zip(self.last_result.boxes, self.last_result.scores, self.last_result.classes), start=1):
                x1_px, y1_px, x2_px, y2_px = box
                cx_px = (x1_px + x2_px) / 2.0
                cy_px = (y1_px + y2_px) / 2.0
                cx_geo, cy_geo = rasterio.transform.xy(raster_transform, cy_px, cx_px)
                
                class_name = resolve_class_name(int(cls), class_names)
                
                feat = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [cx_geo, cy_geo]
                    },
                    "properties": {
                        "id": i,
                        "kelas": class_name,
                        "confidence": round(float(score), 4),
                        "cx_px": round(float(cx_px), 1),
                        "cy_px": round(float(cy_px), 1)
                    }
                }
                features.append(feat)

            fc = {
                "type": "FeatureCollection",
                "features": features
            }
            if epsg:
                fc["crs"] = {
                    "type": "name",
                    "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}
                }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(fc, f, indent=2)
                
            self.statusBar().showMessage(f"Berhasil mengekspor Centroid GeoJSON ke {path}")
            QMessageBox.information(self, "Sukses", f"Centroid GeoJSON berhasil disimpan di:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal mengekspor Centroid GeoJSON:\n{e}")

    def export_centroid_csv(self):
        if not self.last_result or len(self.last_result.boxes) == 0:
            QMessageBox.warning(self, "Peringatan", "Belum ada hasil deteksi untuk diekspor.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Ekspor Centroid CSV", "centroid_deteksi.csv", "CSV (*.csv)"
        )
        if not path:
            return
            
        try:
            with rasterio.open(self.raster_path) as src:
                raster_transform = src.transform

            class_names = getattr(self.last_result, "class_names", None)
            
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "kelas", "confidence", "cx_px", "cy_px", "cx_geo", "cy_geo"])
                
                for i, (box, score, cls) in enumerate(zip(self.last_result.boxes, self.last_result.scores, self.last_result.classes), start=1):
                    x1_px, y1_px, x2_px, y2_px = box
                    cx_px = (x1_px + x2_px) / 2.0
                    cy_px = (y1_px + y2_px) / 2.0
                    cx_geo, cy_geo = rasterio.transform.xy(raster_transform, cy_px, cx_px)
                    class_name = resolve_class_name(int(cls), class_names)
                    
                    writer.writerow([
                        i, class_name, round(float(score), 4),
                        round(float(cx_px), 1), round(float(cy_px), 1),
                        round(cx_geo, 6), round(cy_geo, 6)
                    ])
                    
            self.statusBar().showMessage(f"Berhasil mengekspor Centroid CSV ke {path}")
            QMessageBox.information(self, "Sukses", f"Centroid CSV berhasil disimpan di:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal mengekspor Centroid CSV:\n{e}")

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
