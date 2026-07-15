"""
app.py
Entry point aplikasi. Jalankan langsung: python app.py
Atau build jadi .exe pakai: pyinstaller build.spec
"""
import sys
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow, DARK_QSS


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    app.setApplicationName("Sawit Vision")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
