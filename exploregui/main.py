# This Python file uses the following encoding: utf-8
import logging
import sys

from exploregui import MainWindow
from PySide6.QtWidgets import QApplication
import faulthandler
import cgitb 
import cProfile

from exploregui.modules.app_settings import Settings


def main():
    faulthandler.enable()
    cgitb.enable(format = 'text')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
