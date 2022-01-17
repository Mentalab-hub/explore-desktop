# This Python file uses the following encoding: utf-8
import sys

from exploregui import MainWindow
from PySide6.QtWidgets import QApplication


def main():
    # faulthandler.enable()
    # cgitb.enable(format = 'text')
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    print(
        "\nPlease do not close this command prompt window."
        "\nIf any error happens, you can use this window to send the report to Mentalab.\n")
    main()
