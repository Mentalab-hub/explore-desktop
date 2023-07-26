"""Conatins implementation for a moving gif.
Not used currently but code can be used in future if desired with small changes
"""
from PySide6.QtCore import (
    Qt,
    QTimer
)
from PySide6.QtGui import QMovie
from PySide6.QtWidgets import (
    QLabel,
    QWidget
)


class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)

        self.label_animation = QLabel(self)

        self.movie = QMovie('images/loading.gif')
        self.label_animation.setMovie(self.movie)

        timer = QTimer(self)
        self.startAnimation()
        timer.singleShot(2000, self.stopAnimation)

        self.show()

    def startAnimation(self):
        self.movie.start()

    def stopAnimation(self):
        self.movie.stop()
        self.close()
