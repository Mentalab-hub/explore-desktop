"""Implement base model and define custom PySide6 signals used in GUI

Classes:
    BaseModel
    SignalsConatiner
"""
from enum import Enum

from PySide6.QtCore import (
    QModelIndex,
    QObject,
    QThreadPool,
    Signal
)


from exploredesktop.modules.explore_interface import ExploreInterface  # isort: skip


class SignalsContainer(QObject):
    """Class containig signals used in the GUI
    """
    # Impedance related signals
    impedanceChanged = Signal(dict, int)
    btnImpMeasureChanged = Signal(str)
    displayDefaultImp = Signal()

    # Footer related signals
    envInfoChanged = Signal(dict)
    devInfoChanged = Signal(dict)

    # BT related signals
    btnConnectChanged = Signal(str)

    # Connection related signals
    connectionStatus = Signal(Enum)

    # Settings related signals
    activeChanChanged = Signal(dict)
    srChanged = Signal(int)

    # Navigation
    pageChange = Signal(str)

    # Visualization
    ornChanged = Signal(list)
    exgChanged = Signal(list)

    tRangeORNChanged = Signal(float)
    tRangeEXGChanged = Signal(float)
    tAxisORNChanged = Signal(list)
    tAxisEXGChanged = Signal(list)

    updateDataAttributes = Signal(list)
    updateYAxis = Signal()

    restartPlot = Signal()

    mkrAdd = Signal(list)
    mkrPlot = Signal(list)
    replotMkrAdd = Signal(float)
    mkrRemove = Signal(float)

    btDrop = Signal(bool)

    rrPeakPlot = Signal(list)
    rrPeakRemove = Signal(list)
    heartRate = Signal(str)

    plotRR = Signal(list)

    dataSettingsChanged = Signal(QModelIndex)

    recordStart = Signal()
    recordEnd = Signal(float)


class BaseModel(object):
    """Base class for models
    """
    signals = SignalsContainer()
    explorer = ExploreInterface()
    threadpool = QThreadPool()

    def __init__(self) -> None:
        pass

    def get_signals(self):
        """Returns model signals

        Returns:
            SignalsContainer: QObject containing signals used in GUI
        """
        return self.signals

    def get_explorer(self):
        """Returns explore interface

        Returns:
            ExploreInterface: interface class for explore
        """
        return self.explorer
