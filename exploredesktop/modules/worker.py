import logging
import sys
import traceback

from PySide6.QtCore import (
    QObject,
    QRunnable,
    Signal
)


logger = logging.getLogger("explorepy." + __name__)


class WorkerSignals(QObject):
    """Defines the signals available from a running worker thread"""
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)


class Worker(QRunnable):
    """Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    Args:
        callback: The function callback to run on this worker thread. Supplied args and
                        kwargs will be passed through to the runner.
        args: Arguments to pass to the callback function
        kwargs: Keywords to pass to the callback function
    """

    def __init__(self, funct, *args, **kwargs) -> None:
        super().__init__()

        self.funct = funct
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._running = True

    def run(self) -> None:
        """Initialise the runner function with passed args, kwargs"""
        try:
            if self._running:
                result = self.funct(*self.args, **self.kwargs)
            self.signals.finished.emit()
        # pylint: disable=bare-except
        except:  # noqa: E722
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
            logger.debug(f"Error in thread {exctype}: {value}")
        else:
            self.signals.result.emit(result)  # Returns the result of the processing
        finally:
            self.signals.finished.emit()  # Done

    def stop(self):
        """Stop the execution of the runner function"""
        self._running = False
