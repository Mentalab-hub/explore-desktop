
import logging

import explorepy
import numpy as np
import pyqtgraph as pg
from pylsl import (
    StreamInlet,
    resolve_stream
)
from PySide6.QtCore import Slot
from PySide6.QtGui import QIntValidator


from exploredesktop.modules.app_settings import (  # isort: skip
    Messages,
    PlotItems,
    Stylesheets
)
from exploredesktop.modules.base_data_module import (  # isort: skip
    BasePlots,
    DataContainer
)
from exploredesktop.modules.utils import display_msg  # isort: skip
from exploredesktop.modules.worker import Worker  # isort: skip


logger = logging.getLogger("explorepy." + __name__)


class MarkerData(DataContainer):
    """Marker data model"""

    def __init__(self) -> None:
        super().__init__()
        self.mrk_plot = {'t': [], 'code': [], 'lines': []}
        self.mrk_replot = {'t': [], 'code': [], 'lines': []}

        self.worker = None
        self.acquire_external_markers = True

    def callback(self, packet: explorepy.packet.EventMarker) -> None:
        """Get marker data from packet and emit signal

        Args:
            packet (explorepy.packet.EventMarker): Event marker packet
        """
        timestamp, code = packet.get_data()
        if DataContainer.vis_time_offset is None:
            DataContainer.vis_time_offset = timestamp[0]
        time_vector = list(np.asarray(timestamp) - DataContainer.vis_time_offset)
        data = [time_vector[0], str(code[0]), False]
        self.signals.mkrAdd.emit(data)

    @Slot(list)
    def add_mkr(self, data: list) -> None:
        """Add marker data to marker dictionary

        Args:
            data (list): list of data containing marker time, code and whether to replot
        """
        t_point, code, replot = data
        if replot is False:
            mrk_dict = self.mrk_plot
        else:
            mrk_dict = self.mrk_replot

        # Add timestamp and code to dictionary, emit signal that will add line
        mrk_dict['t'].append(t_point)
        mrk_dict['code'].append(code)
        self.signals.mkrPlot.emit(data)

    @Slot(float)
    def add_mkr_replot(self, t_thr: float) -> None:
        """Add reploted marker to dictionary

        Args:
            t_thr (float): last time point
        """
        # Currently not in use - replotting markers may lead to lag
        for idx_t in range(len(self.mrk_plot['t'])):
            if self.mrk_plot['t'][idx_t] < t_thr:
                t_point = self.mrk_plot['t'][idx_t] + self.timescale
                code = self.mrk_plot['code'][idx_t]
                self.mrk_replot['t'].append(t_point)
                self.mrk_replot['code'].append(code)
                self.signals.mkrPlot.emit([t_point, code, True])

    def get_lsl_marker(self) -> None:
        """Acquire LSL markers and emit signal to plot
        """
        # NOTE Currently not in use (will be used after proper test of external LSL markers and threading)
        logger.info("looking for a marker stream...")
        streams = resolve_stream('type', 'Markers')
        inlet = StreamInlet(streams[0], processing_flags=1 | 8)
        while self.acquire_external_markers:
            sample, timestamp = inlet.pull_sample()
            if self.acquire_external_markers:
                self.explorer.set_external_marker(timestamp, str(sample[0]))

    def enable_external_markers(self, state: bool) -> None:
        """Enable and disable external marker acquisition

        Args:
            state (bool): whether to acquire
        """
        # NOTE Currently not in use (will be used after proper test of external LSL markers and threading)
        if state:
            self.acquire_external_markers = True
            self.start_lsl_marker_thread()
        else:
            self.acquire_external_markers = False
            self.stop_lsl_marker_thread()

    def start_lsl_marker_thread(self) -> None:
        """Start worker and move to threadpool
        """
        # NOTE Currently not in use (will be used after proper test of external LSL markers and threading)
        self.worker = Worker(self.get_lsl_marker)
        self.threadpool.start(self.worker)

    def stop_lsl_marker_thread(self):
        """Stop LSL marker acquisition
        """
        # NOTE Currently not in use (will be used after proper test of external LSL markers and threading)
        if self.worker is not None:
            logger.info("Stopping LSL marker acquisition")
            self.worker.stop()
            self.threadpool.clear()
            self.threadpool.tryTake(self.worker)


class MarkerPlot(BasePlots):
    """Marker plot"""

    def __init__(self, ui) -> None:
        super().__init__(ui)
        self.model = MarkerData()
        self.plots_list = [
            # self.ui.plot_exg,
            self.ui.plot_orn.getItem(0, 0),
            self.ui.plot_orn.getItem(1, 0),
            self.ui.plot_orn.getItem(2, 0)
        ]
        self.setup_validators()

    def setup_ui_connections(self) -> None:
        """Setup connections between widgets and slots"""
        self.ui.btn_marker.clicked.connect(self.set_marker)
        self.ui.value_event_code.returnPressed.connect(self.set_marker)

    def setup_validators(self) -> None:
        """Setup validators for markers"""
        self.ui.value_event_code.setValidator(QIntValidator(0, 65535))
        self.ui.btn_marker.setEnabled(self.ui.value_event_code.text() != "")
        self.ui.value_event_code.textChanged[str].connect(lambda: self.ui.btn_marker.setEnabled(
            (self.ui.value_event_code.text() != "") or (
                (self.ui.value_event_code.text().isnumeric()) and (8 <= int(self.ui.value_event_code.text())))
        )
        )

    def set_marker(self) -> None:
        """
        Get the value for the event code from the GUI and set the event.
        """
        event_code = int(self.ui.value_event_code.text())
        code_ok = self._verify_code_value(event_code)
        if not code_ok:
            return
        try:
            self.model.explorer.set_marker(event_code)
        except ValueError as error:
            display_msg(msg_text=str(error))

    def _verify_code_value(self, event_code: int) -> bool:
        """Verify that marker code is within limits"""
        code_ok = True
        if event_code > 65535 or event_code < 0:
            display_msg(msg_text=Messages.INVALID_MARKER)
            code_ok = False
        return code_ok

    @Slot(list)
    def plot_marker(self, data):
        """
        Plot and update marker data
        """
        t_point, code, replot = data
        if replot is False:
            mrk_dict = self.model.mrk_plot
            color = Stylesheets.MARKER_LINE_COLOR
        else:
            mrk_dict = self.model.mrk_replot
            color = Stylesheets.MARKER_LINE_COLOR_ALPHA

        pen_marker = pg.mkPen(color=color, dash=[4, 4])

        lines = self._plot_lines(t_point, code, pen_marker)

        mrk_dict['lines'].append(lines)
        if replot:
            idx_remove = self.model.mrk_plot['t'].index([t_point - self.model.timescale])
            for line in self.model.mrk_plot['lines'][idx_remove]:
                self._remove_lines(line)

    def _remove_lines(self, line: pg.InfiniteLine):
        """Remove line from all plots

        Args:
            line (pg.InfiniteLine): line to remove
        """
        self.ui.plot_orn.getItem(0, 0).removeItem(line)
        self.ui.plot_orn.getItem(1, 0).removeItem(line)
        self.ui.plot_orn.getItem(2, 0).removeItem(line)
        self.ui.plot_exg.removeItem(line)

    def _plot_lines(self, t_point: float, code: str, pen_marker: str) -> list:
        """Plot marker lines

        Args:
            t_point (float): time point to add the line
            code (str): marker code
            pen_marker (str): line color

        Returns:
            list: list of plotted lines
        """
        lines = []
        # NOTE uncomment below to add markers to ORN plots. If a lot of markers are set in a
        # short time, the program will freeze
        # line = self.ui.plot_orn.getItem(0, 0).addLine(t_point, label=code, pen=pen_marker)
        # lines.append(line)
        # line = self.ui.plot_orn.getItem(1, 0).addLine(t_point, label=code, pen=pen_marker)
        # lines.append(line)
        # line = self.ui.plot_orn.getItem(2, 0).addLine(t_point, label=code, pen=pen_marker)
        # lines.append(line)
        line = self.ui.plot_exg.addLine(t_point, label=code, pen=pen_marker)
        lines.append(line)
        return lines

    @Slot(float)
    def remove_old_item(self, last_t: float) -> None:
        """Remove old item from the plot

        Args:
            last_t (float): last time point
        """
        item_dict = self.model.mrk_replot
        item_type = PlotItems.VLINES
        to_remove = super().remove_old_item(item_dict, last_t, item_type)
        self.model.mrk_replot, to_remove = self.model.remove_dict_item(item_dict, item_type, to_remove)

    def init_plot(self):
        raise NotImplementedError

    def swipe_plot(self):
        raise NotImplementedError
