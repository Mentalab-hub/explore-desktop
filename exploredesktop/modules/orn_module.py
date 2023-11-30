
import logging
from typing import (
    Optional,
    Tuple
)

import explorepy
import numpy as np
import pyqtgraph as pg
import PySide6
from PySide6.QtCore import Slot


from exploredesktop.modules.app_settings import (  # isort:skip
    DataAttributes,
    GUISettings,
    Stylesheets
)
from exploredesktop.modules.base_data_module import BasePlots, DataContainer   # isort:skip


logger = logging.getLogger("explorepy." + __name__)


ORN_LIST = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'magX', 'magY', 'magZ']


class ORNData(DataContainer):
    """Orientation data model"""

    def __init__(self) -> None:
        super().__init__()
        self.plot_data = {k: np.array([np.NaN] * 200) for k in ORN_LIST}
        self.t_plot_data = np.array([np.NaN] * 200)

        self.signals.updateDataAttributes.connect(self.update_attributes)

    def reset_vars(self):
        """Reset class variables"""
        super().reset_vars()
        self.plot_data = {k: np.array([np.NaN] * 200) for k in ORN_LIST}
        self.t_plot_data = np.array([np.NaN] * 200)
        self.pointer = 0

    def new_t_axis(self, signal: Optional[PySide6.QtCore.Signal] = None) -> None:
        """Update time axis

        Args:
            signal (Optional[PySide6.QtCore.Signal]): Signal to emit when axis is updated. Defaults to None.
        """
        signal = self.signals.tAxisORNChanged
        return super().new_t_axis(signal)

    def update_pointer(self, data: dict, signal: Optional[PySide6.QtCore.Signal] = None, fft=False) -> None:
        """Update pointer

        Args:
            data (dict): ORN data dictionary
            signal (Optional[PySide6.QtCore.Signal]): Signal to emit. Defaults to None.
        """
        signal = self.signals.tRangeORNChanged
        return super().update_pointer(data, signal)

    def update_attributes(self, attributes: list) -> None:
        """Update class attributes

        Args:
            attributes (list): list of attributes to update
        """
        if DataAttributes.ORNPOINTER in attributes:
            self.pointer = 0
        if DataAttributes.ORNDATA in attributes:
            points = self.plot_points(orn=True)
            self.t_plot_data = np.array([np.NaN] * points)
            self.plot_data = {k: np.array([np.NaN] * points) for k in ORN_LIST}

    def callback(self, packet: explorepy.packet.Orientation) -> None:
        """ORN callback"""
        timestamp, orn_data = packet.get_data()
        if DataContainer.vis_time_offset == 0:
            DataContainer.vis_time_offset = timestamp[0]
        time_vector = list(np.asarray(timestamp) - DataContainer.vis_time_offset)

        data = dict(zip(ORN_LIST, np.array(orn_data)[:, np.newaxis]))
        data['t'] = time_vector

        self.insert_new_data(data)
        self.update_pointer(data)
        self.new_t_axis()

        self.emit_orn_data()

    def emit_orn_data(self) -> None:
        """Emit orientation data"""
        try:
            self.signals.ornChanged.emit([self.t_plot_data, self.plot_data])
        # RuntimeError might happen when the app closes
        except RuntimeError as error:
            logger.debug("RuntimeError: %s", str(error))

    def change_timescale(self) -> None:
        """Change plot time scale"""
        self.signals.tRangeORNChanged.emit(DataContainer.last_t)
        self.signals.updateDataAttributes.emit([DataAttributes.ORNPOINTER, DataAttributes.ORNDATA])


class ORNPlot(BasePlots):
    """Orientation data plot
    """

    def __init__(self, ui) -> None:
        super().__init__(ui)
        self.model = ORNData()

        self.plot_acc = None
        self.plot_gyro = None
        self.plot_mag = None

        self.plots_list = [self.plot_acc, self.plot_gyro, self.plot_mag]

        self.lines = [None, None, None]

        self.init_plot()

    def setup_ui_connections(self):
        super().setup_ui_connections()
        self.ui.value_timeScale.currentTextChanged.connect(self.model.change_timescale)

    def reset_vars(self) -> None:
        """Reset attributes"""
        self.plot_acc = None
        self.plot_gyro = None
        self.plot_mag = None

        self.plots_list = [self.plot_acc, self.plot_gyro, self.plot_mag]

        self.lines = [None, None, None]

        self.model.reset_vars()

    def init_plot(self) -> None:
        """Initialize ORN plot"""
        layout_wdgt = self.ui.plot_orn

        if self.ui.plot_orn.getItem(0, 0) is not None:
            layout_wdgt.clear()
            self.lines = [None, None, None]

        # Set Background color
        layout_wdgt.setBackground(Stylesheets.PLOT_BACKGROUND)

        # Add subplots
        self._add_subplots()

        # Link all plots to bottom axis
        self._link_subplots()

        # Add legend, axis label and grid to all the plots
        timescale = self.time_scale
        for plt, lbl in zip(self.plots_list, GUISettings.ORN_LEGEND):
            # plt.addLegend(horSpacing=20, colCount=3, brush='k', offset=(0, -125))
            plt.addLegend(horSpacing=20, colCount=3, brush='k', offset=(0, 0))
            plt.getAxis('left').setWidth(80)
            plt.getAxis('left').setLabel(lbl)
            plt.showGrid(x=True, y=True, alpha=0.5)
            plt.setXRange(0, timescale, padding=0.01)
            plt.setMouseEnabled(x=False, y=False)

        # Initialize curves for each plot
        self._add_acc_curves()
        self._add_gyro_curves()
        self._add_mag_curves()

    def _link_subplots(self) -> None:
        """Link subplots to the bottom one. Only display bottom x axis
        """
        self.plot_acc.setXLink(self.plot_mag)
        self.plot_gyro.setXLink(self.plot_mag)
        # Remove x axis in upper plots
        self.plot_acc.getAxis('bottom').setStyle(showValues=False)
        self.plot_gyro.getAxis('bottom').setStyle(showValues=False)

    def _add_subplots(self) -> None:
        """Add accelerometer, gyroscope and magnetometer subplots to layout widget
        """
        layout_wdgt = self.ui.plot_orn
        self.plot_acc = layout_wdgt.addPlot()
        layout_wdgt.nextRow()
        self.plot_gyro = layout_wdgt.addPlot()
        layout_wdgt.nextRow()
        self.plot_mag = layout_wdgt.addPlot()

        self.plots_list = [self.plot_acc, self.plot_gyro, self.plot_mag]

    def _add_mag_curves(self) -> None:
        """Add magnetometer curves to plot"""
        self.curve_mx = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[0], name='magX ')
        self.curve_my = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[1], name='magY ')
        self.curve_mz = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[2], name='magZ ')
        self.plot_mag.addItem(self.curve_mx)
        self.plot_mag.addItem(self.curve_my)
        self.plot_mag.addItem(self.curve_mz)

    def _add_gyro_curves(self) -> None:
        """Add gyroscope curves to plot"""
        self.curve_gx = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[0], name='gyroX')
        self.curve_gy = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[1], name='gyroY')
        self.curve_gz = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[2], name='gyroZ')
        self.plot_gyro.addItem(self.curve_gx)
        self.plot_gyro.addItem(self.curve_gy)
        self.plot_gyro.addItem(self.curve_gz)

    def _add_acc_curves(self) -> None:
        """Add accelerometer curves to plot"""
        self.curve_ax = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[0], name=' accX ')
        self.curve_ay = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[1], name=' accY ')
        self.curve_az = pg.PlotCurveItem(pen=Stylesheets.ORN_LINE_COLORS[2], name=' accZ ')
        self.plot_acc.addItem(self.curve_ax)
        self.plot_acc.addItem(self.curve_ay)
        self.plot_acc.addItem(self.curve_az)

    @Slot(dict)
    def swipe_plot(self, data: Tuple[np.array, dict]) -> None:
        """plot orientation data

        Args:
            data (Tuple[np.array, dict]): time vector, dictionary with orn data
        """
        t_vector, plot_data = data

        # Reset plot if position line is not properly set
        if None in self.lines:
            self.ui.plot_orn.clear()
            self.init_plot()

        # position line
        self._add_pos_line(t_vector)

        # connection vector
        connection = self._connection_vector(len(t_vector), n_nans=2)

        self.set_curve_data(t_vector, plot_data, connection)

    def set_curve_data(self, t_vector: np.array, plot_data: dict, connection: np.array) -> None:
        """Set orientation data to plot curves

        Args:
            t_vector (np.array): time vector
            plot_data (dict): data to plot
            connection (np.array): connection vector
        """

        try:
            self.curve_ax.setData(t_vector, plot_data['accX'], connect=connection)
            self.curve_ay.setData(t_vector, plot_data['accY'], connect=connection)
            self.curve_az.setData(t_vector, plot_data['accZ'], connect=connection)
            self.curve_gx.setData(t_vector, plot_data['gyroX'], connect=connection)
            self.curve_gy.setData(t_vector, plot_data['gyroY'], connect=connection)
            self.curve_gz.setData(t_vector, plot_data['gyroZ'], connect=connection)
            self.curve_mx.setData(t_vector, plot_data['magX'], connect=connection)
            self.curve_my.setData(t_vector, plot_data['magY'], connect=connection)
            self.curve_mz.setData(t_vector, plot_data['magZ'], connect=connection)
        except ValueError as error:
            # Error coming from pyqtgraph can be ignored
            logger.debug("ValueError: %s" % error)
