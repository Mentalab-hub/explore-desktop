"""Base module for data classes"""
import logging
import math
from abc import abstractmethod
from enum import Enum
from typing import (
    Tuple,
    Union
)

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Slot


from exploredesktop.modules.app_settings import (  # isort: skip
    Settings,
    Stylesheets
)
from exploredesktop.modules.base_model import BaseModel  # isort: skip

logger = logging.getLogger("explorepy." + __name__)


class DataContainer(BaseModel):
    """_summary_
    """
    vis_time_offset = None
    last_t = 0

    def __init__(self) -> None:
        super().__init__()
        self.plot_data = {}
        self.t_plot_data = np.array([])

        self.pointer = 0

        self.timescale = 10

    def reset_vars(self) -> None:
        """Reset class and instance variables values"""
        self.plot_data = {}
        self.t_plot_data = np.array([])

        self.pointer = 0

        self.vis_time_offset = None

        self.timescale = 10
        self.last_t = 0

    @abstractmethod
    def callback(self, packet):
        """callback"""
        raise NotImplementedError

    @abstractmethod
    def update_attributes(self, attributes: list) -> None:
        """update class attributes"""
        raise NotImplementedError

    @staticmethod
    def remove_dict_item(item_dict: dict, item_type: Enum, to_remove: list) -> Tuple[dict, list]:
        """Remove item from a dictionary

        Args:
            item_dict (dict): dictionary with the items to remvoe
            item_type (str): type of item to remove.
            to_remove (list): list with the item to remove

        Returns:
            Tuple[dict, list]: `item_dict` and `to_remove` without the removed items
        """
        # if to_remove is emtpy, return
        if len(to_remove) < 1:
            return item_dict, to_remove

        key = item_type.value[1]
        item_key = item_type.value[0]

        for item in to_remove:
            item_dict['t'].remove(item[0])
            item_dict[key].remove(item[1])
            item_dict[item_key].remove(item[2])
            to_remove.remove(item)

        return item_dict, to_remove

    def change_timescale(self) -> None:
        """Write in log file time scale change
        """
        logger.debug("Time scale has been changed to %.0f", self.timescale)

    def plot_points(self, orn: bool = False, downsampling: bool = Settings.DOWNSAMPLING) -> int:
        """Calculate number of points in the plot vectors

        Args:
            orn (bool, optional): whether the plot is for ORN data. Defaults to False.
            downsampling (bool, optional): whether to apply downsampling. Defaults to Settings.DOWNSAMPLING.

        Returns:
            int: number of points
        """
        time_scale = self.timescale
        s_rate = self.explorer.sampling_rate

        # block below to handle TypeError that may occur on exploredesktop initialization
        if s_rate is None:
            logger.debug("s_rate is None, setting it to 250 (default)")
            s_rate = 250

        if not orn:
            if downsampling:
                points = (time_scale * s_rate) / (s_rate / Settings.EXG_VIS_SRATE)
            else:
                points = (time_scale * s_rate)
        else:
            points = time_scale * Settings.ORN_SRATE

        return int(points)

    @staticmethod
    def get_n_new_points(data: dict) -> int:
        """Get number of new points in vector

        Args:
            data (dict): dictionary with new points

        Returns:
            int: number of points
        """
        # IndexError might happen on disconnect, when some signals are terminated before others
        n_new_points = len(data[list(data.keys())[0]])
        return n_new_points

    def insert_new_data(self, data: dict, fft: bool = False, exg=None):
        """Insert new data into plot vectors

        Args:
            data (dict): data to insert
            fft (bool, optional): whether data is for FFT plot. Defaults to False.
        """
        # The parts commented out correspond to the implementation for handling the bluetooth drop

        # if fft is False and data['t'][0] < DataContainer.last_t:
        #     bt_drop = True
        # else:
        #     bt_drop = False
        bt_drop = False
        n_new_points = self.get_n_new_points(data)
        idxs = np.arange(self.pointer, self.pointer + n_new_points)

        if fft is False:
            # if bt_drop and exg:
            #     print(f"\n{data['t']=}")
            #     print(f"{DataContainer.last_t=}")
            #     a = np.arange(idxs[0] - 10, idxs[-1] + 10)
            #     try:
            #         print(f"Before adding: t={self.t_plot_data[a]}")
            #     except:
            #         pass

            if bt_drop and exg:
                # print(f"\n{data['t']=}")
                # print(f"{DataContainer.last_t=}")
                # print("drop")
                # a = np.arange(idxs[0] - 10, idxs[-1] + 10)
                # try:
                #   print(f"Before adding: t={self.t_plot_data[a]}")
                # except:
                #     pass
                try:
                    i = np.where(self.t_plot_data < data['t'][0])[0][-1] + 1
                    """
                    Different approach. This one relies on inserting the delay packet on the original position
                    idxs = np.arange(i, i+n_new_points)
                    self.t_plot_data = np.insert(self.t_plot_data, idxs, data['t'])
                    self.t_plot_data = self.t_plot_data[:-len(idxs)]"""
                    self.pointer = i
                    idxs = np.arange(self.pointer, self.pointer + n_new_points)
                    # DataContainer.last_t = data['t'][-1]
                except IndexError:
                    print("IndexError - ", np.where(self.t_plot_data < data['t'][0]))

            # Different approach. This one relies on the fact that the time between samples
            # should be constant (0.004 for 8 chan)
            # if len(self.t_plot_data[~np.isnan(self.t_plot_data)])>0 \
            # and data['t'][0] - self.t_plot_data[~np.isnan(self.t_plot_data)][-1] > 0.005:
            #     data['t'] = np.arange(self.t_plot_data[~np.isnan(self.t_plot_data)][-1], data['t'][0], 0.004)
            #     idxs = np.arange(self.pointer, len(data['t']))

            # else:
            self.t_plot_data.put(idxs, data['t'], mode='wrap')  # replace values with new points

            # if exg:
            # if bt_drop and exg:
            #   try:
            #     print(f"After adding: t={self.t_plot_data[a]}")
            #   except:
            #     pass

        for key, val in self.plot_data.items():
            if bt_drop is True:
                val.put(idxs, [np.NaN for i in range(n_new_points)], mode='wrap')
            else:
                try:
                    val.put(idxs, data[key], mode='wrap')
                # KeyError might happen when active chanels are changed
                # if this happens, add nans instead of data coming from packet
                except KeyError:
                    val.put(idxs, [np.NaN for i in range(n_new_points)], mode='wrap')

    def update_pointer(self, data: list, signal=None, fft: bool = False) -> None:
        """Update pointer and emit signal

        Args:
            data (list): data from incoming packet
            signal (PySide6 Signal, optional): Signal to emit when pointer is updated. Defaults to None.
            fft (bool, optional): Whether it is computing FFT. Defaults to False.
        """

        # update pointer by adding number of new points
        self.pointer += self.get_n_new_points(data)

        # pointer is bigger than length at the end of the screen,
        # then restart and emit corresponding signal
        if self.pointer >= len(self.t_plot_data):
            self.pointer -= len(self.t_plot_data)
            if fft is False:
                self.on_wrap(signal)

    def on_wrap(self, signal):
        """Actions to perform when pointer reaches end of the graph

        Args:
            signal (PySide6 Signal): signal to emit.
        """
        # Change time vector to view previous points by adding the time scale value
        self.t_plot_data[self.pointer:] += self.timescale
        # emit signal with smallest time point
        signal.emit(np.nanmin(self.t_plot_data))
        # if replotting markers, uncomment line below (may cause lagging, not completely tested)
        # self.signals.replotMkrAdd.emit(self.t_plot_data[0])

    def new_t_axis(self, signal):
        """
        Update t-axis

        Args:
            t_vector (np.array): time vector
            pointer (int): index with current position in time
        """
        # as long as time vector is smaller than timescale no need to change
        if np.nanmax(self.t_plot_data) < self.timescale:
            return

        # set time ticks based on time vector values
        t_ticks = self.t_plot_data.copy()
        t_ticks[self.pointer:] -= self.timescale
        t_ticks = t_ticks.astype(int)

        # number of points equals to one per second
        l_points = int(len(self.t_plot_data) / int(self.timescale))
        vals = self.t_plot_data[::l_points]
        ticks = t_ticks[::l_points]

        # Emit signal to update ticks
        try:
            signal.emit([vals, ticks])
        # RuntimeError might happen when the app closes
        except RuntimeError as error:
            logger.debug("RuntimeError: %s", str(error))


class BasePlots:
    """Base View class for plots
    """

    def __init__(self, ui) -> None:
        self.ui = ui
        self.model = DataContainer()

        self.lines = []
        self.plots_list = []

        self.set_dropdowns()

    def setup_ui_connections(self):
        """Connect ui elements to corresponding slot"""
        self.ui.value_timeScale.currentTextChanged.connect(self.set_time_scale)

    def get_model(self):
        """Returns data model"""
        return self.model

    @property
    def time_scale(self) -> int:
        """Returns timescale set in GUI
        """
        t_str = self.ui.value_timeScale.currentText()
        t_int = int(Settings.TIME_RANGE_MENU[t_str])
        return t_int

    def set_time_scale(self, value: Union[str, float]) -> None:
        """Set time scale value

        Args:
            value (Union[str, float]): value for timescale. If it is a string in is converted to a float
        """
        if isinstance(value, str):
            value = Settings.TIME_RANGE_MENU[value]
        self.model.timescale = value

    def set_dropdowns(self) -> None:
        """Initialize dropdowns"""
        # Avoid double initialization
        if self.ui.value_yAxis.count() > 0:
            return

        # value_yaxis
        self.ui.value_yAxis.addItems(Settings.SCALE_MENU.keys())
        self.ui.value_yAxis.setCurrentText("1 mV")

        # value_time_scale
        self.ui.value_timeScale.addItems(Settings.TIME_RANGE_MENU.keys())

    @abstractmethod
    def init_plot(self):
        """initialize the plot"""
        raise NotImplementedError

    @abstractmethod
    def swipe_plot(self, data):
        """swipping plot"""
        raise NotImplementedError

    # TODO implement later and have option in gui
    # @abstractmethod
    # def moving_plot(self):
    #     """moving window plot"""
    #     raise NotImplementedError

    def add_active_curves(self, all_curves: list, plot_widget: pg.PlotWidget) -> list:
        """Add curves from a list to a plot widget if the corresponding channel is active

        Args:
            all_curves (list): list of all potential curves
            chan_dict (dict): dictionary with active channels
            plot_widget (pg.PlotWidget): pyqtgraph plotWidget

        Returns:
            list: list of curves added to plot
        """
        # Verify curves and chan dict have the same length, if not reset chan_dict
        chan_dict = self.model.explorer.get_chan_dict_list()
        if len(all_curves) != len(self.model.explorer.active_chan_list()):
            logger.debug(
                "Number of plot curves doesn't match number of active channels. "
                "Updating chan_dict_list from base_data_module")
            self.model.explorer.set_chan_dict_list(self.ui.table_settings.model().chan_data)

        active_curves = []

        # Add curves if the channel is active, else remove it
        for curve, active_state in zip(all_curves, [one_chan_dict['enable'] for one_chan_dict in chan_dict]):
            if active_state == 1:
                plot_widget.addItem(curve)
                active_curves.append(curve)
            else:
                plot_widget.removeItem(curve)
        return active_curves

    @Slot(float)
    def set_t_range(self, data: float) -> None:
        """Set plot x range

        Args:
            data (float): minimum value for x range
        """
        # t_min comes from the signal emitted from the model (on_wrap)
        t_min = data
        t_max = t_min + self.time_scale
        for plt in self.plots_list:
            try:
                plt.setXRange(t_min, t_max, padding=0.01)
            # Exception coming from pyqtgraph library, can be ignored
            except Exception:
                pass

    @Slot(list)
    def set_t_axis(self, data: list) -> None:
        """Set ticks in plot x axis

        Args:
            data (list): values and corresponding ticks
        """
        # values and ticks come from signal emitted from the model (new_t_axis)
        values, ticks = data
        for plt in self.plots_list:
            try:
                plt.getAxis('bottom').setTicks([[(t, str(tick)) for t, tick in zip(values, ticks)]])
            # AttributeError might happen closing the app (signal send but object already desctructed)
            except AttributeError:
                pass

    def _connection_vector(self, length, n_nans=10, id_th=None) -> np.array:
        """Create connection vector to connect old and new data with a gap

        Args:
            length (int): length of the connection vector. Must be the same as the array to plot
            id_th (int): threshold obtained when dev is disconnected
            n_nans (int): number of nans to introduce

        Returns:
            np.array: connection vector with 0s and 1s
        """
        # connection vector contains all ones at the beggining (all points connected)
        connection = np.full(length, 1)

        # a gap is needed around the position line, indicated by the model pointer, so 0s are added
        # the size of the gap is decided in the n_nans input parameter
        connection[self.model.pointer - int(n_nans / 2): self.model.pointer + int(n_nans / 2)] = 0

        # more zeros are added where the plot data is nan (we want to have gaps)
        # this is especially relevant if adding nans when BT drops
        first_key = list(self.model.plot_data.keys())[0]
        idx_nan = np.argwhere(np.isnan(self.model.plot_data[first_key]))
        idx_nan = np.delete(idx_nan, np.where(idx_nan >= [length]))
        connection[idx_nan] = 0
        if id_th is not None and id_th > 100:
            connection[:id_th] = 0

        return connection

    def _add_pos_line(self, t_vector: list) -> list:
        """
        Add position line to plot based on last value in the time vector

        Args:
            lines (list): list of position line

        Returns:
            list: position lines with updated time pos
        """
        # the x coordinate is the last updated point of the time vector
        pos = t_vector[self.model.pointer - 1]
        if math.isnan(pos):
            return self.lines
        # at the beggining lines have to be initialized
        if None in self.lines:
            for idx, plt in enumerate(self.plots_list):
                self.lines[idx] = plt.addLine(pos, pen=Stylesheets.POS_LINE_COLOR)
        # afterwards only the position has to be updated
        else:
            for line in self.lines:
                try:
                    line.setPos(pos)
                # RuntimeError might happen when the app closes/device desconnects - set the lines to default value
                except RuntimeError:
                    self.lines = [None for i in range(len(self.lines))]

        return self.lines

    def remove_old_item(self, item_dict: dict, last_t: np.array, item_type: Enum) -> list:
        """
        Remove line or point element from plot widget

        Args:
            item_dict (dict): dictionary with items to remove
            t_vector (np.array): time vector used as a condition to remove
            item_type (str): specifies item to remove (line or points).
            plot_widget (pyqtgraph PlotWidget): plot widget containing item to remove

        Returns:
            list: list with objects to remove
        """
        assert 't' in item_dict.keys(), 'the items dictionary must have the key \'t\''

        # if there are no lines/points in the dict, return
        if len(item_dict[item_type.value[0]]) == 0:
            return []

        to_remove = []
        for idx_t in range(len(item_dict['t'])):
            # remove the items that are older than the last timepoint acquired
            if item_dict['t'][idx_t] < last_t:
                for plt_wdgt in self.plots_list:
                    for item in item_dict[item_type.value[0]][idx_t]:
                        plt_wdgt.removeItem(item)
                to_remove.append([item_dict[key][idx_t] for key in item_dict.keys()])
                # [item_dict['t'][idx_t], item_dict['r_peak'][idx_t], item_dict['points'][idx_t]])
        return to_remove
