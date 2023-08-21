"""ExG visualization module"""
import logging

import explorepy
import numpy as np
import pyqtgraph as pg
from explorepy.tools import HeartRateEstimator
from PySide6.QtCore import (
    QTimer,
    Slot
)


from exploredesktop.modules.app_settings import (  # isort:skip
    ConnectionStatus,
    DataAttributes,
    EnvVariables,
    ExGModes,
    Messages,
    Settings,
    Stylesheets,
    VisModes
)
from exploredesktop.modules.base_data_module import (  # isort:skip
    BasePlots,
    DataContainer
)
from exploredesktop.modules.utils import _remove_old_plot_item, display_msg   # isort:skip


logger = logging.getLogger("explorepy." + __name__)

visualization_option = 5
# 1: 9 channels with scroll and lines for min/max. Offsetts are 1
# 2: 19 channels with scroll and line for central channel. Offsets are 0.5
# 3: 19 channels with scroll and line for min/max. Offsets are 0.5
# 4: 32 channels wo scroll and line for central channel. Offsets are 0.5
# 5: 32 channels wo scroll and line for min/max. Offsets are 0.5
# 6: 19 channels scroll and no lines
# 7: 9 channels with scroll and no lines. Offsets are 1


class ExGData(DataContainer):
    """_summary_"""

    def __init__(self, filters) -> None:
        super().__init__()

        self.filters = filters

        self._baseline = None
        self.offsets = np.array([])
        self.y_unit = Settings.DEFAULT_SCALE
        self.y_string = '1 mV'
        # DataContainer.last_t = 0

        self.packet_count = 0
        self.t_bt_drop = None
        self.bt_drop_warning_displayed = False

        self.rr_estimator = None
        self.r_peak = {'t': [], 'r_peak': [], 'points': []}
        self.r_peak_replot = {'t': [], 'r_peak': [], 'points': []}
        self.rr_warning_displayed = False

        self.mode = ExGModes.EEG
        self.vis_mode = VisModes.SCROLL

    def reset_vars(self) -> None:
        """Reset class variables"""
        self._baseline = None
        self.offsets = np.array([])
        self.y_unit = Settings.DEFAULT_SCALE
        self.y_string = '1 mV'
        # DataContainer.last_t = 0

        self.packet_count = 0
        self.t_bt_drop = None
        self.bt_drop_warning_displayed = False

        self.rr_estimator = None
        self.r_peak = {'t': [], 'r_peak': [], 'points': []}
        self.r_peak_replot = {'t': [], 'r_peak': [], 'points': []}
        self.rr_warning_displayed = False

        DataContainer.vis_time_offset = None
        self.pointer = 0

    def new_t_axis(self, signal=None):
        signal = self.signals.tAxisEXGChanged
        return super().new_t_axis(signal)

    def update_pointer(self, data, signal=None, fft=False):
        signal = self.signals.tRangeEXGChanged
        return super().update_pointer(data, signal)

    def on_wrap(self, signal):
        super().on_wrap(signal)
        # if self.mode == ExGModes.ECG:
        #     self.remove_r_peak()
        #     self.add_r_peaks_replot()
        for idx_t in range(len(self.r_peak['t'])):
            if self.r_peak['t'][idx_t] < DataContainer.last_t:
                new_t = self.r_peak['t'][idx_t] + self.timescale
                # self.model.r_peak_replot = self.plot_rr_point(
                #     [new_t, self.model.r_peak['r_peak'][idx_t], True])
                self.signals.plotRR.emit(
                    [new_t, self.r_peak['r_peak'][idx_t], True])

        to_remove = _remove_old_plot_item(
            self.r_peak,
            t_vector=self.t_plot_data[:self.pointer],
            item_type='points')  # , plot_widget=self.ui.plot_exg)
        self.r_peak, to_remove = self.remove_rpeaks(self.r_peak, to_remove)

    @Slot(list)
    def update_attributes(self, attributes: list) -> None:
        """Update class attributes

        Args:
            attributes (list): list of attributes to update
        """
        if DataAttributes.OFFSETS in attributes:

            n_chan = self.explorer.n_active_chan
            # pyqtgraph starts plotting at the bottom, we want to add ch at the top of the plot -> reversed
            # if visualization_option in [2, 3, 4, 5, 6]:
            if self.vis_mode == VisModes.FULL:
                self.offsets = [i for i in reversed(np.arange(0.5, (n_chan + 1) / 2, 0.5)[:, np.newaxis].astype(float))]
            # elif visualization_option in [1, 7]:
            elif self.vis_mode == VisModes.SCROLL:
                self.offsets = [i for i in reversed(np.arange(1, n_chan + 1)[:, np.newaxis].astype(float))]

        if DataAttributes.BASELINE in attributes:
            self._baseline = None

        if DataAttributes.DATA in attributes:
            active_chan = self.explorer.active_chan_list()
            points = self.plot_points()
            self.plot_data = {ch: np.array([np.NaN] * points) for ch in active_chan}
            self.t_plot_data = np.array([np.NaN] * points)

        if DataAttributes.POINTER in attributes:
            self.pointer = 0

    def handle_disconnection(self, timestamp: list) -> None:
        """Handle disconnection errors

        Args:
            timestamp (list): list of timestamps
        """
        if self.vis_time_offset is None or timestamp[0] > self.vis_time_offset:
            return
        self.reset_vars()
        self.signals.updateDataAttributes.emit([DataAttributes.DATA])
        self.signals.tRangeEXGChanged.emit(0)

    def handle_bt_drop(self, data: dict, sec_th: int = 10) -> None:
        """Handle bluetooth drop

        Args:
            data (dict): exg data
            sec_th (int): threshold of seconds to display the warning again. Defaults to 10
        """
        t_point = data['t'][0]
        data_rate = self.explorer.sampling_rate
        allowed_time_period = 2 * np.round(1 / data_rate, 3)
        # check if there is packet drop: we check it by checking the difference between
        # two consecutive timestamps: ideally it should be 1/sps
        sample_diff = np.round(t_point - DataContainer.last_t, 3)
        if self.explorer.is_explore_plus_device() is True:
            is_unstable = sample_diff > allowed_time_period
        else:
            is_unstable = sample_diff <= -1 * allowed_time_period
        if t_point < 0:
            return
        elif is_unstable and self.bt_drop_warning_displayed is False:
            logger.warning(
                "BlueTooth drop:\nt_point={}\nDataContainer.last_t={}\n".format(t_point, DataContainer.last_t))
            self.bt_drop_warning_displayed = True
            self.t_bt_drop = t_point
            # self.signals.btDrop.emit(True)
            self.signals.devInfoChanged.emit({EnvVariables.DEVICE_NAME: ConnectionStatus.UNSTABLE.value})
        elif (self.t_bt_drop is not None) and not is_unstable and \
                (t_point - self.t_bt_drop > sec_th) and self.bt_drop_warning_displayed:
            self.bt_drop_warning_displayed = False
            connection_label = ConnectionStatus.CONNECTED.value.replace("dev_name", self.explorer.device_name)
            self.signals.devInfoChanged.emit({EnvVariables.DEVICE_NAME: connection_label})

    def callback(self, packet: explorepy.packet.EEG) -> None:
        """Callback to get EEG data

        Args:
            packet (explorepy.packet.EEG): EEG packet
        """
        chan_list = self.explorer.active_chan_list()
        exg_fs = self.explorer.sampling_rate
        timestamp, exg = packet.get_data(exg_fs)

        # Remove channels not active
        exg = np.array([e for e, val in zip(exg, self.explorer.chan_mask) if val])

        # self.handle_disconnection(timestamp)
        # From timestamp to seconds
        if DataContainer.vis_time_offset is None:
            DataContainer.vis_time_offset = timestamp[0]

        time_vector = timestamp - DataContainer.vis_time_offset

        # Downsampling
        if Settings.DOWNSAMPLING:
            time_vector, exg = self.downsampling(time_vector, exg, exg_fs)

        # Baseline Correction
        if self.filters.current_filters is not None and self.filters.current_filters['offset']:
            exg = self.baseline_correction(exg)

        # ValueError thrown when changing the channels. Can be ignored
        try:
            exg = self.update_unit(exg)
        except ValueError as error:
            logger.warning("ValueError: %s", str(error))

        # pyqtgraph starts plotting at the bottom, we want to add ch at the top of the plot -> reversed
        data = dict(zip(reversed(chan_list), reversed(exg)))
        data['t'] = time_vector

        self.insert_new_data(data, exg=True)
        self.update_pointer(data)
        self.new_t_axis()
        self.handle_bt_drop(data, sec_th=10)

        DataContainer.last_t = data['t'][-1]
        self.packet_count += 1

        try:
            self.signals.exgChanged.emit([self.t_plot_data, self.plot_data])
        # RuntimeError might happen when the app closes
        except RuntimeError as error:
            logger.debug("RuntimeError: %s", str(error))

    def downsampling(self, time_vector, exg, exg_fs):
        """Downsample"""
        # Correct packet for 4 chan device
        if len(time_vector) == 33 and self.decide_drop(exg_fs):
            exg = exg[:, 1:]
            time_vector = time_vector[1:]

        # Downsample
        exg = exg[:, ::int(exg_fs / Settings.EXG_VIS_SRATE)]
        time_vector = time_vector[::int(exg_fs / Settings.EXG_VIS_SRATE)]
        return time_vector, exg

    def decide_drop(self, exg_fs: int) -> bool:
        """Decide whether to drop a data point from the packet based on the sampling rate

        Args:
            exg_fs (int): sampling rate

        Returns:
            bool: whether to drop a data point
        """
        drop = True
        if exg_fs == 1000 and self.packet_count % 8 == 0:
            drop = False
        elif exg_fs == 500 and self.packet_count % 4 == 0:
            drop = False
        elif exg_fs == 250 and self.packet_count % 2 == 0:
            drop = False
        return drop

    def baseline_correction(self, exg):
        """baseline correction"""
        samples_avg = exg.mean(axis=1)

        if self._baseline is None:
            self._baseline = samples_avg
        else:
            try:
                self._baseline = self._baseline - (
                    (self._baseline - samples_avg) / Settings.BASELINE_MA_LENGTH * exg.shape[1]
                )
            except ValueError:
                self._baseline = samples_avg

        exg = exg - self._baseline[:, np.newaxis]

        return exg

    def update_unit(self, exg):
        """Update exg signal to match selected scale

        Args:
            exg (list): exg signal

        Returns:
            list: scaled exg signal
        """
        exg = self.offsets + exg / self.y_unit
        return exg

    def change_timescale(self):
        """Change time scale"""
        super().change_timescale()
        self.signals.tRangeEXGChanged.emit(DataContainer.last_t)
        self.signals.updateDataAttributes.emit([DataAttributes.POINTER, DataAttributes.DATA])

    @Slot(str)
    def change_scale(self, new_val: str):
        """
        Change y-axis scale in ExG plot
        """
        old = Settings.SCALE_MENU[self.y_string]
        new = Settings.SCALE_MENU[new_val]
        logger.debug("ExG scale has been changed from %s to %s", self.y_string, new_val)

        old_unit = 10 ** (-old)
        new_unit = 10 ** (-new)

        self.y_string = new_val
        self.y_unit = new_unit

        self.rescale_signal(old_unit, new_unit)
        # TODO
        # # Rescale r_peaks
        self.rescale_peaks(old_unit)

        self.r_peak_replot['r_peak'] = list((np.array(self.r_peak_replot['r_peak']) - self.offsets[0]
                                             ) * (old_unit / self.y_unit) + self.offsets[0])
        self.rescale_peaks(old_unit, replot=True)

        # Update acis
        self.signals.updateYAxis.emit()

    def rescale_signal(self, old_unit, new_unit):
        """Rescale signal

        Args:
            old_unit (float): old axis unit
            new_unit (float): new axis unit
        """
        chan_list = self.explorer.active_chan_list()
        for chan, value in self.plot_data.items():
            if chan in chan_list:
                temp_offset = self.offsets[chan_list.index(chan)]
                self.plot_data[chan] = (value - temp_offset) * (old_unit / new_unit) + temp_offset

    def rescale_peaks(self, old_unit, replot=False):
        """Rescale plotted peaks

        Args:
            old_unit (float): old y axis unit
            replot (bool, optional): Whether to rescale replotted peaks. Defaults to False.
        """
        if replot is False:
            r_peak_dict = self.r_peak
        else:
            r_peak_dict = self.r_peak_replot

        r_peak_dict['r_peak'] = list(
            (np.array(r_peak_dict['r_peak']) - self.offsets[0]) * (old_unit / self.y_unit) + self.offsets[0])

        self.signals.rrPeakRemove.emit(r_peak_dict['points'])
        to_replot = r_peak_dict.copy()
        r_peak_dict = {'t': [], 'r_peak': [], 'points': []}
        # # Plot rescaled rpeaks
        for i in range(len(to_replot['t'])):
            self.signals.plotRR.emit(
                [to_replot['t'][i], to_replot['r_peak'][i], replot])

    def obtain_r_peaks(self):
        """Obtain RR peaks"""
        if self.mode == ExGModes.EEG:
            return

        first_chan = list(self.plot_data.keys())[0]

        if self.rr_estimator is None:
            self.rr_estimator = HeartRateEstimator(fs=self.explorer.sampling_rate)

        s_rate = Settings.EXG_VIS_SRATE if Settings.DOWNSAMPLING else self.explorer.sampling_rate
        start = self.pointer - (2 * s_rate)
        start = start if start >= 0 else 0
        end = self.pointer if start + self.pointer >= (2 * s_rate) else (2 * s_rate)
        # f = self.exg_pointer
        ecg_data = (np.array(self.plot_data[first_chan])[start:end] - self.offsets[0]) * self.y_unit
        time_vector = np.array(self.t_plot_data)[start:end]

        # Check if the peak2peak value is bigger than threshold
        if (np.ptp(ecg_data) < Settings.V_TH[0]) or (np.ptp(ecg_data) > Settings.V_TH[1]):
            msg = 'P2P value larger or less than threshold. Cannot compute heart rate!'
            logger.warning(msg)
            return

        try:
            self.peaks_time, self.peaks_val = self.rr_estimator.estimate(ecg_data, time_vector)
        except IndexError:
            return

        self.peaks_val = (np.array(self.peaks_val) / self.y_unit) + self.offsets[0]

        if self.peaks_time:
            for i, pk_time in enumerate(self.peaks_time):
                if pk_time not in self.r_peak['t']:
                    # self.model.r_peak = self.plot_rr_point(pk_time, self.peaks_val[i],)
                    self.signals.plotRR.emit([pk_time, self.peaks_val[i], False])

        self.estimate_heart_rate()
        return self.peaks_time, self.peaks_val

    def estimate_heart_rate(self):
        """Estimate heart rate"""
        estimated_heart_rate = self.rr_estimator.heart_rate
        self.signals.heartRate.emit(str(estimated_heart_rate))

    def remove_rpeaks(self, peaks_dict, to_remove):
        """
        Remove rpeak points specified in the list to_remove.
        Returns original dictionary and list without the removed points

        Args:
            peaks_dict (dict): dict containing the r_peaks.
            to_remove (list): list with the points to remvove.
        """
        for point in to_remove:
            peaks_dict['t'].remove(point[0])
            peaks_dict['r_peak'].remove(point[1])
            peaks_dict['points'].remove(point[2])
            to_remove.remove(point)

        return peaks_dict, to_remove

    @Slot()
    def set_packet_offset(self):
        self.packet_offset = self.packet_count

    @Slot(float)
    def log_n_packets(self, rec_time):
        n_packets = self.packet_count - self.packet_offset
        if self.explorer.device_chan == 4:
            sample_per_packet = 33
        elif self.explorer.device_chan == 8:
            sample_per_packet = 16
        elif self.explorer.device_chan in [16, 32]:
            sample_per_packet = 4

        expected_packets = rec_time * self.explorer.sampling_rate / sample_per_packet
        logger.info("Total number of packets in recording (%f): %i" % (rec_time, n_packets))
        logger.info("Expected number of packets in recording (%f): %i" % (rec_time, expected_packets))

        percentage_recieved = round(n_packets * 100 / expected_packets)
        percentage_recieved = percentage_recieved if percentage_recieved <= 100 else 100
        msg = (
            "Recording complete.\n\n"
            f"{percentage_recieved}% of the expected packets received"
            # f"Recorded packets: {n_packets}\n"
            # f"Estimated expected packets: {int(expected_packets)}\n"
        )
        if expected_packets * 0.95 < n_packets < expected_packets * 1.05:
            # msg += "At least 95% of the expected packets recieved"
            logger.info(msg)
        else:
            logger.info(msg)
            # msg += "Less than 95% of the expected packets recieved"
        display_msg(msg_text=msg, popup_type='info')

    def change_vis_mode(self, mode):
        print(f"new mode: {mode}")
        self.vis_mode = mode
        self.update_attributes([DataAttributes.OFFSETS])


class ExGPlot(BasePlots):
    """_summary_
    """

    def __init__(self, ui, filters) -> None:
        super().__init__(ui)
        self.model = ExGData(filters)

        self.lines = [None]

        self.plots_list = [self.ui.plot_exg]

        self.timer = QTimer()

    def setup_ui_connections(self) -> None:
        """Setup connections between widgets and slots"""

        super().setup_ui_connections()
        self.ui.value_timeScale.currentTextChanged.connect(self.model.change_timescale)
        self.ui.value_yAxis.currentTextChanged.connect(self.model.change_scale)
        # TODO: this will depend on new chan dict
        # self.ui.value_signal.currentTextChanged.connect(self.change_signal_mode)
        # TODO: uncomment when implemented
        # self.ui.cb_antialiasing.stateChanged.connect(self.antialiasing)
        self.ui.verticalScrollBar.valueChanged.connect(self.scroll)

    def setup_scrollbar(self):
        """Add maximum and minimum to explorepy
        """
        # if visualization_option in [1, 7]:
        if self.model.vis_mode == VisModes.SCROLL and self.model.explorer.device_chan > 16:
            self.ui.verticalScrollBar.setMinimum(1)
            self.ui.verticalScrollBar.setMaximum(25)
        elif self.model.vis_mode == VisModes.SCROLL and self.model.explorer.device_chan > 9:
            self.ui.verticalScrollBar.setMinimum(1)
            self.ui.verticalScrollBar.setMaximum(9)
        else:
            self.ui.verticalScrollBar.setMinimum(18)
            self.ui.verticalScrollBar.setMaximum(26)

    def scroll(self):
        """Change the plot range when useing scrollbar
        """
        value = self.ui.verticalScrollBar.value()
        n_chan = self.model.explorer.n_active_chan
        up_lim = (2 - value) + n_chan
        low_lim = up_lim - 9
        self.ui.plot_exg.setYRange(low_lim, up_lim)

    @Slot(bool)
    def antialiasing(self, cb_status):
        """Activate or deactivate antialiasing option

        Args:
            cb_status (bool): status of the checkbox
        """
        if cb_status:
            # TODO add question to confirm
            # msg = "Antialiasing might impact the visualization speed"
            # response = display_msg(msg)
            pg.setConfigOptions(antialias=True)
        else:
            pg.setConfigOptions(antialias=False)

    def reset_vars(self) -> None:
        """Reset variables"""
        self.lines = [None]
        self.plots_list = [self.ui.plot_exg]
        self.bt_drop_warning_displayed = False
        self.ui.value_yAxis.setCurrentText("1 mV")
        self.ui.value_timeScale.setCurrentText("10 s")

        self.model.reset_vars()

    def init_plot(self) -> None:
        """Initialize plot"""
        plot_wdgt = self.ui.plot_exg

        if self.ui.plot_orn.getItem(0, 0) is not None:
            plot_wdgt.clear()
            self.lines = [None]

        # Set Background color
        plot_wdgt.setBackground(Stylesheets.PLOT_BACKGROUND)

        # Disable zoom
        plot_wdgt.setMouseEnabled(x=False, y=False)

        # Add chan ticks to y axis to left and right axis
        self._setup_left_axis(plot_wdgt)
        self._setup_righ_axis(plot_wdgt)

        # Add range of time axis
        self._setup_plot_range(plot_wdgt)

        all_curves_list = [
            pg.PlotCurveItem(pen=Stylesheets.EXG_LINE_COLOR) for i in range(self.model.explorer.device_chan)]
        self.active_curves_list = self.add_active_curves(all_curves_list, plot_wdgt)

        self.setup_scrollbar()
        # if visualization_option in [4, 5] or self.model.explorer.device_chan < 9:
        if self.model.vis_mode == VisModes.FULL or self.model.explorer.device_chan < 9:
            self.ui.verticalScrollBar.setHidden(True)
        else:
            self.ui.verticalScrollBar.setHidden(False)

    def _setup_plot_range(self, plot_wdgt: pg.PlotWidget):
        """Setup time axis"""
        n_chan = self.model.explorer.n_active_chan
        print(f"{n_chan=}")
        timescale = self.time_scale
        value = self.ui.verticalScrollBar.value()

        if self.model.explorer.device_chan < 9:
            y_range = (-0.5, n_chan + 1)
        # elif visualization_option in [2, 3, 6, 7]:
        elif self.model.vis_mode == VisModes.SCROLL:
            up_lim = (2 - value) + n_chan + 0.5
            y_range = (up_lim - 9, up_lim)
        else:
            y_range = (0, 32)

        print(f"{y_range=}")
        plot_wdgt.setRange(
            yRange=y_range,
            xRange=(0, int(timescale)), padding=0.01)
        plot_wdgt.setLabel('bottom', 'time (s)')

    def _setup_righ_axis(self, plot_wdgt: pg.PlotWidget):
        """Setup right axis"""
        plot_wdgt.showAxis('right')
        plot_wdgt.getAxis('right').linkToView(plot_wdgt.getViewBox())
        self.add_right_axis_ticks()
        plot_wdgt.getAxis('right').setGrid(200)

    def _setup_left_axis(self, plot_wdgt: pg.PlotWidget):
        plot_wdgt.setLabel('left', 'Voltage')
        self.add_left_axis_ticks()
        plot_wdgt.getAxis('left').setWidth(60)
        if visualization_option in [1, 2, 4]:
            plot_wdgt.getAxis('left').setPen(color=(255, 255, 255, 50))
            plot_wdgt.getAxis('left').setGrid(50)
        else:
            plot_wdgt.getAxis('left').setPen(color=(255, 255, 255, 50))
            # plot_wdgt.getAxis('left').setGrid(0)

    def add_right_axis_ticks(self) -> None:
        """
        Add upper and lower lines delimiting the channels in exg plot
        """
        active_chan = self.model.explorer.active_chan_list()

        if visualization_option == 1:
            ticks_right = [(idx + 1.5, '') for idx, _ in enumerate(active_chan)]
            ticks_right += [(0.5, '')]
        # elif visualization_option in [2, 4, 6, 7]:
        elif self.model.vis_mode in [VisModes.FULL, VisModes.SCROLL]:
            ticks_right = []
        elif visualization_option in [3, 5]:
            ticks_right = [(i, '') for i in np.arange(0.25, 17, 0.5)]

        self.ui.plot_exg.getAxis('right').setTicks([ticks_right])

    def add_left_axis_ticks(self) -> None:
        """
        Add central lines and channel name ticks in exg plot
        """
        active_chan = self.model.explorer.active_chan_list(custom_name=True)

        if visualization_option in [1]:
            ticks = [
                (
                    idx + 1, f'{ch}\n' + '(\u00B1' + f'{self.model.y_string})'
                ) for idx, ch in enumerate(reversed(active_chan))]

        # elif visualization_option in [7]:
        elif self.model.vis_mode in [VisModes.SCROLL]:
            ticks = [
                (
                    idx + 1, f'{ch}'
                ) for idx, ch in enumerate(reversed(active_chan))]

        else:
            ticks = [
                (
                    idx / 2 + 0.5, f'{ch}'
                    # idx + 1, f'{ch}\n' + '(\u00B1' + f'{self.model.y_string})'
                ) for idx, ch in enumerate(reversed(active_chan))]

        # ticks = [
        #     (
        #         idx / 2 + 0.5, f'{ch}'
        #     ) for idx, ch in enumerate(reversed(active_chan))]

        self.ui.plot_exg.getAxis('left').setTicks([ticks])

    @Slot(dict)
    def swipe_plot(self, data):
        t_vector, plot_data = data

        # position line
        self._add_pos_line(t_vector)

        # connection vector
        connection = self._connection_vector(len(t_vector))

        # Paint curves
        for curve, chan in zip(self.active_curves_list, self.model.explorer.active_chan_list()):
            try:
                curve.setData(t_vector, plot_data[chan], connect=connection)
            # KeyError might happen when (de)activating channels during visualization
            except KeyError:
                pass

        # remove reploted markers
        self.model.signals.mkrRemove.emit(self.model.last_t)

        # Remove reploted r_peaks
        to_remove_replot = _remove_old_plot_item(
            self.model.r_peak_replot,
            t_vector=t_vector[:self.model.pointer],
            item_type='points', plot_widget=self.ui.plot_exg)
        self.model.r_peak_replot, to_remove_replot = self.model.remove_rpeaks(
            self.model.r_peak_replot, to_remove_replot)

    @Slot(bool)
    def display_bt_drop(self, bt_drop: bool) -> None:
        """Display bluetooth drop warning

        Args:
            bt_drop (bool): whether there is a bluetooth drop
        """
        if bt_drop:
            title = "Unstable Bluetooth connection"
            display_msg(msg_text=Messages.BT_DROP, title=title, popup_type="info")

    def plot_rr_point(self, data) -> dict:
        """Plot r peak points

        Args:
            t_point ([type]): [description]
            r_peak ([type]): [description]
            replot (bool, optional): [description]. Defaults to False.

        Returns:
            dict: [description]
        """
        t_point, r_peak, replot = data
        # NOTE should split into add data to dict and plot
        if replot is False:
            brush = (200, 0, 0)
            r_peak_dict = self.model.r_peak
        else:
            brush = (200, 0, 0, 150)
            r_peak_dict = self.model.r_peak_replot

        point = self.ui.plot_exg.plot(
            [t_point], [r_peak], pen=None,
            symbolBrush=brush, symbol='o', symbolSize=8)

        if t_point not in r_peak_dict:
            r_peak_dict['t'].append(t_point)
            r_peak_dict['r_peak'].append(r_peak)
            r_peak_dict['points'].append(point)

        return r_peak_dict

    @Slot(str)
    def change_signal_mode(self, new_mode):
        """
        Log mode change (EEG or ECG)
        """
        logger.debug("ExG mode has been changed to %s", new_mode)
        if new_mode == ExGModes.ECG.value:
            self.model.mode = ExGModes.ECG
            if self.timer.isActive():
                return
            self.timer.setInterval(2000)
            # self.timer.timeout.connect(self.model.add_r_peaks)
            self.timer.timeout.connect(self.model.obtain_r_peaks)
            self.timer.start()

        elif new_mode == ExGModes.EEG.value:
            self.model.mode = ExGModes.EEG
            if self.timer.isActive:
                self.timer.stop()

    @Slot(list)
    def remove_old_r_peak(self, to_remove):
        """Remove old r peaks from plot"""
        plt_widget = self.plots_list[0]
        for point in to_remove:
            plt_widget.removeItem(point)
        # if replot:
        #     peaks_dict = self.model.r_peak_replot
        # else:
        #     peaks_dict = self.model.r_peak

        # to_remove = []
        # for idx_t in range(len(peaks_dict['t'])):
        #     if peaks_dict['t'][idx_t] < self.model.last_t:
        #         for plt_wdgt in self.plots_list:
        #             plt_wdgt.removeItem(peaks_dict['points'][idx_t])
        #             to_remove.append([peaks_dict[key][idx_t] for key in peaks_dict.keys()])

        # if to_remove:
        #     self.model.signals.rrPeakRemove.emit(to_remove)
        # return to_remove
