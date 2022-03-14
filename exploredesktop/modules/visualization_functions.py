import copy
import logging
import time
import warnings

import numpy as np
import pyqtgraph as pg
from exploredesktop.modules.app_functions import AppFunctions
from exploredesktop.modules.app_settings import Settings
from exploredesktop.modules.dialogs import PlotDialog
from explorepy.stream_processor import TOPICS
from explorepy.tools import HeartRateEstimator
from PySide6.QtCore import Slot
from scipy.ndimage.filters import gaussian_filter1d


warnings.simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger("explorepy." + __name__)
NANS = [False, False]  # exg, orn


class VisualizationFunctions(AppFunctions):
    """[summary]

    Args:
        AppFunctions ([type]): [description]
    """
    def __init__(self, ui, explorer, signals):
        super().__init__(ui, explorer)
        self.signal_exg = signals['exg']
        self.signal_mkr = signals['mkr']
        self.signal_orn = signals['orn']

        self._vis_time_offset = None
        self._baseline_corrector = {'MA_length': 1.5 * Settings.EXG_VIS_SRATE,
                                    'baseline': 0}

        self.y_unit = Settings.DEFAULT_SCALE
        self.y_string = '1 mV'
        self.last_t = 0
        self.bt_drop_warning_displayed = False
        self.t_drop = None

        self.line = None
        self.exg_pointer = 0
        self.orig_exg_pointer = 0
        self.mrk_plot = {'t': [], 'code': [], 'line': []}
        self.mrk_replot = {'t': [], 'code': [], 'line': []}
        self.packet_count = 1

        self.lines_orn = [None, None, None]
        self.orn_pointer = 0
        self.orn_plot = {k: np.array([np.NaN] * 200) for k in Settings.ORN_LIST}
        self.t_orn_plot = np.array([np.NaN] * 200)

        self.rr_estimator = None
        self.r_peak = {'t': [], 'r_peak': [], 'points': []}
        self.r_peak_replot = {'t': [], 'r_peak': [], 'points': []}
        self.rr_warning_displayed = False

    #########################
    # Init Functions
    #########################
    def init_plots(self):
        """
        Initialize EXG, ORN and FFT plots
        """
        if self.ui.plot_orn.getItem(0, 0) is not None:
            self.ui.plot_exg.clear()
            self.ui.plot_fft.clear()
            self.ui.plot_orn.clear()
            self.line = None
            self.lines_orn = [None, None, None]

        self.init_plot_exg()
        self.init_plot_orn()
        self.init_plot_fft()

    def init_plot_exg(self):
        """
        Initialize EXG plot
        """
        # Count number of active channels
        n_chan_sp = self.explorer.stream_processor.device_info['adc_mask'].count(1)
        n_chan = list(self.chan_dict.values()).count(1)
        if n_chan != n_chan_sp:
            print('ERROR chan count does not match')

        # Create offsets for each chan line
        self.offsets = np.arange(1, n_chan + 1)[:, np.newaxis].astype(float)

        # Set Background color
        pw = self.ui.plot_exg
        pw.setBackground(Settings.PLOT_BACKGROUND)

        # Disable zoom
        pw.setMouseEnabled(x=False, y=False)

        # Add chan ticks to y axis
        # Left axis
        self.active_chan = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
        pw.setLabel('left', 'Voltage')
        self.add_left_axis_ticks()
        pw.getAxis('left').setWidth(60)
        pw.getAxis('left').setPen(color=(255, 255, 255, 50))
        pw.getAxis('left').setGrid(50)

        # Right axis
        pw.showAxis('right')
        pw.getAxis('right').linkToView(pw.getViewBox())
        self.add_right_axis_ticks()
        pw.getAxis('right').setGrid(200)

        # Add range of time axis
        timescale = self.get_timeScale()
        pw.setRange(yRange=(-0.5, n_chan + 1), xRange=(0, int(timescale)), padding=0.01)
        pw.setLabel('bottom', 'time (s)')

        # Initialize curves for each channel
        self.curve_ch1 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)
        self.curve_ch2 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)
        self.curve_ch3 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)
        self.curve_ch4 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)
        self.curve_ch5 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)
        self.curve_ch6 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)
        self.curve_ch7 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)
        self.curve_ch8 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)  # , skipFiniteCheck=True)

        all_curves_list = [
            self.curve_ch1, self.curve_ch2, self.curve_ch3, self.curve_ch4,
            self.curve_ch5, self.curve_ch6, self.curve_ch7, self.curve_ch8
        ]

        # Add curves to plot only if channel is active
        self.curves_list = []
        for curve, act in zip(all_curves_list, list(self.chan_dict.values())):
            if act == 1:
                pw.addItem(curve)
                self.curves_list.append(curve)

    def init_plot_orn(self):
        """
        Initialize plot ORN
        """
        pw = self.ui.plot_orn

        # Set Background color
        pw.setBackground(Settings.PLOT_BACKGROUND)

        # Add subplots
        self.plot_acc = pw.addPlot()
        pw.nextRow()
        self.plot_gyro = pw.addPlot()
        pw.nextRow()
        self.plot_mag = pw.addPlot()

        self.plots_orn_list = [self.plot_acc, self.plot_gyro, self.plot_mag]

        # Link all plots to bottom axis
        self.plot_acc.setXLink(self.plot_mag)
        self.plot_gyro.setXLink(self.plot_mag)

        # Remove x axis in upper plots
        self.plot_acc.getAxis('bottom').setStyle(showValues=False)
        self.plot_gyro.getAxis('bottom').setStyle(showValues=False)

        # Add legend, axis label and grid to all the plots
        timescale = int(self.get_timeScale())
        for plt, lbl in zip(self.plots_orn_list, ['Acc [mg/LSB]', 'Gyro [mdps/LSB]', 'Mag [mgauss/LSB]']):
            # plt.addLegend(horSpacing=20, colCount=3, brush='k', offset=(0, -125))
            plt.addLegend(horSpacing=20, colCount=3, brush='k', offset=(0, 0))
            plt.getAxis('left').setWidth(80)
            plt.getAxis('left').setLabel(lbl)
            plt.showGrid(x=True, y=True, alpha=0.5)
            plt.setXRange(0, timescale, padding=0.01)
            plt.setMouseEnabled(x=False, y=False)

        # Initialize curves for each plot
        self.curve_ax = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[0], name=' accX ')
        self.curve_ay = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[1], name=' accY ')
        self.curve_az = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[2], name=' accZ ')
        self.plot_acc.addItem(self.curve_ax)
        self.plot_acc.addItem(self.curve_ay)
        self.plot_acc.addItem(self.curve_az)

        self.curve_gx = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[0], name='gyroX')
        self.curve_gy = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[1], name='gyroY')
        self.curve_gz = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[2], name='gyroZ')
        self.plot_gyro.addItem(self.curve_gx)
        self.plot_gyro.addItem(self.curve_gy)
        self.plot_gyro.addItem(self.curve_gz)

        self.curve_mx = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[0], name='magX ')
        self.curve_my = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[1], name='magY ')
        self.curve_mz = pg.PlotCurveItem(pen=Settings.ORN_LINE_COLORS[2], name='magZ ')
        self.plot_mag.addItem(self.curve_mx)
        self.plot_mag.addItem(self.curve_my)
        self.plot_mag.addItem(self.curve_mz)

    def init_plot_fft(self):
        """
        Initialize FFT plot
        """
        pw = self.ui.plot_fft
        pw.setBackground(Settings.PLOT_BACKGROUND)
        # pw.setXRange(0, 70, padding=0.01)
        pw.showGrid(x=True, y=True, alpha=0.5)
        pw.addLegend(horSpacing=20, colCount=2, brush='k', offset=(0, -300))
        pw.setLabel('left', 'Amplitude (uV)')
        pw.setLabel('bottom', 'Frequency (Hz)')
        pw.setLogMode(x=False, y=True)
        pw.setMouseEnabled(x=False, y=False)
        self.curve_fft_ch1 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[0], name='ch1', skipFiniteCheck=True)
        self.curve_fft_ch2 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[1], name='ch2', skipFiniteCheck=True)
        self.curve_fft_ch3 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[2], name='ch3', skipFiniteCheck=True)
        self.curve_fft_ch4 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[3], name='ch4', skipFiniteCheck=True)
        self.curve_fft_ch5 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[4], name='ch5', skipFiniteCheck=True)
        self.curve_fft_ch6 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[5], name='ch6', skipFiniteCheck=True)
        self.curve_fft_ch7 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[6], name='ch7', skipFiniteCheck=True)
        self.curve_fft_ch8 = pw.getPlotItem().plot(pen=Settings.FFT_LINE_COLORS[7], name='ch8', skipFiniteCheck=True)

        all_curves_fft_list = [
            self.curve_fft_ch1, self.curve_fft_ch2, self.curve_fft_ch3, self.curve_fft_ch4,
            self.curve_fft_ch5, self.curve_fft_ch6, self.curve_fft_ch7, self.curve_fft_ch8
        ]

        # Add curves to plot only if channel is active
        self.curves_fft_list = []
        for curve, act in zip(all_curves_fft_list, list(self.chan_dict.values())):
            if act == 1:
                pw.addItem(curve)
                self.curves_fft_list.append(curve)
            else:
                pw.removeItem(curve)

    #########################
    # Emit Functions
    #########################
    def emit_signals(self):
        """
        Emit EXG, Marker and ORN signals
        """
        self.emit_orn()
        self.emit_exg()
        self.emit_marker()

    def add_original_exg(self, orig_exg):

        first_chan = list(self.exg_plot_data[2].keys())[0]
        n_new_points = len(orig_exg[list(orig_exg.keys())[0]])

        idxs = np.arange(self.orig_exg_pointer, self.orig_exg_pointer + n_new_points)

        for chan in self.exg_plot_data[2].keys():
            try:
                chan_data = orig_exg[chan]
            except KeyError:
                chan_data = np.array([np.NaN for i in range(n_new_points)])
            self.exg_plot_data[2][chan].put(idxs, chan_data, mode='wrap')

        self.orig_exg_pointer += n_new_points
        if self.orig_exg_pointer >= len(self.exg_plot_data[2][first_chan]):
            while self.orig_exg_pointer >= len(self.exg_plot_data[2][first_chan]):
                self.orig_exg_pointer -= len(self.exg_plot_data[2][first_chan])

    def emit_exg(self, stop=False):
        """
        Get EXG data from packet and emit signal
        """

        stream_processor = self.explorer.stream_processor

        def callback(packet):
            self.chan_list = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
            exg_fs = stream_processor.device_info['sampling_rate']
            timestamp, exg = packet.get_data(exg_fs)

            # Original data
            orig_exg = dict(zip(self.active_chan, exg))
            self.add_original_exg(orig_exg)

            # From timestamp to seconds
            if self._vis_time_offset is not None and timestamp[0] < self._vis_time_offset:
                self.reset_vis_vars()
                new_size = self.plot_points()
                self.exg_plot_data[0] = np.array([np.NaN] * new_size)
                self.exg_plot_data[1] = {
                    ch: np.array([np.NaN] * new_size) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
                self.exg_plot_data[2] = {
                    ch: np.array([np.NaN] * self.plot_points(downsampling=False)
                                 ) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}

                t_min = 0
                t_max = t_min + self.get_timeScale()
                self.ui.plot_exg.setXRange(t_min, t_max, padding=0.01)

            if self._vis_time_offset is None:
                self._vis_time_offset = timestamp[0]

            time_vector = timestamp - self._vis_time_offset

            # Downsampling
            if Settings.DOWNSAMPLING:
                # correct packet for 4 chan device
                if len(time_vector) == 33 and self.decide_drop(exg_fs):
                    exg = exg[:, 1:]
                    time_vector = time_vector[1:]
                exg = exg[:, ::int(exg_fs / Settings.EXG_VIS_SRATE)]
                time_vector = time_vector[::int(exg_fs / Settings.EXG_VIS_SRATE)]

            # Baseline correction
            if self.plotting_filters is not None and self.plotting_filters['offset']:
                samples_avg = exg.mean(axis=1)
                if self._baseline_corrector['baseline'] is None:
                    self._baseline_corrector['baseline'] = samples_avg
                else:
                    try:
                        self._baseline_corrector['baseline'] = \
                            self._baseline_corrector['baseline'] - \
                            ((self._baseline_corrector['baseline'] - samples_avg
                              ) / self._baseline_corrector['MA_length'] * exg.shape[1]
                             )
                    except ValueError:
                        self._baseline_corrector['baseline'] = samples_avg

                exg = exg - self._baseline_corrector['baseline'][:, np.newaxis]
            else:
                self._baseline_corrector['baseline'] = None

            # Update ExG unit
            try:
                exg = self.offsets + exg / self.y_unit
                data = dict(zip(self.active_chan, exg))
                data['t'] = time_vector
                self.signal_exg.emit(data)

            except ValueError as error:
                logger.warning("ValueError: %s", str(error))
            except RuntimeError as error:
                logger.warning("RuntimeError: %s", str(error))
            self.packet_count += 1

        if stop:
            stream_processor.unsubscribe(topic=TOPICS.filtered_ExG, callback=callback)
        else:
            stream_processor.subscribe(topic=TOPICS.filtered_ExG, callback=callback)

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

    def emit_orn(self, stop=False):
        """
        Get orientation data and emit signal
        """
        stream_processor = self.explorer.stream_processor

        def callback(packet):
            timestamp, orn_data = packet.get_data()
            if self._vis_time_offset is None:
                self._vis_time_offset = timestamp[0]
            time_vector = list(np.asarray(timestamp) - self._vis_time_offset)

            data = dict(zip(Settings.ORN_LIST, np.array(orn_data)[:, np.newaxis]))
            data['t'] = time_vector
            try:
                self.signal_orn.emit(data)
            except RuntimeError as error:
                logger.warning("RuntimeError: %s", str(error))

        if stop:
            stream_processor.unsubscribe(topic=TOPICS.raw_orn, callback=callback)
        else:
            stream_processor.subscribe(topic=TOPICS.raw_orn, callback=callback)

    def emit_marker(self):
        """
        Get marker data from packet and emit signal
        """
        stream_processor = self.explorer.stream_processor

        def callback(packet):
            timestamp, code = packet.get_data()
            if self._vis_time_offset is None:
                self._vis_time_offset = timestamp[0]
            time_vector = list(np.asarray(timestamp) - self._vis_time_offset)

            data = [time_vector[0], str(code[0])]
            self.signal_mkr.emit(data)

        stream_processor.subscribe(topic=TOPICS.marker, callback=callback)

    #########################
    # Swipping Plot Functions
    #########################
    @Slot(dict)
    def plot_exg(self, data):
        """
        Plot and update exg data
        """
        self.active_chan = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
        n_new_points = len(data['t'])
        idxs = np.arange(self.exg_pointer, self.exg_pointer + n_new_points)

        self.exg_plot_data[0].put(idxs, data['t'], mode='wrap')  # replace values with new points

        if data['t'][0] < self.last_t and self.bt_drop_warning_displayed is False:
            self.bt_drop_warning_displayed = True
            self.t_drop = data['t'][0]
            msg = (
                "The bluetooth connection is unstable. This may affect the ExG visualization."
                "\nPlease read the troubleshooting section of the user manual for more."
            )
            title = "Unstable Bluetooth connection"
            self.display_msg(msg_text=msg, title=title, type="info")

        elif (self.t_drop is not None) and (data['t'][0] > self.last_t) and \
                (data['t'][0] - self.t_drop > 10) and self.bt_drop_warning_displayed is True:
            self.bt_drop_warning_displayed = False

        for chan in self.exg_plot_data[1].keys():
            try:
                chan_data = data[chan]
            except KeyError:
                chan_data = np.array([np.NaN for i in range(n_new_points)])

            self.exg_plot_data[1][chan].put(idxs, chan_data, mode='wrap')

        self.exg_pointer += n_new_points

        self.last_t = data['t'][-1]

        # if wrap happen -> pointer>length:
        if self.exg_pointer >= len(self.exg_plot_data[0]):
            while self.exg_pointer >= len(self.exg_plot_data[0]):
                self.exg_pointer -= len(self.exg_plot_data[0])

            self.exg_plot_data[0][self.exg_pointer:] += self.get_timeScale()

            t_min = self.last_t
            t_max = t_min + self.get_timeScale()
            self.ui.plot_exg.setXRange(t_min, t_max, padding=0.01)

            id_th = np.where(self.exg_plot_data[0] - t_min >= 0.5)[0][0]
            if id_th > 100:
                for chan in self.exg_plot_data[1].keys():
                    self.exg_plot_data[1][chan][:id_th] = np.NaN

            # Remove marker line and replot in the new axis
            for idx_t in range(len(self.mrk_plot['t'])):
                if self.mrk_plot['t'][idx_t] < self.exg_plot_data[0][0]:
                    self.ui.plot_exg.removeItem(self.mrk_plot['line'][idx_t])
                    new_data = [
                        self.mrk_plot['t'][idx_t] + self.get_timeScale(),
                        self.mrk_plot['code'][idx_t]
                    ]
                    self.plot_mkr(new_data, replot=True)

            # Remove rr peaks and replot in new axis
            to_remove = []
            for idx_t in range(len(self.r_peak['t'])):
                if self.r_peak['t'][idx_t] < data['t'][-1]:
                    new_t = self.r_peak['t'][idx_t] + self.get_timeScale()
                    new_point = self.ui.plot_exg.plot([new_t],
                                                      [self.r_peak['r_peak'][idx_t]],
                                                      pen=None,
                                                      symbolBrush=(200, 0, 0, 200),
                                                      symbol='o',
                                                      symbolSize=8)
                    self.r_peak_replot['t'].append(new_t)
                    self.r_peak_replot['r_peak'].append(self.r_peak['r_peak'][idx_t])
                    self.r_peak_replot['points'].append(new_point)

                    self.ui.plot_exg.removeItem(self.r_peak['points'][idx_t])
                    to_remove.append([self.r_peak['t'][idx_t],
                                      self.r_peak['r_peak'][idx_t],
                                      self.r_peak['points'][idx_t]
                                      ])
            for point in to_remove:
                self.r_peak['t'].remove(point[0])
                self.r_peak['r_peak'].remove(point[1])
                self.r_peak['points'].remove(point[2])
                to_remove.remove(point)

        # Position line:
        if self.line is not None:
            self.line.setPos(data['t'][-1])

        else:
            self.line = self.ui.plot_exg.addLine(data['t'][-1], pen='#FF0000')

        # Add nans between new and old data
        if NANS[0]:
            exg_plot_nan = copy.deepcopy(self.exg_plot_data[1])
            for chan in exg_plot_nan.keys():
                exg_plot_nan[chan][self.exg_pointer - 1: self.exg_pointer + 9] = np.NaN

        # Define connection vector for lines
        else:
            connection = np.full(len(self.exg_plot_data[0]), 1)
            connection[self.exg_pointer - 5: self.exg_pointer + 5] = 0
            first_key = list(self.exg_plot_data[1].keys())[0]
            connection[np.argwhere(np.isnan(self.exg_plot_data[1][first_key]))] = 0
            try:
                if id_th > 100:
                    connection[:id_th] = 0
            except UnboundLocalError:
                pass

        # Set t axis
        if np.nanmax(self.exg_plot_data[0]) < self.get_timeScale():
            pass
        else:
            t_ticks = self.exg_plot_data[0].copy()
            t_ticks[self.exg_pointer:] -= self.get_timeScale()
            t_ticks = t_ticks.astype(int)
            l_points = int(len(self.exg_plot_data[0]) / int(self.get_timeScale()))
            vals = self.exg_plot_data[0][::l_points]
            ticks = t_ticks[::l_points]
            self.ui.plot_exg.getAxis('bottom').setTicks([[(t, str(tick)) for t, tick in zip(vals, ticks)]])

        # Paint curves
        for curve, chan in zip(self.curves_list, self.active_chan):
            try:
                if NANS[0]:
                    curve.setData(self.exg_plot_data[0], exg_plot_nan[chan], connect='finite')
                else:
                    curve.setData(self.exg_plot_data[0], self.exg_plot_data[1][chan], connect=connection)
            except KeyError:
                pass

        # Remove reploted markers
        for idx_t in range(len(self.mrk_replot['t'])):
            if self.mrk_replot['t'][idx_t] < data['t'][-1]:
                self.ui.plot_exg.removeItem(self.mrk_replot['line'][idx_t])

        # Remove reploted r_peaks
        to_remove_replot = []
        for idx_t in range(len(self.r_peak_replot['t'])):
            # self.r_peak_replot['t'][idx_t]
            if self.r_peak_replot['t'][idx_t] < data['t'][-1]:
                self.ui.plot_exg.removeItem(self.r_peak_replot['points'][idx_t])
                to_remove_replot.append([self.r_peak_replot['t'][idx_t],
                                         self.r_peak_replot['r_peak'][idx_t],
                                         self.r_peak_replot['points'][idx_t]])
        for point in to_remove_replot:
            self.r_peak_replot['t'].remove(point[0])
            self.r_peak_replot['r_peak'].remove(point[1])
            self.r_peak_replot['points'].remove(point[2])
            to_remove_replot.remove(point)

    @Slot(dict)
    def plot_mkr(self, data, replot=False):
        """
        Plot and update marker data
        """
        t_point, code = data
        if replot is False:
            mrk_dict = self.mrk_plot
            color = Settings.MARKER_LINE_COLOR
        else:
            mrk_dict = self.mrk_replot
            color = Settings.MARKER_LINE_COLOR_ALPHA

        mrk_dict['t'].append(t_point)
        mrk_dict['code'].append(code)
        pen_marker = pg.mkPen(color=color, dash=[4, 4])

        line = self.ui.plot_exg.addLine(t_point, label=code, pen=pen_marker)
        mrk_dict['line'].append(line)

    @Slot(dict)
    def plot_orn(self, data):
        """
        Plot and update ORN data
        """
        n_new_points = len(data['t'])
        idxs = np.arange(self.orn_pointer, self.orn_pointer + n_new_points)

        self.t_orn_plot.put(idxs, data['t'], mode='wrap')  # replace values with new points

        for k in self.orn_plot.keys():
            self.orn_plot[k].put(idxs, data[k], mode='wrap')

        self.orn_pointer += n_new_points

        # if wrap happen -> pointer>length:
        if self.orn_pointer >= len(self.t_orn_plot):
            while self.orn_pointer >= len(self.t_orn_plot):
                self.orn_pointer -= len(self.t_orn_plot)

            self.t_orn_plot[self.orn_pointer:] += self.get_timeScale()

            t_min = np.nanmin(self.t_orn_plot)
            t_max = t_min + self.get_timeScale()
            for plt in self.plots_orn_list:
                plt.setXRange(t_min, t_max, padding=0.01)

        # Position line
        if None in self.lines_orn:
            self.ui.plot_orn.clear()
            self.init_plot_orn()
            for i, plt in enumerate(self.plots_orn_list):
                self.lines_orn[i] = plt.addLine(data['t'][-1], pen='#FF0000')
        else:
            for line in self.lines_orn:
                try:
                    line.setPos(data['t'][-1])
                except RuntimeError:
                    self.lines_orn = [None, None, None]

        # Add nans between new and old data
        if NANS[1]:
            orn_plot_nan = copy.deepcopy(self.orn_plot)
            for k in orn_plot_nan.keys():
                orn_plot_nan[k][self.orn_pointer - 1: self.orn_pointer + 1] = np.NaN

        # Define connection vector for lines
        else:
            connection = np.full(len(self.t_orn_plot), 1)
            connection[self.orn_pointer - 1: self.orn_pointer + 1] = 0

        # Paint curves
        if NANS[1]:
            self.curve_ax.setData(self.t_orn_plot, orn_plot_nan['accX'], connect='finite')
            self.curve_ay.setData(self.t_orn_plot, orn_plot_nan['accY'], connect='finite')
            self.curve_az.setData(self.t_orn_plot, orn_plot_nan['accZ'], connect='finite')
            self.curve_gx.setData(self.t_orn_plot, orn_plot_nan['gyroX'], connect='finite')
            self.curve_gy.setData(self.t_orn_plot, orn_plot_nan['gyroY'], connect='finite')
            self.curve_gz.setData(self.t_orn_plot, orn_plot_nan['gyroZ'], connect='finite')
            self.curve_mx.setData(self.t_orn_plot, orn_plot_nan['magX'], connect='finite')
            self.curve_my.setData(self.t_orn_plot, orn_plot_nan['magY'], connect='finite')
            self.curve_mz.setData(self.t_orn_plot, orn_plot_nan['magZ'], connect='finite')
        else:
            self.curve_ax.setData(self.t_orn_plot, self.orn_plot['accX'], connect=connection)
            self.curve_ay.setData(self.t_orn_plot, self.orn_plot['accY'], connect=connection)
            self.curve_az.setData(self.t_orn_plot, self.orn_plot['accZ'], connect=connection)
            self.curve_gx.setData(self.t_orn_plot, self.orn_plot['gyroX'], connect=connection)
            self.curve_gy.setData(self.t_orn_plot, self.orn_plot['gyroY'], connect=connection)
            self.curve_gz.setData(self.t_orn_plot, self.orn_plot['gyroZ'], connect=connection)
            self.curve_mx.setData(self.t_orn_plot, self.orn_plot['magX'], connect=connection)
            self.curve_my.setData(self.t_orn_plot, self.orn_plot['magY'], connect=connection)
            self.curve_mz.setData(self.t_orn_plot, self.orn_plot['magZ'], connect=connection)

    @Slot()
    def plot_fft(self):
        """
        Plot FFT
        """
        plot_wdgt = self.ui.plot_fft
        # pw.clear()
        plot_wdgt.setXRange(0, 70, padding=0.01)

        exg_fs = self.explorer.stream_processor.device_info['sampling_rate']
        exg_data = np.array([self.exg_plot_data[2][key][~np.isnan(self.exg_plot_data[2][key]
                                                                  )] for key in self.exg_plot_data[2].keys()],
                            dtype=object)
        if (len(exg_data.shape) == 1) or (exg_data.shape[1] < exg_fs * 5):
            return

        fft_content, freq = self.get_fft(exg_data, exg_fs)
        data = dict(zip(self.exg_plot_data[2].keys(), fft_content))
        data['f'] = freq

        for curve, chan in zip(self.curves_fft_list, self.active_chan):
            try:
                curve.setData(data['f'], data[chan])
            except KeyError:
                pass

    #########################
    # Moving Plot Functions
    #########################
    def plot_exg_moving(self, data):

        # max_points = 100
        max_points = AppFunctions._plot_points(self)
        # if len(self.t_exg_plot)>max_points:

        # time_scale = AppFunctions._get_timeScale(self)
        # if len(self.t_exg_plot) and self.t_exg_plot[-1]>time_scale:
        if len(self.t_exg_plot) > max_points:
            # self.plot_ch8.clear()
            # self.curve_ch8 = self.plot_ch8.plot(pen=Settings.EXG_LINE_COLOR)
            new_points = len(data['t'])
            self.t_exg_plot = self.t_exg_plot[new_points:]
            for ch in self.exg_plot.keys():
                self.exg_plot[ch] = self.exg_plot[ch][new_points:]

            # Remove marker line
            for idx_t in range(len(self.mrk_plot['t'])):
                if self.mrk_plot['t'][idx_t] < self.t_exg_plot[0]:
                    '''for i, plt in enumerate(self.plots_list):
                        plt.removeItem(self.mrk_plot['line'][idx_t][i])'''
                    self.ui.plot_exg.removeItem(self.mrk_plot['line'][idx_t])

            # Remove rr peaks
            id2remove = []
            for idx_t in range(len(self.r_peak['t'])):
                if self.r_peak['t'][idx_t][0] < self.t_exg_plot[0]:
                    self.ui.plot_exg.removeItem(self.r_peak['points'][idx_t])
                    id2remove.append(idx_t)
            for idx_t in id2remove:
                self.r_peak['t'].remove(self.r_peak['t'][idx_t])
                self.r_peak['r_peak'].remove(self.r_peak['r_peak'][idx_t])
                self.r_peak['points'].remove(self.r_peak['points'][idx_t])

            # Update axis
            if len(self.t_exg_plot) - max_points > 0:
                extra = int(len(self.t_exg_plot) - max_points)
                self.t_exg_plot = self.t_exg_plot[extra:]
                for ch in self.exg_plot.keys():
                    self.exg_plot[ch] = self.exg_plot[ch][extra:]

        self.t_exg_plot.extend(data['t'])
        for ch in self.exg_plot.keys():
            self.exg_plot[ch].extend(data[ch])

        for curve, ch in zip(self.curves_list, self.active_chan):
            curve.setData(self.t_exg_plot, self.exg_plot[ch])

    def plot_orn_moving(self, data):

        # time_scale = AppFunctions._get_timeScale(self)

        max_points = AppFunctions._plot_points(self) / 7  # / (2*7)
        # if len(self.t_orn_plot) and self.t_orn_plot[-1]>time_scale:
        if len(self.t_orn_plot) > max_points:
            self.t_orn_plot = self.t_orn_plot[1:]
            for k in self.orn_plot.keys():
                self.orn_plot[k] = self.orn_plot[k][1:]
            if len(self.t_orn_plot) - max_points > 0:
                extra = int(len(self.t_orn_plot) - max_points)
                self.t_orn_plot = self.t_orn_plot[extra:]
                for k in self.orn_plot.keys():
                    self.orn_plot[k] = self.orn_plot[k][extra:]

        self.t_orn_plot.extend(data['t'])
        for k in self.orn_plot.keys():
            self.orn_plot[k].extend(data[k])

        self.curve_ax.setData(self.t_orn_plot, self.orn_plot['accX'])
        self.curve_ay.setData(self.t_orn_plot, self.orn_plot['accY'])
        self.curve_az.setData(self.t_orn_plot, self.orn_plot['accZ'])
        self.curve_gx.setData(self.t_orn_plot, self.orn_plot['gyroX'])
        self.curve_gy.setData(self.t_orn_plot, self.orn_plot['gyroY'])
        self.curve_gz.setData(self.t_orn_plot, self.orn_plot['gyroZ'])
        self.curve_mx.setData(self.t_orn_plot, self.orn_plot['magX'])
        self.curve_my.setData(self.t_orn_plot, self.orn_plot['magY'])
        self.curve_mz.setData(self.t_orn_plot, self.orn_plot['magZ'])

    def plot_marker_moving(self, data):
        t, code = data
        self.mrk_plot['t'].append(data[0])
        self.mrk_plot['code'].append(data[1])

        pen_marker = pg.mkPen(color='#7AB904', dash=[4, 4])

        # lines = []
        '''for plt in self.plots_list:
        # plt = self.plot_ch8
            line = self.ui.plot_exg.addLine(t, label=code, pen=pen_marker)
            lines.append(line)'''
        line = self.ui.plot_exg.addLine(t, label=code, pen=pen_marker)
        # lines.append(line)
        # self.mrk_plot['line'].append(lines)
        self.mrk_plot['line'].append(line)

    #########################
    # Functions
    #########################
    @Slot()
    def set_marker(self):
        """
        Get the value for the event code from the GUI and set the event.
        """
        event_code = int(self.ui.value_event_code.text())
        if event_code > 65535 or event_code < 8:
            self.display_msg(msg_text='Marker code value is not valid')
            return
        try:
            self.explorer.set_marker(event_code)
        except ValueError as error:
            self.display_msg(msg_text=str(error))

    @Slot()
    def change_timescale(self):
        """
        Change ExG and ORN plots time scale
        """
        logger.debug("Time scale has been changed to %.0f", self.get_timeScale())
        t_min = self.last_t
        t_max = t_min + self.get_timeScale()
        self.ui.plot_exg.setXRange(t_min, t_max, padding=0.01)
        for plt in self.plots_orn_list:
            plt.setXRange(t_min, t_max, padding=0.01)

        new_size = self.plot_points()
        self.exg_pointer = 0
        self.orig_exg_pointer = 0
        self.exg_plot_data[0] = np.array([np.NaN] * new_size)
        self.exg_plot_data[1] = {
            ch: np.array([np.NaN] * new_size) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
        self.exg_plot_data[2] = {
            ch: np.array([np.NaN] * self.plot_points(downsampling=False)
                         ) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}

        new_size_orn = self.plot_points(orn=True)
        self.orn_pointer = 0
        self.t_orn_plot = np.array([np.NaN] * new_size_orn)
        self.orn_plot = {k: np.array([np.NaN] * new_size_orn) for k in Settings.ORN_LIST}

    @Slot()
    def change_scale(self):
        """
        Change y-axis scale in ExG plot
        """
        old = Settings.SCALE_MENU[self.y_string]
        new = Settings.SCALE_MENU[self.ui.value_yAxis.currentText()]
        logger.debug("ExG scale has been changed from %s to %s", self.y_string, self.ui.value_yAxis.currentText())

        old_unit = 10 ** (-old)
        new_unit = 10 ** (-new)

        self.y_string = self.ui.value_yAxis.currentText()
        self.y_unit = new_unit

        stream_processor = self.explorer.stream_processor
        self.chan_key_list = [Settings.CHAN_LIST[i].lower()
                              for i, mask in enumerate(reversed(stream_processor.device_info['adc_mask'])) if
                              mask == 1]

        for chan, value in self.exg_plot_data[1].items():
            if self.chan_dict[chan] == 1:
                temp_offset = self.offsets[self.chan_key_list.index(chan)]
                self.exg_plot_data[1][chan] = (value - temp_offset) * (old_unit / new_unit) + temp_offset

        # Rescale r_peaks
        self.r_peak['r_peak'] = list((np.array(self.r_peak['r_peak']) - self.offsets[0]
                                      ) * (old_unit / self.y_unit) + self.offsets[0])

        # Remove old rpeaks
        for p in self.r_peak['points']:
            self.ui.plot_exg.removeItem(p)
        self.r_peak['points'] = []

        # plot scaled rpeaks
        for i in range(len(self.r_peak['t'])):
            point = self.ui.plot_exg.plot([self.r_peak['t'][i]],
                                          [self.r_peak['r_peak'][i]],
                                          pen=None,
                                          symbolBrush=(200, 0, 0),
                                          symbol='o',
                                          symbolSize=8)
            self.r_peak['points'].append(point)

        # Rescale replotted rpeaks
        self.r_peak_replot['r_peak'] = list((np.array(self.r_peak_replot['r_peak']) - self.offsets[0]
                                             ) * (old_unit / self.y_unit) + self.offsets[0])

        # Remove old rpeaks
        for p in self.r_peak_replot['points']:
            self.ui.plot_exg.removeItem(p)
        self.r_peak_replot['points'] = []

        # plot scaled rpeaks
        for i in range(len(self.r_peak_replot['t'])):
            point = self.ui.plot_exg.plot([self.r_peak_replot['t'][i]],
                                          [self.r_peak_replot['r_peak'][i]],
                                          pen=None,
                                          symbolBrush=(200, 0, 0, 200),
                                          symbol='o',
                                          symbolSize=8)
            self.r_peak_replot['points'].append(point)

        self.add_left_axis_ticks()

    def add_right_axis_ticks(self):
        """
        Add upper and lower lines delimiting the channels in exg plot
        """
        ticks_right = [(idx + 1.5, '') for idx, _ in enumerate(self.active_chan)]
        ticks_right += [(0.5, '')]

        self.ui.plot_exg.getAxis('right').setTicks([ticks_right])

    def add_left_axis_ticks(self):
        """
        Add central lines and channel name ticks in exg plot
        """
        pw = self.ui.plot_exg
        ticks = [
            (idx + 1, f'{ch}\n' + '(\u00B1' + f'{self.y_string})') for idx, ch in enumerate(self.active_chan)]
        pw.getAxis('left').setTicks([ticks])

    def plot_heart_rate(self):
        """
        Detect R-peaks and update the plot and heart rate
        """

        if self.ui.value_signal.currentText() == 'EEG':
            return

        if 'ch1' not in self.exg_plot_data[2].keys():
            msg = 'Heart rate estimation works only when channel 1 is enabled.'
            logger.warning(msg)
            if self.rr_warning_displayed is False:
                self.display_msg(msg_text=msg, type='info')
                self.rr_warning_displayed = True
            return

        exg_fs = self.explorer.stream_processor.device_info['sampling_rate']

        if self.rr_estimator is None:
            self.rr_estimator = HeartRateEstimator(fs=exg_fs)

        sr = Settings.EXG_VIS_SRATE if Settings.DOWNSAMPLING else self.get_samplingRate
        # last_n_sec
        i = self.exg_pointer - (2 * sr)
        i = i if i >= 0 else 0
        f = self.exg_pointer if i + self.exg_pointer >= (2 * sr) else (2 * sr)
        # f = self.exg_pointer
        ecg_data = (np.array(self.exg_plot_data[1]['ch1'])[i:f] - self.offsets[0]) * self.y_unit
        time_vector = np.array(self.exg_plot_data[0])[i:f]

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
            for i in range(len(self.peaks_time)):
                if self.peaks_time[i] not in self.r_peak['t']:
                    self.r_peak['t'].append(self.peaks_time[i])
                    self.r_peak['r_peak'].append(self.peaks_val[i])

                    point = self.ui.plot_exg.plot([self.peaks_time[i]],
                                                  [self.peaks_val[i]],
                                                  pen=None,
                                                  symbolBrush=(200, 0, 0),
                                                  symbol='o',
                                                  symbolSize=8)

                    self.r_peak['points'].append(point)

        # Update heart rate cell
        estimated_heart_rate = self.rr_estimator.heart_rate
        self.ui.value_heartRate.setText(str(estimated_heart_rate))

    def get_fft(self, exg, s_rate):
        """
        Compute FFT
        Args:
            exg: exg data from ExG packet
            s_rate (int): sampling rate
        """
        n_point = 1024
        exg -= exg.mean(axis=1)[:, np.newaxis]
        freq = s_rate * np.arange(int(n_point / 2)) / n_point
        fft_content = np.fft.fft(exg, n=n_point) / n_point
        fft_content = np.abs(fft_content[:, range(int(n_point / 2))])
        fft_content = gaussian_filter1d(fft_content, 1)
        return fft_content[:, 1:], freq[1:]

    def popup_filters(self):
        """
        Open plot filter dialog and apply filters
        """
        remove = True if self.plotting_filters is not None and AppFunctions.plotting_filters is not None else False
        wait = True if self.plotting_filters is None and AppFunctions.plotting_filters is None else False
        s_rate = self.explorer.stream_processor.device_info['sampling_rate']
        dialog = PlotDialog(sr=s_rate, current_filters=self.plotting_filters)
        filters = dialog.exec()  # returns false if popup is closed/click on cancel
        if filters is False:
            return False
        elif self._check_same_filters(new_filters=filters):
            return True
        else:
            self.plotting_filters = filters
            AppFunctions.plotting_filters = self.plotting_filters
            if remove:
                self.explorer.stream_processor.remove_filters()
            self.apply_filters()
            # self.loading = LoadingScreen()
            if wait:
                time.sleep(1.5)
            return True

    def _check_same_filters(self, new_filters):
        """
        Compare new filters to existing ones

        Args:
            new_filters (dict): filters to compare
        """
        same = True if self.plotting_filters == new_filters else False
        return same

    def reset_vis_vars(self):
        self._vis_time_offset = None
        self._baseline_corrector = {'MA_length': 1.5 * Settings.EXG_VIS_SRATE,
                                    'baseline': 0}

        self.y_unit = Settings.DEFAULT_SCALE
        self.y_string = '1 mV'
        self.ui.value_yAxis.setCurrentText(self.y_string)
        self.last_t = 0
        self.bt_drop_warning_displayed = False
        self.t_drop = None

        self.line = None
        self.exg_pointer = 0
        self.orig_exg_pointer = 0
        self.mrk_plot = {'t': [], 'code': [], 'line': []}
        self.mrk_replot = {'t': [], 'code': [], 'line': []}
        self.packet_count = 1

        self.lines_orn = [None, None, None]
        self.orn_pointer = 0
        self.orn_plot = {k: np.array([np.NaN] * 200) for k in Settings.ORN_LIST}
        self.t_orn_plot = np.array([np.NaN] * 200)

        self.rr_estimator = None
        self.r_peak = {'t': [], 'r_peak': [], 'points': []}
        self.r_peak_replot = {'t': [], 'r_peak': [], 'points': []}
        self.rr_warning_displayed = False

        self.plotting_filters = None

    def _mode_change(self):
        """
        Log mode change (EEG or ECG)
        """
        logger.debug("ExG mode has been changed to %s", self.ui.value_signal.currentText())