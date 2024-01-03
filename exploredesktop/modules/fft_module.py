"""FFT visualization module"""
import logging
from typing import (
    Optional,
    Tuple
)

import explorepy
import numpy as np
from PySide6.QtCore import (
    QTimer,
    Slot
)
from scipy.ndimage.filters import gaussian_filter1d


from exploredesktop.modules.app_settings import (  # isort:skip
    DataAttributes,
    Stylesheets
)
from exploredesktop.modules.base_data_module import (  # isort:skip
    BasePlots,
    DataContainer
)


logger = logging.getLogger("explorepy." + __name__)


class FFTData(DataContainer):
    """FFT data model"""

    def __init__(self) -> None:
        super().__init__()

        self.signals.updateDataAttributes.connect(self.update_attributes)

    @Slot(list)
    def update_attributes(self, attributes: list) -> None:
        """Update class attributes

        Args:
            attributes (list): list of attributes to update
        """
        active_chan = self.explorer.active_chan_list()
        if DataAttributes.DATA in attributes:
            points = self.plot_points(downsampling=False)
            self.plot_data = {ch: np.array([np.NaN] * points) for ch in active_chan}

    @staticmethod
    def get_fft(exg: np.array, s_rate: int) -> Tuple[np.array, np.array]:
        """Compute FFT

        Args:
            exg (np.array): exg data from ExG packet
            s_rate (int): sampling rate

        Returns:
            Tuple[np.array, np.array]: ExG data fourier transform, frequencies
        """
        n_point = 1024
        exg -= exg.mean(axis=1)[:, np.newaxis]
        freq = s_rate * np.arange(int(n_point / 2)) / n_point
        fft_content = np.fft.fft(exg, n=n_point) / n_point
        fft_content = np.abs(fft_content[:, range(int(n_point / 2))])
        fft_content = gaussian_filter1d(fft_content, 1)
        return fft_content[:, 1:], freq[1:]

    def callback(self, packet: explorepy.packet.EEG) -> None:
        """Callback to obtain raw ExG data

        Args:
            packet (explorepy.packet.EEG): EEG packet
        """
        chan_list = self.explorer.active_chan_list()
        exg_fs = self.explorer.sampling_rate

        _, exg = packet.get_data(exg_fs)
        orig_exg = dict(zip(chan_list, exg))
        self.insert_new_data(data=orig_exg, fft=True)
        self.update_pointer(data=orig_exg, fft=True)

    def fft_plot_data(self) -> Optional[dict]:
        """Returns FFT data to plot
        """
        exg_fs = self.explorer.sampling_rate
        exg_data = np.array(
            [self.plot_data[key][~np.isnan(self.plot_data[key])] for key in self.plot_data.keys()], dtype=object)

        if (len(exg_data.shape) == 1) or (exg_data.shape[1] < exg_fs * 5):
            return

        fft_content, freq = self.get_fft(exg_data, exg_fs)
        data = dict(zip(self.plot_data.keys(), fft_content))
        data['f'] = freq

        return data


class FFTPlot(BasePlots):
    """FFT Ploting class"""
    def __init__(self, ui) -> None:
        super().__init__(ui)
        self.model = FFTData()
        self.timer = QTimer()

    def init_plot(self) -> None:
        """Initialize FFT plot"""
        if self.ui.plot_orn.getItem(0, 0) is not None:
            self.ui.plot_fft.clear()

        plot_wdgt = self.ui.plot_fft
        plot_wdgt.setBackground(Stylesheets.PLOT_BACKGROUND)
        col = 2 if self.model.explorer.device_chan <= 8 else 4
        plot_wdgt.addLegend(horSpacing=20, colCount=col, brush='k', offset=(0, -300))
        plot_wdgt.showGrid(x=True, y=True, alpha=0.5)
        plot_wdgt.setLabel('left', 'Amplitude (uV)')
        plot_wdgt.setLabel('bottom', 'Frequency (Hz)')
        plot_wdgt.setLogMode(x=False, y=True)
        plot_wdgt.setMouseEnabled(x=False, y=False)

        active_chan = self.model.explorer.full_chan_list(custom_name=True)

        all_curves_list = [
            plot_wdgt.getPlotItem().plot(
                pen=Stylesheets.FFT_LINE_COLORS[idx], name=f'{ch}', skipFiniteCheck=True
            ) for idx, ch in enumerate(active_chan)
        ]
        self.active_curves_list = self.add_active_curves(all_curves_list, plot_wdgt)

    def plot(self) -> None:
        """Plot FFT data
        """
        plot_wdgt = self.ui.plot_fft
        # NOTE uncomment below to have FFT range = Sampling Rate / 2
        # max_x_range = round(self.model.explorer.sampling_rate / 2)
        max_x_range = 70
        plot_wdgt.setXRange(0, max_x_range, padding=0.01)
        data = self.model.fft_plot_data()
        if data is None:
            return

        for curve, chan in zip(self.active_curves_list, self.model.explorer.active_chan_list()):
            try:
                curve.setData(data['f'], data[chan])
            except KeyError:
                pass

    def reset_vars(self) -> None:
        """Reset timer"""
        if self.timer.isActive():
            self.stop_timer()

    def start_timer(self) -> None:
        """Start plotting timer"""
        if self.timer.isActive():
            return
        # NOTE Change value to control refresh rate (im msec)
        refresh_rate = 500  # msec
        self.timer.setInterval(refresh_rate)
        self.timer.timeout.connect(self.plot)
        self.timer.start()

    def stop_timer(self) -> None:
        """Stop plotting timer"""
        if not self.timer.isActive():
            return
        self.timer.stop()

    def swipe_plot(self, data):
        raise NotImplementedError
