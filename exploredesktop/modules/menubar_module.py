
import logging
import os
import shutil
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from exploredesktop.modules.app_settings import Stylesheets
from explorepy.tools import (
    compare_recover_from_bin,
    generate_eeglab_dataset
)
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget
)
from scipy import signal


from exploredesktop.modules import (  # isort:skip
    BaseModel
)
from exploredesktop.modules.dialogs import (  # isort:skip
    ConvertBinDialog,
    RepairDataDialog,
    EdfToEeglabDialogue
)
from exploredesktop.modules.utils import (  # isort:skip
    display_msg
)


logger = logging.getLogger("explorepy." + __name__)


class MenuBarActions(BaseModel):
    """Class containing actions triggered by menubar items"""

    def export_eeglab_dataset(self):
        """Export eeglab dataset
        """
        # Select folder containing edf files and make sure it exists
        dialog = EdfToEeglabDialogue()
        data = dialog.exec()
        folder_name = data['bdf_path']
        if folder_name in [False, '']:
            return

        if not os.path.isdir(folder_name):
            display_msg("Directory does not exist. Please select an existing folder")
            return

        self.handle_bdf_conversion(folder_name)

    def handle_bdf_conversion(self, folder_name):
        # Create subfolder to store bdf files
        folder_bdfs = os.path.join(folder_name, "bdf")
        if not os.path.isdir(folder_bdfs):
            os.mkdir(folder_bdfs)
            logger.info("Creating folder %s to store bdf files" % folder_bdfs)
        # Create subfolder to store eeglab dataset files
        folder_datasets = os.path.join(folder_name, "datasets")
        if not os.path.isdir(folder_datasets):
            os.mkdir(folder_datasets)
            logger.info("Creating folder %s to store dataset files" % folder_datasets)
        n_files = 0
        for file in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file)
            # if file is .edf, fist step is to convert to .bdf
            if file_path.endswith(".edf") and os.path.isfile(file_path):
                bdf_file = os.path.splitext(file)[0] + ".bdf"
                dataset_file = os.path.splitext(file)[0] + ".set"
                bdf_path = os.path.join(folder_bdfs, bdf_file)
                dataset_path = os.path.join(folder_datasets, dataset_file)
                shutil.copy2(file_path, bdf_path)  # convert to .bdf
                generate_eeglab_dataset(bdf_path, dataset_path)
                n_files += 1
            # if file is .bdf it can be exported directly
            elif file_path.endswith(".bdf") and os.path.isfile(file_path):
                dataset_file = os.path.splitext(file)[0] + ".set"
                dataset_path = os.path.join(folder_datasets, dataset_file)
                generate_eeglab_dataset(file_path, dataset_path)
                n_files += 1
        # if files were originally bdfs (no conversion needed), remove subfolder
        if len(os.listdir(folder_bdfs)) == 0:
            os.rmdir(folder_bdfs)
        # Display confirmation message with number of exported datasets
        folder_datasets = folder_datasets.replace("/", "\\")
        msg = f"{n_files} datasets exported in folder {folder_datasets}"
        logger.info(msg)
        display_msg(msg, popup_type="info")

    def convert_bin(self) -> None:
        """Convert BIN file to csv/edf
        """
        dialog = ConvertBinDialog()
        data = dialog.exec()
        if data is False:
            return
        self.explorer.convert_bin(
            bin_file=data['bin_path'],
            out_dir=data['dst_folder'],
            file_type=data['file_type'],
            out_dir_is_full=True
        )
        display_msg("Conversion finished", popup_type="info")

    def repair_data(self) -> None:
        """Repair recorded csv file by comparison with the binary file
        """
        dialog = RepairDataDialog()
        data = dialog.exec()

        if data is False:
            return

        folder_path = Path(data['bin_path']).parent.absolute()
        outfile_bin = self._get_filename_repair(data['bin_path'])
        outfile_csv = self._get_filename_repair(data['csv_path'])

        # first step - convert bin to csv
        try:
            self.explorer.convert_bin(
                bin_file=data['bin_path'],
                out_dir=folder_path,
                file_type='csv',
                out_dir_is_full=True
            )
        # if csv file exists, ask user if they want to overwrite
        except FileExistsError:
            msg = (
                "A csv file already exists for the selected .BIN file. Do you want to overwite?"
                # "\nOtherwise, the repare will be done with the existing file"
            )
            response = display_msg(msg_text=msg, popup_type="question")

            if response == QMessageBox.StandardButton.Yes:
                self.explorer.convert_bin(
                    bin_file=data['bin_path'],
                    out_dir=folder_path,
                    file_type='csv',
                    out_dir_is_full=True,
                    do_overwrite=True
                )
            else:
                return

        # repair data
        try:
            compare_recover_from_bin(outfile_csv, outfile_bin)
            display_msg("Repair finished", popup_type="info")
        except Exception as e:
            logger.debug(f"An error occured during csv recovery: {type(e)}: {e}")
            display_msg("An error occured during csv repair")

    @staticmethod
    def _get_filename_repair(path: str) -> str:
        """Return file name for the given path without file extension"""
        _, full_filename = os.path.split(path)
        filename, _ = os.path.splitext(full_filename)
        out_file = os.path.join(Path(path).parent.absolute(), filename)
        out_file = out_file.replace("_ExG", "").replace("_Meta", "").replace("ORN", "").replace("_Marker", "")
        return out_file

    def launch_wiki(self):
        webbrowser.open("https://wiki.mentalab.com/explore-desktop-guide/")

    def recorded_visualization(self):
        filepath = self.explorer.record_filename + "_ExG.csv"
        self.window = PlotWindow(filepath, self.explorer.filters, self.explorer.sampling_rate)
        self.window.show()


class PlotWindow(QMainWindow):
    def __init__(self, file_path, filters, sr):
        super().__init__()
        self.centralWidget = CSVReader(file_path, filters, sr)
        self.setCentralWidget(self.centralWidget)


class CSVReader(QWidget):
    def __init__(self, file_path, filters, sampling_rate):
        super().__init__()

        self.file_path = file_path
        self.filters = filters
        self.sampling_rate = sampling_rate

        self.setup_ui()
        self.style_plot()
        self.plot_data()

        self.btn_refresh.clicked.connect(self.plot_data)

    def setup_ui(self):
        """Add widgets to UI
        """
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.setMouseEnabled(x=True, y=True)  # Disable x-axis mouse interaction
        self.plotWidget.setDownsampling(mode='peak')
        self.plotWidget.setClipToView(True)
        self.viewBox = self.plotWidget.plotItem.getViewBox()
        self.viewBox.setMouseMode(pg.ViewBox.PanMode)

        self.btn_refresh = QPushButton(text="Refresh")

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.btn_refresh)
        self.layout.addWidget(self.plotWidget)

    def style_plot(self):
        """Style plot widget by changing color and adding labels
        """
        self.plotWidget.setBackground(Stylesheets.PLOT_BACKGROUND)
        self.plotWidget.setLabel('bottom', "Time (s)")
        self.plotWidget.setLabel('left', "Amplitude (uV)")

        self.plotWidget.addLegend()

    def plot_data(self):
        self.read_csv()
        self.apply_filters()
        self.apply_offsets()

        for idx, col in enumerate(reversed(list(self.data.columns))):
            if col != "TimeStamp":
                self.plotWidget.plot(
                    self.data["TimeStamp"],
                    self.data[col],
                    pen=Stylesheets.FFT_LINE_COLORS[idx],
                    name=col
                )
        self.plotWidget.setXRange(0, max(self.data["TimeStamp"]))

    def read_csv(self):
        self.plotWidget.clear()
        df = pd.read_csv(self.file_path)
        # substract offset to start at 0
        df['TimeStamp'] -= df.iloc[0, 0]
        self.data = df

    def apply_filters(self):
        notch_freq = self.filters["notch"]
        high_freq = self.filters["high_cutoff"]
        low_freq = self.filters["low_cutoff"]

        for col in list(self.data.columns):
            if col != "TimeStamp":
                if notch_freq is not None:
                    self.data[col] = self.notch_filter(self.data[col], self.sampling_rate, notch_freq)
                if high_freq is not None and low_freq is not None:
                    filter_type = "bandpass"
                elif high_freq is None and low_freq is not None:
                    filter_type = "lowpass"
                elif high_freq is not None and low_freq is None:
                    filter_type = "highpass"
                self.data[col] = self.bp_filter(
                    self.data[col], low_freq, high_freq, self.sampling_rate, filter_type=filter_type)

    def apply_offsets(self):
        offsets = [i for i in reversed(np.arange(0.5, (len(self.data.columns)) / 2, 0.5)[:, np.newaxis].astype(float))]
        for i, offset in enumerate(offsets):
            self.data.iloc[:, i + 1] += offset

    @staticmethod
    def notch_filter(exg, fs, f0):
        Q = 30.0  # Quality factor
        # Design notch filter
        b, a = signal.iirnotch(f0, Q, fs)
        return signal.filtfilt(b, a, exg)

    @staticmethod
    def bp_filter(exg, lf=None, hf=None, fs=250, filter_type='bandpass'):
        N = 4
        if filter_type == "bandpass":
            freq = [lf / fs, hf / fs]
        elif filter_type == "lowpass":
            freq = hf
        elif filter_type == "highpass":
            freq = lf
        b, a = signal.butter(N, freq, filter_type)
        return signal.filtfilt(b, a, exg)
