import numpy as np
from exploregui.modules import (
    AppFunctions,
    Settings
)
from exploregui.modules.bt_functions import DISABLED_STYLESHEET
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QMessageBox
)


class ConfigFunctions(AppFunctions):
    def __init__(self, ui, explorer, vis_functions):
        super().__init__(ui, explorer)
        self.vis_functions = vis_functions

    @Slot()
    def format_memory(self):
        r"""
        Display a popup asking for confirmation.
        If yes, memory is formatted.
        """

        question = "Are you sure you want to format the memory?"
        response = self.display_msg(msg_text=question, type="question")

        if response == QMessageBox.StandardButton.Yes:
            with self.wait_cursor():
                self.explorer.format_memory()
                pass
            self.display_msg(msg_text="Memory formatted", type="info")
        else:
            return

    # def update_label_calibration(self):
    #     sec = 100
    #     self.ui.ft_label_device_3.setText(f"Calibrating ORN ({sec}s left)")
    #     sec -= 1

    # def timer_calibration(self):
    #     self.timer_cal = QTimer()
    #     print(f"{self.timer_cal=}")
    #     self.timer_cal.setInterval(1000)
    #     self.timer_cal.timeout.connect(lambda: self.update_label_calibration())
    #     self.timer_cal.start()

    @Slot()
    def calibrate_orn(self):
        r"""
        Calibrate the orientation
        """
        lbl = self.ui.ft_label_device_3.text()
        question = ("Do you want to continue with the orientation sensors calibration?\n"
                    "This will overwrite the calibration data if it already exists\n\n"
                    "If yes, you would need to move and rotate the device for 100 seconds\n"
                    )
        # response = QMessageBox.question(self, "Confirmation", question)
        response = self.display_msg(msg_text=question, type="question")

        if response == QMessageBox.StandardButton.Yes:
            # QMessageBox.information(self, "", "Calibrating...\nPlease move and rotate the device")
            self.ui.ft_label_device_3.setText("Calibrating ORN ... ")
            self.ui.ft_label_device_3.repaint()
            with self.wait_cursor():
                self.explorer.calibrate_orn(do_overwrite=True)
                pass
            self.ui.ft_label_device_3.setText(lbl)
            self.ui.ft_label_device_3.repaint()
            self.display_msg(msg_text="Calibration Complete", title="Done", type="info")
        else:
            return

    @Slot()
    def reset_settings(self):
        r"""
        Display a popup asking for confirmation.
        If yes, the settinngs are set to default.
        """

        question = "Are you sure you want to reset your settings?"
        response = self.display_msg(msg_text=question, type="question")

        if response == QMessageBox.StandardButton.Yes:

            with self.wait_cursor():
                self.change_settings(reset=True)
                self.ui.value_sampling_rate.setCurrentText("250")
                for w in self.ui.frame_cb_channels.findChildren(QCheckBox):
                    w.setChecked(True)
                pass

            # self.display_msg(msg_text="Settings reset", type="info")

        else:
            return

    def change_sampling_rate(self, reset=False):
        """
        Change the sampling rate
        """

        sr = self.explorer.stream_processor.device_info['sampling_rate']
        str_value = self.ui.value_sampling_rate.currentText()
        value = int(str_value)
        if reset:
            value = 250
        if int(sr) != value:
            if self.plotting_filters is not None:
                self.check_filters_new_sr()

            print(
                "Old Sampling rate: ",
                self.explorer.stream_processor.device_info['sampling_rate'])

            self.explorer.set_sampling_rate(sampling_rate=value)

            print(
                "New Sampling rate: ",
                self.explorer.stream_processor.device_info['sampling_rate'])
        else:
            print("Same sampling rate")
            return

    def change_active_channels(self, reset=False):
        """
        Read selected checkboxes and set the channel mask of the device
        """

        active_chan = []

        for w in self.ui.frame_cb_channels.findChildren(QCheckBox):
            status = str(1) if w.isChecked() else str(0)
            active_chan.append(status)

        active_chan_int = [int(i) for i in active_chan]
        active_chan = [i for i in reversed(active_chan)]

        if active_chan_int != self.explorer.stream_processor.device_info['adc_mask']:
            if AppFunctions.plotting_filters is not None:
                self.vis_functions._baseline_corrector["baseline"] = None
                self.explorer.stream_processor.remove_filters()

            mask = "".join(active_chan)
            if reset:
                mask = "11111111"
            int_mask = int(mask, 2)
            try:
                self.explorer.set_channels(int_mask)
            except TypeError:
                self.explorer.set_channels(mask)

            n_chan = self.explorer.stream_processor.device_info['adc_mask']
            n_chan = [i for i in reversed(n_chan)]

            self.chan_dict = dict(zip([c.lower() for c in Settings.CHAN_LIST], n_chan))
            AppFunctions.chan_dict = self.chan_dict

            self.vis_functions.offsets = np.arange(1, n_chan.count(1) + 1)[:, np.newaxis].astype(float)
            self.vis_functions._baseline_corrector["baseline"] = None
            if self.plotting_filters is not None:
                self.apply_filters()
            self.init_imp()

        else:
            print("Same channel mask")
            return

    @Slot()
    def change_settings(self, reset=False):
        """
        Apply changes in device settings
        """

        stream_processor = self.explorer.stream_processor

        with self.wait_cursor():
            points = self.plot_points()
            self.exg_plot_data[0] = np.array([np.NaN] * points)
            self.exg_plot_data[1] = {
                ch: np.array([np.NaN] * points) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
            self.exg_plot_data[2] = {
                ch: np.array([np.NaN] * self.plot_points(downsampling=False
                                                         )) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1
            }
            AppFunctions.exg_plot_data = self.exg_plot_data

            self.change_active_channels(reset)
            self.change_sampling_rate(reset)

            self.exg_plot_data[0] = np.array([np.NaN] * points)
            self.exg_plot_data[1] = {
                ch: np.array([np.NaN] * points) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
            self.exg_plot_data[2] = {
                ch: np.array([np.NaN] * self.plot_points(downsampling=False
                                                         )) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1
            }
            AppFunctions.exg_plot_data = self.exg_plot_data

            pass

        act_chan = ", ".join([ch for ch in self.chan_dict if self.chan_dict[ch] == 1])
        msg = (
            "Device settings have been changed:"
            f"\nSampling Rate: {int(stream_processor.device_info['sampling_rate'])}"
            f"\nActive Channels: {act_chan}"
        )
        self.display_msg(msg_text=msg, type="info")

        self.vis_functions.init_plots()

    def check_filters_new_sr(self):

        if self.plotting_filters is None:
            return

        reapply = False
        # r_value = self.plotting_filters["highpass"]
        # l_value = self.plotting_filters["lowpass"]

        r_value = "" if self.plotting_filters["highpass"] in [None, 'None'] else self.plotting_filters["highpass"]
        l_value = "" if self.plotting_filters["lowpass"] in [None, 'None'] else self.plotting_filters["lowpass"]

        str_value = self.ui.value_sampling_rate.currentText()
        sr = int(str_value)

        nyq_freq = sr / 2.

        max_hc_freq = round(nyq_freq - 1, 2)
        min_lc_freq = round(0.003 * nyq_freq, 2)

        warning = ""

        hc_freq_warning = (
            "High cutoff frequency cannot be larger than or equal to the nyquist frequency.\n"
            f"The high cutoff frequency has changed to {max_hc_freq:.2f} Hz!"
        )

        lc_freq_warning = (
            "Transient band for low cutoff frequency was too narrow.\n"
            f"The low cutoff frequency has changed {min_lc_freq:.2f} Hz!"
        )

        if (l_value != "") and (float(l_value) / nyq_freq <= 0.003):
            warning += lc_freq_warning
            self.plotting_filters["lowpass"] = min_lc_freq
            reapply = True

        if (r_value != "") and (float(r_value) >= nyq_freq):
            warning += hc_freq_warning
            self.plotting_filters["highpass"] = max_hc_freq
            reapply = True

        if reapply:
            self.explorer.stream_processor.remove_filters()
            self.apply_filters()
            AppFunctions.plotting_filters = self.plotting_filters

        if warning != "":
            self.display_msg(msg_text=warning, type="info")

    def enable_settings(self, enable=True):
        if enable is False:
            for w in self.ui.frame_cb_channels.findChildren(QCheckBox):
                w.setEnabled(False)
                # w.setToolTip("Changing channels during visualization is not allowed")
                w.setStyleSheet("color: gray")

            self.ui.value_sampling_rate.setEnabled(False)
            # self.ui.value_sampling_rate.setToolTip(
            #     "Changing the sampling rate during visualization is not allowed")
            self.ui.value_sampling_rate.setStyleSheet("color: gray;\nborder-color: gray;")

            self.ui.btn_apply_settings.setStyleSheet(DISABLED_STYLESHEET)
            self.ui.btn_apply_settings.setEnabled(False)
            self.ui.btn_apply_settings.setToolTip(
                "Changing the settings during recording is not possible")
        else:
            for w in self.ui.frame_cb_channels.findChildren(QCheckBox):
                w.setEnabled(True)
                w.setToolTip("")
                w.setStyleSheet("")

            self.ui.value_sampling_rate.setEnabled(False)
            self.ui.value_sampling_rate.setToolTip("")
            self.ui.value_sampling_rate.setStyleSheet("")

            self.ui.btn_apply_settings.setStyleSheet("")
            self.ui.btn_apply_settings.setEnabled(True)
            self.ui.btn_apply_settings.setToolTip("")
