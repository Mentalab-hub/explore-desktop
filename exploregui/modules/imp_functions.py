from exploregui.modules.app_functions import AppFunctions
from exploregui.modules.app_settings import Settings
from explorepy.stream_processor import TOPICS
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox
)


class IMPFunctions(AppFunctions):
    def __init__(self, ui, explorer, signal_imp):
        super().__init__(ui, explorer)
        self.signal_imp = signal_imp
        self.is_imp_measuring = False

    def disable_imp(self):
        if self.is_connected:
            self.explorer.stream_processor.disable_imp()
            self.reset_impedance()
            self.ui.btn_imp_meas.setText("Measure Impedances")
            self.is_imp_measuring = False
            # AppFunctions.is_imp_measuring = False
        else:
            return

    def emit_imp(self):
        """
        Update impedances
        """

        stream_processor = self.explorer.stream_processor
        print(f"emit_imp - {self.chan_dict=}")
        active_chan = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
        print(active_chan)
        data = {ch: ["", self._impedance_stylesheet_wet("")] for ch in active_chan}

        def callback(packet):
            mode = "dry" if self.ui.imp_mode.currentText() == "Dry electrodes" else "wet"

            imp_values = packet.get_impedances()
            for chan, value in zip(active_chan, imp_values):
                # print(chan, value)
                value = value / 2
                # print(value)
                if value < 5:
                    str_value = "<5 K\u03A9"
                elif (mode == "wet" and value > Settings.COLOR_RULES_WET["open"]) or \
                     (mode == "dry" and value > Settings.COLOR_RULES_DRY["open"]):
                    str_value = "Open"
                else:
                    str_value = str(int(round(value, 0))) + " K\u03A9"

                if mode == "dry":
                    ch_stylesheet = self._impedance_stylesheet_dry(value=value)
                else:
                    ch_stylesheet = self._impedance_stylesheet_wet(value=value)

                data[chan] = [str_value, ch_stylesheet]

            self.signal_imp.emit(data)

        if self.is_connected is False:
            self.display_msg(msg_text="Please connect an Explore device first")
            return

        else:
            if self.is_imp_measuring is False:
                sr_ok = self._verify_samplingRate()
                if sr_ok is False:
                    return
                self.ui.btn_imp_meas.setText("Stop")
                QApplication.processEvents()
                stream_processor.imp_initialize(notch_freq=50)
                stream_processor.subscribe(callback=callback, topic=TOPICS.imp)
                self.is_imp_measuring = True
                # AppFunctions.is_imp_measuring = self.is_imp_measuring

            else:
                # self.signal_imp.disconnect()
                self.disable_imp()

    @Slot(dict)
    def update_impedance(self, chan_dict_data):
        for chan in chan_dict_data.keys():
            new_stylesheet = chan_dict_data[chan][1]
            frame_name = f"frame_{chan}_color"
            ch_frame = self.get_widget_by_objName(frame_name)
            ch_frame.setStyleSheet(new_stylesheet)

            new_value = chan_dict_data[chan][0]
            label_name = f"label_{chan}_value"
            ch_label = self.get_widget_by_objName(label_name)
            ch_label.setText(new_value)

    def check_is_imp(self):
        if self.is_imp_measuring:
            self.display_msg(msg_text="Impedance mode will be disabled", type="info")
            self.disable_imp()
        else:
            return

    def _impedance_stylesheet_wet(self, value):
        """
        Return the stylesheet corresponding to the impedance value
        """
        if type(value) == str:
            imp_stylesheet = Settings.GRAY_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_WET["red"]:  # 500
            imp_stylesheet = Settings.BLACK_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_WET["orange"]:  # 100
            imp_stylesheet = Settings.RED_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_WET["yellow"]:  # 50
            imp_stylesheet = Settings.ORANGE_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_WET["green"]:  # 20
            imp_stylesheet = Settings.YELLOW_IMPEDANCE_STYLESHEET
        else:
            imp_stylesheet = Settings.GREEN_IMPEDANCE_STYLESHEET

        return imp_stylesheet

    def _impedance_stylesheet_dry(self, value):
        """
        Return the stylesheet corresponding to the impedance value
        """
        if type(value) == str:
            imp_stylesheet = Settings.GRAY_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_DRY["red"]:  # 500
            imp_stylesheet = Settings.BLACK_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_DRY["orange"]:  # 200
            imp_stylesheet = Settings.RED_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_DRY["yellow"]:  # na
            imp_stylesheet = Settings.ORANGE_IMPEDANCE_STYLESHEET
        elif value > Settings.COLOR_RULES_DRY["green"]:  # 100
            imp_stylesheet = Settings.YELLOW_IMPEDANCE_STYLESHEET
        else:
            imp_stylesheet = Settings.GREEN_IMPEDANCE_STYLESHEET

        return imp_stylesheet

    def reset_impedance(self):
        if self.is_connected:
            # stream_processor = self.explorer.stream_processor
            # n_chan = stream_processor.device_info['adc_mask']
            # self.chan_dict = dict(zip([c.lower() for c in Settings.CHAN_LIST], n_chan))
            active_chan = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
            chan_dict_data = {ch: ["NA", self._impedance_stylesheet_wet("NA")] for ch in active_chan}

            for chan in chan_dict_data.keys():
                # print(chan_dict[chan])
                new_stylesheet = chan_dict_data[chan][1]
                frame_name = f"frame_{chan}_color"
                ch_frame = self.get_widget_by_objName(frame_name)
                ch_frame.setStyleSheet(new_stylesheet)

                new_value = chan_dict_data[chan][0]
                label_name = f"label_{chan}_value"
                ch_label = self.get_widget_by_objName(label_name)
                ch_label.setText(new_value)
                QApplication.processEvents()

            self.ui.frame_ch1_color.setStyleSheet(self._impedance_stylesheet_wet("NA"))
            self.ui.label_ch1_value.setText("NA")
            QApplication.processEvents()

        else:
            return

    def _verify_samplingRate(self):
        self.explorer._check_connection()
        sr = int(self.get_samplingRate())
        if sr != 250:
            question = (
                "Impedance mode only works in 250 Hz sampling rate!"
                f"\nThe current sampling rate is {sr}."
                "Click on Confirm to change the sampling rate.")

            # response = QMessageBox.question(self, "Confirmation", question)
            response = self.display_msg(msg_text=question, type="question")

            if response == QMessageBox.StandardButton.Yes:
                self.explorer.set_sampling_rate(sampling_rate=250)
                self.ui.value_sampling_rate.setCurrentText(str(250))
                ok = True
            else:
                ok = False
        else:
            ok = True
        return ok

    def reset_imp_vars(self):
        self.is_imp_measuring = False
