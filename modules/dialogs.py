from PySide6.QtGui import QDoubleValidator
from modules.dialog_plot_settings import Ui_Dialog as Ui_PlotDialog
from modules.dialog_recording_settings import Ui_Dialog as Ui_RecordingDialog
from main import *
from PySide6.QtWidgets import QDialog

stylesheet_cancel = (
    "QPushButton{\n"
    "	color: #FFF;\n"
    "	background-color: transparent;\n"
    "	border: 2px solid #FFF;\n"
    "	padding: 5px;\n"
    "	border-radius: 5px;\n"

    "}\n"
    "\n"
    "QPushButton:hover{\n"
    "	background-color: rgb(61, 64, 89);\n"
    "}\n"
    "\n"
    "QPushButton:pressed{\n"
    "	background-color: rgb(101, 106, 141);\n"
    "	border:  2px solid rgb(61, 64, 89);\n"
    "}\n"
    ""
)

class PlotDialog(QDialog):
    def __init__(self, sr, current_filters, parent=None):
        super().__init__(parent)
        self.ui = Ui_PlotDialog()
        self.ui.setupUi(self)
        self.ui.lbl_warning.hide()
        self.ui.cb_offset.setToolTip("Remove the DC offset of the signal based on the previous signal values")
        self.setWindowTitle("Visualization Settings")

        self.ui.buttonBox.button(QDialogButtonBox.Cancel).setStyleSheet(
            stylesheet_cancel
        )


        self.s_rate = float(sr)
        if current_filters is None:
            self.offset = False
            self.notch = ""
            self.lowpass = ""
            self.highpass = ""
        else:
            self.offset = current_filters["offset"]
            self.notch = str(current_filters["notch"])
            self.lowpass = str(current_filters["lowpass"])
            self.highpass = str(current_filters["highpass"])

        # self.double_validator = QDoubleValidator(decimals=2)
        
        # Add options to notch combobox
        self.ui.value_notch.addItems(["", "50", "60"])
        
        # Set current values 
        self.ui.value_notch.setCurrentText(self.notch)
        self.ui.value_highpass.setText(self.highpass)
        self.ui.value_lowpass.setText(self.lowpass)
        self.ui.cb_offset.setChecked(self.offset)

        # Set validators (only accept doubles)
        regex = QRegularExpression("([0-9]+\.?[0-9]|\.[0-9])")
        self.ui.value_highpass.setValidator(QRegularExpressionValidator(regex))
        self.ui.value_lowpass.setValidator(QRegularExpressionValidator(regex))
        # self.ui.value_lowpass.setValidator(QDoubleValidator(decimals=1))

        # Verify input with output message when editing is done
        # self.ui.value_lowpass.editingFinished.connect(lambda: self.verify_input())
        # self.ui.value_highpass.editingFinished.connect(lambda: self.verify_input())

        # Verify input without output message when typing
        self.ui.value_lowpass.textChanged.connect(lambda: self.verify_input(borderOnly=False))
        self.ui.value_highpass.textChanged.connect(lambda: self.verify_input(borderOnly=False))

    def verify_input(self, borderOnly=False):
        nyq_freq = self.s_rate / 2.

        max_hc_freq = nyq_freq-1
        min_lc_freq = 0.003 * nyq_freq

        hc_freq_warning = (
            "High cutoff frequency cannot be larger than or equal to the nyquist frequency."
            f"The maximum high cutoff frequency is {max_hc_freq:.1f} Hz!"
            )

        lc_freq_warning = (
            "'Transient band for low cutoff frequency is too narrow."
            f"The minimum low cutoff frequency is {min_lc_freq:.2f} Hz!"
        )

        bp_freq_warning = ("High cutoff frequency must be larger than low cutoff frequency.")

        r_value = "" if self.ui.value_highpass.text() in [None, 'None'] else self.ui.value_highpass.text()
        l_value = "" if self.ui.value_lowpass.text() in [None, 'None'] else self.ui.value_lowpass.text()
        # print(r_value, l_value)

        r_stylesheet = ""
        l_stylesheet = ""
        lbl_txt = ""

        accepted = True
        
        if r_value == "." or l_value == ".":
            accepted = False
            return

        if r_value != "" and l_value == "": # lowpass
            if float(r_value) >= nyq_freq:
                # print(hc_freq_warning)
                lbl_txt = hc_freq_warning
                r_stylesheet = "border: 1px solid rgb(217, 0, 0)"
                l_stylesheet = ""
                accepted = False


        elif r_value == "" and l_value != "":
            lc_freq = float(l_value)  / nyq_freq
            if lc_freq <= 0.003:
                print(lc_freq_warning)
                lbl_txt = lc_freq_warning
                r_stylesheet = ""
                l_stylesheet = "border: 1px solid rgb(217, 0, 0)"
                accepted = False

        elif r_value != "" and l_value != "":
            lc_freq = float(l_value)  / nyq_freq
            if float(l_value) >= float(r_value):
                print(bp_freq_warning)
                lbl_txt = bp_freq_warning
                r_stylesheet = "border: 1px solid rgb(217, 0, 0)"
                l_stylesheet = "border: 1px solid rgb(217, 0, 0)"
                accepted = False
                
            elif float(r_value) >= nyq_freq:
                print(hc_freq_warning)
                lbl_txt = hc_freq_warning
                r_stylesheet = "border: 1px solid rgb(217, 0, 0)"
                l_stylesheet = ""
                accepted = False

            elif lc_freq <= 0.003:
                print(lc_freq_warning)
                lbl_txt = lc_freq_warning
                r_stylesheet = ""
                l_stylesheet = "border: 1px solid rgb(217, 0, 0)"
                accepted = False

        else:
            r_stylesheet = ""
            l_stylesheet = ""
            lbl_txt = ""

        self.ui.value_highpass.setStyleSheet(r_stylesheet)
        self.ui.value_lowpass.setStyleSheet(l_stylesheet)
        if not borderOnly:
            self.ui.lbl_warning.setText(lbl_txt)
            if lbl_txt != "":
                self.ui.lbl_warning.setHidden(False)
            else:
                self.ui.lbl_warning.setHidden(True)


        # QTimer.singleShot(1000, lambda: self.ui.value_highpass.setStyleSheet(''))
        # QTimer.singleShot(1000, lambda: self.ui.value_lowpass.setStyleSheet(''))
        return accepted


    def exec(self):
        ok = self.verify_input()
        
        if ok:
            super().exec()
            return {
                "offset": self.ui.cb_offset.isChecked(),
                "notch": None if self.ui.value_notch.currentText() == "" == "" else int(self.ui.value_notch.currentText()),
                "lowpass": None if self.ui.value_lowpass.text() == "" else float(self.ui.value_lowpass.text()),
                "highpass": None if self.ui.value_highpass.text() == "" else float(self.ui.value_highpass.text())
                }


class RecordingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_RecordingDialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Recording Settings")
        self.ui.buttonBox.button(QDialogButtonBox.Cancel).setStyleSheet(
            stylesheet_cancel
        )

        self.recording_time = self.ui.spinBox.value()
        self.recording_mode = "csv"
        self.recording_path = ""


        self.ui.btn_browse.clicked.connect(self.save_filename)
        self.ui.spinBox.setMaximum(10000000)
        self.ui.rdbtn_csv.setChecked(True)

    def save_filename(self):
        """
        Open a dialog to select file name to be saved
        """
        if self.ui.rdbtn_csv.isChecked():
            self.recording_mode = "csv"
        elif self.ui.rdbtn_edf.isChecked():
            self.recording_mode = "edf"
        else:
            self.recording_mode = "csv"

        file_types = "CSV files(*.csv);;EFD files (*.efd)"
        dialog = QFileDialog()
        # dialog.setDefaultSuffix(".csv")
        options = dialog.Options()
        file_name, _ = dialog.getSaveFileName(
            self,
            "Save File As",
            "",
            filter=file_types, options=options
            )

        self.recording_path = file_name  # + "." + self.recording_mode
        self.ui.input_filepath.setText(self.recording_path)
        QApplication.processEvents()

    def exec(self):
        super().exec()
        return {
            "file_path": self.recording_path,
            "file_type": self.recording_mode,
            "duration": self.recording_time
            }

if __name__ == "__main__":
    app = QApplication(sys.argv)
    plotting_filters = {'offset': True, 'notch': 50, 'lowpass': 0.5, 'highpass': 30.0}

    dialog = PlotDialog(sr=250, current_filters=plotting_filters)
    # dialog = RecordingDialog()
    dialog.show()
    sys.exit(app.exec())

    
