# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys


from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QFileDialog
from PySide6.QtCore import QTimer, Qt, Signal, QTimer
from PySide6.QtGui import QIcon
import explorepy as xpy
# xpy.set_bt_interface("pybluez")
# from pyqtgraph.Qt import App
# from pyqtgraph.functions import disconnect
from modules import *
import time
import numpy as np
from datetime import datetime
from modules.dialogs import RecordingDialog, PlotDialog
from modules.stylesheets.stylesheet_centralwidget import CENTRAL_STYLESHEET, MAINBODY_STYLESHEET
# from modules.app_settings import Settings
# from modules.app_functions import AppFunctions, Plots
# from modules.ui_functions import UIFunctions
# from modules.ui_main_window import Ui_MainWindow

# pyside6-uic ui_main_window.ui > ui_main_window.py

# pyside6-uic dialog_plot_settings_light.ui > dialog_plot_settings.py
# pyside6-uic dialog_recording_settings_light.ui > dialog_recording_settings.py

# pyside6-uic dialog_plot_settings_dark.ui > dialog_plot_settings.py
# pyside6-uic dialog_recording_settings_dark.ui > dialog_recording_settings.py

VERSION_APP = 'v0.18'

class MainWindow(QMainWindow):
    signal_exg = Signal(object)
    # signal_fft = Signal(object)
    signal_orn = Signal(object)
    signal_imp = Signal(object)
    signal_mkr = Signal(object)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("images/MentalabLogo.png"))

        self.explorer = xpy.Explore()
        self.is_connected = self.explorer.is_connected
        self.is_streaming = False
        self.battery_percent_list = []
        self.chan_dict = {}
        self.mode = "home"
        self.is_recording = False
        self.is_imp_measuring = False
        self.file_names = None
        self.is_started = False

        self.n_chan = 8
        self.chan_list = Settings.CHAN_LIST[:self.n_chan]

        self.is_pushing = False
        self.run = True
        self.th = None
        
        # self.ui.duration_push_lsl.hide()
        # self.ui.frame_6.hide()
        self.ui.frame_impedance_widgets_16.hide()
        self.ui.imp_meas_info.setHidden(True)
        self.ui.label_3.setHidden(self.file_names is None)
        self.ui.label_7.setHidden(self.file_names is None)

        self.plotting_filters = None
        # self.plotting_filters = {'offset': True, 'notch': 50, 'lowpass': 0.5, 'highpass': 30.0}

        self.downsampling = False
        self.t_exg_plot = np.array([np.NaN]*2500)
        self.exg_pointer = 0
        self.exg_plot = {}
        self.mrk_plot = {"t": [], "code": [], "line": []}
        self.mrk_replot = {"t": [], "code": [], "line": []}

        self.orn_plot = {k: np.array([np.NaN]*200) for k in Settings.ORN_LIST}
        self.t_orn_plot = np.array([np.NaN]*200)
        self.orn_pointer = 0


        self._vis_time_offset = None
        self._baseline_corrector = {"MA_length": 1.5 * Settings.EXG_VIS_SRATE,
                                    "baseline": 0}
        self.y_unit = Settings.DEFAULT_SCALE
        self.y_string = "1 mV"
        self.line = None

        self.lines_orn = [None, None, None]
        self.last_t = 0
        self.last_t_orn = 0

        self.rr_estimator = None
        self.r_peak = {"t": [], "r_peak": [], "points": []}

        self._lambda_exg = lambda data: AppFunctions.plot_exg(self, data)
        self._lambda_orn = lambda data: AppFunctions.plot_orn(self, data)
        self._lambda_marker = lambda data: AppFunctions.plot_marker(self, data)

        # Hide os bar
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Add app version to footer
        self.ui.ft_label_version.setText(VERSION_APP)

        # Hide frame of device settings when launching app
        self.ui.frame_device.hide()
        self.ui.line_2.hide()

        # Set UI definitions (close, restore, etc)
        UIFunctions.ui_definitions(self)

        # Initialize values
        AppFunctions.init_dropdowns(self)

        # Apply stylesheets
        self.ui.centralwidget.setStyleSheet(CENTRAL_STYLESHEET)
        self.ui.main_body.setStyleSheet(MAINBODY_STYLESHEET)
        self.ui.line.setStyleSheet(
            """background-color: #FFFFFF;
            border:none;""")
        QFontDatabase.addApplicationFont("./modules/stylesheets/DMSans-Regular.ttf")


        # List devices when starting the app
        test = False
        if test:
            # pass
            self.explorer.connect(device_name="Explore_CA18")
            # self.explorer.connect(device_name="Explore_CA4C")
            # self.explorer.connect(device_name="Explore_CA07")
            self.is_connected = True
            # self.n_chan = 4
            self.n_chan = len(self.explorer.stream_processor.device_info['adc_mask'])
            self.chan_list = Settings.CHAN_LIST[:self.n_chan]

            AppFunctions.info_device(self)
            AppFunctions.update_frame_dev_settings(self)
            stream_processor = self.explorer.stream_processor
            n_chan = stream_processor.device_info['adc_mask']
            n_chan = [i for i in reversed(n_chan)]
            self.chan_dict = dict(zip([c.lower() for c in Settings.CHAN_LIST], n_chan))
            # self.exg_plot = {ch:[] for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
            self.exg_plot = {ch: np.array([np.NaN]*2500) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
            # AppFunctions.init_plot_exg(self)
            # AppFunctions.init_plot_orn(self)
            # AppFunctions.init_plot_fft(self)
            AppFunctions.init_plots(self)
            AppFunctions.init_imp(self)


        '''else:
            AppFunctions.scan_devices(self)'''

        # Slidable left panel
        self.ui.btn_left_menu_toggle.clicked.connect(lambda: UIFunctions.slideLeftMenu(self))

        # Stacked pages - default open home
        # self.ui.stackedWidget.setCurrentWidget(self.ui.page_settings_testing)
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_bt)
        # self.ui.stackedWidget.setCurrentWidget(self.ui.page_plotsNoWidget)
        # AppFunctions.emit_signals(self)

        # Stacked pages - navigation
        for w in self.ui.left_side_menu.findChildren(QPushButton):
            w.clicked.connect(self.leftMenuButtonClicked)

        # SETTINNGS PAGE BUTTONS
        self.ui.btn_connect.clicked.connect(lambda: AppFunctions.connect2device(self))
        self.ui.dev_name_input.returnPressed.connect(lambda: AppFunctions.connect2device(self))
        self.ui.btn_scan.clicked.connect(lambda: AppFunctions.scan_devices(self))
        self.ui.btn_import_data.clicked.connect(lambda: self.import_recorded_data())
        self.ui.btn_format_memory.clicked.connect(lambda: AppFunctions.format_memory(self))
        self.ui.btn_reset_settings.clicked.connect(lambda: AppFunctions.reset_settings(self))
        self.ui.btn_apply_settings.clicked.connect(lambda: AppFunctions.change_settings(self))
        self.ui.btn_calibrate.clicked.connect(lambda: AppFunctions.calibrate_orn(self))
        self.ui.n_chan.currentTextChanged.connect(lambda: AppFunctions._on_n_chan_change(self))

        # IMPEDANCE PAGE
        self.ui.imp_meas_info.setToolTip("Sum of impedances on REF and individual channels divided by 2")
        self.ui.btn_imp_meas.clicked.connect(lambda: AppFunctions.emit_imp(self))
        self.signal_imp.connect(lambda data: AppFunctions._update_impedance(self, data))
        # self.ui.label_6.linkActivated.connect(lambda: AppFunctions.disable_imp(self))
        self.ui.label_6.setHidden(True)

        # PLOTTING PAGE
        self.ui.btn_record.clicked.connect(self.start_record)
        self.ui.btn_plot_filters.clicked.connect(lambda: self.plot_filters())

        self.ui.btn_marker.setEnabled(False)
        self.ui.value_event_code.textChanged[str].connect(lambda: self.ui.btn_marker.setEnabled(
                (self.ui.value_event_code.text() != "")
                or
                ((self.ui.value_event_code.text().isnumeric()) and (8 <= int(self.ui.value_event_code.text())))
            )
        )
        # self.ui.value_event_code.setEnabled(self.ui.btn_record.text()=="Stop")
        self.ui.btn_marker.clicked.connect(lambda: AppFunctions.set_marker(self))
        self.ui.value_event_code.returnPressed.connect(lambda: AppFunctions.set_marker(self))


        # self.ui.btn_marker.clicked.connect(lambda: self.ui.value_event_code.setText(""))
        self.ui.value_event_code.returnPressed.connect(lambda: AppFunctions.set_marker(self))
        # self.ui.btn_marker.clicked.connect(lambda: self.ui.value_event_code.setText(""))

        self.ui.value_yAxis.currentTextChanged.connect(lambda: AppFunctions._change_scale(self))
        self.ui.value_timeScale.currentTextChanged.connect(lambda: AppFunctions._change_timescale(self))

        '''self.signal_exg.connect(lambda data: AppFunctions.plot_exg(self, data))
        # self.signal_fft.connect(lambda data: AppFunctions.plot_fft(self, data))
        self.signal_orn.connect(lambda data: AppFunctions.plot_orn(self, data))
        self.signal_mkr.connect(lambda data: AppFunctions.plot_marker(self, data))'''

        # self.signal_exg.connect(self._lambda_exg)
        # self.signal_orn.connect(self._lambda_orn)
        # self.signal_mkr.connect(self._lambda_marker)
        AppFunctions._connect_signals(self)

        self.ui.btn_stream.hide()
        self.ui.btn_stream.clicked.connect(lambda: AppFunctions.emit_exg(self, stop=True))
        self.ui.btn_stream.clicked.connect(lambda: self.update_fft())
        self.ui.btn_stream.clicked.connect(lambda: self.update_heart_rate())

        # self.signal_exg.connect(lambda data: AppFunctions.plot_exg_moving(self, data))
        # self.signal_orn.connect(lambda data: AppFunctions.plot_orn_moving(self, data))
        # self.signal_mkr.connect(lambda data: AppFunctions.plot_marker_moving(self, data))

        self.ui.label_7.linkActivated.connect(
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_plotsRecorded)
        )

        # Recorded data plotting page
        self.ui.label_3.linkActivated.connect(
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_plotsNoWidget)
        )
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_bt)

        self.ui.btn_stream_rec.clicked.connect(lambda: self.start_recorded_plots())

        # INTEGRATION PAGE
        self.ui.btn_push_lsl.clicked.connect(lambda: AppFunctions.push_lsl(self))

        # /////////////////////////////// START TESTING ///////////////////////
        '''self.signal_exg.connect(lambda data: AppFunctions.plot_exg(self, data))
        # self.ui.pushButton_2.clicked.connect(lambda: AppFunctions.emit_exg(self))
        self.signal_fft.connect(lambda data: AppFunctions.plot_fft(self, data))
        # self.ui.pushButton_2.clicked.connect(lambda: AppFunctions.emit_fft(self))
        self.signal_orn.connect(lambda data: AppFunctions.plot_orn(self, data))
        self.ui.pushButton_2.clicked.connect(lambda: AppFunctions.emit_orn(self))
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.ui.pushButton_3.clicked.connect(lambda: self.df.to_csv(f"output_{dt_string}.csv"))
        # self.ui.pushButton_3.clicked.connect(lambda: self.ui.graphicsView.clear())'''

        # /////////////////////////////// END TESTING ///////////////////////

    def update_fft(self):
        self.timer_fft = QTimer(self)
        self.timer_fft.setInterval(2000)
        self.timer_fft.timeout.connect(lambda: AppFunctions.plot_fft(self))
        self.timer_fft.start()

    def update_heart_rate(self):
        self.timer_hr = QTimer(self)
        self.timer_hr.setInterval(2000)
        self.timer_hr.timeout.connect(lambda: AppFunctions._plot_heart_rate(self))
        self.timer_hr.start()

    def import_recorded_data(self):
        '''
        Open file dialog to select file to import
        '''
        file_types = "CSV files(*.csv);;EDF files (*.edf);;BIN files (*.BIN)"
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        self.file_names, _ = dialog.getOpenFileNames(
            self,
            "Select Files to import",
            "",
            filter=file_types
            )

        files = ", ".join(self.file_names)
        self.ui.le_data_path.setText(files)
        print(self.file_names)

    def plot_filters(self):
        '''
        Open plot filter dialog and apply filters
        '''
        wait = True if self.plotting_filters is None else False
        sr = self.explorer.stream_processor.device_info['sampling_rate']
        dialog = PlotDialog(sr=sr, current_filters=self.plotting_filters)
        self.plotting_filters = dialog.exec()
        AppFunctions._apply_filters(self)
        # self.loading = LoadingScreen()
        if wait:
            time.sleep(1.5)

    def start_timer_recorder(self):
        '''
        Start timer to display recording time
        '''
        print("clicked")
        self.start_time = datetime.now()

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(lambda: self.displayTime())
        self.timer.start()

    def displayTime(self):
        '''
        Display recording time in label
        '''
        time = datetime.now() - self.start_time
        strtime = str(time).split(".")[0]
        self.ui.label_recording_time.setText(strtime)

    def start_record(self):
        '''
        Start/Stop signal recording
        '''
        if self.is_recording is False:
            dialog = RecordingDialog()
            data = dialog.exec()
            print(data)

            file_name = data["file_path"]
            file_type = data["file_type"]
            duration = data["duration"] if data["duration"] != 0 else None

            self.ui.btn_record.setIcon(QIcon(
                u":icons/icons/icon_maximize.png"))
            self.ui.btn_record.setText("Stop")
            QApplication.processEvents()

            self.explorer.record_data(
                file_name=file_name,
                file_type=file_type,
                duration=duration)
            self.is_recording = True
            self.start_timer_recorder()

        else:
            self.explorer.stop_recording()
            self.ui.btn_record.setIcon(QIcon(
                    u":icons/icons/cil-circle.png"))
            self.ui.btn_record.setText("Record")
            QApplication.processEvents()
            self.is_recording = False
            self.timer.stop()

    def start_recorded_plots(self):
        '''
        Start plotting recorded data
        '''

        '''if self.file_names is None:
            QMessageBox.critical(self, "Error", "Import data first")'''

        # if self.ui.cb_swiping.isChecked():
        if self.ui.cb_swipping_rec.isChecked():
            time_scale = self.ui.value_timeScale_rec.currentText()
        else:
            time_scale = None

        if self.is_started is False:
            self.is_started = True
            exg_wdgt = self.ui.plot_exg_rec
            orn_wdgt = self.ui.plot_orn_rec
            fft_wdgt = self.ui.plot_fft_rec
            if any("exg" in s.lower() for s in self.file_names):
                self.plot_exg_recorded = Plots("exg", self.file_names, exg_wdgt, time_scale)
                plot_fft = Plots("fft", self.file_names, fft_wdgt, time_scale)

            if any("orn" in s.lower() for s in self.file_names):
                self.plot_orn_recorded = Plots("orn", self.file_names, orn_wdgt, time_scale)

        # if self.is_streaming is False and self.ui.cb_swiping.isChecked():
        if self.is_streaming is False and self.ui.cb_swipping_rec.isChecked():
            self.ui.btn_stream_rec.setText("Stop Data Stream")
            self.ui.btn_stream_rec.setStyleSheet(Settings.STOP_BUTTON_STYLESHEET)
            self.is_streaming = True
            QApplication.processEvents()

            self.timer_exg = QTimer()
            self.timer_exg.setInterval(1)
            self.timer_exg.timeout.connect(lambda: self.plot_exg_recorded.update_plot_exg())
            self.timer_exg.start()

            self.timer_orn = QTimer()
            self.timer_orn.setInterval(50)
            self.timer_orn.timeout.connect(lambda: self.plot_orn_recorded.update_plot_orn())
            self.timer_orn.start()

        else:
            self.ui.btn_stream_rec.setText("Start Data Stream")
            self.ui.btn_stream_rec.setStyleSheet(Settings.START_BUTTON_STYLESHEET)
            self.is_streaming = False
            try:
                self.timer_exg.stop()
                self.timer_orn.stop()
            except AttributeError as e:
                print(str(e))

    def changePage(self, btn_name):
        """
        Change the active page when the object is clicked
        Args:
            btn_name
        """
        # btn = self.sender()
        # btn_name = btn.objectName()

        if btn_name == "btn_home":
            self.mode = "home"
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            # if self.is_imp_measuring:
            #     self.explorer.stream_processor.disable_imp()

        elif btn_name == "btn_settings":
            self.mode = "settings"
            if self.is_imp_measuring:
                QMessageBox.information(self, "", "Impedance mode will be disabled")
                AppFunctions.disable_imp(self)

            self.ui.stackedWidget.setCurrentWidget(self.ui.page_bt)

        elif btn_name == "btn_plots":
            self.mode = "exg"
            # self.ui.label_3.setHidden(self.file_names is None)
            # self.ui.label_7.setHidden(self.file_names is None)

            if self.is_imp_measuring:
                QMessageBox.information(self, "", "Impedance mode will be disabled")
                AppFunctions.disable_imp(self)

            if self.is_connected is False and self.file_names is None:
                msg = "Please connect an Explore device or import data before attempting to visualize the data"
                QMessageBox.information(self, "!", msg)
                return
            
            elif self.file_names is None:
                self.ui.stackedWidget.setCurrentWidget(self.ui.page_plotsNoWidget)

                if self.plotting_filters is None:
                    self.plot_filters()

                # if self.ui.plot_orn.getItem(0, 0) is None:
                #     AppFunctions.init_plot_orn(self)
                #     AppFunctions.init_plot_exg(self)
                #     AppFunctions.init_plot_fft(self)

                if not self.is_streaming:
                    AppFunctions.emit_signals(self)
                    self.update_fft()
                    self.update_heart_rate()
                    self.is_streaming = True

                for w in self.ui.frame_cb_channels.findChildren(QCheckBox):
                    w.setEnabled(False)
                    w.setToolTip("Changing channels during visualization is not allowed")
                    w.setStyleSheet("color: gray")

            else:
                self.ui.stackedWidget.setCurrentWidget(self.ui.page_plotsRecorded)

            # self.ui.stackedWidget.setCurrentWidget(self.ui.page_plots)
            # self.ui.stackedWidget.setCurrentWidget(self.ui.page_settings_testing)

            

        elif btn_name == "btn_impedance":
            self.mode = "imp"
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_impedance)
            AppFunctions._reset_impedance(self)

        elif btn_name == "btn_integration":
            self.mode = "integration"
            if self.is_imp_measuring:
                QMessageBox.information(self, "", "Impedance mode will be disabled")
                AppFunctions.disable_imp(self)
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_integration)

    def leftMenuButtonClicked(self):
        """
        Change style of the button clicked and move to the selected page
        """

        btn = self.sender()
        btn_name = btn.objectName()

        # Navigate to active page
        self.changePage(btn_name)

        if btn_name != "btn_left_menu_toggle":
            # Reset style for other buttons
            for w in self.ui.left_side_menu.findChildren(QPushButton):
                if w.objectName() != btn_name:
                    defaultStyle = w.styleSheet().replace(Settings.BTN_LEFT_MENU_SELECTED_STYLESHEET, "")
                    # Apply default style
                    w.setStyleSheet(defaultStyle)

            # Apply new style
            newStyle = btn.styleSheet() + (Settings.BTN_LEFT_MENU_SELECTED_STYLESHEET)
            btn.setStyleSheet(newStyle)

    def mousePressEvent(self, event):
        '''
        Get mouse current position to move the window
        Args: mouse press event
        '''
        self.clickPosition = event.globalPos()

    '''def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)'''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())