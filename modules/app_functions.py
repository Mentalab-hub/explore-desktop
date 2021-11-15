# import time
# from explorepy import stream_processor
from datetime import timezone
import time
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QIntValidator
from explorepy import stream_processor
from explorepy.stream_processor import TOPICS
from pyqtgraph.Qt import App
from scipy import signal
# from scipy.ndimage.measurements import label
from main import *
import numpy as np
from modules.app_settings import Settings
from explorepy.tools import bt_scan
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from PySide6.QtWidgets import QApplication, QCheckBox, QMessageBox
from contextlib import contextmanager
import pyqtgraph as pg
import mne
from explorepy.tools import HeartRateEstimator
from modules.workers import Thread
import explorepy._exceptions as xpy_ex

class AppFunctions(MainWindow):

    def __init__(self):
        super().__init__()

    # ///////////////////////////////////
    # ///// START GENERAL FUNCTIONS/////

    def init_dropdowns(self):
        '''
        Initilize the GUI dropdowns with the values specified above
        '''

        # value number of channels:
        # combo_nchan = self.ui.n_chan
        # model_nchan = combo_nchan.model()
        # for n in Settings.N_CHAN_LIST:
        #     item = QStandardItem(str(n))
        #     item.setForeground(QColor('red'))
        #     model_nchan.appendRow(item)
        self.ui.n_chan.addItems(Settings.N_CHAN_LIST)
        self.ui.n_chan.setCurrentText("8")


        # value_signal_type
        self.ui.value_signal.addItems(Settings.MODE_LIST)
        self.ui.value_signal_rec.addItems(Settings.MODE_LIST)

        # value_yaxis
        self.ui.value_yAxis.addItems(Settings.SCALE_MENU.keys())
        self.ui.value_yAxis.setCurrentText("1 mV")

        self.ui.value_yAxis_rec.addItems(Settings.SCALE_MENU.keys())
        self.ui.value_yAxis_rec.setCurrentText("1 mV")
        
        # value_time_scale
        self.ui.value_timeScale.addItems(Settings.TIME_RANGE_MENU.keys())
        self.ui.value_timeScale_rec.addItems(Settings.TIME_RANGE_MENU.keys())

        # value_sampling_rate
        self.ui.value_sampling_rate.addItems([str(int(sr)) for sr in Settings.SAMPLING_RATES])

        self.ui.value_event_code.setValidator(QIntValidator(8, 65535))

        self.ui.cb_swipping_rec.setChecked(True)

        self.ui.imp_mode.addItems(["Wet electrodes", "Dry electrodes"])

    def change_footer(self):
        """
        Change the fields for device and firmware in the GUI footer
        """
        if self.is_connected:
            dev_name = self.explorer.stream_processor.device_info[
                "device_name"]
            device_lbl = f"Connected to {dev_name}"
            firmware = self.explorer.stream_processor.device_info[
                "firmware_version"]
        else:
            device_lbl = "Not connected"
            firmware = "NA"
            stylesheet = AppFunctions._battery_stylesheet(self, value="NA")
            AppFunctions._update_battery(self, "NA", new_stylesheet=stylesheet)
            AppFunctions._update_temperature(self, "NA")

        AppFunctions._update_device_name(self, new_value=device_lbl)
        AppFunctions._update_firmware(self, new_value=firmware)
        # self.ui.ft_label_device_3.setText(device_lbl)
        # self.ui.ft_label_firmware_value.setText(firmware)

    # ///// END GENERAL FUNCTIONS/////

    # //////////////////// ///////////////
    # ///// START FUNCTIONS SETTINGS PAGE /////

    def change_btn_connect(self):
        '''
        Change connect buttonn to Connect/Disconnect depending on explore status
        '''
        if self.is_connected:
            self.ui.btn_connect.setText("Disconnect")
        else:
            self.ui.btn_connect.setText("Connect")

    def update_frame_dev_settings(self):
        """
        Update the frame with the device settings.
        Only shown if a device is connected
        """
        # self.explorer._check_connection()

        if self.is_connected:
            explore = self.explorer
            stream_processor = explore.stream_processor

            # ///// CONFIGURE DEVICE FRAME /////
            # Set device name
            self.ui.label_explore_name.setText(
                stream_processor.device_info["device_name"])

            # Set active channels
            chan = stream_processor.device_info['adc_mask']
            chan = [i for i in reversed(chan)]

            self.chan_dict = dict(zip([c.lower() for c in Settings.CHAN_LIST], chan))
            
            # print("update frame dev")
            # print(f"{self.explorer.stream_processor.device_info['adc_mask']=}")
            # print(f"{self.chan_dict=}")

            for w in self.ui.frame_cb_channels.findChildren(QCheckBox):
                # print(w)
                w.setChecked(self.chan_dict[w.objectName().replace("cb_", "")])
                if w.objectName().replace("cb_", "") not in self.chan_list:
                    w.hide()
                if w.isHidden() and w.objectName().replace("cb_", "") in self.chan_list:
                    w.show()

            # if self.n_chan < 16:
            #     self.ui.frame_cb_channels_16.hide()

            points = AppFunctions._plot_points(self)
            self.exg_plot = {ch: np.array([np.NaN]*points) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
            self.t_exg_plot = np.array([np.NaN]*points)

            # Set sampling rate (value_sampling_rate)
            sr = stream_processor.device_info['sampling_rate']
            self.ui.value_sampling_rate.setCurrentText(str(int(sr)))
            # ////////

            # Show the device frame
            self.ui.frame_device.show()
            self.ui.line_2.show()

        else:
            self.ui.frame_device.hide()
            self.ui.line_2.hide()

    def scan_devices(self):
        """"
        Scans for available explore devices.
        """

        self.ui.ft_label_device_3.setText("Scanning ...")
        self.ui.ft_label_device_3.repaint() 
        # QApplication.processEvents()

        self.ui.list_devices.clear()
        with AppFunctions._wait_cursor():
            try:
                explore_devices = bt_scan()
            except ValueError:
                print("Error opening socket. Check if bt is on")
            explore_devices = [dev[0] for dev in explore_devices]
            if len(explore_devices) == 0:
                print("No explore devices found. Please make sure it is turn on and click on reescan")

            self.ui.list_devices.addItems(explore_devices)
            pass

        self.ui.ft_label_device_3.setText("Not connected")
        self.ui.ft_label_device_3.repaint()
        # QApplication.processEvents()        

    def get_device_from_le(self):

        input_name = self.ui.dev_name_input.text()

        if not input_name.startswith("Explore_") and len(input_name) == 4:
            device_name = "Explore_" + input_name
        elif input_name.startswith("Explore_"):
            device_name = input_name
        else:
            device_name = ""

        return device_name

    def get_device_from_list(self):
        try:
            device_name = self.ui.list_devices.selectedItems()[0].text()
        except IndexError:
            device_name = ""
            
        return device_name

    def connect2device(self):
        """
        Connect to the explore device.
        """
        if self.is_connected is False:
            device_name_le = AppFunctions.get_device_from_le(self)
            device_name_list = AppFunctions.get_device_from_list(self)
            
            if device_name_le != "":
                device_name = device_name_le
            elif device_name_list != "":
                device_name = device_name_list
            else:
                msg = "Please select a device or provide a valid name (Explore_XXXX or XXXX) before connecting."
                # QMessageBox.critical(self, "Error", msg)
                AppFunctions._display_msg(self, msg)
                return

            self.ui.ft_label_device_3.setText("Connecting ...")
            self.ui.ft_label_device_3.repaint()
            # QApplication.processEvents()

            with AppFunctions._wait_cursor():
                try:
                    self.explorer.connect(device_name=device_name)
                    self.is_connected = True
                except xpy_ex.DeviceNotFoundError as e:
                    msg = str(e)
                    # QMessageBox.critical(self, "Error", msg)
                    AppFunctions._display_msg(self, msg)
                    return
                except TypeError as e:
                    msg = "Please select a device or provide a valid name (Explore_XXXX or XXXX) before connecting."
                    # QMessageBox.critical(self, "Error", msg)
                    AppFunctions._display_msg(self, msg)
                    return
                except AssertionError as e:
                    msg = str(e)
                    # QMessageBox.critical(self, "Error", msg)
                    AppFunctions._display_msg(self, msg)
                    return
                except ValueError:
                    msg = "Error opening socket.\nPlease make sure the bluetooth is on."
                    # QMessageBox.critical(self, "Error", msg)
                    AppFunctions._display_msg(self, msg)
                    return
                except Exception as e:
                    msg = str(e)
                    # QMessageBox.critical(self, "Error", msg)
                    AppFunctions._display_msg(self, msg)
                    return
                pass

        else:
            try:
                self.explorer.disconnect()
                self.is_connected = False
            except AssertionError as e:
                msg = str(e)
                AppFunctions._display_msg(self, msg)
                # QMessageBox.critical(self, "Error", msg)
            except Exception as e:
                msg = str(e)
                AppFunctions._display_msg(self, msg)
                # QMessageBox.critical(self, "Error", msg)

        print(self.is_connected)
        AppFunctions._on_connection(self)

    def info_device(self):
        r"""
        Update device information
        """
        # self.explorer._check_connection()

        def callback(packet):
            # print(packet)
            new_info = packet.get_data()
            # print(new_info)
            for key in new_info.keys():
                if key == "temperature":
                    new_value = str(new_info[key][0]) if self.is_connected else "NA"
                    AppFunctions._update_temperature(self, new_value=new_value)
                    # print("\n\ntemperature", new_info[key])

                elif key == "battery":
                    self.battery_percent_list.append(new_info[key][0])
                    if len(self.battery_percent_list) > Settings.BATTERY_N_MOVING_AVERAGE:
                        del self.battery_percent_list[0]
                    value = int(np.mean(self.battery_percent_list))
                    value = 1 if value < 1 else value
                    new_value = value if self.is_connected else "NA"
                    stylesheet = AppFunctions._battery_stylesheet(self, value=new_value)
                    AppFunctions._update_battery(self, new_value=str(new_value), new_stylesheet=stylesheet)

                elif key == "fimrware":
                    # print("firmware callback: ", new_info[key])
                    new_value = new_info[key] if self.is_connected else "NA"
                    AppFunctions._update_firmware(self, new_value=new_value)

                elif key == "device_name":
                    # print("name callback: ", new_info[key])
                    connected_lbl = f"Connected to {new_info[key]}"
                    not_connected_lbl = "Not connected"
                    new_value = connected_lbl if self.is_connected else not_connected_lbl
                    AppFunctions._update_device_name(self, new_value=new_value)

            QApplication.processEvents()

        self.explorer.stream_processor.subscribe(callback=callback, topic=TOPICS.device_info)
        self.explorer.stream_processor.subscribe(callback=callback, topic=TOPICS.env)

    def format_memory(self):
        r"""
        Display a popup asking for confirmation.
        If yes, memory is formatted.
        """

        question = "Are you sure you want to format the memory?"
        # response = QMessageBox.question(self, "Confirmation", question)
        # response.setStyleSheet(Settings.POPUP_STYLESHEET)
        response = AppFunctions._display_msg(self, msg_text=question, type="question")
        if response == QMessageBox.StandardButton.Yes:
            print("yes")
            self.explorer.format_memory()
            AppFunctions._display_msg(self, msg_text="Memory formatted", type="info")
        else:
            print("no")
            return

    def reset_settings(self):
        r"""
        Display a popup asking for confirmation.
        If yes, the settinngs are set to default.
        """

        question = "Are you sure you want to reset your settings?"
        # response = QMessageBox.question(self, "Confirmation", question)
        response = AppFunctions._display_msg(self, msg_text=question, type="question")
        
        if response == QMessageBox.StandardButton.Yes:
            self.explorer.reset_soft()

            self.explorer.set_sampling_rate(sampling_rate=250)
            mask = "11111111"
            int_mask = int(mask, 2)
            self.explorer.set_channels(int_mask)

            print(self.explorer.stream_processor.device_info['sampling_rate'])
            AppFunctions.update_frame_dev_settings(self)
            AppFunctions._display_msg(self, msg_text="Settings reset", type="info")
        else:
            return

    def calibrate_orn(self):
        r"""
        Calibrate the orientation
        """
        question = ("Do you want to continue with the orientation sensors calibration?\n"
                    "This will overwrite the calibration data if it already exists\n\n"
                    "If yes, you would need to move and rotate the device for 100 seconds\n"
                    )
        # response = QMessageBox.question(self, "Confirmation", question)
        response = AppFunctions._display_msg(self, msg_text=question, type="question")

        if response == QMessageBox.StandardButton.Yes:
            # QMessageBox.information(self, "", "Calibrating...\nPlease move and rotate the device")
            self.explorer.calibrate_orn(do_overwrite=True)
            msg = "Calibration Complete"
            title = "Done"
            # QMessageBox.information(self, title, msg)
            AppFunctions._display_msg(self, msg_text=msg, title=title, type="info")
        else:
            return

    def change_sampling_rate(self):
        """
        Change the sampling rate
        """

        sr = self.explorer.stream_processor.device_info['sampling_rate']
        str_value = self.ui.value_sampling_rate.currentText()
        value = int(str_value)
        if int(sr) != value:
            AppFunctions._check_filters_new_sr(self)
            # print(value)
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

    def change_active_channels(self):
        """
        Read selected checkboxes and set the channel mask of the device
        """

        active_chan = []

        for w in self.ui.frame_cb_channels.findChildren(QCheckBox):
            status = str(1) if w.isChecked() else str(0)
            active_chan.append(status)
            # print(w.objectName(), w.isChecked())
            # w.setChecked(self.chan_dict[w.objectName().replace("cb_", "")])
        active_chan_int = [int(i) for i in active_chan]
        active_chan = [i for i in reversed(active_chan)]

        if active_chan_int != self.explorer.stream_processor.device_info['adc_mask']:

            # TEST FILTERS
            if self.plotting_filters is not None:
                self._baseline_corrector["baseline"] = None
                self.explorer.stream_processor.remove_filters()
                
            mask = "".join(active_chan)
            int_mask = int(mask, 2)
            
            self.explorer.set_channels(int_mask)
            self.offsets = np.arange(1, active_chan.count(1) + 1)[:, np.newaxis].astype(float)

            n_chan = self.explorer.stream_processor.device_info['adc_mask']
            n_chan = [i for i in reversed(n_chan)]
            
            self.chan_dict = dict(
                    zip([c.lower() for c in Settings.CHAN_LIST], n_chan))

            points = AppFunctions._plot_points(self)
            self.exg_plot = {ch: np.array([np.NaN]*points) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
             
            print('changed')
            print(f"{self.explorer.stream_processor.device_info['adc_mask']=}")
            print(f"{self.chan_dict=}")

            self.offsets = np.arange(1, n_chan.count(1) + 1)[:, np.newaxis].astype(float)
            self._baseline_corrector["baseline"] = None
            AppFunctions._apply_filters(self)
            AppFunctions.init_imp(self)

        else:
            print("Same channel mask")
            return

    def change_settings(self):
        """
        Apply changes in device settings
        """
        
        # AppFunctions._disconnect_signals(self)
        # print('disconnected signal')
        stream_processor = self.explorer.stream_processor
        
        with AppFunctions._wait_cursor():
            AppFunctions.change_active_channels(self)
            AppFunctions.change_sampling_rate(self)
            # sr = AppFunctions._get_samplingRate(self)
            # ts = AppFunctions._get_timeScale(self)
            # points = sr * ts
            points = AppFunctions._plot_points(self)
            self.exg_plot = {ch: np.array([np.NaN]*points) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}
            self.t_exg_plot = np.array([np.NaN]*points)
            pass
        AppFunctions.init_plots(self)

        act_chan = ", ".join([ch for ch in self.chan_dict if self.chan_dict[ch]==1])
        title = "!"
        msg = (
            "Device settings have been changed:"
            f"\nSampling Rate: {int(stream_processor.device_info['sampling_rate'])}"
            f"\nActive Channels: {act_chan}"
        )
        # QMessageBox.information(self, title, msg)
        AppFunctions._display_msg(self, msg_text=msg, type="info")

        # AppFunctions._connect_signals(self)
        # print("connected signal")

        # AppFunctions.emit_exg(self)

    def _on_n_chan_change(self):
        AppFunctions._set_n_chan(self)
        print()
        AppFunctions.update_frame_dev_settings(self)

    # ///// END SETTINGS PAGE FUNCTIONS /////

    # //////////////////// ///////////////
    # ///// START PLOT PAGE FUNCTIONS/////

    def start_stop_streaming(self):
        '''
        Start or stop acquiring signal
        '''
        # if self.is_connected:
        if self.is_streaming is False:
            self.ui.btn_stream.setText("Stop Data Stream")
            print(Settings.START_BUTTON_STYLESHEET)
            self.ui.btn_stream.setStyleSheet(Settings.STOP_BUTTON_STYLESHEET)
            self.explorer.acquire()
            self.is_streaming = True
        else:
            self.ui.btn_stream.setText("Start Data Stream")
            self.ui.btn_stream.setStyleSheet(Settings.START_BUTTON_STYLESHEET)
            self.is_streaming = False

    def set_marker(self):
        '''
        Get the value for the event code and mark it
        '''
        event_code = self.ui.value_event_code.text()
        try:
            self.explorer.set_marker(int(event_code))
        except ValueError as e:
            AppFunctions._display_msg(self, msg_text=str(e))
            # QMessageBox.critical(self, "Error", str(e))


        # Clean input text box
        self.ui.value_event_code.setText("")

    # ///// END PLOT PAGE FUNCTIONS/////

    # ////////////////////////////////////////
    # ///// START IMPEDANCE PAGE FUNCTIONS/////
    def init_imp(self):
        active_chan = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]

        if self.n_chan < 16:
            self.ui.frame_impedance_widgets_16.hide()

        for chan in Settings.CHAN_LIST:
            frame_name = f"frame_{chan}"
            try:
                ch_frame = AppFunctions._get_widget_by_objName(self, frame_name)
            except AttributeError:
                print(chan, frame_name)
                pass
            if chan not in active_chan:
            # if chan not in self.chan_list:
                ch_frame.hide()
            elif chan in active_chan and ch_frame.isHidden():
                ch_frame.show()


    def disable_imp(self):
        if self.is_connected:
            self.explorer.stream_processor.disable_imp()
            AppFunctions._reset_impedance(self)
            self.ui.btn_imp_meas.setText("Measure Impedances")
            self.is_imp_measuring = False
        else:
            return

    def emit_imp(self):
        """
        Update impedances
        """

        stream_processor = self.explorer.stream_processor
        # n_chan = stream_processor.device_info['adc_mask']
        # self.chan_dict = dict(zip([c.lower() for c in Settings.CHAN_LIST], n_chan))
        print(self.chan_dict)
        chan_list = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
        print(chan_list)
        # chan_list = 
        data = {ch: ["", AppFunctions._impedance_stylesheet_wet(self, "")] for ch in chan_list}

        def callback(packet):

            imp_values = packet.get_impedances()
            # print(imp_values)
            mode = "dry" if self.ui.imp_mode.currentText() == "Dry electrodes" else "wet"
            for chan, value in zip(chan_list, imp_values):
                # print(chan, value)
                # value = value/2
                # print(value)
                if value < 5:
                    str_value = "<5 K\u03A9"
                # elif value < 100 and mode == "dry"
                elif value > 500:
                    str_value = "Open"
                else:
                    str_value = str(int(round(value, 0))) + " K\u03A9"
                # str_value = "<5" if value < 5 else str(round(value, 0))
                # str_value += " K\u03A9"
                if mode == "dry":
                    ch_stylesheet = AppFunctions._impedance_stylesheet_dry(self, value=value)
                else:
                    ch_stylesheet = AppFunctions._impedance_stylesheet_wet(self, value=value)

                data[chan] = [str_value, ch_stylesheet]

            self.signal_imp.emit(data)


        if self.is_connected is False:
            QMessageBox.critical(self, "Error", "Please connect an Explore device first")
            return

        else:
            if self.is_imp_measuring is False:
            # print("start")
                sr_ok = AppFunctions._verify_samplingRate(self)
                if sr_ok is False:
                    return
                self.ui.btn_imp_meas.setText("Stop")
                QApplication.processEvents()
                stream_processor.imp_initialize(notch_freq=50)
                stream_processor.subscribe(callback=callback, topic=TOPICS.imp)
                self.is_imp_measuring = True

            else:
            #    self.signal_imp.disconnect()
               AppFunctions.disable_imp(self)

    # ///// END IMPEDANCE PAGE FUNCTIONS/////
    def _verify_samplingRate(self):
        self.explorer._check_connection()
        sr = int(AppFunctions._get_samplingRate(self))
        if sr != 250:
            question = (
                "Impedance mode only works in 250 Hz sampling rate!"
                f"\nThe current sampling rate is {sr}."
                "Click on Confirm to change the sampling rate.")

            # response = QMessageBox.question(self, "Confirmation", question)
            response = AppFunctions._display_msg(self, msg_text=question, type="question")

            if response == QMessageBox.StandardButton.Yes:
                self.explorer.set_sampling_rate(sampling_rate=250)
                self.ui.value_sampling_rate.setCurrentText(str(250))
                ok = True
            else:
                ok = False
        else:
            ok = True
        return ok
    # ////////////////////////////////////////
    # ///// START INTEGRATION PAGE FUNCTIONS/////

    def push_lsl(self):
        if self.is_connected is False:
            QMessageBox.critical(self, "Error", "Please connect an Explore device first")
            return

        spinbox_val = self.ui.spinBox.value()
        duration = None if spinbox_val == 0 else spinbox_val
        
        if self.is_pushing is False:
            self.is_pushing = True
            self.explorer.push2lsl(duration=None, block=False)
            self.ui.btn_push_lsl.setText("Stop")
        else:
            self.is_pushing = False
            self.explorer.stop_lsl()
            self.ui.btn_push_lsl.setText("Push")

    def enable_lsl_duration(self):
        enable = self.ui.checkBox.isChecked()
        # if self.ui.cb_lsl_duration.isChecked():
        self.ui.label_13.setEnabled(enable)
        # self.ui.label_lsl_duration.setEnabled(enable)
        self.ui.spinBox.setEnabled(enable)
        # self.ui.lsl_duration_.setEnabled(enable)


    # ///// END INTEGRATION PAGE FUNCTIONS/////

    # ////////////////////////////////////////
    # ///// START PLOTTING FUNCTIONS/////
    # ########### Start Emit Signals Functions ################
    def emit_signals(self):
        AppFunctions.emit_orn(self)
        AppFunctions.emit_exg(self)
        AppFunctions.emit_marker(self)

    def emit_exg(self, stop=False):
        """
        Get EXG data and plot
        """
        print(self.y_string)
        stream_processor = self.explorer.stream_processor
        chan_list = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]

        def callback(packet):
            exg_fs = stream_processor.device_info['sampling_rate']
            timestamp, exg = packet.get_data(exg_fs)

            # From timestamp to seconds
            if self._vis_time_offset is None:
                self._vis_time_offset = timestamp[0]
            time_vector = timestamp - self._vis_time_offset

            # Downsampling
            if self.downsampling:
                exg = exg[:, ::int(exg_fs / Settings.EXG_VIS_SRATE)]
                time_vector = time_vector[::int(exg_fs / Settings.EXG_VIS_SRATE)]

            # Baseline correction
            if self.plotting_filters is not None and self.plotting_filters["offset"]: 
            # if not True: # if testing
                samples_avg = exg.mean(axis=1)
                if self._baseline_corrector["baseline"] is None:
                    self._baseline_corrector["baseline"] = samples_avg
                else:
                    # self._baseline_corrector["baseline"] -= (
                    #             (self._baseline_corrector["baseline"] - samples_avg) / self._baseline_corrector["MA_length"] *
                    #             exg.shape[1])
                    # TEST FILTERS
                    try:
                        self._baseline_corrector["baseline"] -= (
                                (self._baseline_corrector["baseline"] - samples_avg) / self._baseline_corrector["MA_length"] *
                                exg.shape[1])
                        exg -= self._baseline_corrector["baseline"][:, np.newaxis]
                        
                    except ValueError:
                        self._baseline_corrector["baseline"] = None
                        print("ValueError line 733")
                # TEST FITLERS
                # exg -= self._baseline_corrector["baseline"][:, np.newaxis]
            else:
                self._baseline_corrector["baseline"] = None

            # Update ExG unit
            
            # exg = self.offsets + exg / self.y_unit

            # data = dict(zip(chan_list, exg))
            # data['t'] = time_vector
            # self.signal_exg.emit(data)
            # TEST FILTERS
            try:
                exg = self.offsets + exg / self.y_unit

                data = dict(zip(chan_list, exg))
                data['t'] = time_vector
                self.signal_exg.emit(data)
            except Exception as e:
                print('emit exg exception', e)

        if stop:
            stream_processor.unsubscribe(topic=TOPICS.filtered_ExG, callback=callback)
            print("unsubscribe")
            return

        AppFunctions._apply_filters(self)
        stream_processor.subscribe(topic=TOPICS.filtered_ExG, callback=callback)
         
    def emit_orn(self):
        """"
        Get orientation data
        """
        stream_processor = self.explorer.stream_processor
        chan_list = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
        # import pandas as pd
        # self.df = pd.DataFrame(columns=chan_list.append("t"))

        def callback(packet):
            timestamp, orn_data = packet.get_data()
            if self._vis_time_offset is None:
                self._vis_time_offset = timestamp[0]
            time_vector = list(np.asarray(timestamp) - self._vis_time_offset)

            data = dict(zip(Settings.ORN_LIST, np.array(orn_data)[:, np.newaxis]))
            data['t'] = time_vector
            # dftemp = pd.DataFrame.from_dict(data)
            # self.df = self.df.append(dftemp)

            self.signal_orn.emit(data)

        stream_processor.subscribe(topic=TOPICS.raw_orn, callback=callback)
    
    def emit_marker(self):
        '''
        '''
        stream_processor = self.explorer.stream_processor

        def callback(packet):
            timestamp, _ = packet.get_data()
            if self._vis_time_offset is None:
                self._vis_time_offset = timestamp[0]
            time_vector = list(np.asarray(timestamp) - self._vis_time_offset)

            '''new_data = dict(zip(
                ['marker', 't', 'code'],
                [np.array([0.01, self.n_chan + 0.99, None], dtype=np.double),
                    np.array([timestamp[0], timestamp[0], None], dtype=np.double)]))'''

            data = [time_vector[0], self.ui.value_event_code.text()]
            self.signal_mkr.emit(data)

        stream_processor.subscribe(topic=TOPICS.marker, callback=callback)

    # ########### End Emit Signals Functions ################

    # #######################################################
    # ########### Start Init Plot Functions ################
    def init_plots(self):
        if self.ui.plot_orn.getItem(0, 0) != None:
            self.ui.plot_exg.clear()
            self.ui.plot_fft.clear()
            self.ui.plot_orn.clear()
            self.line = None
            print("cleared plots")

        AppFunctions.init_plot_exg(self)
        AppFunctions.init_plot_orn(self)
        AppFunctions.init_plot_fft(self)

    def init_plot_exg(self):
        # pw = self.ui.graphicsView #testinng
        n_chan_sp = self.explorer.stream_processor.device_info['adc_mask'].count(1)
        n_chan = list(self.chan_dict.values()).count(1)
        if n_chan != n_chan_sp:
            print("ERROR chan count does not match")

        self.offsets = np.arange(1, n_chan + 1)[:, np.newaxis].astype(float)
        timescale = AppFunctions._get_timeScale(self)

        pw = self.ui.plot_exg
        pw.setBackground(Settings.PLOT_BACKGROUND)

        self.active_chan = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
        ticks = [(idx+1, ch) for idx, ch in enumerate(self.active_chan)]

        pw.getAxis("left").setTicks([ticks])

        pw.getAxis("left").setWidth(50)
        pw.showGrid(x=False, y=True, alpha=0.5)
        pw.setRange(yRange=(-0.5, n_chan+1), xRange=(0, int(timescale)), padding=0.01)
        # pw.setRange(yRange=(-0.5, n_chan+1))
        # pw.setRange(yRange=(-2, 1+2), xRange=(0, int(timescale)))
        pw.setLabel("bottom", "time (s)")
        pw.setLabel("left", "Voltage")

        self.curve_ch1 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)
        self.curve_ch2 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)
        self.curve_ch3 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)
        self.curve_ch4 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)
        self.curve_ch5 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)
        self.curve_ch6 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)
        self.curve_ch7 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)
        self.curve_ch8 = pg.PlotCurveItem(pen=Settings.EXG_LINE_COLOR)

        all_curves_list = [
            self.curve_ch1, self.curve_ch2, self.curve_ch3, self.curve_ch4,
            self.curve_ch5, self.curve_ch6, self.curve_ch7, self.curve_ch8
        ]

        self.curves_list = []
        for curve, act in zip(all_curves_list, list(self.chan_dict.values())):
            if act == 1:
                pw.addItem(curve)
                self.curves_list.append(curve)

    def init_plot_orn(self):
        # pw = self.ui.graphicsView #testinng
        pw = self.ui.plot_orn
        pw.setBackground(Settings.PLOT_BACKGROUND)

        timescale = int(AppFunctions._get_timeScale(self))

        self.plot_acc = pw.addPlot()
        pw.nextRow()
        self.plot_gyro = pw.addPlot()
        pw.nextRow()
        self.plot_mag = pw.addPlot()

        self.plots_orn_list = [self.plot_acc, self.plot_gyro, self.plot_mag]

        self.plot_acc.setXLink(self.plot_mag)
        self.plot_gyro.setXLink(self.plot_mag)

        self.plot_acc.getAxis("bottom").setStyle(showValues=False)
        self.plot_gyro.getAxis("bottom").setStyle(showValues=False)

        for plt, lbl in zip(self.plots_orn_list, ['Acc [mg/LSB]', 'Gyro [mdps/LSB]', 'Mag [mgauss/LSB]']):
            plt.addLegend(horSpacing=20, colCount=3, brush="k", offset=(0,-125))
            # plt.addLegend(horSpacing=20, colCount=3, brush="k", offset=(0,0))
            plt.getAxis("left").setWidth(80)
            plt.getAxis("left").setLabel(lbl)
            plt.showGrid(x=True, y=True, alpha=0.5)
            plt.setXRange(0, timescale, padding=0.01)

        self.curve_az = self.plot_acc.plot(pen=Settings.ORN_LINE_COLORS[2], name="accZ")
        self.curve_ay = self.plot_acc.plot(pen=Settings.ORN_LINE_COLORS[1], name="accY")
        self.curve_ax = self.plot_acc.plot(pen=Settings.ORN_LINE_COLORS[0], name="accX")

        self.curve_gx = self.plot_gyro.plot(pen=Settings.ORN_LINE_COLORS[0], name="gyroX")
        self.curve_gy = self.plot_gyro.plot(pen=Settings.ORN_LINE_COLORS[1], name="gyroY")
        self.curve_gz = self.plot_gyro.plot(pen=Settings.ORN_LINE_COLORS[2], name="gyroZ")

        self.curve_mx = self.plot_mag.plot(pen=Settings.ORN_LINE_COLORS[0], name="magX")
        self.curve_my = self.plot_mag.plot(pen=Settings.ORN_LINE_COLORS[1], name="magY")
        self.curve_mz = self.plot_mag.plot(pen=Settings.ORN_LINE_COLORS[2], name="magZ")

    def init_plot_fft(self):

        pw = self.ui.plot_fft
        pw.setBackground(Settings.PLOT_BACKGROUND)
        pw.setXRange(0, 70, padding=0.01)
        pw.showGrid(x=True, y=True, alpha=0.5)
        pw.addLegend(horSpacing=20, colCount=2, brush="k", offset=(0,-300))
        pw.setLabel('left', "Amplitude (uV)")
        pw.setLabel('bottom', "Frequency (Hz)")
        pw.setLogMode(x=False, y=True)

    # ########### End Init Plot Functions ################

    # #######################################################
    # ########### Start Swiping Plot Functions ################
    def plot_exg(self, data):

        # max_points = 100
        max_points = AppFunctions._plot_points(self)
        # if len(self.t_exg_plot)>max_points:
        time_scale = AppFunctions._get_timeScale(self)

        # if not np.isnan(np.sum(self.t_exg_plot)):

        n_new_points = len(data["t"]) 
        # n_new_points = len(data["t"]) + 1 #!!!
        idxs = np.arange(self.exg_pointer, self.exg_pointer+n_new_points)

        self.t_exg_plot.put(idxs, data["t"], mode="wrap")  # replace values with new points
        self.last_t = data["t"][-1]

        for i, ch in enumerate(self.exg_plot.keys()):
            # d = data[ch]
            # TEST FLTERS
            try:
                d = data[ch]
                # d = np.concatenate((data[ch], np.array([np.NaN]))) #!!!

            except KeyError as e:
                d = np.array([np.NaN for i in range(n_new_points)])
                print(str(e))
            self.exg_plot[ch].put(idxs, d, mode="wrap")
            # self.exg_plot[ch][self.exg_pointer+n_new_points]=np.NaN

        self.exg_pointer += n_new_points

        # if wrap happen -> pointer>length:
        if self.exg_pointer >= len(self.t_exg_plot):
            while self.exg_pointer >= len(self.t_exg_plot):
                self.exg_pointer -= len(self.t_exg_plot)

            self.t_exg_plot[self.exg_pointer:] += AppFunctions._get_timeScale(self)

            t_min = int(round(np.mean(data["t"])))
            # t_min = int(data["t"][-1])
            t_max = int(t_min + AppFunctions._get_timeScale(self))
            self.ui.plot_exg.setXRange(t_min, t_max, padding=0.01)

            # Remove marker line and replot in the new axis
            for idx_t in range(len(self.mrk_plot["t"])):
                if self.mrk_plot["t"][idx_t] < self.t_exg_plot[0]:
                    self.ui.plot_exg.removeItem(self.mrk_plot["line"][idx_t])
                    new_data = [
                        self.mrk_plot["t"][idx_t] + AppFunctions._get_timeScale(self), 
                        self.mrk_plot["code"][idx_t]
                    ]
                    AppFunctions.plot_marker(self, new_data, replot=True)

        # Position line:
        if self.line is not None:
            self.line.setPos(data["t"][-1])
            # print(self.line)

            # TEST FILTERS
            # try:
            #     self.line.setPos(data["t"][-1])
            # except RuntimeError as e:
            #     self.line = None
            #     print("line error: ", e)
        else:
            self.line = self.ui.plot_exg.addLine(data["t"][-1], pen="#FF0000")

        # connection = np.full(len(self.t_exg_plot), 1)
        # connection[self.exg_pointer-1:self.exg_pointer] = 0

        # Paint curves
        for curve, ch in zip(self.curves_list, self.active_chan):
            # curve.setData(self.t_exg_plot, self.exg_plot[ch], connect=connection)
            curve.setData(self.t_exg_plot, self.exg_plot[ch], connect="finite")

        # Remove reploted markers
        for idx_t in range(len(self.mrk_replot["t"])):
            if self.mrk_replot["t"][idx_t] < data["t"][-1]:
                self.ui.plot_exg.removeItem(self.mrk_replot["line"][idx_t])

    def plot_marker(self, data, replot=False):
        t, code = data

        if replot is False:
            mrk_dict = self.mrk_plot
            color = Settings.MARKER_LINE_COLOR
        else:
            mrk_dict = self.mrk_replot
            color = Settings.MARKER_LINE_COLOR_ALPHA

        mrk_dict["t"].append(t)
        mrk_dict["code"].append(code)
        pen_marker = pg.mkPen(color=color, dash=[4,4])

        line = self.ui.plot_exg.addLine(t, label=code, pen=pen_marker)
        mrk_dict["line"].append(line)

    def plot_orn(self, data):

        '''time_scale = AppFunctions._get_timeScale(self)

        max_points = AppFunctions._plot_points(self) / (7)
        if len(self.t_orn_plot)>max_points:
        # if len(self.t_orn_plot) and self.t_orn_plot[-1]>time_scale:
            self.t_orn_plot = self.t_orn_plot[1:]
            for k in self.orn_plot.keys():
                self.orn_plot[k] = self.orn_plot[k][1:]
            if len(self.t_orn_plot) - max_points > 0:
                extra = int(len(self.t_orn_plot) - max_points)
                self.t_orn_plot = self.t_orn_plot[extra:]
                for k in self.orn_plot.keys():
                    self.orn_plot[k] = self.orn_plot[k][extra:]

        self.t_orn_plot.extend(data["t"])
        for k in self.orn_plot.keys():
            self.orn_plot[k].extend(data[k])'''
        n_new_points = len(data["t"])
        idxs = np.arange(self.orn_pointer, self.orn_pointer+n_new_points)

        self.t_orn_plot.put(idxs, data["t"], mode="wrap")  # replace values with new points
        # self.last_t_orn = data["t"][-1]

        for k in self.orn_plot.keys():
            self.orn_plot[k].put(idxs, data[k], mode="wrap")

        self.orn_pointer += n_new_points

        # if wrap happen -> pointer>length:
        if self.orn_pointer >= len(self.t_orn_plot):
            while self.orn_pointer >= len(self.t_orn_plot):
                self.orn_pointer -= len(self.t_orn_plot)

            self.t_orn_plot[self.orn_pointer:] += AppFunctions._get_timeScale(self)

            t_min = int(round(np.mean(data["t"])))
            # t_min = int(data["t"][-1])
            t_max = int(t_min + AppFunctions._get_timeScale(self))
            for plt in self.plots_orn_list:
                plt.setXRange(t_min, t_max, padding=0.01)
        # Position line
        if None in self.lines_orn:
            for i, plt in enumerate(self.plots_orn_list):
                self.lines_orn[i] = plt.addLine(data["t"][-1], pen="#FF0000")
        else:
            for line in self.lines_orn:
                try:
                    line.setPos(data["t"][-1])
                except RuntimeError:
                    self.lines_orn = [None, None, None]
                    # pass

        # Paint curves
        self.curve_ax.setData(self.t_orn_plot, self.orn_plot["accX"])
        self.curve_ay.setData(self.t_orn_plot, self.orn_plot["accY"])
        self.curve_az.setData(self.t_orn_plot, self.orn_plot["accZ"])
        self.curve_gx.setData(self.t_orn_plot, self.orn_plot["gyroX"])
        self.curve_gy.setData(self.t_orn_plot, self.orn_plot["gyroY"])
        self.curve_gz.setData(self.t_orn_plot, self.orn_plot["gyroZ"])
        self.curve_mx.setData(self.t_orn_plot, self.orn_plot["magX"])
        self.curve_my.setData(self.t_orn_plot, self.orn_plot["magY"])
        self.curve_mz.setData(self.t_orn_plot, self.orn_plot["magZ"])

    def plot_fft(self):

        pw = self.ui.plot_fft
        pw.clear()
        pw.setXRange(0, 70, padding=0.01)

        exg_fs = self.explorer.stream_processor.device_info['sampling_rate']
        exg_data = np.array([self.exg_plot[key][~np.isnan(self.exg_plot[key])] for key in self.exg_plot.keys()])
        # exg_data = np.array([self.exg_plot[key] for key in self.exg_plot.keys()])

        if (len(exg_data.shape) == 1)or (exg_data.shape[1] < exg_fs * 5):
            return

        fft_content, freq = AppFunctions.get_fft(exg_data, exg_fs)
        # data = dict(zip(self.chan_key_list, fft_content))
        data = dict(zip(self.exg_plot.keys(), fft_content))
        data['f'] = freq

        for i in range(len(data.keys())):
            key = list(data.keys())[i]
            if key != "f":
                pw.plot(data["f"], data[key], pen=Settings.FFT_LINE_COLORS[i], name=key)

    # ########### End Swipping Plot Functions ################

    # #######################################################
    # ########### Start Moving Plot Functions ################
    def plot_exg_moving(self, data):
        
        # max_points = 100
        max_points = AppFunctions._plot_points(self) 
        # if len(self.t_exg_plot)>max_points:

        time_scale = AppFunctions._get_timeScale(self)
        # if len(self.t_exg_plot) and self.t_exg_plot[-1]>time_scale:
        if len(self.t_exg_plot)>max_points:
            # self.plot_ch8.clear()
            # self.curve_ch8 = self.plot_ch8.plot(pen=Settings.EXG_LINE_COLOR)
            new_points = len(data['t'])
            self.t_exg_plot = self.t_exg_plot[new_points:]
            for ch in self.exg_plot.keys():
                self.exg_plot[ch] = self.exg_plot[ch][new_points:]
            
            # Remove marker line
            for idx_t in range(len(self.mrk_plot["t"])):
                if self.mrk_plot["t"][idx_t] < self.t_exg_plot[0]:
                    '''for i, plt in enumerate(self.plots_list):
                        plt.removeItem(self.mrk_plot["line"][idx_t][i])'''
                    self.ui.plot_exg.removeItem(self.mrk_plot["line"][idx_t])

            # Remove rr peaks
            id2remove = []
            for idx_t in range(len(self.r_peak["t"])):
                if self.r_peak["t"][idx_t][0]  < self.t_exg_plot[0]:
                    self.ui.plot_exg.removeItem(self.r_peak["points"][idx_t])
                    id2remove.append(idx_t)
            for idx_t in id2remove:
                self.r_peak["t"].remove(self.r_peak["t"][idx_t])
                self.r_peak["r_peak"].remove(self.r_peak["r_peak"][idx_t])
                self.r_peak["points"].remove(self.r_peak["points"][idx_t])
                    

            # Update axis
            if len(self.t_exg_plot) - max_points > 0:
                extra = int(len(self.t_exg_plot) - max_points)
                self.t_exg_plot = self.t_exg_plot[extra:]
                for ch in self.exg_plot.keys():
                    self.exg_plot[ch] = self.exg_plot[ch][extra:]
                    
        self.t_exg_plot.extend(data["t"])
        for ch in self.exg_plot.keys():
            self.exg_plot[ch].extend(data[ch])

        for curve, ch in zip(self.curves_list, self.active_chan):
            curve.setData(self.t_exg_plot, self.exg_plot[ch])

    def plot_orn_moving(self, data):
        
        time_scale = AppFunctions._get_timeScale(self)

        max_points = AppFunctions._plot_points(self) / 7 #/ (2*7)
        if len(self.t_orn_plot)>max_points:
        # if len(self.t_orn_plot) and self.t_orn_plot[-1]>time_scale:
            self.t_orn_plot = self.t_orn_plot[1:]
            for k in self.orn_plot.keys():
                self.orn_plot[k] = self.orn_plot[k][1:]
            if len(self.t_orn_plot) - max_points > 0:
                extra = int(len(self.t_orn_plot) - max_points)
                self.t_orn_plot = self.t_orn_plot[extra:]
                for k in self.orn_plot.keys():
                    self.orn_plot[k] = self.orn_plot[k][extra:]

        self.t_orn_plot.extend(data["t"])
        for k in self.orn_plot.keys():
            self.orn_plot[k].extend(data[k])

        self.curve_ax.setData(self.t_orn_plot, self.orn_plot["accX"])
        self.curve_ay.setData(self.t_orn_plot, self.orn_plot["accY"])
        self.curve_az.setData(self.t_orn_plot, self.orn_plot["accZ"])
        self.curve_gx.setData(self.t_orn_plot, self.orn_plot["gyroX"])
        self.curve_gy.setData(self.t_orn_plot, self.orn_plot["gyroY"])
        self.curve_gz.setData(self.t_orn_plot, self.orn_plot["gyroZ"])
        self.curve_mx.setData(self.t_orn_plot, self.orn_plot["magX"])
        self.curve_my.setData(self.t_orn_plot, self.orn_plot["magY"])
        self.curve_mz.setData(self.t_orn_plot, self.orn_plot["magZ"])

    def plot_marker_moving(self, data):
        t, code = data
        self.mrk_plot["t"].append(data[0])
        self.mrk_plot["code"].append(data[1])

        pen_marker = pg.mkPen(color='#7AB904', dash=[4,4])
        
        # lines = []
        '''for plt in self.plots_list:
        # plt = self.plot_ch8
            line = self.ui.plot_exg.addLine(t, label=code, pen=pen_marker)
            lines.append(line)'''
        line = self.ui.plot_exg.addLine(t, label=code, pen=pen_marker)
        # lines.append(line)
        # self.mrk_plot["line"].append(lines)
        self.mrk_plot["line"].append(line)

    # ########### End Moving Plot Functions ################
    # ///// END PLOTTING FUNCTIONS/////

    # ////////////////////////////////////////
    # ///// START HELPER FUNCTIONS/////

    def _battery_stylesheet(self, value):
        if isinstance(value, str):
            stylesheet = Settings.BATTERY_STYLESHEETS["na"]
        elif value > 60:
            stylesheet = Settings.BATTERY_STYLESHEETS["high"]
        elif value > 30:
            stylesheet = Settings.BATTERY_STYLESHEETS["medium"]
        elif value > 0:
            stylesheet = Settings.BATTERY_STYLESHEETS["low"]
        else:
            stylesheet = Settings.BATTERY_STYLESHEETS["na"]
        return stylesheet

    def _impedance_stylesheet_wet(self, value):
        """
        Return the stylesheet corresponding to the impedance value
        """
        if type(value) == str:
            imp_stylesheet = Settings.GRAY_IMPEDANCE_STYLESHEET
        elif value > 500:
            imp_stylesheet = Settings.BLACK_IMPEDANCE_STYLESHEET
        elif value > 100:
            imp_stylesheet = Settings.RED_IMPEDANCE_STYLESHEET
        elif value > 50:
            imp_stylesheet = Settings.ORANGE_IMPEDANCE_STYLESHEET
        elif value > 10:
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
        elif value > 500:
            imp_stylesheet = Settings.BLACK_IMPEDANCE_STYLESHEET
        elif value > 200:
            imp_stylesheet = Settings.RED_IMPEDANCE_STYLESHEET
        # elif value > 50:
        #     imp_stylesheet = Settings.ORANGE_IMPEDANCE_STYLESHEET
        elif value > 100:
            imp_stylesheet = Settings.YELLOW_IMPEDANCE_STYLESHEET
        else:
            imp_stylesheet = Settings.GREEN_IMPEDANCE_STYLESHEET

        return imp_stylesheet

    def _update_temperature(self, new_value):
        self.ui.ft_label_temp_value.setText(new_value)

    def _update_light(self, new_value):
        self.ui.ft_label_lux_value.setText(new_value)

    def _update_battery(self, new_value, new_stylesheet):
        self.ui.ft_label_battery_value.setText(new_value)
        self.ui.ft_label_battery_value.setStyleSheet(new_stylesheet)

    def _update_device_name(self, new_value):
        self.ui.ft_label_device_3.setText(new_value)

    def _update_firmware(self, new_value):
        self.ui.ft_label_firmware_value.setText(new_value)

    def _update_impedance(self, chan_dict):
        for chan in chan_dict.keys():
            new_stylesheet = chan_dict[chan][1]
            frame_name = f"frame_{chan}_color"
            ch_frame = AppFunctions._get_widget_by_objName(self, frame_name)
            ch_frame.setStyleSheet(new_stylesheet)

            new_value = chan_dict[chan][0]
            label_name = f"label_{chan}_value"
            ch_label = AppFunctions._get_widget_by_objName(self, label_name)
            ch_label.setText(new_value)

    def _reset_impedance(self):
        if self.is_connected:
            # stream_processor = self.explorer.stream_processor
            # n_chan = stream_processor.device_info['adc_mask']
            # self.chan_dict = dict(zip([c.lower() for c in Settings.CHAN_LIST], n_chan))
            active_chan = [ch for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1]
            chan_dict = {ch: ["NA", AppFunctions._impedance_stylesheet_wet(self, "NA")] for ch in active_chan}
            
            for chan in chan_dict.keys():
                # print(chan_dict[chan])
                new_stylesheet = chan_dict[chan][1]
                frame_name = f"frame_{chan}_color"
                ch_frame = AppFunctions._get_widget_by_objName(self, frame_name)
                ch_frame.setStyleSheet(new_stylesheet)
                # QApplication.processEvents()
                # ch_frame.repaint()

                new_value = chan_dict[chan][0]
                label_name = f"label_{chan}_value"
                ch_label = AppFunctions._get_widget_by_objName(self, label_name)
                ch_label.setText(new_value)
                QApplication.processEvents()

                # ch_label.repaint()
            
            self.ui.frame_ch1_color.setStyleSheet(AppFunctions._impedance_stylesheet_wet(self, "NA"))
            self.ui.label_ch1_value.setText("NA")
            QApplication.processEvents()

        else:
            return

    def _get_widget_by_objName(self, name):
        widgets = QApplication.instance().allWidgets()
        for x in widgets:
            # print(x)
            if str(x.objectName()) == name:
                return x
        print(f"Could not find {name}")
        return None

    def _get_timeScale(self):
        t_str = self.ui.value_timeScale.currentText()
        t = Settings.TIME_RANGE_MENU[t_str]
        return t

    def _get_samplingRate(self):
        stream_processor = self.explorer.stream_processor
        sr = stream_processor.device_info['sampling_rate']
        return sr

    def _plot_points(self, orn=False):
        time_scale = AppFunctions._get_timeScale(self)
        sr = AppFunctions._get_samplingRate(self)
        # points = (time_scale * sr)
        if not orn:
            if self.downsampling:
                points = (time_scale * sr) / (sr / Settings.EXG_VIS_SRATE)
            else:
                points = (time_scale * sr)
        else:
            points = time_scale * Settings.ORN_SRATE
        # points = (time_scale * sr) / (sr / Settings.EXG_VIS_SRATE)
        # points = Settings.EXG_VIS_SRATE * time_scale
        return int(points)

    def _apply_filters(self):
        stream_processor = self.explorer.stream_processor

        if self.plotting_filters is None:
            notch_freq = None
            high_freq = None
            low_freq = None
        else:
            notch_freq = self.plotting_filters["notch"]
            high_freq = self.plotting_filters["highpass"]
            low_freq = self.plotting_filters["lowpass"]

        if notch_freq is not None:
            stream_processor.add_filter(cutoff_freq=notch_freq, filter_type='notch')

        if high_freq is not None and low_freq is not None:
            stream_processor.add_filter(
                cutoff_freq=(low_freq, high_freq), filter_type='bandpass')
        elif high_freq is not None:
            stream_processor.add_filter(cutoff_freq=high_freq, filter_type='highpass')
        elif low_freq is not None:
            stream_processor.add_filter(cutoff_freq=low_freq, filter_type='lowpass')

        print(self.plotting_filters)

    def _check_filters_new_sr(self):
        
        if self.plotting_filters is None:
            return

        # r_value = self.plotting_filters["highpass"]
        # l_value = self.plotting_filters["lowpass"]

        r_value = "" if self.plotting_filters["highpass"] in [None, 'None'] else self.plotting_filters["highpass"]
        l_value = "" if self.plotting_filters["lowpass"] in [None, 'None'] else self.plotting_filters["lowpass"]

        str_value = self.ui.value_sampling_rate.currentText()
        sr = int(str_value)

        nyq_freq = sr / 2.

        max_hc_freq = round(nyq_freq-1, 2)
        min_lc_freq = round(0.003 * nyq_freq, 2)

        warning = ""

        hc_freq_warning = (
            "High cutoff frequency cannot be larger than or equal to the nyquist frequency."
            f"The high cutoff frequency has changed to {max_hc_freq:.2f} Hz!"
            )

        lc_freq_warning = (
            "Transient band for low cutoff frequency was too narrow."
            f"The low cutoff frequency has changed {min_lc_freq:.2f} Hz!"
        )

        if (l_value != "") and (float(l_value) / nyq_freq <= 0.003):
            warning += lc_freq_warning
            self.plotting_filters["lowpass"] = min_lc_freq

        if (r_value != "") and (float(r_value) >= nyq_freq):
            warning += hc_freq_warning
            self.plotting_filters["highpass"] = max_hc_freq
        
        AppFunctions._apply_filters(self)
        if warning != "":
            print(warning)
            AppFunctions._display_msg(self, msg_text=warning, type="info")
        

    def _change_scale(self):
        old = Settings.SCALE_MENU[self.y_string]
        new = Settings.SCALE_MENU[self.ui.value_yAxis.currentText()]

        old_unit = 10 ** (-old)
        new_unit = 10 ** (-new)

        self.y_string = self.ui.value_yAxis.currentText()
        self.y_unit = new_unit

        stream_processor = self.explorer.stream_processor
        self.chan_key_list = [Settings.CHAN_LIST[i].lower()
                              for i, mask in enumerate(reversed(stream_processor.device_info['adc_mask'])) if
                              mask == 1]
        for chan, value in self.exg_plot.items():

            if self.chan_dict[chan] == 1:
                temp_offset = self.offsets[self.chan_key_list.index(chan)]
                # self.exg_plot[chan] = [i * (old_unit / new_unit) for i in val]
                # self.exg_plot[chan] = list((value - temp_offset) * (old_unit / new_unit) + temp_offset)
                self.exg_plot[chan] = (value - temp_offset) * (old_unit / new_unit) + temp_offset

        self.r_peak['r_peak'] = (np.array(self.r_peak['r_peak']) - self.offsets[0]) * \
            (old_unit / self.y_unit) + self.offsets[0]

    @contextmanager
    def _wait_cursor():
        try:
            QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
            yield
        finally:
            QApplication.restoreOverrideCursor()

    def _plot_tabs(self):
        # exg = 0, orn = 1, fft = 2
        tab_idx = int(self.ui.tabWidget.currentIndex())
        if tab_idx == 0:
            self.signal_exg.connect(lambda data: AppFunctions.plot_exg(self, data))
            self.signal_mkr.connect(lambda data: AppFunctions.plot_marker(self, data))

        elif tab_idx == 1:
            self.signal_orn.connect(lambda data: AppFunctions.plot_orn(self, data))

        elif tab_idx == 2:
            self.signal_fft.connect(lambda data: AppFunctions.plot_fft(self, data))

    def _change_timescale(self):
        """current_size = len(self.t_exg_plot)
        new_size = AppFunctions._plot_points(self)
        ts = AppFunctions._get_timeScale(self)

        print(f"{self.t_exg_plot[-1] - ts =}")
        print(f"{np.where(np.isclose(self.t_exg, self.t_exg_plot[-1]-ts))=}")
        print(f"{np.nanmax(self.t_exg)=}\n")

        # df = pandas.DataFrame(data={"t_exg": list(self.t_exg), "t_plot": list(self.t_exg_plot)})
        dict_ = {"t_exg": list(self.t_exg), "t_plot": list(self.t_exg_plot)}
        df = pd.DataFrame({ key:pd.Series(value) for key, value in dict_.items() })
        df.to_csv(f'filename_{self.t_exg_plot[-1]}.csv', sep=',',index=False)

        diff = int(new_size - current_size)
        if diff>0:
            self.t_exg_plot = np.concatenate((np.full((diff,), np.NaN), self.t_exg_plot))
            for ch in self.exg_plot.keys():
                self.exg_plot[ch] = np.concatenate((np.full((diff,), np.NaN), self.exg_plot[ch]))

        elif diff<0:
            diff = abs(diff)
            self.t_exg_plot = self.t_exg_plot[diff:]
            for ch in self.exg_plot.keys():
                self.exg_plot[ch] = self.exg_plot[ch][diff:]

        # print(f"{current_size=}")
        # print(f"{new_size=}")
        # print(f"{len(self.t_exg_plot)=}")

        try:
            t_min = int(np.nanmin(self.t_exg_plot))
        except: #if all nans (at the beginig)
            t_min = 0
        t_max = int(t_min + AppFunctions._get_timeScale(self))
        self.ui.plot_exg.setXRange(t_min,t_max)

        #"""

        # Based on PyCorder approach
        t_min = self.last_t
        t_max = int(t_min + AppFunctions._get_timeScale(self))
        self.ui.plot_exg.setXRange(t_min, t_max, padding=0.01)
        for plt in self.plots_orn_list:
            plt.setXRange(t_min, t_max, padding=0.01)

        new_size = AppFunctions._plot_points(self)
        self.exg_pointer = 0
        self.t_exg_plot = np.array([np.NaN]*new_size)
        self.exg_plot = {ch: np.array([np.NaN]*new_size) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1}

        new_size_orn = AppFunctions._plot_points(self, orn=True)
        self.orn_pointer = 0
        self.t_orn_plot = np.array([np.NaN]*new_size_orn)
        self.orn_plot = {k: np.array([np.NaN]*new_size_orn) for k in Settings.ORN_LIST}

    def get_fft(exg, s_rate):
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

    def _plot_heart_rate(self):
        if self.ui.value_signal.currentText() == "EEG":
            return

        if "ch1" not in self.exg_plot.keys():
            print('Heart rate estimation works only when channel 1 is enabled.')
            msg = "Heart rate estimation works only when channel 1 is enabled."
            # QMessageBox.information(self, "!", msg)
            AppFunctions._display_msg(self, msg_text=msg, type="info")
            return

        # first_chan = self.exg_plot.keys()[0]

        exg_fs = self.explorer.stream_processor.device_info['sampling_rate']

        if self.rr_estimator is None:
            self.rr_estimator = HeartRateEstimator(fs=exg_fs)
            
            '''# Init R-peaks plot
            self.exg_plot.circle(x='t', y='r_peak', source=self._r_peak_source,
                                 fill_color="red", size=8)'''

        ecg_data = (np.array(self.exg_plot['ch1'])[-2 * Settings.EXG_VIS_SRATE:] - self.offsets[0]) * self.y_unit
        # ecg_data = (np.array(self.exg_plot[first_chan])[-2 * Settings.EXG_VIS_SRATE:] - self.offsets[0]) * self.y_unit
        time_vector = np.array(self.t_exg_plot)[-2 * Settings.EXG_VIS_SRATE:]

        # Check if the peak2peak value is bigger than threshold
        if (np.ptp(ecg_data) < Settings.V_TH[0]) or (np.ptp(ecg_data) > Settings.V_TH[1]):
            print("P2P value larger or less than threshold. Cannot compute heart rate!")
            return

        peaks_time, peaks_val = self.rr_estimator.estimate(ecg_data, time_vector)
        peaks_val = (np.array(peaks_val) / self.y_unit) + self.offsets[0]
        if peaks_time:
            self.r_peak['t'].append(peaks_time)
            list(self.r_peak['r_peak']).append(peaks_val)

            points = self.ui.plot_exg.plot(peaks_time, peaks_val,
                        pen = None, symbolBrush =(200, 0, 0), symbol ='o', symbolSize = 8)
            
            self.r_peak["points"].extend([points])
            
            """self.r_peak['t'].extend(peaks_time)
            self.r_peak['r_peak'].extend(peaks_val)

            for idx in range(len(self.r_peak['t'])):
                t = np.array([self.r_peak['t'][idx]])
                peak = np.array([self.r_peak['r_peak'][idx]])
                point = pg.PlotCurveItem(t, peak, 
                        pen = None, symbolBrush =(200, 0, 0), symbol ='o', symbolSize = 8)
                '''self.ui.plot_exg.plot(t, peak, 
                        pen = None, symbolBrush =(200, 0, 0), symbol ='o', symbolSize = 8)
                '''
                self.ui.plot_exg.addItem(point)
                self.r_peak["points"].extend([point])
            # print(dict(zip(['r_peak', 't'], [peaks_val, peaks_time])))"""
            print(dict(zip(['r_peak', 't'], [peaks_val, peaks_time])))

        # Update heart rate cell
        estimated_heart_rate = self.rr_estimator.heart_rate
        self.ui.value_heartRate.setText(str(estimated_heart_rate))

    def _display_msg(self, msg_text, title=None, type="error"):
        # msg = QMessageBox.critical(self, title="Error", text=msg)
        msg = QMessageBox()
        msg.setText(msg_text)
        msg.setStyleSheet(Settings.POPUP_STYLESHEET)
        
        if type == "error":
            wdw_title = "Error" if title is None else title
            msg.setIcon(QMessageBox.Critical)
        elif type == "info":
            wdw_title = "!" if title is None else title
            msg.setIcon(QMessageBox.Information)
        elif type == "question":
            wdw_title = "Confirmation" if title is None else title
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setIcon(QMessageBox.Question)
        
        msg.setWindowTitle(wdw_title)
        response = msg.exec()
        return response
    

    def _set_n_chan(self):

        self.n_chan = int(self.ui.n_chan.currentText())
        self.chan_list = Settings.CHAN_LIST[:self.n_chan]
        
        # og_mask = self.explorer.stream_processor.device_info['adc_mask']
        # chan = og_mask.count(1)
        # if chan > 8:
        #     self.n_chan = 16
        # elif chan > 4:
        #     self.n_chan = 8
        # else:
        #     test_mask = int("11111111", 2)
        #     self.explorer.set_channels(test_mask)
        #     new_mask = self.explorer.stream_processor.device_info['adc_mask']
        #     if new_mask == og_mask: 
        #         # if the mask is the same after "activating 8 chan the device is 4chan"
        #         self.n_chan = 4
        #     else:
        #         self.n_chan = 8
        print(f"{self.n_chan=}")

    def _on_connection(self):
        # set number of channels:
        # self.n_chan = len(self.explorer.stream_processor.device_info['adc_mask'])
        AppFunctions._set_n_chan(self)
        # self.n_chan = 8
        # self.chan_list = Settings.CHAN_LIST[:self.n_chan]

        # change footer & button text:            
        AppFunctions.change_footer(self)
        AppFunctions.change_btn_connect(self)

        # if self.is_connected:
        # AppFunctions.info_device(self)
        AppFunctions.info_device(self)

        # (un)hide settings frame
        AppFunctions.update_frame_dev_settings(self)

        # init plots and impedances
        AppFunctions.init_plots(self)
        AppFunctions.init_imp(self)

    def _disconnect_signals(self):
        self.signal_exg.disconnect(self._lambda_exg)
        self.signal_orn.disconnect(self._lambda_orn)
        self.signal_mkr.disconnect(self._lambda_marker)

    def _connect_signals(self):
        self.signal_exg.connect(self._lambda_exg)
        self.signal_orn.connect(self._lambda_orn)
        self.signal_mkr.connect(self._lambda_marker)


            

    # ///// END FUNCTIONS/////


class Plots(MainWindow):

    def __init__(self, vis_mode, paths, plot_widget, time_scale):
        # super(Plots, self).__init__()
        self.vis_mode = vis_mode
        self.paths = paths

        self.data = self.read_data(paths)
        self.layoutWidget = plot_widget
        self.time_scale = time_scale
        self.is_started = False

        if self.vis_mode == "exg":
            self.init_plot_exg()
            self.init_data_exg()
            self.config_plot_exg()
            self.start_curves_exg()

        elif self.vis_mode == "orn":
            self.init_plot_orn()
            self.init_data_orn()
            self.config_plot_orn()
            self.start_curves_orn()

        if self.vis_mode == "fft":
            # self.init_plot_fft()
            self.init_data_fft()
            self.config_plot_fft()
            self.start_curves_fft()



    def init_plot_exg(self):
        '''
        Initialize exg plot
        '''
        self.plot_ch8 = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_ch7 = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_ch6 = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_ch5 = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_ch4 = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_ch3 = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_ch2 = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_ch1 = self.layoutWidget.addPlot()

        self.plots_list = [
            self.plot_ch1, self.plot_ch2, self.plot_ch3, self.plot_ch4,
            self.plot_ch5, self.plot_ch6, self.plot_ch7, self.plot_ch8
        ]

    def init_plot_orn(self):
        '''
        Initialize orientation plot
        '''
        self.plot_acc = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_gyro = self.layoutWidget.addPlot()
        self.layoutWidget.nextRow()
        self.plot_mag = self.layoutWidget.addPlot()

    def init_data_exg(self):
        '''
        Initialize exg data
        '''
        self.data_exg = self.data["exg"]
        self.t_exg = self.data_exg["t"]
        self.ch1 = self.data_exg["ch1"]
        self.ch2 = self.data_exg["ch2"]
        self.ch3 = self.data_exg["ch3"]
        self.ch4 = self.data_exg["ch4"]
        self.ch5 = self.data_exg["ch5"]
        self.ch6 = self.data_exg["ch6"]
        self.ch7 = self.data_exg["ch7"]
        self.ch8 = self.data_exg["ch8"]

        self.data_list = [
            self.t_exg,
            self.ch1, self.ch2, self.ch3, self.ch4,
            self.ch5, self.ch6, self.ch7, self.ch8
        ]

        if self.time_scale is None:
            self.windowWidth_exg = len(self.t_exg)
        else:
            self.windowWidth_exg = self.time_scale_points(self.t_exg)
        self.t_plot = list(self.t_exg[:self.windowWidth_exg])
        self.ch1_plot = list(self.ch1[:self.windowWidth_exg])
        self.ch2_plot = list(self.ch2[:self.windowWidth_exg])
        self.ch3_plot = list(self.ch3[:self.windowWidth_exg])
        self.ch4_plot = list(self.ch4[:self.windowWidth_exg])
        self.ch5_plot = list(self.ch5[:self.windowWidth_exg])
        self.ch6_plot = list(self.ch6[:self.windowWidth_exg])
        self.ch7_plot = list(self.ch7[:self.windowWidth_exg])
        self.ch8_plot = list(self.ch8[:self.windowWidth_exg])

        print("exg data init")

    def init_data_orn(self):
        '''
        Initialize orientation data
        '''
        self.data_orn = self.data["orn"]
        self.t_orn = self.data_orn["t"]

        self.ax = self.data_orn["ax"]
        self.ay = self.data_orn["ay"]
        self.az = self.data_orn["az"]

        self.gx = self.data_orn["gx"]
        self.gy = self.data_orn["gy"]
        self.gz = self.data_orn["gz"]

        self.mx = self.data_orn["mx"]
        self.my = self.data_orn["my"]
        self.mz = self.data_orn["mz"]

        if self.time_scale is None:
            self.windowWidth_orn = len(self.t_orn)
        else:
            self.windowWidth_orn = self.time_scale_points(self.t_orn)

        self.t_plot_orn = list(self.t_orn[:self.windowWidth_orn])

        self.ax_plot = list(self.ax[:self.windowWidth_orn])
        self.ay_plot = list(self.ay[:self.windowWidth_orn])
        self.az_plot = list(self.az[:self.windowWidth_orn])

        self.gx_plot = list(self.gx[:self.windowWidth_orn])
        self.gy_plot = list(self.gy[:self.windowWidth_orn])
        self.gz_plot = list(self.gz[:self.windowWidth_orn])

        self.mx_plot = list(self.mx[:self.windowWidth_orn])
        self.my_plot = list(self.my[:self.windowWidth_orn])
        self.mz_plot = list(self.mz[:self.windowWidth_orn])

    def init_data_fft(self):
        '''
        Initialize FFT data
        '''
        self.data_exg = self.data["exg"]
        exg_array = np.array([self.data_exg[i] for i in self.data_exg.columns[1:]])
        self.data_fft, self.f = AppFunctions.get_fft(exg_array, s_rate=250)

        self.ch1 = self.data_fft[1-1]
        self.ch2 = self.data_fft[2-1]
        self.ch3 = self.data_fft[3-1]
        self.ch4 = self.data_fft[4-1]
        self.ch5 = self.data_fft[5-1]
        self.ch6 = self.data_fft[6-1]
        self.ch7 = self.data_fft[7-1]
        self.ch8 = self.data_fft[8-1]

        self.data_list = [
            self.ch1, self.ch2, self.ch3, self.ch4,
            self.ch5, self.ch6, self.ch7, self.ch8
        ]

    def config_plot_fft(self):
        '''
        Styling FFT plot
        '''
        min_y = min([min(ch) for ch in self.data_list])
        max_y = max([max(ch) for ch in self.data_list])

        self.layoutWidget.setYRange(min_y, max_y)
        self.layoutWidget.setXRange(0, 70)
        self.layoutWidget.showGrid(x=True, y=True, alpha=0.5)
        self.layoutWidget.addLegend(horSpacing=20, colCount=2, brush="k", offset=(0,-300))
        self.layoutWidget.setLabel('left', "Amplitude (uV)")
        self.layoutWidget.setLabel('bottom', "Frequency (Hz)")
        self.layoutWidget.setLogMode(x=False, y=True)

    def config_plot_exg(self):
        '''
        Styling exg plot
        '''
        min_y = min([min(ch) for ch in self.data_list[1:]])
        max_y = max([max(ch) for ch in self.data_list[1:]])

        for idx, plt in enumerate(self.plots_list):
            plt.setYRange(min_y, max_y)
            plt.getAxis("left").setTicks([[(0, f"ch{idx+1}")]])
            plt.showGrid(x=False, y=True, alpha=0.5)
            if idx != 0:
                plt.setXLink(self.plot_ch1)
                plt.getAxis("bottom").setStyle(showValues=False)
                plt.hideAxis("bottom")

    def config_plot_orn(self):
        '''
        Styling orientation plot
        '''
        self.plot_acc.setXLink(self.plot_mag)
        self.plot_gyro.setXLink(self.plot_mag)

        self.plot_acc.setYRange(
            min([min(self.ax), min(self.ay), min(self.az)]),
            max([max(self.ax), max(self.ay), max(self.az)]))
        self.plot_acc.getAxis("bottom").setStyle(showValues=False)
        self.plot_acc.addLegend(horSpacing=20, colCount=3, brush="k")

        self.plot_gyro.setYRange(
            min([min(self.gx), min(self.gy), min(self.gz)]),
            max([max(self.gx), max(self.gy), max(self.gz)]))
        self.plot_gyro.getAxis("bottom").setStyle(showValues=False)
        self.plot_gyro.addLegend(horSpacing=20, colCount=3, brush="k")

        self.plot_mag.setYRange(
            min([min(self.mx), min(self.my), min(self.mz)]),
            max([max(self.mx), max(self.my), max(self.mz)]))
        self.plot_mag.addLegend(horSpacing=20, colCount=3, brush="k")

    def start_curves_exg(self):
        '''
        Start curves for exg plot
        '''
        '''for plt in self.plots_list:
            plt.clear()'''

        self.curve_ch1 = self.plot_ch1.plot(self.t_plot, self.ch1_plot, pen=Settings.EXG_LINE_COLOR)
        self.curve_ch2 = self.plot_ch2.plot(self.t_plot, self.ch2_plot, pen=Settings.EXG_LINE_COLOR)
        self.curve_ch3 = self.plot_ch3.plot(self.t_plot, self.ch3_plot, pen=Settings.EXG_LINE_COLOR)
        self.curve_ch4 = self.plot_ch4.plot(self.t_plot, self.ch4_plot, pen=Settings.EXG_LINE_COLOR)
        self.curve_ch5 = self.plot_ch5.plot(self.t_plot, self.ch5_plot, pen=Settings.EXG_LINE_COLOR)
        self.curve_ch6 = self.plot_ch6.plot(self.t_plot, self.ch6_plot, pen=Settings.EXG_LINE_COLOR)
        self.curve_ch7 = self.plot_ch7.plot(self.t_plot, self.ch7_plot, pen=Settings.EXG_LINE_COLOR)
        self.curve_ch8 = self.plot_ch8.plot(self.t_plot, self.ch8_plot, pen=Settings.EXG_LINE_COLOR)

    def start_curves_orn(self):
        '''
        Start curves for orn plot
        '''
        pen = pg.mkPen(color=(255, 0, 0))
        self.curve_ax = self.plot_acc.plot(
            self.t_plot_orn, self.ax_plot, pen=Settings.ORN_LINE_COLORS[0], name="accX")
        self.curve_ay = self.plot_acc.plot(
            self.t_plot_orn, self.ay_plot, pen=Settings.ORN_LINE_COLORS[1], name="accY")
        self.curve_az = self.plot_acc.plot(
            self.t_plot_orn, self.az_plot, pen=Settings.ORN_LINE_COLORS[2], name="accZ")

        self.curve_gx = self.plot_gyro.plot(
            self.t_plot_orn, self.gx_plot, pen=Settings.ORN_LINE_COLORS[0], name="gyroX")
        self.curve_gy = self.plot_gyro.plot(
            self.t_plot_orn, self.gy_plot, pen=Settings.ORN_LINE_COLORS[1], name="gyroY")
        self.curve_gz = self.plot_gyro.plot(
            self.t_plot_orn, self.gz_plot, pen=Settings.ORN_LINE_COLORS[2], name="gyroZ")

        self.curve_mx = self.plot_mag.plot(
            self.t_plot_orn, self.mx_plot, pen=Settings.ORN_LINE_COLORS[0], name="magX")
        self.curve_my = self.plot_mag.plot(
            self.t_plot_orn, self.my_plot, pen=Settings.ORN_LINE_COLORS[1], name="magY")
        self.curve_mz = self.plot_mag.plot(
            self.t_plot_orn, self.mz_plot, pen=Settings.ORN_LINE_COLORS[2], name="magZ")

    def start_curves_fft(self):
        '''
        Start curves for fft plot
        '''
        self.curve_ch1 = self.layoutWidget.plot(self.f, self.ch1, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[0]), name="ch1")
        self.curve_ch2 = self.layoutWidget.plot(self.f, self.ch2, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[1]), name="ch2")
        self.curve_ch3 = self.layoutWidget.plot(self.f, self.ch3, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[2]), name="ch3")
        self.curve_ch4 = self.layoutWidget.plot(self.f, self.ch4, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[3]), name="ch4")
        self.curve_ch5 = self.layoutWidget.plot(self.f, self.ch5, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[4]), name="ch5")
        self.curve_ch6 = self.layoutWidget.plot(self.f, self.ch6, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[5]), name="ch6")
        self.curve_ch7 = self.layoutWidget.plot(self.f, self.ch7, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[6]), name="ch7")
        self.curve_ch8 = self.layoutWidget.plot(self.f, self.ch8, pen=pg.mkPen(color=Settings.FFT_LINE_COLORS[7]), name="ch8")

    def update_plot_exg(self):
        '''
        Update exg plot
        '''

        self.t_plot = self.t_plot[1:]
        try:
            self.t_plot.append(self.t_exg[self.windowWidth_exg + 1])
        except KeyError:
            return

        self.ch1_plot = self.ch1_plot[1:]  # Remove the first
        self.ch1_plot.append(self.ch1[self.windowWidth_exg + 1])  # Add a new value
        self.curve_ch1.setData(self.t_plot, self.ch1_plot)  # Update the data.

        self.ch2_plot = self.ch2_plot[1:]
        self.ch2_plot.append(self.ch2[self.windowWidth_exg + 1])
        self.curve_ch2.setData(self.t_plot, self.ch2_plot)

        self.ch3_plot = self.ch3_plot[1:]
        self.ch3_plot.append(self.ch3[self.windowWidth_exg + 1])
        self.curve_ch3.setData(self.t_plot, self.ch3_plot)

        self.ch4_plot = self.ch4_plot[1:]
        self.ch4_plot.append(self.ch4[self.windowWidth_exg + 1])
        self.curve_ch4.setData(self.t_plot, self.ch4_plot)

        self.ch5_plot = self.ch5_plot[1:]
        self.ch5_plot.append(self.ch5[self.windowWidth_exg + 1])
        self.curve_ch5.setData(self.t_plot, self.ch5_plot)

        self.ch6_plot = self.ch6_plot[1:]
        self.ch6_plot.append(self.ch6[self.windowWidth_exg + 1])
        self.curve_ch6.setData(self.t_plot, self.ch6_plot)

        self.ch7_plot = self.ch7_plot[1:]
        self.ch7_plot.append(self.ch7[self.windowWidth_exg + 1])
        self.curve_ch7.setData(self.t_plot, self.ch7_plot)

        self.ch8_plot = self.ch8_plot[1:]
        self.ch8_plot.append(self.ch8[self.windowWidth_exg + 1])
        self.curve_ch8.setData(self.t_plot, self.ch8_plot)

        self.windowWidth_exg += 1

        '''if self.windowWidth_exg >= len(self.ch1)-1:
            self.timer_exg.stop()'''
        
        if self.windowWidth_exg >= len(self.t_exg)-1:
            return

    def update_plot_orn(self):
        '''
        Update orientation plot
        '''

        
        self.t_plot_orn = self.t_plot_orn[1:]
        try:
            self.t_plot_orn.append(self.t_orn[self.windowWidth_orn + 1])
        except KeyError:
            return

        # ########### ACC PLOT ############
        self.ax_plot = self.ax_plot[1:]
        self.ax_plot.append(self.ax[self.windowWidth_orn + 1])
        self.curve_ax.setData(self.t_plot_orn, self.ax_plot)

        self.ay_plot = self.ay_plot[1:]
        self.ay_plot.append(self.ay[self.windowWidth_orn + 1])
        self.curve_ay.setData(self.t_plot_orn, self.ay_plot)

        self.az_plot = self.az_plot[1:]
        self.az_plot.append(self.az[self.windowWidth_orn + 1])
        self.curve_az.setData(self.t_plot_orn, self.az_plot)
        # ###########  ############

        # ########### GYRO PLOT ############
        self.gx_plot = self.gx_plot[1:]
        self.gx_plot.append(self.gx[self.windowWidth_orn + 1])
        self.curve_gx.setData(self.t_plot_orn, self.gx_plot)

        self.gy_plot = self.gy_plot[1:]
        self.gy_plot.append(self.gy[self.windowWidth_orn + 1])
        self.curve_gy.setData(self.t_plot_orn, self.gy_plot)

        self.gz_plot = self.gz_plot[1:]
        self.gz_plot.append(self.gz[self.windowWidth_orn + 1])
        self.curve_gz.setData(self.t_plot_orn, self.gz_plot)
        # ###########  ############

        # ########### MAG PLOT ############
        self.mx_plot = self.mx_plot[1:]
        self.mx_plot.append(self.mx[self.windowWidth_orn + 1])
        self.curve_mx.setData(self.t_plot_orn, self.mx_plot)

        self.my_plot = self.my_plot[1:]
        self.my_plot.append(self.my[self.windowWidth_orn + 1])
        self.curve_my.setData(self.t_plot_orn, self.my_plot)

        self.mz_plot = self.mz_plot[1:]
        self.mz_plot.append(self.mz[self.windowWidth_orn + 1])
        self.curve_mz.setData(self.t_plot_orn, self.mz_plot)
        # ###########  ############

        self.windowWidth_orn += 1

        if self.windowWidth_orn >= len(self.ax)-1:
            return

    def read_data(self, paths):
        '''
        Read data from recorded csv or edf files
        Args:
            paths (list): list of files to read
        '''
        data = {}
        for file in paths:
            if file.endswith(".csv"):
                if "ExG" in file:
                    data["exg"] = self._read_csv(file)
                elif "Marker" in file:
                    try:
                        data["marker"] = self._read_csv(file)
                    except:
                        data["marker"] = None
                elif "ORN" in file:
                    data["orn"] = self._read_csv(file)
                else:
                    print("Not a exg, marker or ORN file")
            elif file.endswith(".edf"):
                print("Not supported yet")
            else:
                print("Only csv and edf files")

        return data

    def time_scale_points(self, time_vec, time=10):
        '''
        Compute points in window based onn time scale
        '''
        time_str = self.time_scale
        time = Settings.TIME_RANGE_MENU[time_str]
        idx = len(time_vec)-2
        for i in range(len(time_vec)):
            if time_vec[i] > time:
                idx = i
                return idx
        # print(idx)
        return idx

    def _read_csv(self, file):
        df = pd.read_csv(file)
        df["t"] = df["TimeStamp"]-df["TimeStamp"][0]
        return df

    def _read_edf(self, file):
        data = mne.io.read_raw_edf(file)
        raw_data = data.get_data()
        cols = data.ch_names
        data = dict(zip(cols, raw_data))
        df = pd.DataFrame.from_dict(data)
        df["t"] = df["TimeStamp"]-df["TimeStamp"][0]
        return df

    def _read_bin(self, file):
        pass

    def _update_plot_data(self, data, data_plot):
        data_plot = data_plot[1:]  # Remove the first
        data_plot.append(data[self.windowWidth_exg + 1])  # Add a new value
