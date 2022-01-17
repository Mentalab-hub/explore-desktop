# from PySide6.QtCore import Signal
import os

import explorepy._exceptions as xpy_ex
import numpy as np
from exploregui.modules.app_functions import AppFunctions
from exploregui.modules.app_settings import Settings
from explorepy.stream_processor import TOPICS
from explorepy.tools import bt_scan
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox
)


# DISABLED_STYLESHEET = """
#     background-color: rgb(129,133,161);
#     color: rgb(78,78,78);
# """
DISABLED_STYLESHEET = """
    background-color: rgb(89,90,111);
    color: rgb(155,155,155);
"""


class BTFunctions(AppFunctions):
    """
    Functions for Bluetooth connection
    """

    def __init__(self, ui, explorer):
        super().__init__(ui, explorer)
        # self.ui = ui
        # self.explorer = explorer
        self._battery_percent_list = []

    #########################
    # Scan/Connect Functions
    #########################

    def scan_devices(self):
        """"
        Scans for available explore devices.
        """
        self.ui.list_devices.clear()

        # Change footer
        self.ui.ft_label_device_3.setText("Scanning ...")
        self.ui.ft_label_device_3.repaint()

        # Change scan button
        self.ui.btn_scan.setText("Scanning")
        self.ui.btn_scan.setStyleSheet(DISABLED_STYLESHEET)
        QApplication.processEvents()

        try:
            with self.wait_cursor():
                explore_devices = bt_scan()
                pass
        except ValueError:
            msg = "Error opening socket.\nPlease make sure the bluetooth is on."
            self._connection_error_gui(msg, scan=True)
            return
        except SystemError as e:
            msg = str(e)
            msg += "\nPlease make sure the Bluetooth is on."
            self._connection_error_gui(msg, scan=True)
            return
        explore_devices = [dev[0] for dev in explore_devices]

        if len(explore_devices) == 0:
            msg = "No explore devices found. Please make sure your device is turned on."
            # if os.name == 'nt':
            #     msg = msg[:-1]
            #     msg += " and the Bluetooth is active."
            self._connection_error_gui(msg, scan=True)
            return

        self.ui.list_devices.addItems(explore_devices)

        # Reset footer
        self.ui.ft_label_device_3.setText("Not connected")
        self.ui.ft_label_device_3.repaint()

        # Reset button
        self.ui.btn_scan.setText("Scan")
        self.ui.btn_scan.setStyleSheet("")
        QApplication.processEvents()

    def get_device_from_le(self):

        input_name = self.ui.dev_name_input.text()

        if not input_name.startswith("Explore_") and len(input_name) == 4:
            device_name = "Explore_" + input_name
        elif input_name.startswith("Explore_"):
            device_name = input_name
        else:
            device_name = ""

        if len(device_name) != 12:
            device_name = ""

        return device_name

    def get_device_from_list(self):
        try:
            device_name = self.ui.list_devices.selectedItems()[0].text()
        except IndexError:
            device_name = ""

        return device_name

    def connect2device(self, dev_name=None):
        """
        Connect to the explore device.
        """
        if self.is_connected is False:
            device_name_le = self.get_device_from_le()
            device_name_list = self.get_device_from_list()

            if dev_name is not None:
                device_name = dev_name
            elif device_name_le != "":
                device_name = device_name_le
            elif device_name_list != "":
                device_name = device_name_list
            else:
                msg = "Please select a device or provide a valid name (Explore_XXXX or XXXX) before connecting."
                # QMessageBox.critical(self, "Error", msg)
                self.display_msg(msg)
                return

            # Change footer
            self.ui.ft_label_device_3.setText(f"Connecting to {device_name}...")
            self.ui.ft_label_device_3.adjustSize()
            self.ui.ft_label_device_3.repaint()

            # Change connect button
            self.ui.btn_connect.setText("Connecting")
            self.ui.btn_connect.setStyleSheet(DISABLED_STYLESHEET)

            # If platform is windows, add instructions
            if os.name == "nt":
                self.ui.lbl_bt_instructions.setText("Follow Windows' instructions to pair your device.")
                self.ui.lbl_bt_instructions.show()
            QApplication.processEvents()

            try:
                with self.wait_cursor():
                    self.explorer.connect(device_name=device_name)
                    self.is_connected = True
                    AppFunctions.is_connected = self.is_connected
                    pass

            except xpy_ex.DeviceNotFoundError as e:
                msg = str(e)
                # if os.name == "nt":
                #     msg += "\nPlease make sure both the Bluetooth and your device are turned ON."
                self._connection_error_gui(msg)
                return
            except TypeError or UnboundLocalError:
                msg = "Please select a device or provide a valid name (Explore_XXXX or XXXX) before connecting."
                # if os.name == "nt":
                #     msg = msg[:-1]
                #     msg += " and make sure that the Bluetooth is on."
                self._connection_error_gui(msg)
                return
            except AssertionError as e:
                msg = str(e)
                self._connection_error_gui(msg)
                return
            except ValueError:
                msg = "Error opening socket.\nPlease make sure the bluetooth is on."
                self._connection_error_gui(msg)
                return
            except SystemError as e:
                msg = str(e)
                msg += "\nPlease make sure the Bluetooth is on."
                self._connection_error_gui(msg)
                return
            except Exception as e:
                msg = str(e)
                self._connection_error_gui(msg)
                return

        else:
            try:
                self.explorer.disconnect()
                self.is_connected = False
                AppFunctions.is_connected = self.is_connected

            except AssertionError as e:
                msg = str(e)
                self.display_msg(msg)
            except Exception as e:
                msg = str(e)
                self.display_msg(msg)

        print(self.is_connected)
        self.ui.btn_connect.setStyleSheet("")
        self.on_connection()
        self.ui.lbl_bt_instructions.hide()

    #########################
    # Visual feedback functions
    #########################
    def _connection_error_gui(self, msg, scan=False):
        """
        Visual feedback when the scan or connection to device is not successful
        """
        # reset footer to Not connected
        self.ui.ft_label_device_3.setText("Not connected")

        # hide instructions
        self.ui.lbl_bt_instructions.hide()

        if scan is False:
            # change button stylesheet
            self.ui.btn_connect.setStyleSheet("")
            self.ui.btn_connect.setText("Connect")
        else:
            self.ui.btn_scan.setStyleSheet("")
            self.ui.btn_scan.setText("Scan")
        QApplication.processEvents()

        # display error message
        self.display_msg(msg)

    def on_connection(self):
        # set number of channels:
        self.set_n_chan()
        # self.n_chan = 8
        # self.chan_list = Settings.CHAN_LIST[:self.n_chan]

        # change footer & button text:
        self.change_footer()
        self.change_btn_connect()

        # if self.self.is_connected:
        # AppFunctions.info_device(self)
        self.info_device()

        # (un)hide settings frame
        self.update_frame_dev_settings()

        # init plots and impedances
        # self.init_plots()
        self.init_imp()

        self.ui.line_2.setHidden(True)

    def change_btn_connect(self):
        '''
        Change connect buttonn to Connect/Disconnect depending on explore status
        '''
        if self.is_connected:
            # self.signalConnectBtnText.emit("Disconnect")
            self.ui.btn_connect.setText("Disconnect")

        else:
            # self.signalConnectBtnText.emit("Connect")
            self.ui.btn_connect.setText("Connect")

    def change_footer(self):
        """
        Change the fields for device and firmware in the GUI footer
        """
        if self.is_connected:
            self.ui.ft_label_firmware.setHidden(False)
            self.ui.ft_label_firmware_value.setHidden(False)
            self.ui.ft_label_battery.setHidden(False)
            self.ui.ft_label_battery_value.setHidden(False)
            self.ui.ft_label_temp.setHidden(False)
            self.ui.ft_label_temp_value.setHidden(False)

            dev_name = self.explorer.stream_processor.device_info[
                "device_name"]
            device_lbl = f"Connected to {dev_name}"
            firmware = self.explorer.stream_processor.device_info[
                "firmware_version"]
            self._update_device_name(new_value=device_lbl)
            self._update_firmware(new_value=firmware)
        else:
            device_lbl = "Not connected"
            self._update_device_name(new_value=device_lbl)

            self.ui.ft_label_firmware.setHidden(True)
            self.ui.ft_label_firmware_value.setHidden(True)
            self.ui.ft_label_battery.setHidden(True)
            self.ui.ft_label_battery_value.setHidden(True)
            self.ui.ft_label_temp.setHidden(True)
            self.ui.ft_label_temp_value.setHidden(True)
            # stylesheet = self._battery_stylesheet(value="NA")
            # self._update_battery("NA", new_stylesheet=stylesheet)
            # self._update_temperature("NA")

    def update_frame_dev_settings(self):
        """
        Update the frame with the device settings.
        Only shown if a device is connected
        """

        if self.is_connected:
            stream_processor = self.explorer.stream_processor

            # ///// CONFIGURE DEVICE FRAME /////
            # Set device name
            self.ui.label_explore_name.setText(
                stream_processor.device_info["device_name"])

            # Set active channels
            chan = stream_processor.device_info['adc_mask']
            chan = [i for i in reversed(chan)]

            self.chan_dict = dict(zip([c.lower() for c in Settings.CHAN_LIST], chan))
            AppFunctions.chan_dict = self.chan_dict

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

            points = self.plot_points()
            points_wo_downsampl = self.plot_points(downsampling=False)
            self.exg_plot_data[0] = np.array([np.NaN] * points)
            self.exg_plot_data[1] = {
                ch: np.array([np.NaN] * points) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1
            }
            self.exg_plot_data[2] = {
                ch: np.array([np.NaN] * points_wo_downsampl) for ch in self.chan_dict.keys() if self.chan_dict[ch] == 1
            }
            AppFunctions.exg_plot_data = self.exg_plot_data

            # Set sampling rate (value_sampling_rate)
            sr = stream_processor.device_info['sampling_rate']
            self.ui.value_sampling_rate.setCurrentText(str(int(sr)))
            # ////////

            # Show the device frame
            self.ui.frame_device.show()
            self.ui.line_2.show()

        else:
            # self.ui.frame_device.hide()
            # self.ui.line_2.hide()
            pass

    def info_device(self):
        r"""
        Update device information
        """

        def callback(packet):
            # print(packet)
            new_info = packet.get_data()
            # print(new_info)
            for key in new_info.keys():
                if key == "temperature":
                    new_value = str(new_info[key][0]) if self.is_connected else "NA"
                    self._update_temperature(new_value=new_value)
                    # print("\n\ntemperature", new_info[key])

                elif key == "battery":
                    self._battery_percent_list.append(new_info[key][0])
                    if len(self._battery_percent_list) > Settings.BATTERY_N_MOVING_AVERAGE:
                        del self._battery_percent_list[0]
                    value = int(np.mean(self._battery_percent_list))
                    value = 1 if value < 1 else value
                    new_value = value if self.is_connected else "NA"
                    stylesheet = self._battery_stylesheet(value=new_value)
                    self._update_battery(new_value=str(new_value), new_stylesheet=stylesheet)

                elif key == "fimrware":
                    # print("firmware callback: ", new_info[key])
                    new_value = new_info[key] if self.is_connected else "NA"
                    self._update_firmware(new_value=new_value)

                elif key == "device_name":
                    # print("name callback: ", new_info[key])
                    connected_lbl = f"Connected to {new_info[key]}"
                    not_connected_lbl = "Not connected"
                    new_value = connected_lbl if self.is_connected else not_connected_lbl
                    self._update_device_name(new_value=new_value)

            QApplication.processEvents()

        self.explorer.stream_processor.subscribe(callback=callback, topic=TOPICS.device_info)
        self.explorer.stream_processor.subscribe(callback=callback, topic=TOPICS.env)

    #########################
    # Updater Functions
    #########################

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

    #########################
    # Reset Functions
    #########################

    def reset_bt_vars(self):
        self._battery_percent_list = []
