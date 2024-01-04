"""Main Application"""
import logging
import os
import sys
from enum import Enum
from typing import Union

import PySide6
from explorepy.log_config import (
    read_config,
    write_config
)
from explorepy.stream_processor import TOPICS
from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    QThreadPool
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeySequence
)
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QMainWindow,
    QMessageBox,
    QPushButton
)


import exploredesktop  # isort:skip
from exploredesktop.modules import (  # isort:skip
    BaseModel,
    GUISettings,
    Stylesheets,
    Ui_MainWindow
)
from exploredesktop.modules.app_settings import (  # isort:skip
    ConnectionStatus,
    DataAttributes,
    EnvVariables,
    Messages
)
from exploredesktop.modules.bt_module import BTFrameView  # isort:skip
from exploredesktop.modules.exg_module import ExGPlot  # isort:skip
from exploredesktop.modules.fft_module import FFTPlot  # isort:skip
from exploredesktop.modules.filters_module import Filters  # isort:skip
from exploredesktop.modules.footer_module import FooterFrameView  # isort:skip
from exploredesktop.modules.imp_module import ImpFrameView  # isort:skip
from exploredesktop.modules.lsl_module import IntegrationFrameView  # isort:skip
from exploredesktop.modules.menubar_module import MenuBarActions   # isort:skip
from exploredesktop.modules.mkr_module import MarkerPlot  # isort:skip
from exploredesktop.modules.orn_module import ORNPlot  # isort:skip
from exploredesktop.modules.real_time_swipe_plot import EXGPlotVispy
from exploredesktop.modules.recording_module import RecordFunctions  # isort:skip
from exploredesktop.modules.settings_module import SettingsFrameView  # isort:skip
from exploredesktop.modules.utils import (  # isort:skip
    display_msg,
    get_widget_by_obj_name
)
VERSION_APP = exploredesktop.__version__
WINDOW_SIZE = False

logger = logging.getLogger("explorepy." + __name__)

if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
    logger.debug("CWD: %s" % os.getcwd())
    dir_main = os.path.dirname(os.path.abspath(__file__))
    ICON_PATH = os.path.join(dir_main, "images", "MentalabLogo.ico")
    logger.debug("Icon path: %s" % ICON_PATH)
elif sys.platform == "win32":
    logger.debug("CWD: %s" % os.getcwd())
    ICON_PATH = os.path.join(os.getcwd(), "MentalabLogo.ico")
    logger.debug("Icon path: %s" % ICON_PATH)


class MainWindow(QMainWindow, BaseModel):
    """
    Main window class. Connect signals and slots
    """

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowIcon(QIcon(ICON_PATH))
        self.setWindowTitle('Explore Desktop')

        self.is_streaming = False

        # Style UI
        self.style_ui()

        # Set UI definitions (close, restore, etc)
        self.drop_shadow()

        # Slidable left panel
        self.ui.btn_left_menu_toggle.clicked.connect(self.slide_main_menu)

        # Stacked pages - default open connect or home if permissions are not set
        existing_permission = self.check_permissions()
        if existing_permission:
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_bt)
            self.highlight_main_menu_item("btn_bt")
        else:
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            self.highlight_main_menu_item("btn_home")
            # Set data sharing permissions
            self.set_permissions()

        # Stacked pages - navigation
        for btn_wdgt in self.ui.left_side_menu.findChildren(QPushButton):
            btn_wdgt.clicked.connect(self.handle_page_navigation)

        # HOME PAGE
        self.ui.cb_permission.stateChanged.connect(self.set_permissions)

        # IMPEDANCE PAGE
        self.imp_frame = ImpFrameView(self.ui)
        self.imp_frame.setup_ui_connections()

        # BLUETOOTH PAGE
        self.bt_frame = BTFrameView(self.ui)
        self.bt_frame.setup_ui_connections()

        # FOOTER
        self.footer_frame = FooterFrameView(self.ui)

        # FILTERS
        self.filters = Filters(self.ui)
        self.filters.setup_ui_connections()

        # SETTINGS PAGE
        self.settings_frame = SettingsFrameView(self.ui, self.filters)
        self.settings_frame.setup_ui_connections()

        # PLOTS
        self.orn_plot = ORNPlot(self.ui)
        self.orn_plot.setup_ui_connections()
        #self.exg_plot = ExGPlot(self.ui, self.filters)
        self.exg_plot_vispy = EXGPlotVispy(self.ui, self.explorer)
        self.exg_plot_vispy.setup_ui_connections()
        #self.exg_plot.setup_ui_connections()
        self.fft_plot = FFTPlot(self.ui)
        #self.mkr_plot = MarkerPlot(self.ui)
        #self.mkr_plot.setup_ui_connections()

        self.ui.tabWidget.currentChanged.connect(self.plot_tab_changed)

        # RECORDING
        self.recording = RecordFunctions(self.ui)
        self.recording.setup_ui_connections()

        # INTEGRATION PAGE
        self.integration_frame = IntegrationFrameView(self.ui)
        self.integration_frame.setup_ui_connections()

        # signal connections
        self.setup_signal_connections()

        # MENUBAR
        self._setup_menubar()

    #########################
    # UI Functions
    #########################
    def reset_vars(self) -> None:
        """Reset all variables"""
        self.is_streaming = False
        #self.exg_plot.reset_vars()
        self.orn_plot.reset_vars()
        # self.orn_plot.get_model().reset_vars()
        self.fft_plot.reset_vars()
        self.footer_frame.get_model().reset_vars()
        self.imp_frame.get_model().reset_vars()
        self.filters.reset_vars()

    def stop_processes(self) -> None:
        """Stop ongoing actions"""
        self._stop_recording()
        self._stop_impedance()
        self._stop_lsl()
        # TODO uncomment after proper test of external LSL markers and threading
        # known issue: thread not stopped on close
        # self.mkr_plot.model.stop_lsl_marker_thread()

    def _stop_lsl(self) -> None:
        """Stop lsl if active
        """
        if self.explorer.is_pushing_lsl:
            self.integration_frame.stop_lsl_push()

    def _stop_impedance(self) -> None:
        """Stop impedance measurement if active
        """
        if self.explorer.is_measuring_imp:
            self.imp_frame.disable_imp()

    def _stop_recording(self) -> None:
        """Stop recording if active
        """
        if self.explorer.is_recording:
            self.recording.stop_record()

    def on_connection_change(self, connection: Enum) -> None:
        """Actions to perfom when connection status changes

        Args:
            connection (Enum): connection status
        """
        # TODO: split, connect and disconnect
        self.footer_frame.print_connection_status(connection)

        if connection == ConnectionStatus.CONNECTED:
            btn_connect_text, btn_scan_enabled = self.on_connect()
            self.exg_plot_vispy.on_connected()

        elif connection == ConnectionStatus.DISCONNECTED:
            btn_connect_text, btn_scan_enabled = self.on_disconnect()
            self.exg_plot_vispy.on_disconnected()

        else:
            return

        self.signals.btnConnectChanged.emit(btn_connect_text)
        self.ui.btn_scan.setEnabled(btn_scan_enabled)

    def on_disconnect(self) -> Union[str, bool]:
        """Actions to perform when device is disconnected

        Returns:
            Union[str, bool]: Connect button text and whether to disable the scann button
        """
        btn_connect_text = "Connect"
        btn_scan_enabled = True
        self.signals.pageChange.emit("btn_bt")
        self._enable_menubar(False)

        self.stop_processes()
        self.reset_vars()
        return btn_connect_text, btn_scan_enabled

    def on_connect(self) -> Union[str, bool]:
        """Actions to perform when device is connected

        Returns:
            Union[str, bool]: Connect button text and whether to disable the scann button
        """
        btn_connect_text = "Disconnect"
        btn_scan_enabled = False

        if self.ui.stackedWidget.currentWidget() == self.ui.page_bt:
            self.signals.pageChange.emit("btn_settings")

        firmware = self.explorer.stream_processor.device_info["firmware_version"]
        data = {EnvVariables.FIRMWARE: firmware}
        self.signals.devInfoChanged.emit(data)

        # Uncomment below to disable change of sampling rate
        # if self.explorer.device_chan == 32:
        #     self.ui.value_sampling_rate.setEnabled(False)
        # else:
        #     self.ui.value_sampling_rate.setEnabled(True)

        # initialize impedance
        self.signals.displayDefaultImp.emit()
        # subscribe environmental data callback
        self.footer_frame.get_model().subscribe_env_callback()
        # initialize settings frame
        self.settings_frame.setup_settings_frame()
        # initialize visualization offsets
        self.signals.updateDataAttributes.emit([DataAttributes.OFFSETS, DataAttributes.DATA])
        # initialize menubar
        self._enable_menubar(True)

        self._init_plots()

        return btn_connect_text, btn_scan_enabled

    def _enable_menubar(self, enable: bool) -> None:
        """Enable or disable menubar actions dependent on Explorepy connection

        Args:
            enable (bool): whether to enable
        """
        self.ui.actionMetadata_import.setEnabled(enable)
        self.ui.actionMetadata_export.setEnabled(enable)
        self.ui.actionLast_Session_Settings.setEnabled(enable)

    def _setup_menubar(self) -> None:
        """Setup menubar actions
        """
        self.menubar_actions = MenuBarActions()
        self.ui.actionNew.triggered.connect(lambda: print("new clicked"))
        # Exit action
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionExit.setShortcut(QKeySequence("Alt+F4"))

        # Hide import actions
        self.ui.actionCSV_data.setEnabled(False)
        self.ui.actionCSV_data.setVisible(False)
        self.ui.actionBIN_data.setVisible(False)
        self.ui.actionBIN_data.setVisible(False)
        self.ui.actionEDF_data.setVisible(False)
        self.ui.actionEDF_data.setVisible(False)

        self.ui.actionNew.setVisible(False)

        # Disable actions requiring connection with explorepy
        self._enable_menubar(False)
        self.ui.actionRecorded_visualization.setEnabled(False)

        # File actions
        self.ui.actionMetadata_import.triggered.connect(self.settings_frame.import_settings)
        self.ui.actionMetadata_export.triggered.connect(self.settings_frame.export_settings)
        self.ui.actionLast_Session_Settings.triggered.connect(self.settings_frame.import_last_session_settings)
        self.ui.actionConvert.triggered.connect(self.menubar_actions.convert_bin)
        self.ui.actionEEGLAB_Dataset.triggered.connect(self.menubar_actions.export_eeglab_dataset)
        self.ui.actionData_Repair.triggered.connect(self.menubar_actions.repair_data)

        # Help actions
        self.ui.actionDocumentation.triggered.connect(self.menubar_actions.launch_wiki)

        # Tools actions
        self.ui.actionRecorded_visualization.triggered.connect(self.menubar_actions.recorded_visualization)

        # NOTE: uncomment below if implementing custom full/scroll view feature
        # View actions
        # self.ui.actionFull_View.triggered.connect(lambda: self.exg_plot.model.change_vis_mode(VisModes.FULL))
        # self.ui.actionScroll_View.triggered.connect(lambda: self.exg_plot.model.change_vis_mode(VisModes.SCROLL))

        # from PySide6.QtGui import QActionGroup
        # view_group = QActionGroup(self)
        # view_group.setExclusive(True)

        # actionFullView = view_group.addAction("Full View AG")
        # actionFullView.setCheckable(True)
        # actionScrollView = view_group.addAction("Scroll View AG")
        # actionScrollView.setCheckable(True)
        # actionScrollView.setChecked(True)

        # self.ui.menuVisualization.addActions(view_group.actions())

        # actionFullView.triggered.connect(lambda: self.exg_plot.model.change_vis_mode(VisModes.FULL))
        # actionFullView.triggered.connect(self._init_plots)
        # actionScrollView.triggered.connect(lambda: self.exg_plot.model.change_vis_mode(VisModes.SCROLL))
        # actionScrollView.triggered.connect(self._init_plots)

        #self.ui.actionReceive_LSL_Markers.triggered.connect(self.mkr_plot.model.enable_external_markers)

        # self.ui.actionReceive_LSL_Markers.setVisible(True)
        # self.ui.actionReceive_LSL_Markers.setChecked(False)

    def _init_plots(self) -> None:
        """Initialize plots"""
        self.orn_plot.init_plot()
        #self.exg_plot.init_plot()
        self.fft_plot.init_plot()

    def setup_signal_connections(self):
        """Connect custom signals to corresponding slots
        """
        # TODO move to appropiate module
        self.signals.btnImpMeasureChanged.connect(self.ui.btn_imp_meas.setText)
        self.signals.btnConnectChanged.connect(self.ui.btn_connect.setText)

        self.signals.envInfoChanged.connect(self.footer_frame.update_env_info)
        self.signals.devInfoChanged.connect(self.footer_frame.update_dev_info)

        self.signals.connectionStatus.connect(self.on_connection_change)

        self.signals.impedanceChanged.connect(self.imp_frame.get_graph().on_new_data)
        self.signals.displayDefaultImp.connect(self.imp_frame.get_graph().display_default_imp)

        self.signals.pageChange.connect(self.handle_page_navigation)

        self.signals.ornChanged.connect(self.orn_plot.swipe_plot)
        #self.signals.exgChanged.connect(self.exg_plot.swipe_plot)

        self.signals.tRangeORNChanged.connect(self.orn_plot.set_t_range)
        self.signals.tAxisORNChanged.connect(self.orn_plot.set_t_axis)

        #self.signals.tRangeEXGChanged.connect(self.exg_plot.set_t_range)
        #self.signals.tAxisEXGChanged.connect(self.exg_plot.set_t_axis)

        #self.signals.updateYAxis.connect(self.exg_plot.add_left_axis_ticks)

        #self.signals.restartPlot.connect(self.exg_plot.init_plot)
        self.signals.restartPlot.connect(self.fft_plot.init_plot)

        #self.signals.mkrPlot.connect(self.mkr_plot.plot_marker)
        #self.signals.mkrAdd.connect(self.mkr_plot.model.add_mkr)
        # self.signals.mkrReplot.connect(lambda data: self.mkr_plot.plot_marker(data, replot=True))
        #self.signals.replotMkrAdd.connect(self.mkr_plot.model.add_mkr_replot)
        #self.signals.mkrRemove.connect(self.mkr_plot.remove_old_item)

        #self.signals.updateDataAttributes.connect(self.exg_plot.model.update_attributes)

        #self.signals.btDrop.connect(self.exg_plot.display_bt_drop)

        #self.signals.rrPeakRemove.connect(self.exg_plot.remove_old_r_peak)
        # self.signals.rrPeakPlot.connect(self.exg_plot.plot_rr_point)

        # self.signals.heartRate.connect(self.ui.value_heartRate.setText)
        #self.signals.plotRR.connect(self.exg_plot.plot_rr_point)

        #self.signals.recordStart.connect(self.exg_plot.model.set_packet_offset)
        #self.signals.recordEnd.connect(self.exg_plot.model.log_n_packets)
        self.signals.recordStart.connect(lambda: self.settings_frame.enable_settings(False))
        self.signals.recordEnd.connect(self.settings_frame.enable_settings)

    def style_ui(self) -> None:
        """Initial style for UI
        """
        # Bold font for device label
        bold_font = QFont()
        bold_font.setBold(True)
        self.ui.ft_label_device_3.setFont(bold_font)
        self.ui.ft_label_device_3.setStyleSheet("font-weight: bold")
        self.ui.menuBar.setStyleSheet("font: 10pt")

        # Hide unnecessary labels
        # settings page
        self.ui.label_warning_disabled.setHidden(True)
        self.ui.lbl_sr_warning.hide()
        self.ui.btn_calibrate.setHidden(True)

        # connect page
        self.ui.lbl_wdws_warning.hide()
        # self.ui.lbl_bt_instructions.hide()

        # integration page
        # TODO: decide if we want to enable duration
        self.ui.lsl_duration_value.hide()
        self.ui.cb_lsl_duration.hide()
        self.ui.label_lsl_duration.setHidden(True)

        # # Hide os bar
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # Add app version to footer
        self.ui.ft_label_version.setText(VERSION_APP)

        # Hide footer
        self.ui.ft_label_firmware.setHidden(True)
        self.ui.ft_label_firmware_value.setHidden(True)
        self.ui.ft_label_battery.setHidden(True)
        self.ui.ft_label_battery_value.setHidden(True)
        self.ui.ft_label_temp.setHidden(True)
        self.ui.ft_label_temp_value.setHidden(True)

        # Start with foucus on line edit for device name
        self.ui.dev_name_input.setFocus()

        # hide eeglabexport from home
        self.ui.label_6.setHidden(True)
        self.ui.btn_generate_bdf.setHidden(True)
        self.ui.btn_import_edf.setHidden(True)
        self.ui.le_import_edf.setHidden(True)

        # hide view menu at start
        self.ui.menuVisualization.menuAction().setVisible(False)

    def _verify_imp(self, btn_name: str) -> bool:
        """Verify if impedance measurement is active before moving to another page

        Args:
            btn_name (str): button name to the page to move

        Returns:
            bool: whether impedance measurement is active

        """
        imp_disabled = True
        if self.explorer.is_measuring_imp and btn_name != "btn_impedance":
            imp_disabled = self.imp_frame.check_is_imp()
        return imp_disabled

    def _verify_settings_changed(self, btn_name: str) -> bool:
        """
        Check if settings have been saved. If not, ask the user if they want to still exit the page
        Args:
            btn_name (str): button name to the page to move

        Returns:
            bool: whether settings are saved or user wants to exit anyway
        """
        changes_saved = True
        if btn_name != "btn_settings":
            changes_saved = self.settings_frame.check_settings_saved()

        if not changes_saved:
            response = display_msg(msg_text=Messages.SETTINGS_NOT_SAVED, popup_type="question")

            if response == QMessageBox.StandardButton.Yes:
                changes_saved = True
        return changes_saved

    def change_page(self, btn_name: str) -> bool:
        """Change the active page when the object is clicked

        Args:
            btn_name (str): button name

        Returns:
            bool: whether page has changed
        """
        # TODO: split this function
        btn_page_map = {
            "btn_home": self.ui.page_home, "btn_bt": self.ui.page_bt,
            "btn_settings": self.ui.page_settings, "btn_plots": self.ui.page_plotsNoWidget,
            "btn_impedance": self.ui.page_impedance}

        # If not navigating to impedance, verify if imp mode is active
        imp_disabled = self._verify_imp(btn_name)
        if not imp_disabled:
            return False

        # If not navigating to settings, verify if settings have been changed
        settings_saved = self._verify_settings_changed(btn_name)
        if not settings_saved:
            return False

        # If the page requires connection to a Explore device, verify
        if btn_name in GUISettings.LEFT_BTN_REQUIRE_CONNECTION and self.explorer.is_connected is False:
            msg = "Please connect an Explore device."
            display_msg(msg_text=msg, popup_type="info")
            return False

        # Actions for specific pages
        if btn_name == "btn_settings":
            self._move_to_settings()

        elif btn_name == "btn_plots":
            # TODO Uncomment when scroll/full view is implemented
            # if self.explorer.device_chan > 9:
            #     self.ui.menuVisualization.menuAction().setVisible(True)
            # else:
            #     self.ui.menuVisualization.menuAction().setVisible(False)

            filt = True
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_plotsNoWidget)

            if self.filters.current_filters is None:
                filt = self.filters.popup_filters()

            # TODO instead of going to settings go back to previous page
            if filt is False:
                self.ui.stackedWidget.setCurrentWidget(self.ui.page_settings)
                self.highlight_main_menu_item("btn_settings")
                return False

            if not self.is_streaming and filt:
                self._subscribe_callbacks()
                self.is_streaming = True
                #self.mkr_plot.model.start_lsl_marker_thread()

        # Move to page
        self.ui.stackedWidget.setCurrentWidget(btn_page_map[btn_name])
        return True

    def plot_tab_changed(self, idx: int) -> None:
        """Activate/deactivate fft measurment when plot tab changes

        Args:
            idx (int): index of the active tab
        """
        if idx == 2:  # FFT tab active
            self.fft_plot.start_timer()
        else:
            self.fft_plot.stop_timer()

    def _subscribe_callbacks(self) -> None:
        """Subscribe signal callbacks
        """
        self.explorer.subscribe(callback=self.orn_plot.model.callback, topic=TOPICS.raw_orn)
        #self.explorer.subscribe(callback=self.exg_plot.model.callback, topic=TOPICS.filtered_ExG)
        self.explorer.subscribe(callback=self.fft_plot.model.callback, topic=TOPICS.filtered_ExG)
        #self.explorer.subscribe(callback=self.mkr_plot.model.callback, topic=TOPICS.marker)

    def _move_to_settings(self) -> None:
        """Actions to perform before moving to settings
        """
        self.settings_frame.setup_settings_frame()
        enable = not self.explorer.is_recording and not self.explorer.is_pushing_lsl
        self.settings_frame.enable_settings(enable)
        self.ui.value_sampling_rate.setEnabled(enable and not self.explorer.device_chan > 8)

    def slide_main_menu(self) -> None:
        """
        Animation to display the whole left menu
        """
        # Get current left menu width
        width, new_width = self._get_main_menu_width()

        # Animate transition
        self._animate_sliding(width, new_width)

    def _animate_sliding(self, width: int, new_width: int) -> None:
        """Animate main menu sliding

        Args:
            width (int): current width
            new_width (int): width to slide to
        """
        self.animation = QPropertyAnimation(
            self.ui.left_side_menu, b"minimumWidth")
        self.animation.setDuration(250)
        self.animation.setStartValue(width)
        self.animation.setEndValue(new_width)
        self.animation.setEasingCurve(QEasingCurve.InOutQuart)
        self.animation.start()

    def _get_main_menu_width(self) -> Union[int, int]:
        """Obtain current and new with of the main menu

        Returns:
            Union[int, int]: current width, new width
        """
        current_width = self.ui.left_side_menu.width()
        if current_width == GUISettings.LEFT_MENU_MIN:
            new_width = GUISettings.LEFT_MENU_MAX
        else:
            new_width = GUISettings.LEFT_MENU_MIN
        return current_width, new_width

    def highlight_main_menu_item(self, btn_name: str) -> None:
        """
        Change style of the button clicked

        Args:
            btn_name (str): name of the button to highlight
        """
        if btn_name != "btn_left_menu_toggle":
            # Reset style for other buttons
            self._reset_menu_item_style(btn_name)

            # Apply new style
            self._apply_menu_item_style(btn_name)

    def _apply_menu_item_style(self, btn_name: str) -> None:
        """
        Highlight main menu button clicked

        Args:
            btn_name (str): name of the button to highlight
        """
        btn = get_widget_by_obj_name(btn_name)
        new_style = btn.styleSheet() + (Stylesheets.BTN_LEFT_MENU_SELECTED_STYLESHEET)
        btn.setStyleSheet(new_style)

    def _reset_menu_item_style(self, btn_name: str) -> None:
        """Reset stylesheet of the items in the main menu that have not been clicked

        Args:
            btn_name (str): name of the button clicked
        """
        for btn_wdgt in self.ui.left_side_menu.findChildren(QPushButton):
            if btn_wdgt.objectName() != btn_name:
                default_style = btn_wdgt.styleSheet().replace(Stylesheets.BTN_LEFT_MENU_SELECTED_STYLESHEET, "")
                btn_wdgt.setStyleSheet(default_style)

    def handle_page_navigation(self, name=False):
        """
        Change style of the button clicked and move to the selected page
        """
        btn_name = self._get_button_name(name)

        # Navigate to active page
        if btn_name != "btn_left_menu_toggle":
            change = self.change_page(btn_name)
            if change is False:
                return
        # Apply stylesheet
        self.highlight_main_menu_item(btn_name)

    def _get_button_name(self, name) -> str:
        """Get button name

        Returns:
            str: button name
        """
        if isinstance(name, str):
            btn_name = name
        else:
            btn = self.sender()
            btn_name = btn.objectName()
        return btn_name

    # pylint: disable=invalid-name
    def mousePressEvent(self, event):
        """
        Get mouse current position to move the window
        Args: mouse press event
        """
        self.clickPosition = event.globalPosition().toPoint()

    def drop_shadow(self) -> None:
        """Drop shadow
        """
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(0)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 0))
        self.ui.centralwidget.setGraphicsEffect(self.shadow)

    def closeEvent(self, event: PySide6.QtGui.QCloseEvent) -> None:
        """Override close event with  actions to perform on close
        """
        QThreadPool().globalInstance().waitForDone()
        self.stop_processes()
        if self.explorer.device_name is not None:
            self.explorer.disconnect()
        return super().closeEvent(event)

    def set_permissions(self) -> None:
        """
        Set data sharing permission to explorepy config file
        """
        share = self.ui.cb_permission.isChecked()
        write_config("user settings", "share_logs", str(share))

    def check_permissions(self) -> bool:
        """Check current data sharing permission

        Returns:
            bool: whether permission exist in config file
        """
        exist = False
        config = read_config("user settings", "share_logs")
        if config != "":
            config = True if config == "True" else False
            self.ui.cb_permission.setChecked(config)
            exist = True
        return exist

    def changeEvent(self, event: PySide6.QtCore.QEvent) -> None:
        if event.type() == QEvent.WindowStateChange:
            self.resize_settings_table()
        return super().changeEvent(event)

    def resize_settings_table(self):
        if self.height() > 800:
            self.ui.table_settings.setFixedHeight(192 * 2)
            self.ui.spacer_frame.setFixedHeight(100)
        else:
            self.ui.table_settings.setFixedHeight(192)
            self.ui.spacer_frame.setFixedHeight(30)

    def resizeEvent(self, event: PySide6.QtGui.QResizeEvent) -> None:
        self.resize_settings_table()
        return super().resizeEvent(event)
