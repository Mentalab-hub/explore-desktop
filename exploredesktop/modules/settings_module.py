"""Settings module"""
import logging

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QMessageBox
)


from exploredesktop.modules import (  # isort: skip
    Messages,
    Settings,
    BaseModel
)
from exploredesktop.modules.app_settings import ConnectionStatus  # isort: skip
from exploredesktop.modules.tools import display_msg, wait_cursor  # isort: skip


logger = logging.getLogger("explorepy." + __name__)


class SettingsFrameView(BaseModel):
    """_summary_
    """
    def __init__(self, ui) -> None:
        super().__init__()
        self.ui = ui

        self.setup_dropdown()

    def setup_dropdown(self):
        """Initialize dropdown
        """
        self.ui.value_sampling_rate.addItems([str(int(sr)) for sr in Settings.SAMPLING_RATES])

    def setup_ui_connections(self):
        """Connect ui widgets to corresponding slot
        """
        for ch_wdgt in self.ui.frame_cb_channels.findChildren(QCheckBox):
            ch_wdgt.stateChanged.connect(self.one_chan_selected)

        self.ui.value_sampling_rate.currentTextChanged.connect(self.display_sr_warning)

        self.ui.btn_reset_settings.clicked.connect(self.reset_settings)
        self.ui.btn_format_memory.clicked.connect(self.format_memory)
        self.ui.btn_apply_settings.clicked.connect(self.change_settings)
        # TODO uncomment when implemented
        # self.ui.btn_calibrate.setHidden(True)

    def setup_settings_frame(self):
        """Setup the settings frame
        """
        # Set device name
        self.ui.label_explore_name.setText(self.explorer.device_name)

        # Set active channels
        chan_dict = self.explorer.get_chan_dict()
        chan_list = Settings.CHAN_LIST[:self.explorer.get_device_chan()]

        for wdgt in self.ui.frame_cb_channels.findChildren(QCheckBox):
            wdgt.setChecked(chan_dict[wdgt.objectName().replace("cb_", "")])
            if wdgt.objectName().replace("cb_", "") not in chan_list:
                wdgt.hide()
            if wdgt.isHidden() and wdgt.objectName().replace("cb_", "") in chan_list:
                wdgt.show()

        # Set sampling rate
        s_rate = int(self.explorer.sampling_rate)
        self.ui.value_sampling_rate.setCurrentText(str(s_rate))

    ###
    # Button slots
    ###
    @Slot()
    def reset_settings(self):
        """
        Display a popup asking for confirmation.
        If yes, the settinngs are set to default.
        """
        reset = False

        response = display_msg(msg_text=Messages.RESET_SETTINGS_QUESTION, popup_type="question")

        if response == QMessageBox.StandardButton.Yes:
            with wait_cursor():
                reset = self.explorer.reset_soft()

        if reset:
            self.explorer.disconnect()
            self.signals.connectionStatus.emit(ConnectionStatus.DISCONNECTED)

        else:
            msg = (
                "There was an error while resetting the settings."
                "\nPlease make sure the bluetooth connection is stable and try again."
            )
            display_msg(msg)

    @Slot()
    def format_memory(self):
        """
        Display a popup asking for confirmation.
        If yes, memory is formatted.
        """

        response = display_msg(msg_text=Messages.FORMAT_MEM_QUESTION, popup_type="question")

        if response != QMessageBox.StandardButton.Yes:
            return

        with wait_cursor():
            result = self.explorer.format_memory()

        if result:
            display_msg(msg_text="Memory formatted", popup_type="info")
        else:
            msg = (
                "There was an error while formatting the memory."
                "\nPlease make sure the bluetooth connection is stable and try again."
            )
            display_msg(msg)

    @Slot()
    def change_settings(self):
        """
        Apply changes in device settings
        """

        # stream_processor = self.explorer.stream_processor

        with wait_cursor():
            # TODO
            # if self.plotting_filters is not None:
            #     self.vis_functions._baseline_corrector["baseline"] = None
            #     self.explorer.stream_processor.remove_filters()

            changed_chan = self.change_active_channels()

            # TODO
            # changed_sr = self.change_sampling_rate()
            # self.reset_exg_plot_data()

            # if self.plotting_filters is not None:
            #     self.apply_filters()

        # if changed_sr or changed_chan:
        if changed_chan:
            chan_dict = self.explorer.get_chan_dict()
            act_chan = ", ".join([item[0] for item in chan_dict.items() if item[1]])
            msg = (
                "Device settings have been changed:"
                f"\nSampling Rate: {self.explorer.sampling_rate}"
                f"\nActive Channels: {act_chan}"
            )
            display_msg(msg_text=msg, popup_type="info")

        # TODO init plots
        # self.vis_functions.init_plots()

    ###
    # Vis feedback slots
    ###

    def change_active_channels(self):
        """
        Read selected checkboxes and set the channel mask of the device

        Returns:
            bool: whether sampling rate has changed
        """

        active_chan = []
        # changed = False

        for wdgt in self.ui.frame_cb_channels.findChildren(QCheckBox):
            status = str(1) if wdgt.isChecked() else str(0)
            active_chan.append(status)

        active_chan = list(reversed(active_chan))
        active_chan_int = [int(i) for i in active_chan]

        # verify at least one channel is selected
        n_active = sum(active_chan_int)
        if n_active == 0:
            display_msg(Messages.SELECT_1_CHAN)
            return

        if active_chan_int != self.explorer.stream_processor.device_info['adc_mask']:
            mask = "".join(active_chan)
            changed = self.explorer.set_channels(mask)

            n_chan = self.explorer.stream_processor.device_info['adc_mask']
            n_chan = list(reversed(n_chan))

            self.explorer.set_chan_dict()

            # TODO when signal data is implemented -  reset offsets and baseline corrector
            # self.vis_functions.offsets = np.arange(1, n_chan.count(1) + 1)[:, np.newaxis].astype(float)
            # self.vis_functions._baseline_corrector["baseline"] = None
            self.signals.displayDefaultImp.emit()

        return changed

    ###
    # Vis feedback slots
    ###
    @Slot()
    def one_chan_selected(self):
        """
        Make sure at least one checkbox is selected.
        If only one checkbox is left it will be disabled so status cannot change. A tooltip will be added.
        """
        cbs = {ch_wdgt: ch_wdgt.isChecked() for ch_wdgt in self.ui.frame_cb_channels.findChildren(QCheckBox)}
        if sum(cbs.values()) == 1:
            unchecked_cb = list(cbs.keys())[list(cbs.values()).index(True)]
            unchecked_cb.setEnabled(False)
            unchecked_cb.setToolTip(Messages.SELECT_1_CHAN)

        else:
            for ch_wdgt in self.ui.frame_cb_channels.findChildren(QCheckBox):
                ch_wdgt.setEnabled(True)
                ch_wdgt.setToolTip("")

    @Slot()
    def display_sr_warning(self):
        """Display warning for 1000 Hz sampling rate
        """
        if int(self.ui.value_sampling_rate.currentText()) == 1000:
            self.ui.lbl_sr_warning.show()
        else:
            self.ui.lbl_sr_warning.hide()