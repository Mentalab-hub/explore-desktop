import logging
import os
from datetime import datetime
from typing import (
    Tuple,
    Union
)

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog
)


from exploredesktop.modules.base_model import BaseModel  # isort: skip
from exploredesktop.modules.dialogs import RecordingDialog  # isort: skip
from PySide6.QtCore import (  # isort: skip
    QSettings,
    QTimer,
    Slot
)

logger = logging.getLogger("explorepy." + __name__)


class RecordFunctions(BaseModel):
    """
    Functions for recording functionality
    """

    def __init__(self, ui) -> None:
        super().__init__()
        self.ui = ui
        self.timer = QTimer()
        self.t_start_record = None

    def setup_ui_connections(self) -> None:
        """Setup connections between widgets and slots"""
        self.ui.btn_record.clicked.connect(self.on_record_clicked)

    @Slot()
    def on_record_clicked(self) -> None:
        """
        Start or stop recording when button is pressed
        """
        logger.debug("Pressed record button -> %s", not self.explorer.is_recording)
        if self.explorer.is_recording is False:
            self.start_record()
        else:
            self.stop_record()

    def start_record(self) -> None:
        """
        Start signal recording
        """
        default_file_name, default_dir, data = self.get_dialog_data()

        if data is False:
            return

        file_name = self._get_file_name(default_file_name, data)
        file_path = data["file_path"] if data["file_path"] != "" else default_dir
        record_duration = data["duration"] if data["duration"] != 0 else None
        file_type = data["file_type"]

        self.explorer.record_data(
            file_name=os.path.join(file_path, file_name),
            file_type=file_type,
            duration=record_duration,
            exg_ch_names=self.explorer.active_chan_list(custom_name=True)
        )

        self.start_timer_recorder(duration=record_duration)
        self.signals.recordStart.emit()
        self.t_start_record = datetime.now()
        self._update_button()

        self.explorer.record_filename = os.path.join(file_path, file_name)
        self.ui.actionRecorded_visualization.setEnabled(True)

    def get_dialog_data(self) -> Tuple[str, str, Union[bool, dict]]:
        """Get data from recording popup dialog

        Returns:
            Tuple[str, str, Union[bool, dict]]:
                default file name, default directory, popup data (False if dialog is closed or canceled)
        """
        dialog = RecordingDialog()
        default_file_name = self._set_filename_placeholder(dialog)
        default_dir = self._set_dir_placeholder(dialog)
        data = dialog.exec()
        return default_file_name, default_dir, data

    def _update_button(self, start=True) -> None:
        """Update record button

        Args:
            start (bool, optional): Whether recording is starting. Defaults to True.
        """
        if start:
            self.ui.btn_record.setIcon(QIcon(u":icons/icons/stop-button.png"))
            # self.ui.btn_record.setText("Stop")
        else:
            self.ui.btn_record.setIcon(QIcon(u":icons/icons/record-button.png"))
            # self.ui.btn_record.setText("Record")
            self.ui.label_recording_time.setText("00:00:00")
        QApplication.processEvents()

    def _get_file_name(self, default_file_name: str, data: dict) -> str:
        """Get file name.

        Args:
            default_file_name (str): default file name
            data (dict): dictionary containing recording dialog data

        Returns:
            str: file name
        """
        file_name = data["file_name"] if data["file_name"] != "" else default_file_name
        if os.path.isfile(file_name + "_ExG.csv"):
            file_name += datetime.now().strftime("_%d%b%Y_%H%M")
        return file_name

    def _set_dir_placeholder(self, dialog: QDialog) -> str:
        """Define default saving directory and set it up as a placeholder.

        Args:
            dialog (QDialog): Recording dialog with the line edit that needs the placeholder

        Returns:
            str: default directory path
        """
        settings = QSettings("Mentalab", "ExploreDesktop")
        default_dir = settings.value("last_record_folder")
        if not default_dir:
            default_dir = str(os.path.expanduser("~"))
        dialog.ui.input_filepath.setPlaceholderText(default_dir)
        return default_dir

    def _set_filename_placeholder(self, dialog: QDialog) -> str:
        """Define default saving file name and set it up as a placeholder.

        Args:
            dialog (QDialog): Recording dialog with the line edit that needs the placeholder

        Returns:
            str: default file name
        """
        default_file_name = self.explorer.device_name
        default_file_name += datetime.now().strftime("_%d%b%Y_%H%M%S")
        dialog.ui.input_file_name.setPlaceholderText(default_file_name)
        return default_file_name

    def stop_record(self) -> None:
        """
        Stop recording
        """
        if self.explorer.is_recording:
            self.explorer.stop_recording()

        if self.t_start_record is not None:
            total_time = datetime.now() - self.t_start_record
            self.signals.recordEnd.emit(total_time.total_seconds())
            self.t_start_record = None

        self.ui.actionRecorded_visualization.setEnabled(False)
        self.explorer.record_filename = ""
        try:
            self.timer.stop()
        except AttributeError:
            pass
        self._update_button(start=False)

    def start_timer_recorder(self, duration: int) -> None:
        """Start timer to display recording time

        Args:
            duration (int): recording duration
        """
        self.start_time = datetime.now()
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(lambda: self.display_rec_time(duration))
        self.timer.start()

    def display_rec_time(self, duration: int) -> None:
        """
        Display recording time in label

        Args:
            duration (int): recording duration
        """
        time = datetime.now() - self.start_time
        total_sec = int(time.total_seconds())
        strtime = str(time).split(".")[0]
        if duration is None or total_sec <= duration:
            self.ui.label_recording_time.setText(strtime)
        else:
            self.stop_record()
