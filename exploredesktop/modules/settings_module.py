"""Settings module"""
import logging
import os
from copy import deepcopy

import yaml
from appdirs import user_config_dir
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSettings,
    Qt,
    Slot
)
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QStyledItemDelegate
)


from exploredesktop.modules.app_settings import (  # isort: skip
    ConnectionStatus,
    DataAttributes,
    ExGModes
)
from exploredesktop.modules import (  # isort: skip
    Messages,
    Settings,
    BaseModel
)
from exploredesktop.modules.utils import display_msg, wait_cursor, ELECTRODES_10_20  # isort: skip

logger = logging.getLogger("explorepy." + __name__)


class SettingsFrameView(BaseModel):
    """_summary_
    """

    def __init__(self, ui, filters) -> None:
        super().__init__()
        self.ui = ui
        self.filters = filters
        self.setup_frame()

        self.ui.table_settings.setModel(ConfigTableModel([]))
        self.setup_tableview()

        # Setup signal connections
        self.signals.dataSettingsChanged.connect(self.disable_apply)

    def setup_tableview(self) -> None:
        """Configure settings table
        """
        # Add delegates
        # self.ui.table_settings.setItemDelegateForColumn(1, CheckBoxDelegate(None))
        self.ui.table_settings.setItemDelegate(_ConfigItemDelegate())

        # Resize to fill all horizontal space
        self.ui.table_settings.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.table_settings.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ui.table_settings.verticalHeader().setVisible(False)

        # Stylesheet
        self.ui.table_settings.setAlternatingRowColors(True)
        self.ui.table_settings.horizontalHeader().setStyleSheet("""
        QHeaderView::section {
            border-bottom: 1px solid black;
            border-right: 0px;
            border-top: 0px;
        }""")

        self.ui.table_settings.setStyleSheet("""
        border: none;""")

        # Remove button to select all
        self.ui.table_settings.setCornerButtonEnabled(False)

        # Hide signal type column
        self.ui.table_settings.setColumnHidden(3, True)

    def setup_frame(self) -> None:
        """Initialize dropdowns and checkboxes
        """
        self.ui.value_sampling_rate.addItems([str(int(sr)) for sr in Settings.SAMPLING_RATES])
        # TODO uncomment later if implemented
        # self.ui.dropdown_signal_type.addItems(ExGModes.all_values())
        # self.ui.cb_multitype_signal.setChecked(False)
        # self.ui.dropdown_signal_type.setHidden(False)

        # Hide multitype signal dropdown and checkbox
        # self.ui.cb_multitype_signal.setHidden(True)
        # self.ui.dropdown_signal_type.setHidden(True)
        # self.ui.cb_1020.setHidden(True)

    def setup_ui_connections(self) -> None:
        """Connect ui widgets to corresponding slot
        """
        self.ui.value_sampling_rate.currentTextChanged.connect(self.display_sr_warning)

        self.ui.btn_reset_settings.clicked.connect(self.reset_settings)
        self.ui.btn_format_memory.clicked.connect(self.format_memory)
        self.ui.btn_apply_settings.clicked.connect(self.change_settings)
        # TODO uncomment when implemented
        # self.ui.btn_calibrate.setHidden(True)

        # TODO uncomment later when implemented
        # self.ui.cb_multitype_signal.stateChanged.connect(self.multisignal_clicked)
        # self.ui.dropdown_signal_type.currentTextChanged.connect(self.signal_type_changed)
        # self.ui.cb_1020.stateChanged.connect(self.enable_10_20)

    def setup_settings_frame(self) -> None:
        """Setup the settings frame
        """
        # Set device name
        self.ui.label_explore_name.setText(self.explorer.device_name)

        # Set active channels
        data = deepcopy(self.explorer.chan_dict_list)
        self.ui.table_settings.setModel(ConfigTableModel(data))
        self.ui.table_settings.viewport().update()

        # Set sampling rate
        s_rate = int(self.explorer.sampling_rate)
        self.ui.value_sampling_rate.setCurrentText(str(s_rate))

    ###
    # Button slots
    ###
    @Slot()
    def reset_settings(self) -> None:
        """
        Display a popup asking for confirmation.
        If yes, the settinngs are set to default.
        """
        reset = False

        response = display_msg(msg_text=Messages.RESET_SETTINGS_QUESTION, popup_type="question")

        if response == QMessageBox.StandardButton.No:
            return reset

        with wait_cursor():
            reset = self.explorer.reset_soft()
            self.explorer.disconnect()
            self.signals.connectionStatus.emit(ConnectionStatus.DISCONNECTED)
            self.signals.pageChange.emit("btn_bt")

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
    def format_memory(self) -> None:
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
    def change_settings(self) -> None:
        """
        Apply changes in device settings
        """
        with wait_cursor():
            self._remove_filters()

            changed_chan = self.change_active_channels()
            changed_sr = self.change_sampling_rate()
            changed_chan_names = self.change_channel_names()
            # Reset exg data and reapply filters
            self.signals.updateDataAttributes.emit([DataAttributes.DATA])
            if self.filters.current_filters is not None:
                self.filters.apply_filters()

        if changed_sr or changed_chan or changed_chan_names:
            self._display_new_settings()
            self.signals.restartPlot.emit()
            self.signals.displayDefaultImp.emit()

    ###
    # Change settings functions
    ###
    def change_channel_names(self) -> bool:
        """Read table and change the channel names

        Returns:
            bool: _description_
        """
        changed = False
        chan_names_table = self.ui.table_settings.model().get_list_names(full=True)

        if chan_names_table != self.explorer.full_chan_list(custom_name=True):
            changed = True
            new_dict = [
                {
                    "input": ch, "enable": active, "name": name, "type": sig_type
                } for ch, active, name, sig_type in zip(
                    [c.lower() for c in Settings.CHAN_LIST],
                    [d["enable"] for d in self.explorer.chan_dict_list],
                    chan_names_table,
                    [d["type"] for d in self.explorer.chan_dict_list])
            ]

            self.explorer.set_chan_dict_list(new_dict)
            self.explorer.settings.set_chan_names(chan_names_table)

            logger.info(f"Channel names changed: {self.ui.table_settings.model().get_list_names()}")
        return changed

    def change_active_channels(self) -> bool:
        """
        Read selected checkboxes and set the channel mask of the device

        Returns:
            bool: whether sampling rate has changed
        """

        changed = False

        active_chan = self.get_active_chan_ui()
        active_chan_int = [int(i) for i in active_chan]
        # verify at least one channel is selected
        n_active = sum(active_chan_int)
        if n_active == 0:
            display_msg(Messages.SELECT_1_CHAN)
            return

        if (active_chan_int != self.explorer.chan_mask):
            # TODO decide how we handle (de)activation of channels for 4, 8 chan
            changed = True
            # mask = "".join(active_chan)
            # changed = self.explorer.set_channels(mask)
            # self.explorer.chan_mask = self.ui.table_settings.model().get_chan_mask()
            mask = self.ui.table_settings.model().get_chan_mask()
            self.explorer.set_chan_mask(mask)

            new_dict = [
                {
                    "input": ch, "enable": active, "name": name, "type": sig_type
                } for ch, active, name, sig_type in zip(
                    [c.lower() for c in Settings.CHAN_LIST],
                    mask,
                    [d["name"] for d in self.explorer.chan_dict_list],
                    [d["type"] for d in self.explorer.chan_dict_list])
            ]

            self.explorer.set_chan_dict_list(new_dict)
            self.update_modules()
            logger.info(f"Active channels changed: {self.explorer.settings.settings_dict}")

        return changed

    def update_modules(self) -> None:
        """Update modules dependent on number of active channels"""
        self.signals.updateDataAttributes.emit([DataAttributes.OFFSETS, DataAttributes.BASELINE])
        self.signals.displayDefaultImp.emit()

    def get_active_chan_ui(self) -> list:
        """Get active channels from UI settings table checkboxes

        Returns:
            list[str]: binary list indicating whether channel is active
        """
        active_chan = [str(int(one_chan_dict["enable"])) for one_chan_dict in self.ui.table_settings.model().chan_data]
        # active_chan = list(reversed(active_chan))
        return active_chan

    def change_sampling_rate(self) -> bool:
        """Change the sampling rate

        Returns:
            bool: whether sampling rate has changed
        """

        current_sr = int(self.explorer.sampling_rate)
        new_sr = int(self.ui.value_sampling_rate.currentText())
        changed = False

        if int(current_sr) != new_sr:
            if self.filters.current_filters is not None:
                self.filters.check_filters_sr(new_sr)
            logger.info("\nOld Sampling rate: %s", self.explorer.sampling_rate)
            changed = self.explorer.set_sampling_rate(sampling_rate=new_sr)
            self.explorer.settings.set_adc_mask(list(reversed(self.explorer.chan_mask)))
            logger.info("\nNew Sampling rate: %s", self.explorer.sampling_rate)
        return changed

    def check_settings_saved(self) -> bool:
        """Check if the settings in UI are the same as in explore, i.e. all settings have been saved

        Returns:
            bool: whether settings have been saved
        """
        saved = True
        if not self.explorer.is_connected:
            return saved
        current_sr = int(self.explorer.sampling_rate)
        ui_sr = int(self.ui.value_sampling_rate.currentText())

        current_chan_names = self.explorer.active_chan_list(custom_name=True)
        ui_chan_names = self.ui.table_settings.model().get_list_names()

        # TODO uncomment when adc mask is implemented
        # current_active_chan = self.explorer.stream_processor.device_info['adc_mask']
        current_active_chan = self.explorer.chan_mask
        ui_active_chan = [int(i) for i in self.get_active_chan_ui()]

        if (
            current_sr != ui_sr
        ) or (
            current_chan_names != ui_chan_names
        ) or (current_active_chan != ui_active_chan):
            saved = False
        return saved

    def _display_new_settings(self) -> None:
        """Display popup with new sampling rate and active channels
        """
        chan_dict = self.explorer.get_chan_dict_list()
        act_chan = ", ".join([
            f'{one_chan_dict["input"]} ({one_chan_dict["name"]})'
            for one_chan_dict in chan_dict if one_chan_dict["enable"]])
        msg = (
            "Device settings have been changed:"
            f"\nSampling Rate: {self.explorer.sampling_rate}"
            f"\nActive Channels: {act_chan}"
        )
        display_msg(msg_text=msg, popup_type="info")

    def _remove_filters(self) -> None:
        """Remove filters"""
        if self.filters.current_filters is not None:
            self.signals.updateDataAttributes.emit([DataAttributes.BASELINE])
            self.explorer.stream_processor.remove_filters()

    ###
    # Vis feedback slots
    ###
    @Slot()
    def display_sr_warning(self) -> None:
        """Display warning for 1000 Hz sampling rate
        """
        if int(self.ui.value_sampling_rate.currentText()) == 1000:
            self.ui.lbl_sr_warning.show()
        else:
            self.ui.lbl_sr_warning.hide()

    @Slot()
    def disable_apply(self, index: QModelIndex) -> None:
        """Disable apply button based on the names of the channels

        Args:
            index (QModelIndex): index of the item changed
        """
        # Only relevant for Name column, return if item changed is another one
        if index.column() != 2:
            return

        # Default values
        enable = True
        tooltip = ""

        custom_names = self.ui.table_settings.model().get_list_names()
        custom_names_alnum = ["".join(e for e in name if e.isalnum()).strip() for name in custom_names]

        # Check for names containing only special characters
        if "" in custom_names_alnum:
            enable = False
            tooltip = "Channel names cannot contain only special characters"

        # Check for repeated names
        if len(custom_names) != len(set(custom_names)):
            enable = False
            tooltip = "Channel names must be unique"

        # Disable button and set tooltip
        self.ui.btn_apply_settings.setEnabled(enable)
        self.ui.btn_apply_settings.setToolTip(tooltip)

    def enable_settings(self, enable=True) -> None:
        """Disable or enable device settings widgets

        Args:
            enable (bool, optional): True will enable, False will disable. Defaults to True.
        """

        enabled = True
        s_rate_stylesheet = ""
        # TODO decide which stylesheet to use
        # stylesheet = ""
        tooltip_apply_settings = ""
        tooltip_reset_settings = ""
        tooltip_format_mem = ""

        if enable is False:
            enabled = False
            s_rate_stylesheet = "color: gray;\nborder-color: gray;"
            # stylesheet = Stylesheets.DISABLED_BTN_STYLESHEET
            tooltip_apply_settings = Messages.DISABLED_SETTINGS
            tooltip_reset_settings = Messages.DISABLED_RESET
            tooltip_format_mem = Messages.DISABLED_FORMAT_MEM

        self.ui.value_sampling_rate.setEnabled(enabled)
        self.ui.value_sampling_rate.setStyleSheet(s_rate_stylesheet)

        self.ui.btn_apply_settings.setEnabled(enabled)
        # self.ui.btn_apply_settings.setStyleSheet(stylesheet)
        self.ui.btn_apply_settings.setToolTip(tooltip_apply_settings)

        self.ui.btn_reset_settings.setEnabled(enabled)
        # self.ui.btn_reset_settings.setStyleSheet(stylesheet)
        self.ui.btn_reset_settings.setToolTip(tooltip_reset_settings)

        self.ui.btn_format_memory.setEnabled(enabled)
        # self.ui.btn_format_memory.setStyleSheet(stylesheet)
        self.ui.btn_format_memory.setToolTip(tooltip_format_mem)

        self.ui.label_warning_disabled.setHidden(enabled)

        self.ui.table_settings.model().change_column_edit('enable', enabled)
        self.ui.table_settings.model().change_column_edit('name', enabled)

    def multisignal_clicked(self) -> None:
        """Allow/Block selection of multiple signal types
        """
        multitype = self.ui.cb_multitype_signal.isChecked()

        self.ui.dropdown_signal_type.setHidden(multitype)
        self.ui.table_settings.model().change_column_edit("type", multitype)
        self.ui.table_settings.viewport().update()

    def signal_type_changed(self) -> None:
        """Change all signal type values based on dropdown value
        """
        if self.ui.dropdown_signal_type.currentText() == ExGModes.EEG.value:
            new_value = ExGModes.EEG.value
        else:
            new_value = ExGModes.ECG.value

        for ch_dict in self.ui.table_settings.model().chan_data:
            ch_dict["type"] = new_value
        self.ui.table_settings.viewport().update()

    def enable_10_20(self) -> None:
        """Enable combobox for 10/20 notation
        """
        if self.ui.cb_1020.isChecked():
            self.ui.table_settings.model().change_column_editor("name", "combobox")
        else:
            self.ui.table_settings.model().change_column_editor("name", "default")

        self.ui.table_settings.viewport().update()

    ###
    # Import/Export settings
    ###
    def export_settings(self):
        """
        Open a dialog to select folder to be saved
        """
        settings = QSettings("Mentalab", "ExploreDesktop")
        path = settings.value("last_settings_save_folder")
        if not path:
            path = os.path.expanduser("~")

        dialog = QFileDialog()
        file_path = dialog.getSaveFileName(
            None,
            "Save As",
            os.path.join(path, "untitled.yaml"),
            "YAML (*.yaml)")

        file_path = file_path[0]
        if file_path == "":
            return

        if path != os.path.dirname(file_path):
            settings.setValue("last_settings_save_folder", os.path.dirname(file_path))

        settings_to_export = self.explorer.settings.settings_dict.copy()
        del settings_to_export["adc_mask"]
        del settings_to_export["firmware_version"]
        del settings_to_export["mac_address"]

        with open(file_path, 'w+') as fp:
            yaml.safe_dump(settings_to_export, fp, default_flow_style=False)
            fp.close()

    def import_settings(self):
        """Import settings
        """
        settings_dict = self._open_settings_file()
        if settings_dict is None:
            return
        if not self._verify_settings(settings_dict):
            return

        self._apply_imported_settings(settings_dict)

    def _apply_imported_settings(self, settings_dict: dict) -> None:
        """Apply imported settings to explorepy

        Args:
            settings_dict (dict): dictionary containing new settings
        """
        if 'channel_name' not in settings_dict.keys():
            settings_dict['channel_name'] = [f"ch{i + 1}" for i in range(len(settings_dict['software_mask']))]
        new_dict_list = [
            {
                'input': f'ch{idx + 1}', 'enable': bool(val[0]),
                'name': val[1], 'type': 'EEG'}
            for idx, val in enumerate(zip(reversed(settings_dict['software_mask']), settings_dict['channel_name']))]
        self.ui.table_settings.setModel(ConfigTableModel(new_dict_list))
        self.ui.value_sampling_rate.setCurrentText(str(int(settings_dict['sampling_rate'])))
        self.change_settings()
        self.explorer.chan_dict_list = new_dict_list

    def _verify_settings(self, settings_dict: dict) -> bool:
        """Check if imported settings can be applied to connected device

        Args:
            settings_dict (dict): dictionary of imported settings

        Returns:
            bool: whether imported settings are ok
        """
        settings_ok = True
        if len(settings_dict['software_mask']) == self.explorer.device_chan:
            return settings_ok

        extra = len(settings_dict['software_mask']) - self.explorer.device_chan
        if extra > 0:
            blurb = "The file selected has too many channels for the current device"
        elif extra < 0:
            blurb = "The file selected is missing some channels"
        display_msg(blurb, title="Invalid file")
        settings_ok = False

        return settings_ok

    def _open_settings_file(self) -> dict:
        """Open settings yaml file

        Returns:
            dict: dictionary with imported settings
        """
        settings = QSettings("Mentalab", "ExploreDesktop")
        path = settings.value("last_settings_import_folder")
        if not path:
            path = os.path.expanduser("~")

        dialog = QFileDialog()
        file_path = dialog.getOpenFileName(
            None,
            "Import",
            path,
            "YAML (*.yaml)")

        file_path = file_path[0]
        if file_path == "":
            return None

        if path != os.path.dirname(file_path):
            settings.setValue("last_settings_import_folder", os.path.dirname(file_path))

        settings_dict = self._read_settings_file(file_path)
        return settings_dict

    @staticmethod
    def _read_settings_file(file_path: str) -> dict:
        """Read yaml settings file

        Args:
            file_path (str): path to the yaml file

        Returns:
            dict: dictionary with the settings included in the yaml file
        """
        stream = open(file_path, 'r')
        settings_dict = yaml.load(stream, Loader=yaml.SafeLoader)
        return settings_dict

    def import_last_session_settings(self) -> None:
        """Import settings from last session"""
        data_path = user_config_dir(appname="Mentalab", appauthor="explorepy", version='archive')
        file_path = os.path.join(data_path, self.explorer.device_name + ".yaml")
        settings_dict = self._read_settings_file(file_path)
        self._apply_imported_settings(settings_dict)


class _ConfigItemDelegate(QStyledItemDelegate):
    '''Combobox item editor
    '''

    # pylint: disable=invalid-name
    def createEditor(self, parent, option, index):
        """
        Create dropdown editor
        """
        if index.model().editorType(index.column()) == 'combobox':
            combobox = QComboBox(parent)
            combobox.addItems(index.model().comboBoxList(index.column()))
            combobox.setEditable(False)
            return combobox

        if index.model().editorType(index.column()) == 'checkbox':
            return None

        if index.model().editorType(index.column()) == 'limit_text':
            editor = QLineEdit(parent)
            max_char = 10
            editor.setMaxLength(max_char)
            return editor

        return QStyledItemDelegate.createEditor(self, parent, option, index)

    # pylint: disable=invalid-name
    def setEditorData(self, editor, index):
        """Set editor data
        """
        if index.model().columns[index.column()]['editor'] == 'combobox':
            text = index.model().data(index, Qt.DisplayRole)
            i = editor.findText(text)
            if i == -1:
                i = 0
            editor.setCurrentIndex(i)

        elif index.model().columns[index.column()]['editor'] == 'limit_text':
            text = index.model().data(index, Qt.DisplayRole)
            editor.setText(text)
        QStyledItemDelegate.setEditorData(self, editor, index)

    # pylint: disable=invalid-name
    def setModelData(self, editor, model, index):
        """Set data in the model
        """
        if model.columns[index.column()]['editor'] == 'combobox':
            model.setData(index, editor.currentText(), Qt.EditRole)
            # model.reset()
        elif model.columns[index.column()]['editor'] == 'limit_text':
            model.setData(index, editor.text(), Qt.EditRole)
        QStyledItemDelegate.setModelData(self, editor, model, index)


class ConfigTableModel(QAbstractTableModel, BaseModel):
    """Table Model for configuration Table View
    """
    def __init__(self, data):
        """_summary_

        Args:
            data (_type_): _description_
        """
        super(ConfigTableModel, self).__init__()
        self.chan_data = data

        # column description
        self.columns = [
            {'property': 'input', 'header': 'Channel', 'edit': False, 'editor': 'default'},
            {'property': 'enable', 'header': 'Enable', 'edit': True, 'editor': 'checkbox'},
            {'property': 'name', 'header': 'Name', 'edit': True, 'editor': 'limit_text'},
            {'property': 'type', 'header': 'Type', 'edit': False, 'editor': 'combobox'},
        ]

    def change_column_edit(self, col_name: str, new_val: bool) -> None:
        """Change edit field of a column to make it (un)editable

        Args:
            col_name (str): name of the property to change
            new_val (bool): whether column is editable
        """
        next(item for item in self.columns if item["property"] == col_name)["edit"] = new_val

    def change_column_editor(self, col_name: str, new_val: str) -> None:
        """Change editor flag of a column
        Args:
            col_name (str): name of the property to change
            new_val (bool): editor type
        """
        next(item for item in self.columns if item["property"] == col_name)["editor"] = new_val

    def data(self, index, role):
        """Abstract method from QAbstactTableModel to get data
        """
        value = self._getitem(index.row(), index.column())

        if (role == Qt.DisplayRole) or (role == Qt.EditRole):
            if self.columns[index.column()]['property'] == 'enable':
                value = True if value == 2 else False
            return value

        elif role == Qt.CheckStateRole and self.columns[index.column()]['property'] == 'enable':
            # print(f"data - {value=}")
            return value

        if role == Qt.BackgroundRole:
            if index.column() == 2 and (
                "".join(
                    e for e in value if e.isalnum()).strip() == "" or self.get_list_names(full=True).count(value) > 1):
                return QBrush("#fa5c62")

        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignHCenter | Qt.AlignVCenter)
        return None

    def get_list_names(self, full=False) -> list:
        """Return list of custom names
        """
        if full is True:
            return [d["name"] for d in self.chan_data]
        return [d["name"] for d in self.chan_data if d["enable"]]

    def get_chan_mask(self) -> list:
        """Return channel mask as list"""
        return [int(d["enable"]) for d in self.chan_data]

    # pylint: disable=invalid-name
    def rowCount(self, index) -> int:
        """Return number of rows
        """
        return len(self.chan_data)

    # pylint: disable=invalid-name
    def columnCount(self, index) -> int:
        """Return number of columns
        """
        if self.chan_data:
            return len(self.chan_data[0])
        return len(self.columns)

    # pylint: disable=invalid-name
    def headerData(self, col, orientation, role):
        """Abstract method from QAbstactTableModel to get the column header
        """
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.columns[col]['header']

    def comboBoxList(self, column: int) -> list:
        """Get list of items for comboboxes

        Args:
            column (int): column number

        Returns:
            list: list of combobox items
        """
        if column >= len(self.columns):
            return None
        if self.columns[column]['property'] == 'type':
            return ExGModes.all_values()
        if self.columns[column]['property'] == 'name':
            return ELECTRODES_10_20

    def editorType(self, column: int) -> str:
        """Get the columns editor type from column description

        Args:
            column (int): column number

        Returns:
            str: editor type
        """
        if column >= len(self.columns):
            return None
        return str(self.columns[column]['editor'])

    def flags(self, index):
        """Abstract method from QAbstactTableModel
        """
        fl = QAbstractTableModel.flags(self, index)
        if self.columns[index.column()]['editor'] == "checkbox":
            fl |= Qt.ItemIsUserCheckable
            # fl |= Qt.ItemIsEditable | Qt.ItemIsUserCheckable
        if not index.isValid():
            return Qt.ItemIsEnabled
        if not self.columns[index.column()]['edit']:
            return Qt.NoItemFlags
        return fl | Qt.ItemIsEditable

    def _setitem(self, row: int, column: int, value: str) -> bool:
        """Set property item based on table row and column

        Args:
            row (int): row number
            column (int): column number
            value (str): value to set

        Returns:
            bool: True if property value was set, False if not
        """
        if (row >= len(self.chan_data)) or (column >= len(self.columns)):
            return False

        # get channel properties
        property = self.chan_data[row]

        # get property name from column description
        property_name = self.columns[column]['property']

        # set channel property
        if property_name == 'enable':
            value = True if value == 2 else False
            property["enable"] = value
            # print(f"set_item - {value=}")
            return True

        if property_name == 'name':
            n = value
            if "".join(n.split()) == "":
                return False
            property["name"] = value
            return True

        if property_name == 'type':
            n = value
            property["type"] = value
            return True

        return False

    def _getitem(self, row: int, column: int) -> str:
        """Get property item based on table row and column

        Args:
            row (int): row number
            column (int): column number

        Returns:
            str: property value
        """
        if (row >= len(self.chan_data)) or (column >= len(self.columns)):
            return None

        # get channel properties
        property = self.chan_data[row]
        # get property name from column description
        property_name = self.columns[column]['property']
        # get property value
        if property_name in ['input', 'name', 'type']:
            d = str(property[property_name])
        elif property_name == 'enable':
            d = 2 if property[property_name] is True else 0
        else:
            d = None
        return d

    # pylint: disable=invalid-name
    def setData(self, index, value, role):
        """Abstract method from QAbstactTableModel to set cell data based on role
        """
        if not index.isValid():
            return False

        if role == Qt.CheckStateRole or role == Qt.EditRole:
            if not self._setitem(index.row(), index.column(), value):
                return False
            self.signals.dataSettingsChanged.emit(index)
            return True
        return False
