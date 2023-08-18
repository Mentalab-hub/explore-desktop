from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
import exploredesktop.main_window as mw
import explorepy

from exploredesktop.modules import utils


def test_reset_settings(qtbot, monkeypatch):
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    window.hide()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    # By default, should redirect to Settings Frame view, but just to be sure:
    settings_btn = window.ui.btn_settings
    qtbot.addWidget(settings_btn)
    qtbot.mouseClick(settings_btn, Qt.LeftButton)

    reset_btn = window.settings_frame.ui.btn_reset_settings
    qtbot.addWidget(reset_btn)

    # Handle dialog about the Explore device disconnection
    def handle_dialog():
        # this is a confirmation dialogue box
        messagebox = utils.get_widget_by_obj_name('disp_box')
        yes_button = messagebox.button(QMessageBox.Yes)
        qtbot.mouseClick(yes_button, Qt.LeftButton)

    QTimer.singleShot(1000, handle_dialog)

    qtbot.mouseClick(reset_btn, Qt.LeftButton)

    assert not window.explorer.is_connected
    assert window.ui.stackedWidget.currentWidget().objectName() == "page_bt"
    mw.MainWindow().close()


def test_apply_settings(qtbot, monkeypatch):
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    window.hide()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    # window.show()
    table = window.settings_frame.ui.table_settings
    model = table.model()
    # index for the second channel tick box
    cell_2 = model.index(1, 1)
    rect = table.visualRect(cell_2).center()
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, pos=rect)

    # Change sampling rate
    sampling_rate = window.settings_frame.ui.value_sampling_rate
    qtbot.addWidget(sampling_rate)
    qtbot.mouseClick(sampling_rate, Qt.LeftButton)

    qtbot.keyClick(sampling_rate, Qt.Key_Down, delay=3)
    qtbot.keyClick(sampling_rate, Qt.Key_Enter, delay=3)

    # apply settings
    btn_apply = window.settings_frame.ui.btn_apply_settings
    qtbot.addWidget(btn_apply)

    def handle_dialog():
        # this is basically an information dialogue box
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Ok)

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(btn_apply, Qt.LeftButton)
    mw.MainWindow().close()


def test_format_memory_no(qtbot, monkeypatch):
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    format_mem = window.settings_frame.ui.btn_format_memory
    qtbot.addWidget(format_mem)

    def handle_dialog():
        # this is a confirmation dialogue box
        messagebox = utils.get_widget_by_obj_name('disp_box')
        yes_button = messagebox.button(QMessageBox.Yes)
        qtbot.mouseClick(yes_button, Qt.LeftButton)

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(format_mem, Qt.LeftButton, delay=1)
    mw.MainWindow().close()
