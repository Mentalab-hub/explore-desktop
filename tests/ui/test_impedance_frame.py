import explorepy
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt, QTimer
from exploredesktop.modules.app_settings import ImpModes
import exploredesktop.main_window as mw
from exploredesktop.modules.app_settings import Messages
from exploredesktop.modules.utils import get_widget_by_obj_name


def navigate_to_impedance_view(qtbot, window):
    imp_button = window.ui.btn_impedance
    # change to impedance view
    qtbot.addWidget(imp_button)
    qtbot.mouseClick(imp_button, Qt.LeftButton)


def handle_dialog(qtbot):
    # Get an instance of the currently open window and answer it
    messagebox = QApplication.activeWindow()
    assert messagebox.text() == Messages.IMP_INFO
    ok_button = messagebox.button(QMessageBox.Ok)
    qtbot.mouseClick(ok_button, Qt.LeftButton, delay=1)


def message_box_closed_callback():
    print('#############################################')


def test_impedance_modes(qtbot):
    # connect to device
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    # window.show()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    # switch to imp view
    imp_button = window.ui.btn_impedance
    # change to impedance view
    qtbot.addWidget(imp_button)
    qtbot.mouseClick(imp_button, Qt.LeftButton)
    qtbot.wait(2000)
    # qtbot.wait(2000)
    ###############################
    # window.show()
    imp = window.imp_frame
    # navigate_to_impedance_view(qtbot, window)
    # press dropdown and select Wet Electrodes
    drop_down = imp.ui.imp_mode
    qtbot.addWidget(drop_down)
    qtbot.keyClicks(drop_down, ImpModes.WET.value)
    # Press measure btn
    meas_btn = imp.ui.btn_imp_meas

    with qtbot.waitSignal(window.signals.btnImpMeasureChanged, timeout=2000):
        qtbot.mouseClick(meas_btn, Qt.LeftButton)
    print('----------------')
    # Check the color
    # get Impframe view call get_stylesheet check color?
    imp_model = imp.get_graph().model
    assert imp_model.mode == ImpModes.WET
    # disable Impedance measurement
    qtbot.mouseClick(meas_btn, Qt.LeftButton)
    # Change to Dry Electrodes
    qtbot.keyClicks(drop_down, ImpModes.DRY.value)
    assert imp_model.mode == ImpModes.DRY
    # stop measuring Impedance
    qtbot.mouseClick(meas_btn, Qt.LeftButton)
    window.close()


def test_info_pop_up(qtbot):


    # connect to device
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    window.show()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    # switch to imp view
    imp_button = window.ui.btn_impedance
    # change to impedance view
    qtbot.addWidget(imp_button)
    qtbot.mouseClick(imp_button, Qt.LeftButton)

    # App should be now in imp view
    # Get the reference to the question mark button in the Impedance Frame
    info = window.imp_frame.ui.imp_meas_info
    # Give the reference to qtbot so it know it
    qtbot.addWidget(info)

    def handle_dialog():
        # Get an instance of the currently open window and answer it
        messagebox = QApplication.activeWindow()
        assert messagebox.text() == Messages.IMP_INFO
        ok_button = messagebox.button(QMessageBox.Ok)
        qtbot.mouseClick(ok_button, Qt.LeftButton)

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(info, Qt.LeftButton, delay=1)
    window.close()
