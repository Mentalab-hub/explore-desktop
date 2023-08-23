from PySide6.QtWidgets import QApplication, QMessageBox
from pytestqt.qtbot import QtBot
from PySide6.QtCore import Qt, QTimer
from exploredesktop.modules.app_settings import ImpModes
from test_window import connect_device
from exploredesktop.modules.app_settings import Messages


def navigate_to_impedance_view(qtbot, window):
    imp_button = window.ui.btn_impedance
    # change to impedance view
    qtbot.addWidget(imp_button)
    qtbot.mouseClick(imp_button, Qt.LeftButton)


def test_impedance_modes(qtbot):
    window = connect_device(qtbot)
    # window.show()
    imp = window.imp_frame
    navigate_to_impedance_view(qtbot, window)
    # press dropdown and select Wet Electrodes
    drop_down = imp.ui.imp_mode
    qtbot.addWidget(drop_down)
    qtbot.keyClicks(drop_down, ImpModes.WET.value)
    # Press measure btn
    meas_btn = imp.ui.btn_imp_meas
    qtbot.mouseClick(meas_btn, Qt.LeftButton)
    qtbot.wait(1000)
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


def test_info_pop_up(qtbot):
    # Helper method where Qtbot connects the device and returns QMainWindow instance
    window = connect_device(qtbot)
    # Only for visualisation
    window.show()
    # Helper method where Qtbot navigates to the Impedance View Frame
    navigate_to_impedance_view(qtbot, window)
    imp = window.imp_frame
    # Get the reference to the question mark button in the Impedance Frame
    info = imp.ui.imp_meas_info
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
