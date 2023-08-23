from pytestqt.qtbot import QtBot
# from PySide6.QtTest import QTest
import exploredesktop.main_window as mw
from PySide6.QtCore import Qt
from time import sleep


# Move this to test_BlueToothFrame?
def connect_device(qtbot):
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    qtbot.keyClicks(input_field, "855E")
    qtbot.keyClick(input_field, Qt.Key_Enter)
    sleep(5)
    qtbot.wait(5000)
    return window


"""
# This test only works if working with window.show()
def test_close_window(qtbot):
    window = mw.MainWindow()
    window.show()
    qtbot.addWidget(window)
    qtbot.wait(1000)
    assert window.isVisible()
    QTest.qWaitForWindowExposed(window)
    window.close()
    qtbot.wait(1000)
    assert not window.isVisible()
"""
