import pytest
from time import sleep
from pytestqt.qtbot import QtBot
from PySide6.QtCore import Qt
import exploredesktop.main_window as mw


@pytest.mark.xfail(reason='Might fail due other available devices.')
def test_connect_scanned_device(qtbot):
    # Test the connection via device scan
    window = mw.MainWindow()
    window.show()
    bt = window.bt_frame
    # Scan devices
    scan_btn = bt.ui.btn_scan
    qtbot.addWidget(scan_btn)
    qtbot.mouseClick(scan_btn, Qt.LeftButton)
    sleep(5)
    qtbot.wait(5000)
    # Get scanned devices list and click on the first:
    device_list = bt.ui.list_devices
    item_rect = device_list.visualItemRect(device_list.item(1))  # TODO: change to 0
    qtbot.mouseClick(device_list.viewport(), Qt.LeftButton, pos=item_rect.center())
    # Connect to chosen device
    c_button = window.bt_frame.ui.btn_connect
    qtbot.addWidget(c_button)
    qtbot.mouseClick(c_button, Qt.LeftButton)
    sleep(5)
    qtbot.wait(5000)
    assert bt.explorer.is_connected


def test_connect_w_input_device_short_name(qtbot):
    # Connects a device with only the identification number.
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    qtbot.keyClicks(input_field, "855E")
    qtbot.keyClick(input_field, Qt.Key_Enter)


def test_connect_device_long_name(qtbot):
    # Connects a device with the complete name of the device.
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    qtbot.keyClicks(input_field, "Explore_855E")
    qtbot.keyClick(input_field, Qt.Key_Enter)
    sleep(5)
    qtbot.wait(5000)


def test_connect_turns_connecting(qtbot):
    # Tests whether while the device is connecting the button is disabled and button text is 'Connecting'
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    qtbot.keyClicks(input_field, "855E")
    qtbot.keyClick(input_field, Qt.Key_Enter)
    qtbot.wait(1)
    connect_button = bt.ui.btn_connect
    assert connect_button.text() == 'Connecting'
    assert not connect_button.isEnabled()


def test_disconnect(qtbot):
    window = mw.MainWindow()
    window.show()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    qtbot.keyClicks(input_field, "Explore_855E")
    qtbot.keyClick(input_field, Qt.Key_Enter)
    sleep(5)
    qtbot.wait(5000)
    connect_button = bt.ui.btn_connect
    qtbot.addWidget(connect_button)
    qtbot.mouseClick(connect_button, Qt.LeftButton)
    qtbot.wait(5000)
    assert not bt.explorer.is_connected

