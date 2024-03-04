import explorepy
from PySide6.QtCore import Qt
import exploredesktop.main_window as mw


def test_connect(qtbot):
    """
    connects to the device with short name(XXXX)
    Parameters
    ----------
    qtbot

    Returns
    -------

    """
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    print("connect button text is {}".format(bt.ui.btn_connect.text()))
    assert bt.ui.btn_connect.text() == 'Disconnect'
    window.close()


def test_connect_device_full_name(qtbot):
    """
    connects to the device with full name(Explore_XXXX)
    Parameters
    ----------
    qtbot

    Returns
    -------

    """
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    print("connect button text is {}".format(bt.ui.btn_connect.text()))
    assert bt.ui.btn_connect.text() == 'Disconnect'
    window.close()


def test_disconnect(qtbot):
    """
    Connects and disconnects to the device
    Parameters
    ----------
    qtbot

    Returns
    -------

    """
    explorepy.set_bt_interface('mock')
    window = mw.MainWindow()
    bt = window.bt_frame
    input_field = bt.ui.dev_name_input
    qtbot.addWidget(input_field)
    input_field.clearEditText()
    qtbot.keyClicks(input_field, "XXXX")
    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    with qtbot.waitSignal(window.signals.connectionStatus, timeout=2000):
        qtbot.keyClick(input_field, Qt.Key_Enter)

    print("connect button text is {}".format(bt.ui.btn_connect.text()))
    assert bt.ui.btn_connect.text() == 'Connect'
