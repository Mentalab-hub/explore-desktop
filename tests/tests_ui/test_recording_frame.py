import pytest
from pytestqt.qtbot import QtBot
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from test_window import connect_device


def navigate_to_recording_view(qtbot, window):
    record_btn = window.ui.btn_plots
    qtbot.addWidget(record_btn)
    qtbot.mouseClick(record_btn, Qt.LeftButton, delay=1)


def test_cancel_dialog(qtbot):
    window = connect_device(qtbot)
    window.show()
    # Navigate to the recording Frame
    record_btn = window.ui.btn_plots
    qtbot.addWidget(record_btn)

    def handle_dialog():
        filters_dialog = QApplication.activeWindow()
        filters_dialog.reject()

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(record_btn, Qt.LeftButton, delay=1)
    assert not window.explorer.is_recording


def test_recording(qtbot):
    window = connect_device(qtbot)
    window.show()
    # Navigate to the recording Frame
    plot_view_btn = window.ui.btn_plots
    qtbot.addWidget(plot_view_btn)

    def handle_dialog():
        # Get an instance of the currently open window
        filters_dialog = QApplication.activeWindow()
        # either accept here or find the button bound to accept
        filters_dialog.accept()

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(plot_view_btn, Qt.LeftButton, delay=1)
    # 'Start recording' button
    record_functions = window.recording
    record_btn = record_functions.ui.btn_record
    qtbot.addWidget(record_btn)

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(record_btn, Qt.LeftButton, delay=1)
    qtbot.wait(10000)  # Recording duration

    def handle_dialog_2():
        # Get an instance of the currently open window
        messagebox = QApplication.activeWindow()
        ok_button = messagebox.button(QMessageBox.Ok)
        qtbot.mouseClick(ok_button, Qt.LeftButton)

    QTimer.singleShot(500, handle_dialog_2)
    qtbot.mouseClick(record_btn, Qt.LeftButton)
    qtbot.wait(2000)


def test_visualisation_graph(qtbot):
    window = connect_device(qtbot)
    window.show()
    # Navigate to the recording Frame
    plot_view_btn = window.ui.btn_plots
    qtbot.addWidget(plot_view_btn)

    def handle_dialog():
        # Get an instance of the currently open window
        filters_dialog = QApplication.activeWindow()
        # either accept here or find the button bound to accept
        filters_dialog.accept()

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(plot_view_btn, Qt.LeftButton, delay=1)

    # Change graph view
    tab_widget = window.ui.tabWidget
    tab = tab_widget.tabBar()
    exg_pos = tab.tabRect(0).center()
    orn_pos = tab.tabRect(1).center()
    fft_pos = tab.tabRect(2).center()

    qtbot.mouseClick(tab, Qt.LeftButton, pos=orn_pos, delay=1)
    assert tab_widget.currentIndex() == 1
    qtbot.mouseClick(tab, Qt.LeftButton, pos=fft_pos, delay=1)
    assert tab_widget.currentIndex() == 2
    qtbot.mouseClick(tab, Qt.LeftButton, pos=exg_pos, delay=1)
    assert tab_widget.currentIndex() == 0


@pytest.mark.xfail(reason='This test fails often because of PyQt Graph')
def test_change_graph_scaling(qtbot):
    window = connect_device(qtbot)
    window.show()
    # Navigate to the recording Frame
    plot_view_btn = window.ui.btn_plots
    qtbot.addWidget(plot_view_btn)

    def handle_dialog():
        # Get an instance of the currently open window
        filters_dialog = QApplication.activeWindow()
        # either accept here or find the button bound to accept
        filters_dialog.accept()

    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(plot_view_btn, Qt.LeftButton, delay=1)
    qtbot.wait(1000)
    # Select dropdown for Y-Scale
    y_scale = window.ui.value_yAxis
    qtbot.addWidget(y_scale)
    qtbot.mouseClick(y_scale, Qt.LeftButton)
    qtbot.wait(1000)
    qtbot.keyClick(y_scale, Qt.Key_Up, delay=3)
    qtbot.wait(1000)
    qtbot.keyClick(y_scale, Qt.Key_Enter, delay=3)
    qtbot.wait(1000)
    # Select dropdown for time window
    time_scale = window.ui.value_timeScale
    qtbot.addWidget(time_scale)
    qtbot.mouseClick(time_scale, Qt.LeftButton)
    qtbot.wait(1000)
    qtbot.keyClick(time_scale, Qt.Key_Down, delay=3)
    qtbot.wait(1000)
    qtbot.keyClick(time_scale, Qt.Key_Enter, delay=3)

    assert y_scale.currentText() == "500 uV"
    assert time_scale.currentText() == "5 s"


def test_filters(qtbot):
    window = connect_device(qtbot)
    window.show()
    # Navigate to the recording Frame
    plot_view_btn = window.ui.btn_plots
    qtbot.addWidget(plot_view_btn)

    def handle_dialog():
        # Get an instance of the currently open window
        filters_dialog = QApplication.activeWindow()
        # either accept here or find the button bound to accept
        # Fill in an input file

        filters_dialog.accept()
    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(plot_view_btn, Qt.LeftButton, delay=1)

    # Filters button in the plot view
    filters_btn = window.ui.btn_plot_filters
    QTimer.singleShot(100, handle_dialog)
    qtbot.mouseClick(filters_btn, Qt.LeftButton, delay=1)
