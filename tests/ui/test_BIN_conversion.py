import sys

from pytestqt.qtbot import QtBot
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import os
import exploredesktop.main_window as mw


def test_convert_file(qtbot):
    data_folder = 'data'
    file_name = "DATA000_8_channel.BIN"
    base_path = os.path.dirname(os.path.realpath(__file__))
    input_path = os.path.join(base_path, data_folder, file_name)
    print(input_path)
    # File path to where the result will be written:
    output_path = os.path.join(base_path, data_folder)
    print(output_path)

    qtbot.wait(1000)

    # Fills in the Dialog with the input and output path
    def handle_dialog():
        # Get an instance of the currently open window and handle the dialog
        dialog = QApplication.activeWindow()
        input_field = dialog.ui.input_filepath
        output_field = dialog.ui.input_dest_folder
        qtbot.addWidget(input_field)
        qtbot.keyClicks(input_field, input_path)
        qtbot.addWidget(output_field)
        qtbot.keyClicks(output_field, output_path)
        dialog.accept()

    # Handles the confirmation pop up
    def handle_pop_up():
        dialog = QApplication.activeWindow()
        dialog.accept()

    QTimer.singleShot(100, handle_dialog)
    #  Triggers the pop-up handler
    QTimer.singleShot(500, handle_pop_up)

    # Easy way:
    # -> ConvertBinDialog().exec()
    mw.MainWindow().ui.actionConvert.trigger()

    # Checks for CSV files:
    exg_file = os.path.join(base_path, data_folder, "DATA000_8_channel_ExG.csv")
    marker_file = os.path.join(base_path, data_folder, "DATA000_8_channel_Marker.csv")
    meta_file = os.path.join(base_path, data_folder, "DATA000_8_channel_Meta.csv")
    orn_file = os.path.join(base_path, data_folder, "DATA000_8_channel_ORN.csv")

    assert os.path.exists(exg_file)
    assert os.path.exists(marker_file)
    assert os.path.exists(meta_file)
    assert os.path.exists(orn_file)

    try:
        # Removes the created files
        os.remove(exg_file)
        os.remove(marker_file)
        os.remove(meta_file)
        os.remove(orn_file)
    except FileExistsError:
        raise FileExistsError('Error in file conversion')
    except PermissionError:
        # test shouldn't fail if a file can't be removed
        print("Could not remove previously created file.", file=sys.stderr)

    mw.MainWindow().close()


def test_invalid_path(qtbot):
    input_path = 'This is a wrong path'
    output_path = 'This is a wrong path also'

    try:
        def handle_dialog():
            # Get an instance of the currently open window and handle the dialog (with wrong paths)
            dialog = QApplication.activeWindow()
            input_field = dialog.ui.input_filepath
            output_field = dialog.ui.input_dest_folder
            qtbot.addWidget(input_field)
            qtbot.keyClicks(input_field, input_path)
            qtbot.addWidget(output_field)
            qtbot.keyClicks(output_field, output_path)
            dialog.accept()

        QTimer.singleShot(100, handle_dialog)

        # Handles the thrown exception so the test does not crash
    except Exception as e:
        if isinstance(e, AssertionError):
            pass

        def handle_pop_up():
            # Handles the error pop up
            dialog = QApplication.activeWindow()
            dialog.accept()

        # Triggers the alert pop-up handler
        QTimer.singleShot(500, handle_pop_up)

        window.ui.actionConvert.trigger()
