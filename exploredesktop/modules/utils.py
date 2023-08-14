"""
Tools/utils for GUI

Functions:
    get_widget_by_object_name
    display_message
    wait_cursor
"""
import logging
import os
from contextlib import contextmanager

import numpy as np
from PySide6.QtCore import (
    QSettings,
    Qt
)
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox
)


from exploredesktop.modules.app_settings import (  # isort:skip
    FilterTypes,
    Settings
)
logger = logging.getLogger("explorepy." + __name__)


def get_widget_by_obj_name(name: str):
    """Get widget object by its name

    Args:
        name (str): widget name

    Returns:
        Union(None, PySide6.QtWidget): widget object corresponding to the input name.
                                        If no matching widget is found, returns None
    """
    widgets = QApplication.instance().allWidgets()
    # iterate over all widget until match is found
    for wdgt in widgets:
        if str(wdgt.objectName()) == name:
            return wdgt
    logger.warning("Could not find %s", name)
    return None


def display_msg(msg_text: str, title: str = None, popup_type: str = "error"):
    """Display pop up with given message.

    Args:
        msg_text (str): message to display
        title (str, optional): title of the pop-up. Defaults to None.
        type (str, optional): pop-up type. Options are "error", "info", "question" Defaults to "error".

    Returns:
        PySide6.QtWidgets.QMessageBox.StandardButton: button clicked
    """
    msg = QMessageBox()
    msg.setObjectName('disp_box')
    msg.setText(msg_text)
    msg.raise_()

    if popup_type == "error":
        wdw_title = "Error" if title is None else title
        msg.setIcon(QMessageBox.Critical)
    elif popup_type == "info":
        wdw_title = "Information" if title is None else title
        msg.setIcon(QMessageBox.Information)
    elif popup_type == "question":
        wdw_title = "Confirmation" if title is None else title
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setIcon(QMessageBox.Question)

    msg.setWindowTitle(wdw_title)
    response = msg.exec()
    return response


@contextmanager
def wait_cursor() -> None:
    """Display wait cursor"""
    try:
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        yield
    finally:
        QApplication.restoreOverrideCursor()


def identify_filter(filter_values: tuple):
    """Identify whether filter is low, high or band pass

    Args:
        filter_values (tuple): values of the filter

    Returns:
        Enum: Filer Type
    """
    if filter_values[0] is not None and filter_values[1] is None:
        filter_type = FilterTypes.HIGHPASS
    elif filter_values[0] is None and filter_values[1] is not None:
        filter_type = FilterTypes.LOWPASS
    elif filter_values[0] is not None and filter_values[1] is not None:
        filter_type = FilterTypes.BANDPASS
    else:
        logger.warning(f"Filter with values {filter_values} not identified")
        filter_type = None
    return filter_type


def verify_filters(filter_values: tuple, sampling_rate: int) -> dict:
    """Verify filters comply with Nyquist theorem

    Args:
        filter_values (tuple): value of the filters
        sampling_rate (int): sampling rate

    Returns:
        dict: key is filter type and value is bool, whether filter is good
    """
    lc_valid = True
    hc_valid = True
    bp_valid = True

    lc_freq = float(filter_values[0]) if filter_values[0] != "" else None
    hc_freq = float(filter_values[1]) if filter_values[1] != "" else None

    nyq_freq = sampling_rate / 2.
    min_lc_freq = Settings.MIN_LC_WEIGHT * nyq_freq

    filter_type = identify_filter((lc_freq, hc_freq))
    if (filter_type == FilterTypes.HIGHPASS) and (lc_freq <= min_lc_freq):
        lc_valid = False

    elif (filter_type == FilterTypes.LOWPASS) and (hc_freq >= nyq_freq):
        hc_valid = False

    elif (filter_type == FilterTypes.BANDPASS):
        if lc_freq >= hc_freq:
            bp_valid = False
        elif hc_freq >= nyq_freq:
            hc_valid = False
        elif lc_freq <= min_lc_freq:
            lc_valid = False

    return {'lc_freq': lc_valid, 'hc_freq': hc_valid, 'bp_valid': bp_valid}


def get_filter_limits(s_rate: int):
    """Get filter limits based on sampling rate

    Args:
        s_rate (int): sampling rate

    Returns:
        _type_: minimum and maximum accepted frequency
    """

    nyq_freq = s_rate / 2.
    max_hc_freq = round(nyq_freq - 1, 1)
    min_lc_freq = round(0.0035 * nyq_freq, 1)

    return min_lc_freq, max_hc_freq


def _remove_old_plot_item(item_dict: dict, t_vector: np.array, item_type: str, plot_widget=None) -> list:
    """
    Remove line or point element from plot widget

    Args:
        item_dict (dict): dictionary with items to remove
        t_vector (np.array): time vector used as a condition to remove
        item_type (str): specifies item to remove (line or points).
        plot_widget (pyqtgraph PlotWidget): plot widget containing item to remove

    Return:
        list: list with objects to remove
    """
    assert item_type in ['line', 'points'], 'item type parameter must be line or points'
    assert 't' in item_dict.keys(), 'the items dictionary must have the keys \'t\''

    if not len(t_vector):
        return []

    to_remove = []
    for idx_t in range(len(item_dict['t'])):
        if item_dict['t'][idx_t] < t_vector[-1]:
            if plot_widget:
                plot_widget.removeItem(item_dict[item_type][idx_t])
            to_remove.append([item_dict[key][idx_t] for key in item_dict.keys()])
            # [item_dict['t'][idx_t], item_dict['r_peak'][idx_t], item_dict['points'][idx_t]])
    return to_remove


def get_path_settings(settings: QSettings, key: str) -> str:
    """Returns last used directory.
    If running for the first time, Returns user directory

    Args:
        settings (QSettings): QSettings
        key (str): key to get from settings
    """
    path = settings.value(key)
    if not path:
        path = os.path.expanduser("~")
    return path


ELECTRODES_10_20 = [
    'A1', 'A2',
    'C3', 'C4', 'Cz',
    'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz',
    'O1', 'O2',
    'P3', 'P4', 'Pz',
    'T3', 'T4', 'T5', 'T6']
