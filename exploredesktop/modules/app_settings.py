"""Module with app settings, stylesheets and messages displayed in the app
"""
from enum import (
    Enum,
    auto
)


class BaseEnum(Enum):
    """Base class for Enums
    """
    @classmethod
    def all_values(cls):
        """Returns a list of all the values in the Enumeration
        """
        return [m.value for m in cls]


class FilterTypes(BaseEnum):
    """Enum for filter types"""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"


class EnvVariables(BaseEnum):
    """Enum for environmental keys
    """
    BATTERY = "battery"
    TEMPERATURE = "temperature"
    LIGHT = "light"

    DEVICE_NAME = "device_name"
    FIRMWARE = "firmware"


class ConnectionStatus(BaseEnum):
    """Enum for connection status
    """
    CONNECTED = "Connected to dev_name"
    RECONNECTING = "Reconnecting ..."
    DISCONNECTED = "Not connected"
    UNSTABLE = (
        '<html><head/><body><p><span style=" font-weight:600; '
        'color:#c80000;">Unstable BT connection</span></p></body></html>'
    )


class ImpModes(BaseEnum):
    """Enum for impedance measuring modes
    """
    WET = "Wet Electrodes"
    DRY = "Dry Electrodes"


class DataAttributes(BaseEnum):
    """Enum for data attributes
    """
    OFFSETS = auto()
    BASELINE = auto()
    DATA = auto()
    # FILTERS = auto()
    INIT = auto()
    ORNDATA = auto()
    POINTER = auto()
    ORNPOINTER = auto()


class VisModes(BaseEnum):
    FULL = "full"
    SCROLL = "scroll"


class ExGModes(BaseEnum):
    """Enum for supported ExG modes"""
    EEG = "EEG"
    ECG = "ECG"


class FileTypes(BaseEnum):
    """Enum for supported file types"""
    CSV = "csv"
    BDF = "edf"


class PlotItems(BaseEnum):
    """Enum for plot item types and corresponding dict key"""
    VLINES = ["lines", "code"]
    POINTS = ["points", "r_peak"]


class QSettingsKeys(BaseEnum):
    BIN_FOLDER = "last_bin_folder"
    BIN_EXPORT = "last_bin_export"
    RECORD_FOLDER = "last_record_folder"
    REPAIR_FOLDER = "last_repair_folder"


class Stylesheets():
    """
    Class containig stylesheets for GUI
    """
    #########################
    # Buttons stylesheets
    #########################
    DISABLED_BTN_STYLESHEET = """
    background-color: rgb(89,90,111);
    color: rgb(155,155,155);
    """
    BTN_LEFT_MENU_SELECTED_STYLESHEET = """
    background-color: rgb(113, 120, 159);
    """

    #########################
    # Impedance stylesheets
    #########################
    GRAY_IMPEDANCE_STYLESHEET = "#A9A9A9"

    BLACK_IMPEDANCE_STYLESHEET = "#111111"

    RED_IMPEDANCE_STYLESHEET = "#CB0000"

    ORANGE_IMPEDANCE_STYLESHEET = "#CF8F25"

    YELLOW_IMPEDANCE_STYLESHEET = "#C5C527"

    GREEN_IMPEDANCE_STYLESHEET = "#2B851A"

    #########################
    # Battery stylesheet
    #########################
    BATTERY_STYLESHEETS = {
        "low": "color: #FD0000;",
        "medium": "color: #CDB900;",
        "high": "color: #00FD19;",
        "na": "color: #000000"
    }

    #########################
    # Plot colors
    #########################
    EXG_LINE_COLOR = "#42C4F7"

    # TODO category8 palette bookeh
    FFT_LINE_COLORS = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
        '#2ca02c', '#98df8a', '#d62728', '#ff9896',
        '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
        '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
        '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        #
        '#84ee15', '#6e9f23', '#096013', '#d6790b',
        '#f34207', '#6108e8', "#68affc", "#b3dfc1",
        "#154e56", "#49edc9", "#659c7e", "#c0e15c"]

    MARKER_LINE_COLOR = '#7AB904'  # ALPHA = 1

    MARKER_LINE_COLOR_ALPHA = '#7AB90480'  # ALPHA = .5

    ORN_LINE_COLORS = ["#00FF00", "#42C4F7", "#FF0000"]

    PLOT_BACKGROUND = "#120b28"

    POS_LINE_COLOR = "#FF0000"  # red

    #########################
    # Not in use for now
    #########################
    # Start button stylesheet
    START_BUTTON_STYLESHEET = """
    QPushButton{
        border: 1px solid 25541C;
        border-radius: 5px;
        background-color: #2B851A;
        color: #FFF;}
    QPushButton:hover{
        border: 2px solid #25541C;
        border-radius: 5px;
        background-color: #2B851A;
        color: #FFF;
    }
    QPushButton:pressed{
        border: 2px solid #25541C;
        border-radius: 5px;
        background-color: #4D9033;
        color: #FFF;
    }
    """

    STOP_BUTTON_STYLESHEET = """
    QPushButton{
        border: 1px solid #7D0101;
        border-radius: 5px;
        background-color: #EA0000;
        color: #FFF;
    }
    QPushButton:hover{
        border: 2px solid #FFF;
        border-radius: 5px;
        background-color: #EA0000;
        color: #FFF;
    }
    QPushButton:pressed{
        border: 2px solid #7D0101;
        border-radius: 5px;
        background-color: rgb(215, 74, 61);
        color: #FFF;
    }"""

    POPUP_STYLESHEET = """
    QMessageBox {
    background-color: rgb(28, 30, 42);
    }

    QMessageBox QLabel {
        color: rgb(200, 200, 200);
    }
        """


class GUISettings():
    """Class containing GUI settings"""
    CUSTOM_TITLE_BAR = True

    LEFT_MENU_MIN = 60
    LEFT_MENU_MAX = 200

    LEFT_BTN_REQUIRE_CONNECTION = ["btn_settings", "btn_plots", "btn_impedance", "btn_integration"]

    ORN_LEGEND = ['Acc [mg/LSB]', 'Gyro [mdps/LSB]', 'Mag [mgauss/LSB]']

    RESERVED_CHARS = r"[|\\?*<\":>[\]+/']"


class Settings():
    """Class containing explore settings
    """
    DOWNSAMPLING = True

    ORN_SRATE = 20  # Hz
    EXG_VIS_SRATE = 125
    WIN_LENGTH = 10  # Seconds
    # MODE_LIST = ['EEG', 'ECG']
    MAX_CHANNELS = 32
    CHAN_LIST = [f'ch{i}' for i in range(1, MAX_CHANNELS + 1)]

    DEFAULT_SCALE = 10 ** 3  # Volt
    BATTERY_N_MOVING_AVERAGE = 60
    V_TH = [10, 5 * 10 ** 3]  # Noise threshold for ECG (microVolt)

    SCALE_MENU = {
        "1 uV": 0, "5 uV": -0.66667, "10 uV": -1, "50 uV": -1.66667, "100 uV": -2,
        "200 uV": -2.33333, "500 uV": -2.66667, "1 mV": -3,
        "5 mV": -3.66667, "10 mV": -4, "100 mV": -5}

    TIME_RANGE_MENU = {"10 s": 10., "5 s": 5., "20 s": 20.}
    SAMPLING_RATES = [250, 500, 1000]

    # Max value for each color
    COLOR_RULES_DRY = {
        "green": 20,
        "yellow": 35,
        "orange": 50,
        "red": 70,
        "open": 500
    }
    COLOR_RULES_WET = {
        "green": 10,
        "yellow": 20,
        "orange": 30,
        "red": 50,
        "open": 250
    }

    BASELINE_MA_LENGTH = 1.5 * EXG_VIS_SRATE

    MIN_LC_WEIGHT = 0.0035


class Messages():
    """Class containing GUI messages
    """
    #########################
    # bt functions
    #########################
    NO_BT_CONNECTION = "No Bluetooth connection available.\nPlease make sure the bluetooth is on."
    NO_EXPLORE_DEVICES = "No explore devices found. Please make sure your device is turned on."
    WARNING_PAIRED_DEV_WINDOWS = "Note: Listed paired devices might not be advertising"
    INVALID_EXPLORE_NAME = "Please select a device or provide a valid name (Explore_XXXX or XXXX) before connecting."
    WINDOWS_PAIR_INSTRUCTIONS = "Follow Windows' instructions to pair your device."
    CONNECTION_REFUSED = "Please unpair Explore device manually or use a Bluetooth dongle"

    #########################
    # config functions
    #########################
    FORMAT_MEM_QUESTION = "Are you sure you want to format the memory?"
    CALIBRATE_ORN_QUESTION = (
        "Do you want to continue with the orientation sensors calibration?\n"
        "This will overwrite the calibration data if it already exists\n\n"
        "If yes, you would need to move and rotate the device for 100 seconds\n")
    RESET_SETTINGS_QUESTION = (
        "Are you sure you want to reset your settings?\n"
        "The Explore device will disconnect after the soft reset.")
    SELECT_1_CHAN = "At least one channel must be active."
    DISABLED_SETTINGS = "Changing the settings during recording and LSL streaming is not possible"
    DISABLED_RESET = "Resetting the settings during recording and LSL streaming is not possible"
    DISABLED_FORMAT_MEM = "Formatting the memory during recording and LSL streaming is not possible"
    SETTINGS_NOT_SAVED = "Settings were not applied and will not be saved. Do you still want to exit?"

    #########################
    # dialogs
    #########################
    OFFSET_EXPLANATION = "Remove the DC offset of the signal based on the previous signal values"
    SPECIAL_CHAR_WARNING = (
        '<html><head/><body><p><span style=" color:#d90000;">A file/folder name can\'t contain any of the'
        r' following characters:</span></p><p align="center"><span style=" color:#d90000;">| \ ? * &lt; '
        '&quot; : &gt; + [ ] / \'</span></p></body></html>'
    )
    FILE_EXISTS = (
        '<html><head/><body><p><span style=" color:#d90000;">There is already a file with the same'
        ' name in this location.</span></p><p align="center"><span style=" color:#d90000;">Please select another folder'
        ' or file name</span></p></body></html>'
    )

    #########################
    # imp functions
    #########################
    DISABLE_IMP_QUESTION = "Impedance mode will be disabled. Do you want to continue?"
    SET_SR_TO_250_QUESTION = (
        "Impedance mode only works in 250 Hz sampling rate!"
        "\nThe current sampling rate is s_rate. "
        "Click on Yes to change the sampling rate.")
    DISABLE_FAIL = "Couldn't disable impedance measurement mode.\nPlease restart your device manually."

    #########################
    # vis functions
    #########################
    INVALID_MARKER = 'Marker code value is not valid. Please select a value in the range 8 - 65535'
    BT_DROP = (
        "The bluetooth connection is unstable. This may affect the ExG visualization."
        "\nPlease read the troubleshooting section of the user manual for more."
    )
    #########################
    # main window functions
    #########################
    IMP_NOISE = (
        "Impedance measurement will introduce noise to the signal"
        " and affect the visualization, recording, and LSL stream."
        "\nAre you sure you want to continue?")
    IMP_INFO = "The displayed values are an approximation. Please refer to the manual for more information."
    CONNECTION_REQUIRED = "Please connect an Explore device."
