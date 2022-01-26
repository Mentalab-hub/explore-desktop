class Settings():
    CUSTOM_TITLE_BAR = True

    DOWNSAMPLING = True

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

    LEFT_MENU_MIN = 60
    LEFT_MENU_MAX = 200

    BTN_LEFT_MENU_SELECTED_STYLESHEET = """
    background-color: rgb(113, 120, 159);
    """

    GRAY_IMPEDANCE_STYLESHEET = """
border: 2px solid rgb(145, 145, 145);
border-radius: 30px;
background-color: rgb(169, 169, 169);"""

    BLACK_IMPEDANCE_STYLESHEET = """
border: 2px solid #000000;
border-radius: 30px;
background-color: #111111;"""

    RED_IMPEDANCE_STYLESHEET = """
border: 2px solid #830000;
border-radius: 30px;
background-color: #CB0000;"""

    ORANGE_IMPEDANCE_STYLESHEET = """
border: 2px solid #976D28;
border-radius: 30px;
background-color: #CF8F25;"""

    YELLOW_IMPEDANCE_STYLESHEET = """
border: 2px solid #A0A036;
border-radius: 30px;
background-color: #C5C527;"""

    GREEN_IMPEDANCE_STYLESHEET = """
border: 2px solid #25541C;
border-radius: 30px;
background-color: #2B851A;"""

    BATTERY_STYLESHEETS = {
        "low": "color: #FD0000;",
        "medium": "color: #FDE400;",
        "high": "color: #00FD19;",
        "na": "color: #FFFFFF"
    }

    POPUP_STYLESHEET = """
QMessageBox {
background-color: rgb(28, 30, 42);
}

QMessageBox QLabel {
    color: rgb(200, 200, 200);
}
    """

    EXG_LINE_COLOR = "#42C4F7"

    FFT_LINE_COLORS = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
        '#2ca02c', '#98df8a', '#d62728', '#ff9896']

    MARKER_LINE_COLOR = '#7AB904'  # ALPHA = 1
    # MARKER_LINE_COLOR_ALPHA = '#7AB904CC'  # ALPHA = .8
    MARKER_LINE_COLOR_ALPHA = '#7AB90480'  # ALPHA = .5

    ORN_LINE_COLORS = ["#00FF00", "#42C4F7", "#FF0000"]

    ORN_SRATE = 20  # Hz
    EXG_VIS_SRATE = 125  # todo: check somewhere in code that this is above 250
    WIN_LENGTH = 10  # Seconds
    MODE_LIST = ['EEG', 'ECG']
    # CHAN_LIST = [
    #     'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8',
    #     'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16']
    CHAN_LIST = [f'ch{i}' for i in range(1, 9)]
    DEFAULT_SCALE = {"value": 10 ** 3, "string": "1 mV"}  # Volt
    BATTERY_N_MOVING_AVERAGE = 60
    V_TH = [10, 5 * 10 ** 3]  # Noise threshold for ECG (microVolt)
    ORN_LIST = [
        'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ',
        'magX', 'magY', 'magZ']

    SCALE_MENU = {
        "1 uV": 0, "5 uV": -0.66667, "10 uV": -1, "100 uV": -2,
        "200 uV": -2.33333, "500 uV": -2.66667, "1 mV": -3,
        "5 mV": -3.66667, "10 mV": -4, "100 mV": -5}

    TIME_RANGE_MENU = {"10 s": 10., "5 s": 5., "20 s": 20.}
    SAMPLING_RATES = [250, 500, 1000]
    N_CHAN_LIST = ["4", "8"]

    PLOT_BACKGROUND = "#120b28"

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
        "open": 100
    }

    BASELINE_MA_LENGTH = 1.5 * EXG_VIS_SRATE
