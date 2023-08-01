import sys
import unittest
from  PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
import exploredesktop.main_window as mw
import explorepy


def check_button_text(button, expected_text, timeout_ms=5000):
    start_time = QTest.qTickCount()
    while button.text() != expected_text:
        QTest.qWait(100)  # Wait for 100 ms before checking again
        if (QTest.qTickCount() - start_time) > timeout_ms:
            raise TimeoutError(f"Timeout: Button text did not change to '{expected_text}'")



class TestBluetoothFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the QApplication instance once for the entire test class
        cls.app = QApplication(sys.argv)

    @classmethod
    def tearDownClass(cls):
        # Destroy the QApplication instance after all test cases in the class have run
        cls.app.quit()

    def setUp(self):
        explorepy.set_bt_interface('mock')
        self.window = mw.MainWindow()
        self.window.show()
        self.window.setHidden(True)
        self.bt = self.window.bt_frame
        self.input_field = self.bt.ui.dev_name_input
        self.input_field.clearEditText()




    def tearDown(self):
        self.window.close()
        self.app = None

    def test_connect(self):
        QTest.keyClicks(self.input_field, "XXXX")
        QTest.keyClick(self.input_field, Qt.Key_Enter)

        QTest.qWaitForWindowActive(self.bt.ui.btn_connect , 1000)
        print("connect button text is {}".format(self.bt.ui.btn_connect.text()))
        assert self.bt.ui.btn_connect.text() == 'Disconnect'

    def test_disconnect(self):
        QTest.keyClicks(self.input_field, "XXXX")
        QTest.keyClick(self.input_field, Qt.Key_Enter)

        QTest.qWaitForWindowActive(self.bt.ui.btn_connect , 500)
        print("connect button text is {}".format(self.bt.ui.btn_connect.text()))

        # try to disconnect the device
        QTest.keyClick(self.input_field, Qt.Key_Enter)
        QTest.qWaitForWindowActive(self.bt.ui.btn_connect, 500)
        assert self.bt.ui.btn_connect.text() == 'Connect'