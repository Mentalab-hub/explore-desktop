# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialog_plot_settings_dark.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(396, 260)
        Dialog.setStyleSheet(u"QWidget{\n"
"	font: 11pt;\n"
"}\n"
"\n"
"QDialog{\n"
"	background-color: rgb(28, 30, 42);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QCheckBox{\n"
"	color:#FFF;\n"
"	border: none;\n"
"	background-color: rgb(28, 30, 42);\n"
"}\n"
"\n"
"QLineEdit{\n"
"	color: #FFF;\n"
"	background-color: rgb(28, 30, 42);\n"
"	/*background-color: rgb(50, 53, 74);\n"
"	border: 1px solid rgb(84, 89, 124);*/\n"
"	border: 1px solid #FFF;\n"
"}\n"
"\n"
"QComboBox{\n"
"	color: #FFF;\n"
"	border: 1px solid #FFF;\n"
"	background-color: rgb(28, 30, 42);\n"
"}\n"
"\n"
"QComboBox QAbstractItemView{\n"
"	color: #FFF;\n"
"	background-color: rgb(28, 30, 42);\n"
"}\n"
"\n"
"QFrame#frame{\n"
"	border: none;\n"
"}\n"
"\n"
"QPushButton{\n"
"	color: #FFF;\n"
"	background-color: rgb(84, 89, 124);\n"
"	border: 2px solid rgb(84, 89, 124);\n"
"	padding: 5px;\n"
"	border-radius: 5px;\n"
"	width: 75px;\n"
"	height: 15px;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"\n"
"QPushButton:pres"
                        "sed{\n"
"	background-color: rgb(101, 106, 141);\n"
"	border:  2px solid rgb(61, 64, 89);\n"
"}\n"
"/*\n"
"QPushButton{\n"
"	color: #FFF;\n"
"	background-color: transparent;\n"
"	border: 2px solid #FFF;\n"
"	padding: 5px;\n"
"	border-radius: 5px;\n"
"	width: 65px;\n"
"	heigth: 20px;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"	background-color: rgb(101, 106, 141);\n"
"	border:  2px solid rgb(61, 64, 89);\n"
"}*/\n"
"")
        self.verticalLayout_3 = QVBoxLayout(Dialog)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 25, -1, -1)
        self.cb_offset = QCheckBox(Dialog)
        self.cb_offset.setObjectName(u"cb_offset")
        self.cb_offset.setCursor(QCursor(Qt.ArrowCursor))

        self.verticalLayout_3.addWidget(self.cb_offset)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(Dialog)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.value_notch = QComboBox(Dialog)
        self.value_notch.setObjectName(u"value_notch")

        self.horizontalLayout_2.addWidget(self.value_notch)


        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_3 = QLabel(Dialog)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_3)

        self.value_lowpass = QLineEdit(Dialog)
        self.value_lowpass.setObjectName(u"value_lowpass")
        self.value_lowpass.setMinimumSize(QSize(0, 20))
        self.value_lowpass.setStyleSheet(u"")

        self.verticalLayout.addWidget(self.value_lowpass)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_5 = QLabel(Dialog)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_5)

        self.value_highpass = QLineEdit(Dialog)
        self.value_highpass.setObjectName(u"value_highpass")
        self.value_highpass.setMinimumSize(QSize(0, 20))

        self.verticalLayout_2.addWidget(self.value_highpass)


        self.horizontalLayout.addLayout(self.verticalLayout_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.lbl_warning = QLabel(Dialog)
        self.lbl_warning.setObjectName(u"lbl_warning")
        font = QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setItalic(False)
        self.lbl_warning.setFont(font)
        self.lbl_warning.setStyleSheet(u"color: #d90000;")
        self.lbl_warning.setWordWrap(True)

        self.verticalLayout_3.addWidget(self.lbl_warning)

        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setCursor(QCursor(Qt.PointingHandCursor))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout_3.addWidget(self.buttonBox)


        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
#if QT_CONFIG(whatsthis)
        self.cb_offset.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.cb_offset.setText(QCoreApplication.translate("Dialog", u"Baseline Correction", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"Notch Filter", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"Low Cutoff Frequency (Hz)", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"High Cutoff Frequency (Hz)", None))
        self.lbl_warning.setText("")
    # retranslateUi

