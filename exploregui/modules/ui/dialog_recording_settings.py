# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialog_recording_settings_dark.ui'
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
        Dialog.resize(400, 300)
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
"QLabe#label_4l{\n"
"	font: italic 9pt;\n"
"}\n"
"\n"
"QCheckBox{\n"
"	color: #FFF\n"
"}\n"
"\n"
"QLineEdit{\n"
"	color: #FFF;\n"
"	background-color: transparent;\n"
"	/*background-color: rgb(50, 53, 74);\n"
"	border: 1px solid rgb(84, 89, 124);*/\n"
"	border: 1px solid #FFF;\n"
"}\n"
"\n"
"QComboBox{\n"
"	color: #FFF;\n"
"	border: 1px solid #FFF;\n"
"	background-color: transparent;\n"
"}\n"
"\n"
"QRadioButton{\n"
"	color: #FFF;\n"
"	background-color: rgb(28, 30, 42);\n"
"}\n"
"\n"
"QSpinBox{\n"
"	color: #FFF;\n"
"	border: 1px solid #FFF;\n"
"	background-color: transparent;\n"
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
"	background-color: rgb("
                        "61, 64, 89);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"	background-color: rgb(101, 106, 141);\n"
"	border:  2px solid rgb(61, 64, 89);\n"
"}\n"
"")
        self.verticalLayout_2 = QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(-1, 25, -1, -1)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.lbl_step2 = QLabel(Dialog)
        self.lbl_step2.setObjectName(u"lbl_step2")

        self.horizontalLayout_3.addWidget(self.lbl_step2)

        self.rdbtn_csv = QRadioButton(Dialog)
        self.rdbtn_csv.setObjectName(u"rdbtn_csv")

        self.horizontalLayout_3.addWidget(self.rdbtn_csv)

        self.rdbtn_edf = QRadioButton(Dialog)
        self.rdbtn_edf.setObjectName(u"rdbtn_edf")

        self.horizontalLayout_3.addWidget(self.rdbtn_edf)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.lbl_step1 = QLabel(Dialog)
        self.lbl_step1.setObjectName(u"lbl_step1")

        self.verticalLayout.addWidget(self.lbl_step1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.input_filepath = QLineEdit(Dialog)
        self.input_filepath.setObjectName(u"input_filepath")

        self.horizontalLayout.addWidget(self.input_filepath)

        self.btn_browse = QPushButton(Dialog)
        self.btn_browse.setObjectName(u"btn_browse")
        self.btn_browse.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayout.addWidget(self.btn_browse)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.spinBox = QSpinBox(Dialog)
        self.spinBox.setObjectName(u"spinBox")

        self.horizontalLayout_2.addWidget(self.spinBox)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.label_4 = QLabel(Dialog)
        self.label_4.setObjectName(u"label_4")
        font = QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        self.label_4.setFont(font)

        self.verticalLayout.addWidget(self.label_4)

        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setCursor(QCursor(Qt.PointingHandCursor))
        self.buttonBox.setStyleSheet(u"")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.lbl_step2.setText(QCoreApplication.translate("Dialog", u"1. Select the file format :     ", None))
        self.rdbtn_csv.setText(QCoreApplication.translate("Dialog", u"csv", None))
        self.rdbtn_edf.setText(QCoreApplication.translate("Dialog", u"edf", None))
        self.lbl_step1.setText(QCoreApplication.translate("Dialog", u"2. Select the folder and name to store the file:", None))
        self.btn_browse.setText(QCoreApplication.translate("Dialog", u"Browse", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"3. Select recording time (s):", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"If recording time is 0, the default (3600 sec) will be used", None))
    # retranslateUi

