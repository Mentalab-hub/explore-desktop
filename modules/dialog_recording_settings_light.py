# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialog_recording_settings.ui'
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
        self.layoutWidget = QWidget(Dialog)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(10, 10, 381, 271))
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lbl_title_recording_settings = QLabel(self.layoutWidget)
        self.lbl_title_recording_settings.setObjectName(u"lbl_title_recording_settings")
        self.lbl_title_recording_settings.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lbl_title_recording_settings)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.lbl_step2 = QLabel(self.layoutWidget)
        self.lbl_step2.setObjectName(u"lbl_step2")

        self.horizontalLayout_3.addWidget(self.lbl_step2)

        self.rdbtn_csv = QRadioButton(self.layoutWidget)
        self.rdbtn_csv.setObjectName(u"rdbtn_csv")

        self.horizontalLayout_3.addWidget(self.rdbtn_csv)

        self.rdbtn_edf = QRadioButton(self.layoutWidget)
        self.rdbtn_edf.setObjectName(u"rdbtn_edf")

        self.horizontalLayout_3.addWidget(self.rdbtn_edf)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.lbl_step1 = QLabel(self.layoutWidget)
        self.lbl_step1.setObjectName(u"lbl_step1")

        self.verticalLayout.addWidget(self.lbl_step1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.input_filepath = QLineEdit(self.layoutWidget)
        self.input_filepath.setObjectName(u"input_filepath")

        self.horizontalLayout.addWidget(self.input_filepath)

        self.btn_browse = QPushButton(self.layoutWidget)
        self.btn_browse.setObjectName(u"btn_browse")

        self.horizontalLayout.addWidget(self.btn_browse)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout_2.addWidget(self.label)

        self.spinBox = QSpinBox(self.layoutWidget)
        self.spinBox.setObjectName(u"spinBox")

        self.horizontalLayout_2.addWidget(self.spinBox)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.label_4 = QLabel(self.layoutWidget)
        self.label_4.setObjectName(u"label_4")
        font = QFont()
        font.setPointSize(10)
        font.setItalic(True)
        self.label_4.setFont(font)

        self.verticalLayout.addWidget(self.label_4)

        self.buttonBox = QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.lbl_title_recording_settings.setText(QCoreApplication.translate("Dialog", u"<html><head/><body><p><span style=\" font-weight:600;\">Recording Settingns</span></p></body></html>", None))
        self.lbl_step2.setText(QCoreApplication.translate("Dialog", u"1. Select the file format :     ", None))
        self.rdbtn_csv.setText(QCoreApplication.translate("Dialog", u"csv", None))
        self.rdbtn_edf.setText(QCoreApplication.translate("Dialog", u"edf", None))
        self.lbl_step1.setText(QCoreApplication.translate("Dialog", u"2. Select the folder and name to store the file:", None))
        self.btn_browse.setText(QCoreApplication.translate("Dialog", u"Broswe", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"3. Select recording time (s):", None))
        self.label_4.setText(QCoreApplication.translate("Dialog", u"If recording time is 0, the default (3600 sec) will be used", None))
    # retranslateUi

