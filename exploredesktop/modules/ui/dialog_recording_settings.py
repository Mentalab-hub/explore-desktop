# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialog_recording_settings_light.ui'
##
## Created by: Qt User Interface Compiler version 6.2.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTime,
    QUrl
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout
)


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(460, 285)
        self.verticalLayout = QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.layout_file_format = QHBoxLayout()
        self.layout_file_format.setObjectName(u"layout_file_format")
        self.lbl_step1_2 = QLabel(Dialog)
        self.lbl_step1_2.setObjectName(u"lbl_step1_2")

        self.layout_file_format.addWidget(self.lbl_step1_2)

        self.rdbtn_csv = QRadioButton(Dialog)
        self.rdbtn_csv.setObjectName(u"rdbtn_csv")

        self.layout_file_format.addWidget(self.rdbtn_csv)

        self.rdbtn_edf = QRadioButton(Dialog)
        self.rdbtn_edf.setObjectName(u"rdbtn_edf")

        self.layout_file_format.addWidget(self.rdbtn_edf)


        self.verticalLayout.addLayout(self.layout_file_format)

        self.lbl_step2 = QLabel(Dialog)
        self.lbl_step2.setObjectName(u"lbl_step2")

        self.verticalLayout.addWidget(self.lbl_step2)

        self.layout_folder_browse = QHBoxLayout()
        self.layout_folder_browse.setObjectName(u"layout_folder_browse")
        self.lbl_folder = QLabel(Dialog)
        self.lbl_folder.setObjectName(u"lbl_folder")
        self.lbl_folder.setMinimumSize(QSize(54, 0))

        self.layout_folder_browse.addWidget(self.lbl_folder)

        self.input_filepath = QLineEdit(Dialog)
        self.input_filepath.setObjectName(u"input_filepath")

        self.layout_folder_browse.addWidget(self.input_filepath)

        self.btn_browse = QPushButton(Dialog)
        self.btn_browse.setObjectName(u"btn_browse")

        self.layout_folder_browse.addWidget(self.btn_browse)


        self.verticalLayout.addLayout(self.layout_folder_browse)

        self.layout_file_name = QHBoxLayout()
        self.layout_file_name.setObjectName(u"layout_file_name")
        self.lbl_file_name = QLabel(Dialog)
        self.lbl_file_name.setObjectName(u"lbl_file_name")
        self.lbl_file_name.setMinimumSize(QSize(54, 0))

        self.layout_file_name.addWidget(self.lbl_file_name)

        self.input_file_name = QLineEdit(Dialog)
        self.input_file_name.setObjectName(u"input_file_name")

        self.layout_file_name.addWidget(self.input_file_name)


        self.verticalLayout.addLayout(self.layout_file_name)

        self.warning_label = QLabel(Dialog)
        self.warning_label.setObjectName(u"warning_label")
        self.warning_label.setMinimumSize(QSize(0, 0))
        self.warning_label.setWordWrap(True)

        self.verticalLayout.addWidget(self.warning_label)

        self.layout_recording_time = QHBoxLayout()
        self.layout_recording_time.setObjectName(u"layout_recording_time")
        self.lbl_step3 = QLabel(Dialog)
        self.lbl_step3.setObjectName(u"lbl_step3")

        self.layout_recording_time.addWidget(self.lbl_step3)

        self.spinBox_recording_time = QSpinBox(Dialog)
        self.spinBox_recording_time.setObjectName(u"spinBox_recording_time")
        self.spinBox_recording_time.setMaximum(1000000)
        self.spinBox_recording_time.setValue(3600)

        self.layout_recording_time.addWidget(self.spinBox_recording_time)


        self.verticalLayout.addLayout(self.layout_recording_time)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(54, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.buttonBox = QDialogButtonBox(Dialog)
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
        self.lbl_step1_2.setText(QCoreApplication.translate("Dialog", u"1. Select the file format :     ", None))
        self.rdbtn_csv.setText(QCoreApplication.translate("Dialog", u"csv", None))
        self.rdbtn_edf.setText(QCoreApplication.translate("Dialog", u"bdf", None))
        self.lbl_step2.setText(QCoreApplication.translate("Dialog", u"2. Select the folder and name to store the file:", None))
        self.lbl_folder.setText(QCoreApplication.translate("Dialog", u"Folder:", None))
        self.btn_browse.setText(QCoreApplication.translate("Dialog", u"Browse", None))
        self.lbl_file_name.setText(QCoreApplication.translate("Dialog", u"File name:", None))
        self.warning_label.setText(QCoreApplication.translate("Dialog", u"<html><head/><body><p><span style=\" color:#d90000;\">A file name can't contain any of the following characters:</span></p><p align=\"center\"><span style=\" color:#d90000;\">| \\ ? * &lt; &quot; : &gt; + [ ] / '</span></p></body></html>", None))
        self.lbl_step3.setText(QCoreApplication.translate("Dialog", u"3. Select recording time (s):", None))
    # retranslateUi

