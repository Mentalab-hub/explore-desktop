# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialog_plot_settings_light.ui'
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
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout
)


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(430, 236)
        self.verticalLayout_3 = QVBoxLayout(Dialog)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 25, -1, -1)
        self.cb_offset = QCheckBox(Dialog)
        self.cb_offset.setObjectName(u"cb_offset")

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

        self.groupBox = QGroupBox(Dialog)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout_3 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.label_3)

        self.value_lowcutoff = QLineEdit(self.groupBox)
        self.value_lowcutoff.setObjectName(u"value_lowcutoff")
        self.value_lowcutoff.setMinimumSize(QSize(0, 20))
        self.value_lowcutoff.setStyleSheet(u"")

        self.verticalLayout.addWidget(self.value_lowcutoff)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_5)

        self.value_highcutoff = QLineEdit(self.groupBox)
        self.value_highcutoff.setObjectName(u"value_highcutoff")
        self.value_highcutoff.setMinimumSize(QSize(0, 20))

        self.verticalLayout_2.addWidget(self.value_highcutoff)


        self.horizontalLayout.addLayout(self.verticalLayout_2)


        self.horizontalLayout_3.addLayout(self.horizontalLayout)


        self.verticalLayout_3.addWidget(self.groupBox)

        self.lbl_warning = QLabel(Dialog)
        self.lbl_warning.setObjectName(u"lbl_warning")
        font = QFont()
        font.setPointSize(10)
        font.setItalic(True)
        self.lbl_warning.setFont(font)
        self.lbl_warning.setStyleSheet(u"color: #d90000;")
        self.lbl_warning.setWordWrap(True)

        self.verticalLayout_3.addWidget(self.lbl_warning)

        self.buttonBox = QDialogButtonBox(Dialog)
        self.buttonBox.setObjectName(u"buttonBox")
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
        self.cb_offset.setText(QCoreApplication.translate("Dialog", u"Baseline Correction", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"Notch Filter", None))
        self.groupBox.setTitle(QCoreApplication.translate("Dialog", u"Bandpass filter", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"Low Cutoff Frequency (Hz)", None))
        self.label_5.setText(QCoreApplication.translate("Dialog", u"High Cutoff Frequency (Hz)", None))
        self.lbl_warning.setText("")
    # retranslateUi

