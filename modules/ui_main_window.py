# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.2.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QSplitter,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from pyqtgraph import (GraphicsLayoutWidget, PlotWidget)
import app_resources_rc
import app_resources_rc
import app_resources_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1116, 691)
        self.actionExport_Metadata = QAction(MainWindow)
        self.actionExport_Metadata.setObjectName(u"actionExport_Metadata")
        self.actionDocumentation = QAction(MainWindow)
        self.actionDocumentation.setObjectName(u"actionDocumentation")
        self.actionReport_an_issue = QAction(MainWindow)
        self.actionReport_an_issue.setObjectName(u"actionReport_an_issue")
        self.actionReport_a_bug = QAction(MainWindow)
        self.actionReport_a_bug.setObjectName(u"actionReport_a_bug")
        self.actionContact = QAction(MainWindow)
        self.actionContact.setObjectName(u"actionContact")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(140, 0))
        self.centralwidget.setStyleSheet(u"/*GENERAL*/\n"
"QWidget{\n"
"	font: 13pt  \"DM Sans\";\n"
"}\n"
" \n"
"QFrame{\n"
"	background-color: rgb(28, 30, 42);\n"
"	border: none;\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #ffffff;\n"
"	/*font: 13pt \"\"*/\n"
"}\n"
"\n"
"QLabel:disabled{\n"
"	color: rgb(128, 128, 128);\n"
"}\n"
"\n"
"\n"
"QPushButton{\n"
"	color: #FFF;\n"
"	background-color: rgb(84, 89, 124);\n"
"	border: 2px solid rgb(84, 89, 124);\n"
"	padding: 5px;\n"
"	border-radius: 5px;\n"
"	font: 12pt;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"	background-color: rgb(101, 106, 141);\n"
"	border:  2px solid rgb(61, 64, 89);\n"
"}\n"
"\n"
"QCheckBox{\n"
"	color: #fff;	\n"
"	background-color: transparent;\n"
"	border-color: #fff;	\n"
"\n"
"}\n"
"\n"
"QToolTip{\n"
"	background-color: transparent;\n"
"	color: #FFF;\n"
"	border: 1px solid rgb(218, 217, 219);\n"
"}\n"
"\n"
"QComboBox{\n"
"	color: #FFF;\n"
"	border: 1px solid #FFF;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView{\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QSp"
                        "inBox{\n"
"	color: #FFF;\n"
"	background-color: transparent;\n"
"	border-color: #FFF;\n"
"} \n"
"\n"
"QSpinBox:disabled{\n"
"	color: rgb(128, 128, 128);\n"
"	background-color: transparent;\n"
"	border-color: rgb(128, 128, 128);\n"
"} \n"
"\n"
"#centralwidget{\n"
"	background-color: #2F4858;\n"
"}\n"
"\n"
"/*HEADER*/\n"
"#main_header{\n"
"	background-color: rgb(27, 29, 39);\n"
"	border-bottom: 2px solid rgb(95, 197, 201);\n"
"}\n"
"\n"
"#main_header .QFrame{\n"
"	border: none;\n"
"}\n"
"\n"
"#top_right_btns .QPushButton{\n"
"	background-color: rgb(27, 29, 39);\n"
"	border-radius: 5px;\n"
"	border: none\n"
"}\n"
"#top_right_btns .QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"\n"
"\n"
"\n"
"/*FOOTER*/\n"
"#main_footer{\n"
"	background-color: rgb(27, 29, 39);\n"
"	border-top: 1px solid rgb(95, 197, 201); \n"
"}\n"
"\n"
"/*LEFT SIDE MENU*/\n"
"#left_side_menu {\n"
"	border:none;\n"
"	border-right: 1px solid rgb(95, 197, 201);\n"
"}\n"
"#toggle_left_menu{\n"
"	border:none;\n"
"}\n"
"\n"
"#btns"
                        "_left_menu{\n"
"	border:none;\n"
"}\n"
"\n"
"#left_side_menu .QPushButton{\n"
"	border-radius: 5px;\n"
"	background-position: left center;\n"
"    	background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"	color: rgb(189, 189, 189);\n"
"}\n"
"\n"
"#left_side_menu .QPushButton:hover {\n"
"	background-color: rgb(61, 64, 89);\n"
"	border-left: 20px solid rgb(61, 64, 89);\n"
"\n"
"}\n"
"#left_side_menu .QPushButton:pressed {\n"
"	background-color: rgb(113, 120, 159);\n"
"	border-left:  20px solid rgb(113, 120, 159);\n"
"}\n"
"\n"
"/*button icons*/\n"
"#btn_home{\n"
"	background-image: url(:/icons/icons/cil-home.png);\n"
"}\n"
"\n"
"#btn_impedance{\n"
"	background-image: url(:/icons/icons/cil-speedometer.png);\n"
"}\n"
"\n"
"#btn_integration{\n"
"	background-image: url(:/icons/icons/cil-share-boxed.png);\n"
"}\n"
"\n"
"#btn_plots{\n"
"	background-image: url(:/icons/icons/cil-chart-line.png);\n"
"}\n"
"\n"
"#btn_settings{\n"
"	back"
                        "ground-image: url(:/icons/icons/cil-settings.png);\n"
"}\n"
"\n"
"#btn_left_menu_toggle{\n"
"	background-image: url(:/icons/icons/icon_menu.png);\n"
"}\n"
"\n"
"/*TITLES FONT*/\n"
"\n"
"QLabel#integration_title{\n"
"	color: #FFF;\n"
"	border:none;\n"
"	font: 22pt \"DM Sans\";\n"
"}\n"
"\n"
"QFrame#frame_integration_title{\n"
"	border-bottom: 1px solid #FFF;\n"
"	border-top: none;\n"
"	border-left: none;\n"
"	border-right: none;\n"
"}\n"
"\n"
"QLabel#home_title{\n"
"	color: #FFF;\n"
"	border:none;\n"
"	font: 22pt \"DM Sans\";\n"
"}\n"
"\n"
"QFrame#frame_home_title{\n"
"	border-bottom: 1px solid #FFF;\n"
"	border-top: none;\n"
"	border-left: none;\n"
"	border-right: none;\n"
"}\n"
"\n"
"QLabel#impedance_title{\n"
"	color: #FFF;\n"
"	border:none;\n"
"	font: 22pt \"DM Sans\";\n"
"}\n"
"\n"
"QFrame#frame_impedance_title{\n"
"	border-bottom: 1px solid #FFF;\n"
"	border-top: none;\n"
"	border-left: none;\n"
"	border-right: none;\n"
"}\n"
"\n"
"QLabel#settings_title{\n"
"	color: #FFF;\n"
"	border:none;\n"
"	font: 22pt"
                        " \"DM Sans\";\n"
"}\n"
"\n"
"QFrame#frame_settings_title{\n"
"	border-bottom: 1px solid #FFF;\n"
"	border-top: none;\n"
"	border-left: none;\n"
"	border-right: none;\n"
"}\n"
"\n"
"\n"
"/*TABS*/\n"
"QTabWidget::pane{\n"
"	border: 1px solid #FFF\n"
"}\n"
"QTabWidget::tab-bar{\n"
"	alignment: left;\n"
"}\n"
"QTabBar::tab \n"
"{\n"
"    background: transparent;\n"
"    color: white;\n"
"	border: 1px solid rgb(228, 227, 229);\n"
"	/*margin-left: 2;\n"
"	margin-right: 2;*/\n"
"	width: 100px;\n"
"	padding: 2px\n"
"}\n"
"\n"
"QTabBar::tab:selected\n"
"{\n"
"	background: rgba(49, 52, 86, 215);\n"
"    color: white;   \n"
"    border: 2px solid rgb(255, 255, 255);\n"
"}, \n"
"QTabBar::tab:hover \n"
"{\n"
"	text-decoration: underline;\n"
"}")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.main_header = QFrame(self.centralwidget)
        self.main_header.setObjectName(u"main_header")
        self.main_header.setEnabled(True)
        self.main_header.setMinimumSize(QSize(0, 30))
        self.main_header.setMaximumSize(QSize(16777215, 50))
        self.main_header.setStyleSheet(u"")
        self.main_header.setFrameShape(QFrame.WinPanel)
        self.main_header.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.main_header)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.title_bar_container = QFrame(self.main_header)
        self.title_bar_container.setObjectName(u"title_bar_container")
        self.title_bar_container.setMaximumSize(QSize(16777215, 50))
        self.title_bar_container.setStyleSheet(u"")
        self.title_bar_container.setFrameShape(QFrame.StyledPanel)
        self.title_bar_container.setFrameShadow(QFrame.Raised)
        self.mentalab_logo = QLabel(self.title_bar_container)
        self.mentalab_logo.setObjectName(u"mentalab_logo")
        self.mentalab_logo.setGeometry(QRect(0, 3, 121, 21))
        self.mentalab_logo.setMaximumSize(QSize(16777215, 50))
        self.mentalab_logo.setPixmap(QPixmap(u":/image/images/MentalabLogo_full.png"))
        self.mentalab_logo.setScaledContents(True)
        self.mentalab_logo.setWordWrap(True)
        self.mentalab_logo.setOpenExternalLinks(False)

        self.horizontalLayout_2.addWidget(self.title_bar_container)

        self.top_right_btns = QFrame(self.main_header)
        self.top_right_btns.setObjectName(u"top_right_btns")
        self.top_right_btns.setMaximumSize(QSize(100, 16777215))
        self.top_right_btns.setStyleSheet(u"")
        self.top_right_btns.setFrameShape(QFrame.WinPanel)
        self.top_right_btns.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.top_right_btns)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.btn_minimize = QPushButton(self.top_right_btns)
        self.btn_minimize.setObjectName(u"btn_minimize")
        self.btn_minimize.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_minimize.setStyleSheet(u"")
        icon = QIcon()
        icon.addFile(u":/icons/icons/icon_minimize.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_minimize.setIcon(icon)
        self.btn_minimize.setIconSize(QSize(24, 24))

        self.horizontalLayout_3.addWidget(self.btn_minimize)

        self.btn_restore = QPushButton(self.top_right_btns)
        self.btn_restore.setObjectName(u"btn_restore")
        self.btn_restore.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_restore.setStyleSheet(u"")
        icon1 = QIcon()
        icon1.addFile(u":/icons/icons/icon_maximize.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_restore.setIcon(icon1)
        self.btn_restore.setIconSize(QSize(24, 24))

        self.horizontalLayout_3.addWidget(self.btn_restore)

        self.btn_close = QPushButton(self.top_right_btns)
        self.btn_close.setObjectName(u"btn_close")
        self.btn_close.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_close.setStyleSheet(u"")
        icon2 = QIcon()
        icon2.addFile(u":/icons/icons/cil-x.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_close.setIcon(icon2)
        self.btn_close.setIconSize(QSize(24, 24))

        self.horizontalLayout_3.addWidget(self.btn_close)


        self.horizontalLayout_2.addWidget(self.top_right_btns)


        self.verticalLayout.addWidget(self.main_header)

        self.main_body = QFrame(self.centralwidget)
        self.main_body.setObjectName(u"main_body")
        self.main_body.setMinimumSize(QSize(0, 50))
        self.main_body.setStyleSheet(u"background-color: rgb(28, 30, 42);")
        self.main_body.setFrameShape(QFrame.StyledPanel)
        self.main_body.setFrameShadow(QFrame.Raised)
        self.horizontalLayout = QHBoxLayout(self.main_body)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.left_side_menu = QFrame(self.main_body)
        self.left_side_menu.setObjectName(u"left_side_menu")
        self.left_side_menu.setMinimumSize(QSize(60, 0))
        self.left_side_menu.setMaximumSize(QSize(60, 16777215))
        self.left_side_menu.setStyleSheet(u"#left_side_menu .QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"\n"
"\n"
"#left_side_menu .QPushButton:pressed{\n"
"	background-color: rgb(113, 120, 159);\n"
"	\n"
"}\n"
"")
        self.left_side_menu.setFrameShape(QFrame.StyledPanel)
        self.left_side_menu.setFrameShadow(QFrame.Raised)
        self.verticalLayout_29 = QVBoxLayout(self.left_side_menu)
        self.verticalLayout_29.setSpacing(0)
        self.verticalLayout_29.setObjectName(u"verticalLayout_29")
        self.verticalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.toggle_left_menu = QFrame(self.left_side_menu)
        self.toggle_left_menu.setObjectName(u"toggle_left_menu")
        self.toggle_left_menu.setStyleSheet(u"")
        self.toggle_left_menu.setFrameShape(QFrame.StyledPanel)
        self.toggle_left_menu.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_67 = QHBoxLayout(self.toggle_left_menu)
        self.horizontalLayout_67.setSpacing(0)
        self.horizontalLayout_67.setObjectName(u"horizontalLayout_67")
        self.horizontalLayout_67.setContentsMargins(0, 0, 0, 0)
        self.btn_left_menu_toggle = QPushButton(self.toggle_left_menu)
        self.btn_left_menu_toggle.setObjectName(u"btn_left_menu_toggle")
        self.btn_left_menu_toggle.setMinimumSize(QSize(0, 45))
        self.btn_left_menu_toggle.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_left_menu_toggle.setStyleSheet(u"")

        self.horizontalLayout_67.addWidget(self.btn_left_menu_toggle)


        self.verticalLayout_29.addWidget(self.toggle_left_menu, 0, Qt.AlignTop)

        self.btns_left_menu = QFrame(self.left_side_menu)
        self.btns_left_menu.setObjectName(u"btns_left_menu")
        self.btns_left_menu.setStyleSheet(u"")
        self.btns_left_menu.setFrameShape(QFrame.StyledPanel)
        self.btns_left_menu.setFrameShadow(QFrame.Raised)
        self.verticalLayout_30 = QVBoxLayout(self.btns_left_menu)
        self.verticalLayout_30.setSpacing(0)
        self.verticalLayout_30.setObjectName(u"verticalLayout_30")
        self.verticalLayout_30.setContentsMargins(0, 0, 0, 0)
        self.btn_home = QPushButton(self.btns_left_menu)
        self.btn_home.setObjectName(u"btn_home")
        self.btn_home.setMinimumSize(QSize(0, 45))
        self.btn_home.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_home.setStyleSheet(u"")

        self.verticalLayout_30.addWidget(self.btn_home)

        self.btn_settings = QPushButton(self.btns_left_menu)
        self.btn_settings.setObjectName(u"btn_settings")
        self.btn_settings.setMinimumSize(QSize(0, 45))
        self.btn_settings.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_settings.setStyleSheet(u"")

        self.verticalLayout_30.addWidget(self.btn_settings)

        self.btn_plots = QPushButton(self.btns_left_menu)
        self.btn_plots.setObjectName(u"btn_plots")
        self.btn_plots.setMinimumSize(QSize(0, 45))
        self.btn_plots.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_plots.setStyleSheet(u"")

        self.verticalLayout_30.addWidget(self.btn_plots)

        self.btn_impedance = QPushButton(self.btns_left_menu)
        self.btn_impedance.setObjectName(u"btn_impedance")
        self.btn_impedance.setMinimumSize(QSize(0, 45))
        self.btn_impedance.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_impedance.setStyleSheet(u"")

        self.verticalLayout_30.addWidget(self.btn_impedance)

        self.btn_integration = QPushButton(self.btns_left_menu)
        self.btn_integration.setObjectName(u"btn_integration")
        self.btn_integration.setMinimumSize(QSize(0, 45))
        self.btn_integration.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_integration.setStyleSheet(u"")

        self.verticalLayout_30.addWidget(self.btn_integration)


        self.verticalLayout_29.addWidget(self.btns_left_menu, 0, Qt.AlignTop)

        self.left_menu_frame = QFrame(self.left_side_menu)
        self.left_menu_frame.setObjectName(u"left_menu_frame")
        self.left_menu_frame.setMinimumSize(QSize(0, 600))
        self.left_menu_frame.setStyleSheet(u"")
        self.left_menu_frame.setFrameShape(QFrame.StyledPanel)
        self.left_menu_frame.setFrameShadow(QFrame.Raised)

        self.verticalLayout_29.addWidget(self.left_menu_frame)


        self.horizontalLayout.addWidget(self.left_side_menu)

        self.center_main_items = QFrame(self.main_body)
        self.center_main_items.setObjectName(u"center_main_items")
        self.center_main_items.setStyleSheet(u"")
        self.center_main_items.setFrameShape(QFrame.WinPanel)
        self.center_main_items.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.center_main_items)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.stackedWidget = QStackedWidget(self.center_main_items)
        self.stackedWidget.setObjectName(u"stackedWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setMaximumSize(QSize(16777215, 16777215))
        self.stackedWidget.setStyleSheet(u"f")
        self.page_home = QWidget()
        self.page_home.setObjectName(u"page_home")
        self.verticalLayout_18 = QVBoxLayout(self.page_home)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.frame_home_title = QFrame(self.page_home)
        self.frame_home_title.setObjectName(u"frame_home_title")
        self.frame_home_title.setMaximumSize(QSize(16777215, 50))
        self.frame_home_title.setStyleSheet(u"")
        self.frame_home_title.setFrameShape(QFrame.StyledPanel)
        self.frame_home_title.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_19 = QHBoxLayout(self.frame_home_title)
        self.horizontalLayout_19.setSpacing(0)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.horizontalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.home_logo = QLabel(self.frame_home_title)
        self.home_logo.setObjectName(u"home_logo")
        self.home_logo.setMaximumSize(QSize(51, 51))
        self.home_logo.setStyleSheet(u"")
        self.home_logo.setPixmap(QPixmap(u":/image/images/MentalabLogo.png"))
        self.home_logo.setScaledContents(True)

        self.horizontalLayout_19.addWidget(self.home_logo)

        self.home_title = QLabel(self.frame_home_title)
        self.home_title.setObjectName(u"home_title")
        self.home_title.setMinimumSize(QSize(0, 0))
        font = QFont()
        font.setFamilies([u"DM Sans"])
        font.setPointSize(22)
        font.setBold(False)
        font.setItalic(False)
        self.home_title.setFont(font)
        self.home_title.setStyleSheet(u"")
        self.home_title.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_19.addWidget(self.home_title)


        self.verticalLayout_18.addWidget(self.frame_home_title)

        self.label = QLabel(self.page_home)
        self.label.setObjectName(u"label")
        self.label.setStyleSheet(u"")
        self.label.setMargin(15)

        self.verticalLayout_18.addWidget(self.label)

        self.stackedWidget.addWidget(self.page_home)
        self.page_integration = QWidget()
        self.page_integration.setObjectName(u"page_integration")
        self.verticalLayout_12 = QVBoxLayout(self.page_integration)
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.frame_integration_title = QFrame(self.page_integration)
        self.frame_integration_title.setObjectName(u"frame_integration_title")
        self.frame_integration_title.setMaximumSize(QSize(16777215, 50))
        self.frame_integration_title.setStyleSheet(u"")
        self.frame_integration_title.setFrameShape(QFrame.StyledPanel)
        self.frame_integration_title.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_29 = QHBoxLayout(self.frame_integration_title)
        self.horizontalLayout_29.setSpacing(0)
        self.horizontalLayout_29.setObjectName(u"horizontalLayout_29")
        self.horizontalLayout_29.setContentsMargins(0, 0, 0, 0)
        self.integration_log = QLabel(self.frame_integration_title)
        self.integration_log.setObjectName(u"integration_log")
        self.integration_log.setMaximumSize(QSize(51, 51))
        self.integration_log.setPixmap(QPixmap(u":/image/images/MentalabLogo.png"))
        self.integration_log.setScaledContents(True)

        self.horizontalLayout_29.addWidget(self.integration_log)

        self.integration_title = QLabel(self.frame_integration_title)
        self.integration_title.setObjectName(u"integration_title")
        self.integration_title.setMinimumSize(QSize(0, 0))
        self.integration_title.setFont(font)
        self.integration_title.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_29.addWidget(self.integration_title)


        self.verticalLayout_12.addWidget(self.frame_integration_title)

        self.frame_integration = QFrame(self.page_integration)
        self.frame_integration.setObjectName(u"frame_integration")
        self.frame_integration.setStyleSheet(u"QPushButton{\n"
"	color: #FFF;\n"
"	background-color: rgb(84, 89, 124);\n"
"	border: 2px solid rgb(84, 89, 124);\n"
"	padding: 5px;\n"
"	border-radius: 5px;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(61, 64, 89);\n"
"	border:  2px solid rgb(61, 64, 89);\n"
"}\n"
"\n"
"QCheckBox{\n"
"	color: #fff;	\n"
# "	background-color: transparent;\n"
"\n"
"}"
# "\n"
# "QCheckBox::indicator{\n"
# "	border-color: #fff;	\n"
# # "	background-color: transparent;\n"
# "\n"
# "}"
)
        self.frame_integration.setFrameShape(QFrame.StyledPanel)
        self.frame_integration.setFrameShadow(QFrame.Raised)
        self.verticalLayout_36 = QVBoxLayout(self.frame_integration)
        self.verticalLayout_36.setObjectName(u"verticalLayout_36")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_2 = QLabel(self.frame_integration)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMaximumSize(QSize(16777215, 30))
        font1 = QFont()
        font1.setFamilies([u"DM Sans"])
        font1.setPointSize(13)
        font1.setBold(False)
        font1.setItalic(False)
        font1.setKerning(True)
        self.label_2.setFont(font1)

        self.verticalLayout_3.addWidget(self.label_2)

        self.label_11 = QLabel(self.frame_integration)
        self.label_11.setObjectName(u"label_11")
        font2 = QFont()
        font2.setFamilies([u"DM Sans"])
        font2.setPointSize(13)
        font2.setBold(False)
        font2.setItalic(False)
        self.label_11.setFont(font2)
        self.label_11.setWordWrap(True)

        self.verticalLayout_3.addWidget(self.label_11)

        self.checkBox = QCheckBox(self.frame_integration)
        self.checkBox.setObjectName(u"checkBox")

        self.verticalLayout_3.addWidget(self.checkBox)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_13 = QLabel(self.frame_integration)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setEnabled(False)

        self.horizontalLayout_13.addWidget(self.label_13)

        self.spinBox = QSpinBox(self.frame_integration)
        self.spinBox.setObjectName(u"spinBox")
        self.spinBox.setEnabled(False)
        self.spinBox.setMaximumSize(QSize(150, 16777215))

        self.horizontalLayout_13.addWidget(self.spinBox)

        self.empty_frame_4 = QFrame(self.frame_integration)
        self.empty_frame_4.setObjectName(u"empty_frame_4")
        self.empty_frame_4.setFrameShape(QFrame.StyledPanel)
        self.empty_frame_4.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_13.addWidget(self.empty_frame_4)

        self.empty_frame_3 = QFrame(self.frame_integration)
        self.empty_frame_3.setObjectName(u"empty_frame_3")
        self.empty_frame_3.setFrameShape(QFrame.StyledPanel)
        self.empty_frame_3.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_13.addWidget(self.empty_frame_3)

        self.empty_frame_2 = QFrame(self.frame_integration)
        self.empty_frame_2.setObjectName(u"empty_frame_2")
        self.empty_frame_2.setFrameShape(QFrame.StyledPanel)
        self.empty_frame_2.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_13.addWidget(self.empty_frame_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_13)

        self.btn_push_lsl = QPushButton(self.frame_integration)
        self.btn_push_lsl.setObjectName(u"btn_push_lsl")
        self.btn_push_lsl.setMinimumSize(QSize(100, 30))
        self.btn_push_lsl.setMaximumSize(QSize(100, 30))

        self.verticalLayout_3.addWidget(self.btn_push_lsl, 0, Qt.AlignHCenter)


        self.verticalLayout_36.addLayout(self.verticalLayout_3)

        self.verticalSpacer = QSpacerItem(20, 400, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.verticalLayout_36.addItem(self.verticalSpacer)

        self.empty_frame = QFrame(self.frame_integration)
        self.empty_frame.setObjectName(u"empty_frame")
        self.empty_frame.setFrameShape(QFrame.StyledPanel)
        self.empty_frame.setFrameShadow(QFrame.Raised)

        self.verticalLayout_36.addWidget(self.empty_frame)


        self.verticalLayout_12.addWidget(self.frame_integration)

        self.stackedWidget.addWidget(self.page_integration)
        self.page_bt = QWidget()
        self.page_bt.setObjectName(u"page_bt")
        self.verticalLayout_23 = QVBoxLayout(self.page_bt)
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.frame_settings_title = QFrame(self.page_bt)
        self.frame_settings_title.setObjectName(u"frame_settings_title")
        self.frame_settings_title.setMaximumSize(QSize(16777215, 50))
        self.frame_settings_title.setStyleSheet(u"QLabel{\n"
"	color: #FFF;\n"
"	border:none;\n"
"}\n"
"\n"
"QFrame{\n"
"	border-bottom: 1px solid #FFF;\n"
"	border-top: none;\n"
"	border-left: none;\n"
"	border-right: none;\n"
"}\n"
"")
        self.frame_settings_title.setFrameShape(QFrame.StyledPanel)
        self.frame_settings_title.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_settings_title)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.settings_log = QLabel(self.frame_settings_title)
        self.settings_log.setObjectName(u"settings_log")
        self.settings_log.setMaximumSize(QSize(51, 51))
        self.settings_log.setPixmap(QPixmap(u":/image/images/MentalabLogo.png"))
        self.settings_log.setScaledContents(True)

        self.horizontalLayout_5.addWidget(self.settings_log)

        self.settings_title = QLabel(self.frame_settings_title)
        self.settings_title.setObjectName(u"settings_title")
        self.settings_title.setMinimumSize(QSize(0, 0))
        self.settings_title.setFont(font)
        self.settings_title.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_5.addWidget(self.settings_title)


        self.verticalLayout_23.addWidget(self.frame_settings_title)

        self.frame_settings = QFrame(self.page_bt)
        self.frame_settings.setObjectName(u"frame_settings")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.frame_settings.sizePolicy().hasHeightForWidth())
        self.frame_settings.setSizePolicy(sizePolicy1)
        self.frame_settings.setStyleSheet(u"QPushButton{\n"
"	color: #FFF;\n"
"	background-color: rgb(84, 89, 124);\n"
"	border: 2px solid rgb(84, 89, 124);\n"
"	padding: 5px;\n"
"	border-radius: 5px;\n"
"	font: 11pt;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(61, 64, 89);\n"
"	border:  2px solid rgb(61, 64, 89);\n"
"}\n"
"\n"
"QLineEdit{\n"
"	color: #FFF;\n"
"	border: 1px solid rgb(84, 89, 124);\n"
"}\n"
"\n"
"QListWidget{\n"
"	border: 1px solid rgb(84, 89, 124);\n"
"	color: #FFFFFF;\n"
"}\n"
"\n"
"Line{\n"
"	background-color: rgb(255, 255, 255);\n"
"	border: none\n"
"}\n"
"\n"
"QCheckBox{\n"
"	color:#FFF;\n"
"	border: none;\n"
"}\n"
"\n"
"#label_10{\n"
"	font: 11pt;\n"
"}\n"
"")
        self.frame_settings.setFrameShape(QFrame.StyledPanel)
        self.frame_settings.setFrameShadow(QFrame.Raised)
        self.verticalLayout_19 = QVBoxLayout(self.frame_settings)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.horizontalLayout_31 = QHBoxLayout()
        self.horizontalLayout_31.setObjectName(u"horizontalLayout_31")
        self.frame_BT = QFrame(self.frame_settings)
        self.frame_BT.setObjectName(u"frame_BT")
        self.frame_BT.setMaximumSize(QSize(400, 400))
        self.frame_BT.setStyleSheet(u"")
        self.frame_BT.setFrameShape(QFrame.StyledPanel)
        self.frame_BT.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.frame_BT)
        self.verticalLayout_6.setSpacing(20)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label_explore_name_2 = QLabel(self.frame_BT)
        self.label_explore_name_2.setObjectName(u"label_explore_name_2")
        self.label_explore_name_2.setMaximumSize(QSize(16777215, 30))
        self.label_explore_name_2.setStyleSheet(u"")
        self.label_explore_name_2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.label_explore_name_2)

        self.label_10 = QLabel(self.frame_BT)
        self.label_10.setObjectName(u"label_10")
        font3 = QFont()
        font3.setFamilies([u"DM Sans"])
        font3.setPointSize(11)
        font3.setBold(False)
        font3.setItalic(False)
        self.label_10.setFont(font3)
        self.label_10.setStyleSheet(u"")
        self.label_10.setWordWrap(True)

        self.verticalLayout_6.addWidget(self.label_10)

        self.frame = QFrame(self.frame_BT)
        self.frame.setObjectName(u"frame")
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setStyleSheet(u"border:none\n"
"")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.frame)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.frame_5 = QFrame(self.frame)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Raised)
        self.verticalLayout_24 = QVBoxLayout(self.frame_5)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.list_devices = QListWidget(self.frame_5)
        self.list_devices.setObjectName(u"list_devices")
        self.list_devices.viewport().setProperty("cursor", QCursor(Qt.ArrowCursor))
        self.list_devices.setStyleSheet(u"border: 1px solid rgb(84, 89, 124);\n"
"color: #FFFFFF;")

        self.verticalLayout_24.addWidget(self.list_devices)


        self.horizontalLayout_9.addWidget(self.frame_5)

        self.frame_btns_scan_connect = QFrame(self.frame)
        self.frame_btns_scan_connect.setObjectName(u"frame_btns_scan_connect")
        self.frame_btns_scan_connect.setMaximumSize(QSize(16777215, 80))
        self.frame_btns_scan_connect.setStyleSheet(u"")
        self.frame_btns_scan_connect.setFrameShape(QFrame.StyledPanel)
        self.frame_btns_scan_connect.setFrameShadow(QFrame.Raised)
        self.verticalLayout_17 = QVBoxLayout(self.frame_btns_scan_connect)
        self.verticalLayout_17.setSpacing(10)
        self.verticalLayout_17.setObjectName(u"verticalLayout_17")
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.btn_scan = QPushButton(self.frame_btns_scan_connect)
        self.btn_scan.setObjectName(u"btn_scan")
        sizePolicy2 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.btn_scan.sizePolicy().hasHeightForWidth())
        self.btn_scan.setSizePolicy(sizePolicy2)
        self.btn_scan.setMinimumSize(QSize(140, 30))
        self.btn_scan.setMaximumSize(QSize(140, 30))
        self.btn_scan.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_scan.setStyleSheet(u"")
        self.btn_scan.setFlat(False)

        self.verticalLayout_17.addWidget(self.btn_scan)

        self.btn_connect = QPushButton(self.frame_btns_scan_connect)
        self.btn_connect.setObjectName(u"btn_connect")
        sizePolicy.setHeightForWidth(self.btn_connect.sizePolicy().hasHeightForWidth())
        self.btn_connect.setSizePolicy(sizePolicy)
        self.btn_connect.setMinimumSize(QSize(140, 30))
        self.btn_connect.setMaximumSize(QSize(140, 30))
        self.btn_connect.setCursor(QCursor(Qt.PointingHandCursor))

        self.verticalLayout_17.addWidget(self.btn_connect)


        self.horizontalLayout_9.addWidget(self.frame_btns_scan_connect)


        self.verticalLayout_6.addWidget(self.frame)

        self.frame_6 = QFrame(self.frame_BT)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setMinimumSize(QSize(0, 30))
        self.frame_6.setFrameShape(QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_10 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_10.setSpacing(10)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.label_9 = QLabel(self.frame_6)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setStyleSheet(u"")

        self.horizontalLayout_10.addWidget(self.label_9)

        self.dev_name_input = QLineEdit(self.frame_6)
        self.dev_name_input.setObjectName(u"dev_name_input")
        self.dev_name_input.setStyleSheet(u"\n"
"/*background-color: rgb(83, 88, 123);*/\n"
"")

        self.horizontalLayout_10.addWidget(self.dev_name_input)


        self.verticalLayout_6.addWidget(self.frame_6)

        self.line = QFrame(self.frame_BT)
        self.line.setObjectName(u"line")
        self.line.setStyleSheet(u"")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_6.addWidget(self.line)

        self.label_8 = QLabel(self.frame_BT)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_6.addWidget(self.label_8)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.le_data_path = QLineEdit(self.frame_BT)
        self.le_data_path.setObjectName(u"le_data_path")
        self.le_data_path.setFont(font2)

        self.horizontalLayout_12.addWidget(self.le_data_path)

        self.btn_import_data = QPushButton(self.frame_BT)
        self.btn_import_data.setObjectName(u"btn_import_data")
        sizePolicy.setHeightForWidth(self.btn_import_data.sizePolicy().hasHeightForWidth())
        self.btn_import_data.setSizePolicy(sizePolicy)
        self.btn_import_data.setMinimumSize(QSize(140, 30))
        self.btn_import_data.setMaximumSize(QSize(140, 30))
        self.btn_import_data.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayout_12.addWidget(self.btn_import_data)


        self.verticalLayout_6.addLayout(self.horizontalLayout_12)


        self.horizontalLayout_31.addWidget(self.frame_BT)

        self.line_2 = QFrame(self.frame_settings)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setMaximumSize(QSize(1, 16777215))
        self.line_2.setStyleSheet(u"background-color: rgb(205, 207, 207);\n"
"width: 1px")
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.horizontalLayout_31.addWidget(self.line_2)

        self.frame_device = QFrame(self.frame_settings)
        self.frame_device.setObjectName(u"frame_device")
        self.frame_device.setMinimumSize(QSize(0, 0))
        self.frame_device.setMaximumSize(QSize(420, 16777215))
        self.frame_device.setStyleSheet(u"")
        self.frame_device.setFrameShape(QFrame.StyledPanel)
        self.frame_device.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.frame_device)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_explore_name = QLabel(self.frame_device)
        self.label_explore_name.setObjectName(u"label_explore_name")
        self.label_explore_name.setMinimumSize(QSize(0, 0))
        self.label_explore_name.setMaximumSize(QSize(16777215, 75))
        self.label_explore_name.setStyleSheet(u"border: none;\n"
"color: #FFF")
        self.label_explore_name.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_explore_name)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_12 = QLabel(self.frame_device)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_8.addWidget(self.label_12)

        self.n_chan = QComboBox(self.frame_device)
        self.n_chan.setObjectName(u"n_chan")

        self.horizontalLayout_8.addWidget(self.n_chan)


        self.verticalLayout_2.addLayout(self.horizontalLayout_8)

        self.frame_channels = QFrame(self.frame_device)
        self.frame_channels.setObjectName(u"frame_channels")
        self.frame_channels.setStyleSheet(u"")
        self.frame_channels.setFrameShape(QFrame.StyledPanel)
        self.frame_channels.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame_channels)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_activeChannels = QLabel(self.frame_channels)
        self.label_activeChannels.setObjectName(u"label_activeChannels")

        self.verticalLayout_5.addWidget(self.label_activeChannels)

        self.frame_cb_channels = QFrame(self.frame_channels)
        self.frame_cb_channels.setObjectName(u"frame_cb_channels")
        self.frame_cb_channels.setFrameShape(QFrame.StyledPanel)
        self.frame_cb_channels.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_cb_channels)
        self.horizontalLayout_6.setSpacing(5)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.cb_ch1 = QCheckBox(self.frame_cb_channels)
        self.cb_ch1.setObjectName(u"cb_ch1")
        self.cb_ch1.setCursor(QCursor(Qt.PointingHandCursor))
        self.cb_ch1.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch1)

        self.cb_ch2 = QCheckBox(self.frame_cb_channels)
        self.cb_ch2.setObjectName(u"cb_ch2")
        self.cb_ch2.setCursor(QCursor(Qt.PointingHandCursor))
        self.cb_ch2.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch2)

        self.cb_ch3 = QCheckBox(self.frame_cb_channels)
        self.cb_ch3.setObjectName(u"cb_ch3")
        self.cb_ch3.setCursor(QCursor(Qt.PointingHandCursor))
        self.cb_ch3.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch3)

        self.cb_ch4 = QCheckBox(self.frame_cb_channels)
        self.cb_ch4.setObjectName(u"cb_ch4")
        self.cb_ch4.setCursor(QCursor(Qt.PointingHandCursor))
        self.cb_ch4.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch4)

        self.cb_ch5 = QCheckBox(self.frame_cb_channels)
        self.cb_ch5.setObjectName(u"cb_ch5")
        self.cb_ch5.setCursor(QCursor(Qt.PointingHandCursor))
        self.cb_ch5.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch5)

        self.cb_ch6 = QCheckBox(self.frame_cb_channels)
        self.cb_ch6.setObjectName(u"cb_ch6")
        self.cb_ch6.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch6)

        self.cb_ch7 = QCheckBox(self.frame_cb_channels)
        self.cb_ch7.setObjectName(u"cb_ch7")
        self.cb_ch7.setCursor(QCursor(Qt.PointingHandCursor))
        self.cb_ch7.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch7)

        self.cb_ch8 = QCheckBox(self.frame_cb_channels)
        self.cb_ch8.setObjectName(u"cb_ch8")
        self.cb_ch8.setCursor(QCursor(Qt.PointingHandCursor))
        self.cb_ch8.setStyleSheet(u"")

        self.horizontalLayout_6.addWidget(self.cb_ch8)


        self.verticalLayout_5.addWidget(self.frame_cb_channels)


        self.verticalLayout_2.addWidget(self.frame_channels)

        self.frame_samplingrate = QFrame(self.frame_device)
        self.frame_samplingrate.setObjectName(u"frame_samplingrate")
        self.frame_samplingrate.setStyleSheet(u"QLabel{\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QFrame{\n"
"	border:none;\n"
"}\n"
"\n"
"QSpinBox{\n"
"	color: #FFF\n"
"\n"
"}")
        self.frame_samplingrate.setFrameShape(QFrame.StyledPanel)
        self.frame_samplingrate.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_samplingrate)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_samping_rate = QLabel(self.frame_samplingrate)
        self.label_samping_rate.setObjectName(u"label_samping_rate")

        self.horizontalLayout_7.addWidget(self.label_samping_rate)

        self.value_sampling_rate = QComboBox(self.frame_samplingrate)
        self.value_sampling_rate.setObjectName(u"value_sampling_rate")
        self.value_sampling_rate.setStyleSheet(u"")

        self.horizontalLayout_7.addWidget(self.value_sampling_rate)


        self.verticalLayout_2.addWidget(self.frame_samplingrate)

        self.btn_apply_settings = QPushButton(self.frame_device)
        self.btn_apply_settings.setObjectName(u"btn_apply_settings")
        self.btn_apply_settings.setMinimumSize(QSize(150, 30))
        self.btn_apply_settings.setMaximumSize(QSize(150, 30))
        self.btn_apply_settings.setCursor(QCursor(Qt.PointingHandCursor))

        self.verticalLayout_2.addWidget(self.btn_apply_settings, 0, Qt.AlignHCenter)

        self.frame_device_buttons = QFrame(self.frame_device)
        self.frame_device_buttons.setObjectName(u"frame_device_buttons")
        sizePolicy.setHeightForWidth(self.frame_device_buttons.sizePolicy().hasHeightForWidth())
        self.frame_device_buttons.setSizePolicy(sizePolicy)
        self.frame_device_buttons.setStyleSheet(u"")
        self.frame_device_buttons.setFrameShape(QFrame.StyledPanel)
        self.frame_device_buttons.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_device_buttons)
        self.horizontalLayout_4.setSpacing(20)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.btn_format_memory = QPushButton(self.frame_device_buttons)
        self.btn_format_memory.setObjectName(u"btn_format_memory")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.btn_format_memory.sizePolicy().hasHeightForWidth())
        self.btn_format_memory.setSizePolicy(sizePolicy3)
        self.btn_format_memory.setMinimumSize(QSize(120, 30))
        self.btn_format_memory.setMaximumSize(QSize(16777215, 30))
        self.btn_format_memory.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayout_4.addWidget(self.btn_format_memory)

        self.btn_reset_settings = QPushButton(self.frame_device_buttons)
        self.btn_reset_settings.setObjectName(u"btn_reset_settings")
        sizePolicy3.setHeightForWidth(self.btn_reset_settings.sizePolicy().hasHeightForWidth())
        self.btn_reset_settings.setSizePolicy(sizePolicy3)
        self.btn_reset_settings.setMinimumSize(QSize(120, 30))
        self.btn_reset_settings.setMaximumSize(QSize(16777215, 30))
        self.btn_reset_settings.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_reset_settings.setStyleSheet(u"")

        self.horizontalLayout_4.addWidget(self.btn_reset_settings)

        self.btn_calibrate = QPushButton(self.frame_device_buttons)
        self.btn_calibrate.setObjectName(u"btn_calibrate")
        sizePolicy3.setHeightForWidth(self.btn_calibrate.sizePolicy().hasHeightForWidth())
        self.btn_calibrate.setSizePolicy(sizePolicy3)
        self.btn_calibrate.setMinimumSize(QSize(120, 30))
        self.btn_calibrate.setMaximumSize(QSize(16777215, 30))
        self.btn_calibrate.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayout_4.addWidget(self.btn_calibrate)


        self.verticalLayout_2.addWidget(self.frame_device_buttons, 0, Qt.AlignHCenter)


        self.horizontalLayout_31.addWidget(self.frame_device)


        self.verticalLayout_19.addLayout(self.horizontalLayout_31)


        self.verticalLayout_23.addWidget(self.frame_settings)

        self.stackedWidget.addWidget(self.page_bt)
        self.page_plotsNoWidget = QWidget()
        self.page_plotsNoWidget.setObjectName(u"page_plotsNoWidget")
        self.page_plotsNoWidget.setStyleSheet(u"QFrame{\n"
"	border:none;\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"	padding:2px;\n"
"}\n"
"\n"
"QLineEdit{\n"
"	background-color: #53576F;\n"
"	border: 1px solid #FFF;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QComboBox{\n"
"	background-color: #53576F;\n"
"	border: 1px solid #FFF;\n"
"	color: #FFF;\n"
"}\n"
"QComboBox::drop-down{\n"
"	color: #FFF;\n"
"}\n"
"QComboBox QAbstractItemView{\n"
"	color: #FFF;\n"
"}\n"
"QComboBox QListView{\n"
"	color: #FFF;\n"
"}\n"
"QListView{\n"
"	color:#FFF;\n"
"}\n"
"QPushButton{\n"
"	border-radius: 1px;\n"
"	border: 1px solid #FFFFFF;\n"
"	background-position: left center;\n"
"    	background-repeat: no-repeat;\n"
"	color: #FFFFFF;\n"
"	text-align: center;\n"
"	padding: 2px\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"QPushButton:disabled{\n"
"	background-color: #3B3F57;\n"
"	color: #797979;\n"
"}\n"
"\n"
"/*QPushButton{\n"
"	border-radius: 5px;\n"
"	background-position: left center;\n"
"    	background-repeat: no-repeat;\n"
"	border: none;\n"
"	bor"
                        "der-left: 20px solid transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"	color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"	border-left: 20px solid rgb(61, 64, 89);*/\n"
"")
        self.verticalLayout_21 = QVBoxLayout(self.page_plotsNoWidget)
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.horizontalLayout_32 = QHBoxLayout()
        self.horizontalLayout_32.setObjectName(u"horizontalLayout_32")
        self.label_signal = QLabel(self.page_plotsNoWidget)
        self.label_signal.setObjectName(u"label_signal")
        self.label_signal.setFont(font2)
        self.label_signal.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_32.addWidget(self.label_signal)

        self.value_signal = QComboBox(self.page_plotsNoWidget)
        self.value_signal.setObjectName(u"value_signal")
        self.value_signal.setMinimumSize(QSize(85, 0))
        self.value_signal.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_32.addWidget(self.value_signal)

        self.label_yAxis = QLabel(self.page_plotsNoWidget)
        self.label_yAxis.setObjectName(u"label_yAxis")
        self.label_yAxis.setFont(font2)
        self.label_yAxis.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_32.addWidget(self.label_yAxis)

        self.value_yAxis = QComboBox(self.page_plotsNoWidget)
        self.value_yAxis.setObjectName(u"value_yAxis")
        self.value_yAxis.setMinimumSize(QSize(85, 0))
        self.value_yAxis.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_32.addWidget(self.value_yAxis)

        self.label_timeScale = QLabel(self.page_plotsNoWidget)
        self.label_timeScale.setObjectName(u"label_timeScale")
        self.label_timeScale.setFont(font2)
        self.label_timeScale.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_32.addWidget(self.label_timeScale)

        self.value_timeScale = QComboBox(self.page_plotsNoWidget)
        self.value_timeScale.setObjectName(u"value_timeScale")
        self.value_timeScale.setMinimumSize(QSize(85, 0))
        self.value_timeScale.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_32.addWidget(self.value_timeScale)

        self.label_heartRate = QLabel(self.page_plotsNoWidget)
        self.label_heartRate.setObjectName(u"label_heartRate")
        self.label_heartRate.setFont(font2)
        self.label_heartRate.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_32.addWidget(self.label_heartRate)

        self.value_heartRate = QLabel(self.page_plotsNoWidget)
        self.value_heartRate.setObjectName(u"value_heartRate")
        self.value_heartRate.setMaximumSize(QSize(16777215, 25))
        self.value_heartRate.setStyleSheet(u"border: 1px solid #FFF;\n"
"background-color: rgb(49, 52, 86)")
        self.value_heartRate.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_32.addWidget(self.value_heartRate)

        self.frame_2 = QFrame(self.page_plotsNoWidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_32.addWidget(self.frame_2)

        self.btn_stream = QPushButton(self.page_plotsNoWidget)
        self.btn_stream.setObjectName(u"btn_stream")
        self.btn_stream.setMinimumSize(QSize(150, 30))
        self.btn_stream.setMaximumSize(QSize(16777215, 16777215))
        font4 = QFont()
        font4.setFamilies([u"DM Sans"])
        font4.setPointSize(12)
        font4.setBold(False)
        font4.setItalic(False)
        self.btn_stream.setFont(font4)
        self.btn_stream.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_stream.setStyleSheet(u"QPushButton{\n"
"	border: 1px solid #25541C;\n"
"	border-radius: 5px;\n"
"	background-color: #2B851A;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	border: 2px solid #FFF;\n"
"	border-radius: 5px;\n"
"	background-color: #2B851A;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"	border: 2px solid #25541C;\n"
"	border-radius: 5px;\n"
"	background-color: #4D9033;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"")

        self.horizontalLayout_32.addWidget(self.btn_stream)


        self.verticalLayout_21.addLayout(self.horizontalLayout_32)

        self.horizontalLayout_35 = QHBoxLayout()
        self.horizontalLayout_35.setObjectName(u"horizontalLayout_35")
        self.horizontalLayout_35.setContentsMargins(5, -1, -1, -1)
        self.horizontalLayout_33 = QHBoxLayout()
        self.horizontalLayout_33.setObjectName(u"horizontalLayout_33")
        self.btn_record = QPushButton(self.page_plotsNoWidget)
        self.btn_record.setObjectName(u"btn_record")
        self.btn_record.setMinimumSize(QSize(100, 30))
        self.btn_record.setCursor(QCursor(Qt.PointingHandCursor))
        icon3 = QIcon()
        icon3.addFile(u":/icons/icons/cil-circle.png", QSize(), QIcon.Normal, QIcon.Off)
        icon3.addFile(u":/icons/icons/icon_maximize.png", QSize(), QIcon.Normal, QIcon.On)
        self.btn_record.setIcon(icon3)

        self.horizontalLayout_33.addWidget(self.btn_record)

        self.label_recording_time = QLabel(self.page_plotsNoWidget)
        self.label_recording_time.setObjectName(u"label_recording_time")
        self.label_recording_time.setMinimumSize(QSize(100, 0))
        self.label_recording_time.setMaximumSize(QSize(150, 30))
        self.label_recording_time.setStyleSheet(u"\n"
"border: 1px solid #FFF;\n"
"background-color: rgb(49, 52, 86)")
        self.label_recording_time.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_33.addWidget(self.label_recording_time)


        self.horizontalLayout_35.addLayout(self.horizontalLayout_33)

        self.frame_4 = QFrame(self.page_plotsNoWidget)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_35.addWidget(self.frame_4)

        self.horizontalLayout_34 = QHBoxLayout()
        self.horizontalLayout_34.setObjectName(u"horizontalLayout_34")
        self.verticalLayout_20 = QVBoxLayout()
        self.verticalLayout_20.setSpacing(0)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.value_event_code = QLineEdit(self.page_plotsNoWidget)
        self.value_event_code.setObjectName(u"value_event_code")
        self.value_event_code.setMaximumSize(QSize(75, 30))
        self.value_event_code.setFont(font3)
        self.value_event_code.setStyleSheet(u"font: 11pt ")

        self.verticalLayout_20.addWidget(self.value_event_code)


        self.horizontalLayout_34.addLayout(self.verticalLayout_20)

        self.btn_marker = QPushButton(self.page_plotsNoWidget)
        self.btn_marker.setObjectName(u"btn_marker")
        self.btn_marker.setMinimumSize(QSize(80, 30))
        self.btn_marker.setMaximumSize(QSize(80, 16777215))
        self.btn_marker.setCursor(QCursor(Qt.PointingHandCursor))

        self.horizontalLayout_34.addWidget(self.btn_marker)


        self.horizontalLayout_35.addLayout(self.horizontalLayout_34)

        self.frame_3 = QFrame(self.page_plotsNoWidget)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setMinimumSize(QSize(450, 0))
        self.frame_3.setFrameShape(QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_35.addWidget(self.frame_3)

        self.btn_plot_filters = QPushButton(self.page_plotsNoWidget)
        self.btn_plot_filters.setObjectName(u"btn_plot_filters")
        self.btn_plot_filters.setMinimumSize(QSize(100, 30))
        self.btn_plot_filters.setCursor(QCursor(Qt.PointingHandCursor))
        icon4 = QIcon()
        icon4.addFile(u":/icons/icons/cil-options.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_plot_filters.setIcon(icon4)
        self.btn_plot_filters.setIconSize(QSize(16, 12))

        self.horizontalLayout_35.addWidget(self.btn_plot_filters)


        self.verticalLayout_21.addLayout(self.horizontalLayout_35)

        self.tabWidget = QTabWidget(self.page_plotsNoWidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setStyleSheet(u"")
        self.exg = QWidget()
        self.exg.setObjectName(u"exg")
        self.horizontalLayout_21 = QHBoxLayout(self.exg)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.plot_exg = PlotWidget(self.exg)
        self.plot_exg.setObjectName(u"plot_exg")

        self.horizontalLayout_21.addWidget(self.plot_exg)

        self.tabWidget.addTab(self.exg, "")
        self.orn = QWidget()
        self.orn.setObjectName(u"orn")
        self.horizontalLayout_22 = QHBoxLayout(self.orn)
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.plot_orn = GraphicsLayoutWidget(self.orn)
        self.plot_orn.setObjectName(u"plot_orn")

        self.horizontalLayout_22.addWidget(self.plot_orn)

        self.tabWidget.addTab(self.orn, "")
        self.fft = QWidget()
        self.fft.setObjectName(u"fft")
        self.horizontalLayout_23 = QHBoxLayout(self.fft)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.plot_fft = PlotWidget(self.fft)
        self.plot_fft.setObjectName(u"plot_fft")

        self.horizontalLayout_23.addWidget(self.plot_fft)

        self.tabWidget.addTab(self.fft, "")

        self.verticalLayout_21.addWidget(self.tabWidget)

        self.label_7 = QLabel(self.page_plotsNoWidget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_7.setOpenExternalLinks(True)

        self.verticalLayout_21.addWidget(self.label_7)

        self.stackedWidget.addWidget(self.page_plotsNoWidget)
        self.page_plotsRecorded = QWidget()
        self.page_plotsRecorded.setObjectName(u"page_plotsRecorded")
        self.page_plotsRecorded.setStyleSheet(u"QFrame{\n"
"	border:none;\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"	padding:2px;\n"
"}\n"
"\n"
"QLineEdit{\n"
"	background-color: #53576F;\n"
"	border: 1px solid #FFF;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QComboBox{\n"
"	background-color: #53576F;\n"
"	border: 1px solid #FFF;\n"
"	color: #FFF;\n"
"}\n"
"QComboBox::drop-down{\n"
"	color: #FFF;\n"
"}\n"
"QComboBox QAbstractItemView{\n"
"	color: #FFF;\n"
"}\n"
"QComboBox QListView{\n"
"	color: #FFF;\n"
"}\n"
"QListView{\n"
"	color:#FFF;\n"
"}\n"
"\n"
"QPushButton{\n"
"	border-radius: 1px;\n"
"	border: 1px solid #FFFFFF;\n"
"	background-position: left center;\n"
"    	background-repeat: no-repeat;\n"
"	color: #FFFFFF;\n"
"	text-align: center;\n"
"	padding: 2px\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"\n"
"/*QPushButton{\n"
"	border-radius: 5px;\n"
"	background-position: left center;\n"
"    	background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px"
                        ";\n"
"	color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"	border-left: 20px solid rgb(61, 64, 89);*/\n"
"")
        self.verticalLayout_25 = QVBoxLayout(self.page_plotsRecorded)
#ifndef Q_OS_MAC
        self.verticalLayout_25.setSpacing(-1)
#endif
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.horizontalLayout_36 = QHBoxLayout()
        self.horizontalLayout_36.setObjectName(u"horizontalLayout_36")
        self.label_signal_rec = QLabel(self.page_plotsRecorded)
        self.label_signal_rec.setObjectName(u"label_signal_rec")
        self.label_signal_rec.setFont(font2)
        self.label_signal_rec.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_36.addWidget(self.label_signal_rec)

        self.value_signal_rec = QComboBox(self.page_plotsRecorded)
        self.value_signal_rec.setObjectName(u"value_signal_rec")
        self.value_signal_rec.setMinimumSize(QSize(85, 0))
        self.value_signal_rec.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_36.addWidget(self.value_signal_rec)

        self.label_yAxis_rec = QLabel(self.page_plotsRecorded)
        self.label_yAxis_rec.setObjectName(u"label_yAxis_rec")
        self.label_yAxis_rec.setFont(font2)
        self.label_yAxis_rec.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_36.addWidget(self.label_yAxis_rec)

        self.value_yAxis_rec = QComboBox(self.page_plotsRecorded)
        self.value_yAxis_rec.setObjectName(u"value_yAxis_rec")
        self.value_yAxis_rec.setMinimumSize(QSize(85, 0))
        self.value_yAxis_rec.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_36.addWidget(self.value_yAxis_rec)

        self.label_timeScale_rec = QLabel(self.page_plotsRecorded)
        self.label_timeScale_rec.setObjectName(u"label_timeScale_rec")
        self.label_timeScale_rec.setFont(font2)
        self.label_timeScale_rec.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_36.addWidget(self.label_timeScale_rec)

        self.value_timeScale_rec = QComboBox(self.page_plotsRecorded)
        self.value_timeScale_rec.setObjectName(u"value_timeScale_rec")
        self.value_timeScale_rec.setMinimumSize(QSize(85, 0))
        self.value_timeScale_rec.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout_36.addWidget(self.value_timeScale_rec)

        self.label_heartRate_rec = QLabel(self.page_plotsRecorded)
        self.label_heartRate_rec.setObjectName(u"label_heartRate_rec")
        self.label_heartRate_rec.setFont(font2)
        self.label_heartRate_rec.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_36.addWidget(self.label_heartRate_rec)

        self.value_heartRate_rec = QLabel(self.page_plotsRecorded)
        self.value_heartRate_rec.setObjectName(u"value_heartRate_rec")
        self.value_heartRate_rec.setMaximumSize(QSize(16777215, 25))
        self.value_heartRate_rec.setStyleSheet(u"border: 1px solid #FFF;\n"
"background-color: rgb(49, 52, 86)")
        self.value_heartRate_rec.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_36.addWidget(self.value_heartRate_rec)

        self.btn_stream_rec = QPushButton(self.page_plotsRecorded)
        self.btn_stream_rec.setObjectName(u"btn_stream_rec")
        self.btn_stream_rec.setMinimumSize(QSize(150, 30))
        self.btn_stream_rec.setMaximumSize(QSize(16777215, 16777215))
        self.btn_stream_rec.setFont(font4)
        self.btn_stream_rec.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_stream_rec.setStyleSheet(u"QPushButton{\n"
"	border: 1px solid #25541C;\n"
"	border-radius: 5px;\n"
"	background-color: #2B851A;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	border: 2px solid #FFF;\n"
"	border-radius: 5px;\n"
"	background-color: #2B851A;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"	border: 2px solid #25541C;\n"
"	border-radius: 5px;\n"
"	background-color: #4D9033;\n"
"	color: #FFF;\n"
"}\n"
"\n"
"")

        self.horizontalLayout_36.addWidget(self.btn_stream_rec)


        self.verticalLayout_25.addLayout(self.horizontalLayout_36)

        self.cb_swipping_rec = QCheckBox(self.page_plotsRecorded)
        self.cb_swipping_rec.setObjectName(u"cb_swipping_rec")

        self.verticalLayout_25.addWidget(self.cb_swipping_rec)

        self.tabWidget_rec = QTabWidget(self.page_plotsRecorded)
        self.tabWidget_rec.setObjectName(u"tabWidget_rec")
        self.tabWidget_rec.setStyleSheet(u"")
        self.exg_rec = QWidget()
        self.exg_rec.setObjectName(u"exg_rec")
        self.horizontalLayout_24 = QHBoxLayout(self.exg_rec)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.plot_exg_rec = GraphicsLayoutWidget(self.exg_rec)
        self.plot_exg_rec.setObjectName(u"plot_exg_rec")

        self.horizontalLayout_24.addWidget(self.plot_exg_rec)

        self.tabWidget_rec.addTab(self.exg_rec, "")
        self.orn_rec = QWidget()
        self.orn_rec.setObjectName(u"orn_rec")
        self.horizontalLayout_30 = QHBoxLayout(self.orn_rec)
        self.horizontalLayout_30.setObjectName(u"horizontalLayout_30")
        self.plot_orn_rec = GraphicsLayoutWidget(self.orn_rec)
        self.plot_orn_rec.setObjectName(u"plot_orn_rec")

        self.horizontalLayout_30.addWidget(self.plot_orn_rec)

        self.tabWidget_rec.addTab(self.orn_rec, "")
        self.fft_rec = QWidget()
        self.fft_rec.setObjectName(u"fft_rec")
        self.horizontalLayout_40 = QHBoxLayout(self.fft_rec)
        self.horizontalLayout_40.setObjectName(u"horizontalLayout_40")
        self.plot_fft_rec = PlotWidget(self.fft_rec)
        self.plot_fft_rec.setObjectName(u"plot_fft_rec")

        self.horizontalLayout_40.addWidget(self.plot_fft_rec)

        self.tabWidget_rec.addTab(self.fft_rec, "")

        self.verticalLayout_25.addWidget(self.tabWidget_rec)

        self.label_3 = QLabel(self.page_plotsRecorded)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font2)
        self.label_3.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_3.setOpenExternalLinks(True)

        self.verticalLayout_25.addWidget(self.label_3)

        self.stackedWidget.addWidget(self.page_plotsRecorded)
        self.page_impedance = QWidget()
        self.page_impedance.setObjectName(u"page_impedance")
        self.verticalLayout_22 = QVBoxLayout(self.page_impedance)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.frame_impedance_title = QFrame(self.page_impedance)
        self.frame_impedance_title.setObjectName(u"frame_impedance_title")
        self.frame_impedance_title.setMaximumSize(QSize(16777215, 50))
        self.frame_impedance_title.setStyleSheet(u"QLabel{\n"
"	color: #FFF;\n"
"	border:none;\n"
"}\n"
"\n"
"QFrame{\n"
"	border-bottom: 1px solid #FFF;\n"
"	border-top: none;\n"
"	border-left: none;\n"
"	border-right: none;\n"
"}\n"
"")
        self.frame_impedance_title.setFrameShape(QFrame.StyledPanel)
        self.frame_impedance_title.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_17 = QHBoxLayout(self.frame_impedance_title)
        self.horizontalLayout_17.setSpacing(0)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.label_4 = QLabel(self.frame_impedance_title)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(51, 51))
        self.label_4.setPixmap(QPixmap(u":/image/images/MentalabLogo.png"))
        self.label_4.setScaledContents(True)

        self.horizontalLayout_17.addWidget(self.label_4)

        self.impedance_title = QLabel(self.frame_impedance_title)
        self.impedance_title.setObjectName(u"impedance_title")
        self.impedance_title.setMinimumSize(QSize(0, 0))
        self.impedance_title.setFont(font)
        self.impedance_title.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_17.addWidget(self.impedance_title)


        self.verticalLayout_22.addWidget(self.frame_impedance_title)

        self.frame_impedance = QFrame(self.page_impedance)
        self.frame_impedance.setObjectName(u"frame_impedance")
        self.frame_impedance.setStyleSheet(u"QPushButton{\n"
"	color: #FFF;\n"
"	background-color: rgb(84, 89, 124);\n"
"	border: 2px solid rgb(84, 89, 124);\n"
"	padding: 5px;\n"
"	border-radius: 5px;\n"
"\n"
"}\n"
"\n"
"QPushButton:hover{\n"
"	background-color: rgb(61, 64, 89);\n"
"}\n"
"QPushButton:pressed{\n"
"	background-color: rgb(61, 64, 89);\n"
"	border:  2px solid rgb(61, 64, 89);\n"
"}")
        self.frame_impedance.setFrameShape(QFrame.StyledPanel)
        self.frame_impedance.setFrameShadow(QFrame.Raised)
        self.verticalLayout_35 = QVBoxLayout(self.frame_impedance)
        self.verticalLayout_35.setObjectName(u"verticalLayout_35")
        self.frame_impedance_widgets = QFrame(self.frame_impedance)
        self.frame_impedance_widgets.setObjectName(u"frame_impedance_widgets")
        self.frame_impedance_widgets.setMaximumSize(QSize(16777215, 200))
        self.frame_impedance_widgets.setStyleSheet(u"border: none;")
        self.frame_impedance_widgets.setFrameShape(QFrame.StyledPanel)
        self.frame_impedance_widgets.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_18 = QHBoxLayout(self.frame_impedance_widgets)
        self.horizontalLayout_18.setSpacing(5)
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.frame_ch1 = QFrame(self.frame_impedance_widgets)
        self.frame_ch1.setObjectName(u"frame_ch1")
        self.frame_ch1.setMaximumSize(QSize(80, 150))
        self.frame_ch1.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch1.setFrameShape(QFrame.StyledPanel)
        self.frame_ch1.setFrameShadow(QFrame.Raised)
        self.verticalLayout_7 = QVBoxLayout(self.frame_ch1)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.frame_ch1_color = QFrame(self.frame_ch1)
        self.frame_ch1_color.setObjectName(u"frame_ch1_color")
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.frame_ch1_color.sizePolicy().hasHeightForWidth())
        self.frame_ch1_color.setSizePolicy(sizePolicy4)
        self.frame_ch1_color.setMinimumSize(QSize(61, 61))
        self.frame_ch1_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch1_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch1_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169);")
        self.frame_ch1_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch1_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_7.addWidget(self.frame_ch1_color)

        self.label_ch1 = QLabel(self.frame_ch1)
        self.label_ch1.setObjectName(u"label_ch1")
        self.label_ch1.setMinimumSize(QSize(61, 0))
        self.label_ch1.setMaximumSize(QSize(16777215, 30))
        self.label_ch1.setFont(font2)
        self.label_ch1.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_ch1)

        self.label_ch1_value = QLabel(self.frame_ch1)
        self.label_ch1_value.setObjectName(u"label_ch1_value")
        self.label_ch1_value.setMinimumSize(QSize(61, 0))
        self.label_ch1_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch1_value.setFont(font4)
        self.label_ch1_value.setStyleSheet(u"font:12pt")
        self.label_ch1_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_7.addWidget(self.label_ch1_value)


        self.horizontalLayout_18.addWidget(self.frame_ch1)

        self.frame_ch2 = QFrame(self.frame_impedance_widgets)
        self.frame_ch2.setObjectName(u"frame_ch2")
        self.frame_ch2.setMaximumSize(QSize(80, 150))
        self.frame_ch2.setStyleSheet(u"font:12px")
        self.frame_ch2.setFrameShape(QFrame.StyledPanel)
        self.frame_ch2.setFrameShadow(QFrame.Raised)
        self.verticalLayout_8 = QVBoxLayout(self.frame_ch2)
        self.verticalLayout_8.setSpacing(0)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.frame_ch2_color = QFrame(self.frame_ch2)
        self.frame_ch2_color.setObjectName(u"frame_ch2_color")
        sizePolicy4.setHeightForWidth(self.frame_ch2_color.sizePolicy().hasHeightForWidth())
        self.frame_ch2_color.setSizePolicy(sizePolicy4)
        self.frame_ch2_color.setMinimumSize(QSize(61, 61))
        self.frame_ch2_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch2_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch2_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch2_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch2_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_8.addWidget(self.frame_ch2_color)

        self.label_ch2 = QLabel(self.frame_ch2)
        self.label_ch2.setObjectName(u"label_ch2")
        self.label_ch2.setMinimumSize(QSize(61, 0))
        self.label_ch2.setMaximumSize(QSize(16777215, 30))
        font5 = QFont()
        font5.setFamilies([u"DM Sans"])
        font5.setBold(False)
        font5.setItalic(False)
        self.label_ch2.setFont(font5)
        self.label_ch2.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.label_ch2)

        self.label_ch2_value = QLabel(self.frame_ch2)
        self.label_ch2_value.setObjectName(u"label_ch2_value")
        self.label_ch2_value.setMinimumSize(QSize(61, 0))
        self.label_ch2_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch2_value.setFont(font4)
        self.label_ch2_value.setStyleSheet(u"font:12pt")
        self.label_ch2_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_8.addWidget(self.label_ch2_value)


        self.horizontalLayout_18.addWidget(self.frame_ch2)

        self.frame_ch3 = QFrame(self.frame_impedance_widgets)
        self.frame_ch3.setObjectName(u"frame_ch3")
        self.frame_ch3.setMaximumSize(QSize(80, 150))
        self.frame_ch3.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch3.setFrameShape(QFrame.StyledPanel)
        self.frame_ch3.setFrameShadow(QFrame.Raised)
        self.verticalLayout_9 = QVBoxLayout(self.frame_ch3)
        self.verticalLayout_9.setSpacing(0)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.frame_ch3_color = QFrame(self.frame_ch3)
        self.frame_ch3_color.setObjectName(u"frame_ch3_color")
        sizePolicy4.setHeightForWidth(self.frame_ch3_color.sizePolicy().hasHeightForWidth())
        self.frame_ch3_color.setSizePolicy(sizePolicy4)
        self.frame_ch3_color.setMinimumSize(QSize(61, 61))
        self.frame_ch3_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch3_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch3_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch3_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch3_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_9.addWidget(self.frame_ch3_color)

        self.label_ch3 = QLabel(self.frame_ch3)
        self.label_ch3.setObjectName(u"label_ch3")
        self.label_ch3.setMinimumSize(QSize(61, 0))
        self.label_ch3.setMaximumSize(QSize(16777215, 30))
        self.label_ch3.setFont(font2)
        self.label_ch3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.label_ch3)

        self.label_ch3_value = QLabel(self.frame_ch3)
        self.label_ch3_value.setObjectName(u"label_ch3_value")
        self.label_ch3_value.setMinimumSize(QSize(61, 0))
        self.label_ch3_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch3_value.setFont(font4)
        self.label_ch3_value.setStyleSheet(u"font:12pt")
        self.label_ch3_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_9.addWidget(self.label_ch3_value)


        self.horizontalLayout_18.addWidget(self.frame_ch3)

        self.frame_ch4 = QFrame(self.frame_impedance_widgets)
        self.frame_ch4.setObjectName(u"frame_ch4")
        self.frame_ch4.setMaximumSize(QSize(80, 150))
        self.frame_ch4.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch4.setFrameShape(QFrame.StyledPanel)
        self.frame_ch4.setFrameShadow(QFrame.Raised)
        self.verticalLayout_10 = QVBoxLayout(self.frame_ch4)
        self.verticalLayout_10.setSpacing(0)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.frame_ch4_color = QFrame(self.frame_ch4)
        self.frame_ch4_color.setObjectName(u"frame_ch4_color")
        sizePolicy4.setHeightForWidth(self.frame_ch4_color.sizePolicy().hasHeightForWidth())
        self.frame_ch4_color.setSizePolicy(sizePolicy4)
        self.frame_ch4_color.setMinimumSize(QSize(61, 61))
        self.frame_ch4_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch4_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch4_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch4_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch4_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_10.addWidget(self.frame_ch4_color)

        self.label_ch4 = QLabel(self.frame_ch4)
        self.label_ch4.setObjectName(u"label_ch4")
        self.label_ch4.setMinimumSize(QSize(61, 0))
        self.label_ch4.setMaximumSize(QSize(16777215, 30))
        self.label_ch4.setFont(font2)
        self.label_ch4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_10.addWidget(self.label_ch4)

        self.label_ch4_value = QLabel(self.frame_ch4)
        self.label_ch4_value.setObjectName(u"label_ch4_value")
        self.label_ch4_value.setMinimumSize(QSize(61, 0))
        self.label_ch4_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch4_value.setFont(font4)
        self.label_ch4_value.setStyleSheet(u"font:12pt")
        self.label_ch4_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_10.addWidget(self.label_ch4_value)


        self.horizontalLayout_18.addWidget(self.frame_ch4)

        self.frame_ch5 = QFrame(self.frame_impedance_widgets)
        self.frame_ch5.setObjectName(u"frame_ch5")
        self.frame_ch5.setMaximumSize(QSize(80, 150))
        self.frame_ch5.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch5.setFrameShape(QFrame.StyledPanel)
        self.frame_ch5.setFrameShadow(QFrame.Raised)
        self.verticalLayout_11 = QVBoxLayout(self.frame_ch5)
        self.verticalLayout_11.setSpacing(0)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.frame_ch5_color = QFrame(self.frame_ch5)
        self.frame_ch5_color.setObjectName(u"frame_ch5_color")
        sizePolicy4.setHeightForWidth(self.frame_ch5_color.sizePolicy().hasHeightForWidth())
        self.frame_ch5_color.setSizePolicy(sizePolicy4)
        self.frame_ch5_color.setMinimumSize(QSize(61, 61))
        self.frame_ch5_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch5_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch5_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch5_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch5_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_11.addWidget(self.frame_ch5_color)

        self.label_ch5 = QLabel(self.frame_ch5)
        self.label_ch5.setObjectName(u"label_ch5")
        self.label_ch5.setMinimumSize(QSize(61, 0))
        self.label_ch5.setMaximumSize(QSize(16777215, 30))
        self.label_ch5.setFont(font2)
        self.label_ch5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_11.addWidget(self.label_ch5)

        self.label_ch5_value = QLabel(self.frame_ch5)
        self.label_ch5_value.setObjectName(u"label_ch5_value")
        self.label_ch5_value.setMinimumSize(QSize(61, 0))
        self.label_ch5_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch5_value.setFont(font4)
        self.label_ch5_value.setStyleSheet(u"font:12pt")
        self.label_ch5_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_11.addWidget(self.label_ch5_value)


        self.horizontalLayout_18.addWidget(self.frame_ch5)

        self.frame_ch6 = QFrame(self.frame_impedance_widgets)
        self.frame_ch6.setObjectName(u"frame_ch6")
        self.frame_ch6.setMaximumSize(QSize(80, 150))
        self.frame_ch6.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch6.setFrameShape(QFrame.StyledPanel)
        self.frame_ch6.setFrameShadow(QFrame.Raised)
        self.verticalLayout_13 = QVBoxLayout(self.frame_ch6)
        self.verticalLayout_13.setSpacing(0)
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.frame_ch6_color = QFrame(self.frame_ch6)
        self.frame_ch6_color.setObjectName(u"frame_ch6_color")
        sizePolicy4.setHeightForWidth(self.frame_ch6_color.sizePolicy().hasHeightForWidth())
        self.frame_ch6_color.setSizePolicy(sizePolicy4)
        self.frame_ch6_color.setMinimumSize(QSize(61, 61))
        self.frame_ch6_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch6_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch6_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch6_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch6_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_13.addWidget(self.frame_ch6_color)

        self.label_ch6 = QLabel(self.frame_ch6)
        self.label_ch6.setObjectName(u"label_ch6")
        self.label_ch6.setMinimumSize(QSize(61, 0))
        self.label_ch6.setMaximumSize(QSize(16777215, 30))
        self.label_ch6.setFont(font2)
        self.label_ch6.setAlignment(Qt.AlignCenter)

        self.verticalLayout_13.addWidget(self.label_ch6)

        self.label_ch6_value = QLabel(self.frame_ch6)
        self.label_ch6_value.setObjectName(u"label_ch6_value")
        self.label_ch6_value.setMinimumSize(QSize(61, 0))
        self.label_ch6_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch6_value.setFont(font4)
        self.label_ch6_value.setStyleSheet(u"font:12pt")
        self.label_ch6_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_13.addWidget(self.label_ch6_value)


        self.horizontalLayout_18.addWidget(self.frame_ch6)

        self.frame_ch7 = QFrame(self.frame_impedance_widgets)
        self.frame_ch7.setObjectName(u"frame_ch7")
        self.frame_ch7.setMaximumSize(QSize(80, 150))
        self.frame_ch7.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch7.setFrameShape(QFrame.StyledPanel)
        self.frame_ch7.setFrameShadow(QFrame.Raised)
        self.verticalLayout_14 = QVBoxLayout(self.frame_ch7)
        self.verticalLayout_14.setSpacing(0)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.frame_ch7_color = QFrame(self.frame_ch7)
        self.frame_ch7_color.setObjectName(u"frame_ch7_color")
        sizePolicy4.setHeightForWidth(self.frame_ch7_color.sizePolicy().hasHeightForWidth())
        self.frame_ch7_color.setSizePolicy(sizePolicy4)
        self.frame_ch7_color.setMinimumSize(QSize(61, 61))
        self.frame_ch7_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch7_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch7_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch7_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch7_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_14.addWidget(self.frame_ch7_color)

        self.label_ch7 = QLabel(self.frame_ch7)
        self.label_ch7.setObjectName(u"label_ch7")
        self.label_ch7.setMinimumSize(QSize(61, 0))
        self.label_ch7.setMaximumSize(QSize(16777215, 30))
        self.label_ch7.setFont(font2)
        self.label_ch7.setAlignment(Qt.AlignCenter)

        self.verticalLayout_14.addWidget(self.label_ch7)

        self.label_ch7_value = QLabel(self.frame_ch7)
        self.label_ch7_value.setObjectName(u"label_ch7_value")
        self.label_ch7_value.setMinimumSize(QSize(61, 0))
        self.label_ch7_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch7_value.setFont(font4)
        self.label_ch7_value.setStyleSheet(u"font:12pt")
        self.label_ch7_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_14.addWidget(self.label_ch7_value)


        self.horizontalLayout_18.addWidget(self.frame_ch7)

        self.frame_ch8 = QFrame(self.frame_impedance_widgets)
        self.frame_ch8.setObjectName(u"frame_ch8")
        self.frame_ch8.setMaximumSize(QSize(80, 150))
        self.frame_ch8.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch8.setFrameShape(QFrame.StyledPanel)
        self.frame_ch8.setFrameShadow(QFrame.Raised)
        self.verticalLayout_15 = QVBoxLayout(self.frame_ch8)
        self.verticalLayout_15.setSpacing(0)
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.frame_ch8_color = QFrame(self.frame_ch8)
        self.frame_ch8_color.setObjectName(u"frame_ch8_color")
        sizePolicy4.setHeightForWidth(self.frame_ch8_color.sizePolicy().hasHeightForWidth())
        self.frame_ch8_color.setSizePolicy(sizePolicy4)
        self.frame_ch8_color.setMinimumSize(QSize(61, 61))
        self.frame_ch8_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch8_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch8_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch8_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch8_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_15.addWidget(self.frame_ch8_color)

        self.label_ch8 = QLabel(self.frame_ch8)
        self.label_ch8.setObjectName(u"label_ch8")
        self.label_ch8.setMinimumSize(QSize(61, 0))
        self.label_ch8.setMaximumSize(QSize(16777215, 30))
        self.label_ch8.setFont(font2)
        self.label_ch8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_15.addWidget(self.label_ch8)

        self.label_ch8_value = QLabel(self.frame_ch8)
        self.label_ch8_value.setObjectName(u"label_ch8_value")
        self.label_ch8_value.setMinimumSize(QSize(61, 0))
        self.label_ch8_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch8_value.setFont(font4)
        self.label_ch8_value.setStyleSheet(u"font:12pt")
        self.label_ch8_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_15.addWidget(self.label_ch8_value)


        self.horizontalLayout_18.addWidget(self.frame_ch8)


        self.verticalLayout_35.addWidget(self.frame_impedance_widgets)

        self.frame_impedance_widgets_16 = QFrame(self.frame_impedance)
        self.frame_impedance_widgets_16.setObjectName(u"frame_impedance_widgets_16")
        self.frame_impedance_widgets_16.setMaximumSize(QSize(16777215, 200))
        self.frame_impedance_widgets_16.setStyleSheet(u"font:12px")
        self.frame_impedance_widgets_16.setFrameShape(QFrame.StyledPanel)
        self.frame_impedance_widgets_16.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_20 = QHBoxLayout(self.frame_impedance_widgets_16)
        self.horizontalLayout_20.setSpacing(5)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.frame_ch9 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch9.setObjectName(u"frame_ch9")
        self.frame_ch9.setMaximumSize(QSize(80, 150))
        self.frame_ch9.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch9.setFrameShape(QFrame.StyledPanel)
        self.frame_ch9.setFrameShadow(QFrame.Raised)
        self.verticalLayout_16 = QVBoxLayout(self.frame_ch9)
        self.verticalLayout_16.setSpacing(0)
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.frame_ch9_color = QFrame(self.frame_ch9)
        self.frame_ch9_color.setObjectName(u"frame_ch9_color")
        sizePolicy4.setHeightForWidth(self.frame_ch9_color.sizePolicy().hasHeightForWidth())
        self.frame_ch9_color.setSizePolicy(sizePolicy4)
        self.frame_ch9_color.setMinimumSize(QSize(61, 61))
        self.frame_ch9_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch9_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch9_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169);")
        self.frame_ch9_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch9_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_16.addWidget(self.frame_ch9_color)

        self.label_ch9 = QLabel(self.frame_ch9)
        self.label_ch9.setObjectName(u"label_ch9")
        self.label_ch9.setMinimumSize(QSize(61, 0))
        self.label_ch9.setMaximumSize(QSize(16777215, 30))
        self.label_ch9.setFont(font5)
        self.label_ch9.setAlignment(Qt.AlignCenter)

        self.verticalLayout_16.addWidget(self.label_ch9)

        self.label_ch9_value = QLabel(self.frame_ch9)
        self.label_ch9_value.setObjectName(u"label_ch9_value")
        self.label_ch9_value.setMinimumSize(QSize(61, 0))
        self.label_ch9_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch9_value.setFont(font4)
        self.label_ch9_value.setStyleSheet(u"font:12pt")
        self.label_ch9_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_16.addWidget(self.label_ch9_value)


        self.horizontalLayout_20.addWidget(self.frame_ch9)

        self.frame_ch10 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch10.setObjectName(u"frame_ch10")
        self.frame_ch10.setMaximumSize(QSize(80, 150))
        self.frame_ch10.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch10.setFrameShape(QFrame.StyledPanel)
        self.frame_ch10.setFrameShadow(QFrame.Raised)
        self.verticalLayout_26 = QVBoxLayout(self.frame_ch10)
        self.verticalLayout_26.setSpacing(0)
        self.verticalLayout_26.setObjectName(u"verticalLayout_26")
        self.frame_ch10_color = QFrame(self.frame_ch10)
        self.frame_ch10_color.setObjectName(u"frame_ch10_color")
        sizePolicy4.setHeightForWidth(self.frame_ch10_color.sizePolicy().hasHeightForWidth())
        self.frame_ch10_color.setSizePolicy(sizePolicy4)
        self.frame_ch10_color.setMinimumSize(QSize(61, 61))
        self.frame_ch10_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch10_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch10_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch10_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch10_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_26.addWidget(self.frame_ch10_color)

        self.label_ch10 = QLabel(self.frame_ch10)
        self.label_ch10.setObjectName(u"label_ch10")
        self.label_ch10.setMinimumSize(QSize(61, 0))
        self.label_ch10.setMaximumSize(QSize(16777215, 30))
        self.label_ch10.setFont(font5)
        self.label_ch10.setAlignment(Qt.AlignCenter)

        self.verticalLayout_26.addWidget(self.label_ch10)

        self.label_ch10_value = QLabel(self.frame_ch10)
        self.label_ch10_value.setObjectName(u"label_ch10_value")
        self.label_ch10_value.setMinimumSize(QSize(61, 0))
        self.label_ch10_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch10_value.setFont(font4)
        self.label_ch10_value.setStyleSheet(u"font:12pt")
        self.label_ch10_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_26.addWidget(self.label_ch10_value)


        self.horizontalLayout_20.addWidget(self.frame_ch10)

        self.frame_ch11 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch11.setObjectName(u"frame_ch11")
        self.frame_ch11.setMaximumSize(QSize(80, 150))
        self.frame_ch11.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch11.setFrameShape(QFrame.StyledPanel)
        self.frame_ch11.setFrameShadow(QFrame.Raised)
        self.verticalLayout_27 = QVBoxLayout(self.frame_ch11)
        self.verticalLayout_27.setSpacing(0)
        self.verticalLayout_27.setObjectName(u"verticalLayout_27")
        self.frame_ch11_color = QFrame(self.frame_ch11)
        self.frame_ch11_color.setObjectName(u"frame_ch11_color")
        sizePolicy4.setHeightForWidth(self.frame_ch11_color.sizePolicy().hasHeightForWidth())
        self.frame_ch11_color.setSizePolicy(sizePolicy4)
        self.frame_ch11_color.setMinimumSize(QSize(61, 61))
        self.frame_ch11_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch11_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch11_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch11_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch11_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_27.addWidget(self.frame_ch11_color)

        self.label_ch11 = QLabel(self.frame_ch11)
        self.label_ch11.setObjectName(u"label_ch11")
        self.label_ch11.setMinimumSize(QSize(61, 0))
        self.label_ch11.setMaximumSize(QSize(16777215, 30))
        self.label_ch11.setFont(font5)
        self.label_ch11.setAlignment(Qt.AlignCenter)

        self.verticalLayout_27.addWidget(self.label_ch11)

        self.label_ch11_value = QLabel(self.frame_ch11)
        self.label_ch11_value.setObjectName(u"label_ch11_value")
        self.label_ch11_value.setMinimumSize(QSize(61, 0))
        self.label_ch11_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch11_value.setFont(font4)
        self.label_ch11_value.setStyleSheet(u"font:12pt")
        self.label_ch11_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_27.addWidget(self.label_ch11_value)


        self.horizontalLayout_20.addWidget(self.frame_ch11)

        self.frame_ch12 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch12.setObjectName(u"frame_ch12")
        self.frame_ch12.setMaximumSize(QSize(80, 150))
        self.frame_ch12.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch12.setFrameShape(QFrame.StyledPanel)
        self.frame_ch12.setFrameShadow(QFrame.Raised)
        self.verticalLayout_28 = QVBoxLayout(self.frame_ch12)
        self.verticalLayout_28.setSpacing(0)
        self.verticalLayout_28.setObjectName(u"verticalLayout_28")
        self.frame_ch12_color = QFrame(self.frame_ch12)
        self.frame_ch12_color.setObjectName(u"frame_ch12_color")
        sizePolicy4.setHeightForWidth(self.frame_ch12_color.sizePolicy().hasHeightForWidth())
        self.frame_ch12_color.setSizePolicy(sizePolicy4)
        self.frame_ch12_color.setMinimumSize(QSize(61, 61))
        self.frame_ch12_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch12_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch12_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch12_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch12_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_28.addWidget(self.frame_ch12_color)

        self.label_ch12 = QLabel(self.frame_ch12)
        self.label_ch12.setObjectName(u"label_ch12")
        self.label_ch12.setMinimumSize(QSize(61, 0))
        self.label_ch12.setMaximumSize(QSize(16777215, 30))
        self.label_ch12.setFont(font5)
        self.label_ch12.setAlignment(Qt.AlignCenter)

        self.verticalLayout_28.addWidget(self.label_ch12)

        self.label_ch12_value = QLabel(self.frame_ch12)
        self.label_ch12_value.setObjectName(u"label_ch12_value")
        self.label_ch12_value.setMinimumSize(QSize(61, 0))
        self.label_ch12_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch12_value.setFont(font4)
        self.label_ch12_value.setStyleSheet(u"font:12pt")
        self.label_ch12_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_28.addWidget(self.label_ch12_value)


        self.horizontalLayout_20.addWidget(self.frame_ch12)

        self.frame_ch13 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch13.setObjectName(u"frame_ch13")
        self.frame_ch13.setMaximumSize(QSize(80, 150))
        self.frame_ch13.setStyleSheet(u"font:12px")
        self.frame_ch13.setFrameShape(QFrame.StyledPanel)
        self.frame_ch13.setFrameShadow(QFrame.Raised)
        self.verticalLayout_31 = QVBoxLayout(self.frame_ch13)
        self.verticalLayout_31.setSpacing(0)
        self.verticalLayout_31.setObjectName(u"verticalLayout_31")
        self.frame_ch13_color = QFrame(self.frame_ch13)
        self.frame_ch13_color.setObjectName(u"frame_ch13_color")
        sizePolicy4.setHeightForWidth(self.frame_ch13_color.sizePolicy().hasHeightForWidth())
        self.frame_ch13_color.setSizePolicy(sizePolicy4)
        self.frame_ch13_color.setMinimumSize(QSize(61, 61))
        self.frame_ch13_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch13_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch13_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch13_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch13_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_31.addWidget(self.frame_ch13_color)

        self.label_ch13 = QLabel(self.frame_ch13)
        self.label_ch13.setObjectName(u"label_ch13")
        self.label_ch13.setMinimumSize(QSize(61, 0))
        self.label_ch13.setMaximumSize(QSize(16777215, 30))
        self.label_ch13.setFont(font5)
        self.label_ch13.setAlignment(Qt.AlignCenter)

        self.verticalLayout_31.addWidget(self.label_ch13)

        self.label_ch13_value = QLabel(self.frame_ch13)
        self.label_ch13_value.setObjectName(u"label_ch13_value")
        self.label_ch13_value.setMinimumSize(QSize(61, 0))
        self.label_ch13_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch13_value.setFont(font4)
        self.label_ch13_value.setStyleSheet(u"font:12pt")
        self.label_ch13_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_31.addWidget(self.label_ch13_value)


        self.horizontalLayout_20.addWidget(self.frame_ch13)

        self.frame_ch14 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch14.setObjectName(u"frame_ch14")
        self.frame_ch14.setMaximumSize(QSize(80, 150))
        self.frame_ch14.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch14.setFrameShape(QFrame.StyledPanel)
        self.frame_ch14.setFrameShadow(QFrame.Raised)
        self.verticalLayout_32 = QVBoxLayout(self.frame_ch14)
        self.verticalLayout_32.setSpacing(0)
        self.verticalLayout_32.setObjectName(u"verticalLayout_32")
        self.frame_ch14_color = QFrame(self.frame_ch14)
        self.frame_ch14_color.setObjectName(u"frame_ch14_color")
        sizePolicy4.setHeightForWidth(self.frame_ch14_color.sizePolicy().hasHeightForWidth())
        self.frame_ch14_color.setSizePolicy(sizePolicy4)
        self.frame_ch14_color.setMinimumSize(QSize(61, 61))
        self.frame_ch14_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch14_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch14_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch14_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch14_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_32.addWidget(self.frame_ch14_color)

        self.label_ch14 = QLabel(self.frame_ch14)
        self.label_ch14.setObjectName(u"label_ch14")
        self.label_ch14.setMinimumSize(QSize(61, 0))
        self.label_ch14.setMaximumSize(QSize(16777215, 30))
        self.label_ch14.setFont(font5)
        self.label_ch14.setAlignment(Qt.AlignCenter)

        self.verticalLayout_32.addWidget(self.label_ch14)

        self.label_ch14_value = QLabel(self.frame_ch14)
        self.label_ch14_value.setObjectName(u"label_ch14_value")
        self.label_ch14_value.setMinimumSize(QSize(61, 0))
        self.label_ch14_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch14_value.setFont(font4)
        self.label_ch14_value.setStyleSheet(u"font:12pt")
        self.label_ch14_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_32.addWidget(self.label_ch14_value)


        self.horizontalLayout_20.addWidget(self.frame_ch14)

        self.frame_ch15 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch15.setObjectName(u"frame_ch15")
        self.frame_ch15.setMaximumSize(QSize(80, 150))
        self.frame_ch15.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch15.setFrameShape(QFrame.StyledPanel)
        self.frame_ch15.setFrameShadow(QFrame.Raised)
        self.verticalLayout_33 = QVBoxLayout(self.frame_ch15)
        self.verticalLayout_33.setSpacing(0)
        self.verticalLayout_33.setObjectName(u"verticalLayout_33")
        self.frame_ch15_color = QFrame(self.frame_ch15)
        self.frame_ch15_color.setObjectName(u"frame_ch15_color")
        sizePolicy4.setHeightForWidth(self.frame_ch15_color.sizePolicy().hasHeightForWidth())
        self.frame_ch15_color.setSizePolicy(sizePolicy4)
        self.frame_ch15_color.setMinimumSize(QSize(61, 61))
        self.frame_ch15_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch15_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch15_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch15_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch15_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_33.addWidget(self.frame_ch15_color)

        self.label_ch15 = QLabel(self.frame_ch15)
        self.label_ch15.setObjectName(u"label_ch15")
        self.label_ch15.setMinimumSize(QSize(61, 0))
        self.label_ch15.setMaximumSize(QSize(16777215, 30))
        self.label_ch15.setFont(font5)
        self.label_ch15.setAlignment(Qt.AlignCenter)

        self.verticalLayout_33.addWidget(self.label_ch15)

        self.label_ch15_value = QLabel(self.frame_ch15)
        self.label_ch15_value.setObjectName(u"label_ch15_value")
        self.label_ch15_value.setMinimumSize(QSize(61, 0))
        self.label_ch15_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch15_value.setFont(font4)
        self.label_ch15_value.setStyleSheet(u"font:12pt")
        self.label_ch15_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_33.addWidget(self.label_ch15_value)


        self.horizontalLayout_20.addWidget(self.frame_ch15)

        self.frame_ch16 = QFrame(self.frame_impedance_widgets_16)
        self.frame_ch16.setObjectName(u"frame_ch16")
        self.frame_ch16.setMaximumSize(QSize(80, 150))
        self.frame_ch16.setStyleSheet(u"QFrame{\n"
"	\n"
"	border-color: rgb(255, 98, 92);\n"
"}\n"
"\n"
"QLabel{\n"
"	color: #FFF;\n"
"}")
        self.frame_ch16.setFrameShape(QFrame.StyledPanel)
        self.frame_ch16.setFrameShadow(QFrame.Raised)
        self.verticalLayout_34 = QVBoxLayout(self.frame_ch16)
        self.verticalLayout_34.setSpacing(0)
        self.verticalLayout_34.setObjectName(u"verticalLayout_34")
        self.frame_ch16_color = QFrame(self.frame_ch16)
        self.frame_ch16_color.setObjectName(u"frame_ch16_color")
        sizePolicy4.setHeightForWidth(self.frame_ch16_color.sizePolicy().hasHeightForWidth())
        self.frame_ch16_color.setSizePolicy(sizePolicy4)
        self.frame_ch16_color.setMinimumSize(QSize(61, 61))
        self.frame_ch16_color.setMaximumSize(QSize(1000, 61))
        self.frame_ch16_color.setSizeIncrement(QSize(1, 1))
        self.frame_ch16_color.setStyleSheet(u"border: 2px solid rgb(145, 145, 145);\n"
"border-radius: 30px;\n"
"background-color: rgb(169, 169, 169)")
        self.frame_ch16_color.setFrameShape(QFrame.StyledPanel)
        self.frame_ch16_color.setFrameShadow(QFrame.Raised)

        self.verticalLayout_34.addWidget(self.frame_ch16_color)

        self.label_ch16 = QLabel(self.frame_ch16)
        self.label_ch16.setObjectName(u"label_ch16")
        self.label_ch16.setMinimumSize(QSize(61, 0))
        self.label_ch16.setMaximumSize(QSize(16777215, 30))
        self.label_ch16.setFont(font5)
        self.label_ch16.setAlignment(Qt.AlignCenter)

        self.verticalLayout_34.addWidget(self.label_ch16)

        self.label_ch16_value = QLabel(self.frame_ch16)
        self.label_ch16_value.setObjectName(u"label_ch16_value")
        self.label_ch16_value.setMinimumSize(QSize(61, 0))
        self.label_ch16_value.setMaximumSize(QSize(16777215, 30))
        self.label_ch16_value.setFont(font4)
        self.label_ch16_value.setStyleSheet(u"font:12pt")
        self.label_ch16_value.setAlignment(Qt.AlignCenter)

        self.verticalLayout_34.addWidget(self.label_ch16_value)


        self.horizontalLayout_20.addWidget(self.frame_ch16)


        self.verticalLayout_35.addWidget(self.frame_impedance_widgets_16)

        self.imp_mode = QComboBox(self.frame_impedance)
        self.imp_mode.setObjectName(u"imp_mode")
        self.imp_mode.setMinimumSize(QSize(200, 0))
        self.imp_mode.setMaximumSize(QSize(200, 16777215))
        self.imp_mode.setStyleSheet(u"color: #FFF;\n"
"border: 1px solid #FFF")

        self.verticalLayout_35.addWidget(self.imp_mode, 0, Qt.AlignHCenter)

        self.frame_ = QFrame(self.frame_impedance)
        self.frame_.setObjectName(u"frame_")
        self.frame_.setMaximumSize(QSize(16777215, 50))
        self.horizontalLayout_11 = QHBoxLayout(self.frame_)
        self.horizontalLayout_11.setSpacing(5)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.btn_imp_meas = QPushButton(self.frame_)
        self.btn_imp_meas.setObjectName(u"btn_imp_meas")
        sizePolicy4.setHeightForWidth(self.btn_imp_meas.sizePolicy().hasHeightForWidth())
        self.btn_imp_meas.setSizePolicy(sizePolicy4)
        self.btn_imp_meas.setMinimumSize(QSize(140, 30))
        self.btn_imp_meas.setMaximumSize(QSize(200, 16777215))
        self.btn_imp_meas.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_imp_meas.setToolTipDuration(-1)

        self.horizontalLayout_11.addWidget(self.btn_imp_meas)

        self.imp_meas_info = QLabel(self.frame_)
        self.imp_meas_info.setObjectName(u"imp_meas_info")
        self.imp_meas_info.setMinimumSize(QSize(20, 20))
        self.imp_meas_info.setMaximumSize(QSize(20, 20))
        self.imp_meas_info.setCursor(QCursor(Qt.ArrowCursor))
        self.imp_meas_info.setStyleSheet(u"")
        self.imp_meas_info.setPixmap(QPixmap(u":/icons/icons/pngfind.com-png-circle-1194554.png"))
        self.imp_meas_info.setScaledContents(True)

        self.horizontalLayout_11.addWidget(self.imp_meas_info)


        self.verticalLayout_35.addWidget(self.frame_, 0, Qt.AlignHCenter)

        self.label_6 = QLabel(self.frame_impedance)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMaximumSize(QSize(16777215, 30))

        self.verticalLayout_35.addWidget(self.label_6)


        self.verticalLayout_22.addWidget(self.frame_impedance)

        self.stackedWidget.addWidget(self.page_impedance)
        self.page = QWidget()
        self.page.setObjectName(u"page")
        self.btn_scan_2 = QPushButton(self.page)
        self.btn_scan_2.setObjectName(u"btn_scan_2")
        self.btn_scan_2.setGeometry(QRect(770, 140, 119, 25))
        self.btn_scan_2.setMinimumSize(QSize(0, 25))
        self.btn_scan_2.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_scan_2.setStyleSheet(u"background-image: url(:/icons/icons/cil-reload.png);\n"
"background-repeat: no-repeat;\n"
"background-position: center;\n"
"/*\n"
"border-radius: 5px;\n"
"	background-position: left center;\n"
"    	background-repeat: no-repeat;\n"
"	border: none;\n"
"	border-left: 20px solid transparent;\n"
"	text-align: left;\n"
"	padding-left: 44px;\n"
"	color: rgb(189, 189, 189);\n"
"*/")
        self.stackedWidget.addWidget(self.page)
        self.page_settings_testing = QWidget()
        self.page_settings_testing.setObjectName(u"page_settings_testing")
        self.label_5 = QLabel(self.page_settings_testing)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 30, 701, 51))
        self.label_5.setFont(font2)
        self.label_5.setStyleSheet(u"color:#fff")
        self.label_5.setAlignment(Qt.AlignCenter)
        self.label_5.setWordWrap(True)
        self.graphicsView = GraphicsLayoutWidget(self.page_settings_testing)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(10, 100, 981, 371))
        self.splitter = QSplitter(self.page_settings_testing)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setGeometry(QRect(20, 500, 431, 61))
        self.splitter.setOrientation(Qt.Horizontal)
        self.pushButton_2 = QPushButton(self.splitter)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.splitter.addWidget(self.pushButton_2)
        self.pushButton_3 = QPushButton(self.splitter)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.splitter.addWidget(self.pushButton_3)
        self.stackedWidget.addWidget(self.page_settings_testing)

        self.verticalLayout_4.addWidget(self.stackedWidget)

        self.main_footer = QFrame(self.center_main_items)
        self.main_footer.setObjectName(u"main_footer")
        self.main_footer.setMinimumSize(QSize(0, 40))
        self.main_footer.setMaximumSize(QSize(16777215, 40))
        self.main_footer.setStyleSheet(u"")
        self.main_footer.setFrameShape(QFrame.StyledPanel)
        self.main_footer.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_28 = QHBoxLayout(self.main_footer)
        self.horizontalLayout_28.setSpacing(0)
        self.horizontalLayout_28.setObjectName(u"horizontalLayout_28")
        self.horizontalLayout_28.setContentsMargins(0, 0, 0, 0)
        self.ft_label_device_3 = QLabel(self.main_footer)
        self.ft_label_device_3.setObjectName(u"ft_label_device_3")
        self.ft_label_device_3.setStyleSheet(u"")

        self.horizontalLayout_28.addWidget(self.ft_label_device_3)

        self.horizontalLayout_25 = QHBoxLayout()
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.ft_label_firmware = QLabel(self.main_footer)
        self.ft_label_firmware.setObjectName(u"ft_label_firmware")
        self.ft_label_firmware.setStyleSheet(u"")

        self.horizontalLayout_25.addWidget(self.ft_label_firmware)

        self.ft_label_firmware_value = QLabel(self.main_footer)
        self.ft_label_firmware_value.setObjectName(u"ft_label_firmware_value")

        self.horizontalLayout_25.addWidget(self.ft_label_firmware_value)


        self.horizontalLayout_28.addLayout(self.horizontalLayout_25)

        self.horizontalLayout_26 = QHBoxLayout()
        self.horizontalLayout_26.setSpacing(0)
        self.horizontalLayout_26.setObjectName(u"horizontalLayout_26")
        self.ft_label_battery = QLabel(self.main_footer)
        self.ft_label_battery.setObjectName(u"ft_label_battery")
        self.ft_label_battery.setStyleSheet(u"")

        self.horizontalLayout_26.addWidget(self.ft_label_battery, 0, Qt.AlignHCenter)

        self.ft_label_battery_value = QLabel(self.main_footer)
        self.ft_label_battery_value.setObjectName(u"ft_label_battery_value")
        self.ft_label_battery_value.setStyleSheet(u"")

        self.horizontalLayout_26.addWidget(self.ft_label_battery_value)


        self.horizontalLayout_28.addLayout(self.horizontalLayout_26)

        self.horizontalLayout_27 = QHBoxLayout()
        self.horizontalLayout_27.setSpacing(10)
        self.horizontalLayout_27.setObjectName(u"horizontalLayout_27")
        self.ft_label_temp = QLabel(self.main_footer)
        self.ft_label_temp.setObjectName(u"ft_label_temp")
        self.ft_label_temp.setStyleSheet(u"")

        self.horizontalLayout_27.addWidget(self.ft_label_temp)

        self.ft_label_temp_value = QLabel(self.main_footer)
        self.ft_label_temp_value.setObjectName(u"ft_label_temp_value")

        self.horizontalLayout_27.addWidget(self.ft_label_temp_value)


        self.horizontalLayout_28.addLayout(self.horizontalLayout_27)

        self.ft_label_version = QLabel(self.main_footer)
        self.ft_label_version.setObjectName(u"ft_label_version")
        self.ft_label_version.setStyleSheet(u"")
        self.ft_label_version.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_28.addWidget(self.ft_label_version)

        self.frame_size_grip = QFrame(self.main_footer)
        self.frame_size_grip.setObjectName(u"frame_size_grip")
        self.frame_size_grip.setMinimumSize(QSize(20, 20))
        self.frame_size_grip.setMaximumSize(QSize(20, 20))
        self.frame_size_grip.setCursor(QCursor(Qt.SizeFDiagCursor))
        self.frame_size_grip.setFrameShape(QFrame.NoFrame)
        self.frame_size_grip.setFrameShadow(QFrame.Raised)

        self.horizontalLayout_28.addWidget(self.frame_size_grip, 0, Qt.AlignBottom)


        self.verticalLayout_4.addWidget(self.main_footer)


        self.horizontalLayout.addWidget(self.center_main_items)


        self.verticalLayout.addWidget(self.main_body)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.stackedWidget.setCurrentIndex(1)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_rec.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionExport_Metadata.setText(QCoreApplication.translate("MainWindow", u"Export Metadata", None))
        self.actionDocumentation.setText(QCoreApplication.translate("MainWindow", u"Documentation", None))
        self.actionReport_an_issue.setText(QCoreApplication.translate("MainWindow", u"Report an issue", None))
        self.actionReport_a_bug.setText(QCoreApplication.translate("MainWindow", u"Report a bug", None))
        self.actionContact.setText(QCoreApplication.translate("MainWindow", u"Contact", None))
        self.mentalab_logo.setText("")
        self.btn_minimize.setText("")
        self.btn_restore.setText("")
        self.btn_close.setText("")
        self.btn_left_menu_toggle.setText(QCoreApplication.translate("MainWindow", u"Hide", None))
        self.btn_home.setText(QCoreApplication.translate("MainWindow", u"HOME", None))
        self.btn_settings.setText(QCoreApplication.translate("MainWindow", u"SETTINGS", None))
        self.btn_plots.setText(QCoreApplication.translate("MainWindow", u"VISUALIZATION", None))
        self.btn_impedance.setText(QCoreApplication.translate("MainWindow", u"IMPEDANCE", None))
        self.btn_integration.setText(QCoreApplication.translate("MainWindow", u"INTEGRATION", None))
        self.home_logo.setText("")
        self.home_title.setText(QCoreApplication.translate("MainWindow", u"Welcome to Mentalab's ExplorePy", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">HOW TO</span></p><p><img src=\":/icons/icons/cil-settings.png\"/> Connect to your device and change its settings through the Settings page</p><p><img src=\":/icons/icons/cil-chart-line.png\"/> Filter and visualize the ExG signal, its spectral analysis and the orientation data</p><p><img src=\":/icons/icons/cil-speedometer.png\"/> Measure and visualize the channel's impedance</p><p><img src=\":/icons/icons/cil-share-boxed.png\"/> Integrate with other platforms</p><p><br/></p></body></html>", None))
        self.integration_log.setText("")
        self.integration_title.setText(QCoreApplication.translate("MainWindow", u"INTEGRATION", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:600;\">Push to LSL</span></p></body></html>", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Stream data from other software such as OpenVibe or other programming languages. </p><p>It will create three LSL streams for ExG, Orientation and markers. </p></body></html>", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"Set duration (the default stream duration is 3600 seconds.)", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Duration (s):", None))
        self.btn_push_lsl.setText(QCoreApplication.translate("MainWindow", u"Push", None))
        self.settings_log.setText("")
        self.settings_title.setText(QCoreApplication.translate("MainWindow", u"SETTINGS", None))
        self.label_explore_name_2.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Connect your device</p></body></html>", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Scan for nearby devices or directly input the name of your device</p></body></html>", None))
        self.btn_scan.setText(QCoreApplication.translate("MainWindow", u"Scan", None))
        self.btn_connect.setText(QCoreApplication.translate("MainWindow", u"Connect", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Device Name: ", None))
        self.dev_name_input.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Explore_XXXX or XXXX", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Import data", None))
        self.btn_import_data.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_explore_name.setText(QCoreApplication.translate("MainWindow", u"Explore_XXXXX", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Number of channels: ", None))
        self.label_activeChannels.setText(QCoreApplication.translate("MainWindow", u"Active channels", None))
        self.cb_ch1.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.cb_ch2.setText(QCoreApplication.translate("MainWindow", u"2", None))
        self.cb_ch3.setText(QCoreApplication.translate("MainWindow", u"3", None))
        self.cb_ch4.setText(QCoreApplication.translate("MainWindow", u"4", None))
        self.cb_ch5.setText(QCoreApplication.translate("MainWindow", u"5", None))
        self.cb_ch6.setText(QCoreApplication.translate("MainWindow", u"6", None))
        self.cb_ch7.setText(QCoreApplication.translate("MainWindow", u"7", None))
        self.cb_ch8.setText(QCoreApplication.translate("MainWindow", u"8", None))
        self.label_samping_rate.setText(QCoreApplication.translate("MainWindow", u"Sampling Rate", None))
        self.btn_apply_settings.setText(QCoreApplication.translate("MainWindow", u"Apply changes", None))
        self.btn_format_memory.setText(QCoreApplication.translate("MainWindow", u"Format MEM", None))
        self.btn_reset_settings.setText(QCoreApplication.translate("MainWindow", u"Reset Settings", None))
        self.btn_calibrate.setText(QCoreApplication.translate("MainWindow", u"Calibrate ORN", None))
        self.label_signal.setText(QCoreApplication.translate("MainWindow", u"Signal", None))
        self.label_yAxis.setText(QCoreApplication.translate("MainWindow", u"Y-axis Scale", None))
        self.label_timeScale.setText(QCoreApplication.translate("MainWindow", u"Time window", None))
        self.label_heartRate.setText(QCoreApplication.translate("MainWindow", u"Heart Rate (bpm)", None))
        self.value_heartRate.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.btn_stream.setText(QCoreApplication.translate("MainWindow", u"Start Data Stream", None))
        self.btn_record.setText(QCoreApplication.translate("MainWindow", u" Record", None))
        self.label_recording_time.setText(QCoreApplication.translate("MainWindow", u"00:00:00", None))
        self.value_event_code.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Event Code", None))
        self.btn_marker.setText(QCoreApplication.translate("MainWindow", u" Set", None))
        self.btn_plot_filters.setText(QCoreApplication.translate("MainWindow", u"Filters", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.exg), QCoreApplication.translate("MainWindow", u"  ExG  ", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.orn), QCoreApplication.translate("MainWindow", u"  Orientation  ", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.fft), QCoreApplication.translate("MainWindow", u"  FFT  ", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><a href=\"mentalab.com\"><span style=\" text-decoration: underline; color:#0069d9;\">Back to recorded data visualization</span></a></p></body></html>", None))
        self.label_signal_rec.setText(QCoreApplication.translate("MainWindow", u"Signal", None))
        self.label_yAxis_rec.setText(QCoreApplication.translate("MainWindow", u"Y-axis Scale", None))
        self.label_timeScale_rec.setText(QCoreApplication.translate("MainWindow", u"Time window", None))
        self.label_heartRate_rec.setText(QCoreApplication.translate("MainWindow", u"Heart Rate (bpm)", None))
        self.value_heartRate_rec.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.btn_stream_rec.setText(QCoreApplication.translate("MainWindow", u"Start Visualization", None))
        self.cb_swipping_rec.setText(QCoreApplication.translate("MainWindow", u"Moving window", None))
        self.tabWidget_rec.setTabText(self.tabWidget_rec.indexOf(self.exg_rec), QCoreApplication.translate("MainWindow", u"  ExG  ", None))
        self.tabWidget_rec.setTabText(self.tabWidget_rec.indexOf(self.orn_rec), QCoreApplication.translate("MainWindow", u"  Orientation  ", None))
        self.tabWidget_rec.setTabText(self.tabWidget_rec.indexOf(self.fft_rec), QCoreApplication.translate("MainWindow", u"  FFT  ", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><a href=\"mentalab.com\"><span style=\" text-decoration: underline; color:#0069d9;\">Back to RT visualization</span></a></p></body></html>", None))
        self.label_4.setText("")
        self.impedance_title.setText(QCoreApplication.translate("MainWindow", u"IMPEDANCE MEASUREMENT", None))
        self.label_ch1.setText(QCoreApplication.translate("MainWindow", u"Ch1", None))
        self.label_ch1_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch2.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:13pt;\">Ch2</span></p></body></html>", None))
        self.label_ch2_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch3.setText(QCoreApplication.translate("MainWindow", u"Ch3", None))
        self.label_ch3_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch4.setText(QCoreApplication.translate("MainWindow", u"Ch4", None))
        self.label_ch4_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch5.setText(QCoreApplication.translate("MainWindow", u"Ch5", None))
        self.label_ch5_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch6.setText(QCoreApplication.translate("MainWindow", u"Ch6", None))
        self.label_ch6_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch7.setText(QCoreApplication.translate("MainWindow", u"Ch7", None))
        self.label_ch7_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch8.setText(QCoreApplication.translate("MainWindow", u"Ch8", None))
        self.label_ch8_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch9.setText(QCoreApplication.translate("MainWindow", u"Ch9", None))
        self.label_ch9_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch10.setText(QCoreApplication.translate("MainWindow", u"Ch10", None))
        self.label_ch10_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch11.setText(QCoreApplication.translate("MainWindow", u"Ch11", None))
        self.label_ch11_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch12.setText(QCoreApplication.translate("MainWindow", u"Ch12", None))
        self.label_ch12_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch13.setText(QCoreApplication.translate("MainWindow", u"Ch13", None))
        self.label_ch13_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch14.setText(QCoreApplication.translate("MainWindow", u"Ch14", None))
        self.label_ch14_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch15.setText(QCoreApplication.translate("MainWindow", u"Ch15", None))
        self.label_ch15_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.label_ch16.setText(QCoreApplication.translate("MainWindow", u"Ch16", None))
        self.label_ch16_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
#if QT_CONFIG(tooltip)
        self.btn_imp_meas.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.btn_imp_meas.setText(QCoreApplication.translate("MainWindow", u"Measure Impedances", None))
#if QT_CONFIG(tooltip)
        self.imp_meas_info.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Sum of impedances of REF and inndividual channels divided by 2</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.imp_meas_info.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Please restart the device after you have finished measuring impedances or <a href=\"www.mentalab.com\"><span style=\" text-decoration: underline; color:#0069d9;\">click here</span></a></p></body></html>", None))
        self.btn_scan_2.setText(QCoreApplication.translate("MainWindow", u"Scan nearby devices", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"TESTING PLOTS", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Draw", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.ft_label_device_3.setText(QCoreApplication.translate("MainWindow", u"Not connnected", None))
        self.ft_label_firmware.setText(QCoreApplication.translate("MainWindow", u"Firmware Version ", None))
        self.ft_label_firmware_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.ft_label_battery.setText(QCoreApplication.translate("MainWindow", u"Battery %", None))
        self.ft_label_battery_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.ft_label_temp.setText(QCoreApplication.translate("MainWindow", u"Device temperature:", None))
        self.ft_label_temp_value.setText(QCoreApplication.translate("MainWindow", u"NA", None))
        self.ft_label_version.setText(QCoreApplication.translate("MainWindow", u"v0.1", None))
    # retranslateUi

