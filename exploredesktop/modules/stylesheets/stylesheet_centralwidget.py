###############################
########## VARIABLES ##########
###############################
FONT_SIZE = "11pt"
TITLE_FONT_SIZE = "20pt"

WHITE = "#FCFCFC"
LIGHT_BLUE = "rgb(95, 197, 201)"
LIGHT_GRAY = "rgb(173, 173, 173)"

###############################
##### CENTRAL STYLE SHEET #####
###############################

CENTRAL_STYLESHEET = f"""
#centralwidget{{
	border-top: 1px solid {LIGHT_GRAY};
}}

QWidget{{
	font:{FONT_SIZE};
}}

QFrame{{
	border:none;
}}

#main_header{{
	border:none;
	border-bottom: 1px solid {LIGHT_BLUE};
}}

#main_header .QPushButton{{
	border: none;
}}

#main_footer{{
	border:none;
	border-top: 1px solid {LIGHT_BLUE};
}}

/*#page_home .QPushButton{{
	border:none
}}
*/
#btn_bt_2{{
	border:none
}}

#btn_settings_2{{
	border:none
}}

#btn_plots_2{{
	border:none
}}


#btn_impedance_2{{
	border:none
}}

#btn_integration_2{{
	border:none
}}


#value_heartRate{{
	border: 1px solid
}}

#label_recording_time{{
	border: 1px solid
}}

#list_devices{{
	border: 1px solid;
}}


/*LEFT SIDE MENU*/
#left_side_menu {{
	border:none;
	border-right: 1px solid {LIGHT_BLUE};
}}
#toggle_left_menu{{
	border:none;
}}

#btns_left_menu{{
	border:none;
}}

#left_side_menu .QPushButton{{
	border-radius: 5px;
	border: none;
	text-align: left;
	padding-left: 20px;
}}


/*TITLES FONT*/

QLabel#integration_title{{
	font: {TITLE_FONT_SIZE};
}}

QLabel#home_title{{
	font: {TITLE_FONT_SIZE};
}}

QLabel#impedance_title{{
	font: {TITLE_FONT_SIZE};
}}

QLabel#settings_title{{
	font: {TITLE_FONT_SIZE};
}}

QLabel#bt_title{{
	font: {TITLE_FONT_SIZE};
}}

QTabBar::tab{{
	height: 25px}}
"""