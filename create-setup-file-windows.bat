@REM @echo off
@REM Check we are in master branch

@REM git branch | find "* master" > NUL & IF ERRORLEVEL 1 (
@REM     echo "Please checkout to master branch and try again!"
@REM     exit
@REM )


@REM # Conda virtual env
call conda config --append channels conda-forge
call conda create -n gui_installer python=3.8.10 -y openssl=1.1.1l
call conda activate gui_installer
@REM call which python
call python -m pip install --upgrade pip

@REM Install qt and qt-ifw (TO BE USED IN FUTURE)
call mkdir temp || rm -rfv temp/*
call cd temp
call pip install aqtinstall
@REM aqt install-qt linux desktop 6.2.1
call aqt install-tool windows desktop tools_ifw
@REM aqt install-tool linux desktop tools_maintenance
call cd ..

@REM Install Pyinstaller
call pip install pyinstaller==4.7

@REM Install ExploreGUI
call pip install .

@REM  Clean required directories
call dir
call set exploregui_path="installer\ExploreGuiInstaller\ExploreGUI\packages\com.Mentalab.ExploreGUI\"
call rd /S /Q %exploregui_path%data
call md %exploregui_path%data
call rd /S /Q dist

@REM Create executable files
call pyinstaller --onedir --console ExploreGUI.spec --log-level TRACE

@REM Copy files to data dir
call xcopy /I /E /H /R /Q dist\ExploreGUI %exploregui_path%data\ExploreGUI
call xcopy %exploregui_path%extras\MentalabLogo.ico %exploregui_path%data


@REM Create installer file
call set config_path= "installer\ExploreGuiInstaller\ExploreGUI\config\config.xml"
call set package_path="installer\ExploreGuiInstaller\ExploreGUI\packages"
call %~dp0\temp\Tools\QtInstallerFramework\4.2\bin\binarycreator -c %config_path% -p %package_path% --verbose ExploreGUIInstaller.exe
