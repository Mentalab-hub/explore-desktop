@REM @echo off
@REM Check we are in master branch
@REM git branch | find "* master" > NUL & IF ERRORLEVEL 1 (
@REM     echo "Please checkout to master branch and try again!"
@REM     exit /B 2
@REM )

call set mode=%1
if %mode%==update goto input_ok
if %mode%==create goto input_ok

echo "Wrong input, please specify mode [update OR create]"
exit /B 2

:input_ok
@REM Set path website repo
@REM call set /p repo_wiki_path="Enter the path for explore-desktop-release folder in the wiki repository (<your path>\wiki\static\explore-desktop-release)"
call set repo_wiki_path="C:\Users\ss\Desktop\mentalab-dev\wiki\wiki\static\explore-desktop-release"
if EXIST %repo_wiki_path% (
    echo "Wiki path is correct"
) else (
    echo:
    echo %repo_wiki_path%
    echo "Wiki path is not correct"
    exit /B 2
)

@REM Conda virtual env
call conda config --append channels conda-forge
call conda create -n gui_installer python=3.8.10 -y
call conda activate gui_installer
call python -m pip install --upgrade pip

@REM Install qt and qt-ifw (TO BE USED IN FUTURE)
@REM mkdir temp || rm -rfv temp/*
@REM cd temp
@REM pip install aqtinstall
@REM aqt install-qt linux desktop 6.2.1
@REM aqt install-tool linux desktop tools_ifw
@REM aqt install-tool linux desktop tools_maintenance
@REM cd ..

@REM Install Pyinstaller
call pip install pyinstaller==4.7

@REM Uncomment below if ExploreDesktop required the develop branch of explorepy
call pip uninstall -y explorepy
call pip install git+https://github.com/Mentalab-hub/explorepy.git@develop

@REM Install ExploreDesktop
call pip install eeglabio
call pip install mne
call pip install -e .


@REM  Clean required directories
call set exploredesktop_path="installer\ExploreDesktopInstaller\ExploreDesktop\packages\com.Mentalab.ExploreDesktop\"
call rd /S /Q %exploredesktop_path%data
call md %exploredesktop_path%data
call rd /S /Q dist

@REM Create executable files
call pyinstaller --onedir --console --noconfirm ExploreDesktop.spec

@REM Copy files to data dir
call xcopy /I /E /H /R /Q dist\ExploreDesktop %exploredesktop_path%data\ExploreDesktop
call xcopy %exploredesktop_path%extras\MentalabLogo.ico %exploredesktop_path%data
call set /p asd="Files copied to data dir. Verify and hit enter to continue"

@REM Set path variables
call set main_explore_path=%CD%
call set repo_path=%CD%\repository
call set config_path=%CD%\installer\ExploreDesktopInstaller\ExploreDesktop\config\config.xml
call set package_path=%CD%\installer\ExploreDesktopInstaller\ExploreDesktop\packages

@REM Open repository or update it
if %mode%==create call C:\Qt\Tools\QtInstallerFramework\4.5\bin\repogen.exe -p %package_path% -r --verbose %repo_wiki_path%
if %mode%==update call C:\Qt\Tools\QtInstallerFramework\4.5\bin\repogen.exe --update -p %package_path% --verbose %repo_wiki_path%


@REM Go to wiki repo and push changes
cd %repo_wiki_path%
cd ..\..
call hugo -t ace-documentation
@REM need to commit two times
call git pull
call git add .
call git commit -m "Commit from %DATE%"
call set /p asd="Hit enter to continue"
call git push origin main
cd public
call git pull
call git add .
call git commit -m "Commit from %DATE%"
call set /p asd="Hit enter to continue"
call git push origin main




@REM Create installer file
if %mode%==create (
    call cd %main_explore_path%
    call C:\Qt\Tools\QtInstallerFramework\4.5\bin\binarycreator.exe --online-only -c %config_path% -p %package_path% --verbose ExploreDesktopInstaller.exe
)
