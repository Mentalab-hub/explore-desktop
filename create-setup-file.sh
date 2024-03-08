#!/usr/bin/env bash




# Conda virtual env
conda config --append channels conda-forge

#conda create -n gui_installer python=3.8.10 -y
conda activate gui_installer
python -m pip install --upgrade pip

# Install qt and qt-ifw (TO BE USED IN FUTURE)
#mkdir temp || rm -rfv temp/*
#cd temp
#pip install aqtinstall
#aqt install-qt linux desktop 6.2.1
#aqt install-tool linux desktop tools_ifw
#aqt install-tool linux desktop tools_maintenance
#cd ..

# Install Pyinstaller
pip install pyinstaller==4.7
pip install --upgrade pyinstaller-hooks-contrib==2023.2

# Install ExploreDesktop
conda install -c conda-forge liblsl==1.15.2 -y
pip install pylsl
pip install -e .
pip uninstall explorepy -y
pip install git+https://github.com/Mentalab-hub/explorepy.git@feature-mac-dev

# Copy files to data dir
exploredesktop_path="installer/ExploreDesktopInstaller/ExploreDesktop/packages/com.Mentalab.ExploreDesktop/"

# Clean required directories
rm -rfv "$exploredesktop_path"data/*
rm -rfv dist/*

# Create executable files
pyinstaller --onedir --console ExploreDesktop.spec



if [[ "$uname" == "Darwin" ]]
then
  cp -r dist/ExploreDesktop "$exploredesktop_path"data
  cp "$exploredesktop_path"extras/MentalabLogo.png "$exploredesktop_path"data/
  cp "$exploredesktop_path"extras/exploredesktop.desktop "$exploredesktop_path"data/
else
  cp -r dist/ExploreDesktop.app "$exploredesktop_path"data
fi


# Extensions
if [[ "$uname" == "Darwin" ]]
then
  extension=".run"
  binarycreator_path=/Users/"$(whoami)"/Qt/Tools/QtInstallerFramework/4.6/bin/
else
  extension=""
  binarycreator_path=/Users/"$(whoami)"/Qt/Tools/QtInstallerFramework/4.6/bin/
fi
"$binarycreator_path"binarycreator -c "$exploredesktop_path"../../config/config.xml -p "$exploredesktop_path"../ --verbose ExploreDesktopInstaller_x64"$extension"
