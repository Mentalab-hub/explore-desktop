call conda create -n generate_offline_executable_exploredesktop
call conda activate generate_offline_executable_exploredesktop
call conda install pip
call pip install pyinstaller
call pip install -e .
call pyinstaller ExploreDesktop.spec
call xcopy .\exploredesktop\images\MentalabLogo.ico .\dist\ExploreDesktop\
call conda deactivate
call conda remove -n generate_offline_executable_exploredesktop --all
@pause