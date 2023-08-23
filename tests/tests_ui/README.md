# ExploreDesktop

### Notes for the UI Tests

- Currently, each test requires a physical device to run.
- The devices name is per default set to 'Explore_855E' or alternatively just '855E'.
- Additionally ```qtbot.wait()``` and ```sleep()``` statements are often implemented that can be omitted one a working mock instance is implemented.
- When the mock instance is up and running, replace the calls to ```connect_device``` with the calls to the mock server. It is important that a ```MainWindow``` is returned.
- The directory ```reports``` holds successful reports HTML reports for each existing test.
- The directory ```test_files``` holds files that can/are be/being used for UI-tests.




