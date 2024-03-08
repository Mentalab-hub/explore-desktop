import logging
import os
import subprocess
from typing import Optional

from PySide6.QtWidgets import QMessageBox


from exploredesktop.modules.utils import display_msg  # isort:skip


logger = logging.getLogger("explorepy.exploredesktop.main")
maintenance_path = os.path.join(os.getcwd(), "maintenancetool")


def check_updates() -> Optional[str]:
    """Check if there are available updates

    Returns:
        Optional[str]: Version number if new version is available
    """

    logger.debug("Maintainance tool path: %s" % maintenance_path)
    process = subprocess.Popen(
        f"{maintenance_path} --checkupdates",
        shell=True, stdout=subprocess.PIPE)
    subprocess_return = process.stdout.read().decode("utf-8")
    print('subprocess return is {}'.format(subprocess_return))
    logger.debug("Check updates output: %s" % subprocess_return)
    print(subprocess_return)
    if 'Warning' in subprocess_return:
        return None

    try:
        new_version = get_version(subprocess_return)
    except IndexError:
        logger.debug("Error opening the maintenancetool. - The system cannot find the path specified.")
        return None
    return new_version


def get_version(string: str) -> str:
    """Get version number

    Args:
        string (str): string output from checkupdates subprocess

    Returns:
        str: version number
    """
    string = string.split('version=')[1]
    string = string.split('size')[0]
    string = string.split('/')[0]
    string = string.split('name')[0]
    string = string.split('id')[0]
    string = string.replace('"', '')
    string = string.replace(' ', '')
    return string


def update_version() -> bool:
    """Update exploredesktop version

    Returns:
        bool: whether version has been updated
    """
    new_version = check_updates()
    if new_version is None:
        return False

    msg = f"New version available: {new_version}.\t\t\n\nUpdate now?"
    response = display_msg(
        msg,
        title="New version found",
        popup_type="question")

    if response == QMessageBox.StandardButton.Yes:
        subprocess.Popen(f"{maintenance_path} --updater", shell=False)
        logger.debug('Updater launched')
        return True
    return False
