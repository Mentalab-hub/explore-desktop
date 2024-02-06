# -*- mode: python ; coding: utf-8 -*-
from os import path
import shutil
import sys
import pylsl
import mne
import eeglabio
from distutils.sysconfig import get_python_lib
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files
import vispy.glsl
import vispy.io
import vispy.visuals
import freetype

import glob

global DISTPATH

icon_base_path = path.join(".", "installer", "ExploreDesktopInstaller", "ExploreDesktop", "packages", "com.Mentalab.ExploreDesktop", "extras")

exe_name = "ExploreDesktop"

if sys.platform == "darwin":
    icon_path = path.join(icon_base_path, "MentalabLogo.icns")
else:
    icon_path = path.join(icon_base_path, "MentalabLogo.ico")

block_cipher = None
main_path = path.join('exploredesktop', 'main.py')

liblsl_path = next(pylsl.pylsl.find_liblsl_libraries())
liblsl_dir = os.path.dirname(liblsl_path)
liblsl_v = pylsl.pylsl.library_version()

if sys.platform == "linux" or sys.platform == "linux2":
    # TODO paths should not be hardcoded
    binaries = [(liblsl_path, 'pylsl/lib'), (liblsl_path[:-2], 'pylsl/lib'), (liblsl_path[:-2]+'.1.16.0', 'pylsl/lib')]
elif sys.platform == "darwin":
    liblsl_dylib = os.path.join(liblsl_dir, 'liblsl.dylib')
    liblsl_dylib_major_minor = glob.glob(os.path.join(liblsl_dir, f'liblsl.{liblsl_v//100}.{liblsl_v%100}.*.dylib'))[0]
    binaries = [(liblsl_dylib, 'pylsl/lib'),
                (liblsl_dylib_major_minor, 'pylsl/lib')]
elif sys.platform == "win32":
    binaries = None

hidden_imports = [
    "vispy.ext._bundled.six",
    "vispy.app.backends._pyside6",
    "freetpye"
]
hidden_imports += collect_submodules('pandas._libs')

a = Analysis([main_path],
             pathex=[get_python_lib()],
             binaries=binaries,
             datas=[],
             hiddenimports=hidden_imports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
#a.datas += Tree(path.dirname(pylsl.__file__), prefix='pylsl', excludes='__pycache__')
a.datas += Tree(path.dirname(mne.__file__), prefix='mne', excludes='__pycache__')
a.datas += Tree(path.dirname(eeglabio.__file__), prefix='eeglabio', excludes='__pycache__')
a.datas += Tree(os.path.dirname(vispy.glsl.__file__), os.path.join("vispy", "glsl"))
a.datas += Tree(os.path.join(os.path.dirname(vispy.io.__file__), "_data"), os.path.join("vispy", "io", "_data"))
a.datas += Tree(os.path.join(os.path.dirname(vispy.ext.__file__)), os.path.join("vispy", "ext"))
a.datas += Tree(os.path.dirname(freetype.__file__), os.path.join("freetype"))

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name=exe_name,
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon=icon_path)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name=exe_name)

if sys.platform == 'darwin':
    app = BUNDLE(coll,
                 name=f'{exe_name}.app',
                 icon=icon_path,
                 bundle_identifier='com.mentalab.exploredesktop',
                 version='0.7.1',
                 info_plist={
                  'NSPrincipalClass': 'NSApplication',
                  'NSAppleScriptEnabled': False,
                  'NSHighResolutionCapable': True,
                  'LSBackgroundOnly': False,
                  'NSBluetoothPeripheralUsageDescription': 'ExploreDesktop uses Bluetooth to communicate with the Explore devices.'
                 })

target_location = path.join(DISTPATH, exe_name)
shutil.copy(icon_path, target_location)