# -*- mode: python ; coding: utf-8 -*-
from os import path
import sys
import pylsl
import mne
import eeglabio
from distutils.sysconfig import get_python_lib

if sys.platform == "darwin":
    icon_path = "./installer/ExploreDesktopInstaller/ExploreDesktop/packages/com.Mentalab.ExploreDesktop/extras/MentalabLogo.icns"
else:
    icon_path = "./installer/ExploreDesktopInstaller/ExploreDesktop/packages/com.Mentalab.ExploreDesktop/extras/MentalabLogo.ico"

block_cipher = None
main_path = path.join('exploredesktop', 'main.py')
liblsl_path = next(pylsl.pylsl.find_liblsl_libraries())


if sys.platform == "linux" or sys.platform == "linux2":
    # TODO paths should not be hardcoded
    binaries = [(liblsl_path, 'pylsl/lib'), (liblsl_path[:-2], 'pylsl/lib'), (liblsl_path[:-2]+'.1.16.0', 'pylsl/lib')]
elif sys.platform == "darwin":
    binaries = [(liblsl_path, 'pylsl/lib'), (liblsl_path[:-5]+'1.15.2.dylib', 'pylsl/lib')]
elif sys.platform == "win32":
    binaries = [(liblsl_path, 'pylsl/lib'), (liblsl_path[:-7], 'pylsl/lib')]

a = Analysis([main_path],
             pathex=[get_python_lib()],
             binaries=binaries,
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
a.datas += Tree(path.dirname(pylsl.__file__), prefix='pylsl', excludes='__pycache__')
a.datas += Tree(path.dirname(mne.__file__), prefix='mne', excludes='__pycache__')
a.datas += Tree(path.dirname(eeglabio.__file__), prefix='eeglabio', excludes='__pycache__')

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='ExploreDesktop',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
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
               name='ExploreDesktop')

if sys.platform == 'darwin':
    app = BUNDLE(coll,
                 name='ExploreDesktop.app',
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