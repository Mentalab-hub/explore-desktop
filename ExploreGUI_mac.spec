# -*- mode: python ; coding: utf-8 -*-
from os import path
import os
import pylsl
import explorepy
from distutils.sysconfig import get_python_lib
from shutil import copy2

block_cipher = None
main_path = path.join('exploregui', 'main.py')

old_path = path.join(path.dirname(explorepy.__file__), 'btScan')
# /Users/andrea/opt/anaconda3/envs/gui8/lib/python3.8/site-packages/explorepy/btScan
new_path = path.join(os.getcwd(), 'dist', 'ExploreGUI', 'explorepy', 'btScan')
# /Users/andrea/MentalabRepo/explorepy-gui/dist/ExploreGUI/explorepy/btScan

a = Analysis([main_path],
             pathex=[get_python_lib()],
             binaries=[],
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
a.datas += Tree(path.dirname(explorepy.__file__), prefix='explorepy', excludes='__pycache__')
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='ExploreGUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon='MentalabLogo.icns')
  

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ExploreGUI')

app = BUNDLE(coll,
             name='ExploreGUI.app',
             icon='MentalabLogo.icns',
             bundle_identifier='com.mentalab.exploregui',
             version='0.1.0',
             info_plist={
              'NSPrincipalClass': 'NSApplication',
              'NSAppleScriptEnabled': False,
              'NSHighResolutionCapable': True,
              'LSBackgroundOnly': False,
              'NSBluetoothPeripheralUsageDescription': 'ExploreGUI uses Bluetooth to communicate with the Explore devices.'
             })

# copy2(old_path, new_path)
# os.system(f'sudo cp {old_path} {new_path}')
