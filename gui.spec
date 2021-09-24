# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['gui.py'],
             pathex=['C:\\Project\\Python\\Khisoft-DeepLearning-Framework'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False,
             added_files = [
                 ('UI', 'UI'),
                 ( 'model', 'model' ),
                 ( 'data', 'data' ),
                 ( 'raw_datasets', 'raw_datasets' ),
                 ( 'filter_setting.npy', '.' ),
             ]
            )
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='gui',
          debug=true,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False , icon='images\\khisoft.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='gui')
