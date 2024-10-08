# Render360_b3dApp

360度で自動レンダリングするツール。
Blenderのbpyモジュールを利用してツールを作成し、PyInstallerでアプリ化するコードサンプルです。

# 動作環境

Windows11 + Python 3.11.4 で作成されています。bpy・PyInstallerなどのモジュールを利用します。
Macでも動作すると思われます。

動作環境構築までの手順例を下記に記します。

```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## [render360.py](src%2Frender360.py)

任意のglbファイルを自動で360度レンダリングします。

### 使い方

pythonを直接実行します。入力glbファイル以外の引数はすべてオプションです。

```
usage: glbファイルの360度レンダリングを行う [-h] [-o OUTPUT_DIR] [--cam_deg_x [CAM_DEG_X ...]] [--render_size RENDER_SIZE RENDER_SIZE] [--engine {CYCLES,EEVEE,WORKBENCH}] [--no_ground] [--save_blend] [--confirm] [--frames FRAMES] [--gamma GAMMA]
                             [--view_transform {Standard,Khronos PBR Neutral,AgX,Filmic,Filmic Log,False Color,Raw}] [--scale SCALE] [--cycles_samples CYCLES_SAMPLES] [--no_cube_remove]
  --no_ground           地面を表示しない
  --save_blend          blendファイルも出力するか
  --confirm             確認メッセージを表示するか
  --frames FRAMES       レンダリングフレーム数
  --gamma GAMMA         ガンマ補正
  --view_transform {Standard,Khronos PBR Neutral,AgX,Filmic,Filmic Log,False Color,Raw}
                        カラーマネージメント ビュートランスフォームの指定
  --scale SCALE         読み込みオブジェクトの表示倍率
  --cycles_samples CYCLES_SAMPLES
                        Cyclesのサンプル数
  --remove_object_names [REMOVE_OBJECT_NAMES ...]
                        レンダリング時に削除したいオブジェクト名称（正規表現）の指定
  --one_file            テスト用にどこかの1フレームのみレンダリングする
```

### 実行例

OUTPUT_DIR引数はフォルダ指定です。
`python src\render360.py -o "temp" input.glb`

## ツールのアプリ化

PyInstallerでspecファイルを引数指定してください。distフォルダにアプリが出力されます。

```pyinstaller render360.spec```

もしくは

```python -m PyInstaller render360.spec```


アプリの引数はpythonコードを直接実行した場合と同じですが、Windowsの場合はglbファイルのドラッグ＆ドロップでも実行できます。

ドラッグ＆ドロップで実行しつつ引数で細かい挙動を調整したい場合は、以下のようなバッチファイルを作成してバッチファイル側にドラッグ＆ドロップしてください。

``` render.bat
@echo off
render360.exe %1 --render_size 1024 1024 --one_file -o temp --remove_object_names "Cube.*"
```

# ライセンス

GPL-3.0 Licenseです。ご注意ください。
[LICENSE](LICENSE) 
