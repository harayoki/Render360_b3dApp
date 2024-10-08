from typing import List, Literal, Tuple
from pathlib import Path
import datetime
import argparse
import os
import sys
import bpy
import re
from mathutils import Vector
import mathutils
import math
import random


def get_time_stamp() -> str:
    now = datetime.datetime.now()
    # return now.strftime("%Y%m%d%H%M%S")
    return now.strftime("%H:%M:%S")


def resource_path(relative_path_str: str) -> Path:
    """
    PyInstallerの実行時か否かで場合分けしてリソースファイルのパスを取得する
    """
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / Path(relative_path_str)
    return Path(__file__).absolute().parent.parent / relative_path_str


def get_all_children(parent_obj: bpy.types.Object) -> List[bpy.types.Object]:
    """すべての子オブジェクトを取得する"""
    children = []
    for child in parent_obj.children:
        children.append(child)
        children.extend(get_all_children(child))  # 再帰処理
    return children


def get_mesh_objects_in_hierarchy(obj: bpy.types.Object) -> List[bpy.types.Object]:
    """指定オブジェクトの階層以下のメッシュオブジェクトを再帰的に取得する"""
    mesh_objects = []
    try:
        if obj.type == 'MESH':
            mesh_objects.append(obj)
    except ReferenceError:
        # たまに削除したオブジェクト参照が残っている
        pass
    for child in obj.children:
        mesh_objects.extend(get_mesh_objects_in_hierarchy(child))
    return mesh_objects


def join_mesh_objects(mesh_objects: List[bpy.types.Object], context: bpy.context) -> bpy.types.Object:
    """メッシュオブジェクトを一つに結合する"""
    # アクティブオブジェクトとして1つを設定し、それを基準に結合
    context.view_layer.objects.active = mesh_objects[0]
    # すべてのメッシュオブジェクトを選択状態にする
    for obj in mesh_objects:
        # print(f"merge object: {obj.name}")
        obj.select_set(True)
    # メッシュオブジェクトを結合
    bpy.ops.object.join()
    return context.active_object


def scale_to_unit_box(obj: bpy.types.Object, scale: float = 1.0) -> float:
    """
    オブジェクトを指定の大きさにスケーリングする。
    内部処理で、スケールを確定しているので注意。
    :return サイズ変更率
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # オブジェクトのバウンディングボックスを取得
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    # 各軸のバウンディングボックスサイズを計算
    bbox_min = Vector((min(corner[i] for corner in bbox_corners) for i in range(3)))
    bbox_max = Vector((max(corner[i] for corner in bbox_corners) for i in range(3)))
    bbox_size = bbox_max - bbox_min
    # 最大サイズを基準にスケールを計算
    max_bbox_dim = max(bbox_size.x, bbox_size.y, bbox_size.z)
    if max_bbox_dim == 0:
        raise RuntimeError("バウンディングボックスのサイズが0です。")
    current_scale = obj.scale.copy()
    scale_factor = scale / max_bbox_dim
    obj.scale = current_scale * scale_factor
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return scale_factor


def remove_object_tree(obj_name: str, use_re: bool = True) -> None:
    """
    指定オブジェクトを下位ヒエラルキーも含め削除する
    obj_name: 削除対象オブジェクト名
    use_re: obj_name に正規表現を使うか否か
    """
    objects: List[bpy.types.Object] = []
    if use_re:
        objects = [o for o in bpy.data.objects if re.match(obj_name, o.name)]
    else:
        obj: bpy.types.Object = bpy.data.objects.get(obj_name)
        if obj is not None:
            objects.append(obj)
    if len(objects) == 0:
        print(f"{obj_name} not found")
        return

    object_to_remove: List[bpy.types.Object] = []
    for obj in objects:
        object_to_remove.append(obj)
        object_to_remove += get_all_children(obj)
    for obj in object_to_remove:
        print(f"removing... {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)


def move_to_root_keep_rotation(obj: bpy.types.Object) -> None:
    """
    最終的なトランスフォームを維持してオブジェクトをルートに移動する
    """
    world_matrix = obj.matrix_world.copy()
    # bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    obj.parent = None
    obj.matrix_world = world_matrix


def set_material_metallic_and_roughness(
        material: bpy.types.Material, metallic_value: float, roughness_value: float) -> None:
    """
    指定されたマテリアルのPrincipled BSDFノードに対して、メタリックとラフネスを設定する。
    :param material: 対象のマテリアル
    :param metallic_value: メタリックの値（0.0〜1.0）
    :param roughness_value: ラフネスの値（0.0〜1.0）
    """
    node_tree = material.node_tree
    principled_bsdf = None
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_bsdf = node
            break
    principled_bsdf.inputs['Metallic'].default_value = metallic_value
    principled_bsdf.inputs['Roughness'].default_value = roughness_value


def load_model(input_file_path: Path) -> List[bpy.types.Object]:
    objects_before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=input_file_path.as_posix())
    objects_after = set(bpy.data.objects)
    new_objects = objects_after - objects_before
    return list(new_objects)


def fit_object_to_camera(obj: bpy.types.Object, camera: bpy.types.Object, asobi: float = 1.6) -> None:
    """
    オブジェクトのバウンディングボックスがカメラに完全に収まるように、カメラの視野角を調整する
    """

    # オブジェクトのバウンディングボックスのコーナーを取得
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    # カメラの座標と視点ベクトルを取得
    cam_location = camera.matrix_world.translation
    cam_direction = camera.matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))

    # カメラの方向ベクトルから各コーナーまでの距離を取得
    distances = [(cam_location - corner).dot(cam_direction) for corner in bbox_corners]
    max_distance = max(distances)

    # オブジェクトをカメラの前に収めるための最大距離を取得
    furthest_corner = bbox_corners[distances.index(max_distance)]

    # カメラのFOV調整（垂直視野角に基づく）
    scene = bpy.context.scene
    aspect_ratio = scene.render.resolution_x / scene.render.resolution_y

    # オブジェクトの最大高さと幅をカメラに対して計算
    width = max((cam_location - corner).dot(mathutils.Vector((1.0, 0.0, 0.0))) for corner in bbox_corners)
    height = max((cam_location - corner).dot(mathutils.Vector((0.0, 1.0, 0.0))) for corner in bbox_corners)

    # 視野角を計算
    distance_to_object = (cam_location - furthest_corner).length
    fov_y = 2 * math.atan(height / (2 * distance_to_object))

    # アスペクト比に基づいて水平視野角を計算
    fov_x = 2 * math.atan((width / aspect_ratio) / (2 * distance_to_object))

    # カメラの視野角（FOV）を設定 2.0をかけないと大きすぎた
    camera.data.angle = max(fov_y, fov_x) * 2.0 * asobi


def set_keyframe_at_frame(obj: bpy.types.Object, frame: int, location=None, rotation=None, scale=None):
    """
    指定したオブジェクトに、指定したフレームでキーフレームを設定する。
    Parameters:
        obj: キーフレームを設定するオブジェクト
        frame (int): キーフレームを設定するフレーム番号
        location (tuple, optional): 位置 (x, y, z) のタプル
        rotation (tuple, optional): 回転 (x, y, z) のタプル（ラジアンで指定）
        scale (tuple, optional): スケール (x, y, z) のタプル
    """

    # キーフレームの設定
    if location is not None:
        obj.location = location  # 位置を設定
        obj.keyframe_insert(data_path="location", frame=frame)  # 位置にキーフレームを挿入

    if rotation is not None:
        obj.rotation_euler = rotation  # 回転を設定 (Euler回転)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)  # 回転にキーフレームを挿入

    if scale is not None:
        obj.scale = scale  # スケールを設定
        obj.keyframe_insert(data_path="scale", frame=frame)  # スケールにキーフレームを挿入


###################################


ALLOWED_INPUT_EXTENSIONS: List[str] = [".glb"]
RENDERERS: Tuple = ("CYCLES", "EEVEE", "WORKBENCH")
DEG_TO_ROT = math.pi / 180
VIEW_TRANSFORMS: Tuple = ('Standard', 'Khronos PBR Neutral', 'AgX', 'Filmic', 'Filmic Log', 'False Color', 'Raw')


def main():
    print(f"bpy version: {bpy.app.version_string}")
    parser = argparse.ArgumentParser(
        "glbファイルの360度レンダリングを行う"
    )
    parser.add_argument(
        "file_path", type=str,
        help=f"入力ファイル ({', '.join(['*' + ext for ext in ALLOWED_INPUT_EXTENSIONS])})")
    parser.add_argument(
        "-o", "--output_dir", type=str, default="", help=f"出力ディレクトリ")
    parser.add_argument("--cam_deg_x", type=int, nargs="*", default=[15, 30, 45, 60],  # [30]
                        help="360度レンダリングする際にカメラをX軸周りに回転させる角度 複数指定すると複数回レンダリングする")
    parser.add_argument(
        "--render_size", type=int, nargs=2, default=[512, 512], help="レンダリングサイズ")
    parser.add_argument(
        "--engine", type=str, choices=RENDERERS, default="EEVEE",
        help="レンダリングエンジンの指定")
    parser.add_argument("--no_ground", action="store_true", help="地面を表示しない")
    parser.add_argument("--save_blend", action="store_true", help="blendファイルも出力するか")
    parser.add_argument("--confirm", action="store_true", help="確認メッセージを表示するか")
    parser.add_argument("--frames", type=int, default=6, help="レンダリングフレーム数")
    parser.add_argument("--start_frames", type=int, default=1, help="レンダリング開始フレーム数")
    parser.add_argument("--gamma", type=float, default=1.0, help="ガンマ補正")
    parser.add_argument("--view_transform", type=str, choices=VIEW_TRANSFORMS, default="Standard",
                        help="カラーマネージメント ビュートランスフォームの指定")
    parser.add_argument("--scale", type=float, default=1.0, help="読み込みオブジェクトの表示倍率")
    parser.add_argument("--cycles_samples", type=int, default=128, help="Cyclesのサンプル数")
    parser.add_argument("--remove_object_names", type=str, nargs="*", default=[],
                        help="レンダリング時に削除したいオブジェクト名称（正規表現）の指定")
    parser.add_argument("--one_file", action="store_true",
                        help="テスト用にどこかの1フレームのみレンダリングする")
    args = parser.parse_args()
    args.render_size = max(256, args.render_size[0]), max(256, args.render_size[1])

    assert os.path.exists(args.file_path), f"File not found: {args.file_path}"
    assert Path(args.file_path).suffix in ALLOWED_INPUT_EXTENSIONS, \
        f"Unsupported input file format: {Path(args.file_path).suffix}"

    input_file_path: Path = Path(args.file_path)
    output_dir_path: Path
    if args.output_dir == "":
        output_dir_path = input_file_path.parent / Path(input_file_path.stem + "_render")
    else:
        output_dir_path = Path(args.output_dir).absolute()
    if output_dir_path.is_file():
        output_dir_path = output_dir_path.parent
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    output_path_blend = output_dir_path / Path(input_file_path.stem + "_render.blend")

    assert args.frames > 0, "Frames must be greater than 0."
    assert args.start_frames > 0, "Frames must be greater than 0."
    assert args.start_frames <= args.frames, "Start_frames must be less than or equal to Frames."
    assert args.scale > 0, "Scale must be greater than 0."
    assert min(args.cam_deg_x) >= -90, "Cam_deg_x must be greater than or equal to 0."
    assert max(args.cam_deg_x) <= 90, "Cam_deg_x must be less than or equal to 90."

    print("--------------------------------")
    print(f"Input file: {input_file_path}")
    print(f"Output Dir: {output_dir_path}")
    print(f"Texture size: {args.render_size}")
    print(f"Renderer: {args.engine}")
    print(f"Frames: {args.frames}")
    print(f"Scale: {args.scale}")
    print(f"Gamma: {args.gamma}")
    print(f"View Transform: {args.view_transform}")
    print(f"Cam_deg_x: {args.cam_deg_x}")
    print(f"Ground: {args.no_ground}")
    print(f"Remove objects: {args.remove_object_names}")
    if args.engine == "CYCLES":
        print(f"Cycles samples: {args.cycles_samples}")
    if args.save_blend:
        print(f"Save blend file: {output_path_blend}")
    print(f"one_file: {args.one_file}")
    print("--------------------------------")

    if args.confirm:
        prompt = input("Continue? [y/n]: ")
        if prompt.lower() != "y":
            sys.exit("Aborted.")

    if args.one_file:
        args.cam_deg_x = random.choices(args.cam_deg_x, k=1)
        args.start_frames = args.frames = random.choices(range(args.start_frames, args.frames + 1), k=1)[0]

    bpy.ops.wm.open_mainfile(filepath=resource_path("blend/render_template.blend").as_posix())
    bpy.context.scene.render.resolution_x = args.render_size[0]
    bpy.context.scene.render.resolution_y = args.render_size[1]
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = args.frames
    if args.engine == "EEVEE":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    elif args.engine == "CYCLES":
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = args.cycles_samples
    elif args.engine == "WORKBENCH":
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'

    # カラーマネジメント
    bpy.context.scene.view_settings.gamma = args.gamma
    bpy.context.scene.view_settings.view_transform = args.view_transform

    # モデルを読み込む
    new_models = load_model(input_file_path)

    # 読み込んだモデルからレンダリング前に削除したいものがあれば削除
    if args.remove_object_names:
        for remove_name in args.remove_object_names:
            remove_object_tree(remove_name, use_re=True)

    # 読み込んだオブジェクト一覧から削除したものを除外
    for o in new_models:
        try:
            if o.name:  # 例外が発生するかどうかで削除されたか判定
                pass
        except ReferenceError:
            new_models.remove(o)
            continue

    # 全てのモデルをルートに移動して1つにまとめる
    mesh_objects = []
    for m in new_models:
        mesh_objects.extend(get_mesh_objects_in_hierarchy(m))
    for obj in mesh_objects:
        move_to_root_keep_rotation(obj)
    merged: bpy.types.Object = join_mesh_objects(mesh_objects, bpy.context)
    bpy.context.view_layer.objects.active = merged
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    merged.location = (0, 0, 0)
    scale_factor: float = scale_to_unit_box(merged, args.scale)
    # print(f"Scale factor: {scale_factor}")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # カメラに「Track To」コンストレイントを追加
    camera = bpy.data.objects["Camera"]
    track_to = camera.constraints.new(type='TRACK_TO')
    track_to.target = merged
    track_to.track_axis = 'TRACK_NEGATIVE_Z'  # カメラの前方向が -Z
    track_to.up_axis = 'UP_Y'  # 上方向をY軸に設定

    # 地面の位置の調整
    if args.no_ground:
        remove_object_tree(obj_name="Ground")
    else:
        bbox_corners_z = [(merged.matrix_world @ Vector(corner)).z for corner in merged.bound_box]
        merged_min_z = min(bbox_corners_z)
        ground = bpy.data.objects["Ground"]
        ground.location.z = merged_min_z - 0.0

    # unit_cube = bpy.data.objects["UnitCube"]

    # カメラの高さの回転を調整しながら360度レンダリング
    camera_rig = bpy.data.objects["CameraRig"]

    for cam_deg_x in args.cam_deg_x:

        # カメラの回転を設定
        x_rot = - DEG_TO_ROT * cam_deg_x
        y_rot = 0
        set_keyframe_at_frame(camera_rig, 1, rotation=(x_rot, y_rot, 0 * DEG_TO_ROT))
        set_keyframe_at_frame(camera_rig, args.frames + 1, rotation=(x_rot, y_rot, 360 * DEG_TO_ROT))

        # アニメーションのFカーブを取得
        action = camera_rig.animation_data.action
        for fcurve in action.fcurves:
            # 各Fカーブのすべてのキーフレームの補間をLINEARに設定
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'

        # カメラの視野角を調整 毎フレーム行うとおかしくなる
        # bpy.context.scene.frame_set(1)
        # fit_object_to_camera(merged, camera, 1.1)

        # 1フレームずつレンダリング 一括にしていないのは各フレームで調整を入れるかもしれないから & 開発中 途中で止めやすいように
        for frame in range(args.start_frames, args.frames + 1):
            bpy.context.scene.frame_set(frame)
            padded_frame = str(frame).zfill(3)
            output_path = output_dir_path / f"{input_file_path.stem}_rx{cam_deg_x}_{args.engine}_{padded_frame}.png"
            output_path = output_path.absolute()  # 絶対パスに変換しないと Cドライブに保存される
            print(f"Rendering to {output_path}")
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.filepath = output_path.as_posix()
            bpy.context.scene.camera = camera
            bpy.ops.render.render(write_still=True)

    if args.save_blend:
        bpy.context.scene.frame_set(1)
        bpy.ops.wm.save_as_mainfile(filepath=output_path_blend.absolute().as_posix())


if __name__ == "__main__":
    main()

