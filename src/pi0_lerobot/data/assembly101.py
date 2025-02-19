import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from icecream import ic
from jaxtyping import Float32
from numpy import ndarray
from serde import field as serde_field
from serde import from_dict, serde
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from tqdm import tqdm
from einops import rearrange

from pi0_lerobot.video_io import MultiVideoReader


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


@serde
class ExoExtriCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    C10404: Float32[ndarray, "4 4"] = serde_field(rename="C10404:rgb")
    C10118: Float32[ndarray, "4 4"] = serde_field(rename="C10118:rgb")
    C10119: Float32[ndarray, "4 4"] = serde_field(rename="C10119:rgb")
    C10095: Float32[ndarray, "4 4"] = serde_field(rename="C10095:rgb")
    C10379: Float32[ndarray, "4 4"] = serde_field(rename="C10379:rgb")
    C10395: Float32[ndarray, "4 4"] = serde_field(rename="C10395:rgb")
    C10115: Float32[ndarray, "4 4"] = serde_field(rename="C10115:rgb")
    C10390: Float32[ndarray, "4 4"] = serde_field(rename="C10390:rgb")


@serde
class EgoCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    C21176875: Float32[ndarray, "4 4"] = serde_field(rename="21176875:mono10bit")
    C21179183: Float32[ndarray, "4 4"] = serde_field(rename="21179183:mono10bit")
    C21110305: Float32[ndarray, "4 4"] = serde_field(rename="21110305:mono10bit")
    C21176623: Float32[ndarray, "4 4"] = serde_field(rename="21176623:mono10bit")


def load_exo_cameras(
    extrinsics_fixed_path: Path,
    train_assembly_hands_json: Path,
    height: int,
    width: int,
) -> list[PinholeParameters]:
    with open(extrinsics_fixed_path) as f:
        extrinsics_fixed = json.load(f)

    exo_raw_extri: ExoExtriCameras = from_dict(ExoExtriCameras, extrinsics_fixed)
    # assembly101 does not have camera intrinsics, so need to get them from assemblyhands
    with open(train_assembly_hands_json) as f:
        train_assembly_hands_dict: dict = json.load(f)

    all_calib_dict: dict = train_assembly_hands_dict["calibration"]
    # assume that all cameras have the same intrinsics for each capture, so only get a single one
    instrinsics_dict: dict[str, list[list[float]]] = next(
        iter(all_calib_dict.values())
    )["intrinsics"]

    pinhole_list: list[PinholeParameters] = []

    cam_name: str
    exo_camera: Float32[ndarray, "4 4"]
    for cam_name, exo_camera in asdict(exo_raw_extri).items():
        intri: Float32[ndarray, "3 3"] = np.array(
            instrinsics_dict[f"{cam_name}_rgb"], dtype=np.float32
        )
        intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(intri[0, 0]),
            fl_y=float(intri[1, 1]),
            cx=float(intri[0, 2]),
            cy=float(intri[1, 2]),
            height=height,
            width=width,
        )
        extri = Extrinsics(
            world_R_cam=exo_camera[:3, :3],
            world_t_cam=exo_camera[:3, 3],
        )
        pinhole_param = PinholeParameters(
            name=cam_name,
            intrinsics=intri,
            extrinsics=extri,
        )
        pinhole_list.append(pinhole_param)

    return pinhole_list


@serde
class Hand3DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 3"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 3"] = serde_field(rename="1")


@serde
class Hand2DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 2"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 2"] = serde_field(rename="1")


@serde
class Exo2DKeypoints:
    C10395: Hand2DKeypoints = serde_field(rename="C10395:rgb")
    C10379: Hand2DKeypoints = serde_field(rename="C10379:rgb")
    C10404: Hand2DKeypoints = serde_field(rename="C10404:rgb")
    C10119: Hand2DKeypoints = serde_field(rename="C10119:rgb")
    C10115: Hand2DKeypoints = serde_field(rename="C10115:rgb")
    C10390: Hand2DKeypoints = serde_field(rename="C10390:rgb")
    C10118: Hand2DKeypoints = serde_field(rename="C10118:rgb")
    C10095: Hand2DKeypoints = serde_field(rename="C10095:rgb")


@serde
class Ego2DKeypoints:
    C21176875: Hand2DKeypoints = serde_field(rename="21176875:mono10bit")
    C21179183: Hand2DKeypoints = serde_field(rename="21179183:mono10bit")
    C21110305: Hand2DKeypoints = serde_field(rename="21110305:mono10bit")
    C21176623: Hand2DKeypoints = serde_field(rename="21176623:mono10bit")


@dataclass
class Assembly101Dataset:
    exo_video_readers: MultiVideoReader
    exo_pinhole_list: list[PinholeParameters]
    kpts_2d_json: Path
    kpts_3d_json: Path
    all_2d_kpts_exo: dict[int, Exo2DKeypoints] = field(default_factory=dict)
    all_3d_kpts: dict[int, Hand3DKeypoints] = field(default_factory=dict)

    def __post_init__(self):
        with open(self.kpts_2d_json) as f:
            all_2d_kpts: dict[str, dict[str, dict[str, list[list[float]]]]] = (
                json.loads(f.read())
            )

        # sort all_2d_landmarks by frame number
        all_2d_kpts = dict(sorted(all_2d_kpts.items(), key=lambda item: int(item[0])))

        for frame_num, kpts_2d_dict in all_2d_kpts.items():
            exo_kpts_2d_dict = {
                k: v for k, v in kpts_2d_dict.items() if k.startswith("C")
            }
            self.all_2d_kpts_exo[int(frame_num)] = from_dict(
                Exo2DKeypoints, exo_kpts_2d_dict
            )

        # Load 3D keypoints
        with open(self.kpts_3d_json) as f:
            all_3d_landmarks: dict[str, dict[str, list[list[float]]]] = json.loads(
                f.read()
            )
        # convert to Hand3DKeypoints for easier access
        all_3d_landmarks = dict(
            sorted(all_3d_landmarks.items(), key=lambda item: int(item[0]))
        )
        self.all_3d_kpts: dict[int, Hand3DKeypoints] = {
            int(k): from_dict(Hand3DKeypoints, v) for k, v in all_3d_landmarks.items()
        }

    def kpts_3d_to_column(
        self,
    ) -> tuple[
        Float32[ndarray, "num_frames 21 3"], Float32[ndarray, "num_frames 21 3"]
    ]:
        left_kpts_list = []
        right_kpts_list = []
        for frame_number, _ in enumerate(tqdm(self.all_3d_kpts)):
            keypoints: Hand3DKeypoints = self.all_3d_kpts[frame_number]
            left_kpts_list.append(keypoints.left)
            right_kpts_list.append(keypoints.right)

        # Concatenate keypoints from all frames vertically to get a (num_frames 21, 3) array.
        left_kpts_stack: Float32[ndarray, "_ 21 3"] = rearrange(
            left_kpts_list, "num_frames kpts dim -> num_frames kpts dim"
        )
        right_kpts_stack: Float32[ndarray, "_ 21 3"] = rearrange(
            right_kpts_list, "num_frames kpts dim -> num_frames kpts dim"
        )

        return left_kpts_stack, right_kpts_stack

    def kpts_2d_to_column(
        self,
    ) -> tuple[
        dict[str, Float32[ndarray, "_ 21 2"]], dict[str, Float32[ndarray, "_ 21 2"]]
    ]:
        camera_names = (
            Exo2DKeypoints.__annotations__.keys()
        )  # e.g., "C10395", "C10379", etc.
        left_cam_2d_arrays = {cam: [] for cam in camera_names}
        right_cam_2d_arrays = {cam: [] for cam in camera_names}

        for frame_number, _ in enumerate(tqdm(self.all_2d_kpts_exo)):
            exo_kpts_2d: Exo2DKeypoints = self.all_2d_kpts_exo[frame_number]
            for cam_name in camera_names:
                left_cam_2d_arrays[cam_name].append(getattr(exo_kpts_2d, cam_name).left)
                right_cam_2d_arrays[cam_name].append(
                    getattr(exo_kpts_2d, cam_name).right
                )

        for cam_name in camera_names:
            left_cam_2d_arrays[cam_name] = rearrange(
                left_cam_2d_arrays[cam_name],
                "num_frames kpts dim -> num_frames kpts dim",
            )
            right_cam_2d_arrays[cam_name] = rearrange(
                right_cam_2d_arrays[cam_name],
                "num_frames kpts dim -> num_frames kpts dim",
            )

        return left_cam_2d_arrays, right_cam_2d_arrays


def load_assembly101(
    root_directory: Path, example_name: str, encoding: Literal["av1", "h264"] = "av1"
) -> Assembly101Dataset:
    match encoding:
        case "av1":
            video_dir: Path = root_directory / "av1" / example_name
        case "h264":
            video_dir: Path = root_directory / "videos" / example_name

    landmarks_3d_path: Path = (
        root_directory
        / "assembly101_camera_and_hand_poses"
        / "landmarks3D"
        / f"{example_name}.json"
    )
    landmarks_2d_path: Path = (
        root_directory
        / "assembly101_camera_and_hand_poses"
        / "landmarks2D"
        / f"{example_name}.json"
    )
    extrinsics_fixed_path: Path = (
        root_directory
        / "assembly101_camera_and_hand_poses"
        / "camera_extrinsics_fixed"
        / f"{example_name}.json"
    )

    assembly_hands_annotation_path: Path = root_directory / "assembly-hands"
    train_assembly_hands_json: Path = (
        assembly_hands_annotation_path
        / "annotations"
        / "train"
        / "assemblyhands_train_exo_calib_v1-1.json"
    )

    assert video_dir.exists(), f"Video directory {video_dir} does not exist."
    assert landmarks_3d_path.exists(), (
        f"Landmarks 3D directory {landmarks_3d_path} does not exist."
    )
    assert landmarks_2d_path.exists(), (
        f"Landmarks 2D directory {landmarks_2d_path} does not exist."
    )
    assert extrinsics_fixed_path.exists(), (
        f"Extrinsics fixed directory {extrinsics_fixed_path} does not exist."
    )
    assert train_assembly_hands_json.exists(), (
        f"Train assembly hands json {train_assembly_hands_json} does not exist."
    )

    exo_video_files: list[Path] = sorted(
        [
            file
            for file in video_dir.iterdir()
            if file.is_file() and not file.name.startswith("HMC")
        ]
    )

    exo_video_readers: MultiVideoReader = MultiVideoReader(exo_video_files)
    exo_pinhole_list: list[PinholeParameters] = load_exo_cameras(
        extrinsics_fixed_path=extrinsics_fixed_path,
        train_assembly_hands_json=train_assembly_hands_json,
        height=exo_video_readers.height,
        width=exo_video_readers.width,
    )
    # sort by camera name
    exo_pinhole_list.sort(key=lambda x: x.name)

    assembly101_dataset = Assembly101Dataset(
        exo_video_readers=exo_video_readers,
        exo_pinhole_list=exo_pinhole_list,
        kpts_2d_json=landmarks_2d_path,
        kpts_3d_json=landmarks_3d_path,
    )

    return assembly101_dataset
