import json
from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, Int, UInt8
from numpy import ndarray
from rtmlib import Body
from serde import field, from_dict, serde
from simplecv.camera_parameters import (
    Extrinsics,
    Intrinsics,
    PinholeParameters,
)
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm

from pi0_lerobot.skeletons.assembly_hand import HAND_ID2NAME, HAND_IDS, HAND_LINKS
from pi0_lerobot.skeletons.coco_17 import COCO_17_IDS, COCO_ID2NAME, COCO_LINKS
from pi0_lerobot.video_io import VideoReader

np.set_printoptions(suppress=True)

BgrImageType = TypeVar("BgrImageType", bound=UInt8[ndarray, "H W 3"])
RgbImageType = TypeVar("RgbImageType", bound=UInt8[ndarray, "H W 3"])


@dataclass
class VisualzeConfig:
    rr_config: RerunTyroConfig
    root_directory: Path = Path("/mnt/12tbdrive/data/assembly101-new")
    example_name: str = (
        "nusar-2021_action_both_9051-b03a_9051_user_id_2021-02-22_114140"
    )


@serde
class HandKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to field "left"
    left: Float32[ndarray, "21 3"] = field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 3"] = field(rename="1")


@serde
class ExoExtriCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to field "left"
    C10404: Float32[ndarray, "4 4"] = field(rename="C10404:rgb")
    C10118: Float32[ndarray, "4 4"] = field(rename="C10118:rgb")
    C10119: Float32[ndarray, "4 4"] = field(rename="C10119:rgb")
    C10095: Float32[ndarray, "4 4"] = field(rename="C10095:rgb")
    C10379: Float32[ndarray, "4 4"] = field(rename="C10379:rgb")
    C10395: Float32[ndarray, "4 4"] = field(rename="C10395:rgb")
    C10115: Float32[ndarray, "4 4"] = field(rename="C10115:rgb")
    C10390: Float32[ndarray, "4 4"] = field(rename="C10390:rgb")


@serde
class EgoCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to field "left"
    C21176875: Float32[ndarray, "4 4"] = field(rename="21176875:mono10bit")
    C21179183: Float32[ndarray, "4 4"] = field(rename="21179183:mono10bit")
    C21110305: Float32[ndarray, "4 4"] = field(rename="21110305:mono10bit")
    C21176623: Float32[ndarray, "4 4"] = field(rename="21176623:mono10bit")


class MultiVideoReader:
    def __init__(self, video_paths: list[Path]) -> None:
        # check that all video_paths are valid
        for video_path in video_paths:
            assert video_path.exists(), f"{video_path} does not exist"

        self.video_readers: list[VideoReader] = [
            VideoReader(str(video_path)) for video_path in video_paths
        ]

        assert all(
            reader.height == self.video_readers[0].height
            and reader.width == self.video_readers[0].width
            for reader in self.video_readers
        )

    @property
    def height(self) -> int:
        return self.video_readers[0].height

    @property
    def width(self) -> int:
        return self.video_readers[0].width

    def __len__(self) -> int:
        # Use minimum length to ensure safe iteration
        return min(len(reader) for reader in self.video_readers)

    def __iter__(self) -> Generator[list[BgrImageType] | None, None, None]:
        while True:
            bgr_list: list[BgrImageType] = []
            for reader in self.video_readers:
                bgr_image: BgrImageType | None = reader.read()
                match bgr_image:
                    case _ if bgr_image is not None:
                        bgr_list.append(bgr_image)
                    case None:
                        return
            yield bgr_list

    def __getitem__(self, idx: int) -> list[BgrImageType]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return [reader[idx] for reader in self.video_readers]


def project3D(kp3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """

    kp3d: [21, 4] [x, y, z, conf]
    P: [3, 4] (projection matrix - includes extrensic (R, t) and intrinsic (K))

    return kp2d: [u,v, ?]
    """
    if kp3d.shape[1] == 3:
        kp3d = np.concatenate([kp3d, np.ones((kp3d.shape[0], 1))], axis=1)
    assert kp3d.shape[1] == 4
    kp2d = P @ kp3d.T
    kp2d = kp2d[:2, :] / kp2d[2:, :]
    return kp2d.T


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


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def set_pose_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Left Hand", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name)
                        for id, name in HAND_ID2NAME.items()
                    ],
                    keypoint_connections=HAND_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="Right Hand", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name)
                        for id, name in HAND_ID2NAME.items()
                    ],
                    keypoint_connections=HAND_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=3, label="Body", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name)
                        for id, name in COCO_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_LINKS,
                ),
            ]
        ),
        static=True,
    )


def create_blueprint(exo_video_log_paths: list[Path]) -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=[
                rrb.Vertical(
                    contents=[
                        rrb.Spatial3DView(
                            origin="/",
                        ),
                        # take the first 4 video files
                        rrb.Horizontal(
                            contents=[
                                rrb.Tabs(
                                    rrb.Spatial2DView(origin=f"{video_log_path}"),
                                    rrb.Spatial2DView(
                                        origin=f"{video_log_path}".replace(
                                            "video", "image"
                                        )
                                    ),
                                    active_tab=0,
                                )
                                for video_log_path in exo_video_log_paths[:4]
                            ]
                        ),
                    ],
                    row_shares=[3, 1],
                ),
                # do the last 4 videos
                rrb.Vertical(
                    contents=[
                        rrb.Tabs(
                            rrb.Spatial2DView(origin=f"{video_log_path}"),
                            rrb.Spatial2DView(
                                origin=f"{video_log_path}".replace("video", "image")
                            ),
                            active_tab=0,
                        )
                        for video_log_path in exo_video_log_paths[4:]
                    ]
                ),
            ],
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def visualize_data(config: VisualzeConfig):
    parent_log_path: Path = Path("world")
    rr.log("/", rr.ViewCoordinates.BUL, static=True)
    set_pose_annotation_context()
    video_dir: Path = config.root_directory / "av1" / config.example_name

    landmarks_3d_path: Path = (
        config.root_directory
        / "assembly101_camera_and_hand_poses"
        / "landmarks3D"
        / f"{config.example_name}.json"
    )
    landmarks_2d_path: Path = (
        config.root_directory
        / "assembly101_camera_and_hand_poses"
        / "landmarks2D"
        / f"{config.example_name}.json"
    )
    extrinsics_fixed_path: Path = (
        config.root_directory
        / "assembly101_camera_and_hand_poses"
        / "camera_extrinsics_fixed"
        / f"{config.example_name}.json"
    )

    assembly_hands_annotation_path: Path = config.root_directory / "assembly-hands"
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
    # log pinhole parameters
    for cam_params in exo_pinhole_list:
        log_pinhole(
            cam_params,
            cam_log_path=parent_log_path / cam_params.name,
            image_plane_distance=100.0,
            static=True,
        )

    all_timestamps = []
    exo_video_log_paths: list[Path] = [
        parent_log_path / video_file.stem.split("_")[0] / "pinhole" / "video"
        for video_file in exo_video_files
    ]

    blueprint: rrb.Blueprint = create_blueprint(exo_video_log_paths=exo_video_log_paths)
    rr.send_blueprint(blueprint=blueprint)

    for video_file, video_log_path in zip(
        exo_video_files, exo_video_log_paths, strict=True
    ):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        video_asset = rr.AssetVideo(path=video_file)
        rr.log(f"{video_log_path}", video_asset, static=True)

        # Send automatically determined video frame timestamps.
        frame_timestamps_ns: Int[ndarray, "num_frames"] = (  # noqa: UP037
            video_asset.read_frame_timestamps_ns()
        )
        rr.send_columns(
            f"{video_log_path}",
            # Note timeline values don't have to be the same as the video timestamps.
            indexes=[rr.TimeNanosColumn("video_time", frame_timestamps_ns)],
            columns=rr.VideoFrameReference.columns_nanoseconds(frame_timestamps_ns),
        )
        all_timestamps.append(frame_timestamps_ns)

    # Find the timestamp list with the maximum length.
    longest_timestamp = max(all_timestamps, key=len)

    with open(landmarks_3d_path) as f:
        all_3d_landmarks = json.load(f)

    body_estimator = Body(
        mode="balanced",
        backend="onnxruntime",
        device="cuda",
    )

    desired_idx = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    # Create a boolean mask for all rows
    top_half_mask = np.isin(np.arange(17), desired_idx)

    cams_for_wholebody: list[str] = ["C10095", "C10118", "C10119", "C10390"]
    # idx for cam names
    cams_for_wholebody_idx: list[int] = [
        idx
        for idx, exo_cam in enumerate(exo_pinhole_list)
        if any(cam_name in exo_cam.name for cam_name in cams_for_wholebody)
    ]

    projection_all_list: list[Float32[np.ndarray, "3 4"]] = []
    for cam_idx, exo_cam in enumerate(exo_pinhole_list):
        if cam_idx not in cams_for_wholebody_idx:
            continue
        projection_matrix: Float32[ndarray, "3 4"] = exo_cam.projection_matrix.astype(
            np.float32
        )
        projection_all_list.append(projection_matrix)

    P_all: Float32[ndarray, "nViews 3 4"] = np.array([P for P in projection_all_list])

    for ts_idx, timestamp in enumerate(tqdm(longest_timestamp)):
        try:
            rr.set_time_nanos(timeline="video_time", nanos=timestamp)
            bgr_list: list[BgrImageType] = exo_video_readers[ts_idx]
            wb_xyc_list: list[Float32[ndarray, "num_keypoints 3"]] = []

            for cam_idx, (bgr, exo_pinhole) in enumerate(
                zip(bgr_list, exo_pinhole_list, strict=True)
            ):
                if cam_idx not in cams_for_wholebody_idx:
                    continue
                output: tuple[
                    Float64[ndarray, "num_dets num_kpts 2"],
                    Float32[ndarray, "num_dets num_kpts"],
                ] = body_estimator(bgr)
                wb_xy: Float32[ndarray, "num_dets num_kpts 2"] = output[0].astype(
                    np.float32
                )
                scores: Float32[ndarray, "num_dets num_kpts"] = output[1]
                wb_xyc: Float32[ndarray, "num_dets num_kpts 3"] = np.concatenate(
                    [wb_xy, scores[:, :, None]], axis=2
                )
                wb_xyc: Float32[ndarray, "num_kpts 3"] = wb_xyc[0]
                wb_xyc_list.append(wb_xyc)

                # Set rows that are not desired to NaN
                wb_xyc[~top_half_mask, :] = np.nan

                rr.log(
                    f"{parent_log_path / exo_pinhole.name / 'pinhole' / 'video' / 'pred_body'}",
                    rr.Points2D(
                        positions=wb_xyc[:, :2],
                        class_ids=3,
                        keypoint_ids=COCO_17_IDS,
                        show_labels=False,
                    ),
                )

            multiview_wb_xyc: Float32[ndarray, "n_views num_kpts 3"] = np.stack(
                wb_xyc_list
            )

            wb_xyzc: Float64[ndarray, "num_kpts 4"] = batch_triangulate(
                keypoints_2d=multiview_wb_xyc,
                projection_matrices=P_all,
                min_views=3,
            )
            # filter keypoints_3d by confidence
            wb_xyzc[wb_xyzc[:, -1] < 0.5] = np.nan
            # filter keypoints not in the top half of the body
            wb_xyc[~top_half_mask, :] = np.nan

            rr.log(
                "wholebody",
                rr.Points3D(
                    wb_xyzc[:, :3],
                    class_ids=3,
                    keypoint_ids=COCO_17_IDS,
                    show_labels=False,
                ),
            )

            keypoints_dict: dict = all_3d_landmarks[str(ts_idx)]
            keypoints: HandKeypoints = from_dict(HandKeypoints, keypoints_dict)
            for side, color, class_id in (
                ("left", (255, 0, 0), 0),
                ("right", (0, 255, 0), 1),
            ):
                kpts_3d: Float32[ndarray, "21 3"] = getattr(keypoints, side)
                rr.log(
                    side,
                    rr.Points3D(
                        kpts_3d,
                        colors=color,
                        keypoint_ids=HAND_IDS,
                        class_ids=class_id,
                        show_labels=False,
                    ),
                )
                for exo_pinhole in exo_pinhole_list:
                    kpts2d: Float64[ndarray, "21 2"] = project3D(
                        kpts_3d, exo_pinhole.projection_matrix.astype(np.float32)
                    )

                    rr.log(
                        f"{parent_log_path / exo_pinhole.name / 'pinhole' / 'video' / side}",
                        rr.Points2D(
                            kpts2d,
                            colors=color,
                            keypoint_ids=HAND_IDS,
                            class_ids=class_id,
                            show_labels=False,
                        ),
                    )
        except KeyError:
            continue
