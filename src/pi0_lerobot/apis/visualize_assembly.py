from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Float32, Int
from numpy import ndarray
from simplecv.camera_parameters import (
    PinholeParameters,
)
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm

from pi0_lerobot.data.assembly101 import (
    Assembly101Dataset,
    Exo2DKeypoints,
    Hand3DKeypoints,
    load_assembly101,
)
from pi0_lerobot.rerun_log_utils import create_blueprint
from pi0_lerobot.skeletons.assembly_hand import HAND_ID2NAME, HAND_IDS, HAND_LINKS
from pi0_lerobot.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


@dataclass
class VisualzeConfig:
    rr_config: RerunTyroConfig
    root_directory: Path = Path("data/assembly101-sample")
    example_name: str = (
        # "nusar-2021_action_both_9051-b03a_9051_user_id_2021-02-22_114140"
        "nusar-2021_action_both_9015-b05b_9015_user_id_2021-02-02_161800"
    )
    encode_format: Literal["av1", "h264"] = "av1"
    use_columns: bool = True
    num_videos_to_log: Literal[4, 8] = 4


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
            ]
        ),
        static=True,
    )


def log_video(
    video_path: Path, video_log_path: Path, timeline: str = "video_time"
) -> Int[ndarray, "num_frames"]:
    """
    Logs a video asset and its frame timestamps.

    Parameters:
    video_path (Path): The path to the video file.
    video_log_path (Path): The path where the video log will be saved.

    Returns:
    None
    """
    # Log video asset which is referred to by frame references.
    video_asset = rr.AssetVideo(path=video_path)
    rr.log(str(video_log_path), video_asset, static=True)

    # Send automatically determined video frame timestamps.
    frame_timestamps_ns: Int[ndarray, "num_frames"] = (  # noqa: UP037
        video_asset.read_frame_timestamps_ns()
    )
    rr.send_columns(
        f"{video_log_path}",
        # Note timeline values don't have to be the same as the video timestamps.
        indexes=[rr.TimeNanosColumn(timeline, frame_timestamps_ns)],
        columns=rr.VideoFrameReference.columns_nanoseconds(frame_timestamps_ns),
    )
    return frame_timestamps_ns


def visualize_data(config: VisualzeConfig):
    start_time: float = timer()
    parent_log_path: Path = Path("world")
    timeline: str = "video_time"
    rr.log("/", rr.ViewCoordinates.BUL, static=True)
    set_pose_annotation_context()

    # load data
    assembly101_data: Assembly101Dataset = load_assembly101(
        config.root_directory, config.example_name, encoding=config.encode_format
    )
    exo_video_readers: MultiVideoReader = assembly101_data.exo_video_readers
    exo_pinhole_list: list[PinholeParameters] = assembly101_data.exo_pinhole_list
    exo_video_files: list[Path] = exo_video_readers.video_paths

    # log pinhole parameters
    for cam_params in exo_pinhole_list:
        log_pinhole(
            cam_params,
            cam_log_path=parent_log_path / cam_params.name,
            image_plane_distance=100.0,
            static=True,
        )

    all_timestamps: list[Int[ndarray, "num_frames"]] = []  # noqa: UP037
    exo_video_log_paths: list[Path] = [
        parent_log_path / video_file.stem.split("_")[0] / "pinhole" / "video"
        for video_file in exo_video_files
    ]

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=exo_video_log_paths,
        num_videos_to_log=config.num_videos_to_log,
    )
    rr.send_blueprint(blueprint=blueprint)

    for video_file, video_log_path in zip(
        exo_video_files, exo_video_log_paths, strict=True
    ):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(  # noqa: UP037
            video_file, video_log_path, timeline=timeline
        )
        all_timestamps.append(frame_timestamps_ns)

    # Find the timestamp list with the maximum length.
    longest_timestamp: Int[ndarray, "num_frames"] = max(all_timestamps, key=len)  # noqa: UP037

    if config.use_columns:
        # much more efficient to use columns for keypoints, but slightly more complex and verbose
        num_send: int = len(longest_timestamp)

        left_kpts_stack, right_kpts_stack = assembly101_data.kpts_3d_to_column()
        left_kpts_stack: Float32[ndarray, "num_frames 21 3"] = left_kpts_stack[
            :num_send
        ]
        right_kpts_stack: Float32[ndarray, "num_frames 21 3"] = right_kpts_stack[
            :num_send
        ]

        for side, color, class_id, kpts_stack in (
            ("left", (255, 0, 0), 0, left_kpts_stack),
            ("right", (0, 255, 0), 1, right_kpts_stack),
        ):
            num_send = min(num_send, kpts_stack.shape[0])
            rr.log(
                side,
                rr.Points3D.from_fields(
                    colors=color,
                    class_ids=class_id,
                    keypoint_ids=HAND_IDS,
                    show_labels=False,
                ),
                static=True,
            )

            rr.send_columns(
                side,
                indexes=[rr.TimeNanosColumn(timeline, longest_timestamp[0:num_send])],
                columns=[
                    *rr.Points3D.columns(
                        positions=rearrange(
                            kpts_stack,
                            "num_frames kpts dim -> (num_frames kpts) dim",
                        ),
                    ).partition(lengths=[21] * num_send),
                ],
            )

        left_kpts_2d_stack, right_kpts_2d_stack = assembly101_data.kpts_2d_to_column()
        # filter to only number of frames to send
        left_kpts_2d_stack: dict[str, Float32[ndarray, "num_frames 21 2"]] = {
            cam_name: left_kpts_2d_stack[cam_name][:num_send]
            for cam_name in left_kpts_2d_stack
        }
        right_kpts_2d_stack: dict[str, Float32[ndarray, "num_frames 21 2"]] = {
            cam_name: right_kpts_2d_stack[cam_name][:num_send]
            for cam_name in right_kpts_2d_stack
        }

        for exo_cam in exo_pinhole_list:
            cam_name: str = exo_cam.name
            # Rearrange the keypoints to be in the format (num_frames*21, 3)
            left_kpts_2d_stack[cam_name] = rearrange(
                left_kpts_2d_stack[cam_name],
                "num_frames kpts dim -> (num_frames kpts) dim",
            )
            right_kpts_2d_stack[cam_name] = rearrange(
                right_kpts_2d_stack[cam_name],
                "num_frames kpts dim -> (num_frames kpts) dim",
            )

            for side, color, class_id in (
                ("left", (255, 0, 0), 0),
                ("right", (0, 255, 0), 1),
            ):
                rr.log(
                    f"{parent_log_path / cam_name / 'pinhole' / 'video' / side}",
                    rr.Points2D.from_fields(
                        colors=color,
                        class_ids=class_id,
                        keypoint_ids=HAND_IDS,
                        show_labels=False,
                    ),
                    static=True,
                )

                rr.send_columns(
                    f"{parent_log_path / cam_name / 'pinhole' / 'video' / side}",
                    indexes=[
                        rr.TimeNanosColumn(timeline, longest_timestamp[0:num_send])
                    ],
                    columns=[
                        *rr.Points2D.columns(
                            positions=left_kpts_2d_stack[cam_name]
                            if side == "left"
                            else right_kpts_2d_stack[cam_name],
                        ).partition(lengths=[21] * num_send),
                    ],
                )
    else:
        for frame_number, timestamp in enumerate(tqdm(longest_timestamp)):
            try:
                rr.set_time_nanos(timeline=timeline, nanos=timestamp)

                keypoints: Hand3DKeypoints = assembly101_data.all_3d_kpts[frame_number]
                exo_kpts_2d: Exo2DKeypoints = assembly101_data.all_2d_kpts_exo[
                    frame_number
                ]

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
                        kpts2d: Float32[ndarray, "21 2"] = exo_kpts_2d.__getattribute__(
                            exo_pinhole.name
                        ).__getattribute__(side)
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

    print(f"Time taken to load data: {timer() - start_time:.2f} seconds")
