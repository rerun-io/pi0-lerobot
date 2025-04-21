from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int
from numpy import ndarray
from simplecv.apis.view_exoego_data import log_exo_ego_sequence_batch
from simplecv.data.exoego.assembly_101 import Assembely101Sequence
from simplecv.data.exoego.base_exo_ego import BaseExoEgoSequence
from simplecv.data.exoego.hocap import HOCapSequence, SubjectIDs
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_io import MultiVideoReader

from pi0_lerobot.rerun_log_utils import create_blueprint, log_mano_batch

np.set_printoptions(suppress=True)


@dataclass
class VisualzeConfig:
    rr_config: RerunTyroConfig
    dataset: Literal["hocap", "assembly101"] = "hocap"
    root_directory: Path = Path("data/hocap/sample")
    subject_id: SubjectIDs | None = "8"
    sequence_name: str = "20231024_180733"
    num_videos_to_log: Literal[4, 8] = 8
    log_depths: bool = False
    send_as_batch: bool = True


def set_pose_annotation_context(sequence: BaseExoEgoSequence) -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Left Hand", color=(255, 0, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in sequence.hand_id2name.items()
                    ],
                    keypoint_connections=sequence.hand_links,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="Right Hand", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in sequence.hand_id2name.items()
                    ],
                    keypoint_connections=sequence.hand_links,
                ),
            ]
        ),
        static=True,
    )


def visualize_exo_ego(config: VisualzeConfig):
    start_time: float = timer()

    match config.dataset:
        case "hocap":
            sequence: HOCapSequence = HOCapSequence(
                data_path=config.root_directory,
                sequence_name=config.sequence_name,
                subject_id=config.subject_id,
                load_labels=True,
            )
        case "assembly101":
            sequence: Assembely101Sequence = Assembely101Sequence(
                data_path=config.root_directory,
                sequence_name=config.sequence_name,
                subject_id=None,
                load_labels=True,
            )

    set_pose_annotation_context(sequence)
    rr.log("/", sequence.world_coordinate_system, static=True)

    parent_log_path = Path("world")
    timeline: str = "video_time"

    exo_video_readers: MultiVideoReader = sequence.exo_video_readers
    exo_video_files: list[Path] = exo_video_readers.video_paths
    exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in sequence.exo_cam_list]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

    blueprint: rrb.Blueprint = create_blueprint(exo_video_log_paths, num_videos_to_log=config.num_videos_to_log)
    rr.send_blueprint(blueprint)

    # log stationary exo cameras and video assets
    for exo_cam in sequence.exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        image_plane_distance: float = 0.1 if config.dataset == "hocap" else 100.0
        log_pinhole(
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=image_plane_distance,
            static=True,
        )

    all_timestamps: list[Int[ndarray, "num_frames"]] = []  # noqa: UP037
    for video_file, video_log_path in zip(exo_video_files, exo_video_log_paths, strict=True):
        assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
        # Log video asset which is referred to by frame references.
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(  # noqa: UP037
            video_file, video_log_path, timeline=timeline
        )
        all_timestamps.append(frame_timestamps_ns)

    # Find the timestamp list with the maximum length.
    shortest_timestamp: Int[ndarray, "num_frames"] = min(all_timestamps, key=len)  # noqa: UP037
    assert len(shortest_timestamp) == len(sequence), (
        f"Length of timestamps {len(shortest_timestamp)} and sequence {len(sequence)} do not match"
    )

    log_exo_ego_sequence_batch(
        sequence,
        shortest_timestamp=shortest_timestamp,
        parent_log_path=parent_log_path,
        timeline=timeline,
        log_depth=config.log_depths,
    )
    if sequence.exo_batch_data.mano_stack:
        log_mano_batch(
            sequence,
            shortest_timestamp=shortest_timestamp,
            timeline=timeline,
            mano_root_dir=Path("data/mano_clean"),
        )
    print(f"Time taken to load data: {timer() - start_time:.2f} seconds")
