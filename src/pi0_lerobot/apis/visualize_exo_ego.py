from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Int
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.ego_dataset_configs import AnnotatedEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence, CamNameType, EgoLabels
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
    log_pinhole,
    log_video,
)
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


@dataclass
class VisualizeConfig:
    rr_config: RerunTyroConfig
    dataset: AnnotatedEgoDatasetUnion
    max_exo_videos_to_log: Literal[4, 8] = 4


def set_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
    )


def create_blueprint(
    *,
    ego_video_log_paths: list[Path] | None = None,
    exo_video_log_paths: list[Path] | None = None,
    max_exo_videos_to_log: Literal[4, 8] = 8,
) -> rrb.Blueprint:
    """Creates a Rerun blueprint for visualizing ego and exo-centric video streams.

    This function constructs a Rerun blueprint layout. It starts with a main 3D
    spatial view. If ego-centric video paths are provided, it adds a vertical
    panel on the right with a tab for each ego video. If exo-centric video
    paths are provided, it adds a horizontal panel at the bottom with a tab for
    each exo video.

    Args:
        ego_video_log_paths: Optional list of paths to ego-centric video logs.
            If provided, a vertical panel with tabs for each video's 2D view
            is added to the right of the main 3D view.
        exo_video_log_paths: Optional list of paths to exo-centric video logs.
            If provided, a horizontal panel with tabs for each video's 2D view
            is added below the main view.
        max_exo_videos_to_log: The maximum number of exo-centric videos to display
            in the blueprint. Defaults to 8.

    Returns:
        A `rrb.Blueprint` object defining the layout for the Rerun viewer.
    """
    main_view = rrb.Spatial3DView(
        origin="/",
    )

    if ego_video_log_paths is not None:
        ego_view = rrb.Vertical(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in ego_video_log_paths
            ]
        )
        main_view = rrb.Horizontal(
            contents=[main_view, ego_view],
            column_shares=[4, 1],
        )

    if exo_video_log_paths is not None:
        exo_view = rrb.Horizontal(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in exo_video_log_paths[:max_exo_videos_to_log]
            ]
        )
        main_view = rrb.Vertical(
            contents=[main_view, exo_view],
            row_shares=[4, 1],
        )

    contents = [main_view]

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def visualize_exo_ego(config: VisualizeConfig):
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    ego_sequence: BaseEgoSequence | None = exoego_sequence.ego_sequence
    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence

    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    ego_video_log_paths: list[Path] | None = None
    ego_timestamps: list[Int[ndarray, "num_frames"]] = []
    if ego_sequence is not None:
        ego_video_readers: MultiVideoReader = ego_sequence.ego_video_readers
        ego_video_files: list[Path] = ego_video_readers.video_paths
        ego_cam_dict: dict[CamNameType, list[PinholeParameters]] = ego_sequence.ego_cam_dict
        ego_cam_log_paths: list[Path] = [parent_log_path / ego_cam_name for ego_cam_name in ego_cam_dict]
        ego_video_log_paths = [cam_log_paths / "pinhole" / "video" for cam_log_paths in ego_cam_log_paths]

        for video_file, ego_video_log_path in zip(ego_video_files, ego_video_log_paths, strict=True):
            assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
            # Log video asset which is referred to by frame references.
            ego_timestamps_ns: Int[ndarray, "num_frames"] = log_video(  # noqa: UP037
                video_file, ego_video_log_path, timeline=timeline
            )
            ego_timestamps.append(ego_timestamps_ns)

    exo_video_log_paths: list[Path] | None = None
    if exo_sequence is not None:
        exo_video_readers: MultiVideoReader = exo_sequence.exo_video_readers
        exo_video_files: list[Path] = exo_video_readers.video_paths
        exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in exo_sequence.exo_cam_list]
        exo_video_log_paths = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

        # log stationary exo cameras and video assets
        for exo_cam in exo_sequence.exo_cam_list:
            cam_log_path: Path = parent_log_path / exo_cam.name
            log_pinhole(
                camera=exo_cam,
                cam_log_path=cam_log_path,
                image_plane_distance=exo_sequence.image_plane_distance,
                static=True,
            )

        for idx, (video_file, exo_video_log_path) in enumerate(zip(exo_video_files, exo_video_log_paths, strict=True)):
            if idx >= config.max_exo_videos_to_log:
                break
            assert video_file.suffix == ".mp4", f"Video file {video_file} is not an mp4."
            # Log video asset which is referred to by frame references.
            log_video(video_file, exo_video_log_path, timeline=timeline)

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=exo_video_log_paths,
        ego_video_log_paths=ego_video_log_paths,
        max_exo_videos_to_log=config.max_exo_videos_to_log,
    )
    rr.send_blueprint(blueprint)

    if ego_sequence is not None and ego_timestamps:
        # Find the timestamp list with the maximum length.
        shortest_timestamp: Int[ndarray, "num_frames"] = min(ego_timestamps, key=len)  # noqa: UP037
        assert len(shortest_timestamp) == len(ego_sequence), (
            f"Length of timestamps {len(shortest_timestamp)} and sequence {len(ego_sequence)} do not match"
        )

        ego_labels: EgoLabels = ego_sequence.ego_labels
        xyzc_stack: Float[ndarray, "num_frames 133 4"] = ego_labels.xyzc_stack

        for ts_idx, ts in enumerate(shortest_timestamp):
            rr.set_time_nanos(timeline=timeline, nanos=ts)
            ego_cam_param_list: list[PinholeParameters]
            for cam_name, ego_cam_param_list in ego_cam_dict.items():
                try:
                    ego_cam_param: PinholeParameters = ego_cam_param_list[ts_idx]
                except IndexError:
                    print(f"Index {ts_idx} out of bounds for camera {cam_name}")
                    continue
                # get the cam log path that corresponds to the camera name, check cam_log_paths if it exists
                cam_name: CamNameType = ego_cam_param.name
                cam_log_matches: list[Path] = [
                    cam_log_path for cam_log_path in ego_cam_log_paths if cam_name in cam_log_path.name
                ]
                if not cam_log_matches:
                    raise ValueError(f"Camera name {cam_name} not found in all_logs: {ego_cam_log_paths}")
                cam_log_path = cam_log_matches[0]
                log_pinhole(
                    camera=ego_cam_param,
                    cam_log_path=cam_log_path,
                    image_plane_distance=ego_sequence.image_plane_distance,
                    static=False,
                )

                # Log the 3D keypoints
                xyz: Float[ndarray, "133 3"] = xyzc_stack[
                    ts_idx, ..., :3
                ]  # Get the keypoints for the current timestamp and camera
                rr.log(
                    f"{parent_log_path}/keypoints",
                    rr.Points3D(
                        positions=xyz,  # Remove the view dimension
                        colors=(0, 255, 0),  # Assuming a default color for the keypoints
                        class_ids=0,
                        keypoint_ids=COCO_133_IDS,
                        show_labels=False,
                    ),
                )
