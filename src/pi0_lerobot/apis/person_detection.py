from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, Int, UInt8
from numpy import ndarray
from simplecv.camera_parameters import (
    PinholeParameters,
)
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from pi0_lerobot.custom_types import CustomPoints2D, CustomPoints3D
from pi0_lerobot.data.assembly101 import (
    Assembly101Dataset,
    load_assembly101,
)
from pi0_lerobot.multiview_pose_estimator import (
    MultiviewBodyTracker,
    MVOutput,
    projectN3,
)
from pi0_lerobot.rerun_log_utils import create_blueprint
from pi0_lerobot.skeletons.assembly_hand import HAND_ID2NAME, HAND_IDS, HAND_LINKS
from pi0_lerobot.skeletons.coco_17 import COCO_17_IDS, COCO_ID2NAME, COCO_LINKS
from pi0_lerobot.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


@dataclass
class VisualzeConfig:
    rr_config: RerunTyroConfig
    root_directory: Path = Path("data/assembly101-sample")
    example_name: str = (
        "nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345"
    )
    num_videos_to_log: Literal[4, 8] = 4
    log_extrapolated_keypoints: bool = False


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
                    info=rr.AnnotationInfo(
                        id=2, label="Triangulate", color=(0, 255, 255)
                    ),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name)
                        for id, name in COCO_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=3, label="Body", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name)
                        for id, name in COCO_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=4, label="extrap"),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name)
                        for id, name in COCO_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=5, label="t1"),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name)
                        for id, name in COCO_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=6, label="t2"),
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


def run_person_detection(config: VisualzeConfig):
    start_time: float = timer()
    parent_log_path: Path = Path("world")
    timeline: str = "video_time"
    rr.log("/", rr.ViewCoordinates.BUL, static=True)
    set_pose_annotation_context()

    # load data
    assembly101_data: Assembly101Dataset = load_assembly101(
        config.root_directory, config.example_name, load_2d=False, load_3d=False
    )
    exo_video_readers: MultiVideoReader = assembly101_data.exo_video_readers
    exo_gpu_decoders: list[VideoDecoder] = [
        VideoDecoder(exo_path, device="cuda", dimension_order="NHWC")
        for exo_path in assembly101_data.exo_video_readers.video_paths
    ]
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

    print(f"Time taken to load data: {timer() - start_time:.2f} seconds")

    cams_for_detection_idx: list[int] = [0, 2, 3, 5]

    projection_all_list: list[Float32[np.ndarray, "3 4"]] = []
    for exo_cam in exo_pinhole_list:
        projection_matrix: Float32[ndarray, "3 4"] = exo_cam.projection_matrix.astype(
            np.float32
        )
        projection_all_list.append(projection_matrix)

    Pall = np.array([P for P in projection_all_list])
    P_all_filtered: Float32[ndarray, "nViews 3 4"] = np.array(
        [projection_all_list[i] for i in cams_for_detection_idx]
    )

    # filter gpu decoders to only include the cameras we want to detect on
    exo_gpu_decoders = [exo_gpu_decoders[i] for i in cams_for_detection_idx]
    filtered_exo_pinhole_list = [exo_pinhole_list[i] for i in cams_for_detection_idx]

    min_num_frames: int = min(
        [decoder.metadata.num_frames for decoder in exo_gpu_decoders]
    )

    upper_body_filter_idx = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    # Create a boolean mask for all rows
    top_half_mask = np.isin(np.arange(17), upper_body_filter_idx)

    pose_tracker = MultiviewBodyTracker(
        Pall=P_all_filtered,
        mode="lightweight",
        backend="onnxruntime",
        device="cuda",
        filter_body_idxes=upper_body_filter_idx,
    )

    for ts_idx, timestamp in enumerate(tqdm(longest_timestamp[:min_num_frames])):
        rr.set_time_nanos(timeline="video_time", nanos=timestamp)
        bgr_list: list[UInt8[ndarray, "H W 3"]] = [
            decoder[ts_idx].cpu().numpy() for decoder in exo_gpu_decoders
        ]
        mv_output: MVOutput = pose_tracker(bgr_list)
        for cam_idx, (exo_pinhole, (bbox, kpts_2d, scores)) in enumerate(
            zip(filtered_exo_pinhole_list, mv_output, strict=True)
        ):
            cam_log_path: Path = (
                parent_log_path / exo_pinhole.name / "pinhole" / "video"
            )
            if bbox is None:
                continue

            vis_kpts_2d: ndarray = kpts_2d.copy()
            vis_scores_2d: ndarray = scores.copy()
            # filter to only include the desired keypoints
            vis_kpts_2d[~top_half_mask, :] = np.nan
            vis_scores_2d[~top_half_mask] = np.nan

            rr.log(
                f"{cam_log_path / 'bboxes'}",
                rr.Boxes2D(
                    array=bbox,
                    array_format=rr.Box2DFormat.XYXY,
                    colors=[(0, 0, 255)],
                ),
            )
            rr.log(
                f"{cam_log_path / 'pred_body'}",
                CustomPoints2D(
                    positions=vis_kpts_2d,
                    confidences=vis_scores_2d,
                    class_ids=3,
                    keypoint_ids=COCO_17_IDS,
                    show_labels=False,
                ),
            )
            if mv_output.uvc_extrap is not None and config.log_extrapolated_keypoints:
                uvc_extrap: Float64[ndarray, "num_kpts 3"] = mv_output.uvc_extrap[
                    cam_idx
                ]
                u_max: float = np.nanmax(uvc_extrap[:, 0])
                u_min: float = np.nanmin(uvc_extrap[:, 0])
                v_max: float = np.nanmax(uvc_extrap[:, 1])
                v_min: float = np.nanmin(uvc_extrap[:, 1])
                bbox_extrap: Float64[ndarray, "1 4"] = np.array(
                    [[u_min, v_min, u_max, v_max]]
                )
                rr.log(
                    f"{cam_log_path / 'extrap_bbox'}",
                    rr.Boxes2D(
                        array=bbox_extrap,
                        array_format=rr.Box2DFormat.XYXY,
                        colors=[(0, 255, 0)],
                    ),
                )
                rr.log(
                    f"{cam_log_path / 'extrap_body'}",
                    CustomPoints2D(
                        positions=mv_output.uvc_extrap[cam_idx][:, :2],
                        confidences=mv_output.uvc_extrap[cam_idx][:, 2],
                        class_ids=4,
                        keypoint_ids=COCO_17_IDS,
                        show_labels=False,
                    ),
                )

        vis_kpts_3d: ndarray = mv_output.kpts_3d.copy()
        vis_scores_3d: ndarray = mv_output.scores_3d.copy()
        # filter to only include the desired keypoints
        vis_kpts_3d[~top_half_mask, :] = np.nan
        vis_scores_3d[~top_half_mask] = np.nan

        rr.log(
            "wholebody",
            CustomPoints3D(
                positions=vis_kpts_3d,
                confidences=vis_scores_3d,
                class_ids=3,
                keypoint_ids=COCO_17_IDS,
                show_labels=False,
            ),
        )

        scores_expanded = mv_output.scores_3d[
            ..., None
        ]  # Add a dimension to match kpts_3d
        xyzc = np.concatenate([mv_output.kpts_3d, scores_expanded], axis=-1)

        mv_uvc_projected: Float32[ndarray, "n_views n_kpts 3"] = projectN3(
            xyzc, Pall
        ).astype(np.float32)
        for all_exo_pinhole, uvc_projected in zip(
            exo_pinhole_list, mv_uvc_projected, strict=True
        ):
            proj_log_path: Path = (
                parent_log_path / all_exo_pinhole.name / "pinhole" / "video"
            )
            vis_kpts_proj_2d: ndarray = uvc_projected[:, :2].copy()
            vis_scores_proj_2d: ndarray = uvc_projected[:, 2].copy()
            # filter to only include the desired keypoints
            vis_kpts_proj_2d[~top_half_mask, :] = np.nan
            vis_scores_proj_2d[~top_half_mask] = np.nan
            rr.log(
                f"{proj_log_path / 'proj_body'}",
                CustomPoints2D(
                    positions=vis_kpts_proj_2d,
                    confidences=vis_scores_proj_2d,
                    class_ids=2,
                    keypoint_ids=COCO_17_IDS,
                    show_labels=False,
                ),
            )

        kpts3d_list = [
            mv_output.kpts_3d_extrapolated,
            mv_output.kpts_3d_t1,
            mv_output.kpts_3d_t2,
        ]
        class_ids = [4, 5, 6]
        names = ["extrap", "t1", "t2"]
        for kpts3d, class_id, name in zip(kpts3d_list, class_ids, names, strict=True):
            if kpts3d is not None:
                # this has to happen and I don't know why
                kpts3d[~top_half_mask, :] = np.nan
                kpts3d[~top_half_mask] = np.nan
                if config.log_extrapolated_keypoints:
                    rr.log(
                        f"{name}",
                        rr.Points3D(
                            positions=kpts3d[:, :3],
                            class_ids=class_id,
                            keypoint_ids=COCO_17_IDS,
                            show_labels=False,
                        ),
                    )
