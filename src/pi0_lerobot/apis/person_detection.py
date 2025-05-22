from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Float32, Float64, Int, UInt8
from numpy import ndarray
from simplecv.apis.view_exoego_data import log_exo_ego_sequence_batch
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exoego.assembly_101 import Assembly101Sequence
from simplecv.data.exoego.base_exo_ego import BaseExoEgoSequence
from simplecv.data.exoego.hocap import HOCapSequence, SubjectIDs
from simplecv.data.exoego.multicam import MulticamSequence
from simplecv.ops.triangulate import projectN3
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_io import MultiVideoReader
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from pi0_lerobot.custom_types import CustomPoints2D, CustomPoints3D
from pi0_lerobot.mano.kinematic_hand_optim_jax import (
    JointAndScaleOptimization,
    LossWeights,
    OptimizationResults,
    proj_3d_vectorized,
)
from pi0_lerobot.multiview_pose_estimator import (
    MultiviewBodyTracker,
    MVOutput,
)
from pi0_lerobot.rerun_log_utils import create_blueprint
from pi0_lerobot.skeletons.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS

np.set_printoptions(suppress=True)


def new_annotation_context(sequence: BaseExoEgoSequence) -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Right Hand", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in sequence.hand_id2name.items()
                    ],
                    keypoint_connections=sequence.hand_links,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="Left Hand", color=(255, 0, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in sequence.hand_id2name.items()
                    ],
                    keypoint_connections=sequence.hand_links,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=2, label="Wholebody 133", color=(0, 255, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=3, label="Projected Wholebody 133", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
    )


@dataclass
class VisualzeConfig:
    rr_config: RerunTyroConfig
    dataset: Literal["hocap", "assembly101", "multicam"] = "hocap"
    root_directory: Path = Path("data/hocap/sample")
    subject_id: SubjectIDs | None = "8"
    sequence_name: str = "20231024_180733"
    num_videos_to_log: Literal[4, 8] = 8
    log_labels: bool = True
    log_depths: bool = False
    log_extrapolated_keypoints: bool = False
    send_as_batch: bool = True


def run_person_detection(config: VisualzeConfig):
    start_time: float = timer()
    parent_log_path: Path = Path("world")
    timeline: str = "video_time"

    load_labels: bool = config.log_labels
    # load data
    match config.dataset:
        case "hocap":
            sequence: HOCapSequence = HOCapSequence(
                data_path=config.root_directory,
                sequence_name=config.sequence_name,
                subject_id=config.subject_id,
                load_labels=load_labels,
            )
        case "assembly101":
            sequence: Assembly101Sequence = Assembly101Sequence(
                data_path=config.root_directory,
                sequence_name=config.sequence_name,
                subject_id=None,
                load_labels=load_labels,
            )
        case "multicam":
            # multicam does not have labels, raise error if load_labels is True
            if load_labels:
                raise ValueError("Multicam dataset does not have labels.")
            sequence: MulticamSequence = MulticamSequence(
                data_path=config.root_directory,
                sequence_name=config.sequence_name,
                subject_id=None,
                load_labels=load_labels,
            )
    # set_pose_annotation_context(sequence=sequence)
    new_annotation_context(sequence)
    rr.log("/", sequence.world_coordinate_system, static=True)

    exo_video_readers: MultiVideoReader = sequence.exo_video_readers
    exo_video_files: list[Path] = exo_video_readers.video_paths
    exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in sequence.exo_cam_list]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]
    exo_gpu_decoders: list[VideoDecoder] = [
        VideoDecoder(exo_path, device="cuda", dimension_order="NHWC") for exo_path in exo_video_files
    ]

    blueprint: rrb.Blueprint = create_blueprint(exo_video_log_paths, num_videos_to_log=config.num_videos_to_log)
    rr.send_blueprint(blueprint)

    match config.dataset:
        case "hocap":
            image_plane_distance = 0.1
        case "assembly101":
            image_plane_distance = 100.0
        case "multicam":
            image_plane_distance = 25.0
        case _:
            # Default or error case, though Literal type hint should prevent this
            raise ValueError(f"Unexpected dataset value: {config.dataset}")

    # log pinhole parameters
    for exo_cam in sequence.exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        log_pinhole(
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=image_plane_distance,
            static=True,
        )

    all_timestamps: list[Int[ndarray, "num_frames"]] = []  # noqa: UP037

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=exo_video_log_paths,
        num_videos_to_log=config.num_videos_to_log,
    )
    rr.send_blueprint(blueprint=blueprint)

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

    print(f"Time taken to load data: {timer() - start_time:.2f} seconds")

    if load_labels:
        log_exo_ego_sequence_batch(
            sequence=sequence,
            shortest_timestamp=shortest_timestamp,
            parent_log_path=parent_log_path,
            timeline=timeline,
            log_depth=False,
        )

    if config.dataset == "hocap":
        cams_for_detection_idx: list[int] = [1, 3, 5, 6]
    else:
        cams_for_detection_idx: list[int] = [0, 1, 2, 3]

    projection_all_list: list[Float32[np.ndarray, "3 4"]] = []
    for exo_cam in sequence.exo_cam_list:
        projection_matrix: Float32[ndarray, "3 4"] = exo_cam.projection_matrix.astype(np.float32)
        projection_all_list.append(projection_matrix)

    Pall: Float32[np.ndarray, "n_views 3 4"] = np.array([P for P in projection_all_list])

    # filter gpu decoders to only include the cameras we want to detect on
    exo_gpu_decoders: list[VideoDecoder] = [exo_gpu_decoders[i] for i in cams_for_detection_idx]
    filtered_exo_pinhole_list: list[PinholeParameters] = [sequence.exo_cam_list[i] for i in cams_for_detection_idx]
    exo_video_files: list[Path] = [exo_video_readers.video_paths[i] for i in cams_for_detection_idx]
    exo_video_readers: MultiVideoReader = MultiVideoReader(video_paths=exo_video_files)

    min_num_frames: int = min([decoder.metadata.num_frames for decoder in exo_gpu_decoders])

    # 11/12 are the hips
    upper_body_filter_idx = np.array([5, 6, 7, 8, 9, 10])  # , 11, 12])
    face_idx = np.arange(23, 91)
    left_hand_idx = np.arange(91, 112)
    right_hand_idx = np.arange(112, 133)
    wb_upper_body_filter_idx = np.concatenate([upper_body_filter_idx, face_idx, left_hand_idx, right_hand_idx])

    # Create a boolean mask for all rows
    top_half_mask = np.isin(np.arange(17), upper_body_filter_idx)
    top_half_mask = np.isin(np.arange(133), wb_upper_body_filter_idx)

    pose_tracker = MultiviewBodyTracker(
        mode="wholebody",
        backend="onnxruntime",
        device="cuda",
        filter_body_idxes=wb_upper_body_filter_idx,
        cams_for_detection_idx=cams_for_detection_idx,
        perform_tracking=True,
    )

    # kinematic hand skeleton and optimization
    loss_weights = LossWeights(
        keypoint_2d=0.01,
        depth=0.0,
        temp=0.0,
    )

    for ts_idx, timestamp in enumerate(tqdm(shortest_timestamp[:min_num_frames])):
        rr.set_time_nanos(timeline="video_time", nanos=timestamp)
        # this breaks with HDR videos
        # bgr_list: list[UInt8[ndarray, "H W 3"]] = [decoder[ts_idx].cpu().numpy() for decoder in exo_gpu_decoders]
        bgr_list: list[UInt8[ndarray, "H W 3"]] = exo_video_readers[ts_idx]
        mv_output: MVOutput = pose_tracker(bgr_list, Pall)

        # log 2d keypoint outputs
        for cam_idx, (exo_pinhole, (xyxy, uv, uv_scores)) in enumerate(
            zip(filtered_exo_pinhole_list, mv_output, strict=True)
        ):
            cam_log_path: Path = parent_log_path / exo_pinhole.name / "pinhole" / "video"
            if xyxy is None:
                continue

            vis_uv: ndarray = uv.copy()
            vis_scores_uv: ndarray = uv_scores.copy()
            # filter to only include the desired keypoints
            vis_uv[~top_half_mask, :] = np.nan
            vis_scores_uv[~top_half_mask] = np.nan
            # filter out keypoints that are below the threshold (0.75)
            vis_uv[vis_scores_uv < 0.75, :] = np.nan

            rr.log(
                f"{cam_log_path / 'bboxes'}",
                rr.Boxes2D(
                    array=xyxy,
                    array_format=rr.Box2DFormat.XYXY,
                    colors=[(0, 0, 255)],
                ),
            )
            rr.log(
                f"{cam_log_path / 'pred_body'}",
                CustomPoints2D(
                    positions=vis_uv,
                    confidences=vis_scores_uv,
                    class_ids=2,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
            )
            # log extrapolated keypoints
            if mv_output.uvc_extrap is not None and config.log_extrapolated_keypoints:
                uvc_extrap: Float64[ndarray, "num_kpts 3"] = mv_output.uvc_extrap[cam_idx]
                uv_max: Float64[ndarray, "1 2"] = np.nanmax(uvc_extrap[:, 0:2], axis=0, keepdims=True)
                uv_min: Float64[ndarray, "1 2"] = np.nanmin(uvc_extrap[:, 0:2], axis=0, keepdims=True)
                bbox_extrap: Float64[ndarray, "1 4"] = np.concatenate([uv_min, uv_max], axis=1)
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
                        class_ids=0,
                        keypoint_ids=COCO_133_IDS,
                        show_labels=False,
                    ),
                )

        # check if this is the first frame
        if ts_idx == 0:
            # initialize the optimizer
            xyz_template_left: Float32[ndarray, "num_kpts 3"] = mv_output.xyz[91:112, :3].copy()
            xyz_template_right: Float32[ndarray, "num_kpts 3"] = mv_output.xyz[112:133, :3].copy()
            new_joint_and_scale_optimizer = JointAndScaleOptimization(
                xyz_template=np.stack([xyz_template_left, xyz_template_right], axis=0),
                Pall=Pall,
                loss_weights=loss_weights,
                num_iters=30,
            )
        vis_xyz: Float32[ndarray, "num_kpts 3"] = mv_output.xyz.copy()
        vis_scores_3d: Float32[ndarray, "num_kpts"] = mv_output.scores_3d.copy()  # noqa: UP037
        # filter to only include the desired keypoints
        vis_xyz[~top_half_mask, :] = np.nan
        vis_scores_3d[~top_half_mask] = np.nan

        rr.log(
            "wholebody",
            CustomPoints3D(
                positions=vis_xyz,
                confidences=vis_scores_3d,
                class_ids=2,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
        )

        scores_expanded = mv_output.scores_3d[..., None]  # Add a dimension to match kpts_3d
        xyzc = np.concatenate([mv_output.xyz, scores_expanded], axis=-1)

        mv_uvc_projected: Float32[ndarray, "n_views n_kpts 3"] = projectN3(xyzc, Pall).astype(np.float32)
        # uv_projected_left: Float32[ndarray, "n_views 21 2"] = mv_uvc_projected[:, 91:112, 0:2].copy()
        # uv_projected_right: Float32[ndarray, "n_views 21 2"] = mv_uvc_projected[:, 112:133, 0:2].copy()

        uv_projected_left: Float32[ndarray, "n_views 21 2"] = np.stack(mv_output.uv, axis=0)[:, 91:112]
        uv_projected_right: Float32[ndarray, "n_views 21 2"] = np.stack(mv_output.uv, axis=0)[:, 112:133]

        uv_scores_left: Float32[ndarray, "n_views 21"] = np.stack(mv_output.scores_2d, axis=0)[:, 91:112]
        uv_scores_right: Float32[ndarray, "n_views 21"] = np.stack(mv_output.scores_2d, axis=0)[:, 112:133]
        # filter based on scores over threshold
        uv_projected_left[uv_scores_left < 0.75, :] = np.nan
        uv_projected_right[uv_scores_right < 0.75, :] = np.nan

        optim_out: tuple[OptimizationResults, LevenbergMarquardtState] = new_joint_and_scale_optimizer(
            uv_left_pred_batch=uv_projected_left, uv_right_pred_batch=uv_projected_right
        )
        optimized_result: OptimizationResults = optim_out[0]

        xyz_optim_left: Float64[ndarray, "21 3"] = optimized_result.xyz_mano[0]
        xyz_optim_right: Float64[ndarray, "21 3"] = optimized_result.xyz_mano[1]

        rr.log(
            "xyz-optim_left",
            rr.Points3D(
                positions=xyz_optim_left,
                class_ids=0,
                keypoint_ids=sequence.hand_ids,
                show_labels=False,
            ),
        )

        rr.log(
            "xyz-optim_right",
            rr.Points3D(
                positions=xyz_optim_right,
                class_ids=1,
                keypoint_ids=sequence.hand_ids,
                show_labels=False,
            ),
        )

        # get the left and right hand keypoints from projection
        for all_exo_pinhole, uvc_projected in zip(sequence.exo_cam_list, mv_uvc_projected, strict=True):
            proj_log_path: Path = parent_log_path / all_exo_pinhole.name / "pinhole" / "video"
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
                    class_ids=3,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
            )

        kpts3d_list = [
            mv_output.xyz_extrap,
            mv_output.xyz_t1,
            mv_output.xyz_t2,
        ]
        names = ["extrap", "t1", "t2"]
        for kpts3d, name in zip(kpts3d_list, names, strict=True):
            if kpts3d is not None:
                # this has to happen and I don't know why
                kpts3d[~top_half_mask, :] = np.nan
                if config.log_extrapolated_keypoints:
                    rr.log(
                        f"{name}",
                        rr.Points3D(
                            positions=kpts3d[:, :3],
                            class_ids=2,
                            keypoint_ids=COCO_133_IDS,
                            show_labels=False,
                        ),
                    )
