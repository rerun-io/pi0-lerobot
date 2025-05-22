from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float32, Int
from numpy import ndarray
from simplecv.data.exoego.base_exo_ego import BaseExoEgoSequence
from tqdm import tqdm

from pi0_lerobot.mano.mano_utils import MANOLayerJax
from pi0_lerobot.skeletons.coco_17 import COCO_ID2NAME, COCO_LINKS


def set_pose_annotation_context(sequence: BaseExoEgoSequence) -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Left Hand", color=(0, 0, 255)),
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
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=2, label="Triangulate", color=(0, 255, 255)),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in COCO_ID2NAME.items()],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=3, label="Body", color=(0, 0, 255)),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in COCO_ID2NAME.items()],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=4, label="extrap"),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in COCO_ID2NAME.items()],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=5, label="t1"),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in COCO_ID2NAME.items()],
                    keypoint_connections=COCO_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=6, label="t2"),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in COCO_ID2NAME.items()],
                    keypoint_connections=COCO_LINKS,
                ),
            ]
        ),
        static=True,
    )


def create_blueprint(exo_video_log_paths: list[Path], num_videos_to_log: Literal[4, 8] = 8) -> rrb.Blueprint:
    active_tab: int = 0  # 0 for video, 1 for images
    main_view = rrb.Vertical(
        contents=[
            rrb.Tabs(
                rrb.Spatial3DView(
                    origin="/",
                ),
                rrb.Spatial3DView(
                    origin="/",
                    contents=[
                        "+ $origin/**",
                        "- /world/mesh",
                    ],
                ),
                active_tab=active_tab,
            ),
            # take the first 4 video files
            rrb.Horizontal(
                contents=[
                    rrb.Tabs(
                        rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                        rrb.Spatial2DView(
                            origin=f"{video_log_path}".replace("video", "depth"),
                        ),
                        active_tab=active_tab,
                    )
                    for video_log_path in exo_video_log_paths[:4]
                ]
            ),
        ],
        row_shares=[3, 1],
    )
    additional_views = rrb.Vertical(
        contents=[
            rrb.Tabs(
                rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                rrb.Spatial2DView(origin=f"{video_log_path}".replace("video", "depth")),
                active_tab=active_tab,
            )
            for video_log_path in exo_video_log_paths[4:]
        ]
    )
    # do the last 4 videos
    contents = [main_view]
    if num_videos_to_log == 8:
        contents.append(additional_views)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def log_mano_batch(
    sequence: BaseExoEgoSequence,
    shortest_timestamp: Int[ndarray, "num_frames"],
    timeline: str,
    mano_root_dir: Path,
) -> None:
    mano_poses: Float32[Array, "num_frames 2 51"] = jnp.array(sequence.exo_batch_data.mano_stack.poses)

    # order is important here
    hand_sides: list[str] = ["right", "left"]
    mano_layers: list[MANOLayerJax] = [
        MANOLayerJax(
            side=side,
            betas=sequence.exo_batch_data.mano_stack.betas,
            mano_root_dir=mano_root_dir,
        )
        for side in hand_sides
    ]

    pbar = tqdm(
        enumerate(
            zip(
                hand_sides,
                mano_layers,
                strict=True,
            )
        ),
        desc="Logging hand keypoints",
        total=2,
    )

    for hand_idx, (hand_side, mano_layer) in pbar:
        poses: Float32[Array, "num_frames 48"] = mano_poses[:, hand_idx, :48]
        translations: Float32[Array, "num_frames 3"] = mano_poses[:, hand_idx, 48:51]
        mano_outputs: tuple[
            Float32[Array, "num_frames 778 3"],
            Float32[Array, "num_frames 21 3"],
        ] = mano_layer.forward(poses, translations)
        verts: Float32[Array, "num_frames 778 3"] = mano_outputs[0]
        joints: Float32[Array, "num_frames 21 3"] = mano_outputs[1]

        triangle_indices_np = np.array(mano_layer.f)

        rr.log(
            f"mano_{mano_layer.side}",
            rr.Points3D.from_fields(
                # radii=0.005,
                show_labels=False,
            ),
            static=True,
        )

        rr.send_columns(
            f"mano_{mano_layer.side}",
            indexes=[rr.TimeNanosColumn(timeline, shortest_timestamp[0 : len(sequence)])],
            columns=[
                *rr.Points3D.columns(
                    positions=rearrange(
                        np.array(verts),
                        "num_frames kpts dim -> (num_frames kpts) dim",
                    ),
                ).partition(lengths=[778] * len(sequence)),
            ],
        )
