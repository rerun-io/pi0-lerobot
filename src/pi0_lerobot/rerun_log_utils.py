from pathlib import Path
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int
from numpy import ndarray


def create_blueprint(
    exo_video_log_paths: list[Path], num_videos_to_log: Literal[4, 8] = 8
) -> rrb.Blueprint:
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
