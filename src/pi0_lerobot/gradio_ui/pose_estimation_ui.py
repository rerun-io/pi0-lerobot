import time
from dataclasses import dataclass, fields
from pathlib import Path

import beartype
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import UInt16
from tqdm import tqdm
import cv2


@dataclass
class InputComponents:
    polycam_zip_path: gr.File
    max_depth_range_meter: gr.Number
    depth_fusion_resolution: gr.Slider

    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class InputValues:
    polycam_zip_path: str
    max_depth_range_meter: int | float
    depth_fusion_resolution: float

    def __post_init__(self):
        # number always seems to return an int so need to manage that
        self.max_depth_range_meter: float = float(self.max_depth_range_meter)


# @rr.thread_local_stream("rerun_prompt_da")
# def stream_polycam_da(
#     *input_params,
#     progress=gr.Progress(),  # noqa: B008
# ):
#     try:
#         parameters = InputValues(*input_params)
#     except beartype.roar.BeartypeCallHintParamViolation as e:
#         raise gr.Error(  # noqa: B904
#             "Did you make sure the zipfile finished uploading?. Try to hit run again.",
#             duration=20,
#         )
#     except Exception as e:
#         raise gr.Error(  # noqa: B904
#             f"Error: {e}\n Did you wait for zip file to upload?", duration=20
#         )

#     stream: rr.BinaryStream = rr.binary_stream()

#     polycam_zip_path = Path(parameters.polycam_zip_path)
#     parent_log_path: Path = Path("world")
#     rr.log("/", rr.ViewCoordinates.RUB, timeless=True)
#     blueprint: rrb.Blueprint = create_blueprint(parent_log_path=parent_log_path)
#     rr.send_blueprint(blueprint)
#     polycam_dataset: PolycamDataset = load_polycam_data(
#         polycam_zip_or_directory_path=polycam_zip_path
#     )

#     pred_fuser = Open3DFuser(
#         fusion_resolution=parameters.depth_fusion_resolution,
#         max_fusion_depth=parameters.max_depth_range_meter,
#     )

#     progress(progress=0.1, desc="Loading PromptDA model")
#     model = PromptDAPredictor(device="cuda", model_type="large", max_size=1008)
#     pbar = tqdm(polycam_dataset, desc="Inferring", total=len(polycam_dataset))
#     polycam_data: PolycamData
#     for frame_idx, polycam_data in enumerate(pbar):
#         rr.set_time_sequence("frame_idx", frame_idx)
#         # convert image data to tensor
#         depth_pred: CompletionDepthPrediction = model(
#             rgb=polycam_data.rgb_hw3, prompt_depth=polycam_data.original_depth_hw
#         )

#         # filter depthmaps based on confidence, only keep with max confidence
#         pred_filtered_depth_mm: UInt16[np.ndarray, "h w"] = filter_depth(
#             depth_mm=depth_pred.depth_mm,
#             confidence=polycam_data.confidence_hw,
#             confidence_threshold=DepthConfidenceLevel.MEDIUM,
#             max_depth_meter=parameters.max_depth_range_meter,
#         )

#         # fuse the predicted depth and the ground truth depth
#         pred_fuser.fuse_frames(
#             depth_hw=pred_filtered_depth_mm,
#             K_33=polycam_data.pinhole_params.intrinsics.k_matrix,
#             cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
#             rgb_hw3=polycam_data.rgb_hw3,
#         )

#         log_polycam_data(
#             parent_path=parent_log_path,
#             polycam_data=polycam_data,
#             depth_pred=depth_pred.depth_mm,
#             rescale_factor=1,
#         )

#         pred_mesh = pred_fuser.get_mesh()
#         pred_mesh.compute_vertex_normals()

#         rr.log(
#             f"{parent_log_path}/pred_mesh",
#             rr.Mesh3D(
#                 vertex_positions=pred_mesh.vertices,
#                 triangle_indices=pred_mesh.triangles,
#                 vertex_normals=pred_mesh.vertex_normals,
#                 vertex_colors=pred_mesh.vertex_colors,
#             ),
#         )

#         yield stream.read()


@rr.thread_local_stream("rerun_example_streaming_blur")
def streaming_repeated_blur(img):
    stream = rr.binary_stream()

    if img is None:
        raise gr.Error("Must provide an image to blur.")

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="image/original"),
            rrb.Spatial2DView(origin="image/blurred"),
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(blueprint)

    rr.set_time_sequence("iteration", 0)

    rr.log("image/original", rr.Image(img))
    yield stream.read()

    blur = img

    for i in range(100):
        rr.set_time_sequence("iteration", i)

        # Pretend blurring takes a while so we can see streaming in action.
        time.sleep(0.1)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)

        rr.log("image/blurred", rr.Image(blur))

        # Each time we yield bytes from the stream back to Gradio, they
        # are incrementally sent to the viewer. Make sure to yield any time
        # you want the user to be able to see progress.
        yield stream.read()


with gr.Blocks() as pose_estimation_block:
    with gr.Row():
        # It may be helpful to point the viewer to a hosted RRD file on another server.
        # If an RRD file is hosted via http, you can just return a URL to the file.
        choose_rrd = gr.Dropdown(
            label="RRD",
            choices=[
                f"{rr.bindings.get_app_url()}/examples/arkit_scenes.rrd",
                f"{rr.bindings.get_app_url()}/examples/dna.rrd",
                f"{rr.bindings.get_app_url()}/examples/plots.rrd",
                "assembly-pt1.rrd",
            ],
        )
    with gr.Row():
        viewer = Rerun(
            streaming=True,
            panel_states={
                "time": "collapsed",
                "blueprint": "hidden",
                "selection": "hidden",
            },
        )
    choose_rrd.change(lambda x: x, inputs=[choose_rrd], outputs=[viewer])
