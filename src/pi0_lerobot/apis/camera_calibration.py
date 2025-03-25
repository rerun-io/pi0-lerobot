from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, assert_never

import cv2
import numpy as np
import rerun as rr
import torch
from einops import rearrange
from jaxtyping import Float32, UInt8
from mini_dust3r.api.inference import (
    OptimizedResult,
    inferece_dust3r_from_rgb,
    log_optimized_result,
)
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.rerun_log_utils import create_blueprint
from numpy import ndarray
from serde import from_dict
from simplecv.camera_parameters import (
    Extrinsics,
    Intrinsics,
    PinholeParameters,
)
from simplecv.data.exoego.base_exo_ego import ExoData
from simplecv.data.exoego.hocap import HOCapSequence, SubjectIDs
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from torch import Tensor
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from pi0_lerobot.vggt_load_utils import VGGTPredictions, preprocess_images
from pi0_lerobot.vggt_load_utils import create_blueprint as cb_vggt

np.set_printoptions(suppress=True)


@dataclass
class CameraCalibConfig:
    rr_config: RerunTyroConfig
    dataset: Literal["hocap", "assembly101"] = "hocap"
    root_directory: Path = Path("/mnt/12tbdrive/data/HO-cap/sample")
    subject_id: SubjectIDs | None = "8"
    sequence_name: str = "20231024_180733"
    # subject_id: SubjectIDs | None = "5"
    # sequence_name: str = "20231027_113202"
    num_videos_to_log: Literal[4, 8] = 8
    log_depths: bool = False
    send_as_batch: bool = True
    calibration_method: Literal["dust3r", "vggt"] = "vggt"


def calibrate_with_dust3r(sequence: HOCapSequence):
    parent_log_path = Path("world")

    # load dust3r model
    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained("pablovela5620/dust3r").to("cuda")
    exo_data: ExoData = next(iter(sequence))
    # Convert BGR images to RGB format
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in exo_data.bgr_list]

    blueprint = create_blueprint(
        image_name_list=[Path(exo_cam.name) for exo_cam in sequence.exo_cam_list],
        log_path=parent_log_path,
    )

    rr.send_blueprint(blueprint)

    # Run dust3r inference on the RGB images
    optimized_results: OptimizedResult = inferece_dust3r_from_rgb(
        rgb_list=rgb_list,
        model=dust3r_model,
        device="cuda",
        batch_size=1,
        min_conf_thr=0.5,
    )
    log_optimized_result(optimized_results, parent_log_path=parent_log_path, log_depth=True)


def calibrate_with_vggt(sequence: HOCapSequence, confidence_threshold: float = 50.0) -> None:
    parent_log_path = Path("world")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    exo_data: ExoData = next(iter(sequence))
    # Convert BGR images to RGB format
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in exo_data.bgr_list]
    img_tensors: Float32[Tensor, "num_img 3 H W"] = preprocess_images(rgb_list).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    blueprint = cb_vggt(parent_log_path, image_paths=[Path(exo_cam.name) for exo_cam in sequence.exo_cam_list])
    rr.send_blueprint(blueprint)

    load_start = timer()
    print("Loading model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    print("Model loaded in", timer() - load_start, "seconds")

    # Run inference
    print("Running inference...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        # run model and convert to dataclass for type validaton + easy access
        predictions: dict = model(img_tensors)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], img_tensors.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].numpy(force=True)

    # Convert from dict to dataclass and performs runtime type validation for easy access
    pred_class: VGGTPredictions = from_dict(VGGTPredictions, predictions)
    pred_class = pred_class.remove_batch_dim_if_one()

    # Generate world points from depth map,this is usually more accurate than the world points from pose encoding
    depth_maps: Float32[ndarray, "num_cams H W 1"] = pred_class.depth
    world_points: Float32[ndarray, "num_cams H W 3"] = unproject_depth_map_to_point_map(
        depth_maps, pred_class.extrinsic, pred_class.intrinsic
    ).astype(np.float32)

    # Get colors from original images and reshape them to match points
    original_images: Float32[ndarray, "num_cams 3 H W"] = img_tensors.numpy(force=True)
    # Rearrange to match point shape expectation
    original_images: Float32[ndarray, "num_cams H W 3"] = rearrange(original_images, "num_cams C H W -> num_cams H W C")
    # Flatten both points and colors
    flattened_points: Float32[ndarray, "num_points 3"] = rearrange(world_points, "num_cams H W C -> (num_cams H W) C")
    flattened_colors: Float32[ndarray, "num_points 3"] = rearrange(
        original_images, "num_cams H W C -> (num_cams H W) C"
    )

    depth_confs: Float32[ndarray, "num_cams H W"] = pred_class.depth_conf
    conf: Float32[ndarray, "num_points"] = depth_confs.reshape(-1)  # noqa UP037

    # Convert percentage threshold to actual confidence value
    conf_threshold = 0.0 if confidence_threshold == 0.0 else np.percentile(conf, confidence_threshold)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d: Float32[ndarray, "num_points 3"] = flattened_points[conf_mask]
    colors_rgb: Float32[ndarray, "num_points 3"] = flattened_colors[conf_mask]

    rr.set_time_sequence("timeline", 0)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(vertices_3d, colors=colors_rgb),
    )
    for idx, (intri, extri, image, depth_map, depth_conf) in enumerate(
        zip(
            pred_class.intrinsic,
            pred_class.extrinsic,
            original_images,
            depth_maps,
            depth_confs,
            strict=True,
        )
    ):
        cam_name: str = f"camera_{idx}"
        cam_log_path: Path = parent_log_path / cam_name
        intri_param = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(intri[0, 0]),
            fl_y=float(intri[1, 1]),
            cx=float(intri[0, 2]),
            cy=float(intri[1, 2]),
            width=image.shape[1],
            height=image.shape[0],
        )
        extri_param = Extrinsics(
            cam_R_world=extri[:, :3],
            cam_t_world=extri[:, 3],
        )
        pinhole_param = PinholeParameters(name=cam_name, intrinsics=intri_param, extrinsics=extri_param)
        conf_threshold = 0.0 if confidence_threshold == 0.0 else np.percentile(depth_conf, confidence_threshold)
        conf_mask = (depth_conf >= conf_threshold) & (depth_conf > 1e-5)
        # filter depth map based on confidence
        depth_map = depth_map.squeeze()
        depth_map[~conf_mask] = 0.0

        rr.log(f"{cam_log_path}/pinhole/image", rr.Image(image))
        rr.log(f"{cam_log_path}/pinhole/confidence", rr.Image(conf_mask.astype(np.float32)))
        rr.log(f"{cam_log_path}/pinhole/depth", rr.DepthImage(depth_map, draw_order=1))
        log_pinhole(pinhole_param, cam_log_path=cam_log_path, image_plane_distance=0.1)

    # Clean up
    torch.cuda.empty_cache()


def calibrate_camera(config: CameraCalibConfig) -> None:
    start_time: float = timer()

    sequence: HOCapSequence = HOCapSequence(
        data_path=config.root_directory,
        sequence_name=config.sequence_name,
        subject_id=config.subject_id,
        load_labels=True,
    )

    match config.calibration_method:
        case "dust3r":
            calibrate_with_dust3r(sequence)
        case "vggt":
            calibrate_with_vggt(sequence)
        case _:
            assert_never(config.calibration_method)

    print(f"Time taken to load data: {timer() - start_time:.2f} seconds")
