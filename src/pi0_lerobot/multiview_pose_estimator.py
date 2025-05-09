from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from einops import rearrange
from icecream import ic
from jaxtyping import Float, Float32, Float64, Int
from numpy import ndarray
from rtmlib import YOLOX, RTMPose
from simplecv.ops.triangulate import batch_triangulate, projectN3

from pi0_lerobot.custom_types import BgrImageType


def project_multiview(
    xyzc: Float[ndarray, "n_kpts 4"],
    Pall: Float[ndarray, "n_views 3 4"],
) -> Float[ndarray, "n_views n_kpts 3"]:
    """
    Projects 3D keypoints (with confidence) to 2D image coordinates for multiple views.
    Handles potential division by zero by outputting NaN for those keypoints.

    Args:
        xyzc: Array of 3D keypoints and confidence scores (n_kpts, 4).
              The 4th column is confidence, used for masking, not projection.
        Pall: Array of projection matrices for each view (n_views, 3, 4).

    Returns:
        Array of projected 2D keypoints (u, v, w) for each view (n_views, n_kpts, 3).
        The third component 'w' is the depth scale factor, multiplied by the original
        confidence mask (0 or 1). Keypoints with original confidence <= 0 will have w=0.
        Keypoints where perspective division involved division by near-zero w' will have
        NaN values for u and v.
    """
    n_kpts = xyzc.shape[0]
    # Store original confidences for masking later
    original_conf = xyzc[:, 3].copy()

    # --- Vectorized Version ---
    # 1. Prepare homogeneous coordinates for 3D points (n_kpts, 4)
    #    We use xyz coordinates and append 1.
    kp3d_h = np.hstack((xyzc[:, :3], np.ones((n_kpts, 1), dtype=xyzc.dtype)))  # Shape: (K, 4)

    # 2. Perform matrix multiplication for all views at once.
    #    Using einsum: 'vmn,kn->vkm'
    #    Pall (V, 3, 4), kp3d_h (K, 4) -> kp2d_h (V, K, 3)
    #    kp2d_h[v, k, :] contains the (u', v', w') for view v, keypoint k
    kp2d_h = np.einsum("vmn,kn->vkm", Pall, kp3d_h, optimize=True)  # Shape: (V, K, 3)

    # 3. Perform perspective division (handle division by zero by setting to NaN)
    #    Extract w' (the depth scale factor) -> Shape (V, K, 1)
    w_prime = kp2d_h[..., 2:3]  # Keep last dimension

    #    Define mask for valid divisions (where w' is not close to zero)
    valid_division_mask = np.abs(w_prime) > 1e-8  # Shape: (V, K, 1)

    #    Calculate u = u'/w', v = v'/w'. Suppress warnings for division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_raw_division = kp2d_h[..., :2] / w_prime  # Shape: (V, K, 2)

    #    Use the mask to keep valid results and set others to NaN.
    #    The mask (V, K, 1) broadcasts correctly against (V, K, 2).
    uv_normalized = np.where(valid_division_mask, uv_raw_division, np.nan)  # Shape: (V, K, 2)

    # 4. Combine normalized uv with the original w' (depth scale)
    #    Concatenate along the last axis.
    kp2ds = np.concatenate((uv_normalized, w_prime), axis=-1)  # Shape: (V, K, 3)

    # 5. Apply the original confidence mask to the 'w' component.
    #    Keypoints with confidence <= 0 should have their w component zeroed out.
    #    This uses the confidence stored *before* it was potentially modified.
    confidence_mask = (original_conf[:, None] > 0.0).astype(kp2ds.dtype)  # Shape: (K, 1)
    # Broadcast mask (K, 1) to (V, K, 1) and multiply element-wise with w' column
    kp2ds[..., 2:3] *= confidence_mask  # Modifies the last column in place

    return kp2ds


@dataclass
class MVOutput:
    tracked: bool = False
    xyxy_bboxes: list[Float32[ndarray, "1 4"] | None] = field(default_factory=list)
    uv: list[Float32[ndarray, "n_kpts 2"] | None] = field(default_factory=list)
    scores_2d: list[Float32[ndarray, "n_kpts"] | None] = field(default_factory=list)
    xyz: Float32[ndarray, "n_kpts 3"] | None = None
    scores_3d: Float32[ndarray, "n_kpts"] | None = None
    xyz_t1: Float32[ndarray, "n_kpts 3"] | None = None
    xyz_t2: Float32[ndarray, "n_kpts 3"] | None = None
    xyz_extrap: Float32[ndarray, "n_kpts 3"] | None = None
    uvc_extrap: Float32[ndarray, "n_kpts 3"] | None = None

    def __iter__(self):
        # Zips the bboxes, kpts_2d, and scores lists so each iteration
        # returns a tuple (bbox, kpt2d, score)
        return iter(zip(self.xyxy_bboxes, self.uv, self.scores_2d, strict=True))


class MultiviewBodyTracker:
    MODE = {
        "performance": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",  # noqa
            "pose_input_size": (288, 384),
        },
        "lightweight": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",  # noqa
            "det_input_size": (416, 416),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
        "balanced": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",  # noqa
            "pose_input_size": (192, 256),
        },
        "wholebody": {
            "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",  # noqa
            "det_input_size": (640, 640),
            "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",  # noqa
            "pose_input_size": (192, 256),
        },
    }

    def __init__(
        self,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose: str = None,
        pose_input_size: tuple = (288, 384),
        mode: Literal["lightweight", "balanced", "performance", "wholebody"] = "balanced",
        backend: str = "onnxruntime",
        device: str = "cpu",
        keypoint_threshold: float = 0.3,
        cams_for_detection_idx: list[int] | None = None,
        filter_body_idxes: Int[ndarray, "idx"] | None = None,
        perform_tracking: bool = True,
    ) -> None:
        self.keypoint_threshold = keypoint_threshold
        self.filter_body_idxes = filter_body_idxes
        self.cams_for_detection_idx = cams_for_detection_idx
        self.perform_tracking = perform_tracking
        if pose is None:
            pose = self.MODE[mode]["pose"]
            pose_input_size = self.MODE[mode]["pose_input_size"]

        if det is None:
            det = self.MODE[mode]["det"]
            det_input_size = self.MODE[mode]["det_input_size"]

        self.det_model = YOLOX(det, model_input_size=det_input_size, backend=backend, device=device)
        self.pose_model = RTMPose(
            pose,
            model_input_size=pose_input_size,
            to_openpose=False,
            backend=backend,
            device=device,
        )

        self.prev_bboxes = None
        self.xyzc_t1 = None
        self.xyzc_t2 = None
        self.tracked: bool = False

    def __call__(
        self,
        bgr_list: list[BgrImageType],
        Pall: Float32[ndarray, "n_views 3 4"],
    ) -> MVOutput:
        """
        Assumes to be only one person in the image, take the highest score person
        """
        # filter out the cameras that are not used for detection if provided
        if self.cams_for_detection_idx is not None:
            Pall: Float32[ndarray, "n_views 3 4"] = np.array([Pall[i] for i in self.cams_for_detection_idx])
        # initalize with emtpy and fill in the values
        mv_output = MVOutput()

        ##################################################################################
        # if we have keypoints from previous two timesteps, then use them to extrapolate #
        # the 3d keypoints which we can project to 2d and use for detection              #
        ##################################################################################
        xyzc_extrap = None
        if (self.xyzc_t1 is not None and self.xyzc_t2 is not None) and self.perform_tracking:
            xyzc_extrap: Float32[ndarray, "n_kpts 4"] = self.extrapolate_3d_keypoints(
                xyzc_t1=self.xyzc_t1, xyzc_t2=self.xyzc_t2
            )
            mv_output.xyz_extrap = xyzc_extrap
            # project the extrapolated 3d keypoints to 2d
            uvc_extrap: Float[ndarray, "n_views n_kpts 3"] = projectN3(xyzc_extrap, Pall)

            mv_output.uvc_extrap = uvc_extrap

            uv_max: Float[ndarray, "n_views 2"] = np.nanmax(uvc_extrap[:, :, 0:2], axis=1)
            uv_min: Float[ndarray, "n_views 2"] = np.nanmin(uvc_extrap[:, :, 0:2], axis=1)
            bboxes_extrap: Float[ndarray, "n_views 4"] = np.concatenate([uv_min, uv_max], axis=1).astype(np.float32)

        uvc_list = []
        for image_idx, bgr in enumerate(bgr_list):
            if xyzc_extrap is not None:
                bboxes: Float32[ndarray, "n_dets 4"] = rearrange(bboxes_extrap[image_idx], "B -> 1 B")
            else:
                bboxes: Float32[ndarray, "n_dets 4"] = self.det_model(bgr)

            match bboxes.shape[0]:
                # No detections, set outputs to None for this view
                case 0:
                    mv_output.xyxy_bboxes.append(None)
                    mv_output.uv.append(None)
                    mv_output.scores_2d.append(None)
                    uvc_list.append(np.zeros((17, 3)).astype(np.float32))
                case _:
                    # get the first bbox, #TODO do this based on the highest score
                    bboxes: Float32[ndarray, "1 4"] = bboxes[0:1]

                    keypoints: Float64[ndarray, "n_dets n_kpts 2"]
                    scores: Float32[ndarray, "n_dets n_kpts"]
                    keypoints, scores = self.pose_model(bgr, bboxes=bboxes)

                    filtered_keypoints, filtered_scores = self.filter_kpt_outputs(keypoints.astype(np.float32), scores)

                    # can't be nan is it messes with triangulation
                    # filtered_scores[filtered_scores < self.keypoint_threshold] = 0
                    # filtered_scores[filtered_scores >= self.keypoint_threshold] = 1
                    uvc: Float32[ndarray, "n_kpts 3"] = np.concatenate(
                        [filtered_keypoints, filtered_scores[:, None]], axis=1
                    )
                    uvc_list.append(uvc)

                    mv_output.xyxy_bboxes.append(bboxes)
                    mv_output.uv.append(filtered_keypoints)
                    mv_output.scores_2d.append(filtered_scores)

        multiview_uvc: Float32[ndarray, "n_views n_kpts 3"] = np.stack(uvc_list)
        xyzc: Float64[ndarray, "n_kpts 4"] = batch_triangulate(
            keypoints_2d=multiview_uvc,
            projection_matrices=Pall,
            min_views=3,
        )

        filtered_xyzc = xyzc[self.filter_body_idxes]
        # check if more than half the keypoints are below the threshold, if so, don't track
        if np.sum(filtered_xyzc[:, 3] > self.keypoint_threshold) < filtered_xyzc.shape[0] / 2:
            self.xyzc_t1 = None
            self.xyzc_t2 = None
            self.tracked = False

        # save the keypoints for the next frame
        self.xyzc_t2 = self.xyzc_t1
        self.xyzc_t1 = xyzc.astype(np.float32)

        mv_output.xyz = xyzc[:, :3].astype(np.float32)
        mv_output.scores_3d = xyzc[:, 3].astype(np.float32)

        mv_output.xyz_t1 = self.xyzc_t1
        mv_output.xyz_t2 = self.xyzc_t2
        mv_output.tracked = self.tracked

        return mv_output

    def extrapolate_3d_keypoints(
        self, xyzc_t1: Float32[ndarray, "n_kpts 4"], xyzc_t2: Float32[ndarray, "n_kpts 4"]
    ) -> Float32[ndarray, "n_kpts 4"]:
        """
        Extrapolates 3D keypoints with confidence scores using data from
        the previous two timesteps (t-1 and t-2).
        """
        # extrapolate 3d keypoints from the previous frames
        xyzc_extrap = 2 * xyzc_t1 - xyzc_t2
        return xyzc_extrap

    def filter_kpt_outputs(
        self,
        keypoints: Float32[ndarray, "n_dets n_kpts 2"],
        scores: Float32[ndarray, "n_dets n_kpts"],
    ) -> tuple[Float32[ndarray, "n_kpts 2"], Float32[ndarray, "n_kpts"]]:
        """
        Filter keypoints based on the highest score
        """
        max_scores: Float32[ndarray, "n_dets"] = scores.max(axis=1)
        max_idx = max_scores.argmax()

        return keypoints[max_idx], scores[max_idx]
