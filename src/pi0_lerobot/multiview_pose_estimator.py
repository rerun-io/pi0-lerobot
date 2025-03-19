from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from jaxtyping import Float, Float32, Float64, Int
from numpy import ndarray
from rtmlib import YOLOX, RTMPose
from simplecv.ops.triangulate import batch_triangulate

from pi0_lerobot.custom_types import BgrImageType


def projectN3(
    kpts3d: Float[ndarray, "n_kpts 4"],
    Pall: Float[ndarray, "n_views 3 4"],
) -> Float[ndarray, "n_views nJoints 3"]:
    nViews: int = len(Pall)
    # convert to homogenous
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.0)
    return kp2ds


@dataclass
class MVOutput:
    tracked: bool = False
    bboxes: list[Float32[ndarray, "1 4"] | None] = field(default_factory=list)
    kpts_2d: list[Float32[ndarray, "num_kpts 2"] | None] = field(default_factory=list)
    scores_2d: list[Float32[ndarray, "num_kpts"] | None] = field(default_factory=list)
    kpts_3d: Float32[ndarray, "num_kpts 3"] | None = None
    scores_3d: Float32[ndarray, "num_kpts"] | None = None
    kpts_3d_t1: Float32[ndarray, "num_kpts 3"] | None = None
    kpts_3d_t2: Float32[ndarray, "num_kpts 3"] | None = None
    kpts_3d_extrapolated: Float32[ndarray, "num_kpts 3"] | None = None
    uvc_extrap: Float32[ndarray, "num_kpts 3"] | None = None

    def __iter__(self):
        # Zips the bboxes, kpts_2d, and scores lists so each iteration
        # returns a tuple (bbox, kpt2d, score)
        return iter(zip(self.bboxes, self.kpts_2d, self.scores_2d, strict=True))


def log_mvoutput(mv_output: MVOutput): ...


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
    }

    def __init__(
        self,
        det: str = None,
        det_input_size: tuple = (640, 640),
        pose: str = None,
        pose_input_size: tuple = (288, 384),
        mode: Literal["lightweight", "balanced", "performance"] = "balanced",
        backend: str = "onnxruntime",
        device: str = "cpu",
        keypoint_threshold: float = 0.3,
        cams_for_detection_idx: list[int] | None = None,
        filter_body_idxes: Int[ndarray, "idx"] | None = None,
    ) -> None:
        self.keypoint_threshold = keypoint_threshold
        self.filter_body_idxes = filter_body_idxes
        self.cams_for_detection_idx = cams_for_detection_idx
        if pose is None:
            pose = self.MODE[mode]["pose"]
            pose_input_size = self.MODE[mode]["pose_input_size"]

        if det is None:
            det = self.MODE[mode]["det"]
            det_input_size = self.MODE[mode]["det_input_size"]

        self.det_model = YOLOX(
            det, model_input_size=det_input_size, backend=backend, device=device
        )
        self.pose_model = RTMPose(
            pose,
            model_input_size=pose_input_size,
            to_openpose=False,
            backend=backend,
            device=device,
        )

        self.prev_bboxes = None
        self.prev_kpts_3d_t1 = None
        self.prev_kpts_3d_t2 = None
        self.tracked: bool = False

    def __call__(
        self,
        bgr_list: list[BgrImageType],
        Pall: Float32[ndarray, "n_views 3 4"],
    ) -> MVOutput:
        """
        Assumes to be only one person in the image, take the highest score person
        """
        if self.cams_for_detection_idx is not None:
            Pall: Float32[ndarray, "nViews 3 4"] = np.array(
                [Pall[i] for i in self.cams_for_detection_idx]
            )
        # initalize with emtpy and fill in the values
        mv_output = MVOutput()
        uvc_list = []
        # extrapolte 3d keypoints from the previous frames
        xyzc_extrap = None
        if self.prev_kpts_3d_t1 is not None and self.prev_kpts_3d_t2 is not None:
            xyzc_extrap = self.extrapolate_3d_keypoints(
                kpts_3d_t1=self.prev_kpts_3d_t1, kpts_3d_t2=self.prev_kpts_3d_t2
            )
            mv_output.kpts_3d_extrapolated = xyzc_extrap
            # project the extrapolated 3d keypoints to 2d
            uvc_extrap: Float[ndarray, "n_views n_kpts 3"] = projectN3(
                xyzc_extrap, Pall
            )
            mv_output.uvc_extrap = uvc_extrap
            bbox_extrap_list = []
            for uvc_extrap_view in uvc_extrap:
                u_max = np.nanmax(uvc_extrap_view[:, 0])
                u_min = np.nanmin(uvc_extrap_view[:, 0])
                v_max = np.nanmax(uvc_extrap_view[:, 1])
                v_min = np.nanmin(uvc_extrap_view[:, 1])
                bbox_extrap_list.append([u_min, v_min, u_max, v_max])
            bboxes_extrap = np.array(bbox_extrap_list).astype(np.float32)

        for image_idx, bgr in enumerate(bgr_list):
            try:
                bboxes: Float32[ndarray, "num_dets 4"] = bboxes_extrap[image_idx][
                    None, :
                ]
            except NameError:
                bboxes: Float32[ndarray, "num_dets 4"] = self.det_model(bgr)
            if bboxes.shape[0] == 0:
                mv_output.bboxes.append(None)
                mv_output.kpts_2d.append(None)
                mv_output.scores_2d.append(None)
                uvc_list.append(np.zeros((17, 3)).astype(np.float32))
            elif bboxes.shape[0] == 1:
                keypoints: Float64[ndarray, "num_dets num_kpts 2"]
                scores: Float32[ndarray, "num_dets num_kpts"]
                keypoints, scores = self.pose_model(bgr, bboxes=bboxes)

                filtered_keypoints, filtered_scores = self.filter_kpt_outputs(
                    keypoints.astype(np.float32), scores
                )

                # can't be nan is it messes with triangulation
                # filtered_scores[filtered_scores < self.keypoint_threshold] = 0
                # filtered_scores[filtered_scores >= self.keypoint_threshold] = 1
                uvc: Float32[ndarray, "num_kpts 3"] = np.concatenate(
                    [filtered_keypoints, filtered_scores[:, None]], axis=1
                )
                uvc_list.append(uvc)

                mv_output.bboxes.append(bboxes)
                mv_output.kpts_2d.append(filtered_keypoints)
                mv_output.scores_2d.append(filtered_scores)

            else:
                # get the first bbox, #TODO do this based on the highest score
                bboxes = bboxes[0:1]
                keypoints: Float64[ndarray, "num_dets num_kpts 2"]
                scores: Float32[ndarray, "num_dets num_kpts"]
                keypoints, scores = self.pose_model(bgr, bboxes=bboxes)

                filtered_keypoints, filtered_scores = self.filter_kpt_outputs(
                    keypoints.astype(np.float32), scores
                )

                # can't be nan is it messes with triangulation
                # filtered_scores[filtered_scores < self.keypoint_threshold] = 0
                # filtered_scores[filtered_scores >= self.keypoint_threshold] = 1
                uvc: Float32[ndarray, "num_kpts 3"] = np.concatenate(
                    [filtered_keypoints, filtered_scores[:, None]], axis=1
                )
                uvc_list.append(uvc)

                mv_output.bboxes.append(bboxes)
                mv_output.kpts_2d.append(filtered_keypoints)
                mv_output.scores_2d.append(filtered_scores)

        multiview_uvc: Float32[ndarray, "n_views num_kpts 3"] = np.stack(uvc_list)
        xyzc: Float64[ndarray, "num_kpts 4"] = batch_triangulate(
            keypoints_2d=multiview_uvc,
            projection_matrices=Pall,
            min_views=3,
        )

        filtered_xyzc = xyzc[self.filter_body_idxes]
        # check if more than half the keypoints are below the threshold
        if (
            np.sum(filtered_xyzc[:, 3] > self.keypoint_threshold)
            < filtered_xyzc.shape[0] / 2
        ):
            self.prev_kpts_3d_t1 = None
            self.prev_kpts_3d_t2 = None
            self.tracked = False

        self.prev_kpts_3d_t2 = self.prev_kpts_3d_t1
        self.prev_kpts_3d_t1 = xyzc.astype(np.float32)

        mv_output.kpts_3d = xyzc[:, :3].astype(np.float32)
        mv_output.scores_3d = xyzc[:, 3].astype(np.float32)

        mv_output.kpts_3d_t1 = self.prev_kpts_3d_t1
        mv_output.kpts_3d_t2 = self.prev_kpts_3d_t2
        mv_output.tracked = self.tracked

        return mv_output

    def extrapolate_3d_keypoints(
        self, kpts_3d_t1, kpts_3d_t2
    ) -> Float32[ndarray, "num_kpts 4"]:
        """
        Extrapolate 3d keypoints from the previous frames
        t-1 and t-2
        """
        # extrapolate 3d keypoints from the previous frames
        kpts_3d_extrapolated = 2 * kpts_3d_t1 - kpts_3d_t2
        return kpts_3d_extrapolated

    def filter_kpt_outputs(
        self,
        keypoints: Float32[ndarray, "num_dets num_kpts 2"],
        scores: Float32[ndarray, "num_dets num_kpts"],
    ) -> tuple[Float32[ndarray, "num_kpts 2"], Float32[ndarray, "num_kpts"]]:
        """
        Filter keypoints based on the highest score person
        """
        max_scores = scores.max(axis=1)
        max_idx = max_scores.argmax()

        return keypoints[max_idx], scores[max_idx]
