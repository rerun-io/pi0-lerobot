from dataclasses import dataclass
from typing import Literal, TypedDict

import jax
import jax.numpy as npj
import numpy as np
from jax import jit
from jaxopt import LevenbergMarquardt
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Array, Float
from numpy import ndarray

from pi0_lerobot.mano.mano_jax import mano_j_left, mano_j_right


@dataclass
class OptimizationResults:
    """
    Stores results from bounding box detection and hand pose estimation
    """

    xyz_mano: Float[ndarray, "2 21 3"]
    so3: Float[ndarray, "2 48"]
    betas: Float[ndarray, "2 10"]
    trans: Float[ndarray, "2 3"]


class LossWeights(TypedDict):
    keypoint_2d: float
    depth: float
    temp: float


@jit
def project_3d_batch(kp3d: npj.ndarray, P: npj.ndarray) -> npj.ndarray:
    """

    kp3d: [nFrames, 21, 4] [x, y, z, 1]
    P: [nViews, 3, 4] (projection matrix - includes extrensic (R, t) and intrinsic (K))

    return kp2d: [nFrames, nViews, nJoints, 2] (squeeze out if 1)
    """
    # kpts_est: (nFrames, nJoints, 3+1), P: (nViews, 3, 4)
    #   => projection: (nViews, nFrames, nJoints, 3)
    point_cam = npj.einsum("vab,fnb->vfna", P, kp3d)
    kp2d = point_cam[..., :2] / point_cam[..., 2:]

    return npj.squeeze(kp2d)


@jit
def mv_residual(
    param_to_optimize: Float[Array, "51"],
    cameras: Float[Array, "b 3 4"],
    xyz_pred: Float[Array, "b 21 3"],
    loss_weights: dict,
    is_left,
) -> npj.ndarray:
    """
    param_to_optimize - pose/trans should not have a batch dim
    cameras (b, 3, 4)
    xyz_pred (b, 21, 3)
    loss_weight function weighting hyperparameters dict[str, float]
    """
    # extract parameters that are being optimized and add batch dimension
    # (1, 48)
    so3: Float[Array, "b 48"] = param_to_optimize[0:48][npj.newaxis, ...]
    trans: Float[Array, "b 1 3"] = param_to_optimize[48:][npj.newaxis, npj.newaxis, ...]

    def left_func(x):
        return mano_j_left(x[0], x[1])

    def right_func(x):
        return mano_j_right(x[0], x[1])

    # mano poses so3 -> mano 3d joints in meters??
    # 1, 21, 3. There should only ever be a single 3d joint per timestep
    xyz_mano: Float[Array, "b 21 3"] = jax.lax.cond(is_left, left_func, right_func, (so3, trans))

    res = mv_mega_residual(xyz_mano, xyz_pred, cameras, loss_weights)

    return res


@jit
def mv_mega_residual(
    xyz_mano: Float[Array, "b 21 3"],
    xyz_pred: Float[Array, "b 21 3"],
    cameras: Float[Array, "b 3 4"],
    loss_weights: LossWeights,
):
    # (b, 21, 2)
    res_3d: Float[Array, "b 21 3"] = xyz_mano - xyz_pred

    res_kp: Float[Array, "_"] = npj.concatenate(  # noqa: UP037
        [
            loss_weights["keypoint_2d"] * res_3d,
        ],
        axis=2,
    ).flatten()

    res = npj.concatenate([res_kp])

    return res


class LMOptimJointOnly:
    HAND_TYPE: Literal["left", "right"] = ["left", "right"]

    def __init__(
        self,
        Pall: Float[ndarray, "batch 3 4"],
        loss_weights: LossWeights,
        num_iters: int = 30,
        optimize_scale_factor: bool = False,
    ) -> None:
        """
        Pall - n, 3, 4 projection matrix
        loss_weights - dictionary containing how much value to give each portion
            of the cost function (2d, 3d, temporal)
        num_iters - how many iterations to optimize
        """

        batch_size: int = Pall.shape[0]

        self.num_iters: int = num_iters
        # Projection Matrix (n, 3, 4) where n is the number of cameras
        self.Pall: Float[ndarray, "batch 3 4"] = Pall

        self.loss_weights: LossWeights = loss_weights
        # use previous values to initialize, there should only ever be 1
        # hand model per frame
        self.so3_left_prev: Float[Array, "48"] = npj.zeros(48)  # noqa: UP037
        self.trans_left_prev: Float[Array, "3"] = npj.array([0.2, 0, 1.5])  # noqa: UP037

        self.so3_right_prev: Float[Array, "48"] = npj.zeros(48)  # noqa: UP037
        self.trans_right_prev: Float[Array, "3"] = npj.array([-0.2, 0, 1.5])  # noqa: UP037

        # remnant from mano, not needed
        self.beta: Float[Array, "1 10"] = npj.zeros((1, 10))

        # remove the need for two different optimizers, solvers ‘cholesky’, ‘inv’
        self.optimizer = LevenbergMarquardt(
            residual_fun=mv_residual, maxiter=self.num_iters, solver="cholesky", jit=True, xtol=1e-6, gtol=1e-6
        )
        # add jit
        print("Tracing JIT, can take a while...")
        init_params: Float[Array, "51"] = npj.concatenate([self.so3_left_prev, self.trans_left_prev])
        _, _ = self.optimizer.run(
            init_params,
            cameras=npj.array(Pall),
            xyz_pred=npj.zeros((batch_size, 21, 3)),
            loss_weights=self.loss_weights,
            is_left=True,
        )
        self.optimizer = jit(self.optimizer.run)

        print("Trace Done")

    def __call__(
        self,
        xyz_pred_batch: Float[ndarray, "b 21 3"] | None = None,
    ) -> tuple[OptimizationResults, LevenbergMarquardtState]:
        """
        pose_predictions_dict
            pose_predictions
            camera_dict
        """
        so3_optimized: Float[ndarray, "2 48"] = np.zeros((2, 48))
        trans_optimized: Float[ndarray, "2 3"] = np.zeros((2, 3))
        xyz_mano: Float[ndarray, "2 21 3"] = np.zeros((2, 21, 3))
        for hand_type in self.HAND_TYPE:
            # get previous values, and extract pose_predictions
            match hand_type:
                case "left":
                    xyz_pred: Float[ndarray, "21 3"] = xyz_pred_batch[0]
                    so3_prev: Float[Array, "48"] = self.so3_left_prev.copy()
                    trans_prev: Float[Array, "3"] = self.trans_left_prev.copy()

                case "right":
                    xyz_pred: Float[ndarray, "21 3"] = xyz_pred_batch[1]
                    so3_prev: Float[Array, "48"] = self.so3_right_prev.copy()
                    trans_prev: Float[Array, "3"] = self.trans_right_prev.copy()

            # TODO initialize only rotation from wrist form either 3d procustus or mano preds
            so3_init: Float[Array, "48"] = so3_prev
            trans_init: Float[Array, "3"] = trans_prev

            cam_param_list: Float[Array, "batch 3 4"] = npj.array(self.Pall)
            init_params: Float[Array, "51"] = npj.concatenate([so3_init, trans_init])
            params, state = self.optimizer(
                init_params,
                cameras=cam_param_list,
                xyz_pred=xyz_pred[npj.newaxis, ...],
                loss_weights=self.loss_weights,
                is_left=hand_type == "left",
            )

            if np.isnan(params).any():
                continue

            so3: Float[Array, "48"] = params[:48]
            trans: Float[Array, "3"] = params[48:]

            so3_optimized[0 if hand_type == "left" else 1] = np.array(so3)
            trans_optimized[0 if hand_type == "left" else 1] = np.array(trans)

            # pass optimized values to mano to extract 3d joints
            match hand_type:
                case "left":
                    self.so3_left_prev = so3
                    self.trans_left_prev = trans

                    xyz_mano_left: Float[Array, "1 21 3"] = mano_j_left(
                        so3[npj.newaxis, ...], trans[npj.newaxis, npj.newaxis, ...]
                    )
                    xyz_mano[0] = np.array(xyz_mano_left[0])

                case "right":
                    self.so3_right_prev = so3
                    self.trans_right_prev = trans

                    xyz_mano_right: Float[Array, "1 21 3"] = mano_j_right(
                        so3[npj.newaxis, ...], trans[npj.newaxis, npj.newaxis, ...]
                    )
                    xyz_mano[1] = np.array(xyz_mano_right[0])

        optimization_results = OptimizationResults(
            xyz_mano=xyz_mano,
            so3=so3_optimized,
            trans=trans_optimized,
            betas=np.concatenate([self.beta, self.beta], axis=0),
        )

        return optimization_results, state
