import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

import jax
import jax.numpy as npj
import numpy as np
from einops import rearrange
from jax import jit
from jaxopt import LevenbergMarquardt
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Array, Bool, Float
from numpy import ndarray

from pi0_lerobot.mano.kinematic_hand_jax import JointsOnly, mp_to_mano

FwdKinematics = Callable[
    [Float[Array, "b 48"], Float[Array, "b 1 3"], Float[Array, "1 1"]],
    Float[Array, "b 21 3"],
]

# The residual you’ll hand to jaxopt
ResidualFn = Callable[
    [
        Float[Array, "_"],  # flattened params + scale
        Float[Array, "b 3 4"],  # Pall
        Float[Array, "b n_views 21 2"],  # uv_pred
        "LossWeights",
        bool | Bool[Array, ""],
    ],
    Float[Array, "_"],  # flat residual vector
]


@dataclass
class OptimizationResults:
    """
    Stores results from bounding box detection and hand pose estimation
    """

    xyz_mano: Float[ndarray, "2 21 3"]
    so3: Float[ndarray, "2 48"]
    trans: Float[ndarray, "2 3"]
    betas: Float[ndarray, "2 10"] | None = None


class LossWeights(TypedDict):
    keypoint_2d: float
    depth: float
    temp: float


@jit
def proj_3d_vectorized(
    xyz_hom: Float[Array, "n_frames n_joints 4"], P: Float[Array, "n_views 3 4"]
) -> Float[Array, "n_frames n_views n_joints 2"]:
    """
    Projects 3D points to 2D using the projection matrix for a batch of frames and views.

    xyz_hom: [n_frames, 21, 4] [x, y, z, 1]
    P: [n_views, 3, 4] (projection matrix - includes extrensic (R, t) and intrinsic (K))

    return kp2d: [n_frames, n_views, n_joints, 2] (squeeze out if 1)
    """
    # rearrange for batch matrix multiplication
    xyz_hom: Float[Array, "n_frames 1 4 21"] = rearrange(
        xyz_hom, "n_frames n_joints xyz_hom -> n_frames 1 xyz_hom n_joints"
    )
    P: Float[Array, "1 n_views 3 4"] = rearrange(P, "n_views n m -> 1 n_views n m")

    # [1 n_views, 3, 4] @ [n_frames, 1, 4, 21] -> [n_frames, n_views, 3, 21]
    uv_hom: Float[Array, "n_frames n_views 3 21"] = P @ xyz_hom
    uv_hom = rearrange(uv_hom, "n_frames n_views xyz_hom n_joints -> n_frames n_views n_joints xyz_hom")
    # convert back from homogeneous coordinates
    uv: Float[Array, "n_frames n_views 21 2"] = uv_hom[..., :2] / uv_hom[..., 2:]

    return uv


class HandSide(enum.IntEnum):
    """Represents the side of the hand."""

    LEFT = 0
    RIGHT = 1


FwdKinematics = Callable[
    [Float[Array, "b 48"], Float[Array, "b 1 3"], Float[Array, "1 1"]],
    Float[Array, "b 21 3"],
]

# The residual you’ll hand to jaxopt
ResidualFn = Callable[
    [
        Float[Array, "_"],  # flattened params + scale
        Float[Array, "b 3 4"],  # Pall
        Float[Array, "b n_views 21 2"],  # uv_pred
        "LossWeights",
        bool | Bool[Array, ""],
    ],
    Float[Array, "_"],  # flat residual vector
]


def make_mv_scaled_residual(
    xyz_template_left: Float[Array, "21 3"],
    xyz_template_right: Float[Array, "21 3"],
) -> tuple[ResidualFn, FwdKinematics, FwdKinematics]:
    """
    Returns a JIT-compiled residual function that can be dropped straight into
    `jaxopt.LevenbergMarquardt`.  No globals leak out – the MANO forward
    functions are closed over the templates you pass in *once*.

    Example
    -------
    >>> mv_scaled_residual = make_mv_scaled_residual(
    ...     xyz_template_left , xyz_template_right
    ... )
    >>> solver = LevenbergMarquardt(residual_fun=mv_scaled_residual, ...)
    """

    # ------------------------------------------------------------------
    # build per-hand forward kinematics (static because templates are constant)
    # ------------------------------------------------------------------
    joint_fwd_left = jit(JointsOnly(template_joints=xyz_template_left[mp_to_mano, :]))
    joint_fwd_right = jit(JointsOnly(template_joints=xyz_template_right[mp_to_mano, :]))

    # ------------------------------------------------------------------
    # residual – declared once, re-used frame-to-frame
    # ------------------------------------------------------------------
    @jit
    def mv_2d_scaled_residual(
        param_to_optimize: Float[Array, "_"],
        Pall: Float[Array, "b 3 4"],
        uv_pred: Float[Array, "b n_views 21 2"],
        loss_weights: LossWeights,
        is_left: bool | Bool[Array, ""],
    ) -> Float[Array, "_"]:
        """
        Calculates the residual error between projected MANO keypoints and target 2D keypoints.

        Args:
            param_to_optimize: Flattened MANO parameters (pose coefficients and translation).
                            Must be a 1D array (batch_size * 51) because jaxopt optimizers
                            like LevenbergMarquardt expect a flat vector of parameters.
            Pall: Projection matrices for each camera view, shape (b, 3, 4).
                'b' here refers to the batch size (number of frames/samples).
            uv_pred: Target 2D keypoints for each view and joint, shape (b, n_views, 21, 2).
                    'n_views' is the number of camera views.
            loss_weights: Dictionary containing weights for different loss components (e.g., 'keypoint_2d').
            is_left: Boolean indicating whether to use the left or right MANO model.

        Returns:
            A flattened 1D array containing the weighted residual errors for all keypoints, views, and batch items.
        """
        batch_size: int = uv_pred.shape[0]
        # extract parameters that are being optimized and add batch dimension
        scale_param: Float[Array, ""] = param_to_optimize[-1]
        scale_param: Float[Array, "1 1"] = scale_param.reshape(1, 1)
        param_to_optimize: Float[Array, "_"] = param_to_optimize[:-1]  #
        param_to_optimize: Float[Array, "1 51"] = param_to_optimize.reshape(batch_size, 51)

        so3: Float[Array, "b 48"] = param_to_optimize[:, 0:48]
        trans: Float[Array, "b 1 3"] = param_to_optimize[:, npj.newaxis, 48:51]

        def left_func(
            x: tuple[Float[Array, "b 48"], Float[Array, "b 1 3"], Float[Array, "1 1"]],
        ) -> Float[Array, "b 21 3"]:
            return joint_fwd_left(x[0], x[1], x[2])

        def right_func(
            x: tuple[Float[Array, "b 48"], Float[Array, "b 1 3"], Float[Array, "1 1"]],
        ) -> Float[Array, "b 21 3"]:
            return joint_fwd_right(x[0], x[1], x[2])

        xyz_mano: Float[Array, "b 21 3"] = jax.lax.cond(is_left, left_func, right_func, (so3, trans, scale_param))
        xyz_mano_hom: Float[Array, "b 21 4"] = npj.concatenate([xyz_mano, npj.ones_like(xyz_mano)[..., 0:1]], axis=-1)

        uv_mano: Float[Array, "b n_views 21 2"] = proj_3d_vectorized(xyz_hom=xyz_mano_hom, P=Pall)

        # calculate residuals
        res_2d: Float[Array, "b n_views 21 2"] = uv_mano - uv_pred
        res_2d = npj.nan_to_num(res_2d * loss_weights["keypoint_2d"], nan=0.0)

        # Return the flattened vector of valid, weighted residuals
        return res_2d.flatten()

    return mv_2d_scaled_residual, joint_fwd_left, joint_fwd_right


class JointAndScaleOptimization:
    def __init__(
        self,
        xyz_template: Float[ndarray, "2 21 3"],
        Pall: Float[ndarray, "n_views 3 4"],
        loss_weights: LossWeights,
        num_iters: int = 30,
    ) -> None:
        """
        Pall - n, 3, 4 projection matrix
        loss_weights - dictionary containing how much value to give each portion
            of the cost function (2d, 3d, temporal)
        num_iters - how many iterations to optimize
        """

        batch_size = 1
        assert batch_size == 1, "Batch size must be 1 for this optimization"

        n_views: int = Pall.shape[0]

        self.num_iters: int = num_iters
        # Projection Matrix (n, 3, 4) where n is the number of cameras
        self.Pall: Float[Array, "batch 3 4"] = npj.array(Pall)

        self.loss_weights: LossWeights = loss_weights
        # use previous values to initialize, there should only ever be 1
        # hand model per frame
        self.so3_left_prev: Float[Array, "1 48"] = npj.zeros((1, 48))
        self.trans_left_prev: Float[Array, "1 3"] = npj.zeros((1, 3))

        self.so3_right_prev: Float[Array, "1 48"] = npj.zeros((1, 48))
        self.trans_right_prev: Float[Array, "1 3"] = npj.zeros((1, 3))

        # scale parameter is shared between left and right hand
        self.scale_init: Float[Array, "1"] = npj.ones((1))  # noqa UP037

        output_fns: tuple[ResidualFn, FwdKinematics, FwdKinematics] = make_mv_scaled_residual(
            xyz_template_left=npj.array(xyz_template[0]), xyz_template_right=npj.array(xyz_template[1])
        )

        residual_fn: ResidualFn = output_fns[0]
        self.joint_fwd_left: FwdKinematics = output_fns[1]
        self.joint_fwd_right: FwdKinematics = output_fns[2]

        # remove the need for two different optimizers, solvers ‘cholesky’, ‘inv’
        self.optimizer = LevenbergMarquardt(
            residual_fun=residual_fn, maxiter=self.num_iters, solver="cholesky", jit=True, xtol=1e-6, gtol=1e-6
        )
        # add jit
        print("Tracing JIT, can take a while...")
        init_params: Float[Array, "1 51"] = npj.concatenate([self.so3_left_prev, self.trans_left_prev], axis=-1)
        init_params = npj.concatenate([init_params.flatten(), self.scale_init], axis=0)

        uv_batch_init: Float[Array, "n_frames n_views 21 2"] = npj.zeros((1, n_views, 21, 2))
        _, _ = self.optimizer.run(
            init_params.flatten(),
            Pall=self.Pall,
            uv_pred=uv_batch_init,
            loss_weights=loss_weights,
            is_left=True,
        )
        self.optimizer = jit(self.optimizer.run)

        print("Trace Done")

    def __call__(
        self,
        uv_left_pred_batch: Float[ndarray, "n_views 21 2"],
        uv_right_pred_batch: Float[ndarray, "n_views 21 2"],
        calibrate: bool = False,
    ) -> tuple[OptimizationResults, LevenbergMarquardtState]:
        """
        pose_predictions_dict
            pose_predictions
            camera_dict
        """
        so3_optimized: Float[ndarray, "2 48"] = np.zeros((2, 48))
        trans_optimized: Float[ndarray, "2 3"] = np.zeros((2, 3))
        xyz_mano: Float[ndarray, "2 21 3"] = np.zeros((2, 21, 3))

        for hand_enum in HandSide:
            hand_side: Literal["left", "right"] = hand_enum.name.lower()
            hand_idx: int = hand_enum.value
            # get previous values, and extract pose_predictions
            match hand_side:
                case "left":
                    so3_prev: Float[Array, "1 48"] = self.so3_left_prev.copy()
                    trans_prev: Float[Array, "1 3"] = self.trans_left_prev.copy()
                    uv_pred_batch: Float[Array, "1 n_views 21 2"] = npj.array(uv_left_pred_batch)[npj.newaxis, ...]

                case "right":
                    so3_prev: Float[Array, "1 48"] = self.so3_right_prev.copy()
                    trans_prev: Float[Array, "1 3"] = self.trans_right_prev.copy()
                    uv_pred_batch: Float[Array, "1 n_views 21 2"] = npj.array(uv_right_pred_batch)[npj.newaxis, ...]

            # TODO initialize only rotation from wrist form either 3d procustus or mano preds
            if npj.any(npj.isnan(so3_prev)):
                so3_init: Float[Array, "1 48"] = npj.zeros_like(so3_prev)
            else:
                so3_init: Float[Array, "1 48"] = so3_prev

            if npj.any(npj.isnan(trans_prev)):
                trans_init: Float[Array, "1 3"] = npj.zeros_like(trans_prev)
            else:
                trans_init: Float[Array, "1 3"] = trans_prev

            init_params: Float[Array, "1 51"] = npj.concatenate([so3_init, trans_init], axis=-1)
            init_params: Float[Array, "_"] = npj.concatenate([init_params.flatten(), self.scale_init], axis=0)

            optimized_params, state = self.optimizer(
                init_params,
                Pall=self.Pall,
                uv_pred=uv_pred_batch,
                loss_weights=self.loss_weights,
                is_left=hand_side == "left",
            )

            # if np.isnan(optimized_params).any():
            #     continue

            optimized_scale: Float[Array, ""] = optimized_params[-1]
            optimized_params: Float[Array, "_"] = optimized_params[:-1]
            optimized_params: Float[Array, "1 51"] = optimized_params.reshape(1, 51)

            so3: Float[Array, "1 48"] = optimized_params[:, 0:48]
            trans: Float[Array, "1 3"] = optimized_params[:, 48:51]

            so3_optimized[0 if hand_side == "left" else 1] = np.array(so3[0])
            trans_optimized[0 if hand_side == "left" else 1] = np.array(trans[0])

            # pass optimized values to mano to extract 3d joints
            match hand_side:
                case "left":
                    self.so3_left_prev = so3
                    self.trans_left_prev = trans

                    xyz_mano_left: Float[Array, "1 21 3"] = self.joint_fwd_left(so3, trans[:, npj.newaxis, :])
                    xyz_mano[0] = np.array(xyz_mano_left[0])

                case "right":
                    self.so3_right_prev = so3
                    self.trans_right_prev = trans

                    xyz_mano_right: Float[Array, "1 21 3"] = self.joint_fwd_right(so3, trans[:, npj.newaxis, :])
                    xyz_mano[1] = np.array(xyz_mano_right[0])

        optimization_results = OptimizationResults(
            xyz_mano=xyz_mano,
            so3=so3_optimized,
            trans=trans_optimized,
        )

        return optimization_results, state
