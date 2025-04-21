from typing import Literal

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float

# These are in meters
right_joints: Float[Array, "21 3"] = jnp.array(
    [
        [0.09566993, 0.00638343, 0.00618631],
        [0.00757269, 0.00118307, 0.02687230],
        [-0.0251062, 0.00519243, 0.02908936],
        [-0.0472621, 0.00389400, 0.02897524],
        [0.00100949, 0.00490446, 0.00282876],
        [-0.0301731, 0.00676579, -0.0027657],
        [-0.0530778, 0.00551369, -0.0067102],
        [0.02688296, -0.0035569, -0.0370230],
        [0.00986855, -0.0034950, -0.0495218],
        [-0.0059983, -0.0041862, -0.0598537],
        [0.01393438, 0.00242601, -0.0204868],
        [-0.0143799, 0.00449301, -0.0255854],
        [-0.0379004, 0.00280490, -0.0332192],
        [0.07158022, -0.0091389, 0.03199915],
        [0.05194698, -0.0082476, 0.05569870],
        [0.02972924, -0.0136805, 0.07022282],
        [-0.0705249, 0.00461197, 0.03302451],
        [-0.0789928, 0.00614665, -0.0120408],
        [-0.0218988, -0.0016281, -0.0701316],
        [-0.0608042, 0.00734306, -0.0402022],
        [0.00223126, -0.0180950, 0.09091450],
    ]
)

left_joints: Float[Array, "21 3"] = jnp.array(
    [
        [-0.0956699, 0.00638343, 0.00618631],
        [-0.0075726, 0.00118307, 0.02687230],
        [0.02510622, 0.00519243, 0.02908936],
        [0.04726213, 0.00389400, 0.02897524],
        [-0.0010094, 0.00490446, 0.00282876],
        [0.03017318, 0.00676579, -0.0027657],
        [0.05307782, 0.00551369, -0.0067102],
        [-0.0268829, -0.0035569, -0.0370230],
        [-0.0098685, -0.0034950, -0.0495218],
        [0.00599835, -0.0041862, -0.0598537],
        [-0.0139343, 0.00242601, -0.0204868],
        [0.01437990, 0.00449301, -0.0255854],
        [0.03790041, 0.00280490, -0.0332192],
        [-0.0715802, -0.0091389, 0.03199915],
        [-0.0519469, -0.0082476, 0.05569870],
        [-0.0297292, -0.0136805, 0.07022282],
        [0.07052490, 0.00461197, 0.03302451],
        [0.07899282, 0.00614665, -0.0120408],
        [0.02189885, -0.0016281, -0.0701316],
        [0.06080423, 0.00734306, -0.0402022],
        [-0.0022312, -0.0180950, 0.09091450],
    ]
)

joints_template_meters: Float[Array, "2 21 3"] = jnp.stack([left_joints, right_joints], axis=0)


class JointsOnly:
    def __init__(
        self,
        side: Literal["left", "right"] = "right",
    ) -> None:
        self.joints = joints_template_meters[0 if side == "left" else 1]

    def _posemap_axisang(self, pose_vectors: Float[Array, "batch 48"]) -> Float[Array, "batch 144"]:
        """
        batch 3, 16 -> batch 48
        batch 3*48 -> batch 144 because its a 3x3 matrix so an extra 3 gets multipled
        """
        num_joints = int(pose_vectors.shape[1] / 3)
        # merge batch dimension and num_joints dimension (so batch 10 num joints 16 -> 160)
        pose_vec_reshaped = pose_vectors.reshape(-1, 3)
        rot_mats = self._batch_rodrigues(pose_vec_reshaped)
        rot_mats = rot_mats.reshape(pose_vectors.shape[0], num_joints * 9)
        return rot_mats

    def _quat2mat(self, quat: Float[Array, "_ 4"]) -> Float[Array, "_ 3 3"]:
        norm_quat = quat
        norm_quat = norm_quat / jnp.linalg.norm(norm_quat + 1e-8, axis=1, keepdims=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

        batch_size = quat.shape[0]

        w2, x2, y2, z2 = w**2, x**2, y**2, z**2
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = jnp.stack(
            [
                w2 + x2 - y2 - z2,
                2 * xy - 2 * wz,
                2 * wy + 2 * xz,
                2 * wz + 2 * xy,
                w2 - x2 + y2 - z2,
                2 * yz - 2 * wx,
                2 * xz - 2 * wy,
                2 * wx + 2 * yz,
                w2 - x2 - y2 + z2,
            ],
            axis=1,
        ).reshape(batch_size, 3, 3)
        return rotMat

    def _batch_rodrigues(self, axisang: Float[Array, "_ 3"]) -> Float[Array, "_ 9"]:
        # axisang N x 3
        axisang_norm = jnp.linalg.norm(axisang + 1e-8, axis=1)
        angle = axisang_norm[..., jnp.newaxis]
        axisang_normalized = axisang / angle
        angle = angle * 0.5
        v_cos = jnp.cos(angle)
        v_sin = jnp.sin(angle)
        quat = jnp.concatenate((v_cos, v_sin * axisang_normalized), 1)
        rot_mat = self._quat2mat(quat)
        rot_mat = rot_mat.reshape(rot_mat.shape[0], 9)
        return rot_mat

    def _with_zeros(self, tensor: Float[Array, "batch 3 4"]) -> Float[Array, "batch 4 4"]:
        """
        add zeros to the bottom of transformation matrix
        example
        [[[1.         0.         0.         0.09566993]
          [0.         1.         0.         0.00638343]
          [0.         0.         1.         0.00618631]]]

        to

        [[[1.         0.         0.         0.09566993]
          [0.         1.         0.         0.00638343]
          [0.         0.         1.         0.00618631]
          [0.         0.         0.         1.        ]]]
        """
        batch_size = tensor.shape[0]
        padding = jnp.array([0.0, 0.0, 0.0, 1.0])

        # adjust to fit batchsize
        concat_list = (tensor, jnp.tile(padding.reshape(1, 1, 4), (batch_size, 1, 1)))
        cat_res = jnp.concatenate(concat_list, 1)
        return cat_res

    def __call__(
        self,
        pose_coeffs: Float[Array, "batch 48"],
        trans: Float[Array, "batch 1 3"],
    ) -> Float[Array, "batch 21 3"]:
        r"""Produces Joints based on the pose/translation."""
        batch_size: int = pose_coeffs.shape[0]
        # Concatenate back global rot
        full_pose = pose_coeffs
        # compute rotation matrixes from axis-angle while skipping global rotation
        rot_map = self._posemap_axisang(
            full_pose
        )  # (1, 144) or (1, 16*9) for 16 joints and 9 values in 1 (3,3) rot matrix
        root_rot = rot_map[:, :9].reshape(batch_size, 3, 3)
        # seperate root joint rotation from total rotation
        rot_map = rot_map[:, 9:]

        # convert to batched version (b, 21, 3)
        j = jnp.tile(self.joints, (batch_size, 1, 1))

        # Global rigid transformation
        root_j = j[:, 0, :].reshape(batch_size, 3, 1)
        # transformation matrix (rot, trans)
        root_trans = self._with_zeros(jnp.concatenate((root_rot, root_j), 2))  # (b, 4, 4)

        # all other rotations other than the root rotation, convert from flat -> (3, 3)
        all_rots = rot_map.reshape(rot_map.shape[0], 15, 3, 3)  # (b, 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev4_idxs = [16, 17, 18, 19, 20]

        # rotation matricies
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        # finger tip rotations don't matter, just use idx for level 3
        lev4_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]

        # position matricies
        lev1_j = j[:, lev1_idxs]
        lev2_j = j[:, lev2_idxs]
        lev3_j = j[:, lev3_idxs]
        lev4_j = j[:, lev4_idxs]

        # generate list of all transformation matricies, starting with root
        all_transforms = [root_trans[:, jnp.newaxis, ...]]

        # From base to tips
        # Get lev1 results
        lev1_j_rel = lev1_j - root_j.transpose((0, 2, 1))
        lev1_rel_transform_flt = self._with_zeros(
            jnp.concatenate((lev1_rots, lev1_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )
        # generate tiled transformation matrix to match up with all fingers
        root_trans_flt = jnp.tile(root_trans[:, jnp.newaxis, ...], (1, 5, 1, 1))
        # squeeze out extra dimension (1, 5, 4, 4) -> (5, 4, 4) for 5 transformation matricies
        root_trans_flt = root_trans_flt.reshape(root_trans.shape[0] * 5, 4, 4)
        # peroform transform and append to list
        lev1_flt = jnp.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = self._with_zeros(
            jnp.concatenate((lev2_rots, lev2_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )
        lev2_flt = jnp.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = self._with_zeros(
            jnp.concatenate((lev3_rots, lev3_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )
        lev3_flt = jnp.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # Get lev4 results
        lev4_j_rel = lev4_j - lev3_j
        lev4_rel_transform_flt = self._with_zeros(
            jnp.concatenate((lev4_rots, lev4_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )

        lev4_flt = jnp.matmul(lev3_flt, lev4_rel_transform_flt)
        all_transforms.append(lev4_flt.reshape(all_rots.shape[0], 5, 4, 4))

        # w 0 i 1, m , l, r, t
        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15, 16, 17, 18, 19, 20]
        # # convert from list -> array, and reorder
        results = jnp.concatenate(all_transforms, 1)[:, reorder_idxs]
        results_global = results

        jtr = results_global[:, :, :3, 3]
        # Reorder joints to match visualization utilities, w, t, i, m, r, l
        jtr = jtr[:, [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]]
        jtr = jtr + trans
        return jtr


mano_j_right = JointsOnly(side="right")
mano_j_left = JointsOnly(side="left")
mano_j_right = jit(mano_j_right)
mano_j_left = jit(mano_j_left)
