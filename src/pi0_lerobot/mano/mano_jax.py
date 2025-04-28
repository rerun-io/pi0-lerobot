import jax.numpy as jnp
from einops import rearrange, repeat
from jax import jit
from jaxtyping import Array, Float

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


def quat2mat(wxyz_quat: Float[Array, "_ 4"]) -> Float[Array, "_ 3 3"]:
    """Converts a batch of quaternions to rotation matrices.

    Args:
        quat: A JAX array of shape (batch_size, 4) representing quaternions (w, x, y, z).

    Returns:
        A JAX array of shape (batch_size, 3, 3) representing the corresponding rotation matrices.
    """
    norm_wxyz_quat = wxyz_quat
    norm_wxyz_quat = norm_wxyz_quat / jnp.linalg.norm(norm_wxyz_quat + 1e-8, axis=1, keepdims=True)
    w, x, y, z = norm_wxyz_quat[:, 0], norm_wxyz_quat[:, 1], norm_wxyz_quat[:, 2], norm_wxyz_quat[:, 3]

    batch_size = wxyz_quat.shape[0]

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


def batch_rodrigues(axisang: Float[Array, "_ 3"]) -> Float[Array, "_ 9"]:
    """Converts a batch of axis-angle representations to rotation matrices.

    Args:
        axisang: A batch of axis-angle vectors, shape (N, 3).

    Returns:
        A batch of flattened 3x3 rotation matrices, shape (N, 9).
        The conversion is done by first converting axis-angle to quaternions,
        and then converting quaternions to rotation matrices.
    """
    # axisang N x 3
    axisang_norm = jnp.linalg.norm(axisang + 1e-8, axis=1)
    angle = axisang_norm[..., jnp.newaxis]
    axisang_normalized = axisang / angle
    angle = angle * 0.5
    v_cos = jnp.cos(angle)
    v_sin = jnp.sin(angle)
    quat = jnp.concatenate((v_cos, v_sin * axisang_normalized), 1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.reshape(rot_mat.shape[0], 9)
    return rot_mat


def posemap_axisang(pose_vectors: Float[Array, "batch n_betas=48"]) -> Float[Array, "batch n_rotmats_flat=144"]:
    """Converts batch of axis-angle pose vectors to rotation matrices.

    This function takes a batch of pose vectors, where each vector represents
    the axis-angle parameters for multiple hand joints (3 parameters per joint).
    It reshapes the input, applies the Rodrigues' rotation formula via the
    `batch_rodrigues` function to convert each 3D axis-angle vector into a
    3x3 rotation matrix, and then flattens these matrices back into a single
    vector per batch item.

    Args:
        pose_vectors: A JAX array of shape (batch_size, num_joints * 3)
            containing the axis-angle pose parameters. For MANO, num_joints is
            typically 16, resulting in an input shape of (batch_size, 48).

    Returns:
        A JAX array of shape (batch_size, num_joints * 9) containing the
        flattened 3x3 rotation matrices for each joint in the batch. For MANO,
        this results in an output shape of (batch_size, 144).
    """
    num_joints: int = int(pose_vectors.shape[1] / 3)
    # merge batch dimension and num_joints dimension (so batch 10 num joints 16 -> 160)
    pose_vec_reshaped = pose_vectors.reshape(-1, 3)
    rot_mats = batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.reshape(pose_vectors.shape[0], num_joints * 9)
    return rot_mats


def with_zeros(tensor: Float[Array, "batch 3 4"]) -> Float[Array, "batch 4 4"]:
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


class JointsOnly:
    """
    A class to compute joint positions based on pose coefficients, translation, and optional scaling.

    Args:
        template_joints: A JAX array of shape (21, 3) representing the template joint positions.
                         This parameter replaces the old 'side' argument and determines the initial
                         joint configuration (e.g., for left or right hand).
    """

    def __init__(
        self,
        template_joints: Float[Array, "21 3"],
    ) -> None:
        self.joints: Float[Array, "21 3"] = template_joints

    def __call__(
        self,
        pose_coeffs: Float[Array, "batch 48"],
        trans: Float[Array, "batch 1 3"],
        scale: Float[Array, "batch 1"] | None = None,
    ) -> Float[Array, "batch 21 3"]:
        r"""Produces Joints based on the pose/translation."""
        batch_size: int = pose_coeffs.shape[0]
        # ------------------------------------------------------------------
        # Step 1: Get axis-angle from PCA components and coefficients
        # ------------------------------------------------------------------
        full_pose: Float[Array, "b betas=48"] = pose_coeffs

        # ------------------------------------------------------------------
        # Step 2: Get rotation matricies and pose maps from axis-angle
        # pose maps are relative to the current rotation so zero when in rest pose
        # ------------------------------------------------------------------
        rot_map: Float[Array, "b n_rotmat_flat=144"] = posemap_axisang(full_pose)
        root_rot: Float[Array, "b 3 3"] = rot_map[:, :9].reshape(batch_size, 3, 3)
        # seperate root joint rotation from total rotation
        rot_map = rot_map[:, 9:]

        # ------------------------------------------------------------------
        # Step 3 – Regress nominal joint locations (still in rest pose)
        # This is just provided and not regressed in joints only
        # ------------------------------------------------------------------
        # convert to batched version (b, 21, 3)
        # multiply by scale if provided, if not just use 1
        j: Float[Array, "b n_joints=21 dim=3"] = jnp.tile(self.joints, (batch_size, 1, 1))
        if scale is not None:
            j = j * scale.reshape(batch_size, 1, 1)

        # ------------------------------------------------------------------
        # Step 4 – Forward kinematics: build SE(3) for each joint
        # ------------------------------------------------------------------
        root_j: Float[Array, "b dim=3 1"] = j[:, 0, :].reshape(batch_size, 3, 1)
        # transformation matrix (rot, trans)
        root_trans: Float[Array, "b 4 4"] = with_zeros(jnp.concatenate((root_rot, root_j), 2))

        # all other rotations other than the root rotation, convert from flat -> (3, 3)
        all_rots = rot_map.reshape(rot_map.shape[0], 15, 3, 3)  # (b, 15, 3, 3)
        lev1_idxs: list[int] = [1, 4, 7, 10, 13]
        lev2_idxs: list[int] = [2, 5, 8, 11, 14]
        lev3_idxs: list[int] = [3, 6, 9, 12, 15]
        # additional finger tip rotations
        lev4_idxs: list[int] = [16, 17, 18, 19, 20]

        # rotation matricies
        lev1_rots: Float[Array, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots: Float[Array, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots: Float[Array, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        # finger tip rotations don't matter, just use idx for level 3
        lev4_rots: Float[Array, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev3_idxs]]

        # position matricies
        lev1_j: Float[Array, "b 5 3"] = j[:, lev1_idxs]
        lev2_j: Float[Array, "b 5 3"] = j[:, lev2_idxs]
        lev3_j: Float[Array, "b 5 3"] = j[:, lev3_idxs]
        lev4_j: Float[Array, "b 5 3"] = j[:, lev4_idxs]

        # generate list of all transformation matricies, starting with root
        all_transforms: list[Float[Array, "b n_T 4 4"]] = [root_trans[:, jnp.newaxis, ...]]

        # From base to tips
        # Get lev1 results
        lev1_j_rel: Float[Array, "b 5 dim=3"] = lev1_j - rearrange(root_j, "b dim n -> b n dim")
        lev1_rel_transform_flt: Float[Array, "_ 4 4"] = with_zeros(
            jnp.concatenate((lev1_rots, lev1_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )
        # generate tiled transformation matrix to match up with all fingers
        root_trans_flt: Float[Array, "_ 4 4"] = repeat(root_trans, "b m1 m2 -> (b f) m1 m2", f=5)
        # perform transform and append to list
        lev1_flt: Float[Array, "_ 4 4"] = root_trans_flt @ lev1_rel_transform_flt
        all_transforms.append(lev1_flt.reshape(batch_size, 5, 4, 4))

        # Get lev2 results
        lev2_j_rel: Float[Array, "b 5 dim=3"] = lev2_j - lev1_j
        lev2_rel_transform_flt: Float[Array, "_ 4 4"] = with_zeros(
            jnp.concatenate((lev2_rots, lev2_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )
        lev2_flt: Float[Array, "_ 4 4"] = lev1_flt @ lev2_rel_transform_flt
        all_transforms.append(lev2_flt.reshape(batch_size, 5, 4, 4))

        # Get lev3 results
        lev3_j_rel: Float[Array, "b 5 dim=3"] = lev3_j - lev2_j
        lev3_rel_transform_flt: Float[Array, "_ 4 4"] = with_zeros(
            jnp.concatenate((lev3_rots, lev3_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )
        lev3_flt: Float[Array, "_ 4 4"] = lev2_flt @ lev3_rel_transform_flt
        all_transforms.append(lev3_flt.reshape(batch_size, 5, 4, 4))

        # Get lev4 results
        lev4_j_rel: Float[Array, "b 5 dim=3"] = lev4_j - lev3_j
        lev4_rel_transform_flt: Float[Array, "_ 4 4"] = with_zeros(
            jnp.concatenate((lev4_rots, lev4_j_rel[..., jnp.newaxis]), 3).reshape(-1, 3, 4)
        )

        lev4_flt: Float[Array, "_ 4 4"] = lev3_flt @ lev4_rel_transform_flt
        all_transforms.append(lev4_flt.reshape(batch_size, 5, 4, 4))

        # w 0 i 1, m , l, r, t
        reorder_idxs: list[int] = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15, 16, 17, 18, 19, 20]
        # # convert from list -> array, and reorder
        results: Float[Array, "b n_joints=21 4 4"] = jnp.concatenate(all_transforms, 1)[:, reorder_idxs]
        results_global: Float[Array, "b n_joints=21 4 4"] = results

        jtr: Float[Array, "b n_joints=21 3"] = results_global[:, :, :3, 3]
        # Reorder joints to match visualization utilities, w, t, i, m, r, l
        jtr = jtr[:, [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]]
        jtr = jtr + trans
        return jtr


mano_j_right = JointsOnly(template_joints=right_joints)
mano_j_left = JointsOnly(template_joints=left_joints)
mano_j_right = jit(mano_j_right)
mano_j_left = jit(mano_j_left)
