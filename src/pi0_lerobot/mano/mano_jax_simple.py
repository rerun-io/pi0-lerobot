from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jaxtyping import Array, Float, Float64, Int
from numpy import ndarray
from serde.pickle import from_pickle

from pi0_lerobot.mano.kinematic_hand_jax import posemap_axisang, with_zeros
from pi0_lerobot.mano.mano_pytorch_simple import MANOData


def subtract_flat_id(rot_mats: Float[Array, "b 144"]) -> Float[Array, "b 144"]:
    """Subtracts the identity matrix (flattened and repeated) from a batch of flattened rotation matrices.

    Args:
        rot_mats: A tensor of shape (b, 144) representing a batch of 16 flattened 3x3 rotation matrices.

    Returns:
        A tensor of shape (b, 144) where each 9-element chunk has the flattened identity matrix subtracted.
    """
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    # Create a 3x3 identity matrix, reshape to (1, 9)
    eye_matrix_flat = jnp.eye(3, dtype=rot_mats.dtype).reshape(1, 9)
    # Tile it to match the batch size and number of rotation matrices
    id_flat = jnp.tile(eye_matrix_flat, (rot_mats.shape[0], rot_nb))
    # id_flat.requires_grad = False # JAX handles gradients differently
    results = rot_mats - id_flat
    return results


class ManoJaxLayer:
    def __init__(
        self,
        ncomps: Literal[45] = 45,
        side: Literal["left", "right"] = "right",
        mano_root: Path = Path("mano/models"),
        use_pca: bool = True,
    ) -> None:
        """Initializes the ManoSimpleLayer.

        This layer provides a simplified, non-trainable implementation of the MANO hand model,
        primarily intended for understanding, visualization, and debugging purposes. It loads
        pre-defined MANO model parameters from a .pkl file.

        Note:
            This layer is NOT designed for use within a training pipeline due to its
            simplified nature and lack of gradient tracking for some operations.

            ncomps: The number of principal components (PCA) to use for the pose
                representation. Currently fixed at 45 for this simplified version.
                Defaults to 45.
            side: Specifies whether to load the 'left' or 'right' hand model.
                Defaults to "right".
            mano_root: The directory path containing the MANO model .pkl files
                (e.g., MANO_LEFT.pkl, MANO_RIGHT.pkl). Defaults to "mano/models".
            use_pca: Determines whether to use the PCA representation for hand pose.
                If False, raw joint rotations (axis-angle) are expected. Defaults to True.


        Attributes:
            side (Literal["left", "right"]): The side of the hand model loaded.
            use_pca (bool): Flag indicating if PCA pose representation is used.
            joint_rot_mode (Literal["axisang"]): Joint rotation mode (ignored if use_pca is True).
            ncomps (int): Number of PCA components (fixed at 45).
            mano_path (Path): Full path to the loaded MANO .pkl file.
            th_shapedirs (Tensor): Tensor containing shape blend shapes. Shape: (778, 3, 10).
            th_posedirs (Tensor): Tensor containing pose blend shapes. Shape: (778, 3, 135).
            th_v_template (Tensor): Template vertices of the hand model. Shape: (1, 778, 3).
            th_J_regressor (Tensor): Regressor matrix to compute joint locations from vertices. Shape: (16, 778).
            th_weights (Tensor): Linear blend skinning weights. Shape: (778, 16).
            th_faces (Tensor): Triangle face indices for the mesh. Shape: (1538, 3).
            n_verts (int): Number of vertices in the hand model (778).
            th_hands_mean (Tensor): Mean pose in PCA space. Shape: (1, 45).
            th_comps (Tensor): PCA components for pose. Shape: (45, 45).
            kintree_table (ndarray): Kinematic tree structure defining parent-child joint relationships. Shape: (2, 16).
            kintree_parents (list[int]): List mapping each joint to its parent joint index.
        """
        super().__init__()

        self.side: Literal["left", "right"] = side
        self.use_pca: bool = use_pca
        self.ncomps = ncomps

        self.mano_path: Path = mano_root / f"MANO_{side.upper()}.pkl"

        with open(self.mano_path, "rb") as f:
            binary_data: bytes = f.read()
        # use serde to deserialize the binary data, ensures type safety
        mano_data: MANOData = from_pickle(MANOData, binary_data)

        hands_components: Float64[ndarray, "45 45"] = mano_data.hands_components

        # Shape‑blend basis: vertex offsets per β‑shape coefficient
        self.shapedirs: Float[Array, "n_verts=778 dim=3 n_betas=10"] = jnp.array(mano_data.shapedirs)
        # Pose‑blend basis: pose‑dependent wrinkle/volume corrections
        self.posedirs: Float[Array, "n_verts=778 dim=3 n_pose_dims=135"] = jnp.array(mano_data.posedirs)
        # Neutral/rest‑pose mesh in metres (unsqueezed for fake batch dim)
        self.v_template: Float[Array, "1 n_verts=778 dim=3"] = jnp.array(mano_data.v_template)[jnp.newaxis, :]
        # Linear regressor V → J: gives 16 skeletal joint centres
        self.J_regressor: Float[Array, "n_joints=16 n_verts=778"] = jnp.array(mano_data.J_regressor)
        # Skinning weights (rows sum to 1): influence of each joint on each vertex
        self.weights: Float[Array, "n_verts=778 n_joints=16"] = jnp.array(mano_data.weights)
        # Triangle indices for rendering/export (int32 → int64 for PyTorch)
        self.faces: Int[Array, "n_faces=1538 3"] = jnp.array(mano_data.f, dtype=np.int32)
        # Convenience alias for vertex count (778)
        self.n_verts: int = self.v_template.shape[1]

        # Get hand mean
        self.th_hands_mean: Float[Array, "1 45"] = jnp.array(mano_data.hands_mean)[jnp.newaxis, :]
        # PCA basis for hand pose
        self.th_comps: Float[Array, "45 45"] = jnp.array(hands_components)

        # Kinematic chain params
        self.kintree_table: Int[Array, "2 16"] = jnp.array(mano_data.kintree_table)
        parents: list[int] = list(self.kintree_table[0].tolist())
        self.kintree_parents: list[int] = parents

    def forward(
        self,
        pose_coeffs: Float[Array, "b n_poses=48"],
        betas: Float[Array, "b n_betas=10"],
        trans: Float[Array, "b dim=3"],
    ) -> tuple[Float[Array, "b n_verts=778 3"], Float[Array, "b joints_and_tips=21 3"]]:
        """
        Forward pass for the MANO layer.

        PCA is always used for the pose space in this simplified version.

        Applies shape parameters, pose parameters, and global translation to the MANO
        template mesh to produce posed hand vertices and joints.

            th_pose_coeffs: Tensor (batch_size x 48). Hand pose parameters.
                The first `self.rot` (e.g., 3 or 6) elements represent the global root rotation.
                The remaining `self.ncomps` (e.g., 45) elements represent the 15 joint rotations
                (e.g., axis-angle or PCA coefficients).
            th_betas: Tensor (batch_size x 10). Shape parameters (betas) for the hand.
            th_trans: Tensor (batch_size x 3). Global translation applied to vertices and joints.
                If None or zero norm, the output might be centered based on `self.center_idx`.

        Returns:
            A tuple containing:
            - th_verts: Tensor (batch_size x 778 x 3). The posed hand mesh vertices in millimeters.
            - th_jtr: Tensor (batch_size x 21 x 3). The posed hand joints (16 MANO joints + 5 fingertips)
              in millimeters, reordered for compatibility with visualization tools.
        """

        batch_size: int = pose_coeffs.shape[0]

        ###############################################################
        # Step 1: Get axis-angle from PCA components and coefficients #
        ###############################################################
        # Remove global rot coeffs (45 = 15*3)
        hand_pose_coeffs: Float[Array, "b betas=45"] = pose_coeffs[:, 3 : 3 + 45]
        # PCA components --> axis angles if use_pca is True
        # [b 45] @ [45 45]-> [b 45]
        full_hand_pose: Float[Array, "b betas=45"] = hand_pose_coeffs @ self.th_comps

        # Concatenate back global rot
        full_pose: Float[Array, "b betas=48"] = jnp.concatenate(
            [pose_coeffs[:, :3], self.th_hands_mean + full_hand_pose], 1
        )

        # ------------------------------------------------------------------
        # Step 2: Get rotation matricies and pose maps from axis-angle
        # pose maps are relative to the current rotation so zero when in rest pose
        # ------------------------------------------------------------------

        # compute rotation matrixes from axis-angle while skipping global rotation (16*9 = 144)
        rot_map: Float[Array, "b n_rotmat_flat=144"] = posemap_axisang(full_pose)  # used to pose verts
        pose_map: Float[Array, "b n_rotmat_flat=144"] = subtract_flat_id(rot_map)  # used to pose joints

        root_rot: Float[Array, "b 3 3"] = rot_map[:, :9].reshape(batch_size, 3, 3)
        rot_map: Float[Array, "b 135"] = rot_map[:, 9:]
        pose_map: Float[Array, "b 135"] = pose_map[:, 9:]

        # ------------------------------------------------------------------
        # Step 3 – Shape blend: betas deform the template geometry
        # ------------------------------------------------------------------

        # [778, 3, 10] @ [b, 10].T -> [778, 3, 10] @ [10, b] = [778, 3, b]
        v_shaped: Float[Array, "n_verts=778 dim=3 b "] = self.shapedirs @ rearrange(betas, "b n_betas -> n_betas b")
        v_shaped: Float[Array, "b n_verts=778 dim=3 "] = (
            rearrange(v_shaped, "n_verts dim b -> b n_verts dim") + self.v_template
        )

        # ------------------------------------------------------------------
        # Step 4 – Regress nominal joint locations (still in rest pose)
        # ------------------------------------------------------------------

        # [16 778] @ [b, 778, 3] -> [b, 16, 3]
        joints: Float[Array, "b n_joints=16 dim=3"] = self.J_regressor @ v_shaped

        # ------------------------------------------------------------------
        # Step 5 – Pose‑dependent corrective offsets (wrinkles, volume)
        # ------------------------------------------------------------------

        # [778 3 135] @ [b 135].T -> [778 3 135] @ [135 b] = [778 3 b]
        pose_map: Float[Array, "n_verts=778 dim=3 b"] = self.posedirs @ rearrange(
            pose_map, "b n_pose_map -> n_pose_map b"
        )
        v_posed: Float[Array, "b n_verts=778 dim=3"] = v_shaped + rearrange(pose_map, "n_verts dim b -> b n_verts dim")
        # Final T pose with transformation done !

        # ------------------------------------------------------------------
        # Step 6 – Forward kinematics: build SE(3) for each joint
        # ------------------------------------------------------------------
        # get the root joint of each batch
        root_j: Float[Array, "b dim=3 1"] = joints[:, 0, :].reshape(batch_size, 3, 1)
        # trans here refers to transformation matrix
        root_trans: Float[Array, "b 4 4"] = with_zeros(jnp.concatenate([root_rot, root_j], 2))

        all_rots: Float[Array, "b 15 3 3"] = rot_map.reshape(batch_size, 15, 3, 3)
        # levels refer to joint levels in the kinematic tree (does not include fingertips)
        lev1_idxs: list[int] = [1, 4, 7, 10, 13]
        lev2_idxs: list[int] = [2, 5, 8, 11, 14]
        lev3_idxs: list[int] = [3, 6, 9, 12, 15]

        # initialize rotations and translations for each level
        lev1_rots: Float[Array, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots: Float[Array, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots: Float[Array, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j: Float[Array, "b 5 3"] = joints[:, lev1_idxs]
        lev2_j: Float[Array, "b 5 3"] = joints[:, lev2_idxs]
        lev3_j: Float[Array, "b 5 3"] = joints[:, lev3_idxs]

        # From base to level before tips
        # Get lev1 results
        all_transforms: list[Float[Array, "b n_T 4 4"]] = [root_trans[:, jnp.newaxis, :, :]]
        lev1_j_rel: Float[Array, "b 5 dim=3"] = lev1_j - rearrange(root_j, "b dim n -> b n dim")
        lev1_rel_transform_flt: Float[Array, "_ 4 4"] = with_zeros(
            jnp.concatenate([lev1_rots, lev1_j_rel[:, :, :, jnp.newaxis]], axis=3).reshape(-1, 3, 4)
        )
        # generate tiled root transformation matrix to match up with all fingers
        root_trans_flt: Float[Array, "_ 4 4"] = repeat(root_trans, "b m1 m2 -> (b f) m1 m2", f=5)
        # [b*5 4 4] @ [b*5 4 4] -> [b*5 4 4]
        lev1_flt: Float[Array, "_ 4 4"] = root_trans_flt @ lev1_rel_transform_flt
        # reshape back to [b 5 4 4]
        all_transforms.append(lev1_flt.reshape(batch_size, 5, 4, 4))

        # Get lev2 results
        lev2_j_rel: Float[Array, "b 5 dim=3"] = lev2_j - lev1_j
        lev2_rel_transform_flt: Float[Array, "_ 4 4"] = with_zeros(
            jnp.concatenate([lev2_rots, lev2_j_rel[:, :, :, jnp.newaxis]], axis=3).reshape(-1, 3, 4)
        )
        lev2_flt: Float[Array, "_ 4 4"] = lev1_flt @ lev2_rel_transform_flt
        all_transforms.append(lev2_flt.reshape(batch_size, 5, 4, 4))

        # Get lev3 results
        lev3_j_rel: Float[Array, "b 5 dim=3"] = lev3_j - lev2_j
        lev3_rel_transform_flt: Float[Array, "_ 4 4"] = with_zeros(
            jnp.concatenate([lev3_rots, lev3_j_rel[:, :, :, jnp.newaxis]], axis=3).reshape(-1, 3, 4)
        )
        lev3_flt: Float[Array, "_ 4 4"] = jnp.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.reshape(batch_size, 5, 4, 4))

        reorder_idxs: list[int] = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        results: Float[Array, "b n_joints=16 4 4"] = jnp.concatenate(all_transforms, axis=1)[:, reorder_idxs]
        results_global: Float[Array, "b n_joints=16 4 4"] = results

        # ------------------------------------------------------------------
        # Step 7 – Linear Blend Skinning (core of LBS)
        # ------------------------------------------------------------------

        # perform linear blend skinning to get verts
        joint_js: Float[Array, "b n_joints=16 4"] = jnp.concatenate(
            [joints, jnp.zeros((batch_size, 16, 1), dtype=joints.dtype)], axis=2
        )
        # [b 16 4 4] @ [b 16 4 1] -> [b 16 4 4] @ [b 16 4 1] -> [b 16 4 1]
        tmp2: Float[Array, "b n_joints=16 4 1"] = results @ rearrange(joint_js, "b n_joints m -> b n_joints m 1")

        zeros_mat: Float[Array, "b n_joints=16 4 3"] = jnp.zeros((batch_size, 16, 4, 3), dtype=results.dtype)
        zeros_mat: Float[Array, "b n_joints=16 4 4"] = jnp.concatenate([zeros_mat, tmp2], axis=3)

        results2: Float[Array, "b 4 4 n_joints=16"] = rearrange(
            (results - zeros_mat), "b n_joints m n -> b m n n_joints"
        )

        # [b 4 4 16] @ [778 16].T -> [b 4 4 16] @ [16 778] = [b 4 4 778]
        th_T: Float[Array, "b 4 4 n_verts=778"] = results2 @ rearrange(
            self.weights, "n_verts n_joints -> n_joints n_verts"
        )

        # convert to homogeneous coordinates
        rest_shape_h: Float[Array, "b 4 n_verts=778"] = jnp.concatenate(
            [
                jnp.transpose(v_posed, (0, 2, 1)),
                jnp.ones((batch_size, 1, v_posed.shape[1]), dtype=th_T.dtype),
            ],
            axis=1,
        )

        rest_shape_h: Float[Array, "b 1 4 n_verts=778"] = rearrange(rest_shape_h, "b m n_verts -> b 1 m n_verts")
        # [b 4 4 778] * [b 1 4 n_verts=778] -> [b 4 4 n_verts=778].sum(2) -> [b 4 n_verts=778]
        verts: Float[Array, "b 4 n_verts=778"] = (th_T * rest_shape_h).sum(axis=2)
        verts: Float[Array, "b n_verts=778 4"] = jnp.transpose(verts, (0, 2, 1))
        verts: Float[Array, "b n_verts=778 3"] = verts[:, :, :3]

        # ------------------------------------------------------------------
        # Step 8 – Add fingertip pseudo‑joints
        # ------------------------------------------------------------------
        jtr: Float[Array, "b n_joints=16 3"] = results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        tip_indices = jnp.array([745, 319, 444, 556, 673])
        tips: Float[Array, "b 5 3"] = verts[:, tip_indices]
        jtr: Float[Array, "b joints_and_tips=21 3"] = jnp.concatenate([jtr, tips], axis=1)

        # ------------------------------------------------------------------
        # Step 9 – Re‑order joints for downstream visualisers
        # ------------------------------------------------------------------
        reorder_jtr_indices = jnp.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
        jtr: Float[Array, "b joints_and_tips=21 3"] = jtr[:, reorder_jtr_indices]

        # ------------------------------------------------------------------
        # Step 10 – Apply global translation
        # ------------------------------------------------------------------
        jtr: Float[Array, "b joints_and_tips=21 3"] = jtr + rearrange(trans, "b dim -> b 1 dim")
        verts: Float[Array, "b n_verts=778 3"] = verts + rearrange(trans, "b dim -> b 1 dim")

        # ------------------------------------------------------------------
        # Step 11 – Convert from metres → millimetres
        # ------------------------------------------------------------------
        verts: Float[Array, "b n_verts=778 3"] = verts * 1000
        jtr: Float[Array, "b joints_and_tips=21 3"] = jtr * 1000

        return verts, jtr
