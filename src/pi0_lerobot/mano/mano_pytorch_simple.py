import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float, Float64, Int64
from numpy import ndarray
from serde import serde
from serde.pickle import from_pickle
from torch import Tensor
from torch.nn import Module


def quat2mat(quat: Float[Tensor, "_ 4"]) -> Float[Tensor, "_ 3 3"]:
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
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
        dim=1,
    ).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang: Float[Tensor, "_ 3"]) -> Float[Tensor, "_ 9"]:
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def th_posemap_axisang(pose_vectors: Float[Tensor, "b betas=48"]) -> Float[Tensor, "b n_rotmat_flat=144"]:
    """
    Converts batch of axis-angle pose representations to rotation matrices.

    This function takes a batch of pose vectors, where each pose is represented
    by concatenated axis-angle vectors (3 values per rotation). It uses the
    Rodrigues' formula (via `batch_rodrigues`) to convert each axis-angle
    vector into a 3x3 rotation matrix. The resulting rotation matrices are
    then concatenated back for each item in the batch.

    Args:
        pose_vectors (torch.Tensor): A tensor of shape (batch_size, rot_nb * 3)
            containing batch_size poses, where each pose consists of rot_nb
            axis-angle vectors concatenated together.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, rot_nb * 9) containing
            the corresponding rotation matrices concatenated for each pose
            in the batch. Each 9-element block represents a 3x3 rotation
            matrix in row-major order.
    """
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped: Float[Tensor, "_ 3"] = pose_vectors.contiguous().view(-1, 3)
    # convert from axis-angle to rotation matrix
    rot_mats: Float[Tensor, "_ 9"] = batch_rodrigues(pose_vec_reshaped)
    rot_mats: Float[Tensor, "b n_rotmat_flat=144"] = rot_mats.view(pose_vectors.shape[0], rot_nb * 9)

    return rot_mats


def th_with_zeros(tensor: Float[Tensor, "b 3 4"]) -> Float[Tensor, "b 4 4"]:
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def subtract_flat_id(rot_mats: Float[Tensor, "b 144"]) -> Float[Tensor, "b 144"]:
    """Subtracts the identity matrix (flattened and repeated) from a batch of flattened rotation matrices.

    Args:
        rot_mats: A tensor of shape (b, 144) representing a batch of 16 flattened 3x3 rotation matrices.

    Returns:
        A tensor of shape (b, 144) where each 9-element chunk has the flattened identity matrix subtracted.
    """
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(rot_mats.shape[0], rot_nb)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results


@serde
class MANOData:
    """Structure mirroring the data loaded from MANO .pkl files."""

    hands_components: Float64[ndarray, "45 45"]  # PCA components for hand pose
    f: Float64[ndarray, "1538 3"]  # Faces (indices)
    J_regressor: Float64[ndarray, "16 778"]  # Joint regressor (now dense)
    kintree_table: Int64[ndarray, "2 16"]  # Kinematic tree definition
    J: Float64[ndarray, "16 3"]  # Template joint locations
    bs_style: Literal["lbs"]  # Blend shape style (e.g., 'lbs')
    hands_coeffs: Float64[ndarray, "1554 45"]  # Coefficients for hand pose PCA (if applicable)
    weights: Float64[ndarray, "778 16"]  # Skinning weights
    posedirs: Float64[ndarray, "778 3 135"]  # Pose blend shapes
    hands_mean: Float64[ndarray, "45"]  # Mean hand pose (axis-angle)
    v_template: Float64[ndarray, "778 3"]  # Template vertices
    shapedirs: Float64[ndarray, "778 3 10"]  # Shape blend shapes
    bs_type: Literal["lrotmin"]  # Blend shape type (often same as bs_style)

    def __post_init__(self):
        # Ensure that the data is in the expected format
        self.betas: Float64[ndarray, "10"] = np.zeros(self.shapedirs.shape[-1])  # noqa: UP037


class ManoSimpleLayer(Module):
    def __init__(
        self,
        ncomps: Literal[45] = 45,
        side: Literal["left", "right"] = "right",
        mano_root: Path = "mano/models",
        use_pca: bool = True,
        joint_rot_mode: Literal["axisang"] = "axisang",
    ) -> None:
        """
        A simplified version of the MANO layer for understanding and debugging. This should NOT be used in a training pipeline.

        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space - for simple case will always be 45
            side: 'right' or 'left'
            use_pca: Use PCA decomposition for pose space.
            joint_rot_mode: 'axisang' or 'rotmat', ignored if use_pca
        """
        super().__init__()

        self.side: Literal["left", "right"] = side
        self.use_pca: bool = use_pca
        self.joint_rot_mode: Literal["axisang", "rotmat"] = joint_rot_mode
        self.ncomps = ncomps

        self.mano_path: Path = mano_root / f"MANO_{side.upper()}.pkl"

        with open(self.mano_path, "rb") as f:
            binary_data: bytes = f.read()
        # use serde to deserialize the binary data, ensures type safety
        mano_data: MANOData = from_pickle(MANOData, binary_data)

        hands_components: Float64[ndarray, "45 45"] = mano_data.hands_components

        # Store MANO data directly as attributes with type hints
        self.th_shapedirs: Float[Tensor, "n_verts=778 dim=3 n_betas=10"] = torch.Tensor(mano_data.shapedirs)
        self.th_posedirs: Float[Tensor, "n_verts=778 dim=3 n_pose_dims=135"] = torch.Tensor(mano_data.posedirs)
        self.th_v_template: Float[Tensor, "1 n_verts=778 dim=3"] = torch.Tensor(mano_data.v_template).unsqueeze(0)
        self.th_J_regressor: Float[Tensor, "n_joints=16 n_verts=778"] = torch.Tensor(mano_data.J_regressor)
        self.th_weights: Float[Tensor, "n_verts=778 n_joints=16"] = torch.Tensor(mano_data.weights)
        self.th_faces: Int64[Tensor, "n_faces=1538 3"] = torch.Tensor(mano_data.f.astype(np.int32)).long()
        self.n_verts: int = self.th_v_template.shape[1]

        # Get hand mean
        self.th_hands_mean: Float[Tensor, "1 45"] = torch.Tensor(mano_data.hands_mean).unsqueeze(0)
        self.th_comps: Float[Tensor, "45 45"] = torch.Tensor(hands_components)

        # Kinematic chain params
        self.kintree_table: Int64[ndarray, "2 16"] = mano_data.kintree_table
        parents: list[int] = list(self.kintree_table[0].tolist())
        self.kintree_parents: list[int] = parents

    def forward(
        self,
        th_pose_coeffs: Float[Tensor, "b n_poses=48"],
        th_betas: Float[Tensor, "b n_betas=10"],
        th_trans: Float[Tensor, "b dim=3"],
    ) -> tuple[Float[Tensor, "b n_verts=778 3"], Float[Tensor, "b joints_and_tips=21 3"]]:
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

        batch_size: int = th_pose_coeffs.shape[0]

        ###############################################################
        # Step 1: Get axis-angle from PCA components and coefficients #
        ###############################################################
        # Remove global rot coeffs (45 = 15*3)
        th_hand_pose_coeffs: Float[Tensor, "b betas=45"] = th_pose_coeffs[:, 3 : 3 + 45]
        # PCA components --> axis angles if use_pca is True
        # [b 45] @ [45 45]-> [b 45]
        th_full_hand_pose: Float[Tensor, "b betas=45"] = th_hand_pose_coeffs @ self.th_comps

        # Concatenate back global rot
        th_full_pose: Float[Tensor, "b betas=48"] = torch.cat(
            [th_pose_coeffs[:, :3], self.th_hands_mean + th_full_hand_pose], 1
        )

        # ------------------------------------------------------------------
        # Step 2: Get rotation matricies and pose maps from axis-angle
        # pose maps are relative to the current rotation so zero when in rest pose
        # ------------------------------------------------------------------

        # compute rotation matrixes from axis-angle while skipping global rotation (16*9 = 144)
        th_rot_map: Float[Tensor, "b n_rotmat_flat=144"] = th_posemap_axisang(th_full_pose)  # used to pose verts
        th_pose_map: Float[Tensor, "b n_rotmat_flat=144"] = subtract_flat_id(th_rot_map)  # used to pose joints

        root_rot: Float[Tensor, "b 3 3"] = th_rot_map[:, :9].view(batch_size, 3, 3)
        th_rot_map: Float[Tensor, "b 135"] = th_rot_map[:, 9:]
        th_pose_map: Float[Tensor, "b 135"] = th_pose_map[:, 9:]

        # ------------------------------------------------------------------
        # Step 3 – Shape blend: betas deform the template geometry
        # ------------------------------------------------------------------

        # [778, 3, 10] @ [b, 10].T -> [778, 3, 10] @ [10, b] = [778, 3, b]
        th_v_shaped: Float[Tensor, "n_verts=778 dim=3 b "] = self.th_shapedirs @ th_betas.transpose(1, 0)
        th_v_shaped: Float[Tensor, "b n_verts=778 dim=3 "] = (
            rearrange(th_v_shaped, "n_verts dim b -> b n_verts dim") + self.th_v_template
        )

        # ------------------------------------------------------------------
        # Step 4 – Regress nominal joint locations (still in rest pose)
        # ------------------------------------------------------------------

        # [16 778] @ [b, 778, 3] -> [b, 16, 3]
        th_j: Float[Tensor, "b n_joints=16 dim=3"] = self.th_J_regressor @ th_v_shaped

        # ------------------------------------------------------------------
        # Step 5 – Pose‑dependent corrective offsets (wrinkles, volume)
        # ------------------------------------------------------------------

        # [778 3 135] @ [b 135].T -> [778 3 135] @ [135 b] = [778 3 b]
        th_pose_map: Float[Tensor, "n_verts=778 dim=3 b"] = self.th_posedirs @ th_pose_map.transpose(0, 1)
        th_v_posed: Float[Tensor, "b n_verts=778 dim=3"] = th_v_shaped + rearrange(
            th_pose_map, "n_verts dim b -> b n_verts dim"
        )
        # Final T pose with transformation done !

        # ------------------------------------------------------------------
        # Step 6 – Forward kinematics: build SE(3) for each joint
        # ------------------------------------------------------------------

        root_j: Float[Tensor, "b dim=3 1"] = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        # trans here refers to transformation matrix
        root_trans: Float[Tensor, "b 4 4"] = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots: Float[Tensor, "b 15 3 3"] = th_rot_map.view(batch_size, 15, 3, 3)
        # levels refer to joint levels in the kinematic tree (does not include fingertips)
        lev1_idxs: list[int] = [1, 4, 7, 10, 13]
        lev2_idxs: list[int] = [2, 5, 8, 11, 14]
        lev3_idxs: list[int] = [3, 6, 9, 12, 15]

        # initialize rotations and translations for each level
        lev1_rots: Float[Tensor, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots: Float[Tensor, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots: Float[Tensor, "b 5 3 3"] = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j: Float[Tensor, "b 5 3"] = th_j[:, lev1_idxs]
        lev2_j: Float[Tensor, "b 5 3"] = th_j[:, lev2_idxs]
        lev3_j: Float[Tensor, "b 5 3"] = th_j[:, lev3_idxs]

        # From base to level before tips
        # Get lev1 results
        all_transforms: list[Float[Tensor, "b n_T 4 4"]] = [root_trans.unsqueeze(1)]
        lev1_j_rel: Float[Tensor, "b 5 dim=3"] = lev1_j - rearrange(root_j, "b dim n -> b n dim")
        lev1_rel_transform_flt: Float[Tensor, "_ 4 4"] = th_with_zeros(
            torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        # generate tiled root transformation matrix to match up with all fingers
        root_trans_flt: Float[Tensor, "_ 4 4"] = repeat(root_trans, "b m1 m2 -> (b f) m1 m2", f=5)
        # [b*5 4 4] @ [b*5 4 4] -> [b*5 4 4]
        lev1_flt: Float[Tensor, "_ 4 4"] = root_trans_flt @ lev1_rel_transform_flt
        # reshape back to [b 5 4 4]
        all_transforms.append(lev1_flt.view(batch_size, 5, 4, 4))

        # Get lev2 results
        lev2_j_rel: Float[Tensor, "b 5 dim=3"] = lev2_j - lev1_j
        lev2_rel_transform_flt: Float[Tensor, "_ 4 4"] = th_with_zeros(
            torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        lev2_flt: Float[Tensor, "_ 4 4"] = lev1_flt @ lev2_rel_transform_flt
        all_transforms.append(lev2_flt.view(batch_size, 5, 4, 4))

        # Get lev3 results
        lev3_j_rel: Float[Tensor, "b 5 dim=3"] = lev3_j - lev2_j
        lev3_rel_transform_flt: Float[Tensor, "_ 4 4"] = th_with_zeros(
            torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4)
        )
        lev3_flt: Float[Tensor, "_ 4 4"] = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(batch_size, 5, 4, 4))

        reorder_idxs: list[int] = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results: Float[Tensor, "b n_joints=16 4 4"] = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global: Float[Tensor, "b n_joints=16 4 4"] = th_results

        # ------------------------------------------------------------------
        # Step 7 – Linear Blend Skinning (core of LBS)
        # ------------------------------------------------------------------

        # perform linear blend skinning to get verts
        joint_js: Float[Tensor, "b n_joints=16 4"] = torch.cat([th_j, th_j.new_zeros(batch_size, 16, 1)], 2)
        # [b 16 4 4] @ [b 16 4 1] -> [b 16 4 4] @ [b 16 4 1] -> [b 16 4 1]
        tmp2: Float[Tensor, "b n_joints=16 4 1"] = th_results @ rearrange(joint_js, "b n_joints m -> b n_joints m 1")

        zeros_mat: Float[Tensor, "b n_joints=16 4 3"] = th_results.new_zeros(batch_size, 16, 4, 3)
        zeros_mat: Float[Tensor, "b n_joints=16 4 4"] = torch.cat([zeros_mat, tmp2], dim=3)

        th_results2: Float[Tensor, "b 4 4 n_joints=16"] = rearrange(
            (th_results - zeros_mat), "b n_joints m n -> b m n n_joints"
        )

        # [b 4 4 16] @ [778 16].T -> [b 4 4 16] @ [16 778] = [b 4 4 778]
        th_T: Float[Tensor, "b 4 4 n_verts=778"] = th_results2 @ self.th_weights.transpose(0, 1)

        # convert to homogeneous coordinates
        th_rest_shape_h: Float[Tensor, "b 4 n_verts=778"] = torch.cat(
            [
                th_v_posed.transpose(2, 1),
                torch.ones((batch_size, 1, th_v_posed.shape[1]), dtype=th_T.dtype, device=th_T.device),
            ],
            dim=1,
        )

        th_rest_shape_h: Float[Tensor, "b 1 4 n_verts=778"] = rearrange(th_rest_shape_h, "b m n_verts -> b 1 m n_verts")
        # [b 4 4 778] * [b 1 4 n_verts=778] -> [b 4 4 n_verts=778].sum(2) -> [b 4 n_verts=778]
        th_verts: Float[Tensor, "b 4 n_verts=778"] = (th_T * th_rest_shape_h).sum(2)
        th_verts: Float[Tensor, "b n_verts=778 4"] = th_verts.transpose(2, 1)
        th_verts: Float[Tensor, "b n_verts=778 3"] = th_verts[:, :, :3]

        # ------------------------------------------------------------------
        # Step 8 – Add fingertip pseudo‑joints
        # ------------------------------------------------------------------
        th_jtr: Float[Tensor, "b n_joints=16 3"] = th_results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        tips: Float[Tensor, "b 5 3"] = th_verts[:, [745, 319, 444, 556, 673]]
        th_jtr: Float[Tensor, "b joints_and_tips=21 3"] = torch.cat([th_jtr, tips], dim=1)

        # ------------------------------------------------------------------
        # Step 9 – Re‑order joints for downstream visualisers
        # ------------------------------------------------------------------
        th_jtr: Float[Tensor, "b joints_and_tips=21 3"] = th_jtr[
            :, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        ]

        # ------------------------------------------------------------------
        # Step 10 – Apply global translation
        # ------------------------------------------------------------------
        th_jtr: Float[Tensor, "b joints_and_tips=21 3"] = th_jtr + rearrange(th_trans, "b dim -> b 1 dim")
        th_verts: Float[Tensor, "b n_verts=778 3"] = th_verts + rearrange(th_trans, "b dim -> b 1 dim")

        # ------------------------------------------------------------------
        # Step 11 – Convert from metres → millimetres
        # ------------------------------------------------------------------
        th_verts: Float[Tensor, "b n_verts=778 3"] = th_verts * 1000
        th_jtr: Float[Tensor, "b joints_and_tips=21 3"] = th_jtr * 1000

        return th_verts, th_jtr
