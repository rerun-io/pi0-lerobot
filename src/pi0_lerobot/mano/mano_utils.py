from pathlib import Path
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float32, Int64
from torch import Tensor
from torch.nn import Module

from pi0_lerobot.mano.manopt_manolayer import ManoLayer


class MANOLayer(Module):
    """Wrapper layer for manopth ManoLayer."""

    def __init__(self, side: Literal["left", "right"], betas: Float32[np.ndarray, "10"], mano_root_dir: Path) -> None:
        """
        Constructor for MANOLayer.

        Args:
            side (str): MANO hand type. 'right' or 'left'.
            betas (np.ndarray): A numpy array of shape [10] containing the betas.
            mano_root_dir (Path): Path to the MANO root directory
        """
        super().__init__()
        assert mano_root_dir.exists(), f"MANO root directory {mano_root_dir} not found."
        assert mano_root_dir.is_dir(), f"{mano_root_dir} is not a directory."

        self._side: Literal["left", "right"] = side
        self._betas: Float32[np.ndarray, "10"] = betas  # noqa: UP037

        self._mano_layer = ManoLayer(
            side=side,
            mano_root=mano_root_dir,
            flat_hand_mean=False,
            ncomps=45,
            use_pca=True,
        )

        # Register buffer for betas (prescanned for particular subject hand)
        b: Float32[Tensor, "1 10"] = torch.from_numpy(rearrange(betas, "b -> 1 b")).float()
        self.register_buffer("b", b)

        # Register buffer for faces
        f: Int64[Tensor, "num_faces=1538 3"] = self._mano_layer.th_faces
        self.register_buffer("f", f)

        # Register buffer for root translation
        shapedirs: Float32[Tensor, "778 3 10"] = self._mano_layer.th_shapedirs
        v_template: Float32[Tensor, "b=1 num_verts=778 dim=3"] = self._mano_layer.th_v_template
        # [778 3 10] @ [1 10].T -> [778 3 1]
        v: Float32[Tensor, "n_verts=778 dim=3 b=1"] = shapedirs @ self.b.transpose(0, 1)
        v: Float32[Tensor, "b=1 n_verts=778 dim=3"] = rearrange(v, "n_verts dim b -> b n_verts dim") + v_template

        j_regressor: Float32[Tensor, "16 778"] = self._mano_layer.th_J_regressor
        # [1 3 778] @ [1 778 3] -> [b 3 3]
        r: Float32[Tensor, "1 3"] = j_regressor[0] @ v
        self.register_buffer("root_trans", r)

    def forward(
        self, p: Float32[Tensor, "b n_poses=48"], t: Float32[Tensor, "b dim=3"]
    ) -> tuple[Float32[Tensor, "b n_verts=778 dim=3"], Float32[Tensor, "b n_joints=21 dim=3"]]:
        """
        Forward function.

        Args:
            p (Tensor): A tensor of shape [B, 48] containing the pose.
            t (Tensor): A tensor of shape [B, 3] containing the translation.

        Returns:
            tuple[Tensor, Tensor]:
                v: A tensor of shape [B, 778, 3] containing the vertices.
                j: A tensor of shape [B, 21, 3] containing the joints.
        """
        batch_size: int = p.size(0)
        mano_output: tuple[Float32[Tensor, "b n_verts=778 dim=3"], Float32[Tensor, "b n_joints=21 dim=3"]] = (
            self._mano_layer(p, self.b.expand(batch_size, -1), t)
        )

        # Convert to meters.
        verts: Float32[Tensor, "b n_verts=778 dim=3"] = mano_output[0] / 1000.0
        joints: Float32[Tensor, "b n_joints=21 dim=3"] = mano_output[1] / 1000.0
        return verts, joints

    @property
    def th_hands_mean(self) -> Tensor:
        """Return the hand mean tensor."""
        return self._mano_layer.th_hands_mean

    @property
    def th_selected_comps(self) -> Tensor:
        """Return the selected components tensor."""
        return self._mano_layer.th_selected_comps

    @property
    def th_v_template(self) -> Tensor:
        """Return the vertex template tensor."""
        return self._mano_layer.th_v_template

    @property
    def side(self) -> str:
        """Return the side of the hand."""
        return self._side

    @property
    def num_verts(self) -> int:
        """Return the number of vertices."""
        return 778
