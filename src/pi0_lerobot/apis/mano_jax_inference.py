from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import jax  # Added import
import numpy as np
import torch
from jax import numpy as jnp
from simplecv.rerun_log_utils import RerunTyroConfig

from pi0_lerobot.mano.mano_jax_simple import ManoJaxLayer
from pi0_lerobot.mano.mano_pytorch_simple import ManoSimpleLayer


@dataclass
class ManoConfig:
    rr_config: RerunTyroConfig


def mano_inference(config: ManoConfig):
    start_time: float = timer()
    print(f"Start time: {start_time}")
    print(f"rr_config: {config.rr_config}")

    # Initialize models
    jax_mano = ManoJaxLayer(mano_root=Path("data/mano_clean"))
    pytorch_mano = ManoSimpleLayer(mano_root=Path("data/mano_clean"))

    # JIT compile the JAX forward pass
    jax_forward_jit = jax.jit(jax_mano.forward)

    # Warm-up phase
    print("\n--- Warm-up Phase ---")
    warmup_batch_size = 1
    warmup_pose_coeffs = np.random.rand(warmup_batch_size, 48).astype(np.float32)
    warmup_shape_coeffs = np.random.rand(warmup_batch_size, 10).astype(np.float32)
    warmup_trans = np.random.rand(warmup_batch_size, 3).astype(np.float32)

    # PyTorch warm-up
    _ = pytorch_mano(
        torch.from_numpy(warmup_pose_coeffs), torch.from_numpy(warmup_shape_coeffs), torch.from_numpy(warmup_trans)
    )
    print("PyTorch model warmed up.")

    # JAX warm-up (and compilation)
    jax_warmup_out = jax_forward_jit(
        jnp.array(warmup_pose_coeffs), jnp.array(warmup_shape_coeffs), jnp.array(warmup_trans)
    )
    jax_warmup_out[0].block_until_ready()  # Ensure JIT compilation is finished
    print("JAX model JIT compiled and warmed up.")

    # Inference phase
    batch_size = 10000
    print(f"\n--- Inference Phase (batch_size={batch_size}) ---")
    pose_coeffs = np.random.rand(batch_size, 48).astype(np.float32)
    shape_coeffs = np.random.rand(batch_size, 10).astype(np.float32)
    trans = np.random.rand(batch_size, 3).astype(np.float32)

    # PyTorch inference
    torch_pose_coeffs = torch.from_numpy(pose_coeffs)
    torch_shape_coeffs = torch.from_numpy(shape_coeffs)
    torch_trans = torch.from_numpy(trans)

    torch_start_time = timer()
    out_torch = pytorch_mano(torch_pose_coeffs, torch_shape_coeffs, torch_trans)
    torch_end_time = timer()
    print(f"PyTorch inference time: {torch_end_time - torch_start_time:.4f} seconds")

    # JAX inference
    jax_pose_coeffs = jnp.array(pose_coeffs)
    jax_shape_coeffs = jnp.array(shape_coeffs)
    jax_trans = jnp.array(trans)

    jax_start_time = timer()
    out_jax = jax_forward_jit(jax_pose_coeffs, jax_shape_coeffs, jax_trans)
    out_jax[0].block_until_ready()  # Ensure JAX computation is finished before stopping timer
    jax_end_time = timer()
    print(f"JAX inference time: {jax_end_time - jax_start_time:.4f} seconds")

    # Compare vertices
    verts_torch_np = out_torch[0].detach().cpu().numpy()
    verts_jax_np = np.array(out_jax[0])
    all_verts_close = np.allclose(verts_torch_np, verts_jax_np, atol=1e-4)
    print(f"Vertices allclose (atol=1e-4): {all_verts_close}")
    if not all_verts_close:
        print(f"Max verts diff: {np.max(np.abs(verts_torch_np - verts_jax_np))}")

    # Compare joints
    joints_torch_np = out_torch[1].detach().cpu().numpy()
    joints_jax_np = np.array(out_jax[1])
    all_joints_close = np.allclose(
        joints_torch_np, joints_jax_np, atol=1e-4
    )  # Adjusted atol to match vertices for consistency
    print(f"Joints allclose (atol=1e-4): {all_joints_close}")
    if not all_joints_close:
        print(f"Max joints diff: {np.max(np.abs(joints_torch_np - joints_jax_np))}")

    print("\n--- Summary ---")
    print(f"All {batch_size} samples had close vertices: {all_verts_close}")
    print(f"All {batch_size} samples had close joints: {all_joints_close}")
    end_time: float = timer()
    print(f"Total time: {end_time - start_time:.2f} seconds")
