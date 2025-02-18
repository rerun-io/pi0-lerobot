from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Literal

import torch
from icecream import ic
from jaxtyping import Float32
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.policies import PreTrainedConfig
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


@dataclass
class InferenceArgs:
    device: Literal["cpu", "cuda"] = "cuda"
    dataset_repo_id: str = "danaaubakirova/koch_test"
    ckpt_torch_dir: str = "lerobot/pi0"
    warmup_iters: int = 10
    benchmark_iters: int = 30


def inference_pi0(config: InferenceArgs) -> None:
    dataset = LeRobotDataset(config.dataset_repo_id, episodes=[0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
    )

    print(dataset)
    input()

    batch: dict = next(iter(dataloader))

    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=config.device, dtype=torch.float32)

    start_loading = timer()
    print(f"Loading policy from {config.ckpt_torch_dir}")
    cfg = PreTrainedConfig.from_pretrained(config.ckpt_torch_dir)
    cfg.pretrained_path = config.ckpt_torch_dir
    policy: PI0Policy = make_policy(cfg, config.device, ds_meta=dataset.meta)
    print(f"Loading time: {timer() - start_loading:.3f} s")

    # Warmup
    for _ in tqdm(range(config.warmup_iters), desc="Warmup"):
        torch.cuda.synchronize()
        policy.select_action(batch)
        policy.reset()
        torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in tqdm(range(config.benchmark_iters), desc="Benchmark"):
        action: Float32[torch.Tensor, "batch _"] = policy.select_action(batch)
        policy.reset()
    end_event.record()

    # Synchronize and measure time
    torch.cuda.synchronize()
    elapsed_time_ms: float = start_event.elapsed_time(end_event)

    avg_time_per_iter: float = elapsed_time_ms / config.benchmark_iters
    print(f"Average execution time per iteration: {avg_time_per_iter:.3f} ms")


if __name__ == "__main__":
    with torch.inference_mode():
        inference_pi0()
