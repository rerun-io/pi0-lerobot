import torch
import tyro

from pi0_lerobot.apis.benchmark import BenchmarkConfig, benchmark_pi0

if __name__ == "__main__":
    with torch.inference_mode():
        benchmark_pi0(tyro.cli(BenchmarkConfig))
