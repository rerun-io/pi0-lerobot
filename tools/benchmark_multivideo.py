import torch
import tyro

from pi0_lerobot.apis.video_decode_benchmark import BenchmarkConfig, benchmark_decode

if __name__ == "__main__":
    with torch.inference_mode():
        benchmark_decode(tyro.cli(BenchmarkConfig))
