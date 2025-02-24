from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import rerun as rr
import torch
from icecream import ic
from jaxtyping import UInt8
from simplecv.rerun_log_utils import RerunTyroConfig
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from pi0_lerobot.data.assembly101 import Assembly101Dataset, load_assembly101


@dataclass
class BenchmarkConfig:
    rr_config: RerunTyroConfig
    root_directory: Path = Path("/mnt/12tbdrive/data/assembly101-new")
    example_name: str = (
        "nusar-2021_action_both_9051-b03a_9051_user_id_2021-02-22_114140"
    )
    benchmark_iterations: int = 250


def benchmark_decode(config: BenchmarkConfig) -> None:
    ic("Starting video decode benchmark")
    assembly101_data: Assembly101Dataset = load_assembly101(
        config.root_directory, config.example_name, load_2d=False, load_3d=False
    )
    exo_video_paths: list[Path] = assembly101_data.exo_video_readers.video_paths
    gpu_mv_decoder: list[VideoDecoder] = [
        VideoDecoder(exo_path, device="cuda", dimension_order="NHWC")
        for exo_path in exo_video_paths
    ]
    # num_frames = min([decoder.metadata.num_frames for decoder in gpu_mv_decoder])
    num_frames = config.benchmark_iterations
    gpu_num_frames = num_frames * 5

    # CPU Decoding Benchmark
    start_time = timer()
    for frame_idx in tqdm(range(num_frames), desc="CPU Decoding"):
        bgr_list: list[UInt8[np.ndarray, "H W 3"]] = assembly101_data.exo_video_readers[
            frame_idx
        ]
    end_time = timer()
    total_time = end_time - start_time
    cpu_fps = num_frames / total_time
    cpu_message = f"CPU decoding: {total_time:.2f} seconds, FPS: {cpu_fps:.2f}"
    rr.log("log", rr.TextLog(cpu_message))
    ic(cpu_message)

    # GPU Decoding Benchmark (without Jaxtyping overhead)
    start_time = timer()
    for frame_idx in tqdm(range(gpu_num_frames), desc="GPU Decoding"):
        bgr_list: list[torch.Tensor] = [
            decoder[frame_idx] for decoder in gpu_mv_decoder
        ]
    end_time = timer()
    total_time = end_time - start_time
    gpu_fps = gpu_num_frames / total_time
    gpu_message = f"GPU decoding: {total_time:.2f} seconds, FPS: {gpu_fps:.2f}"
    rr.log("log", rr.TextLog(gpu_message))
    ic(gpu_message)

    # GPU Decoding Benchmark with Jaxtyping
    start_time = timer()
    for frame_idx in tqdm(range(gpu_num_frames), desc="GPU Decoding w/ Jaxtyping"):
        bgr_list: list[UInt8[torch.Tensor, "H W 3"]] = [
            decoder[frame_idx] for decoder in gpu_mv_decoder
        ]
    end_time = timer()
    total_time = end_time - start_time
    gpu_jaxtyping_fps = gpu_num_frames / total_time
    gpu_jaxtyping_message = f"GPU w/ Jaxtyping decoding: {total_time:.2f} seconds, FPS: {gpu_jaxtyping_fps:.2f}"
    rr.log("log", rr.TextLog(gpu_jaxtyping_message))
    ic(gpu_jaxtyping_message)

    # GPU with Numpy Memory Copy Decoding Benchmark
    start_time = timer()
    for frame_idx in tqdm(range(gpu_num_frames), desc="GPU Decoding w/ Numpy Mem Copy"):
        bgr_list = [decoder[frame_idx].cpu().numpy() for decoder in gpu_mv_decoder]
    end_time = timer()
    total_time = end_time - start_time
    gpu_numpy_fps = (gpu_num_frames) / total_time
    gpu_numpy_message = f"GPU w/ Numpy mem copy decoding: {total_time:.2f} seconds, FPS: {gpu_numpy_fps:.2f}"
    rr.log("log", rr.TextLog(gpu_numpy_message))
    ic(gpu_numpy_message)

    # GPU with Numpy Memory Copy Decoding Benchmark
    start_time = timer()
    for frame_idx in tqdm(range(gpu_num_frames), desc="GPU Decoding w/ Numpy Mem Copy"):
        bgr_list: list[UInt8[np.ndarray, "H W 3"]] = [
            decoder[frame_idx].cpu().numpy() for decoder in gpu_mv_decoder
        ]
    end_time = timer()
    total_time = end_time - start_time
    gpu_numpy_typing_fps = (gpu_num_frames) / total_time
    gpu_numpy_typing_message = f"GPU w/ Numpy mem copy  + typingdecoding: {total_time:.2f} seconds, FPS: {gpu_numpy_typing_fps:.2f}"
    rr.log("log", rr.TextLog(gpu_numpy_typing_message))
    ic(gpu_numpy_typing_message)
