import torch
import tyro

from pi0_lerobot.apis.pi0_inference import InferenceArgs, inference_pi0

if __name__ == "__main__":
    with torch.inference_mode():
        inference_pi0(tyro.cli(InferenceArgs))
