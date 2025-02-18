from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import tyro
from jaxtyping import Float32
from tensordict import MemoryMappedTensor, tensorclass
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


@dataclass
class TDArgs:
    device: Literal["cpu", "cuda"] = "cuda"


@tensorclass
class FashionMNISTData:
    images: Float32[np.ndarray, "3"]
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset, device=None):
        data = cls(
            images=MemoryMappedTensor.empty(
                (len(dataset), *dataset[0][0].squeeze().shape), dtype=torch.float32
            ),
            targets=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
            batch_size=[len(dataset)],
            device=device,
        )
        for i, (image, target) in enumerate(dataset):
            data[i] = cls(images=image, targets=torch.tensor(target), batch_size=[])
        return data


def test_td(args: TDArgs) -> None:
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    from icecream import ic

    training_data_tc = FashionMNISTData.from_dataset(training_data, device=args.device)
    test_data_tc = FashionMNISTData.from_dataset(test_data, device=args.device)

    ic(type(training_data_tc))


if __name__ == "__main__":
    test_td(tyro.cli(TDArgs))
