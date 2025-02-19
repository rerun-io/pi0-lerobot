from typing import TypeVar

from jaxtyping import UInt8
from numpy import ndarray

BgrImageType = TypeVar("BgrImageType", bound=UInt8[ndarray, "H W 3"])
RgbImageType = TypeVar("RgbImageType", bound=UInt8[ndarray, "H W 3"])
