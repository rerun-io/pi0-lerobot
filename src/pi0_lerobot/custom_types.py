from typing import Any, TypeVar

import pyarrow as pa
import rerun as rr
from jaxtyping import Float, UInt8
from numpy import ndarray

BgrImageType = TypeVar("BgrImageType", bound=UInt8[ndarray, "H W 3"])
RgbImageType = TypeVar("RgbImageType", bound=UInt8[ndarray, "H W 3"])


class ConfidenceBatch(rr.ComponentBatchMixin):
    """A batch of confidence data."""

    def __init__(self, confidence: Float[ndarray, "..."]) -> None:
        self.confidence = confidence

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """The descriptor of the custom component."""
        return rr.ComponentDescriptor("user.Confidence")

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.confidence, type=pa.float32())


class CustomPoints2D(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with a custom component."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "... 2"],
        confidences: Float[ndarray, "..."],
        class_ids: int,
        keypoint_ids: list[int],
        show_labels: bool = False,
    ) -> None:
        self.points2d = rr.Points2D(
            positions=positions,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
            show_labels=show_labels,
        )
        self.confidences = ConfidenceBatch(confidences).or_with_descriptor_overrides(
            archetype_name="user.CustomPoints3D", archetype_field_name="confidences"
        )

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        return (
            list(self.points2d.as_component_batches())  # The components from Points2D
            + [self.confidences]  # Custom confidence data
        )


class CustomPoints3D(rr.AsComponents):
    """A custom archetype that extends Rerun's builtin `Points3D` archetype with a custom component."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "... 3"],
        confidences: Float[ndarray, "..."],
        class_ids: int,
        keypoint_ids: list[int],
        show_labels: bool = False,
    ) -> None:
        self.points3d = rr.Points3D(
            positions=positions,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
            show_labels=show_labels,
        )
        self.confidences = ConfidenceBatch(confidences).or_with_descriptor_overrides(
            archetype_name="user.CustomPoints3D", archetype_field_name="confidences"
        )

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        return (
            list(self.points3d.as_component_batches())  # The components from Points3D
            + [self.confidences]  # Custom confidence data
        )
