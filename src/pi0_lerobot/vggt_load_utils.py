from pathlib import Path

import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from serde import serde
from torchvision import transforms as TF


@serde(deny_unknown_fields=True)
class VGGTPredictions:
    pose_enc: UInt8[ndarray, "*batch num_cams 9"]
    depth: Float32[ndarray, "*batch num_cams H W 1"]
    depth_conf: Float32[ndarray, "*batch num_cams H W"]
    world_points: Float32[ndarray, "*batch num_cams H W 3"]
    world_points_conf: Float32[ndarray, "*batch num_cams H W"]
    images: Float32[ndarray, "*batch num_cams 3 H W"]
    extrinsic: Float32[ndarray, "*batch num_cams 3 4"]
    intrinsic: Float32[ndarray, "*batch num_cams 3 3"]

    def remove_batch_dim_if_one(self) -> "VGGTPredictions":
        """
        Removes the batch dimension from all arrays if batch size is 1.

        Returns:
            VGGTPredictions: A new instance with batch dimension removed if batch=1
        """
        if self.pose_enc.shape[0] != 1:
            return self

        result = VGGTPredictions(
            pose_enc=self.pose_enc.squeeze(0),
            depth=self.depth.squeeze(0),
            depth_conf=self.depth_conf.squeeze(0),
            world_points=self.world_points.squeeze(0),
            world_points_conf=self.world_points_conf.squeeze(0),
            images=self.images.squeeze(0),
            extrinsic=self.extrinsic.squeeze(0),
            intrinsic=self.intrinsic.squeeze(0),
        )
        return result


def preprocess_images(
    rgb_list: list[UInt8[ndarray, "H W 3"]],
) -> Float32[torch.Tensor, "N 3 H W"]:
    """
    A quick start function to preprocess images for model input.

    Args:
        rgb_list (list): List of RGB images as numpy arrays

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - The function ensures width=518px while maintaining aspect ratio
        - Height is adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(rgb_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for rgb in rgb_list:
        # Convert the numpy array to PIL Image to ensure identical processing
        pil_img = Image.fromarray(rgb)

        width, height = pil_img.size
        new_width = 518

        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions using PIL's BICUBIC for exact matching
        pil_img = pil_img.resize((new_width, new_height), Image.BICUBIC)

        # Convert to tensor using the same to_tensor transform
        img = to_tensor(pil_img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518
        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y : start_y + 518, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image is (1, C, H, W)
    if len(rgb_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images


def create_blueprint(parent_log_path: Path, image_paths: list[Path]) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/camera_{i}/pinhole/depth" for i in range(len(image_paths))],
        ],
    )
    view2d = rrb.Vertical(
        contents=[
            rrb.Horizontal(
                contents=[
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Pinhole Content",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/confidence",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Confidence Map",
                    ),
                ]
            )
            # show at most 4 cameras
            for i in range(min(4, len(image_paths)))
        ]
    )

    blueprint = rrb.Blueprint(rrb.Horizontal(contents=[view3d, view2d], column_shares=[3, 1]), collapse_panels=True)
    return blueprint
