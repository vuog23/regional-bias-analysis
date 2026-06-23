from typing import Dict, Any
from torchvision import transforms


class Transform:
    def __init__(self, transform_cfg: Dict[str, Any]):
        self.transform_cfg = transform_cfg
        self.image_size = int(transform_cfg["image_size"])

    def train_transform(self):
        crop_scale = tuple(self.transform_cfg["random_resized_crop_scale"])
        jitter = self.transform_cfg["color_jitter"]

        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=crop_scale,
                ),
                transforms.RandomHorizontalFlip(
                    p=float(self.transform_cfg["horizontal_flip_p"])
                ),
                transforms.ColorJitter(*jitter),
                transforms.RandomGrayscale(
                    p=float(self.transform_cfg["random_grayscale_p"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.transform_cfg["normalize_mean"],
                    std=self.transform_cfg["normalize_std"],
                ),
            ]
        )

    def eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.transform_cfg["normalize_mean"],
                    std=self.transform_cfg["normalize_std"],
                ),
            ]
        )