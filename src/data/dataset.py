from typing import Dict

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, root_path: str | Path, class_to_idx: Dict[str, int], transform=None):
        self.root_path = Path(root_path)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root_path}")

        for region in sorted(self.root_path.iterdir()):
            if not region.is_dir():
                continue

            region_name = region.name

            for category in sorted(region.iterdir()):
                if not category.is_dir():
                    continue

                category_name = category.name
                if category_name not in self.class_to_idx:
                    continue

                label = self.class_to_idx[category_name]

                for img_path in sorted(category.iterdir()):
                    if not img_path.is_file():
                        continue

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "region": region_name,
                            "label": label,
                            "category": category_name,
                        }
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root_path}. Expected root/region/class/image files."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]
        region = sample["region"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, region