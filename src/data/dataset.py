from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class MyDataset(Dataset):
    def __init__(
        self,
        root_path,
        class_to_idx=None,
        transform=None
    ):
        self.root_path = Path(root_path)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        for region in self.root_path.iterdir():
            if not region.is_dir():
                continue

            region_name = region.name

            for category in region.iterdir():
                if not category.is_dir():
                    continue

                category_name = category.name

                if category_name not in self.class_to_idx:
                    continue

                label = self.class_to_idx[category_name]

                for img_path in category.iterdir():
                    self.samples.append({
                        "image_path": img_path,
                        "region": region_name,
                        "label": label,
                        "category": category_name,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]
        region = sample["region"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, region


class LODODataset(Dataset):
    def __init__(
        self,
        root_path,
        class_to_idx,
        transform=None,
        include_regions=None,
        allowed_classes=None,
    ):
        self.root_path = Path(root_path)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        self.include_regions = (
            set(include_regions)
            if include_regions is not None
            else None
        )

        self.allowed_classes = (
            set(allowed_classes)
            if allowed_classes is not None
            else None
        )

        self._collect_samples()

    def _collect_samples(self):
        for region in sorted(self.root_path.iterdir()):
            if not region.is_dir():
                continue

            region_name = region.name

            if self.include_regions is not None:
                if region_name not in self.include_regions:
                    continue

            for category in sorted(region.iterdir()):
                if not category.is_dir():
                    continue

                category_name = category.name

                if self.allowed_classes is not None:
                    if category_name not in self.allowed_classes:
                        continue

                if category_name not in self.class_to_idx:
                    continue

                label = self.class_to_idx[category_name]

                for img_path in category.iterdir():
                    if not img_path.is_file():
                        continue

                    self.samples.append({
                        "image_path": img_path,
                        "region": region_name,
                        "category": category_name,
                        "label": label,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]
        region = sample["region"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, region