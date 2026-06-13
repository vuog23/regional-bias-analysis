from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

def build_class_to_idx(root_path):
    root_path = Path(root_path)
    class_names = set()

    for region in root_path.iterdir():
        region_path = root_path / region

        for category in region_path.iterdir():
            class_names.add(category.name)

    return {
        class_name: idx
        for idx, class_name in enumerate(sorted(class_names))
    }

class DollarStreetDataset(Dataset):
    def __init__(
        self,
        root_path,
        class_to_idx=None,
        transform=None
    ):

        self.root_path = Path(root_path)
        self.class_to_idx = class_to_idx
        self.samples = []
    
        for region in self.root_path.iterdir():
            region_name = region.name
            region_path = root_path / region

            for category in region_path.iterdir():
                category_name = category.name
                category_path = region_path / category

                label = self.class_to_idx[category_name]

                for img_path  in category_path.iterdir():
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

        image = Image.open(sample['image_path']).convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label