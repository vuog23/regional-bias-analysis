from typing import Dict, Any, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MyDataset
from src.data.transform import Transform


class MyDataLoader:
    def __init__(
        self,
        cfg: Dict[str, Any],
        train_dir: str | Path,
        val_dir: str | Path,
        test_dir: str | Path,
        class_to_idx: Dict[str, int],
    ):
        self.cfg = cfg
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)
        self.class_to_idx = class_to_idx

        self.transform_builder = Transform(cfg["transforms"])
        self.training_cfg = cfg["training"]

    def build_train_dataset(self) -> MyDataset:
        return MyDataset(
            root_path=self.train_dir,
            class_to_idx=self.class_to_idx,
            transform=self.transform_builder.train_transform(),
        )

    def build_val_dataset(self) -> MyDataset:
        return MyDataset(
            root_path=self.val_dir,
            class_to_idx=self.class_to_idx,
            transform=self.transform_builder.eval_transform(),
        )

    def build_test_dataset(self) -> MyDataset:
        return MyDataset(
            root_path=self.test_dir,
            class_to_idx=self.class_to_idx,
            transform=self.transform_builder.eval_transform(),
        )

    def get_common_kwargs(self) -> Dict[str, Any]:
        num_workers = int(self.training_cfg["num_workers"])

        return {
            "batch_size": int(self.training_cfg["batch_size"]),
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
        }

    def build_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset = self.build_train_dataset()
        val_dataset = self.build_val_dataset()
        test_dataset = self.build_test_dataset()

        common_kwargs = self.get_common_kwargs()

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **common_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples:   {len(val_dataset)}")
        print(f"Test samples:  {len(test_dataset)}")

        return train_loader, val_loader, test_loader



def resolve_split_dirs(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    paths = cfg["paths"]

    if paths.get("data_root"):
        data_root = Path(paths["data_root"]).expanduser().resolve()
        train_dir = Path(paths["train_dir"]).expanduser().resolve() if paths.get("train_dir") else data_root / paths["train_split"]
        val_dir = Path(paths["val_dir"]).expanduser().resolve() if paths.get("val_dir") else data_root / paths["val_split"]
        test_dir = Path(paths["test_dir"]).expanduser().resolve() if paths.get("test_dir") else data_root / paths["test_split"]
    else:
        required = ["train_dir", "val_dir", "test_dir"]
        missing = [k for k in required if not paths.get(k)]
        if missing:
            raise ValueError(f"Missing paths in config: {missing}. Use data_root or explicit train/val/test dirs.")
        train_dir = Path(paths["train_dir"]).expanduser().resolve()
        val_dir = Path(paths["val_dir"]).expanduser().resolve()
        test_dir = Path(paths["test_dir"]).expanduser().resolve()

    for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_name} directory does not exist: {split_dir}")

    return train_dir, val_dir, test_dir
