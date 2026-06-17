#!/usr/bin/env python3
"""
Standalone image finetuning script for Vast.ai.

This version includes everything inside one file:
- MyDataset
- train/eval transforms
- timm model creation
- class_to_idx builder
- Trainer
- YAML config loading

Example:
    cd /workspace/my_training_project
    python train_vast_standalone.py --config config_geode.yaml

Install dependencies if needed:
    pip install timm pyyaml tqdm pillow torchvision
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import timm
except ImportError as exc:
    raise ImportError("Missing dependency: timm. Install it with: pip install timm") from exc

try:
    import yaml
except ImportError as exc:
    raise ImportError("Missing dependency: PyYAML. Install it with: pip install pyyaml") from exc


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_CONFIGS = {
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "swinv2_base": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "deit3_base": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "geode_finetune",
        "dataset_name": "GeoDE",
        "seed": 42,
    },
    "paths": {
        "data_root": "/workspace/my_training_project/data/GeoDE_split",
        "train_dir": None,
        "val_dir": None,
        "test_dir": None,
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "output_dir": "/workspace/my_training_project/outputs/convnextv2_geode",
    },
    "model": {
        "key": "convnextv2_base",
        "name": None,
        "pretrained": True,
        "num_classes": None,
        "drop_path_rate": 0.1,
        "data_parallel": False,
    },
    "transforms": {
        "image_size": 224,
        "random_resized_crop_scale": [0.7, 1.0],
        "horizontal_flip_p": 0.5,
        "color_jitter": [0.3, 0.3, 0.3, 0.3],
        "random_grayscale_p": 0.1,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
    "training": {
        "epochs": 20,
        "batch_size": 32,
        "num_workers": 2,
        "optimizer": "adam",
        "lr": 0.00003,
        "weight_decay": 0.0,
        "label_smoothing": 0.1,
        "scheduler": "cosine",
        "t_max": None,
        "eta_min": 0.000001,
        "amp": True,
        "device": "auto",
        "save_last": True,
        "test_after_train": True,
        "resume": None,
        "test_only": False,
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return cfg

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    return deep_update(cfg, user_cfg)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_yaml(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def safe_torch_load(path: Path, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -----------------------------------------------------------------------------
# Dataset and transforms
# -----------------------------------------------------------------------------
class MyDataset(Dataset):
    """
    Dataset for structure:

        root_path/
        ├── region_1/
        │   ├── class_a/
        │   └── class_b/
        └── region_2/
            ├── class_a/
            └── class_b/

    Returns: image, label, region
    """

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
                    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
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


def train_transform(transform_cfg: Dict[str, Any]):
    image_size = int(transform_cfg["image_size"])
    crop_scale = tuple(transform_cfg["random_resized_crop_scale"])
    jitter = transform_cfg["color_jitter"]

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=crop_scale),
            transforms.RandomHorizontalFlip(p=float(transform_cfg["horizontal_flip_p"])),
            transforms.ColorJitter(*jitter),
            transforms.RandomGrayscale(p=float(transform_cfg["random_grayscale_p"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_cfg["normalize_mean"],
                std=transform_cfg["normalize_std"],
            ),
        ]
    )


def eval_transform(transform_cfg: Dict[str, Any]):
    image_size = int(transform_cfg["image_size"])

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transform_cfg["normalize_mean"],
                std=transform_cfg["normalize_std"],
            ),
        ]
    )


def build_class_to_idx(root_path: str | Path) -> Dict[str, int]:
    root_path = Path(root_path)
    class_names = set()

    if not root_path.exists():
        raise FileNotFoundError(f"Train path does not exist: {root_path}")

    for region in root_path.iterdir():
        if not region.is_dir():
            continue

        for category in region.iterdir():
            if category.is_dir():
                class_names.add(category.name)

    if not class_names:
        raise RuntimeError(f"No class folders found in {root_path}")

    return {class_name: idx for idx, class_name in enumerate(sorted(class_names))}


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def create_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
    data_parallel: bool = False,
) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    model = model.to(device)

    if data_parallel and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model


def smart_load_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load checkpoints even if DataParallel was toggled on/off."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if not model_keys or not ckpt_keys:
        raise

    model_uses_module = model_keys[0].startswith("module.")
    ckpt_uses_module = ckpt_keys[0].startswith("module.")

    if model_uses_module and not ckpt_uses_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    elif ckpt_uses_module and not model_uses_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[Iterable[Any]]]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1], None
    raise ValueError("Expected batch to be (images, labels) or (images, labels, regions).")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        use_amp: bool,
        output_dir: str | Path,
        model_name: str,
        dataset_name: str,
        class_to_idx: Dict[str, int],
        idx_to_class: Dict[int, str],
        save_last: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.save_last = save_last

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = []

    def autocast_context(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    def get_model_state_dict(self) -> Dict[str, torch.Tensor]:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        smart_load_state_dict(target_model, state_dict)

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_images = 0

        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)

        for batch in pbar:
            images, labels, _ = unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with self.autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_images += batch_size

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / max(total_images, 1):.4f}",
                    "acc": f"{total_correct / max(total_images, 1):.4f}",
                    "lr": f"{self.get_lr():.2e}",
                }
            )

        return {
            "loss": total_loss / max(total_images, 1),
            "acc": total_correct / max(total_images, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "val") -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_images = 0
        region_correct = defaultdict(int)
        region_total = defaultdict(int)
        has_regions = False

        pbar = tqdm(loader, desc=f"Evaluate {split_name}", dynamic_ncols=True)

        for batch in pbar:
            images, labels, regions = unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self.autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct_mask = preds == labels
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct_mask.sum().item()
            total_images += batch_size

            if regions is not None:
                has_regions = True
                for region, is_correct in zip(regions, correct_mask.detach().cpu().tolist()):
                    region = str(region)
                    region_total[region] += 1
                    region_correct[region] += int(is_correct)

            pbar.set_postfix({"object_acc": f"{total_correct / max(total_images, 1):.4f}"})

        results: Dict[str, Any] = {
            "loss": total_loss / max(total_images, 1),
            "object_acc": total_correct / max(total_images, 1),
        }

        if has_regions and len(region_total) > 0:
            region_acc = {
                region: region_correct[region] / region_total[region]
                for region in sorted(region_total.keys())
            }
            best_region = max(region_acc, key=region_acc.get)
            worst_region = min(region_acc, key=region_acc.get)
            best_region_acc = region_acc[best_region]
            worst_region_acc = region_acc[worst_region]

            results.update(
                {
                    "region_object_acc": region_acc,
                    "best_region": best_region,
                    "worst_region": worst_region,
                    "best_region_acc": best_region_acc,
                    "worst_region_acc": worst_region_acc,
                    "region_bias_gap": best_region_acc - worst_region_acc,
                }
            )
        else:
            results.update(
                {
                    "region_object_acc": {},
                    "best_region": None,
                    "worst_region": None,
                    "best_region_acc": None,
                    "worst_region_acc": None,
                    "region_bias_gap": None,
                }
            )

        return results

    def checkpoint_dict(self, epoch: int, val_results: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "model_state_dict": self.get_model_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "val_results": val_results,
            "history": self.history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        return checkpoint

    def save_checkpoint(self, epoch: int, val_results: Dict[str, Any], filename: str) -> None:
        path = self.output_dir / filename
        torch.save(self.checkpoint_dict(epoch, val_results), path)
        print(f"Saved checkpoint: {path}")

    def save_history(self) -> None:
        save_json(self.history, self.output_dir / "history.json")

    def train(self, epochs: int, start_epoch: int = 1) -> None:
        for epoch in range(start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_results = self._train_epoch(epoch)
            val_results = self.evaluate(self.val_loader, split_name="val")

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Train Loss:        {train_results['loss']:.4f}")
            print(f"Train Object Acc:  {train_results['acc']:.4f}")
            print(f"Val Loss:          {val_results['loss']:.4f}")
            print(f"Val Object Acc:    {val_results['object_acc']:.4f}")

            if val_results["region_bias_gap"] is not None:
                print(f"Region Bias Gap:   {val_results['region_bias_gap']:.4f}")
                print("\nObject accuracy per region:")
                for region, acc in val_results["region_object_acc"].items():
                    print(f"  {region}: {acc:.4f}")

            epoch_log = {
                "epoch": epoch,
                "lr": self.get_lr(),
                "train_loss": train_results["loss"],
                "train_object_acc": train_results["acc"],
                "val_loss": val_results["loss"],
                "val_object_acc": val_results["object_acc"],
                "val_region_object_acc": val_results["region_object_acc"],
                "val_best_region": val_results["best_region"],
                "val_worst_region": val_results["worst_region"],
                "val_region_bias_gap": val_results["region_bias_gap"],
            }
            self.history.append(epoch_log)

            if val_results["object_acc"] > self.best_val_acc:
                self.best_val_acc = val_results["object_acc"]
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_results, "best_model.pth")

            if self.save_last:
                self.save_checkpoint(epoch, val_results, "last_model.pth")

            self.save_history()

        print("\nTraining finished")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val object acc: {self.best_val_acc:.4f}")

    def test(self, checkpoint_path: Optional[str | Path] = None) -> Dict[str, Any]:
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / "best_model.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        self.load_model_state_dict(checkpoint["model_state_dict"])

        test_results = self.evaluate(self.test_loader, split_name="test")
        print("\nFINAL TEST RESULTS\n")
        print(f"Test Object Acc:   {test_results['object_acc']:.4f}")

        if test_results["best_region"] is not None:
            print(f"Best Region:       {test_results['best_region']} ({test_results['best_region_acc']:.4f})")
            print(f"Worst Region:      {test_results['worst_region']} ({test_results['worst_region_acc']:.4f})")
            print(f"Region Bias Gap:   {test_results['region_bias_gap']:.4f}")
            print("\nObject accuracy per region:")
            for region, acc in test_results["region_object_acc"].items():
                print(f"  {region}: {acc:.4f}")

        save_json(test_results, self.output_dir / "test_results.json")
        return test_results


# -----------------------------------------------------------------------------
# Build objects from config
# -----------------------------------------------------------------------------
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


def build_loaders(
    cfg: Dict[str, Any],
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    class_to_idx: Dict[str, int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = MyDataset(
        root_path=train_dir,
        class_to_idx=class_to_idx,
        transform=train_transform(cfg["transforms"]),
    )
    val_dataset = MyDataset(
        root_path=val_dir,
        class_to_idx=class_to_idx,
        transform=eval_transform(cfg["transforms"]),
    )
    test_dataset = MyDataset(
        root_path=test_dir,
        class_to_idx=class_to_idx,
        transform=eval_transform(cfg["transforms"]),
    )

    training = cfg["training"]
    num_workers = int(training["num_workers"])

    common_kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common_kwargs)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def build_optimizer(cfg: Dict[str, Any], model: nn.Module):
    training = cfg["training"]
    optimizer_name = str(training["optimizer"]).lower()
    lr = float(training["lr"])
    weight_decay = float(training.get("weight_decay", 0.0))

    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'adam' or 'adamw'.")


def build_scheduler(cfg: Dict[str, Any], optimizer):
    training = cfg["training"]
    scheduler_name = str(training.get("scheduler", "cosine")).lower()

    if scheduler_name in ["none", "null", ""]:
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(training["t_max"] or training["epochs"]),
            eta_min=float(training["eta_min"]),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}. Use 'cosine' or 'none'.")


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # These overrides let you keep one YAML file and quickly change common options.
    if args.data_root is not None:
        cfg["paths"]["data_root"] = args.data_root
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.model_key is not None:
        cfg["model"]["key"] = args.model_key
        cfg["model"]["name"] = None
    if args.model_name is not None:
        cfg["model"]["name"] = args.model_name
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.resume is not None:
        cfg["training"]["resume"] = args.resume
    if args.test_only:
        cfg["training"]["test_only"] = True
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone timm finetuning script with YAML config.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")

    # Common optional overrides
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-key", type=str, default=None, choices=sorted(MODEL_CONFIGS.keys()))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed_everything(int(cfg["project"]["seed"]))
    device = get_device(str(cfg["training"]["device"]))

    train_dir, val_dir, test_dir = resolve_split_dirs(cfg)
    class_to_idx = build_class_to_idx(train_dir)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    model_cfg = cfg["model"]
    model_name = model_cfg["name"] or MODEL_CONFIGS[model_cfg["key"]]
    num_classes = int(model_cfg["num_classes"] or len(class_to_idx))

    if model_cfg["num_classes"] is not None and num_classes != len(class_to_idx):
        print(
            f"Warning: config model.num_classes={num_classes}, but train folder has "
            f"{len(class_to_idx)} classes. Using {num_classes}."
        )

    output_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration")
    print(f"  train_dir:    {train_dir}")
    print(f"  val_dir:      {val_dir}")
    print(f"  test_dir:     {test_dir}")
    print(f"  output_dir:   {output_dir}")
    print(f"  model_name:   {model_name}")
    print(f"  num_classes:  {num_classes}")
    print(f"  device:       {device}")
    print(f"  amp:          {cfg['training']['amp']}")

    save_yaml(cfg, output_dir / "used_config.yaml")
    save_json(class_to_idx, output_dir / "class_to_idx.json")

    train_loader, val_loader, test_loader = build_loaders(cfg, train_dir, val_dir, test_dir, class_to_idx)

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        device=device,
        pretrained=bool(model_cfg["pretrained"]),
        drop_path_rate=float(model_cfg["drop_path_rate"]),
        data_parallel=bool(model_cfg["data_parallel"]),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=bool(cfg["training"]["amp"]),
        output_dir=output_dir,
        model_name=model_name,
        dataset_name=str(cfg["project"]["dataset_name"]),
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        save_last=bool(cfg["training"]["save_last"]),
    )

    start_epoch = 1
    resume_path = cfg["training"].get("resume")
    test_only = bool(cfg["training"].get("test_only", False))

    if resume_path:
        resume_path = Path(resume_path).expanduser().resolve()
        checkpoint = safe_torch_load(resume_path, map_location=device)
        trainer.load_model_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {resume_path}")

        if not test_only:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            trainer.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
            trainer.history = checkpoint.get("history", [])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1

    if test_only:
        trainer.test(checkpoint_path=resume_path or output_dir / "best_model.pth")
        return

    trainer.train(epochs=int(cfg["training"]["epochs"]), start_epoch=start_epoch)

    if bool(cfg["training"].get("test_after_train", True)):
        trainer.test(checkpoint_path=output_dir / "best_model.pth")


if __name__ == "__main__":
    main()
