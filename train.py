from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn

from src.data.class_mapping import build_class_to_idx
from src.data.dataloader import MyDataLoader, resolve_split_dirs
from src.engine.checkpoint import safe_torch_load
from src.engine.optim import build_optimizer, build_scheduler
from src.engine.trainer import Trainer
from src.models.model_config import MODEL_CONFIGS, resolve_model_name
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.io import save_json, save_yaml
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a timm classifier using config.yaml."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file. Defaults to config.yaml.",
    )

    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-key", type=str, default=None, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test-only", action="store_true")

    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Keep config.yaml authoritative while allowing common quick experiments."""
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


def build_data(cfg: Dict[str, Any]):
    train_dir, val_dir, test_dir = resolve_split_dirs(cfg)
    class_to_idx = build_class_to_idx(train_dir)
    dataloaders = MyDataLoader(
        cfg=cfg,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        class_to_idx=class_to_idx,
    )
    train_loader, val_loader, test_loader = dataloaders.build_loaders()
    return train_dir, val_dir, test_dir, class_to_idx, train_loader, val_loader, test_loader


def print_run_summary(
    cfg: Dict[str, Any],
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    output_dir: Path,
    model_name: str,
    num_classes: int,
) -> None:
    print("Configuration")
    print(f"  train_dir:    {train_dir}")
    print(f"  val_dir:      {val_dir}")
    print(f"  test_dir:     {test_dir}")
    print(f"  output_dir:   {output_dir}")
    print(f"  model_key:    {cfg['model']['key']}")
    print(f"  model_name:   {model_name}")
    print(f"  num_classes:  {num_classes}")
    print(f"  device:       {cfg['training']['device']}")
    print(f"  amp:          {cfg['training']['amp']}")
    print(f"  epochs:       {cfg['training']['epochs']}")
    print(f"  batch_size:   {cfg['training']['batch_size']}")
    print(f"  lr:           {cfg['training']['lr']}")


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)

    seed_everything(int(cfg["project"]["seed"]))
    device = get_device(str(cfg["training"]["device"]))
    cfg["training"]["device"] = str(device)

    (
        train_dir,
        val_dir,
        test_dir,
        class_to_idx,
        train_loader,
        val_loader,
        test_loader,
    ) = build_data(cfg)

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    model_cfg = cfg["model"]
    model_name = resolve_model_name(
        model_key=str(model_cfg["key"]),
        model_name=model_cfg.get("name"),
    )
    num_classes = int(model_cfg["num_classes"] or len(class_to_idx))

    if model_cfg["num_classes"] is not None and num_classes != len(class_to_idx):
        print(
            f"Warning: config model.num_classes={num_classes}, but train folder has "
            f"{len(class_to_idx)} classes."
        )

    output_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print_run_summary(
        cfg=cfg,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        output_dir=output_dir,
        model_name=model_name,
        num_classes=num_classes,
    )
    save_yaml(cfg, output_dir / "used_config.yaml")
    save_json(class_to_idx, output_dir / "class_to_idx.json")

    from src.models.create_model import create_model

    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        device=device,
        pretrained=bool(model_cfg["pretrained"]),
        drop_path_rate=float(model_cfg["drop_path_rate"]),
        data_parallel=bool(model_cfg["data_parallel"]),
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg["training"]["label_smoothing"])
    )
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

    resume_path = cfg["training"].get("resume")
    test_only = bool(cfg["training"].get("test_only", False))

    if test_only:
        checkpoint_path = resume_path or output_dir / "best_model.pth"
        trainer.test(checkpoint_path=checkpoint_path)
        return

    start_epoch = 1
    if resume_path:
        resume_path = Path(resume_path).expanduser().resolve()
        checkpoint = safe_torch_load(resume_path, map_location=device)
        trainer.load_model_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        trainer.best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
        trainer.history = checkpoint.get("history", [])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed checkpoint: {resume_path}")

    trainer.train(epochs=int(cfg["training"]["epochs"]), start_epoch=start_epoch)

    if bool(cfg["training"].get("test_after_train", True)):
        trainer.test(checkpoint_path=output_dir / "best_model.pth")


if __name__ == "__main__":
    main()
