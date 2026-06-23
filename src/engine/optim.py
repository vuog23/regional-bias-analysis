from torch.optim import Adam, AdamW
from typing import Any, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn


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