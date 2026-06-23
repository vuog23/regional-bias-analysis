from typing import Any, Dict
import torch
from pathlib import Path
import torch.nn as nn

def safe_torch_load(path: Path, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

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