import torch
import torch.nn as nn
import timm


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

# from build_model import build_model

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = build_model(
#     model_key="convnextv2",
#     num_classes=40,
#     pretrained=True,
#     in_chans=3,
#     device=device,
# )

# print(model.__class__.__name__)
