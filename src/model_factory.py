import torch
import torch.nn as nn
import timm

def create_model(model_name, num_classes, device):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=0.1,
    )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using DataParallel")
        model = nn.DataParallel(model)

    return model