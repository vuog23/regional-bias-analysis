MODEL_CONFIGS = {
    "convnextv2": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "swinv2": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "swinv2_base": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "deit3": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "deit3_base": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "resnet50": "resnet50.a2_in1k",
    "resnet50_base": "resnet50.a2_in1k",
    "clip": "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
    "clip_vitb16": "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
    "mae": "vit_base_patch16_224.mae",
    "mae_vitb16": "vit_base_patch16_224.mae",
    "siglip": "vit_base_patch16_siglip_224.webli",
    "siglip_vitb16": "vit_base_patch16_siglip_224.webli",
}


def resolve_model_name(model_key: str, model_name: str | None = None) -> str:
    if model_name:
        return model_name

    if model_key not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown model key '{model_key}'. Available keys: {available}")

    return MODEL_CONFIGS[model_key]
