from pathlib import Path
from typing import Dict


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