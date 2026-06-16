from pathlib import Path


def class_to_idx(root_path):
    root_path = Path(root_path)
    class_names = set()

    for region in root_path.iterdir():
        if not region.is_dir():
            continue

        for category in region.iterdir():
            if category.is_dir():
                class_names.add(category.name)

    return {
        class_name: idx
        for idx, class_name in enumerate(sorted(class_names))
    }


def get_regions(root_path):
    root_path = Path(root_path)

    return sorted([
        p.name for p in root_path.iterdir()
        if p.is_dir()
    ])


def get_classes_by_region(root_path):
    root_path = Path(root_path)
    classes_by_region = {}

    for region_dir in sorted(root_path.iterdir()):
        if not region_dir.is_dir():
            continue

        region_name = region_dir.name
        classes = set()

        for class_dir in sorted(region_dir.iterdir()):
            if class_dir.is_dir():
                classes.add(class_dir.name)

        classes_by_region[region_name] = classes

    return classes_by_region


def build_class_to_idx_from_classes(classes):
    return {
        class_name: idx
        for idx, class_name in enumerate(sorted(classes))
    }

def prepare_lodo_classes(train_root, use_common_classes=False):
    classes_by_region = get_classes_by_region(train_root)

    all_classes = set()
    for region, classes in classes_by_region.items():
        all_classes.update(classes)

    common_classes = set.intersection(*classes_by_region.values())

    if use_common_classes:
        final_classes = common_classes
    else:
        final_classes = all_classes

    class_to_idx = build_class_to_idx_from_classes(final_classes)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print("All classes:", len(all_classes))
    print("Common classes:", len(common_classes))
    print("Final classes:", len(final_classes))

    return class_to_idx, idx_to_class, final_classes