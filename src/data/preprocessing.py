from pathlib import Path
from PIL import Image
import pandas as pd
import json
import ast
import random
import shutil
from io import BytesIO


def dollarstreet_imagenet_splits(
    root_path,
    save_root,
    train_csv,
    test_csv,
    map_path,
    image_size=256,
    val_ratio=0.25,
    seed=42,
    val_action="move",
    overwrite=False,
):

    root_path = Path(root_path)
    save_root = Path(save_root)
    train_csv = Path(train_csv)
    test_csv = Path(test_csv)
    map_path = Path(map_path)

    train_dir = save_root / "train"
    val_dir = save_root / "val"
    test_dir = save_root / "test"

    if val_action not in ["move", "copy"]:
        raise ValueError("val_action must be either 'move' or 'copy'")

    if overwrite:
        for split_dir in [train_dir, val_dir, test_dir]:
            if split_dir.exists():
                shutil.rmtree(split_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    with open(map_path, "r", encoding="utf-8") as f:
        topic_to_imagenet = json.load(f)

    def _parse_list(x):
        if isinstance(x, list):
            return x

        if pd.isna(x):
            return []

        if isinstance(x, str):
            x = x.strip()

            if x == "":
                return []

            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except Exception:
                return [x]

        return []

    def _get_imagenet_labels(topics):
        labels = []

        for topic in topics:
            if topic in topic_to_imagenet:
                mapped_label = topic_to_imagenet[topic]

                if isinstance(mapped_label, list):
                    labels.extend(mapped_label)
                else:
                    labels.append(mapped_label)

        labels = list(dict.fromkeys(labels))
        return labels

    def _add_imagenet_label(df):
        df = df.copy()

        df["topics"] = df["topics"].apply(_parse_list)
        df["imagenet_labels"] = df["topics"].apply(_get_imagenet_labels)

        df["label"] = df["imagenet_labels"].apply(
            lambda x: x[0] if len(x) > 0 else None
        )

        return df

    def _safe_output_path(path):
        """
        Avoid overwriting if two images have the same filename.
        """
        path = Path(path)

        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def _resize_save(df, split_dir):
        saved = 0
        skipped = 0
        missing = 0

        for _, row in df.iterrows():
            image_rel_path = row["imageRelPath"]
            label = row["label"]
            region = row["region.id"]

            if pd.isna(label) or pd.isna(region):
                skipped += 1
                continue

            input_path = root_path / image_rel_path

            if not input_path.exists():
                print("Missing:", input_path)
                missing += 1
                continue

            save_dir = split_dir / str(region) / str(label)
            save_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(image_rel_path).name
            output_path = _safe_output_path(save_dir / filename)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(
                        (image_size, image_size),
                        Image.Resampling.LANCZOS
                    )
                    img.save(output_path)

                saved += 1

            except Exception as e:
                print("Error:", input_path, "|", e)
                skipped += 1

        return {
            "saved": saved,
            "missing": missing,
            "skipped": skipped,
        }

    def _create_val_set():
        random.seed(seed)

        moved_or_copied = 0
        skipped_classes = 0

        for region_dir in sorted(train_dir.iterdir()):
            if not region_dir.is_dir():
                continue

            region = region_dir.name

            for class_dir in sorted(region_dir.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                images = [p for p in class_dir.iterdir() if p.is_file()]

                n = len(images)

                if n <= 1:
                    print(f"Skip {region}/{class_name}: only {n} image")
                    skipped_classes += 1
                    continue

                random.shuffle(images)

                val_count = round(n * val_ratio)
                val_count = max(1, val_count)
                val_count = min(val_count, n - 1)

                selected_images = images[:val_count]

                val_class_dir = val_dir / region / class_name
                val_class_dir.mkdir(parents=True, exist_ok=True)

                for img_path in selected_images:
                    dst_path = _safe_output_path(val_class_dir / img_path.name)

                    if val_action == "move":
                        shutil.move(str(img_path), str(dst_path))
                    else:
                        shutil.copy2(str(img_path), str(dst_path))

                    moved_or_copied += 1

        return {
            "val_images": moved_or_copied,
            "skipped_classes": skipped_classes,
            "val_action": val_action,
        }

    def _count_images(folder):
        if not folder.exists():
            return 0

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sum(
            1 for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df = _add_imagenet_label(train_df)
    test_df = _add_imagenet_label(test_df)

    train_stats = _resize_save(train_df, train_dir)
    test_stats = _resize_save(test_df, test_dir)
    val_stats = _create_val_set()

    summary = {
        "train_before_val": train_stats,
        "test": test_stats,
        "val": val_stats,
        "final_counts": {
            "train": _count_images(train_dir),
            "val": _count_images(val_dir),
            "test": _count_images(test_dir),
        },
        "output_root": str(save_root),
    }

    return summary


def geode_splits(
    parquet_folder,
    save_root,
    image_size=256,
    val_ratio=0.25,
    test_ratio=0.25,
    seed=42,
    overwrite=False,
):

    parquet_folder = Path(parquet_folder)
    save_root = Path(save_root)

    all_dir = save_root / "_all"
    train_dir = save_root / "train"
    val_dir = save_root / "val"
    test_dir = save_root / "test"

    if overwrite and save_root.exists():
        for split_dir in [all_dir, train_dir, val_dir, test_dir]:
            if split_dir.exists():
                shutil.rmtree(split_dir)

    all_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    def _safe_output_path(path):
        """
        Avoid overwriting if two images have the same filename.
        """
        path = Path(path)

        if not path.exists():
            return path

        parent = path.parent
        stem = path.stem
        suffix = path.suffix

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def _save_parquet_images(df):
        saved = 0
        skipped = 0

        for i in range(len(df)):
            try:
                img_bytes = df.iloc[i]["image"]
                obj = df.iloc[i]["object"]
                region = df.iloc[i]["region"]

                if pd.isna(obj) or pd.isna(region):
                    skipped += 1
                    continue

                img_path = img_bytes.get("path", f"image_{i}.jpg")

                img = Image.open(BytesIO(img_bytes["bytes"])).convert("RGB")
                img = img.resize(
                    (image_size, image_size),
                    Image.Resampling.LANCZOS
                )

                save_dir = all_dir / str(region) / str(obj)
                save_dir.mkdir(parents=True, exist_ok=True)

                filename = Path(img_path).name

                if Path(filename).suffix == "":
                    filename = f"{filename}.jpg"

                output_path = _safe_output_path(save_dir / filename)
                img.save(output_path)

                saved += 1

            except Exception as e:
                print(f"Error at row {i}: {e}")
                skipped += 1

        return saved, skipped

    def _split_all_to_train_val_test():
        moved_train = 0
        moved_val = 0
        moved_test = 0
        skipped_classes = 0

        for region_dir in sorted(all_dir.iterdir()):
            if not region_dir.is_dir():
                continue

            region = region_dir.name

            for class_dir in sorted(region_dir.iterdir()):
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name

                images = [p for p in class_dir.iterdir() if p.is_file()]
                n = len(images)

                if n == 0:
                    continue

                random.shuffle(images)

                if n <= 2:
                    train_images = images
                    val_images = []
                    test_images = []
                    skipped_classes += 1

                else:
                    test_count = round(n * test_ratio)
                    val_count = round(n * val_ratio)

                    test_count = max(1, test_count)
                    val_count = max(1, val_count)

                    while test_count + val_count > n - 1:
                        if test_count >= val_count and test_count > 0:
                            test_count -= 1
                        elif val_count > 0:
                            val_count -= 1
                        else:
                            break

                    test_images = images[:test_count]
                    val_images = images[test_count:test_count + val_count]
                    train_images = images[test_count + val_count:]

                train_class_dir = train_dir / region / class_name
                val_class_dir = val_dir / region / class_name
                test_class_dir = test_dir / region / class_name

                train_class_dir.mkdir(parents=True, exist_ok=True)
                val_class_dir.mkdir(parents=True, exist_ok=True)
                test_class_dir.mkdir(parents=True, exist_ok=True)

                for img_path in train_images:
                    dst_path = _safe_output_path(train_class_dir / img_path.name)
                    shutil.move(str(img_path), str(dst_path))
                    moved_train += 1

                for img_path in val_images:
                    dst_path = _safe_output_path(val_class_dir / img_path.name)
                    shutil.move(str(img_path), str(dst_path))
                    moved_val += 1

                for img_path in test_images:
                    dst_path = _safe_output_path(test_class_dir / img_path.name)
                    shutil.move(str(img_path), str(dst_path))
                    moved_test += 1

        return {
            "train": moved_train,
            "val": moved_val,
            "test": moved_test,
            "skipped_small_classes": skipped_classes,
        }

    def _count_images(folder):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        return sum(
            1 for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )

    parquet_files = sorted(parquet_folder.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in: {parquet_folder}")

    total_saved = 0
    total_skipped = 0

    for parquet_file in parquet_files:
        print("Reading:", parquet_file)

        df = pd.read_parquet(parquet_file)

        saved, skipped = _save_parquet_images(df)

        total_saved += saved
        total_skipped += skipped

    split_stats = _split_all_to_train_val_test()

    if all_dir.exists():
        shutil.rmtree(all_dir)

    summary = {
        "parquet_files": len(parquet_files),
        "total_saved_before_split": total_saved,
        "total_skipped": total_skipped,
        "split": split_stats,
        "final_counts": {
            "train": _count_images(train_dir),
            "val": _count_images(val_dir),
            "test": _count_images(test_dir),
        },
        "output_root": str(save_root),
    }

    return summary