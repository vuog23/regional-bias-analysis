from contextlib import nullcontext
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

import timm
from timm.data import resolve_model_data_config, create_transform

from src.data.dataset import MyDataset, build_class_to_idx


class ImageNetEval:
    def __init__(
        self,
        root_path,
        model_name,
        target_to_imagenet_idx,
        batch_size=64,
        num_workers=2,
        region_index=None,
        use_amp=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.root_path = root_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.region_index = region_index
        self.use_amp = use_amp and self.device.type == "cuda"
        self.target_to_imagenet_idx = target_to_imagenet_idx

        # timm-only model loading
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1000,
        ).to(self.device)

        self.model.eval()

        # Correct timm preprocessing for each checkpoint
        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(
            **data_config,
            is_training=False,
        )

        print("Loaded timm pretrained model")
        print("Model:", model_name)

        self.class_to_idx = build_class_to_idx(self.root_path)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        mapped_classes = [
            c for c in self.class_to_idx
            if len(self.target_to_imagenet_idx.get(c, [])) > 0
        ]

        unmapped_classes = [
            c for c in self.class_to_idx
            if len(self.target_to_imagenet_idx.get(c, [])) == 0
        ]

        print("Dataset classes:", len(self.class_to_idx))
        print("Mapped classes:", len(mapped_classes))
        print("Unmapped / skipped classes:", len(unmapped_classes))
        print("Unmapped:", sorted(unmapped_classes))

    def _create_dataloader(self):
        dataset = MyDataset(
            root_path=self.root_path,
            class_to_idx=self.class_to_idx,
            transform=self.transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    def _unpack_batch(self, batch):
        images = batch[0]
        labels = batch[1]

        regions = None
        if self.region_index is not None and len(batch) > self.region_index:
            regions = batch[self.region_index]

        return images, labels, regions

    def _to_list(self, x):
        if x is None:
            return None

        if isinstance(x, torch.Tensor):
            return x.cpu().tolist()

        return list(x)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        loader = self._create_dataloader()

        total = 0
        correct = 0
        skipped = 0

        per_class_total = defaultdict(int)
        per_class_correct = defaultdict(int)

        skipped_by_class = Counter()

        region_total = defaultdict(int)
        region_correct = defaultdict(int)

        region_class_total = defaultdict(lambda: defaultdict(int))
        region_class_correct = defaultdict(lambda: defaultdict(int))

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp
            else nullcontext()
        )

        for batch in tqdm(loader):
            images, labels, regions = self._unpack_batch(batch)

            images = images.to(self.device, non_blocking=True)
            labels = self._to_list(labels)
            regions = self._to_list(regions)

            with amp_context:
                logits = self.model(images)

            preds = logits.argmax(dim=1).cpu().tolist()

            for i, (y, pred) in enumerate(zip(labels, preds)):
                target_class = self.idx_to_class[int(y)]
                valid_targets = self.target_to_imagenet_idx.get(target_class, [])

                if len(valid_targets) == 0:
                    skipped += 1
                    skipped_by_class[target_class] += 1
                    continue

                valid_set = set(valid_targets)
                is_correct = pred in valid_set

                total += 1
                correct += int(is_correct)

                per_class_total[target_class] += 1
                per_class_correct[target_class] += int(is_correct)

                if regions is not None:
                    region = regions[i]

                    region_total[region] += 1
                    region_correct[region] += int(is_correct)

                    region_class_total[region][target_class] += 1
                    region_class_correct[region][target_class] += int(is_correct)

        overall_acc = correct / total if total > 0 else 0.0

        per_class_rows = []
        for cls in sorted(per_class_total.keys()):
            n = per_class_total[cls]
            per_class_rows.append({
                "class": cls,
                "n": n,
                "accuracy": per_class_correct[cls] / n,
            })

        per_class_df = pd.DataFrame(per_class_rows)

        skipped_df = pd.DataFrame([
            {"class": cls, "skipped_images": n}
            for cls, n in skipped_by_class.items()
        ])

        if len(skipped_df) > 0:
            skipped_df = skipped_df.sort_values("class")
        else:
            skipped_df = pd.DataFrame(columns=["class", "skipped_images"])

        region_df = None
        region_summary = None

        if len(region_total) > 0:
            region_rows = []

            for region in sorted(region_total.keys()):
                n = region_total[region]
                raw_acc = region_correct[region] / n

                class_accs = []

                for cls in region_class_total[region]:
                    cls_n = region_class_total[region][cls]
                    cls_acc = region_class_correct[region][cls] / cls_n
                    class_accs.append(cls_acc)

                object_balanced_acc = sum(class_accs) / len(class_accs)

                region_rows.append({
                    "region": region,
                    "n": n,
                    "accuracy": raw_acc,
                    "object_balanced_accuracy": object_balanced_acc,
                })

            region_df = pd.DataFrame(region_rows)

            best_idx = region_df["accuracy"].idxmax()
            worst_idx = region_df["accuracy"].idxmin()

            best_balanced_idx = region_df["object_balanced_accuracy"].idxmax()
            worst_balanced_idx = region_df["object_balanced_accuracy"].idxmin()

            region_summary = {
                "best_region": region_df.loc[best_idx, "region"],
                "worst_region": region_df.loc[worst_idx, "region"],
                "region_gap": region_df["accuracy"].max() - region_df["accuracy"].min(),
                "region_std": region_df["accuracy"].std(),

                "object_balanced_best_region": region_df.loc[best_balanced_idx, "region"],
                "object_balanced_worst_region": region_df.loc[worst_balanced_idx, "region"],
                "object_balanced_region_gap": (
                    region_df["object_balanced_accuracy"].max()
                    - region_df["object_balanced_accuracy"].min()
                ),
                "object_balanced_region_std": region_df["object_balanced_accuracy"].std(),
            }

        return {
            "model_name": self.model_name,
            "accuracy": overall_acc,
            "evaluated_images": total,
            "skipped_images": skipped,
            "per_class_df": per_class_df,
            "skipped_df": skipped_df,
            "region_df": region_df,
            "region_summary": region_summary,
        }

    def print_summary(self, results):
        print("ImageNet-pretrained evaluation")
        print("Model:", results["model_name"])
        print()
        print(f"Accuracy:           {results['accuracy'] * 100:.2f}%")
        print(f"Evaluated images:   {results['evaluated_images']}")
        print(f"Skipped images:     {results['skipped_images']}")
        print()

        if results["region_summary"] is not None:
            s = results["region_summary"]

            print("Region bias summary")
            print()
            print(f"Best region:                     {s['best_region']}")
            print(f"Worst region:                    {s['worst_region']}")
            print(f"Region gap:                      {s['region_gap'] * 100:.2f}%")
            print(f"Region std:                      {s['region_std'] * 100:.2f}%")

            print()
            print(f"Object-balanced best region:     {s['object_balanced_best_region']}")
            print(f"Object-balanced worst region:    {s['object_balanced_worst_region']}")
            print(f"Object-balanced region gap:      {s['object_balanced_region_gap'] * 100:.2f}%")
            print(f"Object-balanced region std:      {s['object_balanced_region_std'] * 100:.2f}%")