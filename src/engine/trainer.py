from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import torch
from tqdm.auto import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from src.engine.checkpoint import safe_torch_load, smart_load_state_dict
from src.utils.io import save_json


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[Iterable[Any]]]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1], None
    raise ValueError("Expected batch to be (images, labels) or (images, labels, regions).")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        use_amp: bool,
        output_dir: str | Path,
        model_name: str,
        dataset_name: str,
        class_to_idx: Dict[str, int],
        idx_to_class: Dict[int, str],
        save_last: bool = True,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.save_last = save_last

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.history = []

    def autocast_context(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    def get_model_state_dict(self) -> Dict[str, torch.Tensor]:
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        smart_load_state_dict(target_model, state_dict)

    def get_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_images = 0

        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)

        for batch in pbar:
            images, labels, _ = unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with self.autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_images += batch_size

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / max(total_images, 1):.4f}",
                    "acc": f"{total_correct / max(total_images, 1):.4f}",
                    "lr": f"{self.get_lr():.2e}",
                }
            )

        return {
            "loss": total_loss / max(total_images, 1),
            "acc": total_correct / max(total_images, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "val") -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_images = 0
        region_correct = defaultdict(int)
        region_total = defaultdict(int)
        has_regions = False

        pbar = tqdm(loader, desc=f"Evaluate {split_name}", dynamic_ncols=True)

        for batch in pbar:
            images, labels, regions = unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self.autocast_context():
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct_mask = preds == labels
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += correct_mask.sum().item()
            total_images += batch_size

            if regions is not None:
                has_regions = True
                for region, is_correct in zip(regions, correct_mask.detach().cpu().tolist()):
                    region = str(region)
                    region_total[region] += 1
                    region_correct[region] += int(is_correct)

            pbar.set_postfix({"object_acc": f"{total_correct / max(total_images, 1):.4f}"})

        results: Dict[str, Any] = {
            "loss": total_loss / max(total_images, 1),
            "object_acc": total_correct / max(total_images, 1),
        }

        if has_regions and len(region_total) > 0:
            region_acc = {
                region: region_correct[region] / region_total[region]
                for region in sorted(region_total.keys())
            }
            best_region = max(region_acc, key=region_acc.get)
            worst_region = min(region_acc, key=region_acc.get)
            best_region_acc = region_acc[best_region]
            worst_region_acc = region_acc[worst_region]

            results.update(
                {
                    "region_object_acc": region_acc,
                    "best_region": best_region,
                    "worst_region": worst_region,
                    "best_region_acc": best_region_acc,
                    "worst_region_acc": worst_region_acc,
                    "region_bias_gap": best_region_acc - worst_region_acc,
                }
            )
        else:
            results.update(
                {
                    "region_object_acc": {},
                    "best_region": None,
                    "worst_region": None,
                    "best_region_acc": None,
                    "worst_region_acc": None,
                    "region_bias_gap": None,
                }
            )

        return results

    def checkpoint_dict(self, epoch: int, val_results: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "model_state_dict": self.get_model_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "val_results": val_results,
            "history": self.history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        return checkpoint

    def save_checkpoint(self, epoch: int, val_results: Dict[str, Any], filename: str) -> None:
        path = self.output_dir / filename
        torch.save(self.checkpoint_dict(epoch, val_results), path)
        print(f"Saved checkpoint: {path}")

    def save_history(self) -> None:
        save_json(self.history, self.output_dir / "history.json")

    def train(self, epochs: int, start_epoch: int = 1) -> None:
        for epoch in range(start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_results = self._train_epoch(epoch)
            val_results = self.evaluate(self.val_loader, split_name="val")

            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Train Loss:        {train_results['loss']:.4f}")
            print(f"Train Object Acc:  {train_results['acc']:.4f}")
            print(f"Val Loss:          {val_results['loss']:.4f}")
            print(f"Val Object Acc:    {val_results['object_acc']:.4f}")

            if val_results["region_bias_gap"] is not None:
                print(f"Region Bias Gap:   {val_results['region_bias_gap']:.4f}")
                print("\nObject accuracy per region:")
                for region, acc in val_results["region_object_acc"].items():
                    print(f"  {region}: {acc:.4f}")

            epoch_log = {
                "epoch": epoch,
                "lr": self.get_lr(),
                "train_loss": train_results["loss"],
                "train_object_acc": train_results["acc"],
                "val_loss": val_results["loss"],
                "val_object_acc": val_results["object_acc"],
                "val_region_object_acc": val_results["region_object_acc"],
                "val_best_region": val_results["best_region"],
                "val_worst_region": val_results["worst_region"],
                "val_region_bias_gap": val_results["region_bias_gap"],
            }
            self.history.append(epoch_log)

            if val_results["object_acc"] > self.best_val_acc:
                self.best_val_acc = val_results["object_acc"]
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_results, "best_model.pth")

            if self.save_last:
                self.save_checkpoint(epoch, val_results, "last_model.pth")

            self.save_history()

        print("\nTraining finished")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best val object acc: {self.best_val_acc:.4f}")

    def test(self, checkpoint_path: Optional[str | Path] = None) -> Dict[str, Any]:
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / "best_model.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        self.load_model_state_dict(checkpoint["model_state_dict"])

        test_results = self.evaluate(self.test_loader, split_name="test")
        print("\nFINAL TEST RESULTS\n")
        print(f"Test Object Acc:   {test_results['object_acc']:.4f}")

        if test_results["best_region"] is not None:
            print(f"Best Region:       {test_results['best_region']} ({test_results['best_region_acc']:.4f})")
            print(f"Worst Region:      {test_results['worst_region']} ({test_results['worst_region_acc']:.4f})")
            print(f"Region Bias Gap:   {test_results['region_bias_gap']:.4f}")
            print("\nObject accuracy per region:")
            for region, acc in test_results["region_object_acc"].items():
                print(f"  {region}: {acc:.4f}")

        save_json(test_results, self.output_dir / "test_results.json")
        return test_results
