# Uncovering Regional Disparities Across Modern Visual Representation Paradigms

This repository provides a config-driven training entrypoint for fine-tuning timm image classification models on region-aware datasets such as GeoDE and DollarStreet.

The main script is [`train.py`](train.py). Most training options live in [`config.yaml`](config.yaml), while reusable code is organized under [`src/`](src).

## Setup

Create or activate a Python environment with PyTorch installed, then install the project dependencies:

```bash
pip install -r requirements.txt
```

If you use CUDA, install the PyTorch build that matches your CUDA version from the official PyTorch instructions before installing the rest of the requirements.

## Dataset Format

`train.py` expects each split to use this folder layout:

```text
data_root/
  train/
    region_name/
      class_name/
        image_001.jpg
        image_002.jpg
  val/
    region_name/
      class_name/
        image_001.jpg
  test/
    region_name/
      class_name/
        image_001.jpg
```

Example:

```text
GeoDE/
  train/
    africa/
      backpack/
      bicycle/
  val/
    africa/
      backpack/
      bicycle/
  test/
    africa/
      backpack/
      bicycle/
```

Classes are discovered from the training split by scanning:

```text
train/<region>/<class>/
```

The same class names should also appear in validation and test folders.

## Configure a Run

Edit [`config.yaml`](config.yaml) before training.

The most important fields are:

```yaml
paths:
  data_root: processed/GeoDE
  output_dir: results/geode/convnext

model:
  key: convnextv2_base
  name: null
  pretrained: true
  num_classes: null

training:
  epochs: 10
  batch_size: 32
  optimizer: adam
  lr: 0.00003
  scheduler: cosine
  amp: true
  device: auto
```

`data_root` should contain `train`, `val`, and `test` folders by default. If your split folders have different names, change:

```yaml
paths:
  train_split: train
  val_split: val
  test_split: test
```

You can also bypass `data_root` and provide explicit split folders:

```yaml
paths:
  data_root: null
  train_dir: /path/to/train
  val_dir: /path/to/val
  test_dir: /path/to/test
```

## Supported Model Keys

Use one of these values for `model.key`:

```text
convnextv2, convnextv2_base
swinv2, swinv2_base
deit3, deit3_base
dinov2, dinov2_vitb14
resnet50, resnet50_base
clip, clip_vitb16
mae, mae_vitb16
siglip, siglip_vitb16
```

If you want to use any timm model directly, set `model.name`:

```yaml
model:
  key: convnextv2_base
  name: resnet50.a2_in1k
```

When `model.name` is not `null`, it overrides `model.key`.

## Train

Run training with the default config:

```bash
python train.py
```

Or pass a custom config:

```bash
python train.py --config configs/my_experiment.yaml
```

You can override common options from the command line:

```bash
python train.py --epochs 10 --batch-size 32 --lr 3e-5
```

Change dataset or output folder without editing the YAML:

```bash
python train.py \
  --data-root datasets/processed/DollarStreet \
  --output-dir results/dollarstreet/convnext
```

Change model quickly:

```bash
python train.py --model-key resnet50_base
```

Use a raw timm model name:

```bash
python train.py --model-name convnext_tiny.fb_in22k_ft_in1k
```

## Resume Training

Resume from a saved checkpoint:

```bash
python train.py --resume results/geode/convnext/last_model.pth
```

You can also set it in `config.yaml`:

```yaml
training:
  resume: results/geode/convnext/last_model.pth
```

When resuming, the script restores the model, optimizer, scheduler, best validation accuracy, and training history when those values are available in the checkpoint.

## Test Only

Evaluate a checkpoint on the test split:

```bash
python train.py --test-only --resume results/geode/convnext/best_model.pth
```

If `--resume` is not provided, test-only mode tries to load:

```text
<output_dir>/best_model.pth
```

## Outputs

Each run writes artifacts to `paths.output_dir`:

```text
output_dir/
  used_config.yaml
  class_to_idx.json
  history.json
  best_model.pth
  last_model.pth
  test_results.json
```

Files:

- `used_config.yaml`: the final config after command-line overrides.
- `class_to_idx.json`: class label mapping discovered from the training split.
- `history.json`: training and validation metrics for each epoch.
- `best_model.pth`: checkpoint with the best validation object accuracy.
- `last_model.pth`: latest checkpoint when `training.save_last` is `true`.
- `test_results.json`: final test metrics when testing is run.

## Metrics

The trainer reports:

- Training loss
- Training object accuracy
- Validation loss
- Validation object accuracy
- Test object accuracy
- Region-level object accuracy when region labels are available
- Region bias gap, defined as best region accuracy minus worst region accuracy

Because the dataset returns region names from the folder structure, region metrics are computed automatically from:

```text
split/<region>/<class>/<image>
```

## Useful Commands

Show all command-line options:

```bash
python train.py --help
```

Run a short smoke test configuration:

```bash
python train.py --epochs 1 --batch-size 8
```

Train on CPU:

```yaml
training:
  device: cpu
  amp: false
```

Train on GPU automatically when available:

```yaml
training:
  device: auto
  amp: true
```

## Troubleshooting

If Python cannot import `timm`, install the requirements:

```bash
pip install -r requirements.txt
```

If a split folder cannot be found, check `paths.data_root`, `train_split`, `val_split`, and `test_split` in `config.yaml`.

If no images are found, make sure the dataset follows:

```text
split/region/class/image
```

If the number of classes is wrong, leave `model.num_classes` as `null` so the script uses the number of classes discovered from the training split.

## Project Structure

```text
.
  config.yaml
  train.py
  requirements.txt
  src/
    configs/
    data/
    engine/
    models/
    utils/
```

`train.py` is intentionally small. It loads the config, builds datasets and dataloaders, creates the model, configures optimization, and delegates the training loop to `src.engine.trainer.Trainer`.
