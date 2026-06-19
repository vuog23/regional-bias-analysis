# External Sources

This paper workspace reads external experiment artifacts but must not modify them.

## External Paths

- Results root: `D:/Project/DAP_paper/results`
- Global config file: `D:/Project/DAP_paper/config.yaml`
- Training script example: `D:/Project/DAP_paper/scripts/finetune.py`

## Allowed Reads

The paper pipeline may read these external file types when needed:

- Result JSON files, including `test_results.json` and pretrained ImageNet JSON files.
- Configuration YAML files, including `used_config.yaml` and the global `config.yaml`.
- Python source files, including the training script example, for documenting experimental setup.

## Files That Must Never Be Modified

- Anything under `D:/Project/DAP_paper/results`
- `D:/Project/DAP_paper/config.yaml`
- `D:/Project/DAP_paper/scripts/finetune.py`
- Dataset files under `D:/Project/DAP_paper/datasets`
- Any model checkpoint file, including `best_model.pth`, `last_model.pth`, or other `.pth` files

## Checkpoint Policy

Checkpoint files are ignored for paper writing. They are not required to aggregate metrics, make tables, make figures, validate citations, or compile the paper. Do not copy checkpoint files into this workspace.

## Output Policy

All generated analysis outputs belong inside:

- `derived/processed/`
- `derived/tables/`
- `derived/figures/`
- `paper/tables/`

