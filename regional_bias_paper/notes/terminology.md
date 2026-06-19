# Terminology

- `object_acc`: Overall object classification accuracy reported by the evaluation JSON.
- `region_object_acc`: Mapping from region name to object accuracy for that region.
- `best_region`: Region with the highest regional object accuracy.
- `worst_region`: Region with the lowest regional object accuracy.
- `best_region_acc`: Object accuracy in `best_region`.
- `worst_region_acc`: Object accuracy in `worst_region`.
- `region_bias_gap`: Difference between `best_region_acc` and `worst_region_acc`.
- `finetune=yes`: Result from a finetuned evaluation JSON.
- `finetune=no`: Result from a pretrained ImageNet evaluation JSON.
- `regional robustness`: A descriptive term for stable performance across regions. Use cautiously and define it operationally through `region_bias_gap` and worst-region accuracy.
- `model family`: A grouping such as CNN, transformer, self-supervised, or vision-language. Family comparisons are descriptive unless additional causal evidence is available.

