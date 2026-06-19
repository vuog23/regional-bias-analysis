# Paper Plan

## Working Title

Regional Bias in Pretrained and Finetuned Vision Models on GeoDE and Dollar Street

## Main Research Question

How do pretrained and finetuned vision models differ in regional bias across GeoDE and Dollar Street?

## Secondary Research Questions

1. Which models have the highest object accuracy?
2. Which models have the lowest regional bias gap?
3. Does finetuning improve object accuracy?
4. Does finetuning reduce or increase regional bias gap?
5. Are some model families more regionally stable than others?
6. Are high-accuracy models always less biased across regions?
7. How do CNN, transformer, self-supervised, and vision-language models differ in regional robustness?

## Expected Contributions

1. A reproducible analysis pipeline that normalizes pretrained and finetuned result JSON files into a single metric table.
2. A comparative evaluation of object accuracy and regional bias gap across nine model families on GeoDE and Dollar Street.
3. A cautious analysis of whether finetuning is associated with improved accuracy, reduced regional gaps, or trade-offs between the two.
4. A literature-grounded discussion of geographic diversity, dataset bias, domain shift, and evaluation beyond aggregate accuracy.

## Paper Structure

1. Introduction: motivate regional robustness and define the empirical comparison.
2. Related Work: organize prior work on geographic bias, dataset bias, domain shift, GeoDE, Dollar Street, vision-language models, self-supervised learning, and worst-group evaluation.
3. Problem Definition: define object accuracy, regional accuracy, best and worst regions, and regional bias gap.
4. Datasets: describe GeoDE and Dollar Street using cited dataset papers and verified local configuration details.
5. Models: group the evaluated models into CNN, transformer, self-supervised, and vision-language families.
6. Methodology: explain aggregation, metrics, paired comparisons, and evidence rules.
7. Experiments: document external result files, evaluation settings, and reproducibility workflow.
8. Results: report object accuracy, best and worst regions, regional bias gap, and pretrained versus finetuned comparisons.
9. Analysis and Discussion: interpret observed patterns cautiously and connect them to literature.
10. Limitations: state that the paper only analyzes pretrained ImageNet and finetuned settings, plus any missing metadata or experimental constraints.
11. Conclusion: summarize evidence-supported findings and future work.

## Intended Analysis

- Normalize all available result JSON files into `derived/processed/all_metrics.csv`.
- Rank models by object accuracy and by lowest regional bias gap.
- Compare pretrained and finetuned results for each dataset-model pair.
- Summarize dataset-level and model-level averages.
- Inspect whether higher object accuracy is associated with lower, similar, or higher regional bias gap.
- Compare model families without claiming causality.

## Literature Review Strategy

- Start with seed papers on GeoDE, Dollar Street, dataset bias, distribution shift, ImageNet generalization, CLIP/SigLIP, self-supervised representation learning, and model architectures.
- Use `literature/literature_search_protocol.md` to run a transparent search across arXiv, OpenReview, NeurIPS, CVF, ACM/IEEE, Semantic Scholar, Google Scholar, and official dataset/model pages.
- Record screened papers in `literature/paper_inventory.csv`.
- Move rejected or out-of-scope papers into `literature/rejected_papers.csv`.
- Only add BibTeX entries when source metadata is available.

## Evidence-Driven Writing Process

- Draft claims first in `literature/claim_bank.md` or from derived empirical tables.
- Map every planned claim to a source in `literature/evidence_map.md`.
- Use TODO markers when evidence is missing.
- Use cautious language: suggests, indicates, may reflect, is consistent with, and may be associated with.
- Avoid causal wording unless the evidence directly supports a causal claim.

