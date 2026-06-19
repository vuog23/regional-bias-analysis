# Evidence Map

This map tracks what each section needs before prose is finalized.

| paper section | claim needed | supporting source | citation key | evidence strength | status |
|---|---|---|---|---|---|
| Introduction | Regional evaluation matters for object recognition because prior work found uneven performance across geographic and socioeconomic contexts. | Does Object Recognition Work for Everyone? | devries2019object | strong | seeded |
| Introduction | This paper compares pretrained and finetuned settings across GeoDE and Dollar Street. | External result JSON files and derived tables | derived results | strong after validation | TODO: evidence needed |
| Related Work | GeoDE is a geographically diverse object-recognition dataset. | GeoDE paper | ramaswamy2023geode | strong | seeded |
| Related Work | Dollar Street captures geographic and socioeconomic diversity in household-object images. | Dollar Street paper | rojas2022dollarstreet | strong | seeded |
| Related Work | Dataset bias and domain shift affect visual recognition evaluation. | Dataset bias and shift literature | torralba2011datasetbias; taori2020naturalshifts; koh2021wilds | medium | seeded |
| Problem Definition | Worst-region accuracy and regional bias gap complement aggregate object accuracy. | Group/worst-case evaluation literature plus derived metric definition | sagawa2020groupdro | medium | seeded |
| Datasets | Exact local class counts and splits for GeoDE and Dollar Street. | External configs and local result metadata | local files | strong after audit | TODO: evidence needed |
| Models | Exact model identifiers used locally. | `used_config.yaml` and training script/config metadata | local files | strong after audit | TODO: evidence needed |
| Methodology | Aggregation produces the required normalized columns. | `scripts/aggregate_results.py` and `derived/processed/all_metrics.csv` | derived results | strong after validation | TODO: evidence needed |
| Results | Highest object accuracy by model and dataset. | `derived/tables/overall_metrics_table.csv` | derived results | strong after validation | TODO: evidence needed |
| Results | Lowest regional bias gap by model and dataset. | `derived/tables/region_bias_gap_ranking_table.csv` | derived results | strong after validation | TODO: evidence needed |
| Results | Finetuned versus pretrained object-accuracy deltas. | `derived/tables/pretrained_vs_finetuned_comparison_table.csv` | derived results | strong after validation | TODO: evidence needed |
| Results | Finetuned versus pretrained regional-gap deltas. | `derived/tables/pretrained_vs_finetuned_comparison_table.csv` | derived results | strong after validation | TODO: evidence needed |
| Discussion | Some model families may appear more regionally stable. | Derived model-family summaries plus model papers | derived results; model citations | medium after validation | TODO: evidence needed |
| Limitations | Analysis is limited to available pretrained and finetuned result files. | Workspace scope and external source documentation | local documentation | strong | seeded |

