# Evidence Map

| paper section | claim needed | supporting paper | supporting result table or figure if applicable | citation key | strength | TODO if evidence is missing |
|---|---|---|---|---|---|---|
| Introduction | Regional object-recognition evaluation matters. | Does Object Recognition Work for Everyone?; No Classification without Representation | none | devries2019object; shankar2017geodiversity | strong |  |
| Introduction | GeoDE and Dollar Street are appropriate datasets. | GeoDE; Dollar Street dataset papers | none | ramaswamy2023geode; rojas2022dollarstreet | strong |  |
| Related Work | Dataset bias and benchmark representation affect evaluation. | Dataset bias; ImageNet audit papers | none | torralba2011datasetbias; yang2020fairerimagenet; luccioni2022bugs | strong |  |
| Related Work | Distribution shift motivates evaluation beyond aggregate accuracy. | WILDS; natural shifts; DomainBed | none | koh2021wilds; taori2020naturalshifts; gulrajani2021domainbed | strong |  |
| Related Work | Worst-group and hidden-strata evaluation motivate worst-region metrics. | Group DRO; hidden stratification | none | sagawa2020groupdro; sohoni2020hiddenstratification | strong |  |
| Related Work | Context/design/shortcut mechanisms may matter for regional differences. | GeoNet; shortcut and texture-bias papers | none | kalluri2023geonet; geirhos2018texturebias; geirhos2020shortcut | medium | Need local evidence before turning this into an explanation. |
| Methodology | Metrics are normalized and validated. | none | derived/processed/all_metrics.csv; validate_results.py output | derived results | strong |  |
| Results | Finetuning increases object accuracy. | none | derived/tables/pretrained_vs_finetuned_comparison_table.csv; figure pretrained_vs_finetuned_object_acc.png | derived results | strong |  |
| Results | Finetuning increases regional gap. | none | derived/tables/pretrained_vs_finetuned_comparison_table.csv; figure pretrained_vs_finetuned_region_bias_gap.png | derived results | strong |  |
| Results | ConvNeXtV2 is highest-accuracy finetuned model on both datasets. | none | derived/processed/all_metrics.csv | derived results | strong |  |
| Results | MAE is lowest-gap finetuned model on both datasets. | none | derived/processed/all_metrics.csv | derived results | strong |  |
| Results | Dollar Street has larger finetuned regional gap than GeoDE. | none | derived/tables/dataset_level_summary_table.csv | derived results | strong |  |
| Discussion | Accuracy and regional balance are not the same property. | Worst-group literature | accuracy-bias and Pareto figures | sagawa2020groupdro; sohoni2020hiddenstratification | strong |  |
| Discussion | Finetuning may reveal regional gaps because accuracy becomes nontrivial. | none | delta trade-off figure | derived results | medium | TODO: citation needed for metric-floor/compression interpretation. |
| Discussion | Model-family differences are descriptive only. | model papers for identity; no causal evidence | family comparison table/figure | model citations | medium | TODO: add controlled family analysis before stronger claims. |
| Limitations | Local dataset statistics are missing. | dataset papers; local audit needed | none | ramaswamy2023geode; rojas2022dollarstreet | medium | TODO: create local split/region/class table. |
| Limitations | Region gap does not fully capture fairness. | fairness and worst-group papers | none | gustafson2023facet; sagawa2020groupdro | medium | TODO: add confidence intervals or uncertainty estimates. |
