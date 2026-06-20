# Citation Bank

This bank is intentionally selective. Use score 4-5 papers for main claims. Use score 3 papers only for background or model identity. Do not cite rejected papers in the draft.

## Central Dataset and Geographic Bias Papers

- citation key: ramaswamy2023geode
- one-sentence summary: Introduces GeoDE as a geographically diverse object-recognition evaluation dataset.
- why it is relevant: Core source for GeoDE and region-level object recognition.
- exact claim it supports: GeoDE is appropriate for evaluating geographic variation in object recognition.
- where to cite it: Introduction; Datasets; Related Work; Methodology.
- safe citation wording: GeoDE was introduced as a geographically diverse evaluation dataset for object recognition.
- warning if the paper should not be overused: Do not use it to explain Dollar Street results or model-family effects.

- citation key: rojas2022dollarstreet
- one-sentence summary: Introduces Dollar Street as a household-object image dataset designed around geographic and socioeconomic diversity.
- why it is relevant: Core source for Dollar Street.
- exact claim it supports: Dollar Street is appropriate for studying geographic and socioeconomic visual diversity.
- where to cite it: Introduction; Datasets; Related Work.
- safe citation wording: Dollar Street provides household-object imagery designed to capture geographic and socioeconomic diversity.
- warning if the paper should not be overused: Local split statistics still require a local audit.

- citation key: devries2019object
- one-sentence summary: Studies object-recognition performance on Dollar Street across countries and income contexts.
- why it is relevant: Closest prior motivation for object-recognition disparities in this setting.
- exact claim it supports: Object-recognition systems may perform unevenly across geographic and socioeconomic contexts.
- where to cite it: Introduction; Related Work; Discussion.
- safe citation wording: Prior Dollar Street work suggests object-recognition performance may vary across countries and income levels.
- warning if the paper should not be overused: Do not claim the same mechanism or magnitude for the present results without direct evidence.

- citation key: shankar2017geodiversity
- one-sentence summary: Audits geodiversity in open image datasets and classifier performance across locales.
- why it is relevant: Provides broad evidence that geographic representation in visual datasets matters.
- exact claim it supports: Open image datasets can be geographically skewed, which may matter for downstream classifiers.
- where to cite it: Introduction; Related Work; Discussion.
- safe citation wording: Geodiversity audits suggest that open image datasets can exhibit amerocentric and eurocentric representation bias.
- warning if the paper should not be overused: It predates the specific GeoDE/Dollar Street setup in this paper.

- citation key: kalluri2023geonet
- one-sentence summary: Introduces GeoNet for benchmarking unsupervised adaptation across geographies.
- why it is relevant: Provides a vocabulary for geographic shifts such as context, design, and prior shifts.
- exact claim it supports: Geographic robustness can involve shifts in scene context, object design, and label priors.
- where to cite it: Related Work; Discussion.
- safe citation wording: Geographic adaptation work identifies context, design, and prior shifts as possible sources of geographic robustness challenges.
- warning if the paper should not be overused: It studies adaptation benchmarks, not this paper's finetuning comparison.

## Dataset Bias, ImageNet, and Benchmark Evaluation

- citation key: torralba2011datasetbias
- one-sentence summary: Frames dataset bias and cross-dataset generalization in visual recognition.
- why it is relevant: Foundational dataset-bias framing.
- exact claim it supports: Recognition datasets can contain dataset-specific biases affecting generalization.
- where to cite it: Related Work.
- safe citation wording: Dataset-specific biases can affect visual-recognition generalization.
- warning if the paper should not be overused: Pair with newer robustness and benchmark papers.

- citation key: wang2022revise
- one-sentence summary: Presents a tool for auditing object-, person-, and geography-related visual dataset biases.
- why it is relevant: Supports geography-aware dataset auditing.
- exact claim it supports: Geography-aware auditing is a recognized concern in visual datasets.
- where to cite it: Related Work; Limitations.
- safe citation wording: Geography-aware dataset auditing has been proposed as a way to surface visual dataset biases.
- warning if the paper should not be overused: REVISE is not the metric pipeline used here.

- citation key: deng2009imagenet
- one-sentence summary: Introduces ImageNet as a large-scale hierarchical image database.
- why it is relevant: Background for pretrained ImageNet evaluation.
- exact claim it supports: ImageNet is the source benchmark for the pretrained evaluation setting.
- where to cite it: Datasets; Methodology.
- safe citation wording: ImageNet is a large-scale visual recognition dataset widely used for pretraining and evaluation.
- warning if the paper should not be overused: It does not support regional-bias claims.

- citation key: yang2020fairerimagenet
- one-sentence summary: Analyzes and mitigates problematic representation in ImageNet's people subtree.
- why it is relevant: Shows that ImageNet curation and taxonomy can carry representational issues.
- exact claim it supports: Benchmark datasets can require fairness-aware filtering and balancing.
- where to cite it: Related Work; Limitations.
- safe citation wording: Work on the ImageNet people subtree shows that dataset taxonomy and representation can require fairness-aware curation.
- warning if the paper should not be overused: It concerns people categories, not regional object accuracy.

- citation key: luccioni2022bugs
- one-sentence summary: Audits ImageNet wildlife classes for label, geographic, and cultural issues.
- why it is relevant: Supports caution about ImageNet as a neutral reference dataset.
- exact claim it supports: ImageNet can contain label and representation issues beyond aggregate accuracy.
- where to cite it: Related Work; Limitations.
- safe citation wording: Audits of ImageNet wildlife classes report label, geographic, and cultural issues.
- warning if the paper should not be overused: It focuses on biodiversity, not household objects.

- citation key: recht2019imagenet
- one-sentence summary: Tests whether ImageNet classifiers generalize to newly collected ImageNet-like test sets.
- why it is relevant: Supports benchmark generalization caution.
- exact claim it supports: High benchmark accuracy may not fully characterize generalization.
- where to cite it: Related Work; Discussion.
- safe citation wording: Newly collected ImageNet-like test sets can reveal drops in classifier accuracy.
- warning if the paper should not be overused: It does not evaluate geographic groups.

- citation key: taori2020naturalshifts
- one-sentence summary: Measures robustness to natural distribution shifts in image classification.
- why it is relevant: Places regional evaluation in the broader distribution-shift literature.
- exact claim it supports: Natural distribution shifts can remain challenging for image classifiers.
- where to cite it: Related Work; Discussion.
- safe citation wording: Natural image distribution shifts remain an important challenge for image classification.
- warning if the paper should not be overused: Regional shifts are related but not identical.

- citation key: koh2021wilds
- one-sentence summary: Introduces WILDS, a benchmark suite for in-the-wild distribution shifts.
- why it is relevant: Supports standardized real-world shift evaluation.
- exact claim it supports: Real-world shifts require benchmark protocols and subgroup-aware evaluation.
- where to cite it: Related Work; Methodology.
- safe citation wording: WILDS frames in-the-wild distribution shift as a recurring deployment challenge.
- warning if the paper should not be overused: It is broader than regional object recognition.

- citation key: gulrajani2021domainbed
- one-sentence summary: Studies domain generalization evaluation and model selection.
- why it is relevant: Supports caution about protocol and overclaiming.
- exact claim it supports: Domain generalization claims depend on careful evaluation protocols.
- where to cite it: Methodology; Related Work.
- safe citation wording: Domain generalization benchmarks highlight the importance of standardized protocols and model-selection choices.
- warning if the paper should not be overused: This paper does not train DG algorithms.

## Worst-Group, Fairness, and Hidden-Strata Evaluation

- citation key: sagawa2020groupdro
- one-sentence summary: Studies worst-group generalization under group shifts.
- why it is relevant: Supports worst-region accuracy as a complement to average accuracy.
- exact claim it supports: Average accuracy can obscure weak performance for specific groups.
- where to cite it: Problem Definition; Related Work.
- safe citation wording: Worst-group evaluation motivates reporting performance for weaker groups rather than only average accuracy.
- warning if the paper should not be overused: This paper is not using group DRO.

- citation key: idrissi2021databalancing
- one-sentence summary: Shows simple balancing can be competitive for worst-group accuracy.
- why it is relevant: Supports the importance of group/class balance in interpreting worst-group metrics.
- exact claim it supports: Group imbalance and model selection can strongly affect worst-group evaluation.
- where to cite it: Discussion; Limitations.
- safe citation wording: Worst-group benchmarks can be sensitive to group and class imbalance.
- warning if the paper should not be overused: It does not analyze geographic regions.

- citation key: sohoni2020hiddenstratification
- one-sentence summary: Studies hidden subclass failures inside coarse classification labels.
- why it is relevant: Analogous motivation for looking beyond object-level aggregate accuracy.
- exact claim it supports: Coarse labels can hide large performance variation across finer strata.
- where to cite it: Problem Definition; Discussion.
- safe citation wording: Hidden-stratification work shows that coarse labels may hide subgroup-specific failures.
- warning if the paper should not be overused: It is not specifically about geographic strata.

- citation key: gustafson2023facet
- one-sentence summary: Introduces a fairness benchmark for common computer-vision tasks.
- why it is relevant: Supports fairness evaluation beyond aggregate accuracy.
- exact claim it supports: Vision models can show performance disparities across annotated attributes.
- where to cite it: Related Work; Limitations.
- safe citation wording: Computer-vision fairness benchmarks evaluate performance disparities across annotated attributes.
- warning if the paper should not be overused: It focuses on demographics, not geographic regions.

## Spurious Correlations, Context, and Robustness Mechanisms

- citation key: geirhos2018texturebias
- one-sentence summary: Shows ImageNet-trained CNNs can rely on texture rather than shape.
- why it is relevant: Supports possible appearance/context explanations for regional object-recognition gaps.
- exact claim it supports: Visual models may rely on features that do not transfer robustly across conditions.
- where to cite it: Related Work; Discussion.
- safe citation wording: ImageNet-trained CNNs can exhibit texture bias, which is relevant to robustness under appearance shifts.
- warning if the paper should not be overused: It does not test regional datasets.

- citation key: geirhos2020shortcut
- one-sentence summary: Frames shortcut learning as reliance on easy rules that fail outside benchmark conditions.
- why it is relevant: Supports cautious interpretation of high accuracy and regional failures.
- exact claim it supports: Models can exploit shortcuts that work in benchmark settings but fail under harder shifts.
- where to cite it: Related Work; Discussion.
- safe citation wording: Shortcut learning provides one possible explanation for performance that does not transfer across conditions.
- warning if the paper should not be overused: It is a broad perspective, not a direct test here.

- citation key: lapuschkin2019cleverhans
- one-sentence summary: Uses explanation methods to reveal unintended cues behind high model performance.
- why it is relevant: Supports the argument that aggregate accuracy can hide fragile decision rules.
- exact claim it supports: High accuracy can mask reliance on unintended correlations.
- where to cite it: Related Work; Discussion.
- safe citation wording: Explanation-based audits show that models may rely on unintended cues despite strong aggregate performance.
- warning if the paper should not be overused: It does not explain the specific regional gaps in this paper.

## Vision-Language and Model-Family Context

- citation key: radford2021clip
- one-sentence summary: Introduces CLIP and contrastive language-image pretraining.
- why it is relevant: Model-family source for CLIP.
- exact claim it supports: CLIP is a vision-language model trained with natural-language supervision.
- where to cite it: Models.
- safe citation wording: CLIP uses contrastive language-image pretraining for transferable visual representations.
- warning if the paper should not be overused: Do not cite it for regional bias unless discussing model identity.

- citation key: fang2022cliprobustness
- one-sentence summary: Studies distributional robustness in CLIP-like models and the role of pretraining data.
- why it is relevant: Supports cautious VLM robustness discussion.
- exact claim it supports: CLIP-like robustness may be associated with pretraining data distribution.
- where to cite it: Related Work; Discussion.
- safe citation wording: Controlled CLIP-like experiments suggest that pretraining data distribution is associated with robustness.
- warning if the paper should not be overused: It does not test GeoDE or Dollar Street.

- citation key: zhai2023siglip
- one-sentence summary: Introduces SigLIP.
- why it is relevant: Model-family source for SigLIP.
- exact claim it supports: SigLIP is a language-image pretraining model using sigmoid loss.
- where to cite it: Models.
- safe citation wording: SigLIP modifies language-image pretraining with a sigmoid loss.
- warning if the paper should not be overused: Model identity only.

- citation key: oquab2024dinov2
- one-sentence summary: Introduces DINOv2 self-supervised visual features.
- why it is relevant: Model-family source for DINO.
- exact claim it supports: DINOv2 is a self-supervised visual representation model.
- where to cite it: Models.
- safe citation wording: DINOv2 learns visual features without supervised labels.
- warning if the paper should not be overused: Do not infer regional robustness from the model paper.

- citation key: he2022mae
- one-sentence summary: Introduces MAE masked image modeling.
- why it is relevant: Model-family source for MAE.
- exact claim it supports: MAE learns representations through masked image reconstruction.
- where to cite it: Models.
- safe citation wording: MAE learns visual representations through masked image reconstruction.
- warning if the paper should not be overused: Model identity only.

