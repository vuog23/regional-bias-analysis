# Citation Bank

This bank is a seed, not a complete literature review. Use these entries only when the corresponding BibTeX key exists in `paper/references.bib` or `literature/bibtex_raw.bib`.

## GeoDE and Dollar Street

- citation key: ramaswamy2023geode
- short paper summary: Introduces GeoDE, a geographically diverse evaluation dataset for object recognition.
- exact reason to cite it: Dataset source and motivation for region-level object recognition evaluation.
- section support: Related Work; Datasets; Methodology.
- safe citation wording: GeoDE was introduced as a geographically diverse object-recognition evaluation dataset.

- citation key: rojas2022dollarstreet
- short paper summary: Introduces the Dollar Street dataset with household-object images and demographic metadata including region, country, and income.
- exact reason to cite it: Dataset source for Dollar Street and its geographic/socioeconomic motivation.
- section support: Related Work; Datasets.
- safe citation wording: Dollar Street provides household-object imagery designed to capture geographic and socioeconomic diversity.

- citation key: devries2019object
- short paper summary: Evaluates object recognition systems on Dollar Street and reports disparities across countries and income levels.
- exact reason to cite it: Prior work directly motivating regional and socioeconomic object recognition analysis.
- section support: Introduction; Related Work; Discussion.
- safe citation wording: Prior work on Dollar Street suggests object-recognition systems may perform unevenly across countries and income levels.

## Dataset Bias and Distribution Shift

- citation key: torralba2011datasetbias
- short paper summary: Studies dataset bias and cross-dataset generalization in visual recognition.
- exact reason to cite it: Foundational conceptual framing for dataset bias.
- section support: Related Work.
- safe citation wording: Dataset-specific biases can affect cross-dataset generalization in recognition systems.

- citation key: wang2022revise
- short paper summary: Presents REVISE, a tool for auditing object-, person-, and geography-related biases in visual datasets.
- exact reason to cite it: Supports geography-aware dataset auditing context.
- section support: Related Work; Limitations.
- safe citation wording: Geography-based dataset auditing has been proposed as one way to surface visual dataset biases.

- citation key: deng2009imagenet
- short paper summary: Introduces ImageNet as a large-scale hierarchical image database.
- exact reason to cite it: Background for ImageNet-pretrained evaluation.
- section support: Datasets; Experiments.
- safe citation wording: ImageNet is a large-scale image database widely used for visual recognition pretraining and evaluation.

- citation key: recht2019imagenet
- short paper summary: Studies whether ImageNet classifiers generalize to newly collected ImageNet-like test sets.
- exact reason to cite it: Supports caution about benchmark reuse and generalization.
- section support: Related Work; Discussion.
- safe citation wording: Evaluations on newly collected ImageNet-like test sets indicate that benchmark accuracy may not fully characterize generalization.

- citation key: taori2020naturalshifts
- short paper summary: Measures robustness of ImageNet models under natural distribution shifts.
- exact reason to cite it: Supports natural distribution-shift framing.
- section support: Related Work; Discussion.
- safe citation wording: Natural image distribution shifts remain an important challenge for image classification robustness.

- citation key: koh2021wilds
- short paper summary: Introduces WILDS, a benchmark suite for in-the-wild distribution shifts.
- exact reason to cite it: Places regional evaluation within broader real-world shift evaluation.
- section support: Related Work.
- safe citation wording: WILDS frames distribution shifts as a recurring challenge in real-world model deployment.

- citation key: gulrajani2021domainbed
- short paper summary: Presents DomainBed and emphasizes careful model selection and reproducibility in domain generalization.
- exact reason to cite it: Supports transparent evaluation protocol and caution about domain generalization claims.
- section support: Related Work; Methodology.
- safe citation wording: Domain generalization benchmarks highlight the importance of standardized protocols and model-selection choices.

- citation key: barbu2019objectnet
- short paper summary: Introduces ObjectNet, a bias-controlled object-recognition test set.
- exact reason to cite it: Supports evaluating object recognition beyond conventional benchmark conditions.
- section support: Related Work.
- safe citation wording: Bias-controlled object-recognition benchmarks provide complementary evidence about robustness beyond standard test sets.

- citation key: sagawa2020groupdro
- short paper summary: Studies worst-group generalization under group distribution shifts.
- exact reason to cite it: Supports use of worst-region accuracy as a complement to average accuracy.
- section support: Problem Definition; Related Work.
- safe citation wording: Worst-group evaluation is useful because average accuracy can obscure poor performance for specific groups.

## Model Families

- citation key: radford2021clip
- short paper summary: Introduces CLIP, a contrastive language-image pretraining approach.
- exact reason to cite it: Model-family source for CLIP.
- section support: Related Work; Models.
- safe citation wording: CLIP is a vision-language model trained with natural-language supervision over image-text pairs.

- citation key: fang2022cliprobustness
- short paper summary: Performs controlled experiments on CLIP-like model robustness and the role of training data.
- exact reason to cite it: Supports cautious discussion of CLIP robustness.
- section support: Related Work; Discussion.
- safe citation wording: Controlled studies suggest that pretraining data distribution may be associated with CLIP-like robustness.

- citation key: zhai2023siglip
- short paper summary: Introduces SigLIP, which uses sigmoid loss for language-image pretraining.
- exact reason to cite it: Model-family source for SigLIP.
- section support: Models.
- safe citation wording: SigLIP modifies language-image pretraining with a sigmoid loss over image-text pairs.

- citation key: oquab2024dinov2
- short paper summary: Introduces DINOv2, self-supervised visual features trained on curated large-scale image data.
- exact reason to cite it: Model-family source for DINO-style self-supervised evaluation.
- section support: Related Work; Models.
- safe citation wording: DINOv2 is a self-supervised visual representation model trained on curated large-scale image data.

- citation key: he2022mae
- short paper summary: Introduces masked autoencoders for scalable self-supervised vision learning.
- exact reason to cite it: Model-family source for MAE.
- section support: Models.
- safe citation wording: MAE learns visual representations through masked image reconstruction.

- citation key: liu2022convnext
- short paper summary: Introduces ConvNeXt, a modernized ConvNet architecture.
- exact reason to cite it: Model-family source for ConvNeXt.
- section support: Models.
- safe citation wording: ConvNeXt revisits CNN design choices in light of modern vision architectures.

- citation key: liu2021swin
- short paper summary: Introduces Swin Transformer with shifted-window hierarchical attention.
- exact reason to cite it: Model-family source for Swin.
- section support: Models.
- safe citation wording: Swin Transformer uses a hierarchical shifted-window attention design for vision tasks.

- citation key: touvron2021deit
- short paper summary: Introduces data-efficient image transformers and transformer distillation.
- exact reason to cite it: Model-family source for DeiT.
- section support: Models.
- safe citation wording: DeiT studies data-efficient training and distillation for vision transformers.

- citation key: tu2022maxvit
- short paper summary: Introduces MaxViT with multi-axis attention.
- exact reason to cite it: Model-family source for MaxViT.
- section support: Models.
- safe citation wording: MaxViT combines blocked local and dilated global attention in a hierarchical vision backbone.

- citation key: he2016resnet
- short paper summary: Introduces residual learning for deep image-recognition networks.
- exact reason to cite it: Model-family source for ResNet.
- section support: Models.
- safe citation wording: ResNet introduced residual connections for training deep convolutional networks.

