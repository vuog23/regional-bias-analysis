# Deep Research Prompt

You are conducting a comprehensive, transparent literature review for a research paper on regional bias analysis of computer vision models on GeoDE and Dollar Street.

Important constraints:

- Do not fabricate citations, papers, metadata, URLs, DOIs, arXiv IDs, or findings.
- Do not claim that the search found all possible papers.
- Prefer peer-reviewed papers, arXiv papers, OpenReview papers, official dataset papers, official benchmark papers, and official model papers.
- Generate BibTeX entries only when reliable source metadata is available.
- Use cautious language: suggests, indicates, may reflect, is consistent with, and may be associated with.
- Do not claim causality unless the source directly supports it.

Paper topic:

Regional bias analysis of computer vision models on GeoDE and Dollar Street, comparing pretrained ImageNet evaluation and finetuned evaluation.

Main research question:

How do pretrained and finetuned vision models differ in regional bias across GeoDE and Dollar Street?

Secondary research questions:

1. Which models have the highest object accuracy?
2. Which models have the lowest regional bias gap?
3. Does finetuning improve object accuracy?
4. Does finetuning reduce or increase regional bias gap?
5. Are some model families more regionally stable than others?
6. Are high-accuracy models always less biased across regions?
7. How do CNN, transformer, self-supervised, and vision-language models differ in regional robustness?

Search themes:

1. Geographic or regional bias in computer vision.
2. GeoDE and geographically diverse object recognition.
3. Dollar Street and socioeconomic/geographic visual diversity.
4. Dataset bias in object recognition.
5. Domain shift and out-of-distribution generalization.
6. Domain generalization in computer vision.
7. ImageNet bias and dataset representation.
8. Vision-language model bias and robustness.
9. CLIP and SigLIP robustness, fairness, or bias.
10. Self-supervised visual representation learning and robustness.
11. DINOv2 and MAE robustness/generalization.
12. CNN versus transformer robustness.
13. Regional fairness metrics and worst-group accuracy.
14. Model evaluation beyond aggregate accuracy.

Seed papers to verify and expand from:

- GeoDE: A Geographically Diverse Evaluation Dataset for Object Recognition.
- The Dollar Street Dataset: Images Representing the Geographic and Socioeconomic Diversity of the World.
- Does Object Recognition Work for Everyone?
- Unbiased Look at Dataset Bias.
- REVISE: A Tool for Measuring and Mitigating Bias in Visual Datasets.
- ImageNet: A Large-Scale Hierarchical Image Database.
- Do ImageNet Classifiers Generalize to ImageNet?
- Measuring Robustness to Natural Distribution Shifts in Image Classification.
- WILDS: A Benchmark of in-the-Wild Distribution Shifts.
- In Search of Lost Domain Generalization.
- Learning Transferable Visual Models From Natural Language Supervision.
- Data Determines Distributional Robustness in Contrastive Language Image Pre-training (CLIP).
- Sigmoid Loss for Language Image Pre-Training.
- DINOv2: Learning Robust Visual Features without Supervision.
- Masked Autoencoders Are Scalable Vision Learners.
- A ConvNet for the 2020s.
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
- Training Data-Efficient Image Transformers and Distillation through Attention.
- MaxViT: Multi-Axis Vision Transformer.
- Deep Residual Learning for Image Recognition.

Please produce:

1. A ranked paper list with title, authors, year, venue, URL, DOI/arXiv/OpenReview link if available, paper type, dataset focus, model focus, main topic, relevance score from 1 to 5, and reason for inclusion.
2. BibTeX entries for papers with reliable metadata.
3. A related work matrix with columns: citation_key, title, year, topic_group, dataset, models, method, main_finding, limitation, how_it_supports_this_paper, where_to_cite, relevance_score_1_to_5.
4. A citation bank with citation key, short paper summary, exact reason to cite it, which section it supports, and safe citation wording.
5. A claim bank with claim, supporting citation key, evidence from paper, safe wording, section where the claim can be used, and warning if the claim is weak.
6. An evidence map with paper section, claim needed, supporting source, citation key, whether evidence is strong/medium/weak, and TODO where source is missing.
7. A list of rejected or low-relevance papers with title, URL, date checked, and reason for rejection.
8. A short explanation of remaining literature gaps and recommended next searches.

