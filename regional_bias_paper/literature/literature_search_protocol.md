# Literature Search Protocol

This protocol is designed for a comprehensive, transparent search. The current workspace contains a seed bibliography only and does not claim to include all relevant papers.

## Databases and Sources to Search

- arXiv
- OpenReview
- NeurIPS proceedings
- CVF Open Access
- ICML/PMLR proceedings
- ICLR/OpenReview proceedings
- ACM Digital Library
- IEEE Xplore
- Semantic Scholar
- Google Scholar
- Official dataset and benchmark pages
- Official model repositories when they include citation metadata

## Search Queries

- "geographic bias computer vision object recognition"
- "regional bias image classification object recognition"
- "GeoDE geographically diverse evaluation dataset object recognition"
- "Dollar Street dataset geographic socioeconomic diversity computer vision"
- "Does Object Recognition Work for Everyone Dollar Street"
- "dataset bias object recognition cross dataset generalization"
- "ImageNet bias geographic representation computer vision"
- "natural distribution shift image classification robustness"
- "domain generalization computer vision benchmark DomainBed"
- "WILDS benchmark distribution shifts computer vision"
- "worst-group accuracy computer vision fairness"
- "regional fairness metrics image classification"
- "CLIP robustness bias distribution shift"
- "SigLIP robustness bias image-text pretraining"
- "DINOv2 robustness generalization self-supervised vision"
- "MAE robustness generalization self-supervised vision"
- "CNN transformer robustness image classification"
- "ConvNeXt Swin DeiT MaxViT robustness ImageNet shifts"
- "ObjectNet bias controlled object recognition benchmark"
- "visual dataset geography bias audit REVISE"

## Inclusion Criteria

Include papers if they address at least one of the following:

- regional or geographic bias in computer vision
- dataset bias in visual recognition
- visual domain shift or out-of-distribution generalization
- object-recognition robustness
- fairness or worst-group performance in vision
- GeoDE
- Dollar Street
- ImageNet bias, representation, or generalization
- CLIP or SigLIP robustness, fairness, or bias
- DINOv2, MAE, or self-supervised robustness
- domain generalization
- evaluation beyond aggregate accuracy

## Exclusion Criteria

Exclude papers if they are:

- only about NLP bias with no vision relevance
- only about medical imaging unless the method is directly relevant to domain or group robustness
- only about generative image models with no evaluation relevance
- not connected to geographic, domain, dataset bias, or visual robustness
- inaccessible and lacking enough reliable metadata to cite

## Screening Process

1. Record every candidate in `paper_inventory.csv` with status `candidate`.
2. Confirm metadata from a primary source where possible.
3. Assign a relevance score from 1 to 5.
4. Move non-relevant papers to `rejected_papers.csv` with a reason and date checked.
5. Add accepted papers to `related_work_matrix.csv`.
6. Add citation wording to `citation_bank.md`.
7. Add any reusable evidence-backed claims to `claim_bank.md`.
8. Add BibTeX only after metadata is available from a reliable source.

## Relevance Ranking

- 5: Directly about GeoDE, Dollar Street, or geographic/socioeconomic object-recognition bias.
- 4: Directly about dataset bias, distribution shift, ImageNet generalization, or CLIP robustness relevant to the research question.
- 3: Model architecture or method paper required to describe evaluated model families.
- 2: Adjacent fairness or robustness work that may support limitations or future work.
- 1: Background-only paper with weak connection to the final paper.

