# ThreadMask v2 — reproducible Solar Filament CPU project

Local prototype, not submitted. Code and documentation prepared with Codex assistance.

On the same prospective holdout of 100 observations, public self-evaluation v6
pooled PQ improves from 0.009788 (v1) to 0.097488 (v2). Pure public scoring-function
parity was checked on all real overlap matrices with zero score difference using
a NumPy compatibility adapter. Torch and the private Kaggle scoring service were
not executed. See results/public-v6/evaluation.json for exact scope and uncertainty.
No leaderboard score or scientific novelty is claimed.

Historical image-averaged diagnostic results (0.010947 and 0.099168) are retained
in results/v2; they use a different matching/aggregation convention and must not
be presented as public-v6 pooled PQ. Segmentation settings did not change in this audit.

The disk is inferred from the image. Background smoothing is normalized by disk
occupancy; a limb margin reduces edge artifacts. Relative dark contrast, seeded
components, one closing step and area filters produce instances at 1024 resolution;
labels are enlarged to 2048 using nearest neighbor. See method-v2.txt for limitations.

Python 3.12, CPU only, no API keys or paid services. Install requirements.txt.
After obtaining authorized competition data, run from the project root:

```sh
python -m unittest -v
python experiment.py freeze --data /path/to/MAGFiLO_1.0_Kaggle_2026
python experiment.py tune --data /path/to/MAGFiLO_1.0_Kaggle_2026
python experiment.py evaluate --data /path/to/MAGFiLO_1.0_Kaggle_2026
python score_public.py /path/to/MAGFiLO_1.0_Kaggle_2026
python predict_v2.py /path/to/MAGFiLO_1.0_Kaggle_2026 results/v2/candidate.csv
```

pipeline-v2.ipynb documents the complete workflow. Recorded outputs are included;
rerunning them does not make an already inspected holdout blind again. The stored
split predates tuning: 35 images from 32 days for tuning, 100 images from 96 other
days for evaluation. All 24 trials are retained. The older 135-image set informed
this redesign and was excluded from the new assessment. Day grouping cannot rule
out persistence of the same physical structures across adjacent days.

Each predicted RLE is decoded and compared against exact label pixels. The local
candidate covers 180 test images. Audit checks verify counts, IDs and checksums.
Zero detections are represented by zero object rows, consistent with the inspected public v6 evaluator; its unchanged empty decoder was executed. See results/public-v6/empty-prediction-audit.json. Private upload validation and private-scorer equivalence have not been tested.
An acmart-compiled PDF (3 pages) and editable source are in report/. All pages were rendered and visually checked. The earlier ReportLab proof is retained separately. Participant review remains pending.
The participant must understand and defend all major components.

Original observations and annotations are not included. Dataset terms are separate
CC BY-NC 4.0; code is Apache 2.0. Preview images retain separate dataset terms.
See ASSET-LICENSES.txt. The older baseline and its results remain in README-v1.md,
pipeline.py and results/summary.json; use the v2 commands above for the new method.

Official sources:
https://www.kaggle.com/competitions/filament-segmentation-2026/overview
https://www.kaggle.com/competitions/filament-segmentation-2026/rules
https://www.kaggle.com/competitions/filament-segmentation-2026/data

No code in this project joins, publishes or submits to Kaggle.
