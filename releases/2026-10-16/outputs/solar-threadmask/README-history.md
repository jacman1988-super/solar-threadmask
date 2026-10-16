# ThreadMask v3 — learned component rejection

Author: Jacek Adamek.

Local review candidate, unpublished and not submitted. Original code with Codex
assistance; participant review and understanding remain required.

The frozen v2 CPU candidate generator is followed by a regularized logistic
filter with eleven shape/contrast/disk-distance features. It only removes whole
components. Training uses 170 images, threshold tuning 55, and a new once-opened
assessment 106 images from 96 dates; 106 more images remain unused. Every date
from all prior experiments is excluded. Nearby dates may remain correlated.

Matched public-v6 pooled PQ on the same 106 images:
- v2: 0.10528875; v3: 0.17728248.
- v3 date-cluster 95% interval: [0.14937464, 0.20402947].
- paired improvement interval: [0.05540742, 0.08763434], 2000 resamples.
- TP 368 -> 281; FP 2899 -> 588; FN 802 -> 889.
The filter reduces false positives but also loses true filaments. These are
local results, not leaderboard scores. Historical v2 PQ 0.09749 used a different
100-image set and is not the baseline for this comparison.

The promotion rule was written before assessment and is met. The recommended
local CSV is results/v3/candidate.csv, bound by results/recommended-candidate.json.
It covers 180 test observations, 950 objects and 11 zero-detection images. Every
RLE was decoded and compared with exact instance pixels. Zero detections means
zero object rows under the inspected public evaluator. Private upload validation
and private-scorer equivalence remain untested. No placeholder object is added.

Twenty-five tests pass. The known pycocotools/NumPy deprecation warning remains
recorded; mask equality checks pass. NumPy/SciPy/Pillow/pycocotools versions are
pinned in requirements.txt. All coefficients and source/input hashes are included.

Reproduce in a fresh output directory (the original protocol is protected):
```sh
python -m unittest discover -v
python experiment_v3.py freeze /path/to/MAGFiLO_1.0_Kaggle_2026 /path/to/cache --output results/v3-reproduction
python experiment_v3.py fit /path/to/MAGFiLO_1.0_Kaggle_2026 /path/to/cache --output results/v3-reproduction
python experiment_v3.py tune /path/to/MAGFiLO_1.0_Kaggle_2026 /path/to/cache --output results/v3-reproduction
python experiment_v3.py evaluate /path/to/MAGFiLO_1.0_Kaggle_2026 /path/to/cache --output results/v3-reproduction
python predict_v3.py /path/to/MAGFiLO_1.0_Kaggle_2026 results/v3-reproduction/candidate.csv --model-dir results/v3-reproduction
```
Rerunning the inspected assessment reproduces computation, not new blind evidence.
pipeline-v3.ipynb documents this sequence but has not been executed as a notebook;
the actual CLI experiments and their raw records are included in results/v3/.

Current report: report/technical-report-v3-ACM.pdf, compiled from report-v3.tex
with the organizer preamble. All three pages rendered and visually checked.
The report includes the precision/recall tradeoff, prospective protocol, uncertainty,
exact public-reference arithmetic checks, first-three-filename examples and credits.
README-v2.md, earlier sources, CSVs and reports remain historical records.

Code is Apache 2.0; dataset and derivative assets have separate terms (CC BY-NC 4.0).
See ASSET-LICENSES.txt. No original dataset archive is included. No code joins,
publishes or submits to Kaggle, accepts rules, or purchases compute.

Full reproduction now executed: pipeline-v3-executed.ipynb contains actual output
from all seven Python code cells, using the explicitly recorded plain-CPython
engine rather than a Jupyter kernel. A fresh cache and output directory reproduce
the model bytes, CSV bytes and assessment metrics exactly. Audit:
results/v3/reproduction-audit.json. The editable pipeline-v3.ipynb remains the
unexecuted template. This is reproduction of inspected data, not a new blind score.

submission-draft.json records the official organizer form fields and local file
size/hash checks. Author name is confirmed; a public Git URL remains pending; no account
switch, upload, form submission or publication was performed.

Additional final review: `results/v3/final-review.json`; `audit_candidate.py` validates all instances independently. All 29 Solar tests pass. Archive-source prediction reproduced the exact CSV.

Post-hoc image-level tradeoff: results/v3/tradeoff-review.json (73 improved, 18 worse, 15 unchanged); no retuning.

Recall-oriented variant: recall_experiment.py and results/recall-v4/. A tune-only frozen threshold .075 increased recall on 106 previously unused images from .22449 to .29902, with PQ .16127 to .15740 and FP 625 to 1247. Original candidate unchanged; all remaining unused dates are now inspected.

2026-10-15 precision guard: results/precision-guard/; FP 1247 to 1094, TP 337 unchanged, PQ .15740 to .16681 on previously inspected dates. Development evidence only. Actual inference checked on three filenames; original CSV unchanged.

Frozen guard cross-check on earlier 106 assessment images: FP 1165 to 1064, TP 350 unchanged. Both inspected assessments total 212 images, TP 687 unchanged and FP 2412 to 2158. Still no independent validation. Results: results/guard-previous-assessment/.
