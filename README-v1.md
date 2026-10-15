# ThreadMask — Solar Filament CPU baseline

Original local prototype, not submitted. Authorized competition data has now been
downloaded and a frozen holdout evaluation is recorded in results/validation.json.
Measured diagnostic PQ: 0.010915 on 135 observations (229 annotation sets), with
date-cluster bootstrap 95% interval 0.008896–0.013011. This is a weak baseline.
Local predictions cover all 180 test images; nothing has been submitted.
See results/summary.json for measured results and limitations.
Code and documentation prepared with Codex assistance.

The method estimates smooth local background, measures dark residual in robust noise
units, grows weak candidates only when their component contains a strong seed, and
filters small components. It avoids closing/dilation that could merge nearby threads.
These choices are hypotheses, not established advantages. Fixed centered disk geometry
can fail on off-center or poor observations; sunspots and artifacts can be false positives.

Python 3.12 CPU workflow, after the user authorizes Kaggle terms and obtains the data:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python pipeline.py evaluate /path/to/MAGFiLO_1.0_Kaggle_2026 results/validation.json
python pipeline.py predict /path/to/MAGFiLO_1.0_Kaggle_2026 results/candidate.csv
```

See pipeline.ipynb for the same reproducible pipeline. No API keys or paid services.
The package is source code only; data has separate CC BY-NC 4.0 terms and must not be
redistributed under the code license. No pretrained models, training or external metadata.

Evaluation uses a frozen date-group holdout: sha256(YYYYMMDD), first 8 hex digits modulo
5 equals zero. All annotators and instruments for a day remain in the same partition.
Annotation-set PQ is averaged within observation, then across observations. No parameters
have been tuned on these data. The local PQ implementation requires parity checks against
the organizer notebook; empty/overlapping-mask edge cases must be checked before claims.
Do not select parameters using test labels or reverse-engineer the leaderboard.

Prediction format: one row per predicted instance, unique observation_id_suffix, COCO
compressed RLE counts for a 2048x2048 mask. Every encoding is decoded and compared before
writing. An audit sidecar lists input hashes, per-image instance counts and output hash.
Images with no detected objects have no CSV row; official handling needs verification.
This output is a candidate, not a certified submission. The pipeline uses compressed
COCO masks and releases each observation before processing the next to limit memory.

The preview uses the first three holdout filenames, chosen independently of scores.
It reveals false positives on the solar limb and missed filaments. Future development
after inspecting those errors is adaptive; disclose this and reserve untouched days
for a final assessment. No leaderboard result or organizer-metric parity is claimed.

Before contest readiness: download authorized data, inspect actual annotations, validate
metric parity and schema, run grouped validation, compare threshold-only and alternative
baselines, inspect segmentation errors, report runtime/memory and uncertainty, freeze the
configuration, verify full test coverage, and prepare the required 4-page PDF using the
organizer template. Participant must understand and defend every major method component.

Final evaluation also requires a public Git repository, versioned requirements, end-to-end
notebook and the organizer Google form. A CSV alone is insufficient. Publishing, joining,
accepting rules and submitting remain subject to user approval.

Sources: https://www.kaggle.com/competitions/filament-segmentation-2026/overview
https://www.kaggle.com/competitions/filament-segmentation-2026/rules
https://www.kaggle.com/competitions/filament-segmentation-2026/data
