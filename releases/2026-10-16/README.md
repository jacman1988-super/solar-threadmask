# ThreadMask: reproducible solar-filament research

Author: **Jacek Adamek**. Prepared source release: **2026.10.16**.

This package contains the baseline pixel/component models' source, a new full-resolution ridge model's source, frozen experiment protocols, positive and negative outcomes, and a publication report. It prepares an update to `jacman1988-super/solar-threadmask`; it does not publish, upload predictions or modify the competition entry.

## Results

All figures below use the same local public organizer metric, evaluated separately for each annotation set. Thin counts are **annotation-level matches**, not a count of unique physical filaments.

| Evaluation | Baseline PQ | Ridge PQ | Baseline FP | Ridge FP | Baseline thin matches | Ridge thin matches |
|---|---:|---:|---:|---:|---:|---:|
| Original 106 images / 95 dates | 0.271485 | 0.272835 | 550 | 548 | 82/314 | 83/314 |
| Additional 40 images / 40 dates | 0.285986 | 0.287015 | 260 | 259 | 44/128 | 44/128 |
| Combined 146 images / 135 dates | 0.276155 | 0.277402 | 810 | 807 | 126/442 | 127/442 |

The extra dates were excluded from the current model's training, tuning and first assessment splits. Older baseline experiments had nevertheless used the available dataset. **These are development results, not an independent blind test or a new Kaggle score.** The change is small; descriptive confidence intervals include zero. The additional set improves overall PQ/FP but does not independently add thin matches.

## Start here

1. Read [REPORT.pdf](REPORT.pdf) and [REPRODUCE.md](REPRODUCE.md).
2. Install Python 3.12 dependencies with `python -m pip install -r requirements.txt`.
3. Run `python verify_release.py` and `python -m pytest -q` for offline package checks.
4. For local predictions, provide the separately held reference weights:

```text
python predict.py IMAGE.jpeg LABELS.npy --weights PRIVATE_WEIGHT_DIRECTORY --model ridge
```

The directory must contain `pixel-forest.joblib`, `component-filter.joblib` and `ridge-forest.joblib`. Input is a 2048x2048 grayscale image with integer values 0..255. Existing output files are not overwritten. All weight hashes are verified. The result is local instance labels, not a submitted competition CSV.

## Method and scope

The new regressor uses ten original-resolution image features, including dark-ridge Hessian responses at three scales. It was trained on 60 selected training images and 335,163 sampled pixels. It refines existing masks without introducing instance IDs. Parameters were selected on 19 tuning images before the 106-image assessment, then held fixed for the additional 40 dates.

Earlier attempts that added thin components, changed geometric contours, or substituted masks are retained, including failures. Historical files under `outputs/` are the evidence archive; their old readiness/status fields describe earlier stages. Use this root README and root commands as the release entry points. Historical manifests describe the original full local runs; `RELEASE_MANIFEST.json` describes this source-only package.

## Distribution

Competition images, annotations, trained weights, cached predictions and submission CSVs are **not included**. Obtain data through the organizer and observe its separate terms. Original code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). No preview images derived from the competition data are redistributed in this update.
