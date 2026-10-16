# ThreadMask: reproducible solar-filament research

Author: **Jacek Adamek**. Complete reproducibility update: **16 October 2026**.

The complete proposed competition update is at [submission-update-2026-10-16/README.md](submission-update-2026-10-16/README.md). It includes all three fitted models, pinned requirements, exporter, independent verifier, full-pipeline notebook and four-page report. No private weight directory or paid inference service is required; obtain the authorized competition test images separately.

The clean-clone full run regenerated all **180 images / 1136 instances** and exactly matched candidate CSV SHA256 `9d8f64cdca2fcf996ca8efefff5f6444ecee1480195ddb312a8e72029e63a639`. Eleven images have no detections. All six release tests passed. [Verification record](submission-update-2026-10-16/PUBLIC-REPRODUCTION.json).

Local reused-development PQ changes from 0.276155 to 0.277402, FP from 810 to 807 and thin annotation-level matches from 126/442 to 127/442. Confidence intervals include zero; additional 40-date validation adds no thin matches. These modest local results are **not a new Kaggle score**.

The three data-derived model files and attributed example graphics have separate CC BY-NC 4.0 notices. Code remains Apache-2.0. No original competition image/annotation archive, account fields or prediction CSV is published in this update.

## Earlier evidence and source release

The immutable [16 October source release](releases/2026-10-16/README.md), original root code and earlier reports remain available. The older release intentionally omitted weights; Solar's separate complete update now includes its three fitted models. Historical manifests describe their own artifacts and are unchanged.

**No competition update has been submitted by this GitHub publication.**
