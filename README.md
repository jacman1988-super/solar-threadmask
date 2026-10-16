# ThreadMask: solar-filament research

Author: **Jacek Adamek**.

## Complete competition-update materials

Start with [submission-update-2026-10-16/README.md](submission-update-2026-10-16/README.md) for the latest notebook, exact commands, integrity checks and proposed competition-update documents.


Author: **Jacek Adamek**. Latest research update: **16 October 2026**.

The new release adds full-resolution ridge refinement and a dated reproduction package. On the combined development comparison (146 images / 135 dates), PQ changes from **0.276155 to 0.277402**, false positives from **810 to 807**, and thin annotation-level matches from **126/442 to 127/442**.

This is a small development improvement: confidence intervals include zero and the data are not a blind holdout. Additional 40-date validation adds no thin matches. These are local measurements, not a new Kaggle score. The report preserves unsuccessful alternatives and limitations.

Release checks: **6 tests passed**, and image-only prediction exactly reproduced the frozen reference label-array hash.

## Reproduce this update

See [release README](releases/2026-10-16/README.md), [reproduction instructions](releases/2026-10-16/REPRODUCE.md), [research report](releases/2026-10-16/REPORT.pdf) and [validation record](releases/2026-10-16/VALIDATION.json).

Run the documented commands from `releases/2026-10-16/`. The dated release includes its own file manifest and offline integrity verifier. Its preparation records describe the local checks performed before this GitHub update. Full training and fresh model generations were not repeated during packaging.

## Earlier version

The [previous README](README-before-2026-10-16.md), original root code and original reports remain available. This GitHub update does not submit or modify a competition entry. Competition inputs, trained weights and new prediction CSVs are not included in the update.
