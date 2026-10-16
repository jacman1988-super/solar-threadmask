# Solar: complete reproduction materials

Author: Jacek Adamek. Proposed competition update, 16 October 2026.
No competition update has been submitted. The old Kaggle score does not apply to the new candidate.

## Exact candidate reproduction from a fresh clone

Use Python 3.12 on Windows (the audited platform). Obtain the authorized competition dataset from [the organizer](https://www.kaggle.com/competitions/filament-segmentation-2026/data). Only its test/test_images directory is needed for frozen inference. Training images and masks are not needed for the commands below. The three fitted models are included here; no private weight folder or paid service is required.

From the repository root:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r submission-update-2026-10-16/requirements.txt
.venv/Scripts/python submission-update-2026-10-16/verify_materials.py
.venv/Scripts/python releases/2026-10-16/verify_release.py
.venv/Scripts/python submission-update-2026-10-16/predict_submission.py --images AUTHORIZED_DATA/test/test_images --output work/reproduced/candidate-ridge-v8.csv --workers 3
.venv/Scripts/python submission-update-2026-10-16/verify_candidate.py --candidate work/reproduced/candidate-ridge-v8.csv --images AUTHORIZED_DATA/test/test_images
```

Replace AUTHORIZED_DATA with the dataset directory, quoting paths containing spaces. On Linux/macOS use .venv/bin/python; those platforms have not been validated here. The exporter refuses to overwrite an existing CSV. Choose a new output directory for each run. The exact expected CSV SHA256 is `9d8f64cdca2fcf996ca8efefff5f6444ecee1480195ddb312a8e72029e63a639` (180 images; 1136 instances; 11 zero-detection images). No candidate CSV or original images are redistributed: regenerate it with the public fitted models and authorized test inputs.

## Notebook and reports

pipeline-update.ipynb illustrates preprocessing, baseline segmentation, full-resolution refinement, COCO RLE round-trips and full-set export. It can run from the repository root or this directory. Set SOLAR_DATA to the dataset directory; set SOLAR_EXPORT_ALL=1 to enable the full final cell. pipeline-update-executed.ipynb and PUBLIC-REPRODUCTION.json record the actual clean-copy run, including exact CSV comparison. Notebook dependencies are the pinned scientific packages in requirements.txt; use your existing Jupyter frontend. The recorded execution engine is explicit plain CPython execution of code cells, not a claim about a Jupyter server.

technical-report-update.pdf is the four-page organizer-template report; its editable source, preamble and attributed examples are provided. Expected aggregate results and per-input hashes are in the accompanying audit files. The older immutable source release remains under releases/2026-10-16, with full training instructions and study history. Its statement that weights were omitted describes that older source-only release; this separate update supplies them.

## Scope and licensing

The local development improvement is small, confidence intervals include zero, and additional 40-date validation adds no thin matches. No new Kaggle score is known. Source code is Apache-2.0. Data-derived fitted models and example graphics retain separate CC BY-NC 4.0 terms; see weights/LICENSE.txt and ASSET-NOTICES.txt. No observation/annotation archive, account data or competition-form fields are included.
