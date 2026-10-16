# Final local review package

Author: Jacek Adamek. Public research archive; not submitted to the competition.

Start with ODTWORZENIE.txt and final-release.json. Run python -m pytest -q and python verify_final.py.

Recommended candidate: results/v3/candidate.csv (950 instances, 180 images). Current report: report/technical-report-final-ACM.pdf.

Development alternative: results/precision-guard/candidate.csv. It reduced false positives on two inspected assessments without lost hits; no independent validation. Never attach the alternative to the original blind score.

README-history.md and previous report/source files preserve the development record. No code publishes, submits, accepts terms or purchases services.

## Publication scope

This public release was explicitly authorized by the participant. Historical JSON files and local-review documents preserve their original dated publication/submission status; those snapshots do not describe the current visibility of this repository. No Kaggle entry or organizer-form submission is performed by publishing this archive.

Code is Apache-2.0. Data-derived previews and results have separate terms; read ASSET-LICENSES.txt. Full source observations and model-input archives are not distributed. The recommended candidate remains results/v3/candidate.csv; the precision guard is a separately labeled development alternative.

## Complete reproducible release

The root contains readable code, notebooks and reports. **Before running the pipeline or verifiers, run `python unpack_release.py` once.** This validates and expands the bundled `source-and-results.zip` into this checkout, adding every results subdirectory, frozen model/configuration, reference excerpt and historical record. The archive is part of this repository; no additional unpublished file is needed. Input images for Solar and optional Gemma weights remain separately acquired resources as documented. The archive contains no credentials, full input dataset or model weights.

Then install the pinned project requirements in your own Python environment and follow the existing reproduction instructions. Publication is not a Kaggle or organizer-form submission. Dated publication flags inside experimental snapshots are historical records.
