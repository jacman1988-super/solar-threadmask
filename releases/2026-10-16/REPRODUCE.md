# Reproduction and local inference

Use Python 3.12 and the pinned dependencies in `requirements.txt`. PyTorch is not needed for the supported baseline/ridge path; historical U-Net studies have separate archived requirements.

## Offline package checks

```text
python verify_release.py
python -m pytest -q
python predict.py --help
```

The source package deliberately omits weights. The local `--weights` adapter changes only artifact paths; the frozen model digests, features and thresholds are checked by the archived runtime. Never load an untrusted joblib file.

## Train in a new working directory

Do not overwrite frozen evidence. From this release root:

```text
python prepare_reproduction.py /ABSOLUTE/NEW_WORKSPACE
```

The destination must not exist. Supply the authorized dataset there at:

```text
work/solar-data/MAGFiLO_1.0_Kaggle_2026/
    train/train_images/*.jpeg
    train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
```

Then, from the new workspace, run in this order:

```text
python outputs/solar-research-v5/research.py rf
python outputs/solar-research-v5/research.py tune --method rf
python outputs/solar-research-v5/component_followup.py fit
python outputs/solar-ridge-v8/train_evaluate.py
```

These commands reproduce the baseline forest, baseline component filter, ridge training, tuning and the 106-image comparison in a fresh experiment tree. They use only segmentation masks for supervision. They do not require the original legacy-v3 probability cache or train the historical U-Net models. Full training was performed during the recorded study; this release preparation tested workspace setup and unchanged reference inference rather than repeating all training.

For predictions from the new experiment, gather its three newly trained `.joblib` files into a private weights directory and run its generated `predict.py`. That new workspace's own frozen protocols identify its new artifacts. Do not replace digests in the published evidence or claim new serialized weights reproduce the historical byte hashes. Keep fresh outcomes separate and compare measured metrics.

## Recorded evidence

- `outputs/solar-research-v5/`: baseline protocols, component filter and earlier comparisons.
- `outputs/solar-thin-v6/`, `outputs/solar-thin-v7/`: unsuccessful alternatives and diagnostic checks.
- `outputs/solar-ridge-v8/`: full-resolution ridge protocol, training metadata and original assessment.
- `outputs/solar-ridge-validation-v9/`: fixed additional-date protocol, complete rows, mask hashes and combined results.

Some historical scripts expect omitted data/weights or older optional artifacts; they are research records, not the supported root CLI. Their individual historical instructions remain alongside them. A fresh evaluation must retain the distinction between already inspected development data and a genuinely new independent test.
