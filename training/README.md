# Research code guide

## Reusable components

`passord.py` implements the frozen pass-level selector. `models.py`, `data.py`, `features.py`, `labels.py`, `metrics.py`, and `temporal.py` provide model, preprocessing, feature-cache, label, metric, and sequence helpers.

The frame dataset expects a metadata column named `path`. Distributed manifests instead contain `image_relative_path`; users with authorized image access must resolve those relative paths against their own image root and populate `path`. No images are bundled.

## Historical experiment implementations

Other Python modules retain the original training and analysis implementations. They may write to `outputs/`, fit models, or regenerate derived tables and figures. The raw images, feature cache and weights needed by many of these scripts are not supplied. Read each entry point and its arguments before use.

`recompute_classical_nested.py` fits nested SVM / RF baselines and expects `outputs/baselines/features/frame_features.npz` and `frame_features_meta.csv`. `nested_sota.py` retains historical training and fusion logic; it is not the final frozen PASS-Ord inference entry point. Model-search scripts are retained for provenance, not as recommended deployment commands. Filename terms such as `sota` do not describe the scope of the scientific claims.

For a read-only check of the distributed predictions, run `python verify_retained_metrics.py` from the repository root. Historical logs and preserved checkpoints do not recover missing launch arguments.
