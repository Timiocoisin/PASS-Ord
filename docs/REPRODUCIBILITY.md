# Reproducibility boundaries

## Three distinct activities

1. **Retained-result verification:** `verify_retained_metrics.py` recomputes metrics from the distributed prediction CSV files and compares them to `evidence/recomputed_metrics.json` with absolute tolerance `1e-7`. No training takes place.
2. **Frozen inference reuse:** `training.passord.route_pass_predictions` combines already aligned, pass-aggregated expert predictions. Users provide the expert probabilities and CNN progress estimates.
3. **Image-based training:** historical implementations are retained, but raw images, weights, the full historical environment and separate four-seed launch commands are not distributed. This activity cannot be reproduced end to end from this repository alone.

## Evaluation units

Class labels are zero-based. The main comparison operates at pass level, with CNN predictions averaged over frames and four model seeds. Frame-level progress metrics are a separate evaluation unit and must not be ranked directly against pass-level MAE. Test stages are resampled as clusters when estimating stage-aware intervals; pass observations within a stage are not independent specimens.

## Selection and records

The retained final selector is fixed and applies only to five-class predictions. The three-class and progress paths are the CNN ensemble. Score calibration across experts was not established. Inspect `evidence/checkpoint_records.json`, the paired prediction CSV files, and `evidence/classical_nested/frozen_config.json` for the available records.

Selected logs in `logs/` are historical execution records, not a replacement for missing configuration provenance. Their original experiment identifiers and rounded summaries can differ from the final recomputed metric record; `verify_retained_metrics.py` is the supported numerical consistency check.

## Environment

The evaluation requirements are lightweight: NumPy, pandas and SciPy. Optional model helpers require PyTorch, torchvision, scikit-learn and Pillow. These are component requirements, not a historical lockfile. The retained classical baseline configuration records Python 3.13.5 and scikit-learn 1.6.1 for that baseline run; those versions do not establish the historical CNN environment.

## Generalization

The retained study uses one specimen and an interleaved stage-grouped split. It supports evaluation within that trajectory. It does not establish performance on independent specimens, acquisition setups, or future stages outside the observed range. Historical spatial experiments are exploratory and must not be treated as independent validation of this final selector.
