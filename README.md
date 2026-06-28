# HeartFailure_DL

Educational tabular heart-disease classification demo built with pandas and scikit-learn.

The repository name is historical. The bundled dataset target column is `HeartDisease`, and this project does not claim that it specifically predicts heart failure or any other single diagnosis.

## Educational And Medical Disclaimer

This repository is for educational purposes only.

- It is not medical advice.
- It is not a diagnostic tool.
- It must not be used for treatment, triage, screening, or clinical decision-making.
- Any model output in this repository is an educational machine-learning signal, not a health conclusion.

## Dataset Status

The repository currently includes `input/heart.csv`, but the exact upstream source and redistribution license could not be verified from repository contents alone.

- Do not assume the bundled dataset is safe to redistribute publicly.
- Do not assume the dataset label is clinically validated for public medical use.
- See [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md) before publishing, redistributing, or reusing the dataset.

## Current Modeling Approach

The current implementation uses a scikit-learn logistic-regression baseline with a train-only preprocessing pipeline:

- `RestingBP` and `Cholesterol` values equal to `0` are treated as documented anomalies or possible missing values.
- Those zero values are converted to missing values inside the training pipeline.
- Missing numeric values are median-imputed inside the pipeline.
- Nominal categories are one-hot encoded with `handle_unknown="ignore"`.
- Numerical evaluation uses stratified train/test splitting and cross-validation without fitting preprocessing on the full dataset first.

This is a conservative educational baseline intended to avoid evaluation leakage and fragile runtime behavior. It does not establish clinical validity.

## Known Limitations

- Dataset source and license are still unresolved and remain documented limitations for public release.
- The bundled data contains physiologically implausible zeros such as `Cholesterol=0` and `RestingBP=0`.
- The preprocessing policy for those zeros is an educational assumption, not a medically validated correction.
- Metrics printed by the script are for local educational inspection only.
- The project has not been clinically reviewed or validated.

## Setup

Create the virtual environment and install dependencies with `uv`:

```bash
uv venv
uv sync
```

If `uv` cannot access its global cache on your machine, use a repo-local cache instead:

```bash
uv sync --cache-dir .uv-cache
```

## Run Checks

```bash
uv run python -m compileall .
uv run pytest -q
```

If you are using a repo-local `uv` cache, add `--cache-dir .uv-cache` to those commands as well.

## Run The Script

Print evaluation metrics only:

```bash
uv run python heart_failure.py --evaluate-only
```

Run evaluation and then the local interactive demo:

```bash
uv run python heart_failure.py
```

Show CLI help:

```bash
uv run python heart_failure.py --help
```

## Interactive Demo Input Notes

The CLI accepts both dataset-native codes and friendlier labels for categorical inputs. Examples:

- Sex: `M`, `F`, `male`, `female`, `man`, `woman`
- Chest pain type: `TA`, `ATA`, `NAP`, `ASY`, `typical angina`, `atypical angina`, `non-anginal pain`, `asymptomatic`
- Exercise angina: `Y`, `N`, `yes`, `no`, `true`, `false`, `1`, `0`

If an input is invalid, the script prints a clear validation message instead of crashing.

## Public Release Status

This repository is technically ready for educational portfolio use after the owner-approved history privacy cleanup.

- Dataset provenance and redistribution status remain unverified and are documented as a known limitation.
- The repository does not claim dataset license verification.
- The repository does not claim clinical validity.
- See [PUBLIC_RELEASE_BLOCKERS.md](PUBLIC_RELEASE_BLOCKERS.md) and [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md) for the current status notes.

## License

The source code in this repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

That code license should not be assumed to cover the bundled dataset or any future external data assets unless their provenance and license are documented separately.
