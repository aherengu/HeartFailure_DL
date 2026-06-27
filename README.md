# HeartFailure_DL

Educational heart disease prediction experiment using a tabular dataset, scikit-learn preprocessing, logistic regression, and a small Keras neural network.

## Medical Disclaimer

This project is for educational purposes only. It is not medical advice and must not be used for diagnosis, treatment, triage, or clinical decisions. Model outputs are experimental learning signals, not health conclusions. Consult qualified medical professionals for health concerns.

## Dataset

The repository includes `input/heart.csv`. The original dataset source and dataset license are not documented in this repository and should be verified before reuse, redistribution, or publication.

## Setup

Create an environment and install the Python dependencies:

```bash
uv venv
uv pip install numpy pandas scikit-learn scikeras keras tensorflow
```

Run the script:

```bash
.venv\Scripts\python heart_failure.py
```

On macOS or Linux, use `.venv/bin/python heart_failure.py`.

The script trains models, prints cross-validation accuracy statistics, and then asks for example input values for an educational prediction demo.

## License

The source code is licensed under the Apache License 2.0. See `LICENSE`.