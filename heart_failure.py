from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

TARGET_COLUMN = "HeartDisease"
RANDOM_STATE = 32
TEST_SIZE = 0.2

CATEGORICAL_COLUMNS = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
NUMERIC_COLUMNS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]
FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
ZERO_AS_MISSING_COLUMNS = ["RestingBP", "Cholesterol"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "input" / "heart.csv"

CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    "Sex": {
        "m": "M",
        "male": "M",
        "man": "M",
        "f": "F",
        "female": "F",
        "woman": "F",
    },
    "ChestPainType": {
        "ta": "TA",
        "typical": "TA",
        "typical angina": "TA",
        "ata": "ATA",
        "atypical": "ATA",
        "atypical angina": "ATA",
        "nap": "NAP",
        "non anginal": "NAP",
        "non anginal pain": "NAP",
        "nonanginal": "NAP",
        "nonanginal pain": "NAP",
        "asy": "ASY",
        "asymptomatic": "ASY",
    },
    "RestingECG": {
        "normal": "Normal",
        "st": "ST",
        "lvh": "LVH",
    },
    "ExerciseAngina": {
        "y": "Y",
        "yes": "Y",
        "true": "Y",
        "1": "Y",
        "n": "N",
        "no": "N",
        "false": "N",
        "0": "N",
    },
    "ST_Slope": {
        "up": "Up",
        "upsloping": "Up",
        "flat": "Flat",
        "down": "Down",
        "downsloping": "Down",
    },
}

INPUT_PROMPTS = {
    "Age": "Age: ",
    "Sex": "Sex (M/F, male/female): ",
    "ChestPainType": "Chest pain type (TA/ATA/NAP/ASY or friendly label): ",
    "RestingBP": "Resting blood pressure: ",
    "Cholesterol": "Cholesterol: ",
    "FastingBS": "Fasting blood sugar > 120 mg/dl (0/1 or no/yes): ",
    "RestingECG": "Resting ECG (Normal/ST/LVH): ",
    "MaxHR": "Maximum heart rate achieved: ",
    "ExerciseAngina": "Exercise-induced angina (Y/N or yes/no): ",
    "Oldpeak": "Oldpeak: ",
    "ST_Slope": "ST slope (Up/Flat/Down): ",
}


def load_dataset(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )
    return dataset[REQUIRED_COLUMNS].copy()


def replace_documented_zero_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = frame.copy()
    for column in ZERO_AS_MISSING_COLUMNS:
        transformed[column] = transformed[column].replace(0, np.nan)
    return transformed


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "zero_to_missing",
                FunctionTransformer(
                    replace_documented_zero_anomalies,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def split_dataset(
    dataset: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_values = dataset.drop(columns=TARGET_COLUMN)
    y_values = dataset[TARGET_COLUMN]
    return train_test_split(
        x_values,
        y_values,
        test_size=test_size,
        random_state=random_state,
        stratify=y_values,
    )


def evaluate_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_splits: int = 5,
) -> dict[str, Any]:
    cross_validator = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cross_validator,
        scoring="accuracy",
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "cv_scores": cv_scores,
        "holdout_accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, digits=3),
        "predictions": predictions,
        "probabilities": probabilities,
        "model": model,
    }


def canonicalize_category_value(value: str) -> str:
    text = value.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def normalize_category_value(field: str, raw_value: Any) -> str:
    aliases = CATEGORY_ALIASES[field]
    normalized_key = canonicalize_category_value(str(raw_value))
    if normalized_key in aliases:
        return aliases[normalized_key]

    allowed_values = sorted(set(aliases.values()))
    raise ValueError(
        f"Invalid value for {field}: {raw_value!r}. "
        f"Accepted values map to: {', '.join(allowed_values)}."
    )


def parse_float_value(field: str, raw_value: Any) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {raw_value!r}.") from exc


def parse_binary_indicator(field: str, raw_value: Any) -> int:
    normalized_key = canonicalize_category_value(str(raw_value))
    if normalized_key in {"1", "yes", "y", "true"}:
        return 1
    if normalized_key in {"0", "no", "n", "false"}:
        return 0
    raise ValueError(
        f"Invalid value for {field}: {raw_value!r}. Accepted values are 0/1 or no/yes."
    )


def normalize_user_input(raw_inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "Age": parse_float_value("Age", raw_inputs["Age"]),
        "Sex": normalize_category_value("Sex", raw_inputs["Sex"]),
        "ChestPainType": normalize_category_value(
            "ChestPainType", raw_inputs["ChestPainType"]
        ),
        "RestingBP": parse_float_value("RestingBP", raw_inputs["RestingBP"]),
        "Cholesterol": parse_float_value("Cholesterol", raw_inputs["Cholesterol"]),
        "FastingBS": parse_binary_indicator("FastingBS", raw_inputs["FastingBS"]),
        "RestingECG": normalize_category_value("RestingECG", raw_inputs["RestingECG"]),
        "MaxHR": parse_float_value("MaxHR", raw_inputs["MaxHR"]),
        "ExerciseAngina": normalize_category_value(
            "ExerciseAngina", raw_inputs["ExerciseAngina"]
        ),
        "Oldpeak": parse_float_value("Oldpeak", raw_inputs["Oldpeak"]),
        "ST_Slope": normalize_category_value("ST_Slope", raw_inputs["ST_Slope"]),
    }


def prompt_for_raw_user_input() -> dict[str, str]:
    print(
        "\nEducational demo only. This project is not medical advice and must not be "
        "used for diagnosis, treatment, triage, or clinical decisions."
    )
    print("Enter example values for the dataset fields below.")
    return {field: input(prompt) for field, prompt in INPUT_PROMPTS.items()}


def predict_user_input(model: Pipeline, raw_inputs: Mapping[str, Any]) -> dict[str, Any]:
    normalized_input = normalize_user_input(raw_inputs)
    input_frame = pd.DataFrame([normalized_input], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(input_frame)[0, 1])
    predicted_label = int(model.predict(input_frame)[0])
    return {
        "normalized_input": normalized_input,
        "predicted_label": predicted_label,
        "positive_class_probability": probability,
    }


def print_evaluation_summary(results: Mapping[str, Any]) -> None:
    print("Educational evaluation summary")
    print(
        f"- Cross-validation accuracy: mean={np.mean(results['cv_scores']):.3f}, "
        f"std={np.std(results['cv_scores']):.3f}"
    )
    print(f"- Holdout accuracy: {results['holdout_accuracy']:.3f}")
    print("- Holdout classification report:")
    print(results["classification_report"])
    print(
        "These metrics describe this repository's educational experiment only. "
        "They do not establish clinical validity or safety."
    )


def run_interactive_demo(model: Pipeline) -> int:
    try:
        prediction = predict_user_input(model, prompt_for_raw_user_input())
    except ValueError as exc:
        print(f"Input validation error: {exc}")
        return 1
    except EOFError:
        print("Interactive input was interrupted. Re-run with --evaluate-only to skip the demo.")
        return 1

    print("\nEducational prediction summary")
    print(f"- Predicted dataset label: {prediction['predicted_label']}")
    print(
        "- Estimated probability for the positive dataset label: "
        f"{prediction['positive_class_probability']:.3f}"
    )
    print(
        "This output is an educational model signal only and must not be interpreted "
        "as a diagnosis."
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an educational heart-disease classification baseline, print "
            "cross-validation and holdout metrics, and optionally launch a local "
            "interactive prediction demo."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the CSV dataset. Defaults to input/heart.csv.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip the interactive demo after printing evaluation metrics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(args.dataset)
    x_train, x_test, y_train, y_test = split_dataset(dataset)

    model = build_logistic_pipeline()
    evaluation_results = evaluate_model(model, x_train, x_test, y_train, y_test)
    print_evaluation_summary(evaluation_results)

    if args.evaluate_only:
        return 0

    if not sys.stdin.isatty():
        print("No interactive terminal detected. Re-run with --evaluate-only for metrics only.")
        return 0

    return run_interactive_demo(evaluation_results["model"])


if __name__ == "__main__":
    raise SystemExit(main())
