import builtins
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import heart_failure


def test_import_has_no_interactive_side_effects(monkeypatch):
    module_path = REPO_ROOT / "heart_failure.py"
    spec = importlib.util.spec_from_file_location("heart_failure_import_test", module_path)
    module = importlib.util.module_from_spec(spec)

    def fail_on_input(*args, **kwargs):
        raise AssertionError("input() should not be called during import")

    monkeypatch.setattr(builtins, "input", fail_on_input)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert callable(module.main)
    assert callable(module.normalize_user_input)


@pytest.mark.parametrize(
    ("field", "raw_value", "expected"),
    [
        ("Sex", "male", "M"),
        ("Sex", "woman", "F"),
        ("ChestPainType", "typical angina", "TA"),
        ("ChestPainType", "non-anginal pain", "NAP"),
        ("RestingECG", "st", "ST"),
        ("ExerciseAngina", "Yes", "Y"),
        ("ExerciseAngina", "0", "N"),
        ("ST_Slope", "downsloping", "Down"),
    ],
)
def test_normalize_category_value_accepts_friendly_aliases(field, raw_value, expected):
    assert heart_failure.normalize_category_value(field, raw_value) == expected


def test_normalize_user_input_rejects_unknown_category_with_clear_error():
    raw_inputs = {
        "Age": "50",
        "Sex": "robot",
        "ChestPainType": "ATA",
        "RestingBP": "120",
        "Cholesterol": "200",
        "FastingBS": "0",
        "RestingECG": "Normal",
        "MaxHR": "150",
        "ExerciseAngina": "no",
        "Oldpeak": "1.0",
        "ST_Slope": "Flat",
    }

    with pytest.raises(ValueError, match="Invalid value for Sex"):
        heart_failure.normalize_user_input(raw_inputs)


def test_replace_documented_zero_anomalies_only_changes_documented_columns():
    frame = pd.DataFrame(
        {
            "Age": [50.0],
            "RestingBP": [0.0],
            "Cholesterol": [0.0],
            "FastingBS": [0.0],
            "MaxHR": [140.0],
            "Oldpeak": [1.0],
        }
    )

    transformed = heart_failure.replace_documented_zero_anomalies(frame)

    assert np.isnan(transformed.loc[0, "RestingBP"])
    assert np.isnan(transformed.loc[0, "Cholesterol"])
    assert transformed.loc[0, "FastingBS"] == 0.0


def test_pipeline_can_fit_and_predict_on_small_dataset():
    dataset = pd.DataFrame(
        [
            {
                "Age": 40,
                "Sex": "M",
                "ChestPainType": "ATA",
                "RestingBP": 140,
                "Cholesterol": 289,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 172,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Up",
                "HeartDisease": 0,
            },
            {
                "Age": 49,
                "Sex": "F",
                "ChestPainType": "NAP",
                "RestingBP": 160,
                "Cholesterol": 180,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 156,
                "ExerciseAngina": "N",
                "Oldpeak": 1.0,
                "ST_Slope": "Flat",
                "HeartDisease": 1,
            },
            {
                "Age": 55,
                "Sex": "M",
                "ChestPainType": "ASY",
                "RestingBP": 0,
                "Cholesterol": 0,
                "FastingBS": 1,
                "RestingECG": "ST",
                "MaxHR": 110,
                "ExerciseAngina": "Y",
                "Oldpeak": 1.5,
                "ST_Slope": "Flat",
                "HeartDisease": 1,
            },
            {
                "Age": 45,
                "Sex": "F",
                "ChestPainType": "TA",
                "RestingBP": 130,
                "Cholesterol": 237,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 170,
                "ExerciseAngina": "N",
                "Oldpeak": 0.0,
                "ST_Slope": "Up",
                "HeartDisease": 0,
            },
        ]
    )

    model = heart_failure.build_logistic_pipeline()
    x_values = dataset.drop(columns=heart_failure.TARGET_COLUMN)
    y_values = dataset[heart_failure.TARGET_COLUMN]
    model.fit(x_values, y_values)
    predictions = model.predict(x_values)

    assert len(predictions) == len(dataset)
    assert set(predictions).issubset({0, 1})


def test_split_dataset_uses_stratification():
    dataset = heart_failure.load_dataset()
    _, _, y_train, y_test = heart_failure.split_dataset(dataset)
    overall_positive_rate = dataset[heart_failure.TARGET_COLUMN].mean()

    assert abs(y_train.mean() - overall_positive_rate) < 0.02
    assert abs(y_test.mean() - overall_positive_rate) < 0.02
