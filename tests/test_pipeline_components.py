"""Unit tests for core data-pipeline components."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import DataLoader, DataSplitter, DataValidator
from src.features import FeatureEngineer
from src.preprocessing import DataCleaner


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a small employee dataset for testing."""
    return pd.DataFrame(
        {
            "Age": [25, 35, 45, 29],
            "Attrition": ["Yes", "No", "No", "Yes"],
            "YearsAtCompany": [1, 10, 15, 3],
            "TotalWorkingYears": [2, 12, 20, 5],
            "YearsInCurrentRole": [1, 5, 10, 2],
            "YearsWithCurrManager": [1, 4, 8, 2],
            "YearsSinceLastPromotion": [0, 2, 5, 1],
            "MonthlyIncome": [3000, 6000, 9000, 4000],
            "JobLevel": [1, 2, 4, 1],
            "EnvironmentSatisfaction": [2, 4, 3, 1],
            "JobSatisfaction": [2, 3, 4, 2],
            "RelationshipSatisfaction": [3, 4, 3, 2],
            "WorkLifeBalance": [2, 3, 4, 2],
            "EmployeeCount": [1, 1, 1, 1],
            "Over18": ["Y", "Y", "Y", "Y"],
            "StandardHours": [80, 80, 80, 80],
            "EmployeeNumber": [101, 102, 103, 104],
        }
    )


def test_data_loader_loads_csv(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """DataLoader should load a valid CSV file."""
    csv_path = tmp_path / "employees.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    loaded_dataframe = DataLoader(csv_path).load()

    assert loaded_dataframe.shape == sample_dataframe.shape
    assert loaded_dataframe.columns.tolist() == sample_dataframe.columns.tolist()


def test_data_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """DataLoader should reject a file that does not exist."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        DataLoader(missing_path).load()


def test_validator_accepts_expected_data(
    sample_dataframe: pd.DataFrame,
) -> None:
    """DataValidator should accept valid target values and columns."""
    validator = DataValidator(
        target_column="Attrition",
        required_columns=["Age", "Attrition"],
        expected_target_values=["Yes", "No"],
        constant_columns=[
            "EmployeeCount",
            "Over18",
            "StandardHours",
        ],
    )

    report = validator.validate(sample_dataframe)

    assert report["validation_passed"] is True
    assert report["row_count"] == 4
    assert report["target_distribution"] == {
        "Yes": 2,
        "No": 2,
    }


def test_cleaner_removes_non_predictive_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    """DataCleaner should remove configured columns and encode target."""
    cleaner = DataCleaner(
        target_column="Attrition",
        constant_columns=[
            "EmployeeCount",
            "Over18",
            "StandardHours",
        ],
        identifier_columns=["EmployeeNumber"],
    )

    cleaned_dataframe = cleaner.clean(sample_dataframe)

    assert "EmployeeCount" not in cleaned_dataframe.columns
    assert "Over18" not in cleaned_dataframe.columns
    assert "StandardHours" not in cleaned_dataframe.columns
    assert "EmployeeNumber" not in cleaned_dataframe.columns
    assert set(cleaned_dataframe["Attrition"].unique()) == {0, 1}


def test_feature_engineer_adds_expected_features(
    sample_dataframe: pd.DataFrame,
) -> None:
    """FeatureEngineer should create all configured engineered features."""
    cleaned_dataframe = DataCleaner(
        target_column="Attrition",
        constant_columns=[
            "EmployeeCount",
            "Over18",
            "StandardHours",
        ],
        identifier_columns=["EmployeeNumber"],
    ).clean(sample_dataframe)

    engineered_dataframe = FeatureEngineer().transform(
        cleaned_dataframe
    )

    expected_features = {
        "CompanyTenureRatio",
        "CurrentRoleTenureRatio",
        "ManagerTenureRatio",
        "PromotionDelayRatio",
        "IncomePerWorkingYear",
        "IncomePerJobLevel",
        "OverallSatisfaction",
        "LowSatisfactionCount",
        "CareerStage",
    }

    assert expected_features.issubset(engineered_dataframe.columns)


def test_data_splitter_preserves_all_rows() -> None:
    """Train, validation, and test partitions should contain every row."""
    dataframe = pd.DataFrame(
        {
            "feature": range(100),
            "Attrition": [0, 1] * 50,
        }
    )

    split = DataSplitter(
        target_column="Attrition",
        test_size=0.20,
        validation_size=0.20,
        random_state=42,
    ).split(dataframe)

    total_rows = (
        len(split.X_train)
        + len(split.X_validation)
        + len(split.X_test)
    )

    assert total_rows == 100
    assert len(split.X_train) == 60
    assert len(split.X_validation) == 20
    assert len(split.X_test) == 20