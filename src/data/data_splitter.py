"""Training, validation, and testing split utilities."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Container for training, validation, and testing datasets."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class DataSplitter:
    """Create stratified training, validation, and testing datasets."""

    def __init__(
        self,
        target_column: str,
        test_size: float = 0.20,
        validation_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the data splitter.

        Args:
            target_column: Name of the target variable.
            test_size: Proportion of the full dataset used for testing.
            validation_size: Proportion used for model selection.
            random_state: Seed used for reproducibility.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if not 0 < validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        if test_size + validation_size >= 1:
            raise ValueError(
                "test_size and validation_size must sum to less than 1."
            )

        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

    def split(self, dataframe: pd.DataFrame) -> DataSplit:
        """
        Create stratified training, validation, and testing splits.

        Returns:
            DataSplit containing all feature and target partitions.
        """
        if self.target_column not in dataframe.columns:
            raise ValueError(
                f"Target column is missing: {self.target_column}"
            )

        X = dataframe.drop(columns=[self.target_column])
        y = dataframe[self.target_column]

        if y.nunique() < 2:
            raise ValueError(
                "Target column must contain at least two classes."
            )

        X_remaining, X_test, y_remaining, y_test = (
            train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
        )

        adjusted_validation_size = (
            self.validation_size / (1 - self.test_size)
        )

        (
            X_train,
            X_validation,
            y_train,
            y_validation,
        ) = train_test_split(
            X_remaining,
            y_remaining,
            test_size=adjusted_validation_size,
            random_state=self.random_state,
            stratify=y_remaining,
        )

        return DataSplit(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )