"""Train-test splitting utilities."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Container for training and testing datasets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class DataSplitter:
    """Separate features and target and create a stratified train-test split."""

    def __init__(
        self,
        target_column: str,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the data splitter.

        Args:
            target_column: Name of the target variable.
            test_size: Proportion of observations assigned to the test set.
            random_state: Seed used to make the split reproducible.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split(self, dataframe: pd.DataFrame) -> DataSplit:
        """
        Split a dataset into stratified training and testing sets.

        Args:
            dataframe: Processed dataset containing features and target.

        Returns:
            DataSplit containing X_train, X_test, y_train, and y_test.
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

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        return DataSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )