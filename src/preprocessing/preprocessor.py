"""Model preprocessing utilities."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelPreprocessor:
    """Build reusable preprocessing for numeric and categorical features."""

    def __init__(self) -> None:
        """Initialize the model preprocessor."""
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.transformer: ColumnTransformer | None = None

    def identify_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """
        Identify numeric and categorical feature columns.

        Args:
            dataframe: Feature dataframe without the target column.

        Returns:
            Numeric and categorical column-name lists.
        """
        self.numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        self.categorical_columns = dataframe.select_dtypes(
            exclude="number"
        ).columns.tolist()

        if not self.numeric_columns and not self.categorical_columns:
            raise ValueError("No usable feature columns were found.")

        return self.numeric_columns, self.categorical_columns

    def build(
        self,
        dataframe: pd.DataFrame,
    ) -> ColumnTransformer:
        """
        Build the preprocessing transformer.

        Numeric features:
        - median imputation;
        - standard scaling.

        Categorical features:
        - most-frequent imputation;
        - one-hot encoding.
        """
        numeric_columns, categorical_columns = self.identify_columns(
            dataframe
        )

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "one_hot_encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        self.transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

        return self.transformer

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ):
        """Fit preprocessing using training data and transform it."""
        transformer = self.build(dataframe)
        return transformer.fit_transform(dataframe)

    def transform(
        self,
        dataframe: pd.DataFrame,
    ):
        """Transform new data using previously fitted preprocessing."""
        if self.transformer is None:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit_transform() first."
            )

        return self.transformer.transform(dataframe)

    def get_feature_names(self) -> list[str]:
        """Return transformed output feature names."""
        if self.transformer is None:
            raise RuntimeError(
                "Preprocessor has not been built or fitted."
            )

        return self.transformer.get_feature_names_out().tolist()