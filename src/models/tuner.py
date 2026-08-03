"""Hyperparameter tuning utilities."""

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


class ModelTuner:
    """Tune selected classification models using cross-validation."""

    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
    ) -> None:
        """
        Initialize the model tuner.

        Args:
            random_state: Seed used for reproducibility.
            cv_folds: Number of cross-validation folds.
            scoring: Metric optimized during tuning.
            n_jobs: Number of parallel processing jobs.
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs

    def _get_search_spaces(
        self,
    ) -> dict[
        str,
        tuple[
            ClassifierMixin,
            dict[str, list[Any]],
        ],
    ]:
        """
        Return estimators and their hyperparameter search spaces.

        Returns:
            Mapping of model names to estimators and parameter grids.
        """
        return {
            "logistic_regression": (
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=self.random_state,
                ),
                {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"],
                },
            ),
            "xgboost": (
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                ),
                {
                    "n_estimators": [150, 300],
                    "learning_rate": [0.03, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
            ),
        }

    def tune(
        self,
        X_train: Any,
        y_train: Any,
    ) -> tuple[
        dict[str, ClassifierMixin],
        dict[str, dict[str, Any]],
    ]:
        """
        Tune the selected models using grid-search cross-validation.

        Args:
            X_train: Transformed training features.
            y_train: Training target values.

        Returns:
            A tuple containing:
            - best fitted estimator for each model;
            - best cross-validation score and parameters for each model.
        """
        best_models: dict[str, ClassifierMixin] = {}
        tuning_results: dict[str, dict[str, Any]] = {}

        search_spaces = self._get_search_spaces()

        for model_name, (
            estimator,
            parameter_grid,
        ) in search_spaces.items():
            print(f"Tuning {model_name}...")

            search = GridSearchCV(
                estimator=estimator,
                param_grid=parameter_grid,
                scoring=self.scoring,
                cv=self.cv_folds,
                n_jobs=self.n_jobs,
                refit=True,
                verbose=1,
            )

            search.fit(X_train, y_train)

            best_models[model_name] = search.best_estimator_

            tuning_results[model_name] = {
                "best_score": float(search.best_score_),
                "best_params": search.best_params_,
            }

        return best_models, tuning_results