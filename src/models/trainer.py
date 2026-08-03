"""Model training utilities."""

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class ModelTrainer:
    """Create and train baseline classification models."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def get_models(self) -> dict[str, ClassifierMixin]:
        """Return configured baseline models."""
        return {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "xgboost": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }

    def train(
        self,
        X_train: Any,
        y_train: Any,
    ) -> dict[str, ClassifierMixin]:
        """Train all configured models."""
        trained_models: dict[str, ClassifierMixin] = {}

        for model_name, model in self.get_models().items():
            model.fit(X_train, y_train)
            trained_models[model_name] = model

        return trained_models