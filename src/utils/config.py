"""
Configuration manager for the Employee Attrition Pipeline.

Loads the YAML configuration once and provides easy access
to configuration values throughout the project.
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Singleton-style configuration manager.
    """

    _config: dict[str, Any] | None = None

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> None:
        """
        Load configuration from a YAML file.
        """

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            cls._config = yaml.safe_load(file)

    @classmethod
    def get(cls, *keys: str) -> Any:
        """
        Retrieve nested configuration values.

        Example:
            Config.get("paths", "raw_data")
        """

        if cls._config is None:
            cls.load()

        value: Any = cls._config

        for key in keys:
            value = value[key]

        return value