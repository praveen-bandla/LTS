"""Adapter layer for dataset/task-specific configuration.

The goal of this module is to keep the core pipeline (sampling, labeling, clustering,
training) free of hardcoded dataset schemas, prompt semantics, and run-state file
locations. Adding a new dataset should be: new adapter subclass + YAML + prompt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Type

import pandas as pd
import yaml


class Trainer(Protocol):
    """Minimal training/inference interface expected by the core pipeline.

    This intentionally mirrors the few operations the pipeline needs:
    - train on a labeled batch
    - optionally infer labels for filtering
    - enable/disable filtering behavior
    - update and persist the model
    """

    def train(self, df_train: pd.DataFrame, still_unbalanced: bool) -> Dict[str, Any]:
        """Train on a labeled batch and return evaluation metrics."""

    def infer(self, df_unlabeled: pd.DataFrame) -> Any:
        """Infer labels or scores for unlabeled records."""

    def save(self, path: str) -> None:
        """Persist the current model artifacts to `path`."""

    def update_model(self, model_name: str, model_metric: float, save_model: bool) -> None:
        """Update the active model reference (and optionally save it)."""

    def enable_filtering(self, enabled: bool) -> None:
        """Enable or disable filtering in samplers."""

    def is_filtering_enabled(self) -> bool:
        """Return whether filtering is enabled."""

    def get_base_model(self) -> str:
        """Return the current model name."""

    def get_last_model_metrics(self) -> Optional[Dict[str, float]]:
        """Return a dict of last-evaluated metrics for the current model."""


@dataclass(frozen=True)
class AdapterPaths:
    """Convenience container for adapter-derived artifact/state paths."""

    artifact_dir: Path

    @property
    def models_dir(self) -> Path:
        return self.artifact_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.artifact_dir / "logs"

    @property
    def results_dir(self) -> Path:
        return self.artifact_dir / "results"

    @property
    def selected_ids_path(self) -> Path:
        return self.artifact_dir / "selected_ids.txt"

    @property
    def wins_path(self) -> Path:
        return self.artifact_dir / "wins.txt"

    @property
    def losses_path(self) -> Path:
        return self.artifact_dir / "losses.txt"

    @property
    def positive_data_path(self) -> Path:
        return self.artifact_dir / "positive_data.csv"

    @property
    def lda_cache_path(self) -> Path:
        return self.artifact_dir / "lda_cache.csv"

    @property
    def labeled_data_path(self) -> Path:
        return self.artifact_dir / "data_labeled.csv"

    @property
    def training_data_path(self) -> Path:
        return self.artifact_dir / "training_data.csv"

    @property
    def model_results_path(self) -> Path:
        return self.artifact_dir / "model_results.json"


class DataAdapter(ABC):
    """Base class for dataset/task adapters.

    Subclasses implement schema getters, prompt formatting/parsing, and trainer
    construction. The core pipeline calls into this contract instead of reading
    hardcoded columns/paths.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load_yaml(self.config_path)
        self._prompt_template_cache: Optional[str] = None

    def _resolve_path(self, raw: str | Path) -> Path:
        """Resolve a path from YAML.

        Relative paths are interpreted relative to the YAML file location.
        """
        path = Path(raw)
        if path.is_absolute():
            return path
        return (self.config_path.parent / path).resolve()

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Adapter config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Adapter YAML must be a mapping")
        return data

    # ----------------------------
    # Identity / artifacts
    # ----------------------------

    def get_run_name(self) -> str:
        """Return a stable run name used to namespace artifacts/state."""
        return str(self.config.get("run_name") or self.name)

    # Subclasses must override this.
    name: str

    def get_artifact_dir(self) -> Path:
        """Return the directory where this run writes state and outputs."""
        base = Path(self.config.get("artifact_root", "artifacts"))
        artifact_dir = base / self.get_run_name()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def get_paths(self) -> AdapterPaths:
        """Return namespaced artifact/state paths for this adapter."""
        paths = AdapterPaths(artifact_dir=self.get_artifact_dir())
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.results_dir.mkdir(parents=True, exist_ok=True)
        return paths

    # ----------------------------
    # I/O
    # ----------------------------

    def load_pool_df(self) -> pd.DataFrame:
        """Load the unlabeled pool dataframe."""
        path = self._resolve_path(self.config["data"]["pool_csv"])  # required
        return pd.read_csv(path)

    def load_validation_df(self) -> pd.DataFrame:
        """Load the validation dataframe."""
        path = self._resolve_path(self.config["data"]["validation_csv"])  # required
        return pd.read_csv(path)

    # ----------------------------
    # Column getters
    # ----------------------------

    def get_id_col(self) -> str:
        return str(self.config.get("columns", {}).get("id", "id"))

    def get_input_text_col(self) -> str:
        return str(self.config.get("columns", {}).get("input_text", "title"))

    def get_description_col(self) -> Optional[str]:
        return self.config.get("columns", {}).get("description")

    def get_target_col(self) -> str:
        return str(self.config.get("columns", {}).get("target", "label"))

    def get_clean_text_col(self) -> str:
        return str(self.config.get("columns", {}).get("clean_text", "clean_title"))

    def get_cluster_col(self) -> str:
        return str(self.config.get("columns", {}).get("cluster", "label_cluster"))

    def get_predicted_col(self) -> str:
        return str(self.config.get("columns", {}).get("predicted", "predicted_label"))

    def get_answer_col(self) -> str:
        return str(self.config.get("columns", {}).get("answer", "answer"))

    def get_prompt_col(self) -> str:
        return str(self.config.get("columns", {}).get("prompt", "prompt"))

    # ----------------------------
    # Prepare hooks
    # ----------------------------

    def prepare_pool_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize pool df (e.g., ensure id exists)."""
        id_col = self.get_id_col()
        if id_col not in df.columns:
            df = df.copy()
            df[id_col] = df.index.astype(str)
        return df

    def prepare_validation_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize validation df (e.g., ensure id exists)."""
        id_col = self.get_id_col()
        if id_col not in df.columns:
            df = df.copy()
            df[id_col] = df.index.astype(str)
        return df

    # ----------------------------
    # Task hooks
    # ----------------------------

    def _load_prompt_template(self) -> str:
        prompt_path = self._resolve_path(self.config["prompt"]["template_path"])  # required
        template = prompt_path.read_text(encoding="utf-8")
        return template

    def format_prompt(self, text: str) -> str:
        """Format a prompt string for the labeling model."""
        if self._prompt_template_cache is None:
            self._prompt_template_cache = self._load_prompt_template()
        template = self._prompt_template_cache
        # Accept either {title} or {text} placeholders.
        try:
            return template.format(title=text, text=text)
        except Exception:
            return f"{template}\n{text}"

    def parse_model_output(self, output: str) -> int:
        """Map a raw model output into a 0/1 label.

        Default behavior uses adapter config fields:
        - prompt.positive_text (exact match, case-insensitive)
        - prompt.positive_value / negative_value
        """
        positive_text = str(self.config.get("prompt", {}).get("positive_text", "relevant"))
        positive_value = int(self.config.get("prompt", {}).get("positive_value", 1))
        negative_value = int(self.config.get("prompt", {}).get("negative_value", 0))

        normalized = (output or "").strip().lower()
        if not normalized:
            return negative_value
        if normalized == positive_text.lower():
            return positive_value
        if normalized in {"1", "yes", "true", "relevant"}:
            return positive_value
        if normalized in {"0", "no", "false", "irrelevant"}:
            return negative_value
        return negative_value

    # ----------------------------
    # Training
    # ----------------------------

    @abstractmethod
    def build_trainer(self, validation_df: pd.DataFrame) -> Trainer:
        """Build and return a trainer instance configured for this dataset."""


class AdapterRegistry:
    """Registry for resolving `-dataset` keys to adapter classes."""

    _registry: Dict[str, Type[DataAdapter]] = {}

    @classmethod
    def register(cls, key: str, adapter_cls: Type[DataAdapter]) -> None:
        cls._registry[key] = adapter_cls

    @classmethod
    def create(cls, dataset: str, config_path: str | Path) -> DataAdapter:
        if dataset not in cls._registry:
            raise ValueError(f"Unknown dataset adapter: {dataset}. Registered: {sorted(cls._registry)}")
        return cls._registry[dataset](config_path=config_path)
