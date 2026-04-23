"""Wildlife dataset adapter.

This adapter mirrors the current default dataset behavior but routes all schema,
prompt, and artifact/state locations through `DataAdapter`.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from data_adapter import AdapterRegistry, DataAdapter, Trainer
from fine_tune import BertFineTuner


class WildlifeAdapter(DataAdapter):
    """Adapter for the current wildlife-style binary relevance task."""

    name = "wildlife"

    def build_trainer(self, validation_df: pd.DataFrame) -> Trainer:
        """Build a BERT fine-tuner configured by adapter column names."""
        model_name = str(self.config.get("trainer", {}).get("model_name", "bert-base-uncased"))
        text_col = str(self.config.get("trainer", {}).get("text_col", self.get_input_text_col()))
        label_col = str(self.config.get("trainer", {}).get("label_col", self.get_target_col()))

        paths = self.get_paths()
        trainer = BertFineTuner(
            model_name=model_name,
            training_data=None,
            test_data=validation_df,
            text_col=text_col,
            label_col=label_col,
            output_dir=str(paths.results_dir),
            logging_dir=str(paths.logs_dir),
        )
        return trainer


AdapterRegistry.register(WildlifeAdapter.name, WildlifeAdapter)
