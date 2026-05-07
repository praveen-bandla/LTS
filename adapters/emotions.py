"""Emotions dataset adapter.

Multiclass task (6 classes):
    sadness=0  anger=1  fear=2  joy=3  love=4  surprise=5

label_map and num_labels are read from the YAML config.
"""

from __future__ import annotations

import pandas as pd

from data_adapter import AdapterRegistry, DataAdapter, Trainer
from fine_tune import BertFineTuner


class EmotionsAdapter(DataAdapter):
    name = "emotions"

    def parse_model_output(self, output: str) -> int:
        label_map: dict = self.config.get("prompt", {}).get("label_map", {})
        default = int(self.config.get("prompt", {}).get("default_value", 0))
        normalized = (output or "").strip().lower()
        for label, value in label_map.items():
            if str(label).lower() in normalized:
                return int(value)
        return default

    def build_trainer(self, validation_df: pd.DataFrame) -> Trainer:
        model_name = str(self.config.get("trainer", {}).get("model_name", "bert-base-uncased"))
        text_col = str(self.config.get("trainer", {}).get("text_col", self.get_input_text_col()))
        label_col = str(self.config.get("trainer", {}).get("label_col", self.get_target_col()))
        num_labels = int(self.config.get("trainer", {}).get("num_labels", 2))

        paths = self.get_paths()
        return BertFineTuner(
            model_name=model_name,
            training_data=None,
            test_data=validation_df,
            text_col=text_col,
            label_col=label_col,
            output_dir=str(paths.results_dir),
            logging_dir=str(paths.logs_dir),
            num_labels=num_labels,
        )


AdapterRegistry.register(EmotionsAdapter.name, EmotionsAdapter)
