"""CMU WebKB dataset adapter.

Binary task: university student pages = 1, all other page categories = 0.

Data preparation: run scripts/prepare_webkb.py once before using this adapter.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from data_adapter import AdapterRegistry, DataAdapter, Trainer
from fine_tune import BertFineTuner


class WebKBAdapter(DataAdapter):
    name = "webkb"

    def build_trainer(self, validation_df: pd.DataFrame) -> Trainer:
        model_name = str(self.config.get("trainer", {}).get("model_name", "bert-base-uncased"))
        text_col = str(self.config.get("trainer", {}).get("text_col", self.get_input_text_col()))
        label_col = str(self.config.get("trainer", {}).get("label_col", self.get_target_col()))

        paths = self.get_paths()
        return BertFineTuner(
            model_name=model_name,
            training_data=None,
            test_data=validation_df,
            text_col=text_col,
            label_col=label_col,
            output_dir=str(paths.results_dir),
            logging_dir=str(paths.logs_dir),
        )


AdapterRegistry.register(WebKBAdapter.name, WebKBAdapter)
