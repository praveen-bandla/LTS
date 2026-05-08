"""Random sampling baseline for cluster-based active learning.

Draws an equal number of samples from each cluster each iteration.
Used as a comparison baseline against ThompsonSampler. State (selected IDs)
is persisted to disk to prevent re-labeling the same records.
"""

from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path

class RandomSampler:
    """Uniform random sampler that draws evenly from each LDA cluster.

    Serves as the baseline comparison for ThompsonSampler. Sampling is equal
    across all clusters regardless of past labeling outcomes.
    """
    def __init__(
        self,
        n_bandits,
        id_col: str = "id",
        cluster_col: str = "label_cluster",
        predicted_col: str = "predicted_label",
        selected_ids_path: str | Path = "selected_ids.txt",
    ):
        """Random sampler across clusters.

        Adapter refactor changes:
        - selected ids state file is configurable
        - dataframe column names are configurable
        """
        self.n_bandits = n_bandits
        self.id_col = id_col
        self.cluster_col = cluster_col
        self.predicted_col = predicted_col
        self.selected_ids_path = Path(selected_ids_path)

        try:
            self.selected_ids = set(np.loadtxt(self.selected_ids_path, dtype=str))
        except IOError:
            self.selected_ids = set()

    def get_sample_data(self, df, sample_size, filter_label: bool, trainer: Any):
        """Sample evenly from each cluster, optionally filtering by predicted label.

        Excludes already-seen IDs and persists the updated selected set after each call.
        """
        def get_sample(data, size):
            if data.empty:
                return pd.DataFrame()
            else:
                return data.sample(min(size, len(data)), random_state=42)


        unique_clusters = df[self.cluster_col].unique()

        samples_per_cluster = int(sample_size / self.n_bandits)

        sampled_data = []

        df = df[~df[self.id_col].astype(str).isin(self.selected_ids)]
        if filter_label:
            filtering_enabled = False
            if hasattr(trainer, "is_filtering_enabled"):
                filtering_enabled = bool(trainer.is_filtering_enabled())
            elif hasattr(trainer, "get_clf"):
                filtering_enabled = bool(trainer.get_clf())

            if filtering_enabled:
                df = df.copy()
                if hasattr(trainer, "infer"):
                    df[self.predicted_col] = trainer.infer(df)
                else:
                    df[self.predicted_col] = trainer.get_inference(df)

        # Sample data from each cluster
        for cluster in unique_clusters:
            cluster_data = df[df[self.cluster_col] == cluster]

            if filter_label:
                if self.predicted_col in cluster_data.columns:
                    pos = cluster_data[cluster_data[self.predicted_col] == 1]
                    neg = cluster_data[cluster_data[self.predicted_col] == 0]
                    n_sample = int(samples_per_cluster/2)

                    pos_cluster_data = get_sample(pos, n_sample)
                    neg_cluster_data = get_sample(neg, samples_per_cluster-len(pos_cluster_data))

                    sampled_data.append(pd.concat([pos_cluster_data, neg_cluster_data]).sample(frac=1))
                else:
                    sampled_data.append(get_sample(cluster_data, size=samples_per_cluster))
            else:
                sampled_data.append(get_sample(cluster_data, size=samples_per_cluster))


        sampled_data = pd.concat(sampled_data, ignore_index=True)

        # Add the IDs of sampled data to the selected_ids set
        self.selected_ids.update(sampled_data[self.id_col].astype(str))
        self.selected_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.selected_ids_path, 'w') as f:
            f.write('\n'.join(self.selected_ids))

        return sampled_data, "random"




