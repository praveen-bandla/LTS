from typing import Any
import numpy as np
from scipy.stats import beta
import os
import pandas as pd
from pathlib import Path

class ThompsonSampler:
    def __init__(
        self,
        n_bandits,
        alpha=0.5,
        beta=0.5,
        decay=0.99,
        id_col: str = "id",
        cluster_col: str = "label_cluster",
        predicted_col: str = "predicted_label",
        selected_ids_path: str | Path = "selected_ids.txt",
        wins_path: str | Path = "wins.txt",
        losses_path: str | Path = "losses.txt",
    ):
        """Thompson sampling over clusters.

        Adapter refactor changes:
        - state files (selected ids, wins/losses) are configurable
        - dataframe column names are configurable
        """
        self.n_bandits = n_bandits
        self.wins = np.zeros(n_bandits)  # Initialize wins array
        self.losses = np.zeros(n_bandits)  # Initialize losses array
        self.alpha = alpha  # Prior parameter for Beta distribution (successes)
        self.beta = beta   # Prior parameter for Beta distribution (failures)
        self.decay = decay

        self.id_col = id_col
        self.cluster_col = cluster_col
        self.predicted_col = predicted_col
        self.selected_ids_path = Path(selected_ids_path)
        self.wins_path = Path(wins_path)
        self.losses_path = Path(losses_path)

        try:
            self.selected_ids = set(np.loadtxt(self.selected_ids_path, dtype=str))
        except IOError:
            self.selected_ids = set()

        try:
            self.wins = np.loadtxt(self.wins_path)
            self.losses = np.loadtxt(self.losses_path)
        except IOError:
            self.wins = np.zeros(n_bandits)
            self.losses = np.zeros(n_bandits)

    def choose_bandit(self):
        betas = beta(self.wins + self.alpha, self.losses + self.beta)
        sampled_rewards = betas.rvs(size=self.n_bandits)
        return np.argmax(sampled_rewards)

    def update(self, chosen_bandit, reward_difference):
        if reward_difference > 0:
            self.wins[chosen_bandit] += 1
        else:
            self.losses[chosen_bandit] += 1

        self.wins *= self.decay
        self.losses *= self.decay

        self.wins_path.parent.mkdir(parents=True, exist_ok=True)
        self.losses_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(self.wins_path, self.wins)
        np.savetxt(self.losses_path, self.losses)

    # def get_sample_data(self, df, sample_size, filter_label: bool, trainer: Any):
    #     def select_data(df, chosen_bandit, sample_size):
    #         filtered_df = df[df['label_cluster'] == chosen_bandit].sample(min(sample_size, len(df[df['label_cluster'] == chosen_bandit])))
    #         return filtered_df


    #     #remove already used data
    #     df = df[~df['id'].isin(self.selected_ids)]

    #     if filter_label:
    #         if "predicted_label" in df.columns:
    #             pos = df[df["predicted_label"] == 1]
    #             neg = df[df["predicted_label"] == 0]

    #             data = pd.DataFrame()

    #             while data.empty:
    #                 n_sample = sample_size/2
    #                 chosen_bandit = self.choose_bandit()
    #                 print(f"Chosen bandit {chosen_bandit}")
    #                 data = select_data(pos, chosen_bandit, int(n_sample))

    #             neg_data = select_data(neg, chosen_bandit, int(sample_size-len(data)))
    #             data = pd.concat([data, neg_data]).sample(frac=1)
    #         else:
    #             chosen_bandit = self.choose_bandit()
    #             print(f"Chosen bandit {chosen_bandit}")
    #             data = select_data(df, chosen_bandit, sample_size)
    #     else:
    #         chosen_bandit = self.choose_bandit()
    #         print(f"Chosen bandit {chosen_bandit}")
    #         data= select_data(df, chosen_bandit, sample_size)

    #     # Add the IDs of sampled data to the selected_ids set
    #     self.selected_ids.update(data['id'])
    #     with open('selected_ids.txt', 'w') as f:
    #         f.write('\n'.join(self.selected_ids))

    #     return data, chosen_bandit



    def get_sample_data(self, df, sample_size, filter_label: bool, trainer: Any):
        def select_data(df, chosen_bandit, sample_size):
            cluster_df = df[df[self.cluster_col] == chosen_bandit]
            return cluster_df.sample(min(sample_size, len(cluster_df)))


        #remove already used data
        df = df[~df[self.id_col].astype(str).isin(self.selected_ids)]

        data = pd.DataFrame()
        while data.empty:
            chosen_bandit = self.choose_bandit()
            print(f"Chosen bandit {chosen_bandit}")
            bandit_df = df[df[self.cluster_col] == chosen_bandit]
            print(f"length of bendit {len(bandit_df)}")
            if not bandit_df.empty:
                if filter_label:
                    filtering_enabled = False
                    if hasattr(trainer, "is_filtering_enabled"):
                        filtering_enabled = bool(trainer.is_filtering_enabled())
                    elif hasattr(trainer, "get_clf"):
                        filtering_enabled = bool(trainer.get_clf())

                    if filtering_enabled:
                        bandit_df = bandit_df.copy()
                        if hasattr(trainer, "infer"):
                            bandit_df[self.predicted_col] = trainer.infer(bandit_df)
                        else:
                            bandit_df[self.predicted_col] = trainer.get_inference(bandit_df)
                        print("inference results")
                        print(bandit_df[self.predicted_col].value_counts())
                    if self.predicted_col in bandit_df.columns:
                        print("inference results")
                        print(bandit_df[self.predicted_col].value_counts())
                        pos = bandit_df[bandit_df[self.predicted_col] == 1]
                        neg = bandit_df[bandit_df[self.predicted_col] == 0]
                        if pos.empty:
                            print("no positive data available")
                            data=pos
                        else:
                            n_sample = sample_size/2
                            data = select_data(pos, chosen_bandit, int(n_sample))
                            neg_data = select_data(neg, chosen_bandit, int(sample_size-len(data)))
                            data = pd.concat([data, neg_data]).sample(frac=1)
                    else:
                        data = select_data(bandit_df, chosen_bandit, sample_size)
                else:
                    data = select_data(bandit_df, chosen_bandit, sample_size)


        # Add the IDs of sampled data to the selected_ids set
        self.selected_ids.update(data[self.id_col].astype(str))
        self.selected_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.selected_ids_path, 'w') as f:
            f.write('\n'.join(self.selected_ids))

        return data, chosen_bandit
