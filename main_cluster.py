import argparse
import pandas as pd
import numpy as np
from labeling import Labeling
from random_sampling import RandomSampler
from preprocessing import TextPreprocessor
from fine_tune import BertFineTuner
from thompson_sampling import ThompsonSampler
import nltk
import json
nltk.download('punkt')

import os
from LDA import LDATopicModel
from config import RUN_CONFIG

import adapters  # registers built-in adapters
from data_adapter import AdapterRegistry


def _truncate_string(text: str, max_length: int = 2000) -> str:
    """Truncate long strings to keep prompts within reasonable limits."""
    return text[:max_length] + "..." if len(text) > max_length else text

def main():
    parser = argparse.ArgumentParser(prog="Sampling fine-tuning", description='Perform Sampling and fine tune')
    # parser.add_argument('-cluster', type=str, required=False,
    #                     help="Name of cluster type")
    parser.add_argument('-sampling', type=str, required=False,
                        help="Name of sampling method")
    parser.add_argument('-sample_size', type=int, required=False,
                        help="sample size")
    parser.add_argument('-filter_label', type=bool, required=False,
                        help="use model clf results to filter data")
    parser.add_argument('-balance', type=bool, required=False,
                        help="balance positive and neg sample")
    parser.add_argument('-model_finetune', type=str, required=False,
                        help="model base for fine tune")
    parser.add_argument('-labeling', type=str, required=False,
                        help="Model to be used for labeling or file if label already on file")
    parser.add_argument('-baseline', type=float, required=False,
                        help="The initial baseline metric")
    parser.add_argument('-filename', type=str, required=False,
                        help="The initial file to be used")
    parser.add_argument('-model', type=str, required=False,
                        help="The type of model to be finetune")
    parser.add_argument('-metric', type=str, required=False,
                        help="The type of metric to be used for baseline")
    parser.add_argument('-val_path', type=str, required=False,
                        help="path to validation")
    parser.add_argument('-cluster_size', type=str, required=False,
                        help="path to validation")

    # Adapter-driven mode
    parser.add_argument('-dataset', type=str, required=False,
                        help="Adapter key (e.g., wildlife)")
    parser.add_argument('-config', type=str, required=False,
                        help="Path to adapter YAML (defaults to adapters/configs/<dataset>.yaml)")


    args = parser.parse_args()

    # cluster = args.cluster
    sampling = args.sampling if args.sampling is not None else RUN_CONFIG.sampling
    sample_size = args.sample_size if args.sample_size is not None else RUN_CONFIG.sample_size
    filter_label = args.filter_label if args.filter_label is not None else RUN_CONFIG.filter_label
    balance = args.balance if args.balance is not None else RUN_CONFIG.balance
    model_finetune = args.model_finetune if args.model_finetune is not None else RUN_CONFIG.model_finetune
    labeling = args.labeling if args.labeling is not None else RUN_CONFIG.labeling
    baseline = args.baseline if args.baseline is not None else RUN_CONFIG.baseline
    filename = args.filename if args.filename is not None else RUN_CONFIG.filename
    model = args.model if args.model is not None else RUN_CONFIG.model
    metric = args.metric if args.metric is not None else RUN_CONFIG.metric
    validation_path = args.val_path if args.val_path is not None else RUN_CONFIG.val_path
    cluster_size = args.cluster_size if args.cluster_size is not None else RUN_CONFIG.cluster_size


    preprocessor = TextPreprocessor()

    # Adapter-driven configuration (preferred)
    adapter = None
    paths = None
    if args.dataset:
        config_path = args.config or f"adapters/configs/{args.dataset}.yaml"
        adapter = AdapterRegistry.create(args.dataset, config_path)
        paths = adapter.get_paths()
        adapter_balance = adapter.config.get("training", {}).get("balance", None)
        if adapter_balance is not None:
            balance = adapter_balance

    if adapter is not None:
        id_col = adapter.get_id_col()
        input_col = adapter.get_input_text_col()
        desc_col = adapter.get_description_col()
        clean_col = adapter.get_clean_text_col()
        cluster_col = adapter.get_cluster_col()
        pred_col = adapter.get_predicted_col()
        target_col = adapter.get_target_col()
        answer_col = adapter.get_answer_col()

        validation = adapter.prepare_validation_df(adapter.load_validation_df())
        pool_df = adapter.prepare_pool_df(adapter.load_pool_df())

        # Ensure standard directories exist under the artifact namespace.
        _ = paths.models_dir
        _ = paths.logs_dir
        _ = paths.results_dir

        try:
            data = pd.read_csv(paths.lda_cache_path)
            n_cluster = int(data[cluster_col].nunique())
            print("using data saved on disk")
        except Exception:
            print("Creating LDA")
            data = preprocessor.preprocess_df(pool_df, text_col=input_col, desc_col=desc_col, output_col=clean_col)
            lda_topic_model = LDATopicModel(num_topics=int(cluster_size))
            topics = lda_topic_model.fit_transform(data[clean_col].to_list())
            data[cluster_col] = topics
            n_cluster = int(data[cluster_col].nunique())
            data.to_csv(paths.lda_cache_path, index=False)
            print("LDA created")
    else:
        # Legacy fallback path (kept for backwards compatibility)
        validation = pd.read_csv(validation_path)
        validation["training_text"] = validation["title"]

        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("log", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        try:
            data = pd.read_csv(filename+"_lda.csv")
            n_cluster = data['label_cluster'].value_counts().count()
            print("using data saved on disk")
        except Exception:
            print("Creating LDA")
            data = pd.read_csv(filename+".csv")
            data = preprocessor.preprocess_df(data)
            lda_topic_model = LDATopicModel(num_topics=int(cluster_size))
            topics = lda_topic_model.fit_transform(data['clean_title'].to_list())
            data["label_cluster"] = topics
            n_cluster = data['label_cluster'].value_counts().count()
            print(n_cluster)
            data.to_csv(filename + "_lda.csv", index=False)
            print("LDA created")


    baseline = baseline

    if model == "text":
        if adapter is not None:
            trainer = adapter.build_trainer(validation)
        else:
            trainer = BertFineTuner(model_finetune, None, validation)
    else:
        raise ValueError("Currently only text model is supported")

    labeler = Labeling(label_model=labeling)
    labeler.set_model()

    if sampling == "thompson":
        if adapter is not None:
            sampler = ThompsonSampler(
                n_cluster,
                id_col=id_col,
                cluster_col=cluster_col,
                predicted_col=pred_col,
                selected_ids_path=paths.selected_ids_path,
                wins_path=paths.wins_path,
                losses_path=paths.losses_path,
            )
        else:
            sampler = ThompsonSampler(n_cluster)
    elif sampling == "random":
        if adapter is not None:
            sampler = RandomSampler(
                n_cluster,
                id_col=id_col,
                cluster_col=cluster_col,
                predicted_col=pred_col,
                selected_ids_path=paths.selected_ids_path,
            )
        else:
            sampler = RandomSampler(n_cluster)
    else:
        raise ValueError("Choose one of thompson or random")

    for i in range(10):
        sample_data, chosen_bandit = sampler.get_sample_data(data, sample_size, filter_label, trainer)
        ## Generate labels
        if labeling != "file":
            if adapter is not None:
                df = sample_data.copy()
                df[answer_col] = df.apply(
                    lambda x: labeler.predict(
                        adapter.format_prompt(_truncate_string(str(x[clean_col]))),
                        record_id=str(x[id_col]),
                    ),
                    axis=1,
                )
                df[answer_col] = df[answer_col].astype(str).str.strip()
                df[target_col] = df[answer_col].apply(adapter.parse_model_output)

                if paths.labeled_data_path.exists():
                    train_data = pd.read_csv(paths.labeled_data_path)
                    train_data = pd.concat([train_data, df])
                    train_data.to_csv(paths.labeled_data_path, index=False)
                else:
                    df.to_csv(paths.labeled_data_path, index=False)
            else:
                df = labeler.generate_inference_data(sample_data, 'clean_title')
                print("df for inference created")
                df["answer"] = df.apply(lambda x: labeler.predict_animal_product(x), axis=1)
                df["answer"] = df["answer"].str.strip()
                from config import LABEL_CONFIG
                df["label"] = np.where(df["answer"] == LABEL_CONFIG.positive_text, LABEL_CONFIG.positive_value, LABEL_CONFIG.negative_value)
                if os.path.exists(f"{filename}_data_labeled.csv"):
                    train_data = pd.read_csv(f"{filename}_data_labeled.csv")
                    train_data = pd.concat([train_data, df])
                    train_data.to_csv(f"{filename}_data_labeled.csv", index=False)
                else:
                    df.to_csv(f"{filename}_data_labeled.csv", index=False)
        else:
            df = sample_data
        label_counts_col = adapter.get_target_col() if adapter is not None else "label"
        print(df[label_counts_col].value_counts())
        # print(df["answer"].value_counts())

        # ADD POSITIVE DATA IF AVAILABLE

        if adapter is not None:
            if paths.positive_data_path.exists():
                pos = pd.read_csv(paths.positive_data_path)
                df = pd.concat([df, pos]).sample(frac=1)
                print(f"adding positive data: {df[target_col].value_counts()}")
        else:
            if os.path.exists('positive_data.csv'):
                pos = pd.read_csv('positive_data.csv')
                df = pd.concat([df, pos]).sample(frac=1)
                print(f"adding positive data: {df['label'].value_counts()}")
        if balance:
            label_counts = df[label_counts_col].value_counts()
            if len(label_counts) > 1:
                unbalanced = label_counts.max() / label_counts.min() > 2
                if unbalanced:
                    min_count = int(label_counts.min())
                    # Cap every class at 2× the minority count; never over-sample
                    balanced_parts = [
                        grp.sample(min(len(grp), min_count * 2), replace=False)
                        for _, grp in df.groupby(label_counts_col)
                    ]
                    df = pd.concat(balanced_parts).sample(frac=1).reset_index(drop=True)
                    print(f"Balanced data: {df[label_counts_col].value_counts()}")
            # else:
                # if i == 0: # if this is the first model training
                # unbalanced = True
                # print("No positive samples to balance with.")
        
        # Run grid search on the first iteration to lock in best hyperparams
        if i == 0:
            print("\n--- Running Hyperparameter Grid Search ---")
            # This uses the fresh labeled 'df'
            best_params = trainer.run_hyperparameter_search(df, metric=metric)
            print(f"Optimal Params Found: {best_params}\n")

        ## FINE TUNE MODEL (Existing logic continues below)
        model_name = trainer.get_base_model()
        
        ## FINE TUNE MODEL

        #previous model
        model_name = trainer.get_base_model()
        print(f"using model {model_name}")
        model_results = trainer.get_last_model_metrics() if hasattr(trainer, "get_last_model_metrics") else trainer.get_last_model_acc()
        if model_results:
            baseline = model_results[model_name]
            print(f"previous model {metric} metric baseline of: {baseline}")
        else:
            print(f"Starting with metric {metric} baseline {baseline}")
        print(f"Starting training")

        try:
            _lc = df[label_counts_col].value_counts()
            still_unbalenced = _lc.max() / _lc.min() >= 2
        except Exception:
            still_unbalenced = True
        print(f"Unbalanced? {still_unbalenced}")

        if hasattr(trainer, "train"):
            results = trainer.train(df, still_unbalenced)
        else:
            results, huggingface_trainer = trainer.train_data(df, still_unbalenced)
        reward_difference = results[f"eval_{metric}"] - baseline
        if reward_difference > 0:
            print(f"Model improved with {reward_difference}")
            if adapter is not None:
                model_name = str(paths.models_dir / f"fine_tunned_{i}_bandit_{chosen_bandit}")
            else:
                model_name = f"models/fine_tunned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{metric}"], save_model=True)
            # df.to_csv("llama_training_data.csv", index=False)
            if adapter is not None:
                if paths.training_data_path.exists():
                    train_data = pd.read_csv(paths.training_data_path)
                    df = pd.concat([train_data, df])
                df.to_csv(paths.training_data_path, index=False)
                if paths.positive_data_path.exists():
                    os.remove(paths.positive_data_path)
            else:
                if os.path.exists(f'{filename}_training_data.csv'):
                    train_data = pd.read_csv(f'{filename}_training_data.csv')
                    df = pd.concat([train_data, df])
                df.to_csv(f'{filename}_training_data.csv', index=False)
                if os.path.exists('positive_data.csv'):
                    os.remove('positive_data.csv')
            if filter_label:
                if hasattr(trainer, "enable_filtering"):
                    trainer.enable_filtering(True)
                else:
                    trainer.set_clf(True)
                # data["predicted_label"] = trainer.get_inference(data)
                # print(data["predicted_label"].value_counts())
                # if data[data["predicted_label"]==1].empty:
                #     data["predicted_label"] = 1
                # data.to_csv("data_w_predictions.csv", index=False)
            ## save model results
        else:
            #back to initial model
            trainer.update_model(model_name, baseline, save_model=False)
            # save positive sample
            if adapter is not None:
                # Save all minority-class rows so rare classes accumulate across rounds.
                _lc = df[label_counts_col].value_counts()
                _majority = _lc.idxmax()
                df_pos = df[df[label_counts_col] != _majority]
                if paths.positive_data_path.exists():
                    positive = pd.read_csv(paths.positive_data_path)
                    df_pos = pd.concat([df_pos, positive]).drop_duplicates()
                df_pos.to_csv(paths.positive_data_path, index=False)
            else:
                if os.path.exists('positive_data.csv'):
                    positive = pd.read_csv("positive_data.csv")
                    df = df[df["label"]==1]
                    df = pd.concat([df, positive])
                    df = df.drop_duplicates()
                df[df["label"]==1].to_csv("positive_data.csv", index=False)


        results_path = paths.model_results_path if adapter is not None else f"{filename}_model_results.json"
        if os.path.exists(results_path):
            with open(results_path, 'r') as file:
                existing_results = json.load(file)
        else:
            existing_results = {}

        if existing_results.get(str(chosen_bandit)):
            existing_results[str(chosen_bandit)].append(results)
        else:
            existing_results[str(chosen_bandit)] = [results]

        # Write the updated list to the file
        with open(results_path, 'w') as file:
            json.dump(existing_results, file, indent=4)
        if sampling == "thompson":
            sampler.update(chosen_bandit, reward_difference)


    if sampling == "thompson":
        print("Bendt with highest expected improvement:", np.argmax(sampler.wins / (sampler.wins + sampler.losses)))
        print(sampler.wins)
        print(sampler.losses)
    # Save the DataFrame with cluster labels
    # umap_df.to_csv("./data/gpt_training_with_clusters.csv", index=False)




if __name__ == "__main__":
    main()
