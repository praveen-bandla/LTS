"""BERT fine-tuning module for the LTS active learning pipeline.

Provides BertFineTuner, a column-configurable wrapper around HuggingFace Trainer
that supports binary and multiclass sequence classification. Also defines MyTrainer
(class-weighted loss for imbalanced datasets) and EarlyStoppingCallback.
"""

from typing import Any, Optional, Dict, Union, Tuple, Callable, List
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification, PreTrainedModel, PreTrainedTokenizerBase, DataCollatorWithPadding, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import IterableDataset
import pandas as pd
from torch import nn
import os

class BertFineTuner:
    """BERT-based text classifier with iterative fine-tuning support.

    Wraps HuggingFace Trainer to expose the Trainer protocol expected by the
    pipeline (train, infer, save, update_model). Column names and number of
    output labels are fully configurable so the same class works for binary
    and multiclass tasks across different datasets.
    """

    def __init__(
        self,
        model_name: Optional[str],
        training_data: Optional[pd.DataFrame],
        test_data: Optional[pd.DataFrame],
        text_col: str = "title",
        label_col: str = "label",
        output_dir: str = "results",
        logging_dir: str = "./logs",
        learning_rate=2e-5,
        dropout=0.2,
        num_labels: int = 2,
    ):
        self.base_model = model_name
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_model_acc: Dict[str, float] | None = None
        self.training_data = training_data
        self.test_data = test_data
        self.text_col = text_col
        self.label_col = label_col
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.trainer = None
        self.run_clf = False
        self.learning_rate = learning_rate
        self.weight_decay = 0.00 
        self.num_labels = num_labels
        
        if dropout:
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            model.config.hidden_dropout_prob = dropout
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        self.model.to(self.device)

    def run_hyperparameter_search(self, df: pd.DataFrame, metric: str = "accuracy") -> Dict[str, Any]:
        """Runs a grid search over LR and Weight Decay, resetting the model each time."""
        learning_rates = [1e-5, 2e-5, 5e-5]
        weight_decays = [0.0, 0.01, 0.1]
        
        best_score = -float('inf')
        best_params = {"lr": self.learning_rate, "wd": self.weight_decay}
        original_base = self.base_model 

        for lr in learning_rates:
            for wd in weight_decays:
                print(f">>> Testing Grid: LR={lr}, WD={wd}")
                
                # HARD RESET: Fresh model for every trial to prevent weight poisoning
                self.model = BertForSequenceClassification.from_pretrained(
                    original_base, num_labels=self.num_labels
                ).to(self.device)
                
                self.learning_rate = lr
                self.weight_decay = wd
                
                # Run training trial
                results, _ = self.train_data(df, still_unbalenced=True)
                
                current_score = results.get(f"eval_{metric}", 0)
                if current_score > best_score:
                    best_score = current_score
                    best_params = {"lr": lr, "wd": wd}
                    print(f"New best grid score: {best_score}")

        # Apply best found parameters
        self.learning_rate = best_params["lr"]
        self.weight_decay = best_params["wd"]
        
        # Final Reset to base weights before real training begins
        self.model = BertForSequenceClassification.from_pretrained(
            original_base, num_labels=self.num_labels
        ).to(self.device)
        
        return best_params

    def set_clf(self, set: bool):
        """Enable or disable classifier-based filtering during sampling."""
        self.run_clf = set

    def get_clf(self):
        """Return whether classifier filtering is currently enabled."""
        return self.run_clf

    def get_last_model_acc(self):
        """Return the metrics dict from the most recent evaluation."""
        return self.last_model_acc

    def set_train_data(self, train):
        """Replace the current training DataFrame."""
        self.training_data = train

    def get_train_data(self):
        """Return the current training DataFrame."""
        return self.training_data

    def get_base_model(self):
        """Return the name or path of the current base model."""
        return self.base_model

    def enable_filtering(self, enabled: bool) -> None:
        self.set_clf(enabled)

    def is_filtering_enabled(self) -> bool:
        return bool(self.get_clf())

    def get_last_model_metrics(self) -> Dict[str, float] | None:
        return self.get_last_model_acc()

    def train(self, df_train: pd.DataFrame, still_unbalenced: bool) -> Dict[str, Any]:
        results, _ = self.train_data(df_train, still_unbalenced)
        return results

    def infer(self, df_unlabeled: pd.DataFrame) -> torch.Tensor:
        return self.get_inference(df_unlabeled)

    def save(self, path: str) -> None:
        self.save_model(path)

    def create_dataset(self, train, test):
        """Tokenize train and test DataFrames and return a tokenized DatasetDict and collator."""
        def tokenize_function(element):
            return self.tokenizer(element[self.text_col], padding="max_length", truncation=True, max_length=512)

        train = train.copy()
        test = test.copy()
        train[self.label_col] = train[self.label_col].astype(int)
        test[self.label_col] = test[self.label_col].astype(int)

        dataset_train = Dataset.from_pandas(train[[self.text_col, self.label_col]])
        dataset_val = Dataset.from_pandas(test[[self.text_col, self.label_col]])

        dataset = DatasetDict({"train": dataset_train, "val": dataset_val})
        tokenized_data = dataset.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        return tokenized_data, data_collator

    def create_test_dataset(self, df: pd.DataFrame) -> Dataset:
        """Tokenize an unlabeled DataFrame for inference."""
        def tokenize_function(element):
            return self.tokenizer(element[self.text_col], padding="max_length", truncation=True, max_length=512)

        test_dataset = Dataset.from_pandas(df[[self.text_col]])
        dataset = DatasetDict({"test": test_dataset})
        tokenized_data = dataset.map(tokenize_function, batched=True)
        return tokenized_data

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='weighted'),
            'recall': recall_score(labels, preds, average='weighted'),
            'f1': f1_score(labels, preds, average='weighted')
        }

    def train_data(self, df, still_unbalenced):
        """Run a full training + evaluation pass and return (results_dict, trainer)."""
        early_stopping_callback = EarlyStoppingCallback(patience=5, log_dir=self.logging_dir)
        tokenized_data, data_collator = self.create_dataset(df, self.test_data)

        # training_args = TrainingArguments(
        #     output_dir=self.output_dir,
        #     eval_strategy="epoch",
        #     save_strategy="epoch",
        #     metric_for_best_model="eval_accuracy",
        #     per_device_train_batch_size=32,
        #     per_device_eval_batch_size=32,
        #     num_train_epochs=20, # Note: Consider lowering this for grid search trials
        #     learning_rate=self.learning_rate,
        #     weight_decay=self.weight_decay,
        #     save_total_limit=2,
        #     logging_steps=10,
        #     push_to_hub=False,
        #     logging_dir=self.logging_dir,
        #     load_best_model_at_end=True
        # )
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_f1", # Changed this to f1 for you too
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=20, 
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            save_total_limit=2,
            logging_steps=10,
            push_to_hub=False,
            logging_dir=self.logging_dir,
            load_best_model_at_end=True,
            fp16=True  
        )
        
        trainer_class = MyTrainer if still_unbalenced else Trainer
        extra_args = {"num_labels": self.num_labels} if still_unbalenced else {}

        trainer = trainer_class(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=BertFineTuner.compute_metrics,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["val"],
            callbacks=[early_stopping_callback],
            **extra_args
        )

        trainer.train()
        results = trainer.evaluate()
        self.trainer = trainer
        return results, self.trainer

    def get_inference(self, df: pd.DataFrame) -> torch.Tensor:
        """Run inference on df in chunks and return a tensor of predicted class indices."""
        predicted_labels = []
        chunk_size = 10000
        for i in range(0, len(df), chunk_size):
            chunk = df[i : i + chunk_size]
            test_dataset = self.create_test_dataset(chunk)
            predictions = self.trainer.predict(test_dataset["test"])
            batch_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)
            predicted_labels.append(batch_labels)
        return torch.cat(predicted_labels)

    def save_model(self, path: str):
        """Save the current trainer's model artifacts to the given path."""
        if self.trainer:
            self.trainer.save_model(path)

    def update_model(self, model_name, model_acc, save_model: bool):
        """Optionally save and then reload the model from model_name, updating the active base."""
        if save_model:
            self.save_model(model_name)
        self.last_model_acc = {model_name: model_acc}
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels)
        self.base_model = model_name
        self.model.to(self.device)

class MyTrainer(Trainer):
    """HuggingFace Trainer subclass with class-weighted cross-entropy loss.

    For binary tasks applies a [0.2, 0.8] weight to up-weight the positive class,
    which helps with the label imbalance typical in active learning pools.
    For multiclass tasks falls back to standard unweighted cross-entropy.
    """

    def __init__(self, **kwargs):
        self._num_labels = kwargs.pop("num_labels", 2)
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Applying class weights to handle imbalance in binary classification
        if self._num_labels == 2:
            weight = torch.tensor([0.2, 0.8], device=model.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    """Stops training when eval loss stops improving and writes epoch logs to disk."""

    def __init__(self, patience=5, log_dir=None):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        self.log_dir = log_dir

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Look for the evaluation loss in the metrics dictionary
        current_loss = metrics.get("eval_loss")
        
        if current_loss is not None:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping triggered. No improvement for {self.patience} evaluation steps.")
                    control.should_training_stop = True
        
        # Save logs if log_dir is provided
        if self.log_dir and state.is_world_process_zero:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, f"history_epoch_{state.epoch}.txt")
            with open(log_path, "w") as f:
                f.write(str(state.log_history))
        
        return control
