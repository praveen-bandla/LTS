from typing import Any, Optional, Dict, Union, Tuple, Callable, List
# import tensorflow as tf
from transformers import Trainer, TrainingArguments, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import Trainer, PreTrainedTokenizerBase, TrainingArguments, DataCollatorWithPadding, BertForSequenceClassification, PreTrainedModel
import torch
from torch.utils.data import IterableDataset
import pandas as pd
from torch import nn

class BertFineTuner:
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
        """BERT fine-tuner used by the sampling loop.

        The adapter refactor makes `text_col` and `label_col` configurable so the
        core pipeline does not hardcode dataset schemas.
        """
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
        self.weight_decay = 0.00 # previously None, checking if this solves error 4/15/26 RMG
        self.num_labels = num_labels
        if dropout:
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            model.config.hidden_dropout_prob = dropout
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)


    def set_clf(self, set: bool):
        self.run_clf = set

    def get_clf(self):
        return self.run_clf

    def get_last_model_acc(self):
        return self.last_model_acc

    def set_train_data(self, train):
        self.training_data = train

    def get_train_data(self):
        return self.training_data

    def get_base_model(self):
        return self.base_model

    # --- Adapter-facing API ---

    def enable_filtering(self, enabled: bool) -> None:
        """Adapter-facing alias for enabling sampler filtering."""
        self.set_clf(enabled)

    def is_filtering_enabled(self) -> bool:
        """Return whether filtering is enabled."""
        return bool(self.get_clf())

    def get_last_model_metrics(self) -> Dict[str, float] | None:
        """Adapter-facing alias for the last evaluation metrics."""
        return self.get_last_model_acc()

    def train(self, df_train: pd.DataFrame, still_unbalenced: bool) -> Dict[str, Any]:
        """Train and return evaluation metrics."""
        results, _ = self.train_data(df_train, still_unbalenced)
        return results

    def infer(self, df_unlabeled: pd.DataFrame) -> torch.Tensor:
        """Run inference for filtering."""
        return self.get_inference(df_unlabeled)

    def save(self, path: str) -> None:
        """Persist the current model to `path`."""
        self.save_model(path)

    def create_dataset(self, train, test):
        def tokenize_function(element):
            return self.tokenizer(element[self.text_col], padding="max_length", truncation=True, max_length=512)

        dataset_train = Dataset.from_pandas(train[[self.text_col, self.label_col]])
        dataset_val = Dataset.from_pandas(test[[self.text_col, self.label_col]])

        dataset = DatasetDict()
        dataset["train"] = dataset_train
        dataset["val"] = dataset_val

        tokenized_data = dataset.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        return tokenized_data, data_collator

    def create_test_dataset(self, df: pd.DataFrame) -> Dataset:
        def tokenize_function(element):
            return self.tokenizer(element[self.text_col], padding="max_length", truncation=True, max_length=512)

        test_dataset = Dataset.from_pandas(df[[self.text_col]])

        dataset = DatasetDict()
        dataset["test"] = test_dataset

        tokenized_data = dataset.map(tokenize_function, batched=True)

        return tokenized_data

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        accuracy = accuracy_score(labels, preds)

        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train_data(self, df, still_unbalenced):
        early_stopping_callback = EarlyStoppingCallback(patience=5, log_dir=self.logging_dir)

        tokenized_data, data_collator = self.create_dataset(df, self.test_data)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",  # "epoch", "steps", or EvaluationStrategy.EPOCH
            save_strategy="epoch",
            metric_for_best_model="eval_accuracy",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=20,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            save_total_limit=2,
            logging_steps=10,
            push_to_hub=False,
            logging_dir=self.logging_dir,
            load_best_model_at_end=True
        )
        if still_unbalenced:
            print(f"using modified loss function")
            # Create a Trainer
            trainer = MyTrainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                compute_metrics=BertFineTuner.compute_metrics,
                train_dataset=tokenized_data["train"],
                eval_dataset=tokenized_data["val"],
                callbacks=[early_stopping_callback],
                num_labels=self.num_labels,
            )

            # Fine-tune the model
            trainer.train()
            results = trainer.evaluate()
            print(results)

            self.trainer = trainer

            return results, self.trainer
        else:
            # Create a Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                compute_metrics=BertFineTuner.compute_metrics,
                train_dataset=tokenized_data["train"],
                eval_dataset=tokenized_data["val"],
                callbacks=[early_stopping_callback]
            )

            # Fine-tune the model
            trainer.train()
            results = trainer.evaluate()
            print(results)

            self.trainer = trainer

            return results, self.trainer


    def get_inference(self, df: pd.DataFrame) -> torch.Tensor:
        predicted_labels = []
        chunk_size = 10000
        total_records = len(df)
        start_index = 0

        while start_index < total_records:
            end_index = min(start_index + chunk_size, total_records)
            chunk = df[start_index:end_index]
            test_dataset = self.create_test_dataset(chunk)
            # data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            # data_loader = DataLoader(test_dataset["test"], batch_size=batch_size, collate_fn=data_collator)
            predictions = self.trainer.predict(test_dataset["test"])  # Make predictions on the current batch
            prediction_scores = predictions.predictions
            batch_predicted_labels = torch.argmax(torch.tensor(prediction_scores), dim=1)

            predicted_labels.append(batch_predicted_labels)

            start_index = end_index


        # Concatenate the predicted labels from all batches
        predicted_labels = torch.cat(predicted_labels)

        return predicted_labels

    def save_model(self, path: str):
        self.trainer.save_model(path)

    def update_model(self, model_name, model_acc, save_model: bool):
        if save_model:
            self.save_model(model_name)
        self.last_model_acc = {model_name: model_acc}
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels)
        self.base_model = model_name
        self.model.to(self.device)


class MyTrainer(Trainer):
    def __init__(self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[Any] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[Any], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[Any], Dict]] = None,
            callbacks: Optional[List[Any]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            num_labels: int = 2):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self._num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Binary: use class weights to handle imbalance. Multiclass: equal weights.
        if self._num_labels == 2:
            weight = torch.tensor([0.2, 0.8], device=model.device)
        else:
            weight = None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss



from transformers import TrainerCallback, Trainer

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, log_dir=None):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        self.log_dir = log_dir

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and state.log_history:
            current_loss = None
            for log_entry in reversed(state.log_history):
                if 'eval_loss' in log_entry:
                    current_loss = log_entry['eval_loss']
            if current_loss is not None:  # Check if loss is available in log history
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        control.should_training_stop = True
                # Save logs
                if self.log_dir:
                    with open(f"{self.log_dir}/epoch_{state.epoch}.txt", "w") as f:
                        for log in state.log_history:
                            f.write(f"{log}\n")
