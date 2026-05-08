# Data Adapter Layer

The adapter layer decouples dataset-specific details from the core pipeline. Each dataset is defined by one YAML config file and one Python subclass of `DataAdapter` base class. The pipeline queries the adapter for column names, file paths, prompt logic, and trainer configuration — nothing is hardcoded in the core algorithm.

The adapter acts as the "glue" between a raw dataset and the algorithm: it describes what the data looks like (column schema, file locations) and what task is being solved (prompt format, label mapping).

---

## What an Adapter Controls

- **Schema** — maps logical roles (id, input text, label, etc.) to actual column names in the CSV
- **I/O** — where to load the pool and validation CSVs from
- **Prompt** — how to format input text into an LLM prompt; how to parse the raw response into a class label (binary or multiclass)
- **Artifacts** — where to write sampler state, models, and results (namespaced per dataset under `artifacts/<run_name>/`)
- **Trainer** — how to instantiate BERT: number of labels, model variant, sequence length, batch size, etc.

---

## Preconfigured Datasets

| Dataset  | Key      | Task Type  | Classes | Config                         |
|----------|----------|------------|---------|--------------------------------|
| Wildlife | wildlife | Binary     | 2       | adapters/configs/wildlife.yaml |
| Emotions | emotions | Multiclass | 6       | adapters/configs/emotions.yaml |
| Reuters  | reuters  | Multiclass | 9       | adapters/configs/reuters.yaml  |

---

## How It Interacts with the Pipeline

`main_cluster.py` runs in adapter mode when `-dataset` is passed:

1. Loads pool + validation CSVs via `adapter.load_pool_df()` / `adapter.load_validation_df()`
2. Preprocesses text using the adapter's column names (`get_input_text_col()`, `get_clean_text_col()`)
3. Runs LDA and writes cluster IDs into the adapter's cluster column (`get_cluster_col()`)
4. For each sampled row:
   - Builds a prompt via `adapter.format_prompt(text)`
   - Calls the labeling engine: `Labeling.predict(prompt)` → raw response string
   - Maps raw response → integer label via `adapter.parse_model_output(output)`
5. Fine-tunes BERT via a trainer built once with `adapter.build_trainer(validation_df)`
6. Updates the Thompson sampler with rewards; saves all state to the adapter's artifact paths

As a result, core modules carry no dataset-specific logic:
- `labeling.py` is a pure execution engine: prompt in → raw response out
- `fine_tune.py` accepts `text_col`, `label_col`, and `num_labels` as parameters
- `thompson_sampling.py` / `random_sampling.py` accept column names and state file paths as parameters

---

## Artifact Layout

Each adapter writes to its own namespaced directory (default: `artifacts/<run_name>/`):

```
artifacts/<run_name>/
├── selected_ids.txt       # IDs already labeled (prevents re-sampling)
├── wins.txt               # Thompson sampler wins per cluster
├── losses.txt             # Thompson sampler losses per cluster
├── lda_cache.csv          # preprocessed + clustered pool (cached)
├── data_labeled.csv       # all labeled rows accumulated across iterations
├── positive_data.csv      # positive examples used in training
├── training_data.csv      # full training set for the current iteration
├── models/                # saved BERT checkpoints
├── results/
│   └── model_results.json # per-iteration evaluation metrics
└── logs/                  # training logs
```

This prevents one dataset's sampling history or model state from affecting another.

---

## YAML Config Reference

```yaml
# Namespace for artifacts. Must be unique per dataset to avoid state collisions.
run_name: my_dataset

# Root directory for all artifacts (default: "artifacts").
artifact_root: artifacts

data:
  pool_csv: ../../data_use_cases/my_pool.csv          # REQUIRED
  validation_csv: ../../data_use_cases/my_val.csv     # REQUIRED
  # Paths are resolved relative to this YAML file.

columns:
  id: id                        # default: "id"
  input_text: title             # default: "title" — text fed to LDA and the prompt
  description: description      # optional — concatenated with input_text during preprocessing
  target: label                 # default: "label" — ground truth column
  clean_text: clean_title       # default: "clean_title" — output column of preprocessing
  cluster: label_cluster        # default: "label_cluster" — output column of LDA
  predicted: predicted_label    # default: "predicted_label" — model filtering output
  answer: answer                # default: "answer" — raw LLM response
  prompt: prompt                # default: "prompt" — formatted prompt string

# For binary classification:
prompt:
  template_path: ../../prompts/my_dataset/gpt_prompt.txt   # REQUIRED
  positive_text: relevant animal   # case-insensitive substring match against LLM response
  positive_value: 1                # label on match (default: 1)
  negative_value: 0                # label otherwise (default: 0)

# For multiclass classification (replaces positive_text/positive_value/negative_value):
prompt:
  template_path: ../../prompts/my_dataset/gpt_prompt.txt
  label_map:
    sadness: 0
    anger: 1
    fear: 2
    joy: 3
    love: 4
    surprise: 5
  default_value: 0    # label returned when no keyword matches

trainer:
  model_name: bert-base-uncased   # default: "bert-base-uncased"
  num_labels: 2                   # default: 2 — set higher for multiclass
  fp16: true                      # optional — requires CUDA
  max_seq_length: 128             # optional
  per_device_train_batch_size: 16 # optional

training:
  balance: false    # optional — overrides the global -balance flag for this dataset
```

---

## Prompt Templates

Prompt templates are plain `.txt` files. The adapter calls `.format(...)` on them at runtime, replacing either `{text}` or `{title}` with the input text.

**Binary example** (`prompts/wildlife_dataset/gpt_prompt.txt`):
```text
Decide if the following listing is related to a wildlife product.
Reply with "relevant animal" if it is, or "not relevant" if it is not.

Text:
{text}
```

**Multiclass example** (`prompts/emotions/gpt_prompt.txt`):
```text
Classify the emotion expressed in the following text.
Reply with exactly one of: sadness, anger, fear, joy, love, surprise.

Text:
{text}
```

---

## Adding a New Dataset

1. **Create a YAML config** at `adapters/configs/<name>.yaml`
   - Set `data.pool_csv` and `data.validation_csv`
   - Set `columns.*` to match the CSV headers
   - Set `prompt.template_path` and the appropriate prompt fields (binary or multiclass)
   - Set `trainer.num_labels` if more than 2 classes

2. **Write a prompt template** at `prompts/<name>/gpt_prompt.txt`
   - Use `{text}` or `{title}` as the placeholder

3. **Create an adapter class** at `adapters/<name>.py`:
   ```python
   from data_adapter import DataAdapter, AdapterRegistry
   from fine_tune import BertFineTuner

   class MyDatasetAdapter(DataAdapter):
       name = "mydataset"

       def build_trainer(self, validation_df):
           paths = self.get_paths()
           model_name = self.config.get("trainer", {}).get("model_name", "bert-base-uncased")
           num_labels = int(self.config.get("trainer", {}).get("num_labels", 2))
           return BertFineTuner(
               model_name=model_name,
               training_data=None,
               test_data=validation_df,
               text_col=self.get_input_text_col(),
               label_col=self.get_target_col(),
               output_dir=str(paths.results_dir),
               logging_dir=str(paths.logs_dir),
               num_labels=num_labels,
           )

   AdapterRegistry.register(MyDatasetAdapter)
   ```

4. **Register the adapter** by adding an import to `adapters/__init__.py`:
   ```python
   from adapters import mydataset
   ```

5. **Run:**
   ```bash
   python main_cluster.py -dataset mydataset -sampling thompson -sample_size 50 -labeling gpt
   ```

---

## Abstract Base Class Reference

All methods in `DataAdapter` (`data_adapter.py`). Only `build_trainer()` is required — all others have defaults that read from the YAML config.

| Method | Required | Description |
|---|---|---|
| `build_trainer(validation_df)` | Yes | Construct and return a configured `BertFineTuner` instance |
| `get_id_col()` | No | Column name for unique row ID (default: `"id"`) |
| `get_input_text_col()` | No | Column name for raw input text (default: `"title"`) |
| `get_description_col()` | No | Optional secondary text column (default: `None`) |
| `get_target_col()` | No | Column name for ground truth label (default: `"label"`) |
| `get_clean_text_col()` | No | Column name for preprocessed text output (default: `"clean_title"`) |
| `get_cluster_col()` | No | Column name for LDA cluster ID (default: `"label_cluster"`) |
| `get_predicted_col()` | No | Column name for filtering predictions (default: `"predicted_label"`) |
| `get_answer_col()` | No | Column name for raw LLM response (default: `"answer"`) |
| `load_pool_df()` | No | Load unlabeled pool CSV (reads `data.pool_csv` from YAML) |
| `load_validation_df()` | No | Load validation CSV (reads `data.validation_csv` from YAML) |
| `prepare_pool_df(df)` | No | Hook for schema normalization before the pool is used |
| `prepare_validation_df(df)` | No | Hook for schema normalization before validation is used |
| `format_prompt(text)` | No | Format input text into a prompt string using the template file |
| `parse_model_output(output)` | No | Map raw LLM response string → integer class label |
| `get_paths()` | No | Return `AdapterPaths` with all artifact file and directory paths |
