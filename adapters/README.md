# Adapters

Adapters isolate *dataset-specific* details (columns, prompts, and artifact/state paths) from the core pipeline.
After this refactor, adding a dataset should be: **new adapter subclass + YAML + prompt file**.

If you’re new to the repo: think of the adapter as the “glue” that tells the pipeline **what the dataset looks like** (column names + file paths) and **what labeling task we’re doing** (prompt + how to interpret the answer).

## What an adapter controls

Each adapter (a `DataAdapter` subclass) defines:

- **I/O**: how to load pool + validation CSVs
- **Schema**: which columns mean id/text/clean text/cluster/label/predicted label
- **Task**: how to build a prompt and map raw model output -> `0/1`
- **Artifacts/state**: where run outputs and sampler state files live
- **Training**: how to construct the `Trainer` used by the loop

## Quick repo map (where things live)

- `data_adapter.py`: the adapter base class + registry (`DataAdapter`, `AdapterRegistry`)
- `adapters/`: dataset-specific adapter implementations
  - `adapters/wildlife.py`: example adapter
  - `adapters/configs/*.yaml`: dataset configs
- `main_cluster.py`: main pipeline loop (sampling → labeling → fine-tune → update sampler)
- `preprocessing.py`: text cleaning (`TextPreprocessor`)
- `LDA.py`: topic model used to assign cluster IDs (`LDATopicModel`)
- `labeling.py`: labeling *engine* (runs GPT/LLaMA; does not decide prompts/labels anymore)
- `fine_tune.py`: BERT fine-tuning (`BertFineTuner`; now column-configurable)
- `thompson_sampling.py`, `random_sampling.py`: samplers (now column + path configurable)

## How it interacts with the pipeline

`main_cluster.py` runs in **adapter mode** when you pass `-dataset`:

- Loads pool + validation via the adapter (`load_*_df`, `prepare_*_df`)
- Preprocesses text using adapter column names (`TextPreprocessor.preprocess_df`)
- Creates clusters (LDA) and stores them in the adapter’s cluster column
- Labels samples by:
  - building a prompt via `adapter.format_prompt(text)`
  - calling the labeling engine `Labeling.predict(prompt)`
  - mapping raw response -> `0/1` via `adapter.parse_model_output(output)`
- Builds a trainer once via `adapter.build_trainer(validation_df)`
- Creates samplers with adapter-provided **column names** and **state paths**

### End-to-end lifecycle (one run, simplified)

1) Adapter loads pool + validation CSVs
2) Preprocessing writes cleaned text into the adapter’s “clean text” column
3) LDA assigns a cluster id per row into the adapter’s “cluster” column (cached in `lda_cache.csv`)
4) A sampler chooses which rows to label next (and avoids re-sampling ids from `selected_ids.txt`)
5) For each sampled row:
  - adapter builds a prompt string
  - labeling engine returns raw response text
  - adapter converts raw response → `0/1` label
6) Trainer fine-tunes on newly labeled rows (plus any saved positive examples)
7) Reward updates Thompson sampler wins/losses, and model artifacts/metrics are saved under the adapter’s artifact directory

Key couplings removed from core modules:

- `labeling.py` is now an execution engine: prompt in -> raw response out
- `fine_tune.py` no longer assumes `title`/`label` (it accepts `text_col`/`label_col`)
- `thompson_sampling.py` / `random_sampling.py` no longer assume `id`/`label_cluster` nor global `selected_ids.txt`/`wins.txt`/`losses.txt`

## Artifact/state layout

Each adapter chooses an artifact directory (default `artifacts/<run_name>/`).
This directory contains namespaced run state like:

- `selected_ids.txt`, `wins.txt`, `losses.txt`
- `positive_data.csv`
- `lda_cache.csv`
- `data_labeled.csv`, `training_data.csv`
- `model_results.json`
- `models/`, `logs/`, `results/`

This prevents dataset A’s sampling/training history from affecting dataset B.

Notes:

- Paths in the YAML are resolved **relative to the YAML file** (so configs are portable)
- If you run two datasets, each gets its own `artifacts/<run_name>/...` directory

## Existing example

- Adapter: `adapters/wildlife.py`
- YAML: `adapters/configs/wildlife.yaml`
- Prompt template: `prompts/wildlife_dataset/gpt_prompt.txt`

Run it:

```bash
python main_cluster.py -dataset wildlife -config adapters/configs/wildlife.yaml
```

## Adapter YAML (example + meaning)

This is the minimum set of fields most datasets will need:

```yaml
run_name: my_dataset_run
artifact_root: artifacts

data:
  pool_csv: ../../data_use_cases/my_pool.csv
  validation_csv: ../../data_use_cases/my_validation.csv

columns:
  id: id
  input_text: title
  description: description         # optional
  target: label                   # where 0/1 labels live
  clean_text: clean_title          # where preprocessing writes
  cluster: label_cluster           # where LDA writes cluster IDs
  predicted: predicted_label       # where filtering predictions go
  answer: answer                   # where raw model response is stored

prompt:
  template_path: ../../prompts/my_dataset/gpt_prompt.txt
  positive_text: relevant animal   # substring match (case-insensitive)
  positive_value: 1
  negative_value: 0

trainer:
  model_name: bert-base-uncased
```

## Prompt templates

Prompt templates are plain text files. The adapter will call `.format(...)` on them.
Use either placeholder:

- `{text}`
- `{title}`

Example:

```text
Decide if the following text is relevant. Reply with "relevant animal" or "not relevant".

Text:
{text}
```

## Adding a new dataset (checklist)

1) **Create a YAML** (copy `adapters/configs/wildlife.yaml`)
- Set `data.pool_csv` + `data.validation_csv`
- Set `columns.*` to match your CSV schema
- Point `prompt.template_path` to your prompt file
- Set `prompt.positive_text` (substring match) and `positive_value`/`negative_value`

2) **Add a prompt template file**
- Use either `{text}` or `{title}` placeholder (both are supported)

3) **Create an adapter subclass** in `adapters/<dataset>.py`
- Implement `name = "<dataset>"`
- Implement `build_trainer(validation_df)`
- Optionally override:
  - `prepare_pool_df` / `prepare_validation_df` (schema normalization)
  - `format_prompt` / `parse_model_output` (task-specific behavior)

4) **Register it**
- Ensure the adapter module registers itself via `AdapterRegistry.register(...)`
- Add it to `adapters/__init__.py` so it’s imported (and registered) at runtime

5) **Run**

```bash
python main_cluster.py -dataset <dataset> -config adapters/configs/<dataset>.yaml
```

## Common gotchas / troubleshooting

- **"KeyError" on a column name**: your YAML `columns.*` does not match the CSV header.
- **Prompt template not found**: `prompt.template_path` is relative to the YAML file location.
- **No OpenAI key** (GPT labeling): set `OPENAI_API_KEY` in your environment (see `config.py`).
- **Weird sampling repeats**: delete `artifacts/<run_name>/selected_ids.txt` to restart sampling state.
- **Switching datasets but state leaks**: ensure each dataset has a unique `run_name`.
