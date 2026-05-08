# Wildlife Trafficking Detection: LLM-Assisted Active Learning for Text Classification

This repository contains the code associated with the paper:

> **"A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces"**
> Proceedings of the ACM on Management of Data, Volume 3, Issue 3. Article No.: 119, Pages 1–23.
> https://dl.acm.org/doi/10.1145/3725256
> https://arxiv.org/html/2504.21211v1

The codebase extends the original approach to support multiple datasets and task types through a data adapter layer, without modifying the core algorithm.

---

## How It Works

The pipeline iteratively builds a text classifier from an unlabeled data pool using minimal human annotation cost:

1. **LDA** clusters the unlabeled pool into topic groups
2. **GPT** labels a sample of records using a task-specific prompt
3. **BERT** fine-tunes on the newly labeled data
4. **Thompson/Random sampler** selects which cluster to draw from next, guided by model performance

This loop repeats for a fixed number of iterations, accumulating labeled data and improving the classifier with each pass.

```
Unlabeled Pool → LDA Clusters → GPT Labeling → BERT Fine-Tune → Sampler Update → repeat
```

---

## Setup

Experiments were conducted using **Python 3.11.2**.

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the OpenAI key

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=your_key_here
```

The key is loaded centrally in `config.py` and used by `labeling.py`. The `.env` file is gitignored.

**GPU note:** `fp16: true` in adapter configs requires CUDA. CPU fallback works but is significantly slower for fine-tuning.

---

## Quick Start

### Adapter mode (recommended)

```bash
python main_cluster.py \
  -dataset wildlife \
  -sampling thompson \
  -sample_size 50 \
  -labeling gpt \
  -metric f1 \
  -baseline 0.5
```

Replace `-dataset wildlife` with any preconfigured dataset key (see table below). The adapter config is loaded automatically from `adapters/configs/<dataset>.yaml`.

### Legacy mode

The original interface from the paper is preserved for backwards compatibility:

```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/shark_trophy" \
  -val_path "data_use_cases/validation_sharks.csv" \
  -balance True \
  -sampling thompson \
  -filter_label True \
  -model_finetune bert-base-uncased \
  -labeling gpt \
  -metric f1 \
  -baseline 0.5 \
  -cluster_size 5
```

---

## Preconfigured Datasets

| Dataset  | Key      | Task Type  | Classes | Config                         |
|----------|----------|------------|---------|--------------------------------|
| Wildlife | wildlife | Binary     | 2       | adapters/configs/wildlife.yaml |
| Emotions | emotions | Multiclass | 6       | adapters/configs/emotions.yaml |
| Reuters  | reuters  | Multiclass | 9       | adapters/configs/reuters.yaml  |

Data files for all experiments are available at:
https://drive.google.com/drive/folders/1UO4OYjBmvgKcFz71YeB1kefXqQhMvXGA?usp=sharing

See [`adapters/README.md`](adapters/README.md) for documentation on adding a new dataset.

---

## Repo Structure

```
├── main_cluster.py          # Pipeline orchestrator (adapter mode + legacy mode)
├── data_adapter.py          # DataAdapter base class + AdapterRegistry
├── fine_tune.py             # BERT fine-tuning (BertFineTuner)
├── labeling.py              # LLM labeling engine (GPT / LLaMA)
├── thompson_sampling.py     # Thompson bandit sampler
├── random_sampling.py       # Random sampling baseline
├── preprocessing.py         # Text cleaning (TextPreprocessor)
├── LDA.py                   # Topic modeling (LDATopicModel)
├── config.py                # Global defaults and environment config
├── requirements.txt
│
├── adapters/                # Dataset adapters and per-dataset configs
│   ├── __init__.py          # Auto-registers all built-in adapters
│   ├── wildlife.py
│   ├── emotions.py
│   ├── reuters.py
│   ├── configs/             # YAML config files (one per dataset)
│   └── README.md            # Adapter documentation
│
├── prompts/                 # LLM prompt templates (one directory per dataset)
├── data_use_cases/          # Pool and validation CSVs
├── artifacts/               # Run outputs — gitignored
└── scripts/                 # Data preparation scripts
```

---

## Reproducibility

Each dataset run writes all outputs to `artifacts/<dataset>/`, isolated from other datasets:

- **Sampler state** (`selected_ids.txt`, `wins.txt`, `losses.txt`) is preserved between runs, allowing a run to be resumed.
- **Per-iteration metrics** are written to `artifacts/<dataset>/results/model_results.json`.
- To restart a run from scratch, delete the `artifacts/<dataset>/` directory.

All dataset-specific configuration (column names, file paths, prompt logic, trainer settings) lives in `adapters/configs/<dataset>.yaml`. See [`adapters/README.md`](adapters/README.md) for the full YAML reference.
