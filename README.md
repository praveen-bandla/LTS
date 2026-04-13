# 🐾 Wildlife Trafficking Detection: Using LLMs to Automate the Creation of Classifiers for Data Triage

This repository contains the code used in the paper:

> **"A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces"**
> Proceedings of the ACM on Management of Data, Volume 3, Issue 3. Article No.: 119, Pages 1 - 23.  
> https://dl.acm.org/doi/10.1145/3725256
> https://arxiv.org/html/2504.21211v1

---

## 📚 Table of Contents

1. [Requirements](#-requirements)
2. [Setup](#-setup)
3. [Use Cases & Reproducibility](#-use-cases--reproduction)

---

## 📦 Requirements

Experiments were conducted using **Python 3.11.2**. All required dependencies are listed in `requirements.txt` and can be installed via pip.

Before running GPT-based labeling, you need to set your OpenAI key via a `.env` file (see Setup below). **Do not hard-code keys in the repo**.

---

## ⚙️ Setup

### 1. Create a Virtual Environment (Recommended)

Use a virtual environment to avoid dependency conflicts:

```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate     # For Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI key (for `-labeling gpt`)

Create a file named `.env` in the repo root and add:

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
```

Notes:
- `.env` is ignored by git via `.gitignore`.
- The code loads `.env` centrally in `config.py`, and `labeling.py` authenticates using that.

---


### Required columns

Training CSV (the `-filename` CSV) must include:
- `id` (unique identifier per row)
- `title` (text used for preprocessing/clustering/labeling)
- Optional: `description` (if present, it will be concatenated with `title` during preprocessing)

Validation CSV (the `-val_path` CSV) must include:
- `title`
- `label` (0/1)

---

🧪 Use Cases & Reproducibility
To reproduce experiments from the paper, run main_cluster.py with the appropriate flags:

The data needed to run all experiments can be found on:
https://drive.google.com/drive/folders/1UO4OYjBmvgKcFz71YeB1kefXqQhMvXGA?usp=sharing

🔧 Required Parameters:

- -sample_size: Number of samples per iteration.

- -filename: Path to the dataset CSV file. Must contain a text column named 'title'.

  Note: pass the *base path without* the `.csv` extension (the code reads `"<filename>.csv"` and writes `"<filename>_lda.csv"`).

- -val_path: Path to the validation dataset.

- -balance: Whether to balance the dataset via undersampling (bool).

- -sampling: Sampling strategy (string: "thompson" or "random").

- -filter_label: Whether to filter out negative samples. (bool)

- -model_finetune: Model name for fine-tuning in the first iteration (string: e.g., "bert-base-uncased").

- -labeling: Source of labels (string: gpt, llama, or file).

- -model: Choose model type (string: text, multi-modal).

- -metric: Evaluation metric used to compare models between iterations (string: "f1", "accuracy", "recall", "precision").

- -baseline: Initial baseline metric score for first iteration.

- -cluster_size: Number of clusters to use.



👜 Use Case 1: Leather Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/data_leather" \
  -val_path "data_use_cases/leather_validation.csv" \
  -balance False \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 10
```

🦈 Use Case 2: Shark Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/shark_trophy" \
  -val_path "data_use_cases/validation_sharks.csv" \
  -balance True \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 5
```

Use Case 3: Animal Products
```bash
python main_cluster.py \
  -sample_size 200 \
  -filename "data_use_cases/animals" \
  -val_path "data_use_cases/validation_animals.csv" \
  -balance True \
  -sampling "thompson" \
  -filter_label True \
  -model_finetune "bert-base-uncased" \
  -labeling "gpt" \
  -model "text" \
  -baseline 0.5 \
  -metric "f1" \
  -cluster_size 10
```

📫 Contact
For questions or feedback, please open an issue or reach out via the contact information provided in the paper.

