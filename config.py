from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROMPT_SET_DIR = Path("wildlife_dataset")
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"


@dataclass(frozen=True)
class RunConfig:
    # Core pipeline settings
    sampling: str = "thompson"  # "thompson" | "random"
    sample_size: int = 200
    filter_label: bool = True
    balance: bool = True

    # Models
    model: str = "text"  # currently only "text" is supported
    model_finetune: str = "bert-base-uncased"

    # Labeling source
    labeling: str = "gpt"  # "gpt" | "llama" | "file"

    # Evaluation
    metric: str = "f1"  # "f1" | "accuracy" | "recall" | "precision"
    baseline: float = 0.5

    # Data
    filename: str = "data_use_cases/shark_trophy"  # base path without .csv
    val_path: str = "data_use_cases/validation_sharks.csv"

    # Clustering
    cluster_size: int = 5


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None
    model: str = "gpt-4o-mini"
    max_tokens: int = 100
    temperature: float = 0.2


@dataclass(frozen=True)
class PromptConfig:
    # Prompt templates live in prompts/.
    gpt_prompt_path: Path = PROMPTS_DIR / PROMPT_SET_DIR / "gpt_prompt.txt"
    llama_instruction_path: Path = PROMPTS_DIR / PROMPT_SET_DIR / "llama_prompt.txt"


@dataclass(frozen=True)
class LabelConfig:
    positive_text: str = "relevant animal"
    positive_value: int = 1
    negative_value: int = 0


RUN_CONFIG = RunConfig()
OPENAI_CONFIG = OpenAIConfig(api_key=os.getenv(OPENAI_API_KEY_ENV_VAR))
PROMPT_CONFIG = PromptConfig()
LABEL_CONFIG = LabelConfig()
