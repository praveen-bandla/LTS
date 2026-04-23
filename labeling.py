import pandas as pd
from pprint import pprint
import torch
import os
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from openai import OpenAI
import pandas as pd

from config import OPENAI_API_KEY_ENV_VAR, OPENAI_CONFIG, PROMPT_CONFIG


class Labeling:
    """Labeling engine for LLaMA / OpenAI GPT.

    After the adapter refactor, this class acts as an execution engine:
    prompt in -> raw response out. Prompt formatting and response parsing are
    handled by the dataset adapter.
    """

    def __init__(self, label_model= "llama"):
        self.label_model = label_model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._gpt_prompt_template: str | None = None

    def predict(self, prompt: str, record_id: str | None = None) -> str:
        """Return the raw model response for a fully-formed prompt string."""
        if self.label_model == "llama":
            return self._predict_llama(prompt)
        elif self.label_model == "gpt":
            return self._predict_gpt(prompt, record_id=record_id)
        elif self.label_model == "file":
            return self.get_file_label({"id": record_id, "text": prompt})
        raise ValueError("No model selected")

    def generate_prompt(self, title):
        if self.label_model == "llama":
            return self.generate_prompt_llama(title)
        elif self.label_model=="gpt":
            return self.generate_prompt_gpt(title)
        else:
            return None


    def generate_prompt_llama(self, title: str) -> str:
        return f"""### Instruction: {self.prompt_llama}
                ### Input:
                {title.strip()}
                """

    def generate_prompt_gpt(self, title):
        if not self._gpt_prompt_template:
            raise ValueError("GPT prompt template not loaded. Did you call set_model()?")
        return self._gpt_prompt_template.format(title=title)

    def generate_llama_prompt(self) -> str:
        return self._read_prompt_file(PROMPT_CONFIG.llama_instruction_path)

    @staticmethod
    def _read_prompt_file(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip() + "\n"

    def set_model(self):
        if self.label_model == "llama":
            checkpoint = "llama/"
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.prompt_llama = self.generate_llama_prompt()
            print("model Loaded")
        elif self.label_model == "gpt":
            if not OPENAI_CONFIG.api_key:
                raise ValueError(
                    f"Missing OpenAI API key. Set {OPENAI_API_KEY_ENV_VAR} in .env (see .env file)."
                )
            self.model = OpenAI(api_key=OPENAI_CONFIG.api_key)
            self._gpt_prompt_template = self._read_prompt_file(PROMPT_CONFIG.gpt_prompt_path)
        elif self.label_model =="file":
            self.model = None


    def predict_animal_product(self, row):
        # print(f"Prediction Animal with {self.model}")
        label = Labeling.check_already_label(row)
        if label:
            return label
        if self.label_model == "llama":
            return self.get_llama_label(row)
        elif self.label_model == "gpt":
            return self.get_gpt_label(row)
        elif self.label_model == "file":
            return self.get_file_label(row)
        else:
            raise ValueError("No model selected")


    def generate_inference_data(self, data, column):
        def truncate_string(s, max_length=2000):  # Adjust max_length as needed
            return s[:max_length] + '...' if len(s) > max_length else s

        if self.label_model != "file":
            examples = []
            for _, data_point in data.iterrows():
                examples.append(
                {
                    "id": data_point["id"],
                    "title": data_point["title"],
                    "training_text": data_point["clean_title"],
                    "text": self.generate_prompt(truncate_string(data_point[column])),
                }
                )
            data = pd.DataFrame(examples)
        return data



    def _predict_gpt(self, prompt: str, record_id: str | None = None) -> str:
        labels = pd.read_csv("labaled_by_gpt.csv") if os.path.exists("labaled_by_gpt.csv") else None
        if labels is not None and record_id is not None:
            if record_id in labels["id"].astype(str).to_list():
                return labels.loc[labels["id"].astype(str) == str(record_id), "label"].values[0]

        response = self.model.chat.completions.create(
            model=OPENAI_CONFIG.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=OPENAI_CONFIG.max_tokens,
            temperature=OPENAI_CONFIG.temperature,
        )
        return response.choices[0].message.content

    def get_gpt_label(self, row):
        return self._predict_gpt(row["text"], record_id=str(row.get("id")))


    def _predict_llama(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
            results = self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
            try:
                answer = results.split("Response:\n")[2].split("\n")[0]
            except Exception:
                # Handle IndexError separately
                try:
                    answer = results.split("Response:\n")[1].split("\n")[0]
                except Exception:
                    # Handle any other exception
                    answer = 'not a relevant animal'
        return answer

    def get_llama_label(self, row):
        return self._predict_llama(row["text"])


    def get_file_label(self, row):
        raise NotImplementedError()

    @staticmethod
    def check_already_label(row):
        return None
        # labeled_data = pd.read_csv("all_labeled_data_gpt.csv")
        # if row["id"] in labeled_data["id"].values:
        #     # Retrieve the label for the corresponding id
        #     print("data already labeled")
        #     label = labeled_data.loc[labeled_data["id"] == row["id"], "label"].values[0]
        #     return label
        # else:
        #     return None


