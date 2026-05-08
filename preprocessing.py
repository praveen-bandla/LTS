"""Text preprocessing utilities for the LTS pipeline.

Cleans raw text before LDA clustering and LLM labeling: lowercasing,
whitespace normalization, and removal of special characters.
"""

import re
import string
import numpy as np

class TextPreprocessor:
    """Cleans and normalizes raw text columns in a DataFrame."""
    def __init__(self):
        self.punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
        self.weird_chars_regex = re.compile(r'[^a-zA-Z0-9\s]')

    def preprocess_df(self, df, text_col: str = "title", desc_col: str | None = None, output_col: str = "clean_title"):
        """Clean a text column and write the cleaned result to `output_col`.

        Defaults preserve the legacy behavior:
        - input column: `title`
        - optional extra text: `description` (if present)
        - output column: `clean_title`
        """
        if desc_col is None and "description" in df.columns:
            desc_col = "description"

        df = df.dropna(subset=[text_col])
        if desc_col and desc_col in df.columns:
            combined_col = "title_and_desc" if text_col == "title" and desc_col == "description" else f"{text_col}_and_desc"
            df[combined_col] = np.where(df[desc_col].isnull(), df[text_col], df[text_col] + ". " + df[desc_col])
            df[output_col] = df[combined_col].apply(lambda x: self.clean_text(x))
        else:
            df[output_col] = df[text_col].apply(lambda x: self.clean_text(x))
        return df

    def clean_text(self, text):
        try:
            text = text.lower()
            text = text.replace("\n", " ")
            # Remove \xa0 characters
            text = text.replace("\xa0", " ")

            text = text.replace("eBay", "")

            text = self.remove_weird_characters(text)

            text = self.remove_extra_whitespaces(text)
        except Exception:
            print(text)
        return text


    def remove_weird_characters(self, text):
        text = self.weird_chars_regex.sub('', text)
        return text

    def remove_extra_whitespaces(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text
