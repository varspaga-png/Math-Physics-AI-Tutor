import json
from pathlib import Path
from typing import Iterator
from datasets import Dataset, DatasetDict
import pandas as pd


class TrainingDataProcessor:
    @staticmethod
    def load_jsonl(file_path: str | Path) -> list[dict]:
        """Load JSONL training data."""
        data = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    @staticmethod
    def load_csv(file_path: str | Path) -> list[dict]:
        """Load CSV training data."""
        df = pd.read_csv(file_path)
        return df.to_dict("records")

    @staticmethod
    def create_dataset(
        examples: list[dict],
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> DatasetDict:
        """Create a HuggingFace DatasetDict from examples."""
        df = pd.DataFrame(examples)
        dataset = Dataset.from_dict({
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "subject": df.get("subject", ["math"] * len(df)).tolist(),
            "difficulty": df.get("difficulty", ["intermediate"] * len(df)).tolist(),
        })

        # Split data
        train_size = int(len(dataset) * train_split)
        val_size = int(len(dataset) * val_split)
        test_size = len(dataset) - train_size - val_size

        split = dataset.train_test_split(
            train_size=train_size,
            test_size=val_size + test_size,
        )
        val_test = split["test"].train_test_split(
            train_size=val_size,
            test_size=test_size,
        )

        return DatasetDict(
            train=split["train"],
            validation=val_test["train"],
            test=val_test["test"],
        )

    @staticmethod
    def prepare_training_examples(
        dataset: Dataset,
        tokenizer,
        max_length: int = 512,
    ) -> Dataset:
        """Prepare examples for model training."""
        def format_example(example):
            prompt = (
                f"Question: {example['question']}\n"
                f"Subject: {example['subject']}\n"
                f"Difficulty: {example['difficulty']}\n"
                f"Answer: {example['answer']}"
            )
            return {"text": prompt}

        formatted = dataset.map(format_example, remove_columns=dataset.column_names)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        return formatted.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
