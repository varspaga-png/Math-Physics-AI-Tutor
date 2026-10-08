import pytest
from app.training.data_processor import TrainingDataProcessor


def test_create_dataset():
    examples = [
        {"question": "Q1", "answer": "A1", "subject": "math", "difficulty": "beginner"},
        {"question": "Q2", "answer": "A2", "subject": "physics", "difficulty": "intermediate"},
        {"question": "Q3", "answer": "A3", "subject": "math", "difficulty": "advanced"},
    ]
    dataset_dict = TrainingDataProcessor.create_dataset(examples)
    assert "train" in dataset_dict
    assert "validation" in dataset_dict
    assert "test" in dataset_dict
    total = len(dataset_dict["train"]) + len(dataset_dict["validation"]) + len(dataset_dict["test"])
    assert total == 3
