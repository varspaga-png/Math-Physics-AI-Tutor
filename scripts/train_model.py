#!/usr/bin/env python
import logging
import argparse
from pathlib import Path
from app.training.data_processor import TrainingDataProcessor
from app.training.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on math/physics data")
    parser.add_argument("data_file", help="Path to training data (JSONL or CSV)")
    parser.add_argument(
        "--model",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--output",
        default="./fine_tuned_model",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading training data from {args.data_file}")
    if args.data_file.endswith(".jsonl"):
        examples = TrainingDataProcessor.load_jsonl(args.data_file)
    elif args.data_file.endswith(".csv"):
        examples = TrainingDataProcessor.load_csv(args.data_file)
    else:
        raise ValueError("Unsupported file format. Use .jsonl or .csv")

    logger.info(f"Loaded {len(examples)} examples")

    # Create dataset
    dataset_dict = TrainingDataProcessor.create_dataset(examples)
    logger.info(f"Train: {len(dataset_dict['train'])}, Val: {len(dataset_dict['validation'])}, Test: {len(dataset_dict['test'])}")

    # Load model and tokenizer
    trainer = ModelTrainer(model_name=args.model)
    trainer.load_model()
    trainer.apply_lora()

    # Prepare training data
    train_dataset = TrainingDataProcessor.prepare_training_examples(
        dataset_dict["train"],
        trainer.tokenizer,
    )

    # Train
    trainer.train(
        train_dataset=train_dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Save
    trainer.save(args.output)
    logger.info(f"Fine-tuned model saved to {args.output}")


if __name__ == "__main__":
    main()
