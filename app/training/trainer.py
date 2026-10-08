import logging
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def apply_lora(self, rank: int = 8, alpha: int = 16):
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA applied to model")

    def train(
        self,
        train_dataset,
        output_dir: str | Path = "./model_checkpoints",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ):
        """Fine-tune the model."""
        if self.model is None:
            self.load_model()

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed")

    def save(self, path: str | Path):
        """Save the fine-tuned model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
