import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np  # noqa: F401
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from calt.data_loader.utils.preprocessor import AbstractPreprocessor
from calt.data_loader.utils.data_collator import (
    StandardDataset,
    StandardDataCollator,
)


@dataclass
class ScriptConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str]
    train_file: str
    eval_file: Optional[str]
    max_length: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    output_dir: str
    seed: int
    fp16: bool


def parse_args() -> ScriptConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model with HF Trainer using StandardDataset, tokenizer, "
            "and data_collator."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Tokenizer name or path (defaults to model_name_or_path)",
    )
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/hf-trainer",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
    return ScriptConfig(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        seed=args.seed,
        fp16=bool(args.fp16),
    )


class IdentityPreprocessor(AbstractPreprocessor):
    """No-op preprocessor: returns input as-is for both encode/decode."""

    def __init__(self) -> None:
        super().__init__(num_variables=0, max_degree=0, max_coeff=1)

    def encode(self, text: str) -> str:  # type: ignore[override]
        return text

    def decode(self, tokens: str) -> str:  # type: ignore[override]
        return tokens


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    # Build datasets using StandardDataset with an identity preprocessor
    preprocessor = IdentityPreprocessor()
    train_dataset = StandardDataset.load_file(cfg.train_file, preprocessor=preprocessor)
    eval_dataset = (
        StandardDataset.load_file(cfg.eval_file, preprocessor=preprocessor)
        if cfg.eval_file
        else None
    )

    tokenizer_name = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Seq2Seq model works with StandardDataCollator outputs
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)

    data_collator = StandardDataCollator(tokenizer)

    evaluation_strategy = "no" if eval_dataset is None else "epoch"

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        eval_strategy=evaluation_strategy,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        label_names=["labels"],
        load_best_model_at_end=(evaluation_strategy != "no"),
        report_to=[],
        fp16=cfg.fp16,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    trainer.train()
    if eval_dataset is not None:
        _ = trainer.evaluate()
    trainer.save_model(cfg.output_dir)
    if trainer.args.should_save:
        tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
