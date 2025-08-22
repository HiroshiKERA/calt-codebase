import click

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from calt.data_loader.utils.preprocessor import AbstractPreprocessor
from calt.data_loader.utils.data_collator import (
    StandardDataset,
    StandardDataCollator,
)


class IdentityPreprocessor(AbstractPreprocessor):
    """No-op preprocessor: returns input as-is for both encode/decode."""

    def __init__(self) -> None:
        super().__init__(num_variables=0, max_degree=0, max_coeff=1)

    def encode(self, text: str) -> str:  # type: ignore[override]
        return text

    def decode(self, tokens: str) -> str:  # type: ignore[override]
        return tokens


@click.command()
@click.option("--model_name_or_path", type=str, default="distilbert-base-uncased")
@click.option(
    "--tokenizer_name_or_path",
    type=str,
    default=None,
    help="Tokenizer name or path (defaults to model_name_or_path)",
)
@click.option("--train_file", type=str, required=True)
@click.option("--eval_file", type=str, default=None)
@click.option("--max_length", type=int, default=256)
@click.option("--learning_rate", type=float, default=5e-5)
@click.option("--num_train_epochs", type=float, default=3.0)
@click.option("--per_device_train_batch_size", type=int, default=16)
@click.option("--per_device_eval_batch_size", type=int, default=16)
@click.option("--weight_decay", type=float, default=0.01)
@click.option("--output_dir", type=str, default="./outputs/hf-trainer")
@click.option("--seed", type=int, default=42)
@click.option("--fp16", is_flag=True)
def main(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None,
    train_file: str,
    eval_file: str | None,
    max_length: int,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    weight_decay: float,
    output_dir: str,
    seed: int,
    fp16: bool,
) -> None:
    set_seed(seed)

    # Build datasets using StandardDataset with an identity preprocessor
    preprocessor = IdentityPreprocessor()
    train_dataset = StandardDataset.load_file(train_file, preprocessor=preprocessor)
    eval_dataset = (
        StandardDataset.load_file(eval_file, preprocessor=preprocessor)
        if eval_file
        else None
    )

    tokenizer_name = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Seq2Seq model works with StandardDataCollator outputs
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    data_collator = StandardDataCollator(tokenizer)

    evaluation_strategy = "no" if eval_dataset is None else "epoch"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy=evaluation_strategy,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        label_names=["labels"],
        load_best_model_at_end=(evaluation_strategy != "no"),
        remove_unused_columns=False,
        report_to=[],
        fp16=fp16,
        seed=seed,
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
    trainer.save_model(output_dir)
    if trainer.args.should_save:
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
