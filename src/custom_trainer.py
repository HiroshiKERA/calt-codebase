import torch
import numpy as np
from calt import Trainer


class CustomTrainer(Trainer):
    """
    Trainer class in CALT is based on the HuggingFace Trainer class. 
    Refer to the (official documentation)[https://huggingface.co/docs/transformers/en/main_classes/trainer] of HuggingFace Trainer class to see methods to override.

    Below are the methods that are typically overridden.
    - compute_loss
    - compute_metrics
    - evaluate
    - evaluate_and_save_generation  (particular to CALT)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = (
            kwargs["compute_metrics"]
            if "compute_metrics" in kwargs
            else self._compute_metrics
        )
        self.loss_weight = kwargs["loss_weight"] if "loss_weight" in kwargs else 0.1

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        ignore_index=-100,
        num_items_in_batch=None,
    ):
        """
        This method is called at each iteration of training.
        The default implementation is to compute the loss of the model.

        Args:
            model: the model to train
            inputs: the inputs to the model (e.g., input_ids, attention_mask, labels)
            return_outputs: whether to return the outputs of the model
            ignore_index: the index of the ignore token
            num_items_in_batch: the number of items in the batch
        """

        outputs = model(**inputs)  # outputs.loss is the loss of the model.

        # your custom loss (define compute_custom_loss)
        loss = self._custom_loss_fn(outputs, inputs, self.loss_weight)

        return (loss, outputs) if return_outputs else loss

    def _custom_loss_fn(self, outputs, inputs, r, ignore_index=-100):
        """
        This method is called to compute the custom loss.
        """

        base_loss = outputs.loss

        valid_mask = inputs["labels"] != ignore_index
        extra_loss = 1 / (outputs.logits[valid_mask].var() + 1e-6)

        return base_loss + r * extra_loss  # encourage the variance of the logits

    def _compute_metrics(self, eval_preds, ignore_index=-100):
        """This method is called at each prediction step to compute the metrics.

        Parameters
        ----------
        eval_preds: tuple (predictions, labels)
            predictions: shape (batch_size, seq_len)
            labels: shape (batch_size, seq_len)

        Returns
        -------
        dict with accuracy
        """
        predictions, labels = eval_preds

        # Convert to tensors since inputs are often numpy arrays
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        # Mask tokens with ignore_index
        mask = labels != ignore_index
        correct = (predictions == labels) & mask
        acc = correct.sum().item() / mask.sum().item()

        return {"token_accuracy": acc}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        This method is called at the end of training.
        The default implementation is to compute the metrics for test data.
        """

        # currently using the default implementation in Trainer class
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def evaluate_and_save_generation(self, max_length: int = 512):
        """
        This method is called after training is finished in training script (not internally in Trainer class).
        The default implementation is to generate the evaluation results.
        This is particular to Trainer classes in CALT, and NOT a HuggingFace Trainer.
        """
        # currently using the default implementation in Trainer class
        return super().evaluate_and_save_generation(max_length)
