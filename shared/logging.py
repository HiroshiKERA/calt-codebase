"""Custom HuggingFace callback that logs extra metrics to Weights & Biases."""

import torch
import wandb
from transformers import TrainerCallback


class CustomLoggingCallback(TrainerCallback):
    """
    Log gradient norms and GPU memory usage to wandb at each logging step.

    Usage
    -----
    Add to TrainerPipeline after building:

        from shared.logging import CustomLoggingCallback
        trainer_pipeline.trainer.add_callback(CustomLoggingCallback())
    """

    def on_log(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return

        metrics = {}

        # Average L2 norm of trainable parameters
        with torch.no_grad():
            param_norm_sq = sum(
                torch.norm(p).item() ** 2
                for p in model.parameters()
                if p.requires_grad
            )
            param_count = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            if param_count > 0:
                metrics["train/avg_param_norm"] = (param_norm_sq / param_count) ** 0.5

        # GPU memory
        if torch.cuda.is_available():
            metrics["train/gpu_memory_used_MB"] = torch.cuda.memory_allocated() / 1024**2
            metrics["train/gpu_memory_reserved_MB"] = torch.cuda.memory_reserved() / 1024**2

        if metrics:
            wandb.log(metrics)
