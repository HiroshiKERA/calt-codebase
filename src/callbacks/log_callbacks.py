from transformers import TrainerCallback
import torch
import wandb


class CustomLoggingCallback(TrainerCallback):
    """
    This callback is used to log custom metrics to wandb.
    """

    def log_custom_metrics(
        self, model, prefix, ignore_index=-100, metrics=["avg_param_norm", "gpu_usage"]
    ):
        custom_metrics = {}

        if "avg_param_norm" in metrics:
            with torch.no_grad():
                param_norm = 0.0
                param_count = 0
                for param in model.parameters():
                    if param.requires_grad:
                        param_norm += torch.norm(param).item() ** 2
                        param_count += param.numel()
                if param_count > 0:
                    avg_norm = (param_norm / param_count) ** 0.5
                    custom_metrics[f"{prefix}/avg_param_norm"] = avg_norm

        if "gpu_usage" in metrics and torch.cuda.is_available():
            custom_metrics[f"{prefix}/gpu_memory_used_MB"] = (
                torch.cuda.memory_allocated() / 1024**2
            )
            custom_metrics[f"{prefix}/gpu_memory_reserved_MB"] = (
                torch.cuda.memory_reserved() / 1024**2
            )

        return custom_metrics

    def on_log(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return

        metrics = self.log_custom_metrics(
            model=model,
            prefix="train",
            ignore_index=-100,
            metrics=["avg_param_norm", "gpu_usage"],
        )

        if metrics:
            wandb.log(metrics)

    def on_prediction_step(self, args, state, control, **kwargs):
        """
        Optional: implement logging during prediction if needed.
        """
        pass  # Not used currently
