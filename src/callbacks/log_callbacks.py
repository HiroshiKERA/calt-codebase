from transformers import TrainerCallback
import torch
import wandb

class CustomLoggingCallback(TrainerCallback):
    '''
    This callback is used to log custom metrics to wandb.
    '''
    def log_custom_metrics(self, model, prefix, ignore_index=-100, metrics=['avg_param_norm', 'gpu_memory', 'error_rate']):
        custom_metrics = {}
        
        # Metric 1: average of parameter weights
        if 'avg_param_norm' in metrics:
            with torch.no_grad():
                param_norm = 0.0
                param_count = 0
                for param in model.parameters():
                    if param.requires_grad:
                        param_norm += torch.norm(param).item() ** 2
                        param_count += param.numel()
                if param_count > 0:
                    custom_metrics[f"{prefix}/avg_param_norm"] = (param_norm / param_count) ** 0.5

        # Metric 2: GPU memory usage
        if 'gpu_memory' in metrics:
            custom_metrics[f"{prefix}/gpu_memory_used_MB"] = torch.cuda.memory_allocated() / 1024 ** 2
            custom_metrics[f"{prefix}/gpu_memory_reserved_MB"] = torch.cuda.memory_reserved() / 1024 ** 2
        
        # Metric 3: token-level error rate
        if 'error_rate' in metrics:
            if (model.logits is not None and 'labels' in model.inputs):
            
                with torch.no_grad():
                    labels = model.inputs['labels']
                    logits = model.logits
                    
                    valid_mask = labels != ignore_index
                    valid_labels = labels[valid_mask]
                    valid_logits = logits[valid_mask]
                    
                    if len(valid_labels) > 0:
                        predictions = torch.argmax(valid_logits, dim=-1)
                        custom_metrics[f"{prefix}/tl_error_rate"] = (
                            (predictions != valid_labels).float().mean().item()
                        )

        return custom_metrics

    def on_log(self, args, state, control, model=None, **kwargs):
        '''
        This method is called at each logging step.
        '''
        
        if not state.is_world_process_zero:
            return
        metrics = self.log_custom_metrics(model, prefix="train", ignore_index=-100, metrics=['avg_param_norm', 'gpu_memory', 'error_rate'])
        wandb.log(metrics)

    def on_prediction_step(self, args, state, control, model=None, **kwargs):
        '''
        This method is called at each prediction step.
        '''
        if not state.is_world_process_zero:
            return
        metrics = self.log_custom_metrics(model, prefix="eval", ignore_index=-100, metrics=['error_rate'])
        wandb.log(metrics)
