from torch.nn import CrossEntropyLoss
import os
import json
import torch
import wandb

from calt import PolynomialTrainer
from calt.data_loader.utils.preprocessor import SymbolicToInternalProcessor


class PolynomialTrainerPlus(PolynomialTrainer):
    '''
    PolynomialTrainer class is based on the HuggingFace Trainer class. 
    Refer to the (official documentation)[https://huggingface.co/docs/transformers/en/main_classes/trainer] of HuggingFace Trainer class to see methods to override.
    
    Below are the methods that are typically overridden.
    - compute_loss
    - log_metrics
    - generate_evaluation  (particular to CALT)
    - evaluate
    '''
    
    def compute_loss(self, model, inputs, return_outputs=False, ignore_index=-100, num_items_in_batch=None):
        '''
        This method is called at each iteration of training. 
        The default implementation is to compute the loss of the model.
        
        Args:
            model: the model to train
            inputs: the inputs to the model (e.g., input_ids, attention_mask, labels)
            return_outputs: whether to return the outputs of the model
            ignore_index: the index of the ignore token
            num_items_in_batch: the number of items in the batch
        '''
        
        outputs = model(**inputs)  # outputs.loss is the loss of the model.
        
        ## standard loss
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
                
        # your custom loss (define compute_custom_loss)
        loss = self.compute_custom_loss(outputs, inputs) 
        
        # log metrics
        self.log_metrics(outputs, inputs, model, ignore_index)

        return (loss, outputs) if return_outputs else loss
    
    
    def log_metrics(self, outputs, inputs, model, ignore_index=-100):

        if not self.is_world_process_zero():
            return
        
        ## multi GPUs
        loss_value = outputs.loss.mean().item() if outputs.loss is not None else 0.0
        metrics = {"train/loss": loss_value}
        
        # Metric 1: average of parameter weights
        with torch.no_grad():
            param_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.requires_grad:
                    param_norm += torch.norm(param).item() ** 2
                    param_count += param.numel()
            
            if param_count > 0:
                avg_param_norm = (param_norm / param_count) ** 0.5
                metrics["train/avg_param_norm"] = avg_param_norm

        # Metric 2: classification error rate
        if (outputs.logits is not None and 'labels' in inputs):
            
            labels = inputs['labels']
            logits = outputs.logits
            
            valid_mask = labels != ignore_index
            valid_labels = labels[valid_mask]
            valid_logits = logits[valid_mask]
            
            if len(valid_labels) > 0:
                predictions = torch.argmax(valid_logits, dim=-1)
                metrics["train/tl_error_rate"] = (
                    (predictions != valid_labels).float().mean().item()
                )

        # Metric 3: GPU memory usage
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2

        # Add to log history
        self.log_history.append(metrics)

        metrics["gpu_memory_used_MB"] = gpu_memory_allocated
        metrics["gpu_memory_reserved_MB"] = gpu_memory_reserved

        wandb.log(metrics)
        

    def generate_evaluation(self, 
                            tokenizer, 
                            max_length: int = 512):
        '''
        This method is called at the end of training.
        The default implementation is to generate the evaluation results. 
        This is particular to Trainer classes in CALT, and NOT a HuggingFace Trainer.
        '''
        
        pass
        

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        '''
        This method is called at the end of training.
        The default implementation is to compute the metrics for test data.
        '''
        pass
    
