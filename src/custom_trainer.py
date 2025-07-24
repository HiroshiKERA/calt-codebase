from torch.nn import CrossEntropyLoss
import os
import json
import torch
import wandb

from transformers import Trainer
from calt import PolynomialTrainer
from calt.data_loader.utils.preprocessor import SymbolicToInternalProcessor

import numpy as np


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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_metrics = self._compute_metrics
    
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
        # loss = self._custom_loss(outputs, inputs) 
    
        return (loss, outputs) if return_outputs else loss
    

    def _custom_loss(self, outputs, inputs):
        '''
        This method is called to compute the custom loss.
        '''
        raise NotImplementedError("compute_custom_loss is not implemented")
    

    def preprocess_logits_for_metrics(self, logits, labels):
            """
            Args:
                logits: Tensor of shape (batch_size, seq_len, vocab_size)
                labels: Tensor of shape (batch_size, seq_len) — ignored here

            Returns:
                predictions: Tensor of shape (batch_size, seq_len)
            """
            # Extract predicted tokens
            return torch.argmax(logits, dim=-1)

    def _compute_metrics(self, eval_preds, ignore_index=-100):
        """
        Args:
            eval_preds: tuple (predictions, labels)
                - predictions: shape (batch_size, seq_len)
                - labels: shape (batch_size, seq_len)
        
        Returns:
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
    
    # def log_metrics(self, outputs, inputs, ignore_index=-100):
    #     print('agakjfkaemflkaflaknfelk \n\n\n\n\n\n')
        
    #     if not self.is_world_process_zero():
    #         return
        
    #     # Calculate custom metrics
    #     custom_metrics = {}
        
    #     model = self.model
        
    #     # Metric 1: average of parameter weights
    #     with torch.no_grad():
    #         param_norm = 0.0
    #         param_count = 0
    #         for param in model.parameters():
    #             if param.requires_grad:
    #                 param_norm += torch.norm(param).item() ** 2
    #                 param_count += param.numel()
            
    #         if param_count > 0:
    #             avg_param_norm = (param_norm / param_count) ** 0.5
    #             custom_metrics["train/avg_param_norm"] = avg_param_norm

    #     # Metric 2: GPU memory usage
    #     gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
    #     gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
    #     custom_metrics["train/gpu_memory_used_MB"] = gpu_memory_allocated
    #     custom_metrics["train/gpu_memory_reserved_MB"] = gpu_memory_reserved

    #     # Call parent class log_metrics first (this will log parent metrics)
    #     super().log_metrics(outputs, inputs, ignore_index)
        
    #     # Then log custom metrics (this will be a separate log entry but that's OK)
    #     if custom_metrics:
    #         self.log_history.append(custom_metrics)
    #         wandb.log(custom_metrics)
            

    def evaluate_and_save_generation(self, 
                                     max_length: int = 512):
        '''
        This method is called after training is finished in training script (not internally in Trainer class).
        The default implementation is to generate the evaluation results. 
        This is particular to Trainer classes in CALT, and NOT a HuggingFace Trainer.
        '''
        
        return super().generate_evaluation(max_length)
        

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        '''
        This method is called at the end of training.
        The default implementation is to compute the metrics for test data.
        '''

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
