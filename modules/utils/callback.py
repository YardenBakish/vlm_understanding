
from typing import Optional, Union, Any
from torch import nn
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText ,TrainingArguments, Trainer, TrainerCallback
import torch

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)


class CustomTrainer(Trainer):
     def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):

 
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        
        outputs = model(**inputs)
        self.log({"movement_vectors_loss": outputs.movement_vectors_loss.item()})
        self.log({"cap_loss": outputs.cap_loss.item()})

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if Trainer._is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:

            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss




class LossLoggerCallback2(TrainerCallback):
    def __init__(self, log_file="training_loss.txt"):
        self.log_file = log_file

    def on_log(self, args, state, control, model=None,  logs=None, **kwargs):


        #for name, param in model.named_parameters():
        #    print(name, param.grad is not None, None if param.grad is None else param.grad.norm().item())
        if logs is not None and "loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: loss = {logs['loss']:.4f}\n")

        if logs is not None and "movement_vectors_loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: movement_vectors_loss = {logs['movement_vectors_loss']:.4f}\n")

        if logs is not None and "cap_loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: cap_loss = {logs['cap_loss']:.4f}\n")