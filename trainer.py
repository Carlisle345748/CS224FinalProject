from typing import Any, Dict, Union

import torch
import torch.nn as nn
from transformers.file_utils import is_apex_available
from transformers.trainer import Trainer, is_sagemaker_mp_enabled
from transformers import EvalPrediction
from transformers.trainer_pt_utils import nested_numpify, nested_detach

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward


class DenseTrainer(Trainer):
    def __init__(self, compute_train_metric=None, **kwargs):
        super(DenseTrainer, self).__init__( **kwargs)
        self.compute_train_metric = compute_train_metric

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            labels = nested_numpify(nested_detach(inputs['labels']))
            outputs = tuple(v for k, v in outputs.items() if k != 'loss')
            outputs = nested_numpify(nested_detach(outputs))
            self.compute_train_metric(EvalPrediction(predictions=outputs, label_ids=labels))

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()
