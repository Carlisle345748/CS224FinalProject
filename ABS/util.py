import os
import random

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score
from transformers import DataCollatorWithPadding, TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class ABSCallBack(TrainerCallback):
    """
    TrainerCallback for Adaptive Batch Sampler
    """
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        kwargs['train_dataloader'].batch_sampler.save_checkpoint(checkpoint_dir)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step != state.max_steps:
            kwargs['train_dataloader'].batch_sampler.reset()


class ABSCollactor(DataCollatorWithPadding):
    """
    DataCollator For Adaptive Batch Sampler.
    This sampler must be used together with the provided Adaptive Batch Sampler and DenseTrainer.

    Input: List[Dict{"query":str, "positives" str, "labels": List[int]}]
    Output: Dict{"query": tensor[batch_size, sequence_length],
            "passage": tensor[batch_size*passage_per_query, sequence_length],
            "labels": tensor[batch_size, sequence_length]}.
    """
    q_max_len: int = 32
    p_max_len: int = 128

    def __init__(self, p_per_q, tokenizer):
        self.p_per_q = p_per_q
        self.rand = random.Random()
        self.tokenizer = tokenizer

    def __call__(self, batch):
        labels = torch.zeros(len(batch), self.p_per_q, dtype=torch.float32)
        queries = [x['query'] for x in batch]
        passages = []
        for i in range(len(batch)):
            qid = batch[i]['id']
            psg = [batch[i]['positives']]
            for j in range(len(batch)):
                pid = batch[j]['id']
                if pid != qid:
                    psg.append(batch[j]['positives'])
            true_idx = self.rand.randint(0, self.p_per_q - 1)
            psg[0], psg[true_idx] = psg[true_idx], psg[0]
            passages.extend(psg)
            labels[i, true_idx] = 1

        queries = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.q_max_len,
            padding=True,
            return_tensors="pt",
        )
        passages = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.p_max_len,
            padding=True,
            return_tensors="pt",
        )
        return {'query': queries, 'passage': passages, 'labels': labels}


class ComputeMetrics:
    """
    Compute MMR and NDCG
    """
    def __call__(self, eval_preds):
        output, labels = eval_preds
        scores = output[2]
        mmr = self.MRR(scores, labels)
        ndcg = self.NDCG(scores, labels)
        return {'mmr': mmr, "ndcg": ndcg}

    @staticmethod
    def MRR(scores, target):
        """
        scores: [batch_size, num_passages]
        """
        probs = softmax(scores, axis=1)
        rank = np.apply_along_axis(rankdata, axis=1, arr=-probs)
        idx = np.argmax(target, axis=1).reshape(-1, 1)
        rank_top_idx = np.take_along_axis(rank, idx, axis=1)  # The rank of the top 1 item of target in prediction
        return np.mean(1 / rank_top_idx)

    @staticmethod
    def NDCG(scores, target):
        """
        scores: [batch_size, num_passages]
        """
        probs = softmax(scores, axis=1)
        return ndcg_score(target, probs)
