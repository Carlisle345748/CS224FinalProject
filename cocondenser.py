# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers.modeling_outputs import ModelOutput
from datasets import load_dataset
from dataclasses import dataclass
import torch.nn as nn
from torch import Tensor
import os
import numpy as np
from itertools import product
import torch.nn.functional as F
from sklearn.metrics import ndcg_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class CondenserLTR(nn.Module):
    def __init__(self, q_enc: PreTrainedModel, p_enc: PreTrainedModel, ltr: nn.Module, psg_per_qry: int):
        super().__init__()
        self.q_enc = q_enc
        self.p_enc = p_enc
        self.ltr = ltr
        self.psg_per_qry = psg_per_qry

    def encode_query(self, query):
        q_out = self.q_enc(**query, return_dict=True)
        q_hidden = q_out.last_hidden_state
        q_reps = q_hidden[:, 0]
        return q_hidden, q_reps

    def encode_passage(self, passage):
        p_out = self.p_enc(**passage, return_dict=True)
        p_hidden = p_out.last_hidden_state
        p_reps = p_hidden[:, 0]
        return p_hidden, p_reps

    def forward(self, query: Tensor, passage: Tensor):
        # Encode queries and passages
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)

        # Prepare LTR model input
        batch_size = q_reps.size(0)
        q_idx_map = sum(map(lambda x: [x] * self.psg_per_qry, range(batch_size)), [])
        ltr_input = torch.cat([p_reps, q_reps[q_idx_map]], axis=1)
        ltr_input = ltr_input.view(batch_size, -1, q_reps.size(1) + p_reps.size(1))

        # Run LTR model
        scores, loss = self.ltr(ltr_input)

        return DenseOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

    def save(self, output_dir: str):
        os.makedirs(os.path.join(output_dir, 'query_model'))
        os.makedirs(os.path.join(output_dir, 'passage_model'))
        self.q_enc.save_pretrained(os.path.join(output_dir, 'query_model'))
        self.p_enc.save_pretrained(os.path.join(output_dir, 'passage_model'))
        torch.save(self.ltr.state_dict(), "ltr_model")


class ListNet(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, int(input_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 2), int(input_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 4), 1),
            nn.ReLU()
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        scores = self.layers(x).squeeze(-1)

        # ListNet Target.
        # Positive passages have top-one probability 1.
        # Negative passages have top-one probalility 0.
        target = torch.zeros(x.size(0), x.size(1), device=device)
        target[:, 0] = 1
        target.to(device)

        loss = self.cross_entropy(scores, target)
        return scores, loss


class RankNet(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, int(input_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 2), int(input_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(input_dim / 4), 1),
            nn.ReLU()
        )
        self.cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, x):
        scores = self.layers(x).squeeze(-1)

        # RankNet Target.
        # P_ij is the probanility that i rank higner than j
        # if i > j, P_ij = 1
        # if i == j, P_ij = 0.5
        # if i < j, P_ij = 0
        target = torch.zeros(x.size(0), x.size(1), device=device)
        target.to(device)
        pairs_candidates = list(product(range(target.size(1)), repeat=2))

        pairs_target = target[:, pairs_candidates]
        pairs_scores = scores[:, pairs_candidates]

        target_diffs = pairs_target[:, :, 0] - pairs_target[:, :, 1]
        target_diffs[target_diffs == 0] = 0.5
        target_diffs[target_diffs == -1] = 0

        scores_diffs = pairs_scores[:, :, 0] - pairs_scores[:, :, 1]

        loss = self.cross_entropy(scores_diffs, target_diffs)
        return scores, loss


def MRR(scores):
    """
    scores: [batch_size, num_passages]
    """
    probs = F.softmax(scores, dim=1)
    idx = torch.argmax(probs, axis=1) + 1
    return torch.mean(1 / idx).item()


def NDCG(scores):
    """
    scores: [batch_size, num_passages]
    """
    target = np.array([1] + [0] * (scores.size(1) - 1))
    target = np.tile(target, (scores.size(0), 1))
    probs = F.softmax(scores, dim=1)
    probs = probs.detach().cpu().numpy()
    return ndcg_score(target, probs)


class DRCollactor(DataCollatorWithPadding):
    q_max_len: int = 32
    p_max_len: int = 128

    def __call__(self, feature):
        queries = [x['query'] for x in feature]
        passages = [y for x in feature for y in x['passage']]
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
            max_length=self.q_max_len,
            padding=True,
            return_tensors="pt",
        )
        return {'query': queries, 'passage': passages}


if __name__ == '__main__':
    dataset = load_dataset("json", data_files="sample_1000_abs.jsonl", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
    encoder = AutoModel.from_pretrained('Luyu/co-condenser-marco')
    collator = DRCollactor(tokenizer=tokenizer)
    LTR = ListNet(input_dim=768*2)
    model = CondenserLTR(q_enc=encoder, p_enc=encoder, ltr=LTR, psg_per_qry=8)

    training_args = TrainingArguments("model_output",
                                      learning_rate=5e-6,
                                      num_train_epochs=3,
                                      per_device_train_batch_size=16)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()


