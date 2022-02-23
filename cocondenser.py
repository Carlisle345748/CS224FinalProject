# -*- coding: utf-8 -*-

from dataclasses import dataclass
from itertools import product

import torch
import torch.nn as nn
from datasets import load_dataset
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers.modeling_outputs import ModelOutput
from metrics import ComputeMetrics
from trainer import DenseTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None
    hidden_loss: Tensor = None


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

    def forward(self, query: Tensor, passage: Tensor, labels: Tensor):
        # Encode queries and passages
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)

        # Prepare LTR model input
        batch_size = q_reps.size(0)
        q_idx_map = sum(map(lambda x: [x] * self.psg_per_qry, range(batch_size)), [])
        ltr_input = torch.cat([p_reps, q_reps[q_idx_map]], axis=1)
        ltr_input = ltr_input.view(batch_size, -1, q_reps.size(1) + p_reps.size(1))

        # Run LTR model
        scores, loss = self.ltr(ltr_input, labels)
        # hidden loss is a hack to prevent trainer to filter it out
        return DenseOutput(loss=loss, hidden_loss=torch.full((1, 1), loss.item()),
                           scores=scores, q_reps=q_reps, p_reps=p_reps)


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

    def forward(self, x, target):
        scores = self.layers(x).squeeze(-1)

        # ListNet Target.
        # Positive passages have top-one probability 1.
        # Negative passages have top-one probalility 0.

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

    def forward(self, x, target):
        scores = self.layers(x).squeeze(-1)

        # RankNet Target.
        # P_ij is the probanility that i rank higner than j
        # if i > j, P_ij = 1
        # if i == j, P_ij = 0.5
        # if i < j, P_ij = 0
        pairs_candidates = list(product(range(target.size(1)), repeat=2))

        pairs_target = target[:, pairs_candidates]
        pairs_scores = scores[:, pairs_candidates]

        target_diffs = pairs_target[:, :, 0] - pairs_target[:, :, 1]
        target_diffs[target_diffs == 0] = 0.5
        target_diffs[target_diffs == -1] = 0

        scores_diffs = pairs_scores[:, :, 0] - pairs_scores[:, :, 1]

        loss = self.cross_entropy(scores_diffs, target_diffs)
        return scores, loss


class DRCollactor(DataCollatorWithPadding):
    q_max_len: int = 32
    p_max_len: int = 128

    def __call__(self, feature):
        queries = [x['query'] for x in feature]
        passages = [y for x in feature for y in x['passage']]
        labels = torch.tensor([x['labels'] for x in feature], dtype=torch.float32)
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
        return {'query': queries, 'passage': passages, 'labels': labels}


class Preprocessor:
    def __init__(self, split):
        self.split = split
        self.split_map = {"train": 1, "dev": 2, "test": 3}

    def __call__(self, data):
        data['labels'] = [1] + [0] * (len(data['passage']) - 1)
        data['split'] = self.split_map[self.split]
        return data


if __name__ == '__main__':
    dataset = load_dataset("json", data_files="sample_1000_abs.jsonl", split="train")
    train_set = dataset.map(Preprocessor('train'))
    dev_set = dataset.select(range(100)).map(Preprocessor('dev'))

    tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
    encoder = AutoModel.from_pretrained('Luyu/co-condenser-marco')
    collator = DRCollactor(tokenizer=tokenizer)
    LTR = ListNet(input_dim=768*2)
    model = CondenserLTR(q_enc=encoder, p_enc=encoder, ltr=LTR, psg_per_qry=8)

    training_args = TrainingArguments("model_output",
                                      learning_rate=5e-6,
                                      num_train_epochs=3,
                                      per_device_train_batch_size=16,
                                      evaluation_strategy='steps',
                                      eval_steps=100,
                                      save_steps=100,
                                      load_best_model_at_end=True,
                                      metric_for_best_model="MMR",
                                      remove_unused_columns=False)
    trainer = DenseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=ComputeMetrics("dev", "model_metrics"),
        compute_train_metric=ComputeMetrics("train", "model_metrics"),
    )

    trainer.train()
    trainer.save_model()



