# -*- coding: utf-8 -*-
import copy
import os
import random
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, DataCollatorWithPadding, TrainingArguments, \
	TrainerCallback, TrainerState, TrainerControl
from transformers.modeling_outputs import ModelOutput
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

from abs_bm25_sampler import AdaptiveBatchSampler
from metrics import ComputeMetrics
from trainer import DenseTrainer


@dataclass
class DenseOutput(ModelOutput):
	q_reps: Tensor = None
	p_reps: Tensor = None
	loss: Tensor = None
	scores: Tensor = None


class CondenserLTR(nn.Module):
	def __init__(self, q_enc: PreTrainedModel, p_enc: PreTrainedModel):
		super().__init__()
		self.q_enc = q_enc
		self.p_enc = p_enc
		self.loss = nn.CrossEntropyLoss()

	def encode_query(self, query):
		q_out = self.q_enc(**query, return_dict=True, output_hidden_states=True)
		q_hidden = q_out.hidden_states
		q_reps = (q_hidden[0][:, 0] + q_hidden[-1][:, 0]) / 2
		return q_reps

	def encode_passage(self, passage):
		p_out = self.p_enc(**passage, return_dict=True, output_hidden_states=True)
		p_hidden = p_out.hidden_states
		p_reps = (p_hidden[0][:, 0] + p_hidden[-1][:, 0]) / 2
		return p_reps

	def forward(self, query: Tensor, passage: Tensor, labels: Tensor):
		# Encode queries and passages
		q_reps = self.encode_query(query)
		p_reps = self.encode_passage(passage)

		# Prepare LTR model input
		# batch_size = q_reps.size(0)
		# q_idx_map = sum(map(lambda x: [x] * self.psg_per_qry, range(batch_size)), [])
		# ltr_input = torch.cat([p_reps, q_reps[q_idx_map]], dim=1)
		# ltr_input = ltr_input.view(batch_size, -1, q_reps.size(1) + p_reps.size(1))

		# Run LTR model
		# scores, loss = self.ltr(ltr_input, labels)

		# Contractive loss
		batch_size = q_reps.size(0)
		psg_per_qry = int(p_reps.size(0) / q_reps.size(0))
		q_idx_map = sum(map(lambda x: [x] * psg_per_qry, range(batch_size)), [])
		scores = q_reps[q_idx_map] * p_reps
		scores = torch.sum(scores, dim=1).view(batch_size, -1)
		loss = self.loss(scores, labels)

		# hidden loss is a hack to prevent trainer to filter it out
		return DenseOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)


class ListNet(nn.Module):
	def __init__(self, input_dim) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_dim, 1),
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


class EvalCollactor(DataCollatorWithPadding):
	q_max_len: int = 32
	p_max_len: int = 128

	def __call__(self, feature):
		queries = [x['query'] for x in feature]
		if isinstance(feature[0]['passage'], list):
			passages = [y for x in feature for y in x['passage']]
		else:
			passages = [x['passage'] for x in feature]
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
			max_length=self.p_max_len,
			padding=True,
			return_tensors="pt",
		)
		return {'query': queries, 'passage': passages, 'labels': labels}


class ABSCollactor(DataCollatorWithPadding):
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


class DevSetPreprocessor:
	def __init__(self):
		self.rand = random.Random()

	def __call__(self, data):
		labels = [1] + [0] * (len(data['passage']) - 1)
		swap_idx = self.rand.randint(0, len(data['passage']) - 1)
		labels[0], labels[swap_idx] = labels[swap_idx], labels[0]
		data['passage'][0], data['passage'][swap_idx] = data['passage'][swap_idx], data['passage'][0]
		data['labels'] = labels
		return data


class TrainSetPreprocessor:
	def __call__(self, data, idx):
		data['id'] = idx
		return data


class ABSCallBack(TrainerCallback):
	def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
		kwargs['train_dataloader'].batch_sampler.save_checkpoint(checkpoint_dir)

	def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		if state.global_step != state.max_steps:
			kwargs['train_dataloader'].batch_sampler.reset()

	def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		torch.cuda.empty_cache()


if __name__ == '__main__':
	train_set = load_from_disk("Dataset/sample_1000_raw").map(TrainSetPreprocessor(), with_indices=True)
	dev_set = load_dataset("json", data_files="Dataset/sample_1000_abs.json", split='train').map(DevSetPreprocessor())

	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = copy.deepcopy(q_enc)
	LTR = ListNet(input_dim=768 * 2)
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc)
	abs_sampler = AdaptiveBatchSampler(dataset=train_set, tokenizer=tokenizer)

	training_args = TrainingArguments("model_output",
	                                  overwrite_output_dir=True,
	                                  learning_rate=5e-6,
	                                  num_train_epochs=3,
	                                  per_device_train_batch_size=8,
	                                  evaluation_strategy='steps',
	                                  save_strategy="steps",
	                                  save_total_limit=10,
	                                  logging_steps=10,
	                                  eval_steps=500,
	                                  save_steps=500,
	                                  load_best_model_at_end=True,
	                                  metric_for_best_model="mmr",
	                                  lr_scheduler_type="cosine",
	                                  warmup_steps=1,
	                                  remove_unused_columns=False)

	trainer = DenseTrainer(
		model=model,
		args=training_args,
		train_dataset=train_set,
		eval_dataset=dev_set,
		abs_sampler=abs_sampler,
		abs_collator=ABSCollactor(p_per_q=8, tokenizer=tokenizer),
		data_collator=EvalCollactor(tokenizer=tokenizer),
		tokenizer=tokenizer,
		compute_metrics=ComputeMetrics(),
	)

	trainer.add_callback(ABSCallBack())
	trainer.train()
	trainer.save_model()
