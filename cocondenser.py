# -*- coding: utf-8 -*-
import copy
import random
from dataclasses import dataclass
from itertools import product

import faiss
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, DataCollatorWithPadding, TrainingArguments
from transformers.modeling_outputs import ModelOutput

from metrics import ComputeMetrics
from trainer import DenseTrainer


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
		ltr_input = torch.cat([p_reps, q_reps[q_idx_map]], dim=1)
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


class PassageIndex:
	def __init__(self, p_reps: np.ndarray):
		self.dim = p_reps.shape[1]
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		index = faiss.index_factory(self.dim, "IDMap,Flat", faiss.METRIC_INNER_PRODUCT)
		if self.device == 'cuda':
			print("Faiss Use GPU")
			self.faiss_res = faiss.StandardGpuResources()
			index = faiss.index_cpu_to_gpu(self.faiss_res, 0, index)
		self.index = index
		self.add(p_reps)

	def search(self, q_reps: np.ndarray, top_k: int):
		if not q_reps.flags.c_contiguous:
			q_reps = np.asarray(q_reps, order="C")
		print(q_reps.shape)
		return self.index.search(x=q_reps, k=top_k)

	def add(self, p_reps: np.ndarray):
		if not p_reps.flags.c_contiguous:
			p_reps = np.asarray(p_reps, order="C")
		start, end = self.index.ntotal, self.index.ntotal + p_reps.shape[0]
		self.index.add_with_ids(p_reps, np.arange(start, end, dtype='int64'))

	def update(self, p_reps: np.ndarray, ids: np.ndarray):
		if not p_reps.flags.c_contiguous:
			p_reps = np.asarray(p_reps, order="C")
		self.index.add_with_ids(p_reps, ids)

	def __len__(self):
		return self.index.ntotal


class FaissScheduler:
	def __init__(self, p_reps=None, dataset=None, cache=None):
		if cache is None and p_reps is None:
			raise "Should provide p_reps or cache"
		if cache:
			p_reps = np.load(cache)['passage']
		self.psg_idx = PassageIndex(p_reps)
		self.is_sampled = np.zeros(len(self.psg_idx), dtype=bool)
		self.exclude = self.build_exclude(dataset)

	@staticmethod
	def build_exclude(dataset):
		dup_p = {}
		passage_set = {}
		for i in range(len(dataset)):
			passage = dataset[i]['positives']
			if passage in passage_set:
				passage_set[passage].add(i)
				if len(passage_set[passage]) > 1:
					dup_p[passage] = passage_set[passage]
			else:
				passage_set[passage] = {i}

		exclude = {}
		for dup_set in dup_p.values():
			for idx in dup_set:
				exclude[idx] = dup_set - {idx}
		return exclude

	def get_batch(self, q_reps, qids, labels, p_per_q=8):
		if isinstance(q_reps, torch.Tensor):
			q_reps = q_reps.detach().cpu().numpy()
		if isinstance(labels, torch.Tensor):
			labels = labels.detach().cpu().numpy()

		# Search until find enough negative samples
		batch = [None] * len(qids)  # Initially only contain positives samples

		valid_counter = 0
		search_range = p_per_q
		qid_to_search = list(qids)
		rowid_to_search = list(range(len(qids)))
		while valid_counter < len(batch):
			search_range += 100
			qid2rowid = {qids[i]: i for i in range(len(qids))}
			finished_qid, finished_rowid = [], []
			_, candidates = self.psg_idx.search(q_reps[rowid_to_search], top_k=search_range)
			for i in range(len(candidates)):
				qid = qid_to_search[i]
				valid_pids = [qid]
				for pid in candidates[i]:
					if pid != qid and (qid not in self.exclude or pid not in self.exclude[qid]):
						valid_pids.append(pid)
						self.is_sampled[pid] = True
						if len(valid_pids) == p_per_q:
							true_idx = np.argmax(labels[qid2rowid[qid]])
							valid_pids[true_idx], valid_pids[0] = valid_pids[0], valid_pids[true_idx]
							batch[qid2rowid[qid]] = valid_pids
							finished_qid.append(qid)
							finished_rowid.append(qid2rowid[qid])
							valid_counter += 1
							break
			for i in range(len(finished_qid)):
				qid_to_search.remove(finished_qid[i])
				rowid_to_search.remove(finished_rowid[i])

		return batch

	def update(self, p_reps, pids):
		if isinstance(p_reps, torch.Tensor):
			p_reps = p_reps.detach().cpu().numpy()
		if not isinstance(pids, np.ndarray):
			pids = np.array(pids, dtype='int64')
		self.psg_idx.update(p_reps=p_reps, ids=pids)

	def unsampled_rate(self):
		return (1 - np.sum(self.is_sampled)) / self.is_sampled.shape[0]


class DRCollactor(DataCollatorWithPadding):
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
		if 'qid' not in feature[0]:
			return {'query': queries, 'passage': passages, 'labels': labels}
		qid = [x['qid'] for x in feature]
		return {'query': queries, 'passage': passages, 'labels': labels, 'qid': qid}


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
	def __init__(self, p_per_q):
		self.rand = random.Random()
		self.batch_size = p_per_q

	def __call__(self, data, idx):
		labels = [1] + [0] * (self.batch_size - 1)
		swap_idx = self.rand.randint(0, self.batch_size - 1)
		labels[0], labels[swap_idx] = labels[swap_idx], labels[0]
		data['labels'] = labels
		return {'query': data['query'], 'passage': data['positives'], 'labels': labels, 'qid': idx}


if __name__ == '__main__':
	train_set = load_from_disk("Dataset/sample_1000_raw")
	train_set = train_set.map(TrainSetPreprocessor(p_per_q=8), with_indices=True)
	dev_set = load_dataset("json", data_files="Dataset/sample_1000_abs.json", split='train')
	dev_set = dev_set.map(DevSetPreprocessor())

	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = copy.deepcopy(q_enc)
	collator = DRCollactor(tokenizer=tokenizer)
	LTR = ListNet(input_dim=768 * 2)
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc, ltr=LTR, psg_per_qry=8)

	batch_scheduler = FaissScheduler(dataset=train_set, cache="Dataset/sim_cache_1000.npz")

	training_args = TrainingArguments("model_output",
	                                  overwrite_output_dir=True,
	                                  learning_rate=5e-6,
	                                  num_train_epochs=10,
	                                  per_device_train_batch_size=16,
	                                  evaluation_strategy='steps',
	                                  save_strategy="steps",
	                                  save_total_limit=10,
	                                  eval_steps=1,
	                                  save_steps=500,
	                                  load_best_model_at_end=True,
	                                  metric_for_best_model="mmr",
	                                  remove_unused_columns=False)
	trainer = DenseTrainer(
		model=model,
		args=training_args,
		train_dataset=train_set,
		eval_dataset=dev_set,
		data_collator=collator,
		batch_scheduler=batch_scheduler,
		tokenizer=tokenizer,
		compute_metrics=ComputeMetrics("dev", "model_metrics"),
		compute_train_metric=ComputeMetrics("train", "model_metrics", save_step=10),
	)

	trainer.train()
	trainer.save_model()
