import copy
import random
from dataclasses import dataclass

import torch
from datasets import load_from_disk, load_dataset
from torch import Tensor, nn
from transformers import PreTrainedModel, DataCollatorWithPadding, AutoTokenizer, AutoModel, TrainingArguments
from transformers.file_utils import ModelOutput

from CS224FinalProject.ABS.bm25_sampler import AdaptiveBatchSampler
from CS224FinalProject.ABS.trainer import DenseTrainer
from CS224FinalProject.ABS.util import ABSCollactor, ABSCallBack, ComputeMetrics


@dataclass
class DenseOutput(ModelOutput):
	q_reps: Tensor = None
	p_reps: Tensor = None
	loss: Tensor = None
	scores: Tensor = None
	hidden_loss: Tensor = None


class CondenserLTR(nn.Module):
	def __init__(self, q_enc: PreTrainedModel, p_enc: PreTrainedModel, psg_per_qry: int):
		super().__init__()
		self.q_enc = q_enc
		self.p_enc = p_enc
		self.psg_per_qry = psg_per_qry
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

		# Contrastive loss
		batch_size = q_reps.size(0)
		q_idx_map = sum(map(lambda x: [x] * self.psg_per_qry, range(batch_size)), [])
		scores = q_reps[q_idx_map] * p_reps
		scores = torch.sum(scores, dim=1).view(batch_size, -1)
		loss = self.loss(scores, labels)

		# hidden loss is a hack to prevent trainer to filter it out
		return DenseOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)


class EvalCollactor(DataCollatorWithPadding):
	"""
	Input: List[Dict{"query":str, "passage":List[str] | str, "labels": List[int]}]
	Output: Dict{"query": tensor[batch_size, sequence_length],
	             "passage": tensor[batch_size*passage_per_query, sequence_length],
	             "labels": tensor[batch_size, sequence_length]}.

	This step is preparing data for model input.
	"""
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


class DevSetPreprocessor:
	"""
	In original dataset, each data record is organized as {"query":str , "passage":List[str]}.
	This first passage in the passage list is positive sample and the others are negative sample.
	This preprocessor shuffle passage list and add labels to it.

	Input: {"query":str, "passage": List[str]}
	Output: {"query":str, "passage": List[str], "labels":List[int]}
	"""
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
	"""
	This preprocessor adds id to data record.

	Input: ["query":str, "positives": str]
	Output: ["query":str, "positives": str, "id": int]
	"""
	def __call__(self, data, idx):
		data['id'] = idx
		return data


if __name__ == '__main__':
	train_set = load_from_disk("train_1000").map(TrainSetPreprocessor(), with_indices=True)
	dev_set = load_dataset("json", data_files="dev_1000.json", split='train').map(DevSetPreprocessor())

	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = copy.deepcopy(q_enc)
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc, psg_per_qry=8)
	abs_sampler = AdaptiveBatchSampler(dataset=train_set, tokenizer=tokenizer, batch_size=8)

	training_args = TrainingArguments("model_output",
	                                  overwrite_output_dir=True,
	                                  learning_rate=5e-6,
	                                  num_train_epochs=10,
	                                  per_device_train_batch_size=8,
	                                  evaluation_strategy='steps',
	                                  save_strategy="steps",
	                                  save_total_limit=10,
	                                  logging_steps=10,
	                                  eval_steps=500,
	                                  save_steps=500,
	                                  load_best_model_at_end=True,
	                                  metric_for_best_model="mmr",
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

	trainer.add_callback(ABSCallBack())  # Remember to add callback!!!
	trainer.train()
	trainer.save_model()
