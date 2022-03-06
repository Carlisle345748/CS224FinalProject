import copy
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch import Tensor, nn
from transformers import PreTrainedModel, DataCollatorWithPadding, AutoTokenizer, AutoModel, TrainingArguments
from transformers.file_utils import ModelOutput

from CS224FinalProject.ABS.trainer import DenseTrainer
from CS224FinalProject.ABS.util import ComputeMetrics


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

		# Contrastive loss
		batch_size = q_reps.size(0)
		psg_per_qry = int(p_reps.size(0) / q_reps.size(0))
		q_idx_map = sum(map(lambda x: [x] * psg_per_qry, range(batch_size)), [])
		scores = q_reps[q_idx_map] * p_reps
		scores = torch.sum(scores, dim=1).view(batch_size, -1)
		loss = self.loss(scores, labels)

		# hidden loss is a hack to prevent trainer to filter it out
		return DenseOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)


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


if __name__ == '__main__':
	data = load_dataset("Carlisle/msmarco-passage-non-abs")
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = copy.deepcopy(q_enc)
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc)

	training_args = TrainingArguments("model_output",
	                                  overwrite_output_dir=True,
	                                  learning_rate=5e-6,
	                                  num_train_epochs=2,
	                                  per_device_train_batch_size=8,
	                                  evaluation_strategy='steps',
	                                  save_strategy="steps",
	                                  save_total_limit=10,
	                                  logging_steps=10,
	                                  eval_steps=1000,
	                                  save_steps=1000,
	                                  load_best_model_at_end=True,
	                                  metric_for_best_model="mmr",
	                                  remove_unused_columns=False)

	trainer = DenseTrainer(
		model=model,
		args=training_args,
		train_dataset=data['train'],
		eval_dataset=data['dev'],
		data_collator=EvalCollactor(tokenizer=tokenizer),
		tokenizer=tokenizer,
		compute_metrics=ComputeMetrics(),
	)

	trainer.train()
	trainer.save_model()
