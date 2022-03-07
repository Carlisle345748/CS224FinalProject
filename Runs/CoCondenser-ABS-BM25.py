import copy

import torch
from datasets import load_dataset
from torch import Tensor, nn
from transformers import PreTrainedModel, AutoTokenizer, AutoModel, TrainingArguments

from ABS.bm25_sampler import AdaptiveBatchSampler
from ABS.trainer import DenseTrainer
from ABS.util import ComputeMetrics, EvalCollactor, DenseOutput, ABSCallBack, ABSCollactor


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


if __name__ == '__main__':
	train_set = load_dataset("Carlisle/msmarco-passage-abs", split='train')
	dev_set = load_dataset("Carlisle/msmarco-passage-non-abs", split='dev')  # dev set is stored in non-abs dataset
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = copy.deepcopy(q_enc)
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc)

	abs_sampler = AdaptiveBatchSampler(dataset=train_set, tokenizer=tokenizer)

	training_args = TrainingArguments("model_output/ABS_BM25",
	                                  overwrite_output_dir=True,
	                                  learning_rate=5e-6,
	                                  num_train_epochs=3,
	                                  per_device_train_batch_size=8,
	                                  evaluation_strategy='steps',
	                                  save_strategy="steps",
	                                  save_total_limit=3,
	                                  logging_steps=10,
	                                  eval_steps=1000,
	                                  save_steps=1000,
	                                  lr_scheduler_type="cosine",
	                                  warmup_steps=len(train_set),
	                                  load_best_model_at_end=True,
	                                  metric_for_best_model="mmr",
	                                  remove_unused_columns=False)

	trainer = DenseTrainer(
		model=model,
		args=training_args,
		train_dataset=train_set,
		eval_dataset=dev_set,
		data_collator=EvalCollactor(tokenizer=tokenizer),
		abs_sampler=abs_sampler,
		abs_collator=ABSCollactor(tokenizer),
		tokenizer=tokenizer,
		compute_metrics=ComputeMetrics(),
	)

	trainer.add_callback(ABSCallBack())
	trainer.train()
	trainer.save_model()
