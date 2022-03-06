import os

import numpy as np
import torch
from datasets import load_from_disk, Dataset
from torch.utils.data import Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from abs3 import BM25, SimilarityCache


class AdaptiveBatchSampler(Sampler[int]):
	def __init__(self,
	             dataset: Dataset,
	             tokenizer: PreTrainedTokenizerBase,
	             batch_size=8,
	             resume_from_checkpoint=None):

		super().__init__(data_source=dataset)

		self.dataset = dataset
		self.batch_size = batch_size
		self.tokenizer = tokenizer

		self.sim = BM25(dataset['positives'], tokenizer=tokenizer)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.query = self.tokenizer(dataset['query'], return_tensors="pt", padding=True, max_length=32,
		                            return_attention_mask=False, add_special_tokens=False, truncation=True,
		                            return_token_type_ids=False).input_ids.to(self.device)

		if resume_from_checkpoint:
			self.U = np.load(os.path.join(resume_from_checkpoint, "U.npy"))
			self.T = [x.tolist() for x in np.load(os.path.join(resume_from_checkpoint, "T.npy"))]
			self.hardness_log = np.loadtxt(os.path.join(resume_from_checkpoint, "hardness.txt")).tolist()
		else:
			self.hardness_log = []
			self.U = np.arange(len(dataset), dtype=np.int32)
			self.T = []

		# Find passages that are answers to multiple questions
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

		self.dup_set = {}
		for dup_set in dup_p.values():
			for idx in dup_set:
				self.dup_set[idx] = dup_set - {idx}

	@staticmethod
	def get_remove(cache):
		sim_arr = cache.get()
		diff = torch.sum(sim_arr, dim=1) + torch.sum(sim_arr, dim=0)
		min_idx = torch.argmin(diff)
		pair_remove, sub = cache.qid[min_idx], diff[min_idx]
		return pair_remove, sub

	@staticmethod
	def get_add(current_dataset, pair_remove, remain_dataset, cache_b_u=None, cache_u_b=None):
		s1 = cache_u_b.get().sum(axis=1)
		s1 -= cache_u_b.get_passage_sim(pair_remove)
		idx = [cache_u_b.qid2rowid[x] for x in current_dataset]
		s1[idx] = -np.inf

		s2 = cache_b_u.get().sum(axis=0)
		s2 -= cache_b_u.get_query_sim(pair_remove)
		idx = [cache_b_u.pid2colid[x] for x in current_dataset]
		s2[idx] = -np.inf

		diff = s1 + s2
		max_idx = torch.argmax(diff)
		pair_add, add = remain_dataset[max_idx], diff[max_idx]

		return pair_add, add

	def save_checkpoint(self, checkpoint_dir):
		os.makedirs(checkpoint_dir, exist_ok=True)
		np.save(os.path.join(checkpoint_dir, "U.npy"), self.U)
		np.save(os.path.join(checkpoint_dir, "T.npy"), np.vstack(self.T))
		np.savetxt(os.path.join(checkpoint_dir, "hardness.txt"), np.array(self.hardness_log))

	def reset(self):
		pass

	def __iter__(self):
		yield from self.T

		total = len(self.dataset) // self.batch_size - len(self.T)
		for i in range(total):
			B = np.random.choice(self.U, size=self.batch_size, replace=False)
			cache_B = SimilarityCache(self.sim, self.query, self.dup_set, B, B)
			cache_B_U = SimilarityCache(self.sim, self.query, self.dup_set, B, self.U)
			cache_U_B = SimilarityCache(self.sim, self.query, self.dup_set, self.U, B)
			hardness_B = cache_B.get().sum()
			while True:
				d_r, sub = self.get_remove(cache_B)
				d_a, add = self.get_add(B, d_r, self.U, cache_b_u=cache_B_U, cache_u_b=cache_U_B)
				hardness_B_tem = hardness_B - sub + add
				if hardness_B_tem > hardness_B:
					B = np.setdiff1d(B, d_r)
					B = np.append(B, d_a)
					hardness_B = hardness_B_tem
					cache_B.swap_query(d_r, d_a)
					cache_B.swap_passage(d_r, d_a)
					cache_B_U.swap_query(d_r, d_a)
					cache_U_B.swap_passage(d_r, d_a)
				else:
					break
			self.T.append(B.tolist())
			self.U = np.setdiff1d(self.U, B)
			self.hardness_log.append(hardness_B.item())
			yield B.tolist()

	def __len__(self):
		return len(self.dataset) // self.batch_size


if __name__ == '__main__':
	samples = load_from_disk("Dataset/sample_1000_raw")
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	ABS = AdaptiveBatchSampler(dataset=samples, tokenizer=tokenizer)

	it = iter(ABS)
	for batch in range(len(ABS) - 50):
		print(next(it))

	ABS.save_checkpoint("abs_output/checkpoint_test")

	ABS2 = AdaptiveBatchSampler(dataset=samples, tokenizer=tokenizer,
	                            resume_from_checkpoint="abs_output/checkpoint_test")
	for batch in ABS2:
		print(batch)
	ABS2.reset()
