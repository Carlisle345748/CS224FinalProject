import os

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

from bm25_torch import BM25


class SimilarityCache:
	def __init__(self, bm25: BM25, query, dup_set, qid, pid):
		self.bm25 = bm25
		self.query = query
		self.dup_set = dup_set
		self.qid = qid
		self.pid = pid
		self.qid2rowid = {qid[i]: i for i in range(len(qid))}
		self.pid2colid = {pid[i]: i for i in range(len(pid))}
		self.cache = bm25.batch_get_score(query[qid], pid, encoded=True)

		if len(qid) <= len(pid):
			for _id in self.qid:
				# Set S_ii = 0
				if _id in self.pid2colid:
					self.cache[self.qid2rowid[_id], self.pid2colid[_id]] = 0
				# If a passage is answer to multiple queries, it should not be considered as negative sample
				# if _id in self.dup_set:
				# 	for dup_id in self.dup_set[_id]:
				# 		if dup_id in self.pid2colid:
				# 			self.cache[self.qid2rowid[_id], self.pid2colid[dup_id]] = 0
		else:
			for _id in self.pid:
				# Set S_ii = 0
				if _id in self.qid2rowid:
					self.cache[self.qid2rowid[_id], self.pid2colid[_id]] = 0
				# If a passage is answer to multiple queries, it should not be considered as negative sample
				# if _id in self.dup_set:
				# 	for dup_id in self.dup_set[_id]:
				# 		if dup_id in self.qid2rowid:
				# 			self.cache[self.qid2rowid[dup_id], self.pid2colid[_id]] = 0

	def get(self):
		return self.cache

	def swap_query(self, old_qid, new_qid):
		new_score = self.bm25.batch_get_score(self.query[new_qid].reshape(1, -1), self.pid, encoded=True)
		if new_qid in self.pid2colid:
			new_score[0, self.pid2colid[new_qid]] = 0
		# if new_qid in self.dup_set:
		# 	for dup_id in self.dup_set[new_qid]:
		# 		if dup_id in self.pid2colid:
		# 			new_score[0, self.pid2colid[dup_id]] = 0

		row_idx = self.qid2rowid[old_qid]
		self.cache[row_idx] = new_score
		self.qid[row_idx] = new_qid
		del self.qid2rowid[old_qid]
		self.qid2rowid[new_qid] = row_idx

	def swap_passage(self, old_pid, new_pid):
		new_score = self.bm25.batch_get_score(self.query[self.qid], [new_pid], encoded=True).flatten()
		if new_pid in self.qid2rowid:
			new_score[self.qid2rowid[new_pid]] = 0
		# if new_pid in self.dup_set:
		# 	for dup_id in self.dup_set[new_pid]:
		# 		if dup_id in self.qid2rowid:
		# 			new_score[self.qid2rowid[dup_id]] = 0

		col_idx = self.pid2colid[old_pid]
		self.cache[:, col_idx] = new_score
		self.pid[col_idx] = new_pid
		del self.pid2colid[old_pid]
		self.pid2colid[new_pid] = col_idx

	def get_query_sim(self, qid):
		return self.cache[self.qid2rowid[qid]]

	def get_passage_sim(self, pid):
		return self.cache[:, self.pid2colid[pid]]


class AdaptiveBatchSampling:
	def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.bm25 = BM25([x['positives'] for x in dataset], tokenizer=self.tokenizer)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.query = self.tokenizer(dataset['query'], return_tensors="pt", padding=True, max_length=32,
		                            return_attention_mask=False, add_special_tokens=False, truncation=True,
		                            return_token_type_ids=False).input_ids.to(self.device)

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

	def save_result(self, T, output_file):
		result = []
		for batch in T:
			for qid in batch:
				qid = int(qid)
				query_text = self.dataset[qid]['query']
				passages = []
				for pid in batch:
					pid = int(pid)
					if pid != qid:
						passages.append(self.dataset[pid]['positives'])
				passages = [self.dataset[qid]['positives']] + passages
				result.append([query_text, passages])
		df = pd.DataFrame.from_records(result)
		df.columns = ['query', 'passage']
		df.to_json(output_file, orient="records", lines=True)

	def solve(self, batch_size, output_dir=None, checkpoint_step=500, resume_from_checkpoint=None):
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)

		total = len(self.query) // batch_size * batch_size
		progress = tqdm(total=total)

		if resume_from_checkpoint is not None:
			U = np.load(os.path.join(resume_from_checkpoint, "U.npy"))
			T = np.load(os.path.join(resume_from_checkpoint, "T.npy"))
			T = [x for x in T]
			hardness_log = np.loadtxt(os.path.join(resume_from_checkpoint, "hardness.txt")).tolist()
			progress.update(len(T)*batch_size)
			last_save_step = len(T)*batch_size
		else:
			hardness_log = []
			last_save_step = 0
			U = np.arange(len(self.query), dtype=np.int32)
			T = []

		while U.shape[0] >= batch_size:
			B = np.random.choice(U, size=batch_size, replace=False)
			cache_B = SimilarityCache(self.bm25, self.query, self.dup_set, B, B)
			hardness_B = cache_B.get().sum()
			if U.shape[0] > batch_size:
				cache_B_U = SimilarityCache(self.bm25, self.query,  self.dup_set, B, U)
				cache_U_B = SimilarityCache(self.bm25, self.query, self.dup_set, U, B)
				while True:
					d_r, sub = self.get_remove(cache_B)
					d_a, add = self.get_add(B, d_r, U, cache_b_u=cache_B_U, cache_u_b=cache_U_B)
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

			U = np.setdiff1d(U, B)
			T.append(B)
			hardness_log.append(hardness_B.item())
			progress.update(batch_size)
			if output_dir is not None and len(T) * batch_size - last_save_step >= checkpoint_step and len(U) != 0:
				checkpoint_dir = os.path.join(output_dir, f'checkpoint_{len(T) * batch_size}')
				os.makedirs(checkpoint_dir, exist_ok=True)
				np.save(os.path.join(checkpoint_dir, "U.npy"), U)
				np.save(os.path.join(checkpoint_dir, "T.npy"), np.vstack(T))
				np.savetxt(os.path.join(checkpoint_dir, "hardness.txt"), np.array(hardness_log))
				last_save_step = len(T) * batch_size

		if output_dir:
			output_file = os.path.join(output_dir, "abs.json")
			hardness_file = os.path.join(output_dir, "hardness.txt")
			np.savetxt(hardness_file, np.array(hardness_log))
			self.save_result(T, output_file)

		return T


if __name__ == '__main__':
	samples = load_from_disk("Dataset/sample_1000_raw")
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	ABS = AdaptiveBatchSampling(samples, tokenizer)
	batches = ABS.solve(8, output_dir="abs_output")
