import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer


class Similarity:
	def __init__(self, q_enc: PreTrainedModel, p_enc: PreTrainedModel, tokenizer, dataset, cache_file=None):
		self.dataset = dataset
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.q_enc = q_enc.to(self.device)
		self.p_enc = p_enc.to(self.device)
		self.tokenizer = tokenizer

		if cache_file is not None:
			cache = np.load(cache_file)
			self.query = torch.from_numpy(cache['query']).to(self.device)
			self.passage = torch.from_numpy(cache['passage']).to(self.device)
		else:
			self.query = torch.zeros(len(dataset), 768, dtype=torch.float32).to(self.device)
			self.passage = torch.zeros(768, len(dataset), dtype=torch.float32).to(self.device)
			queries = dataset['query']
			passages = dataset['positives']

			self.q_enc.eval()
			self.p_enc.eval()
			with torch.no_grad():
				step = 200
				for i in tqdm(range(0, len(samples), step), desc="Encode query and passage"):
					start, end = i, min(i + step, len(samples))
					q = self.tokenizer(queries[start:end], return_tensors="pt", padding=True,
					                   truncation=True, return_token_type_ids=False, max_length=32).to(self.device)
					p = self.tokenizer(passages[start:end], return_tensors="pt", padding=True,
					                   truncation=True, return_token_type_ids=False, max_length=128).to(self.device)
					self.query[start:end] = self.q_enc(**q).last_hidden_state[:, 0]
					self.passage[:, start:end] = self.p_enc(**p).last_hidden_state[:, 0].T

	def get_sim(self, qid=None, pid=None):
		if qid is not None and pid is not None:
			return self.query[qid] @ self.passage[:, pid]
		elif qid is not None and pid is None:
			return self.query[qid] @ self.passage
		elif qid is None and pid is not None:
			return self.query @ self.passage[:, pid]
		else:
			return None

	def save(self, file: str):
		query = self.query.cpu().detach().numpy()
		passage = self.passage.cpu().detach().numpy()
		np.savez_compressed(file, query=query, passage=passage)


class SimilarityCache:
	def __init__(self, sim: Similarity, qid, pid):
		self.sim = sim
		self.qid = qid
		self.pid = pid
		self.qid2rowid = {qid[i]: i for i in range(len(qid))}
		self.pid2colid = {pid[i]: i for i in range(len(pid))}
		self.cache = sim.get_sim(qid, pid)

		if len(qid) <= len(pid):
			for i in self.qid:
				if i in self.pid2colid:
					self.cache[self.qid2rowid[i], self.pid2colid[i]] = 0
		else:
			for i in self.pid:
				if i in self.qid2rowid:
					self.cache[self.qid2rowid[i], self.pid2colid[i]] = 0

	def get(self):
		return self.cache

	def swap_query(self, old_qid, new_qid):
		new_score = self.sim.get_sim([new_qid], self.pid)
		if new_qid in self.pid:
			new_score[0, self.pid2colid[new_qid]] = 0
		row_idx = self.qid2rowid[old_qid]
		self.cache[row_idx] = new_score
		self.qid[row_idx] = new_qid
		del self.qid2rowid[old_qid]
		self.qid2rowid[new_qid] = row_idx

	def swap_passage(self, old_pid, new_pid):
		new_score = self.sim.get_sim(self.qid, [new_pid]).flatten()
		if new_pid in self.qid2rowid:
			new_score[self.qid2rowid[new_pid]] = 0
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
	def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, sim: Similarity):
		self.dataset = dataset
		self.tokenizer = tokenizer
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.query = self.tokenizer([x['query'] for x in dataset], return_tensors="pt", padding='max_length',
		                            return_attention_mask=False, add_special_tokens=False, truncation=True,
		                            return_token_type_ids=False).input_ids.to(self.device)
		self.sim = sim

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

	def reformat(self, dataset):
		B = []
		for q in dataset:
			q = int(q)
			query_text = self.dataset[q]['query']
			passages = []
			for p in dataset:
				p = int(p)
				if p != q:
					passages.append(self.dataset[p]['positives'])
			passages = [self.dataset[q]['positives']] + passages
			B.append([query_text, passages])
		return B

	def solve(self, batch_size, output_file=None):
		t = tqdm(total=len(self.query))
		U = np.arange(len(self.query), dtype=np.int32)
		T = []
		while U.shape[0] > 0:
			B = U[np.random.randint(low=0, high=len(U), size=batch_size)]
			if U.shape[0] > batch_size:
				cache_B = SimilarityCache(self.sim, B, B)
				cache_B_U = SimilarityCache(self.sim, B, U)
				cache_U_B = SimilarityCache(self.sim, U, B)
				hardness_B = cache_B.get().sum()
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
						print(hardness_B.item())
						break

			U = np.setdiff1d(U, B)
			batch = self.reformat(B)
			T.extend(batch)
			t.update(len(batch))

		if output_file:
			df = pd.DataFrame.from_records(T)
			df.columns = ['query', 'passage']
			df.to_json(output_file, orient="records", lines=True)

		return T


if __name__ == '__main__':
	samples = load_from_disk("Dataset/sample_1000_raw")
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	similarity = Similarity(q_enc=q_enc, p_enc=p_enc, tokenizer=tokenizer,
	                        dataset=samples, cache_file="Dataset/sim_cache_1000.npz")
	ABS = AdaptiveBatchSampling(samples, tokenizer=tokenizer, sim=similarity)
	batches = ABS.solve(8)
