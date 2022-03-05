import faiss
import numpy as np
import torch


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