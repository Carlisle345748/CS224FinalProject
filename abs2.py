import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel


class AdaptiveBatchSampling:
	def __init__(self, dataset, similarity, encode=False):
		self.encode = encode
		self.data = dataset
		self.sim_cache = similarity
		np.fill_diagonal(self.sim_cache, 0)

		if encode:
			self.tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
			self.model = AutoModel.from_pretrained('Luyu/co-condenser-marco')

	def get_hardness(self, hardness, dataset, d_remove=None, d_add=None):
		if d_remove is None and d_add is None:
			hardness = self.subarray(dataset, dataset).sum()
		elif d_remove is not None and d_add is None:
			hardness -= self.sim_cache[d_remove, dataset].sum()
			hardness -= self.sim_cache[dataset, d_remove].sum()
		elif d_remove is None and d_add is not None:
			hardness += self.sim_cache[d_add, dataset].sum()
			hardness += self.sim_cache[dataset, d_add].sum()
		elif d_remove is not None and d_add is not None:
			hardness -= self.sim_cache[d_remove, dataset].sum()
			hardness -= self.sim_cache[dataset, d_remove].sum()
			dataset = np.setdiff1d(dataset, d_remove)
			hardness += self.sim_cache[d_add, dataset].sum()
			hardness += self.sim_cache[dataset, d_add].sum()
		return hardness

	def get_remove(self, dataset):
		sub_arr = self.subarray(dataset, dataset)
		diff = np.sum(sub_arr, axis=1)
		np.add(diff, np.sum(sub_arr, axis=0), out=diff)
		pair_remove = dataset[np.argmin(diff)]
		return pair_remove

	def subarray(self, row, col):
		# prevent create huge array in self.sim_cache[:, col] or self.sim_cache[row, :]
		if len(row) > len(col):
			return np.take(self.sim_cache[:, col], row, axis=0)
		else:
			return np.take(self.sim_cache[row, :], col, axis=1)

	def get_add(self, current_dataset, pair_remove, remain_dataset):
		remain = np.setdiff1d(remain_dataset, current_dataset)
		temp = np.setdiff1d(current_dataset, pair_remove)
		diff = np.sum(self.subarray(remain, temp), axis=1)
		np.add(diff, np.sum(self.subarray(temp, remain), axis=0), out=diff)
		pair_add = remain[np.argmax(diff)]
		return pair_add

	def reformat(self, dataset):
		B = []
		for q in dataset:
			q = int(q)
			query_text = self.data[q]['query']
			passages = []
			for p in dataset:
				p = int(p)
				if p != q:
					passages.append(self.data[p]['positives'])
			passages = [self.data[q]['positives']] + passages
			B.append([query_text, passages])
		return B

	def solve(self, batch_size, output_file=None):
		U = np.arange(len(self.data), dtype=np.int32)
		T = []
		while U.shape[0] > 0:
			B = U[np.random.randint(low=0, high=len(U), size=batch_size)]
			if U.shape[0] > batch_size:
				hardness_B = self.get_hardness(0, B)
				while True:
					d_r = self.get_remove(B)
					d_a = self.get_add(B, d_r, U)
					hardness_B_tem = self.get_hardness(hardness_B, B, d_r, d_a)
					if hardness_B_tem > hardness_B:
						B = np.setdiff1d(B, d_r)
						B = np.append(B, d_a)
						hardness_B = hardness_B_tem
					else:
						break

			U = np.setdiff1d(U, B)
			batch = self.reformat(B)
			print(batch)
			T.extend(batch)

		if output_file:
			df = pd.DataFrame.from_records(T)
			df.columns = ['query', 'passage']
			df.to_json(output_file, orient="records", lines=True)

		return T


if __name__ == '__main__':
	sim = np.load("Similarity/sim_1000.npz")['arr_0']
	n = np.count_nonzero(sim)
	samples = load_from_disk("Dataset/sample_50000_raw")
	ABS = AdaptiveBatchSampling(samples, encode=False, similarity=sim)
	batches = ABS.solve(8, output_file="Dataset/sample_1000_abs.json")
