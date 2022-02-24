import random

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from gensim.summarization.bm25 import BM25
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool


class AdaptiveBatchSampling:
	def __init__(self, dataset, similarity, encode=False):
		self.encode = encode
		self.data = [(x['query'], x['positives']) for x in dataset]
		passage = list(set([x['positives'] for x in dataset]))
		query = list(set([x['query'] for x in dataset]))
		self.p2idx = {passage[i]: i for i in range(len(passage))}
		self.q2idx = {query[i]: i for i in range(len(query))}
		self.sim_cache = {}
		self.sim_cache2 = similarity

		if encode:
			self.tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
			self.model = AutoModel.from_pretrained('Luyu/co-condenser-marco')
		else:
			self.bm25 = BM25([doc.split(" ") for doc in passage])
			self.avg_ifg = sum(float(val) for val in self.bm25.idf.values()) / len(self.bm25.idf)

	def get_similarity(self, query_text, passage_text):
		if query_text in self.sim_cache and passage_text in self.sim_cache[query_text]:
			return self.sim_cache[query_text][passage_text]

		if self.encode:
			self.model.eval()
			sequences = [query_text, passage_text]
			model_inputs = self.tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
			encoding = self.model(**model_inputs, return_dict=True).last_hidden_state[:, 0]
			e1 = torch.nn.functional.normalize(encoding[0], p=2.0, dim=0)
			e2 = torch.nn.functional.normalize(encoding[1], p=2.0, dim=0)
			similarity = e1 @ e2

		else:
			query_tokens = query_text.split()
			self.sim_cache[query_text] = {}
			similarity = self.bm25.get_score(query_tokens, self.p2idx[passage_text])
			self.sim_cache[query_text][passage_text] = similarity

		return similarity

	def get_hardness(self, hardness, dataset, d_remove=None, d_add=None):
		if d_remove is None and d_add is None:
			# for pair1 in temp:
			# 	for pair2 in temp:
			# 		if pair1 != pair2:
			# 			hardness = hardness + self.get_similarity(pair1[0], pair2[1])
			idx = np.tile(dataset, (len(dataset), 1))
			hardness = np.take_along_axis(self.sim_cache2[dataset, :], idx, axis=1).sum()
		elif d_remove is not None and d_add is not None:
			for pair in dataset:
				if pair != d_remove:
					hardness = hardness - self.get_similarity(pair[0], d_remove[1]) - \
					           self.get_similarity(d_remove[0], pair[1]) + \
					           self.get_similarity(pair[0], d_add[1]) + \
					           self.get_similarity(d_add[0], pair[1])

		elif d_remove is not None and d_add is None:
			for pair in dataset:
				if pair != d_remove:
					hardness = hardness - self.get_similarity(pair[0], d_remove[1]) - \
					           self.get_similarity(d_remove[0], pair[1])

		elif d_remove is None and d_add is not None:
			h_copy = hardness
			for pair in dataset:
				hardness = hardness + self.get_similarity(pair[0], d_add[1]) + self.get_similarity(d_add[0], pair[1])
			# h_copy +=

		return hardness

	def get_remove(self, dataset, hardness):
		max_hardness = float("-inf")
		temp = dataset.copy()
		pair_remove = ()

		for pair in dataset:
			temp_hardness = self.get_hardness(hardness, temp, pair, None)
			if temp_hardness > max_hardness:
				max_hardness = temp_hardness
				pair_remove = pair

		return pair_remove

	def get_add(self, current_dataset, pair_remove, remain_dataset, hardness):
		temp = current_dataset.copy()
		remain = remain_dataset.copy()
		remain = remain - set(temp)
		cur_hardness = self.get_hardness(hardness, temp, pair_remove, None)
		temp.remove(pair_remove)
		pair_add = ()
		max_hardness = float("-inf")

		for pair in remain:
			temp_hardness = self.get_hardness(cur_hardness, temp, None, pair)
			if temp_hardness > max_hardness:
				max_hardness = temp_hardness
				pair_add = pair

		return pair_add

	@staticmethod
	def reformat(dataset):
		B = []
		for pair1 in dataset:
			query = pair1[0]
			current_row = []
			for pair2 in dataset:
				if pair2[0] != query:
					current_row.append(pair2[1])
			current_row = [pair1[1]] + current_row
			B.append([query, current_row])
		return B

	def solve(self, batch_size, output_file=None):
		np.random.seed(12)
		U = np.arange(len(self.data))
		T = []
		while U.shape[0] > 0:
			B = U[np.random.randint(low=0, high=len(U), size=batch_size)]
			if U.shape[0] > batch_size:
				hardness_B = self.get_hardness(0, B)
				while True:
					d_r = self.get_remove(B, hardness_B)
					d_a = self.get_add(B, d_r, U, hardness_B)
					hardness_B_tem = self.get_hardness(hardness_B, B, d_r, d_a)
					if hardness_B_tem > hardness_B:
						B.remove(d_r)
						B.append(d_a)
						hardness_B = hardness_B_tem
					else:
						break

			print(B)
			B = set(B)
			U = U - B
			T.extend(self.reformat(B))

		if output_file:
			df = pd.DataFrame.from_records(T)
			df.columns = ['query', 'passage']
			df.to_json(output_file, orient="records", lines=True)

		return T


# samples = load_from_disk("Dataset/sample_1000_raw")
# passage = [x['positives'] for x in samples]
# bm25 = BM25([doc.split(" ") for doc in passage])
# query = list(set([x['query'] for x in samples]))


# def process(x):
# 	print(x)
# 	return np.array(bm25.get_scores(x.split(" ")))


if __name__ == '__main__':
	sim = np.load("sim_1000.npy")
	samples = load_from_disk("Dataset/sample_1000_raw")
	ABS = AdaptiveBatchSampling(samples, encode=False, similarity=sim)
	batches = ABS.solve(8, output_file="sample_1000_abs.json")
	# with Pool() as p:
	# 	sims = p.map(process, query)
	# 	arr = np.vstack(sims)
	# 	np.save("sim_1000.npy", arr)

