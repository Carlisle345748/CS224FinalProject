import random

import pandas as pd
import torch
from datasets import load_from_disk
from gensim.summarization.bm25 import BM25
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel


class AdaptiveBatchSampling:
	def __init__(self, dataset, encode=False):
		self.encode = encode
		self.sim_cache = {}
		self.data = [(x['query'], x['positives']) for x in dataset]

		if encode:
			self.tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
			self.model = AutoModel.from_pretrained('Luyu/co-condenser-marco')
		else:
			passage = list(set([x['positives'] for x in dataset]))
			self.p2idx = {passage[i]: i for i in range(len(passage))}
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
		temp = dataset.copy()
		if d_remove is None and d_add is None:
			for pair1 in temp:
				for pair2 in temp:
					if pair1 != pair2:
						hardness = hardness + self.get_similarity(pair1[0], pair2[1])

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
			for pair in dataset:
				hardness = hardness + self.get_similarity(pair[0], d_add[1]) + self.get_similarity(d_add[0], pair[1])

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

	def clean_cache(self, B):
		for pair in B:
			del self.sim_cache[pair[0]]

	def solve(self, batch_size, output_file=None):
		U = set(self.data)
		T = []
		while U:
			B = random.sample(U, batch_size)
			if len(U) > batch_size:
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
						print(hardness_B)
						break

			print(B)
			B = set(B)
			U = U - B
			T.extend(self.reformat(B))
			self.clean_cache(B)

		if output_file:
			df = pd.DataFrame.from_records(T)
			df.columns = ['query', 'passage']
			df.to_json(output_file, orient="records", lines=True)

		return T


if __name__ == '__main__':
	samples = load_from_disk("Dataset/sample_1000_raw")
	ABS = AdaptiveBatchSampling(samples, encode=False)
	batches = ABS.solve(8)
