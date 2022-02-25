import numpy as np
from datasets import load_from_disk
from gensim.summarization.bm25 import BM25
from multiprocessing import Pool

samples = load_from_disk("Dataset/sample_50000_raw")
passage = [x['positives'] for x in samples]
bm25 = BM25([doc.split(" ") for doc in passage])
query = list(set([x['query'] for x in samples]))


def process(x):
	print(x)
	return np.array(bm25.get_scores(x.split(" ")), np.float16)


if __name__ == '__main__':
	with Pool() as p:
		sims = p.map(process, query)
		arr = np.vstack(sims)
		np.savez_compressed("sim_50000", arr)
