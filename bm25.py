import logging
import math

import numpy as np
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_from_disk
from collections import Counter
from gensim.summarization.bm25 import BM25 as BBM25
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BM25(object):
    def __init__(self, corpus, tokenizer: PreTrainedTokenizer, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = np.zeros((len(corpus), self.vocab_size), dtype=np.uint32)
        self.idf = np.zeros(self.vocab_size)
        self.doc_len = np.zeros(len(corpus))
        self._initialize(corpus)

    def _initialize(self, corpus):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for i in tqdm(range(len(corpus)), desc="Count Frequency"):
            document = self.encode(corpus[i])
            self.doc_len[i] = len(document)
            num_doc += len(document)

            frequencies = Counter(document)
            for token_id in frequencies:
                self.doc_freqs[i, token_id] += frequencies[token_id]
                nd[token_id] = nd.get(token_id, 0) + 1

        self.avgdl = float(num_doc) / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for token_id, freq in tqdm(nd.items(), desc="Count IDF"):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[token_id] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(token_id)
        self.average_idf = float(idf_sum) / len(nd)

        if self.average_idf < 0:
            logger.warning(
                'Average inverse document frequency is less than zero. Your corpus of {} documents'
                ' is either too small or it does not originate from natural text. BM25 may produce'
                ' unintuitive results.'.format(self.corpus_size)
            )

        eps = self.epsilon * self.average_idf
        for token_id in negative_idfs:
            self.idf[token_id] = eps

        self.denominator_constant = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

    def get_score(self, query, index, encoded=False):
        numerator_constant = self.k1 + 1
        denominator_constant = self.denominator_constant[index]
        if not encoded:
            query = self.encode(query)

        df = self.doc_freqs[index, query]
        idf = self.idf[query]
        score = np.sum(idf * df * numerator_constant / (df + denominator_constant))
        return score

    def get_scores(self, query, encoded=False):
        numerator_constant = self.k1 + 1
        if not encoded:
            query = self.encode(query)

        df = self.doc_freqs[:, query]
        idf = self.idf[query]
        scores = np.sum(idf * df * numerator_constant / (df + self.denominator_constant.reshape(-1, 1)), axis=1)
        return scores

    def batch_get_score(self, queries, indexes, encoded=False):
        numerator_constant = self.k1 + 1
        if not encoded:
            queries = self.batch_encode(queries)
        denominator_constant = self.denominator_constant[indexes].reshape(-1, 1, 1)

        if len(queries) >= len(indexes):
            df = np.take(self.doc_freqs[indexes], queries, axis=1)
        else:
            df = np.take(self.doc_freqs, queries, axis=1)[indexes]

        idf = np.take(self.idf, queries)
        idf = idf.reshape(1, idf.shape[0], idf.shape[1])
        n = df * idf * numerator_constant
        d = df + denominator_constant
        score = np.sum(n / d, axis=-1).T
        return score

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="np",  add_special_tokens=False,
                                     padding=False, truncation=True).flatten()

    def batch_encode(self, text):
        return self.tokenizer(text, return_tensors="np", padding=True, return_attention_mask=False,
                              add_special_tokens=False, return_token_type_ids=False).input_ids


if __name__ == '__main__':
    samples = load_from_disk("Dataset/sample_1000_raw")
    tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
    bm25 = BM25(corpus=samples['positives'], tokenizer=tokenizer)
    text = "His mother, Bl. Joan of Aza, was a holy woman in her own right."
    s1 = bm25.get_score(text, 150)
    # s2 = bm25.get_score("His mother, Bl. Joan of Aza, was a holy woman in her own right.", 160)
    # s3 = bm25.get_scores("His mother, Bl. Joan of Aza, was a holy woman in her own right.")
    # s4 = bm25.get_score("His mother, Bl. Joan of Aza", 150)
    # s5 = bm25.get_score("His mother, Bl. Joan of Aza", 160)
    s6 = bm25.batch_get_score(["His mother, Bl. Joan of Aza, was a holy woman in her own right.",
                               "His mother, Bl. Joan of Aza"], [150, 160, 170])

    encode = [tokenizer.encode(seq, return_tensors="np",  add_special_tokens=False,
                               padding=False, truncation=True).flatten() for seq in samples['positives']]
    decode = [tokenizer.convert_ids_to_tokens(x) for x in encode]

    text_1 = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, return_tensors="np",  add_special_tokens=False,
                               padding=False, truncation=True).flatten())

    bbm25 = BBM25(corpus=decode)
    s2 = bbm25.get_score(text_1, 150)
    print(s1, s2)
