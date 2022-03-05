import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score


class ComputeMetrics:
    def __call__(self, eval_preds):
        output, labels = eval_preds
        scores = output[2]
        mmr = self.MRR(scores, labels)
        ndcg = self.NDCG(scores, labels)
        return {'mmr': mmr, "ndcg": ndcg}

    @staticmethod
    def MRR(scores, target):
        """
        scores: [batch_size, num_passages]
        """
        probs = softmax(scores, axis=1)
        rank = np.apply_along_axis(rankdata, axis=1, arr=-probs)
        idx = np.argmax(target, axis=1).reshape(-1, 1)
        rank_top_idx = np.take_along_axis(rank, idx, axis=1)  # The rank of the top 1 item of target in prediction
        return np.mean(1 / rank_top_idx)

    @staticmethod
    def NDCG(scores, target):
        """
        scores: [batch_size, num_passages]
        """
        probs = softmax(scores, axis=1)
        return ndcg_score(target, probs)

