import json
import os
from datetime import datetime

import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score


class ComputeMetrics:
    def __init__(self, split, output_dir, save_step=1, verbose=False):
        self.split = split
        self.output_dir = output_dir
        self.counter = 0
        self.save_step = save_step
        self.verbose = verbose
        self.filename = f"{self.split}_metrics_{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"

    def __call__(self, eval_preds):
        output, labels = eval_preds
        scores, loss = output[2], np.mean(output[3], dtype=np.float64)
        mmr = self.MRR(scores, labels)
        ndcg = self.NDCG(scores, labels)
        metrics = {'loss': loss, 'mmr': mmr, "ndcg": ndcg}
        if self.verbose:
            print(f'{self.split} step{self.counter+1}: loss={loss} mmr={mmr} ndcg={ndcg}')
        self.save_metrics(metrics)
        return metrics

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

    def save_metrics(self, metrics):
        self.counter += 1
        if self.counter % self.save_step != 0:
            return

        metrics = metrics.copy()
        metrics['step'] = self.counter

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        path = os.path.join(self.output_dir, self.filename)
        with open(path, "a") as f:
            json_str = json.dumps(metrics, sort_keys=True)
            f.write(json_str+'\n')
