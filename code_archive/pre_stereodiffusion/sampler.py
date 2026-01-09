from torch.utils.data import Sampler
from collections import defaultdict
import random


class BalancedByBaselineSampler(Sampler):
    """Round-robin über Baseline-Buckets; pro Step zufälliger Index aus Bucket."""
    def __init__(self, samples, batch_size=1, steps_per_epoch=2000):
        self.buckets = defaultdict(list)
        for i, s in enumerate(samples):
            self.buckets[s["btag"]].append(i)
        self.keys = sorted(self.buckets.keys())
        self.batch_size = batch_size
        self.steps = steps_per_epoch

    def __iter__(self):
        k = 0
        for _ in range(self.steps * self.batch_size):
            key = self.keys[k % len(self.keys)]
            yield random.choice(self.buckets[key])
            k += 1

    def __len__(self):
        return self.steps * self.batch_size