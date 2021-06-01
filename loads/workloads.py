import numpy as np

class RandomArrival:
    def __init__(self, load, duration, p):
        self.load = load
        self.duration = duration
        self.p = p
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
    def __call__(self, t):
        if self.rng.random() < self.p:
            return (self.load, self.duration)
        else:
            return (0, 0)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)