import numpy as np

class RandomArrival:
    def __init__(self, load, duration, p):
        self.load = load
        self.duration = duration
        if type(p) == int or type(p) == float:
            self.p = lambda t: p
        elif callable(p):
            self.p = p
        else:
            raise "p not of valid type"
        self.seed(37)
    def seed(self, seed, hour_offset=0):
        self.rng = np.random.default_rng(seed)
    def __call__(self, t):
        if self.rng.random() < self.p(t):
            return (self.load, self.duration)
        else:
            return (0, 0)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)