import numpy as np

class Workload:
    def __init__(self):
        pass
    def seed(self, seed):
        pass
    def __call__(self, t):
        pass
    def min_values(self):
        pass
    def max_values(self):
        pass

class ConstantArrival(Workload):
    def __init__(self, load, duration):
        self.load = load
        self.duration = duration
    def __call__(self, t):
        return (self.load, self.duration)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)

class RandomArrival(Workload):
    def __init__(self, load, duration, p):
        self.load = load
        self.duration = duration
        self.p = p
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
    def __call__(self, t):
        if self.rng.rand() < self.p:
            return (self.load, self.duration)
        else:
            return (0, 0)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)