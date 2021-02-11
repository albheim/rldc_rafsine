import numpy as np

class ConstantArrival:
    def __init__(self, load, duration):
        self.load = load
        self.duration = duration
    def step(self):
        return (self.load, self.duration)

class RandomArrival:
    def __init__(self, load, duration, p):
        self.load = load
        self.duration = duration
        self.p = p
    def step(self):
        if np.random.rand() < self.p:
            return self.load, self.duration
        else:
            return 0, 0
