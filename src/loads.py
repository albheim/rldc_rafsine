import numpy as np

class ConstantArrival:
    def __init__(self, load, duration):
        self.load = load
        self.duration = duration
    def __call__(self, t):
        return (self.load, self.duration)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)

class RandomArrival:
    def __init__(self, load, duration, p):
        self.load = load
        self.duration = duration
        self.p = p
    def __call__(self, t):
        if np.random.rand() < self.p:
            return (self.load, self.duration)
        else:
            return (0, 0)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)

class ConstantTemperature:
    def __init__(self, temp):
        self.temp = temp
    def __call__(self, t):
        return self.temp 
    def min_values(self):
        return 0
    def max_values(self):
        return self.temp
