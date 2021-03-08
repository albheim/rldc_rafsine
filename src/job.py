import numpy as np

class ConstantArrival:
    def __init__(self, load, duration):
        self.load = load
        self.duration = duration
    def __call__(self, t, dt):
        return (self.load * dt, self.duration)
    def min_values(self):
        return (0, 0)
    def max_values(self):
        return (self.load, self.duration)

class SinusArrival:
    def __init__(self, offset, amplitude, period, phase_offset=0, duration=200):
        self.offset = offset
        self.amplitude = amplitude
        self.omega = 2 * np.pi / period
        self.phase_offset = phase_offset

        self.time = 0
        self.duration = duration

    def get_load(self):
        return self.offset + self.amplitude * np.sin(self.omega * self.time + self.phase_offset)
    
    def step(self, dt):
        self.time += dt
        return (self.get_load() * dt, self.duration)

class RandomArrival:
    def __init__(self, load, duration, p):
        self.load = load
        self.duration = duration
        self.p = p
    def step(self, dt):
        if np.random.rand() < self.p:
            return self.load * dt, self.duration
        else:
            return 0, 0