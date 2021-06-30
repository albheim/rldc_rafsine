import numpy as np

class SinusTemperature:
    def __init__(self, offset, amplitude, phase=0, noise=0, period=24*60*60):
        self.period = period
        self.offset = offset
        self.amplitude = amplitude
        self.phase = phase
        self.noise = noise
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.phase = self.rng.random() * 24 * 60 * 60
    def __call__(self, t):
        return self.offset + self.amplitude * np.sin(2 * np.pi * (t - self.phase) / self.period) + self.noise * self.rng.normal()
    def min_values(self):
        return self.offset - self.amplitude - 10 * (self.noise + 0.01)
    def max_values(self):
        return self.offset + self.amplitude + 10 * (self.noise + 0.01)