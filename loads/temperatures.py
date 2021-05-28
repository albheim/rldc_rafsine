import numpy as np

class SinusTemperature:
    def __init__(self, offset, amplitude):
        self.period = 24*60*60
        self.offset = offset
        self.amplitude = amplitude
    def __call__(self, t):
        return self.offset + self.amplitude * np.sin(2 * np.pi * t / self.period)
    def min_values(self):
        return self.offset - self.amplitude - 1
    def max_values(self):
        return self.offset + self.amplitude + 1