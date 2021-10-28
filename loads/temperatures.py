import numpy as np
import pandas as pd

class SinusTemperature:
    def __init__(self, offset, amplitude, noise=0, period=24*60*60):
        self.period = period
        self.offset = offset
        self.amplitude = amplitude
        self.noise = noise
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.phase = self.rng.random() * self.period
    def __call__(self, t):
        return self.offset + self.amplitude * np.sin(2 * np.pi * (t - self.phase) / self.period) + self.noise * self.rng.normal()
    def min_values(self):
        return self.offset - self.amplitude - 10 * (self.noise + 0.01)
    def max_values(self):
        return self.offset + self.amplitude + 10 * (self.noise + 0.01)

class CSVTemperature:
    def __init__(self, path):
        self.start_idx = 0
        # Hourly averages since 2016-10-16 00:00 to 2021-10-15 23:00
        self.data = pd.read_csv(path, sep=";", decimal=",")["Timmedel"].to_numpy()
    
    def seed(self, seed, hour_offset=0):
        self.rng = np.random.default_rng(seed)
        # self.start_idx = hour_offset + 24 * self.rng.integers(low=0, high=len(self.data) // 24)
        start_idx = 33220 + hour_offset # Will generate data from summer with temp in 10-20 range
        self.data = self.data[start_idx:start_idx+24*10]
    
    def __call__(self, t):
        temp = np.interp(t/3600, range(len(self.data)), self.data)
        return temp
