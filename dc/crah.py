import numpy as np

class CRAH:
    def __init__(self, n_crah, air_vol_heatcap):
        self.n_crah = n_crah
        self.air_vol_heatcap = air_vol_heatcap

        self.min_temp = 18
        self.max_temp = 27
        self.min_flow = 0.1 # To avoid divide by zero
        self.max_flow = 2.1 
        # servers are 50.4 / 0.04 watt/flow
        # crah should probably be similar (maybe cheaper) so starting we set it to 2.5 * 50.4 / 0.04 = 2646
        # Assume it is twice as effective
        self.max_fan_power = 750 #2646 / 2

        self.compressor_factor = 0.3 # How efficient is the compressor? Costs this much energy per unit of removed heat energy.

    def reset(self, outdoor_temp):
        self.flow = self.min_flow * np.ones(self.n_crah)
        self.temp_out = 22 * np.ones(self.n_crah)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        # If Tamb < Tout compressor is off, else it extracts the energy flowing through by using an equal amount of energy
        self.compressor_power = self.compressor_factor * np.sum((outdoor_temp > self.temp_out) * self.air_vol_heatcap * self.flow * (outdoor_temp - self.temp_out))

    def update(self, temp_out, flow, temp_in, outdoor_temp):
        # Allows for flow and temp_out to be either numbers or vectors for individual vs combined control
        self.flow = flow * np.ones(self.n_crah)
        self.temp_out = temp_out * np.ones(self.n_crah)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        # If Tamb < Tout compressor is off
        # Assumes Tin > Tout which should always hold for stationary conditions but maybe could be broken if Tout changes too much too quickly.
        self.compressor_power = self.compressor_factor * np.sum((outdoor_temp > self.temp_out) * self.air_vol_heatcap * self.flow * (temp_in - self.temp_out))