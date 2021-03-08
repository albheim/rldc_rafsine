import numpy as np

class CRAH:
    def __init__(self, n_crah):
        self.n_crah = n_crah

        self.min_temp = 18
        self.max_temp = 27
        self.min_flow = 0.4
        self.max_flow = 2.5
        self.max_fan_power = 10000

        # Kinematic viscosity of air (m^2/s)
        nu = 1.568e-5
        # Thermal conductivity (W/m K)
        k = 2.624e-2
        # Prandtl number of air
        Pr = 0.707
        self.air_vol_heatcap = Pr * k / nu 

    def reset(self, ambient_temp):
        self.flow = self.min_flow * np.ones(self.n_crah)
        self.temp_out = 22 * np.ones(self.n_crah)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        # If Tamb < Tout compressor is off
        self.compressor_power = np.sum((ambient_temp > self.temp_out) * self.air_vol_heatcap * self.flow * (ambient_temp - self.temp_out))

    def update(self, temp_out, flow, temp_in, ambient_temp):
        # Maybe allow individual control?
        self.flow = flow * np.ones(self.n_crah)
        self.temp_out = temp_out * np.ones(self.n_crah)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        # If Tamb < Tout compressor is off
        self.compressor_power = np.sum((ambient_temp > self.temp_out) * self.air_vol_heatcap * self.flow * (temp_in - self.temp_out))