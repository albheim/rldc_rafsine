import numpy as np
import heapq

class Servers:
    def __init__(self, n_servers):
        self.n_servers = n_servers

        self.idle_load = 200
        self.max_load = 500  # W
        self.idle_temp_cpu = 35
        self.max_temp_cpu = 85  # C
        self.idle_flow = 0.01
        self.max_flow = 0.04 # m3/s

        self.max_fan_power = 25.2 * 2 

        # Kinematic viscosity of air (m^2/s)
        nu = 1.568e-5
        # Thermal conductivity (W/m K)
        k = 2.624e-2
        # Prandtl number of air
        Pr = 0.707
        self.air_vol_heatcap = Pr * k / nu 
        #self.air_vol_heatcap = 1000 * 1.225

        # Server constants
        # self.R = 0.001
        self.R = 3 / self.air_vol_heatcap

    def reset(self, ambient_temp):
        self.delta_t = np.zeros(self.n_servers)
        self.temp_cpu = ambient_temp * np.ones(self.n_servers)
        self.flow = self.idle_flow * np.ones(self.n_servers)
        self.load = self.idle_load * np.ones(self.n_servers)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        self.running_jobs = []
        self.dropped_jobs = 0

    def update(self, time, placement, load, duration, temp_in):
        self.dropped_jobs = 0

        if load == 0:
            pass # No job
        elif self.load[placement] + load <= self.max_load:
            self.load[placement] += load
            heapq.heappush(self.running_jobs, (time + duration, load, placement))
        else:
            self.dropped_jobs = 1
        while len(self.running_jobs) > 0 and self.running_jobs[0][0] <= time:
            _, load, placement = heapq.heappop(self.running_jobs)
            self.load[placement] -= load

        # Update server
        self.temp_cpu = temp_in + self.R * self.load / self.flow
        cpu_target_temp = self.idle_temp_cpu + (self.max_temp_cpu - self.idle_temp_cpu) * np.clip((self.load - self.idle_load) / (self.max_load - self.idle_load), 0, 1)
        self.flow = np.clip(self.flow * self.temp_cpu / cpu_target_temp, self.idle_flow, self.max_flow)

        self.delta_t = self.load / (self.air_vol_heatcap * self.flow)
        
        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)
