import numpy as np
import heapq

class Servers:
    def __init__(self, n_servers, air_vol_heatcap, R):
        self.n_servers = n_servers
        self.air_vol_heatcap = air_vol_heatcap
        self.R = R

        self.idle_load = 200
        self.max_load = 500  # W
        self.idle_temp_cpu = 35
        self.temp_cpu_target = 60
        self.max_temp_cpu = 85  # C
        self.idle_flow = 0.01
        self.max_flow = 0.04 # m3/s

        # TODO maybe find Ti in better way?
        # Ti is negative since a lower temp than ref means we should lower flow
        self.Ti = -(self.max_temp_cpu - self.idle_temp_cpu) / (self.max_flow - self.idle_flow)

        self.max_fan_power = 25.2 * 2 

    def reset(self, ambient_temp):
        self.delta_t = np.zeros(self.n_servers)
        self.temp_cpu = ambient_temp * np.ones(self.n_servers)
        self.flow = self.idle_flow * np.ones(self.n_servers)
        self.load = self.idle_load * np.ones(self.n_servers)

        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

        self.running_jobs = []
        self.dropped_jobs = 0
        self.overheated_inlets = 0
        self.load_variance = 0

    def update(self, time, dt, placement, load, duration, temp_in):
        # Update server in correct order
        new_temp_cpu = temp_in + self.R * self.load / self.flow
        #cpu_target_temp = self.idle_temp_cpu + (self.max_temp_cpu - self.idle_temp_cpu) * np.clip((self.load - self.idle_load) / (self.max_load - self.idle_load), 0, 1)
        self.flow = np.clip(self.flow + dt / self.Ti * (self.temp_cpu_target - self.temp_cpu), self.idle_flow, self.max_flow)
        #self.flow = np.clip(self.flow * self.temp_cpu / self.temp_cpu_target, self.idle_flow, self.max_flow)
        #self.flow = self.idle_flow + (self.max_flow - self.idle_flow) * np.clip((self.load - self.idle_load) / (self.max_load - self.idle_load), 0, 1)

        self.temp_cpu = new_temp_cpu
        self.overheated_inlets = np.sum(temp_in > 27)
        self.load_variance = np.var(self.load)

        self.delta_t = self.load / (self.air_vol_heatcap * self.flow)
        
        self.fan_power = np.sum(self.max_fan_power * (self.flow / self.max_flow)**3)

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
