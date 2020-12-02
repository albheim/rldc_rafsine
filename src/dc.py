import numpy as np

class SimpleDCEnv:
    def __init__(self):
        self.n_servers = 10

        self.server_max_load = 10000  # W
        self.server_idle_load = 45
        self.server_max_temp_cpu = 85  # C
        self.server_idle_temp_cpu = 35
        self.server_max_flow = 0.1  # m3/s
        self.server_idle_flow = 0.01
        self.server_max_fan_power = 500  # W

        self.crah_min_flow = self.n_servers * self.server_idle_flow
        self.crah_max_flow = self.n_servers * self.server_max_flow * 3
        self.crah_max_power = self.n_servers * self.server_max_fan_power

        self.ambient_temp = 15

        self.job_time = 200
        self.job_rate = 0.6
        self.job_load = 300

        self.air_vol_heatcap = (1000 * 1.225)
        # Ratio between air and water heat capacity adjusted for volume instead of mass
        self.fixed_heatcap_ratio = (1000 * 1.225) / (4200 * 997)

        self.R = 0.001  # TODO totally made up value, check what is reasonable

        self.reset()

    def reset(self):
        self.server_flow = self.server_idle_flow * np.ones(self.n_servers)
        self.server_power = self.server_idle_load * np.ones(self.n_servers)
        self.server_temp_out = self.ambient_temp + self.server_power / \
            (self.server_flow * self.air_vol_heatcap)

        self.job_queue = [[] for i in range(self.job_time)]

        self.job = np.random.rand() > 1 - self.job_rate

        self.time = 0

        return {"time": self.time, "job": self.job, "load": self.server_power.copy(), "in_temp": self.server_temp_out[0], "out_temp": self.server_temp_out.copy(), "cpu_temp": self.server_idle_temp_cpu * np.zeros(self.server_temp_out.shape), "crah_temp_out": self.ambient_temp, "crah_flow": np.sum(self.server_flow), "flow": self.server_flow, "generator": 0, "ambient": self.ambient_temp, "cost": 0}

    def step(self, a):
        # New power deistribution and new crah_temp_out and crah_flow just set, update other stuff
        # Should save srv temp out, srv- crah- and water flow, power distribution, generator status

        placement, crah_temp_out_ratio, crah_flow_ratio = a

        crah_temp_out = 10 + 15 * crah_temp_out_ratio
        crah_flow = self.crah_min_flow + \
            (self.crah_max_flow - self.crah_min_flow) * crah_flow_ratio

        self.ambient_temp = np.sin(self.time / (60*60*24) * 2*np.pi) * 5 + 15

        # Place jobs and remove finished jobs
        if self.job:
            self.server_power[placement] += self.job_load
            self.job_queue.append(placement)
        else:
            self.job_queue.append(-1)
        idx = self.job_queue.pop(0)
        if idx != -1:
            self.server_power[idx] -= self.job_load

        # Find new input temperatures
        server_flow_total = np.sum(self.server_flow)
        server_temp_out = np.dot(self.server_temp_out,
                                 self.server_flow) / server_flow_total
        if server_flow_total > crah_flow:
            # Reflow to server
            crah_temp_in = server_temp_out
            eta = crah_flow / server_flow_total
            server_temp_in = eta * crah_temp_out + (1 - eta) * server_temp_out
        else:
            # Reflow to crah
            server_temp_in = crah_temp_out
            eta = server_flow_total / crah_flow
            crah_temp_in = (1 - eta) * crah_temp_out + eta * server_temp_out

        if crah_temp_out > self.ambient_temp:
            generator = 0
            water_flow = crah_flow * self.fixed_heatcap_ratio * \
                (crah_temp_in - crah_temp_out) / \
                (crah_temp_in - self.ambient_temp)
        else:
            # water_flow = crah_flow * self.fixed_heatcap_ratio * (crah_temp_in - crah_temp_out) / (crah_temp_in - self.ambient_temp + 5) # TODO fix this
            water_flow = 0
            generator = (crah_temp_in - crah_temp_out) * \
                self.air_vol_heatcap * crah_flow
            #generator = np.sum(self.server_power)

        # Get new cpu temp
        server_temp_cpu = server_temp_in + self.R * self.server_power / self.server_flow
        # Set setpoints based on load
        server_temp_set = self.server_idle_temp_cpu + \
            (self.server_max_temp_cpu - self.server_idle_temp_cpu) * \
            np.clip(self.server_power / self.server_max_load, 0, 1)
        # Update fan speed based on relation between them
        self.server_flow = np.clip(self.server_flow * server_temp_cpu /
                                   server_temp_set, self.server_idle_flow, self.server_max_flow)
        self.server_temp_out = server_temp_in + self.server_power / \
            (self.server_flow * self.air_vol_heatcap)

        self.job = np.random.rand() > 1 - self.job_rate

        cost = np.sum(self.server_max_fan_power * (self.server_flow / self.server_max_flow)**3 +
                      self.crah_max_power * (crah_flow / self.crah_max_flow)**3) + generator

        state = {"time": self.time, "job": self.job, "load": np.copy(self.server_power), "in_temp": np.copy(server_temp_in), "out_temp": np.copy(self.server_temp_out), "cpu_temp": np.copy(
            server_temp_cpu), "crah_flow": crah_flow, "crah_temp_out": crah_temp_out, "flow": np.copy(self.server_flow), "generator": generator, "ambient": self.ambient_temp, "cost": cost}

        reward = -100*np.sum(np.maximum(0, server_temp_cpu -
                                        self.server_max_temp_cpu)**2) - cost

        self.time += 1

        return state, reward
