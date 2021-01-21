import numpy as np

class SimpleDCEnv:
    def __init__(self, n_servers):
        self.n_servers = n_servers

        self.server_max_load = 500  # W
        self.server_idle_load = 200
        self.server_max_temp_cpu = 85  # C
        self.server_idle_temp_cpu = 35
        self.server_max_flow = 0.1  # m3/s
        self.server_idle_flow = 0.01
        self.server_max_fan_power = 50  # W

        self.crah_min_flow = self.n_servers * self.server_idle_flow # Minimal CRAH is minimal server
        self.crah_max_flow = self.n_servers * self.server_max_flow * 2 # Allow CRAH to be up to twice as strong as servers
        self.crah_max_power = self.n_servers * self.server_max_fan_power # CRAH is twice as efficient (max is same energy as servers but double flow)

        self.crah_min_temp = 18
        self.crah_max_temp = 27

        self.ambient_temp = 22

        self.job_time = 10#200
        self.job_rate = 1#n_servers / 40 # Just tested to be around a nice number
        self.job_load = 20

        self.air_vol_heatcap = (1000 * 1.225) # J/(m^3 K)
        # Ratio between air and water heat capacity adjusted for volume instead of mass
        self.fixed_heatcap_ratio = (1000 * 1.225) / (4200 * 997)

        self.R = 0.005  # TODO totally made up value, check what is reasonable

        self.reset()

    def reset(self):
        self.server_flow = self.server_idle_flow * np.ones(self.n_servers)
        self.server_load = self.server_idle_load * np.ones(self.n_servers)
        self.server_temp_out = self.ambient_temp + self.server_load / (self.server_flow * self.air_vol_heatcap)
        self.server_temp_in = self.ambient_temp * np.ones(self.n_servers)

        self.job_queue = [[] for i in range(self.job_time)]

        self.jobs = np.random.poisson(self.job_rate)

        self.time = 0

        return {"time": self.time, "jobs": self.jobs, "load": self.server_load.copy(), "in_temp": self.server_temp_out[0], "out_temp": self.server_temp_out.copy(), "cpu_temp": self.server_idle_temp_cpu * np.zeros(self.server_temp_out.shape), "crah_temp_out": self.ambient_temp, "crah_flow": np.sum(self.server_flow), "flow": self.server_flow, "compressor": 0, "ambient": self.ambient_temp, "cost": 0}

    def step(self, a):
        placement, crah = a
        crah_temp_out = crah[0]
        crah_flow = crah[1]

        #self.ambient_temp = np.sin(self.time / (60*60*24) * 2*np.pi) * 5 + 15

        # Place jobs and remove finished jobs
        self.server_load[placement] += self.job_load

        self.job_queue.append(placement)
        i = self.job_queue.pop(0)
        self.server_load[i] -= self.job_load

        # Find new input temperatures
        server_flow_total = np.sum(self.server_flow)
        server_temp_out = np.dot(self.server_temp_out,
                                 self.server_flow) / server_flow_total
        if server_flow_total > crah_flow:
            # Recirculation to server, some servers get hotter air
            crah_temp_in = server_temp_out
            gamma = crah_flow / server_flow_total
            self.server_temp_in = gamma * crah_temp_out + (1 - gamma) * server_temp_out + np.zeros(self.n_servers)
        else:
            # Bypass to crah
            self.server_temp_in[:] = crah_temp_out
            eta = server_flow_total / crah_flow
            crah_temp_in = (1 - eta) * crah_temp_out + eta * server_temp_out

        # # Normal compressor
        # if crah_temp_out > self.ambient_temp:
        #     compressor = 0
        # else:
        #     compressor = (crah_temp_in - crah_temp_out) * self.air_vol_heatcap * crah_flow
        # # Hinge compressor
        # if crah_temp_out > self.ambient_temp:
        #     compressor = 0
        # else:
        #     compressor = (self.ambient_temp - crah_temp_out) * self.air_vol_heatcap * crah_flow
        # Sigmoid compressor
        compressor = 1 / (1 + np.exp(-(self.ambient_temp - crah_temp_out) / 5)) * np.sum(self.server_load)

        # Get new cpu temp
        server_temp_cpu = self.server_temp_in + self.R * self.server_load / self.server_flow
        # Set setpoints based on load
        server_temp_set = self.server_idle_temp_cpu + (self.server_max_temp_cpu - self.server_idle_temp_cpu) * np.clip(self.server_load / self.server_max_load, 0, 1)
        # Update fan speed based on relation between them
        self.server_flow = np.clip(self.server_flow * server_temp_cpu / server_temp_set, self.server_idle_flow, self.server_max_flow)
        self.server_temp_out = self.server_temp_in + self.server_load / (self.server_flow * self.air_vol_heatcap)

        self.jobs = 1#np.random.poisson(self.job_rate) 

        server_fan_energy = np.sum(self.server_max_fan_power * (self.server_flow / self.server_max_flow)**3)
        crah_fan_energy = self.crah_max_power * (crah_flow / self.crah_max_flow)**3 

        cost = server_fan_energy + crah_fan_energy + compressor
        state = {"time": self.time, "jobs": self.jobs, "load": np.copy(self.server_load), 
                 "in_temp": np.copy(self.server_temp_in), "out_temp": np.copy(self.server_temp_out), 
                 "cpu_temp": np.copy(server_temp_cpu), "crah_flow": crah_flow, "crah_temp_out": crah_temp_out, 
                 "flow": np.copy(self.server_flow), "ambient": self.ambient_temp, "cost": cost, "cpu_setpoint": server_temp_set,
                 "energy_server_fans": server_fan_energy, "energy_crah_fans": crah_fan_energy, "energy_compressor": compressor}


        reward = -cost - 100*np.sum(np.maximum(0, server_temp_cpu - self.server_max_temp_cpu)**2) 

        self.time += 1

        return state, reward / 1000
