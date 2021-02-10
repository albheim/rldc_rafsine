import numpy as np
import gym
import heapq

class SimpleDCEnv(gym.Env):
    def __init__(self, config):
        self.n_servers = config["n_servers"]

        # Server constants
        self.server_max_load = 500  # W
        self.server_idle_load = 200
        self.server_max_temp_cpu = 85  # C
        self.server_idle_temp_cpu = 35
        self.server_max_flow = 0.1  # m3/s
        self.server_idle_flow = 0.01
        self.server_max_fan_power = 50  # W

        # CRAH constants
        self.crah_min_temp = 18
        self.crah_max_temp = 27
        self.crah_min_flow = self.n_servers * self.server_idle_flow # Minimal CRAH is minimal server
        self.crah_max_flow = self.n_servers * self.server_max_flow * 2 # Allow CRAH to be up to twice as strong as servers
        self.crah_max_power = self.n_servers * self.server_max_fan_power # CRAH is twice as efficient (max is same energy as servers but double flow)

        # Env constants
        self.ambient_temp = 22

        # Job constants
        self.job_time = 200
        self.job_rate = self.n_servers / 30 # Just tested to be around a nice number
        self.job_load = 20

        # Physical constants
        self.air_vol_heatcap = (1000 * 1.225) # J/(m^3 K)
        self.R = 0.005  # TODO totally made up value, check what is reasonable

        # Gym spaces
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.n_servers), 
            gym.spaces.Box(-1.0, 1.0, shape=(2,))))
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(1 + 3 * self.n_servers,))

        # Used for rescaling
        self.alow = np.array([self.crah_min_temp, self.crah_min_flow])
        self.ahigh = np.array([self.crah_max_temp, self.crah_max_flow])
        self.slow = np.concatenate(
            (self.ambient_temp * np.ones(self.n_servers),
            self.server_idle_flow * np.ones(self.n_servers),
            self.server_idle_load * np.ones(self.n_servers)))
        self.shigh = np.concatenate(
            (self.server_max_temp_cpu * np.ones(self.n_servers),
            self.server_max_flow * np.ones(self.n_servers),
            self.server_max_load * np.ones(self.n_servers)))

    def action_transform(self, action):
        a1, a2 = action
        # a1 = np.random.randint(self.n_servers)
        a1 = (a1, self.job_load, self.job_time)
        a2 = (a2 * (self.ahigh - self.alow) + self.ahigh + self.alow) / 2
        a2 = np.clip(a2, self.alow, self.ahigh)
        self.action = (a1, a2) # Save for plotting
        return self.action
    
    def state_transform(self, state):
        s = np.concatenate((state["out_temp"], state["flow"], state["load"]))
        s = (2 * s - self.slow - self.shigh) / (self.shigh - self.slow)
        return np.concatenate((s, [state["jobs"] - 0.5]))

    def reset(self):
        # States
        self.server_flow = self.server_idle_flow * np.ones(self.n_servers)
        self.server_load = self.server_idle_load * np.ones(self.n_servers)
        self.server_temp_out = self.ambient_temp + self.server_load / (self.server_flow * self.air_vol_heatcap)
        self.server_temp_in = self.ambient_temp * np.ones(self.n_servers)
        self.server_temp_cpu = self.server_idle_temp_cpu * np.ones(self.n_servers)

        self.crah_temp_in = self.ambient_temp

        # Reset other stuff
        self.running_jobs = []

        #self.jobs = np.random.poisson(self.job_rate)
        self.jobs = 1 if np.random.rand() < self.job_rate else 0

        self.time = 0

        state = {"jobs": self.jobs, "load": self.server_load, "out_temp": self.server_temp_out, "flow": self.server_flow}
        return self.state_transform(state)

    def step(self, a):
        job, crah = self.action_transform(a)

        # Set CRAH
        crah_temp_out = crah[0]
        crah_flow = crah[1]

        # Place jobs and remove finished jobs
        srvidx, load, dur = job # Always here, but load and dur is zero if no job
        self.server_load[srvidx] += load
        heapq.heappush(self.running_jobs, (self.time + dur, load, srvidx))
        while len(self.running_jobs) > 0 and self.running_jobs[0][0] <= self.time:
            _, load, srvidx = heapq.heappop(self.running_jobs)
            self.server_load[srvidx] -= load

        #self.ambient_temp = np.sin(self.time / (60*60*24) * 2*np.pi) * 5 + 15

        # Temp vars
        server_flow_total = np.sum(self.server_flow)

        # Find new input temperatures
        bypass = min(1, crah_flow / server_flow_total)
        recirc = min(1, server_flow_total / crah_flow)
        server_temp_in = bypass * crah_temp_out + (1 - bypass) * self.server_temp_out 
        crah_temp_in = recirc * self.server_temp_out + (1 - recirc) * crah_temp_out

        # Server temp out 
        server_temp_out_individual = self.server_temp_in + self.server_load / (self.server_flow * self.air_vol_heatcap)
        server_temp_out = np.dot(self.server_flow, server_temp_out_individual) / server_flow_total

        # Get new cpu temp
        server_temp_cpu = self.server_temp_in + self.R * self.server_load / self.server_flow
        # Set setpoints based on load
        server_temp_set = self.server_idle_temp_cpu + (self.server_max_temp_cpu - self.server_idle_temp_cpu) * np.clip((self.server_load - self.server_idle_load) / (self.server_max_load - self.server_idle_load), 0, 1)
        # Update fan speed based on relation between them
        server_flow = np.clip(self.server_flow * self.server_temp_cpu / server_temp_set, self.server_idle_flow, self.server_max_flow)
        
        # Normal compressor
        if crah_temp_out > self.ambient_temp:
            compressor = 0
        else:
            compressor = (crah_temp_in - crah_temp_out) * self.air_vol_heatcap * crah_flow
        # # Hinge compressor
        # if crah_temp_out > self.ambient_temp:
        #     compressor = 0
        # else:
        #     compressor = (self.ambient_temp - crah_temp_out) * self.air_vol_heatcap * crah_flow
        # Sigmoid compressor
        # compressor = 1 / (1 + np.exp(-(self.ambient_temp - crah_temp_out) / 5)) * (self.ambient_temp - crah_temp_out) * self.air_vol_heatcap * crah_flow

        self.jobs = 1 if np.random.rand() < self.job_rate else 0 # Should maybe generate a load and a duration that is 0,0 if no job

        server_fan_energy = np.sum(self.server_max_fan_power * (server_flow / self.server_max_flow)**3)
        crah_fan_energy = self.crah_max_power * (crah_flow / self.crah_max_flow)**3 

        self.server_temp_cpu = server_temp_cpu
        self.server_temp_in = server_temp_in
        self.server_temp_out = server_temp_out
        self.server_flow = server_flow
        self.crah_temp_in = crah_temp_in

        cost = server_fan_energy + crah_fan_energy + compressor
        state = {"jobs": self.jobs, "load": self.server_load, "out_temp": server_temp_out_individual, "flow": self.server_flow}

        reward = -(cost + 100*np.sum(np.maximum(0, self.server_temp_cpu - self.server_max_temp_cpu)**2)) / 1000

        self.time += 1

        return self.state_transform(state), reward, False, {}
