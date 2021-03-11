import numpy as np
import gym
import heapq

from job import RandomArrival, ConstantArrival
class SimpleDCEnv(gym.Env):
    def __init__(self, config):
        self.dt = config.get("dt", 1)
        self.seed = config.get("seed", 37)
        self.energy_cost = config.get("energy_cost", 0.0001)
        self.job_drop_cost = config.get("job_drop_cost", 10.0)
        self.n_servers = config.get("n_servers", 360)
        self.n_crah = 1 # Simple sim can only have 1 CRAH

        self.setup_physical_constants()

        # Job constants
        self.job_load = 20
        self.job_time = 3600
        self.load_generator = ConstantArrival(load=self.job_load, duration=self.job_time)
        # self.job_time = 200
        # self.job_rate = self.n_servers / 30 # Just tested to be around a nice number
        # self.job_load = 20
        # self.load_generator = RandomArrival(self.job_load, self.job_time, self.job_rate)

        # Gym spaces
        self.load_balanced = config.get("load_balanced", True)
        if self.load_balanced:
            self.action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(self.n_servers), 
                gym.spaces.Box(-1.0, 1.0, shape=(2,))))
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(2 + 2 * self.n_servers,))

        # Used for rescaling
        self.alow = np.array([self.crah_min_flow, self.crah_min_temp])
        self.ahigh = np.array([self.crah_max_flow, self.crah_max_temp])
        self.slow = np.concatenate(
            (self.ambient_temp * np.ones(self.n_servers),
            self.server_idle_load * np.ones(self.n_servers),
            (0, 0)))
        self.shigh = np.concatenate(
            (self.server_max_temp_cpu * np.ones(self.n_servers),
            self.server_max_load * np.ones(self.n_servers),
            (self.job_load, self.job_time)))

    def action_transform(self, crah_settings):
        crah_settings = (np.clip(crah_settings, -1.0, 1.0) * (self.ahigh - self.alow) + self.ahigh + self.alow) / 2
        return crah_settings
    
    def state_transform(self, state):
        state = (2 * state - self.slow - self.shigh) / (self.shigh - self.slow)
        return state

    def reset(self):
        # Server states
        self.server_load = self.server_idle_load * np.ones(self.n_servers)
        self.server_flow = self.server_idle_flow * np.ones(self.n_servers)
        self.server_temp_out = (self.ambient_temp + self.server_load / (self.server_flow * self.air_vol_heatcap)) * np.ones(self.n_servers)
        self.server_temp_in = self.ambient_temp * np.ones(self.n_servers)
        self.server_temp_cpu = self.server_idle_temp_cpu * np.ones(self.n_servers)

        # CRAH states
        self.crah_temp_in = self.ambient_temp * np.ones(self.n_crah)
        self.crah_flow = self.crah_min_flow * np.ones(self.n_crah)
        self.crah_temp_out = self.crah_min_temp * np.ones(self.n_crah)

        # Jobs
        self.job = self.load_generator.step(self.dt)
        self.dropped_jobs = 0

        # Reset other stuff
        self.running_jobs = []
        self.dropped_jobs = 0

        self.time = 0

        state = np.concatenate((self.server_temp_out, self.server_load, self.job))
        return self.state_transform(state)

    def update_crah(self, settings):
        flow, temp_out = settings

        # Maybe allow individual control?
        self.crah_flow = flow * np.ones(self.n_crah)
        self.crah_temp_out = temp_out * np.ones(self.n_crah)

        self.crah_fan_power = np.sum(self.crah_max_fan_power * (self.crah_flow / self.crah_max_flow)**3)

        # If Tamb < Tout compressor is off
        self.compressor_power = np.sum((self.ambient_temp > self.crah_temp_out) * self.air_vol_heatcap * self.crah_flow * (self.crah_temp_in - self.crah_temp_out))

    def update_server(self):
        self.server_temp_cpu = self.server_temp_in + self.R * self.server_load / self.server_flow
        cpu_target_temp = self.server_idle_temp_cpu + (self.server_max_temp_cpu - self.server_idle_temp_cpu) * np.clip((self.server_load - self.server_idle_load) / (self.server_max_load - self.server_idle_load), 0, 1)
        self.server_flow = np.clip(self.server_flow * self.server_temp_cpu / cpu_target_temp, self.server_idle_flow, self.server_max_flow)
        
        self.server_fan_power = np.sum(self.server_max_fan_power * (self.server_flow / self.server_max_flow)**3)

    def reward(self):
        total_energy = (self.server_fan_power + self.crah_fan_power + self.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_drop_cost = self.job_drop_cost * self.dropped_jobs
        total_cost = self.total_energy_cost + self.total_job_drop_cost
        return -total_cost

    def step(self, action):
        if self.load_balanced:
            placement, crah_settings = action
        else:
            crah_settings = action
            placement = np.argmin(self.server_load)

        # Increment time
        self.time += self.dt

        # Place jobs and remove finished jobs
        self.dropped_jobs = 0
        load, dur = self.job # Always here, but load and dur is zero if no job
        if load == 0:
            pass # No job
        elif self.server_load[placement] + load <= self.server_max_load:
            self.server_load[placement] += load
            heapq.heappush(self.running_jobs, (self.time + dur, load, placement))
        else:
            self.dropped_jobs = 1
        while len(self.running_jobs) > 0 and self.running_jobs[0][0] <= self.time:
            _, load, placement = heapq.heappop(self.running_jobs)
            self.server_load[placement] -= load

        # Read data from sensors
        self.read_data()

        # Update servers
        self.update_server()

        # Update CRAH
        self.update_crah(self.action_transform(crah_settings))

        # Vary ambient temp during day between 10 and 20 C
        # self.ambient_temp = np.sin(self.time / (60*60*24) * 2*np.pi) * 5 + 15

        self.job = self.load_generator.step(self.dt)

        state = np.concatenate((self.server_temp_out, self.server_load, self.job))
        return self.state_transform(state), self.reward(), False, {}

    def read_data(self):
        server_flow_total = np.sum(self.server_flow)
        crah_flow_total = np.sum(self.crah_flow)
        # Server temp out 
        self.server_temp_out = self.server_temp_in + self.server_load / (self.server_flow * self.air_vol_heatcap)
        server_temp_out_total = np.dot(self.server_flow, self.server_temp_out) / server_flow_total
        # Find new input temperatures
        bypass = min(1, crah_flow_total / server_flow_total)
        recirc = min(1, server_flow_total / crah_flow_total)
        self.server_temp_in = (bypass * self.crah_temp_out[0] + (1 - bypass) * server_temp_out_total) * np.ones(self.n_servers)
        self.crah_temp_in = (recirc * server_temp_out_total + (1 - recirc) * self.crah_temp_out[0]) * np.ones(self.n_crah)

    def get_time(self):
        return self.time

    def setup_physical_constants(self):
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

        self.server_max_load = 500  # W
        self.server_idle_load = 200
        self.server_max_temp_cpu = 85  # C
        self.server_idle_temp_cpu = 35
        # RPMs for fans
        # server_fan_idle_rpm = 2000
        # server_fan_max_rpm = 11000.0
        # Max air flow in CFM, 2 fans, converted to m3/s
        # self.server_max_flow = 109.7 * 0.000471947443
        # self.server_idle_flow = self.server_max_flow * server_fan_idle_rpm / server_fan_max_rpm
        self.server_idle_flow = 0.01
        self.server_max_flow = 0.03 # If using 5 we get NaN from rafsine
        # Max input power in W, 2 fans
        self.server_max_fan_power = 25.2 * 2 

        # CRAH constants
        self.crah_min_temp = 18
        self.crah_max_temp = 27
        # self.crah_min_flow = self.n_servers * self.server_idle_flow / self.n_crah # Minimal CRAH is minimal server
        # self.crah_max_flow = self.n_servers * self.server_max_flow * 2 / self.n_crah # Allow CRAH to be up to twice as strong as servers
        # print(self.crah_min_flow) # 0.8
        # print(self.crah_max_flow) # 6.4
        self.crah_min_flow = 0.4
        self.crah_max_flow = 2.2
        self.crah_max_fan_power = self.n_servers * self.server_max_fan_power / self.n_crah # Should this be smaller to indicate that CRAH fan more efficient?

        # Env constants
        self.ambient_temp = 17
