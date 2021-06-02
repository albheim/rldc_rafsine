import numpy as np
import gym

from dc.servers import Servers
from dc.crah import CRAH

class DCEnv(gym.Env):
    def __init__(self, config={}):
        self.dt = config.get("dt", 1)
        self.energy_cost = config.get("energy_cost", 0.00001)
        self.job_drop_cost = config.get("job_drop_cost", 50.0)
        self.overheat_cost = config.get("overheat_cost", 0.1)
        self.crah_out_setpoint = config.get("crah_out_setpoint", 22)
        self.crah_flow_setpoint = config.get("crah_flow_setpoint", 0.8)

        if config.get("rafsine_flow", True):
            from dc.rafsineflow import RafsineFlow
            self.flowsim = RafsineFlow(self.dt)
        else:
            from dc.simpleflow import SimpleFlow
            self.flowsim = SimpleFlow(self.dt, config.get("n_servers", 360), config.get("n_racks", 12), config.get("n_crah", 4))

        self.n_servers = self.flowsim.n_servers
        self.n_crah = self.flowsim.n_crah

        self.n_place = config.get("n_place", self.n_servers)

        nu = 1.568e-5 # Kinematic viscosity of air (m^2/s)
        k = 2.624e-2 # Thermal conductivity (W/m K)
        Pr = 0.707 # Prandtl number of air
        air_vol_heatcap = Pr * k / nu 
        R = config.get("kR", 3) / air_vol_heatcap

        self.servers = Servers(self.n_servers, air_vol_heatcap, R)
        self.crah = CRAH(self.n_crah, air_vol_heatcap)

        # Jobs
        self.load_generator = config["load_generator"]()

        self.actions = config.get("actions", ["server_placement", "crah_temp_out", "crah_flow"])
        self.observations = config.get("observations", ["server_temp_out", "server_load", "job"])

        # Ambient temp
        self.ambient_temp = config["ambient_temp"]()

        # Gym environment stuff
        # Generate all individual action spaces
        action_spaces_agent = {
            "none": gym.spaces.Discrete(2), # If running with other algorithms
            "server_placement": gym.spaces.Discrete(self.flowsim.n_servers), 
            "crah_temp_out": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
            "crah_flow": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        }
        action_spaces_env = {
            "none": gym.spaces.Discrete(2),
            "server_placement": gym.spaces.Discrete(self.flowsim.n_servers), 
            "crah_temp_out": gym.spaces.Box(self.crah.min_temp, self.crah.max_temp, shape=(1,)),
            "crah_flow": gym.spaces.Box(self.crah.min_flow, self.crah.max_flow, shape=(1,)),
        }
        # Put it together based on chosen actions
        self.action_space = gym.spaces.Dict({k: action_spaces_agent[k] for k in self.actions})
        self.action_space_env = action_spaces_env

        # All individual observation spaces
        observation_spaces = {
            "server_load": gym.spaces.Box(-100.0, 100.0, shape=(self.n_servers,)),
            "server_temp_out": gym.spaces.Box(-100.0, 100.0, shape=(self.n_servers,)),
            "job": gym.spaces.Box(-100.0, 100.0, shape=(1,)),
        }
        # Put it together based on chosen observations
        # The real space is just made bigger than the target to fit anything that falls outside, only needed for ray to be happy
        self.observation_space = gym.spaces.Dict({k: observation_spaces[k] for k in self.observations})
        self.observation_spaces_agent = {
            "server_load": gym.spaces.Box(-1.0, 1.0, shape=(self.n_servers,)),
            "server_temp_out": gym.spaces.Box(-1.0, 1.0, shape=(self.n_servers,)),
            "job": gym.spaces.Box(-1.0, 1.0, shape=(1,)),
        }
        self.observation_spaces_env = {
            "server_load": gym.spaces.Box(self.servers.idle_load, self.servers.max_load, shape=(self.n_servers,)),
            #"temp_out": gym.spaces.Box(-10, self.servers.max_temp_cpu+10, shape=(self.n_servers,)),
            "server_temp_out": gym.spaces.Box(15, 85, shape=(self.n_servers,)),
            "job": gym.spaces.Box(0, 1, shape=(1,)),
            #"job": gym.spaces.Box(np.array(self.load_generator.min_values()), np.array(self.load_generator.max_values())),
        }

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.load_generator.seed(seed)
        
    def reset(self):
        self.time = 0

        self.servers.reset(self.ambient_temp(self.time))
        self.crah.reset(self.ambient_temp(self.time))

        self.flowsim.reset(self.servers, self.crah)

        total_energy = (self.servers.fan_power + self.crah.fan_power + self.crah.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_drop_cost = self.job_drop_cost * self.servers.dropped_jobs
        self.total_overheat_cost = self.overheat_cost * self.servers.overheated_inlets

        self.job = self.load_generator(self.time)

        state = self.get_state()
        return state

    def step(self, action):
        clipped_action = self.clip_action(action)
        action = self.scale_to(clipped_action, self.action_space, self.action_space_env)
        if "server" in action:
            placement = action.get("server_placement")
        else:
            placement = np.argmin(self.servers.load[:self.n_place])

        self.time += self.dt

        self.servers.update(self.time, self.dt, placement, self.job[0], self.job[1], self.flowsim.server_temp_in)

        # Update CRAH fans
        crah_temp = action.get("crah_temp_out", self.crah_out_setpoint)
        crah_flow = action.get("crah_flow", self.crah_flow_setpoint * self.crah.max_flow)
        self.crah.update(crah_temp, crah_flow, self.flowsim.crah_temp_in, self.ambient_temp(self.time))

        # Run simulation based on current boundary condition
        self.flowsim.step(self.servers, self.crah)

        # Get new job, tuple of expected (load, duration)
        self.job = self.load_generator(self.time)

        total_energy = (self.servers.fan_power + self.crah.fan_power + self.crah.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_drop_cost = self.job_drop_cost * self.servers.dropped_jobs
        # self.total_overheat_cost = self.overheat_cost * self.servers.overheated_inlets
        # For rafsine?
        # avg_temp_in = np.dot(self.flowsim.server_temp_in, self.servers.flow) / np.sum(self.servers.flow)
        # self.total_overheat_cost = 10 * max(0, avg_temp_in - 27)
        self.total_overheat_cost = self.overheat_cost * np.mean(np.maximum(0, self.flowsim.server_temp_in - 27))
        total_cost = self.total_energy_cost + self.total_job_drop_cost + self.total_overheat_cost
        reward = -total_cost

        state = self.get_state()
        return state, reward, False, {}

    def get_state(self):
        """
        Return a dict of rescaled observations based on selected observations in self.observations
        """
        all_states = {
            "server_load": self.servers.load,
            "server_temp_out": self.flowsim.server_temp_out,
            "job": np.array([0 if self.job == (0, 0) else 1]),
        }
        selected_states = {k: all_states[k] for k in self.observations}
        state = self.scale_to(selected_states, self.observation_spaces_env, self.observation_spaces_agent)
        return state
    
    def scale_to(self, x, source_space, target_space):
        """
        If supplied range is of type Box do a linear mapping from source to target
        """
        scaled_x = {}
        for k in x:
            if isinstance(source_space[k], gym.spaces.Box):
                scaled_x[k] = (x[k] - source_space[k].low) * (target_space[k].high - target_space[k].low) / (source_space[k].high - source_space[k].low) + target_space[k].low
            else: # Can't handle rescaling other types
                scaled_x[k] = x[k]
        return scaled_x

    def clip_action(self, a):
        """
        Clip action based on action space, only do it for Box space
        """
        clipped_a = {}
        for k in self.actions:
            space = self.action_space[k]
            if isinstance(space, gym.spaces.Box):
                clipped_a[k] = np.clip(a[k], space.low, space.high)
            else: # Can't handle rescaling other types
                clipped_a[k] = a[k]
        return clipped_a