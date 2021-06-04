import numpy as np
import gym

from dc.servers import Servers
from dc.crah import CRAH

class DCEnv(gym.Env):
    def __init__(self, config={}):
        self.dt = config.get("dt", 1)
        self.energy_cost = config.get("energy_cost", 0.00001)
        self.job_drop_cost = config.get("job_drop_cost", 100.0)
        self.overheat_cost = config.get("overheat_cost", 1.0)
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

        self.actions = config.get("actions", ["server", "crah_out", "crah_flow"])
        self.observations = config.get("observations", ["temp_out", "load", "job"])

        # Ambient temp
        self.outdoor_temp = config["outdoor_temp"]()

        # Gym environment stuff
        # Generate all individual action spaces
        action_spaces_agent = {
            "none": gym.spaces.Discrete(2), # If running with other algorithms
            "rack": gym.spaces.Discrete(self.flowsim.n_racks), 
            "server": gym.spaces.Discrete(self.flowsim.n_servers), 
            "crah_out": gym.spaces.Box(-1, 1, shape=(1,)),
            "crah_flow": gym.spaces.Box(-1, 1, shape=(1,)),
        }
        action_spaces_env = {
            "none": gym.spaces.Discrete(2),
            "rack": gym.spaces.Discrete(self.flowsim.n_racks), 
            "server": gym.spaces.Discrete(self.flowsim.n_servers), 
            "crah_out": gym.spaces.Box(self.crah.min_temp, self.crah.max_temp, shape=(1,)),
            "crah_flow": gym.spaces.Box(self.crah.min_flow, self.crah.max_flow, shape=(1,)),
        }
        # Put it together based on chosen actions
        self.action_space = gym.spaces.Tuple(tuple(map(action_spaces_agent.__getitem__, self.actions)))
        self.action_space_env = gym.spaces.Tuple(tuple(map(action_spaces_env.__getitem__, self.actions)))

        # All individual observation spaces
        observation_spaces = {
            "load": gym.spaces.Box(-100, 100, shape=(self.n_servers,)),
            "temp_out": gym.spaces.Box(-100, 100, shape=(self.n_servers,)),
            "outdoor_temp": gym.spaces.Box(-100, 100, shape=(1,)),
            "job": gym.spaces.Box(-100, 100, shape=(1,)),
        }
        observation_spaces_target = {
            "load": gym.spaces.Box(-1, 1, shape=(self.n_servers,)),
            "temp_out": gym.spaces.Box(-1, 1, shape=(self.n_servers,)),
            "outdoor_temp": gym.spaces.Box(-1, 1, shape=(1,)),
            "job": gym.spaces.Box(-1, 1, shape=(1,)),
        }
        observation_spaces_env = {
            "load": gym.spaces.Box(self.servers.idle_load, self.servers.max_load, shape=(self.n_servers,)),
            #"temp_out": gym.spaces.Box(-10, self.servers.max_temp_cpu+10, shape=(self.n_servers,)),
            "temp_out": gym.spaces.Box(15, 85, shape=(self.n_servers,)),
            "outdoor_temp": gym.spaces.Box(0, 30, shape=(1,)),
            "job": gym.spaces.Box(0, 1, shape=(1,)),
            #"job": gym.spaces.Box(np.array(self.load_generator.min_values()), np.array(self.load_generator.max_values())),
        }
        # Put it together based on chosen observations
        # The real space is just made bigger than the target to fit anything that falls outside, only needed for ray to be happy
        self.observation_space = gym.spaces.Tuple(tuple(map(observation_spaces.__getitem__, self.observations)))
        # The target space is -1..1
        self.observation_space_target = gym.spaces.Tuple(tuple(map(observation_spaces_target.__getitem__, self.observations)))
        # The source space is what we approximate the values to be within in the environment
        self.observation_space_env = gym.spaces.Tuple(tuple(map(observation_spaces_env.__getitem__, self.observations)))

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.load_generator.seed(seed)
        
    def reset(self):
        self.time = 0

        self.servers.reset(self.outdoor_temp(self.time))
        self.crah.reset(self.outdoor_temp(self.time))

        self.flowsim.reset(self.servers, self.crah)

        total_energy = (self.servers.fan_power + self.crah.fan_power + self.crah.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_drop_cost = self.job_drop_cost * self.servers.dropped_jobs
        self.total_overheat_cost = self.overheat_cost * self.servers.overheated_inlets

        self.job = self.load_generator(self.time)

        state = self.get_state()
        return state

    def step(self, action):
        clipped_action = map(lambda x: self.clip_action(*x), zip(action, self.action_space))
        rescaled_action = map(lambda x: self.scale_to(*x), zip(clipped_action, self.action_space, self.action_space_env))
        action = dict(zip(self.actions, rescaled_action))
        if "rack" in action:
            rack_placement = action.get("rack")
            start = rack_placement * self.flowsim.servers_per_rack
            end = (rack_placement + 1) * self.flowsim.servers_per_rack
            placement = start + np.argmin(self.servers.load[start:end])
        elif "server" in action:
            placement = action.get("server")
        else:
            placement = np.argmin(self.servers.load[:self.n_place])

        self.time += self.dt

        self.servers.update(self.time, self.dt, placement, self.job[0], self.job[1], self.flowsim.server_temp_in)

        # Update CRAH fans
        crah_temp = action.get("crah_out", self.crah_out_setpoint)
        crah_flow = action.get("crah_flow", self.crah_flow_setpoint * self.crah.max_flow)
        self.crah.update(crah_temp, crah_flow, self.flowsim.crah_temp_in, self.outdoor_temp(self.time))

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
        Return a tuple of rescaled observations based on selected observations in self.observations
        """
        states = {
            "load": self.servers.load,
            "temp_out": self.flowsim.server_temp_out,
            "outdoor_temp": self.outdoor_temp(self.time),
            "job": 0 if self.job == (0, 0) else 1,
        }
        state = tuple(map(lambda x: self.scale_to(*x), zip(
            map(states.__getitem__, self.observations), 
            self.observation_space_env, 
            self.observation_space_target)))
        return state
    
    def scale_to(self, x, original_range, target_range):
        """
        If supplied range is of type Box do a linear mapping from source to target
        """
        if isinstance(original_range, gym.spaces.Box):
            return (x - original_range.low) * (target_range.high - target_range.low) / (original_range.high - original_range.low) + target_range.low
        else: # Can't handle rescaling other types
            return x

    def clip_action(self, a, allowed_range):
        """
        Clip action based on the allowed range, only do it for Box space
        """
        if isinstance(allowed_range, gym.spaces.Box):
            return np.clip(a, allowed_range.low, allowed_range.high)
        else: # Can't handle rescaling other types
            return a