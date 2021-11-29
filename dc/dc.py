import numpy as np
import gym

from dc.servers import Servers
from dc.crah import CRAH
from loads.workloads import RandomArrival
from loads.temperatures import CSVTemperature, SinusTemperature

class DCEnv(gym.Env):
    def __init__(self, config={}):
        self.dt = config.get("dt", 1)
        self.energy_cost = config.get("energy_cost", 0.00001)
        self.job_misplace_cost = config.get("job_misplace_cost", 100.0)
        self.overheat_cost = config.get("overheat_cost", 1.0)
        self.load_variance_cost = config.get("load_variance_cost", 0.0) # Around 0.0001 seems reasonable, but does not seem to do very well. 
        self.crah_out_setpoint = config.get("crah_out_setpoint", 22)
        self.loglevel = config.get("loglevel", 1)
        self.break_after = config.get("break_after", -1)
        self.placement_block_size = config.get("placement_block_size", 1)

        if config.get("rafsine_flow", False):
            from dc.rafsineflow import RafsineFlow
            self.flowsim = RafsineFlow(self.dt)
        else:
            from dc.simpleflow import SimpleFlow
            self.flowsim = SimpleFlow(self.dt, config.get("n_servers", 360), config.get("n_racks", 12), config.get("n_crah", 4))

        self.n_servers = self.flowsim.n_servers
        self.n_crah = self.flowsim.n_crah

        nu = 1.568e-5 # Kinematic viscosity of air (m^2/s)
        k = 2.624e-2 # Thermal conductivity (W/(m*K))
        Pr = 0.707 # Prandtl number of air
        air_vol_heatcap = Pr * k / nu 
        R = 3 / air_vol_heatcap

        self.servers = Servers(self.n_servers, air_vol_heatcap, R)
        self.crah = CRAH(self.n_crah, air_vol_heatcap)

        self.crah_flow_setpoint = config.get("crah_flow_setpoint", 0.8) * self.crah.max_flow
        self.crah_flow_efficiency = np.ones(self.n_crah)

        # Jobs
        if "load_generator" in config:
            self.load_generator = config["load_generator"]()
        else:
            job_p = config.get("job_p", 0.5)
            # job_pf = lambda t: job_p - 0.8 * job_p * (t > 24*3600) # Why is this hard?
            job_load = config.get("job_load", 20) # Used to be 20
            self.load_generator = RandomArrival(job_load, duration=self.dt * config.get("avg_load", 200) * self.n_servers / (job_load * job_p), p=job_p)

        # Outdoor temp
        outdoor_temp = config.get("outdoor_temp", "loads/smhi_temp.csv")
        if callable(outdoor_temp): # Callable registered with tune that creates temperature object
            self.outdoor_temp = outdoor_temp()
        elif type(outdoor_temp) == str:
            self.outdoor_temp = CSVTemperature(outdoor_temp)
        else:
            self.outdoor_temp = SinusTemperature(offset=outdoor_temp[0], amplitude=outdoor_temp[1])

        self.actions = config.get("actions", ["place", "crah_out", "crah_flow"])
        self.observations = config.get("observations", ["temp_out", "load", "outdoor_temp", "job"])

        self.server_placement_indices = config.get("place_load_indices", range(0, self.n_servers))
        self.autoplace = config.get("autoplace", None)

        # Gym environment stuff
        # Generate all individual action spaces
        assert self.flowsim.servers_per_rack % self.placement_block_size == 0 
        blocks = self.flowsim.n_servers // self.placement_block_size
        action_spaces_agent = {
            "none": gym.spaces.Discrete(2), # If running with other algorithms
            "place": gym.spaces.Discrete(blocks), 
            "crah_out": gym.spaces.Box(-1, 1, shape=(self.n_crah,)),
            "crah_flow": gym.spaces.Box(-1, 1, shape=(self.n_crah,)),
        }
        action_spaces_env = {
            "none": gym.spaces.Discrete(2),
            "place": gym.spaces.Discrete(blocks), 
            "crah_out": gym.spaces.Box(self.crah.min_temp, self.crah.max_temp, shape=(self.n_crah,)),
            "crah_flow": gym.spaces.Box(self.crah.min_flow, self.crah.max_flow, shape=(self.n_crah,)),
        }
        # Put it together based on chosen actions
        self.action_space = gym.spaces.Tuple(tuple(map(action_spaces_agent.__getitem__, self.actions)))
        self.action_space_env = gym.spaces.Tuple(tuple(map(action_spaces_env.__getitem__, self.actions)))

        # All individual observation spaces
        observation_spaces = {
            "load": gym.spaces.Box(-100, 100, shape=(blocks,)),
            "temp_out": gym.spaces.Box(-100, 100, shape=(blocks,)),
            "temp_in": gym.spaces.Box(-100, 100, shape=(blocks,)),
            "flow": gym.spaces.Box(-100, 100, shape=(blocks,)),
            "outdoor_temp": gym.spaces.Box(-100, 100, shape=(1,)),
            "job": gym.spaces.Box(-100, 100, shape=(1,)),
        }
        observation_spaces_target = {
            "load": gym.spaces.Box(-1, 1, shape=(blocks,)),
            "temp_out": gym.spaces.Box(-1, 1, shape=(blocks,)),
            "temp_in": gym.spaces.Box(-1, 1, shape=(blocks,)),
            "flow": gym.spaces.Box(-1, 1, shape=(blocks,)),
            "outdoor_temp": gym.spaces.Box(-1, 1, shape=(1,)),
            "job": gym.spaces.Box(0, 1, shape=(1,)), # We want to use 0 jobs to kill gradient, so no rescale here
        }
        observation_spaces_env = {
            "load": gym.spaces.Box(self.servers.idle_load, self.servers.max_load, shape=(blocks,)),
            "temp_in": gym.spaces.Box(18, 27, shape=(blocks,)),
            "temp_out": gym.spaces.Box(15, 85, shape=(blocks,)),
            "flow": gym.spaces.Box(self.servers.min_flow, self.servers.max_flow, shape=(blocks,)),
            "outdoor_temp": gym.spaces.Box(0, 30, shape=(1,)),
            "job": gym.spaces.Box(0, 1, shape=(1,)),
        }
        # Put it together based on chosen observations
        # The real space is just made bigger than the target to fit anything that falls outside, only needed for ray to be happy
        self.observation_space = gym.spaces.Tuple(tuple(map(observation_spaces.__getitem__, self.observations)))
        # The target space is -1..1
        self.observation_space_target = gym.spaces.Tuple(tuple(map(observation_spaces_target.__getitem__, self.observations)))
        # The source space is what we approximate the values to be within in the environment
        self.observation_space_env = gym.spaces.Tuple(tuple(map(observation_spaces_env.__getitem__, self.observations)))

    def seed(self, seed):
        # env0 will always be seeded the same, env1 has +1 and so on
        self.seed = seed 
        self.rng = np.random.default_rng(seed)
        hour_offset = self.rng.integers(low=0, high=24) # Maybe good to have random offset when trainign? Maybe less realistic?
        self.load_generator.seed(seed, hour_offset)
        self.outdoor_temp.seed(seed, hour_offset)
        
    def reset(self):
        self.time = 0

        self.servers.reset(self.outdoor_temp(self.time))
        self.crah.reset(self.outdoor_temp(self.time))

        self.flowsim.reset(self.servers, self.crah)

        total_energy = (self.servers.fan_power + self.crah.fan_power + self.crah.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_misplace_cost = self.job_misplace_cost * self.servers.misplaced_jobs
        self.total_overheat_cost = self.overheat_cost * self.servers.overheated_inlets
        self.total_load_variance_cost = self.load_variance_cost * np.var(self.servers.load)

        self.job = self.load_generator(self.time)

        state = self.get_state()
        return state

    def step(self, action):
        """
        Do `action` and step environment forward by `dt`.
        """
        clipped_action = map(lambda x: self.clip_action(*x), zip(action, self.action_space))
        rescaled_action = map(lambda x: self.scale_to(*x), zip(clipped_action, self.action_space, self.action_space_env))
        action = dict(zip(self.actions, rescaled_action))
        if "place" in action:
            block_placement = action.get("place")
            start_idx = block_placement * self.placement_block_size
            end_idx = (block_placement + 1) * self.placement_block_size
            if self.autoplace == "minload":
                placement = start_idx + np.argmin(self.servers.load[start_idx:end_idx])
            elif self.autoplace == "minflow":
                placement = start_idx + np.argmin(self.servers.flow[start_idx:end_idx])
            elif self.autoplace == "mintempout":
                placement = start_idx + np.argmin(self.servers.temp_out[start_idx:end_idx])
            else:
                raise "no valid value for autoplace"
        else:
            if self.autoplace == "minload":
                placement = self.server_placement_indices[np.argmin(self.servers.load[self.server_placement_indices])]
            elif self.autoplace == "minflow":
                placement = self.server_placement_indices[np.argmin(self.servers.flow[self.server_placement_indices])]
            elif self.autoplace == "mintempout":
                placement = self.server_placement_indices[np.argmin(self.flowsim.server_temp_out[self.server_placement_indices])]
            else:
                raise "no valid value for autoplace"

        self.time += self.dt

        self.servers.update(self.time, self.dt, placement, self.job[0], self.job[1], self.flowsim.server_temp_in)

        # Update CRAH fans
        if self.break_after >= 0 and self.time >= self.break_after:
            flow_efficiency = 0.8
            crah_idx = 0
            self.crah_flow_efficiency[crah_idx] *= flow_efficiency
            self.crah.max_flow *= self.crah_flow_efficiency
            self.break_after = -1
        crah_temp = action.get("crah_out", self.crah_out_setpoint * np.ones(4))
        crah_flow = action.get("crah_flow", self.crah_flow_setpoint * np.ones(4))
        crah_flow *= self.crah_flow_efficiency
        self.crah.update(crah_temp, crah_flow, self.flowsim.crah_temp_in, self.outdoor_temp(self.time))

        # Run simulation based on current boundary condition
        self.flowsim.step(self.servers, self.crah)

        # Get new job, tuple of expected (load, duration)
        self.job = self.load_generator(self.time)

        total_energy = (self.servers.fan_power + self.crah.fan_power + self.crah.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_misplace_cost = self.job_misplace_cost * self.servers.misplaced_jobs
        # self.total_overheat_cost = self.overheat_cost * self.servers.overheated_inlets
        # For rafsine?
        # avg_temp_in = np.dot(self.flowsim.server_temp_in, self.servers.flow) / np.sum(self.servers.flow)
        # self.total_overheat_cost = 10 * max(0, avg_temp_in - 27)
        self.total_overheat_cost = self.overheat_cost * np.mean(np.maximum(0, self.flowsim.server_temp_in - 27))
        self.total_load_variance_cost = self.load_variance_cost * np.var(self.servers.load)
        total_cost = self.total_energy_cost + self.total_job_misplace_cost + self.total_overheat_cost + self.total_load_variance_cost
        reward = -total_cost

        state = self.get_state()
        return state, reward, False, {}

    def get_state(self):
        """
        Return a tuple of rescaled observations based on selected observations in self.observations
        """
        states = {
            "load": np.mean(self.servers.load.reshape((-1, self.placement_block_size)), axis=1),
            "temp_out": np.mean(self.flowsim.server_temp_out.reshape((-1, self.placement_block_size)), axis=1),
            "temp_in": np.mean(self.flowsim.server_temp_in.reshape((-1, self.placement_block_size)), axis=1),
            "flow": np.mean(self.servers.flow.reshape((-1, self.placement_block_size)), axis=1),
            "job": 0 if self.job == (0, 0) else 1,
            "outdoor_temp": self.outdoor_temp(self.time),
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
        if isinstance(original_range, gym.spaces.Box): # Both are box
            return (x - original_range.low) * (target_range.high - target_range.low) / (original_range.high - original_range.low) + target_range.low
        else: # Otherwise don't rescale
            return x

    def clip_action(self, a, allowed_range):
        """
        Clip action based on the allowed range, only do it for Box space
        """
        if isinstance(allowed_range, gym.spaces.Box):
            return np.clip(a, allowed_range.low, allowed_range.high)
        else: # Can't handle rescaling other types
            return a