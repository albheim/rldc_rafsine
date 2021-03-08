import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input

import pandas as pd
import os
import heapq
import numpy as np
import gym
import time

from .simpleflow import SimpleFlow
from .rafsineflow import RafsineFlow
from .servers import Servers
from .crah import CRAH

class DCEnv(gym.Env):
    def __init__(self, config={}):
        self.dt = config.get("dt", 1)
        self.seed = config.get("seed", 37)
        self.energy_cost = config.get("energy_cost", 0.00001)
        self.job_drop_cost = config.get("job_drop_cost", 10.0)

        if config.get("rafsine_flow", True):
            self.flowsim = RafsineFlow(self.dt)
        else:
            self.flowsim = SimpleFlow(self.dt, config.get("n_servers", 360), config.get("n_crah", 4))

        self.n_servers = self.flowsim.n_servers
        self.n_crah = self.flowsim.n_crah

        self.servers = Servers(self.n_servers)
        self.crah = CRAH(self.n_crah)

        # Jobs
        self.load_generator = config["load_generator"]

        self.actions = config.get("actions", ["placement", "crah_out", "crah_flow"])
        self.observations = config.get("observations", ["load", "temp_out"])

        # Gym environment stuff
        self.load_balanced = config.get("load_balanced", True)
        if self.load_balanced:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        else:
            self.action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(self.n_servers), 
                gym.spaces.Box(-1.0, 1.0, shape=(2,))))
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(2 * self.n_servers + 2,))
        
        # Conversion variables for state/action mapping normalization
        self.alow = np.array((self.crah.min_temp, self.crah.min_flow))
        self.ahigh = np.array((self.crah.max_temp, self.crah.max_flow))
        self.slow = np.concatenate((20 * np.ones(self.n_servers), self.servers.idle_load * np.ones(self.n_servers), self.load_generator.min_values()))
        self.shigh = np.concatenate((80 * np.ones(self.n_servers), self.servers.max_load * np.ones(self.n_servers), self.load_generator.max_values()))

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

        self.ambient_temp = 20

        self.servers.reset(self.ambient_temp)
        self.crah.reset(self.ambient_temp)

        self.flowsim.reset(self.servers, self.crah)

        total_energy = (self.servers.fan_power + self.crah.fan_power + self.crah.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_drop_cost = self.job_drop_cost * self.servers.dropped_jobs

        self.time = 0

        self.job = self.load_generator(self.time, self.dt)

        state = np.concatenate((self.flowsim.server_temp_out, self.servers.load, self.job))
        return self.state_transform(state)

    def step(self, action):
        if self.load_balanced:
            crah_settings = action
            placement = np.argmin(self.servers.load)
        else:
            placement, crah_settings = action

        self.time += self.dt

        self.servers.update(self.time, placement, self.job[0], self.job[1], self.flowsim.server_temp_in)

        # Update CRAH fans
        temp_out, flow = self.action_transform(crah_settings)
        self.crah.update(temp_out, flow, self.flowsim.crah_temp_in, self.ambient_temp)

        # Run simulation based on current boundary condition
        self.flowsim.step(self.servers, self.crah)

        # Get new job, tuple of expected (load, duration)
        self.job = self.load_generator(self.time, self.dt)

        state = np.concatenate((self.flowsim.server_temp_out, self.servers.load, self.job))
        
        total_energy = (self.servers.fan_power + self.crah.fan_power + self.crah.compressor_power) * self.dt
        self.total_energy_cost = self.energy_cost * total_energy 
        self.total_job_drop_cost = self.job_drop_cost * self.servers.dropped_jobs
        total_cost = self.total_energy_cost + self.total_job_drop_cost
        reward = -total_cost

        return self.state_transform(state), reward, False, {}

    def state_transform(self, state):
        return (2 * state - self.slow - self.shigh) / (self.shigh - self.slow)

    def action_transform(self, crah):
        return (np.clip(crah, -1, 1) * (self.ahigh - self.alow) + self.alow + self.ahigh) / 2
    