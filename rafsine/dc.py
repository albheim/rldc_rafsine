import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input

import pandas as pd
import os
import heapq
import numpy as np
import gym

RAFSINE_BASE = "/home/ubuntu/rafsine"
import sys
sys.path.insert(0, RAFSINE_BASE)

from python.simulation import Simulation
from job import ConstantArrival

class DCEnv(gym.Env):
    def __init__(self, config={}):
        # Parameters with default values
        self.dt = config.get("dt", 1)
        self.seed = config.get("seed", 37)

        # Server layout
        self.racks = 12
        self.chassis_per_rack = 10
        self.servers_per_chassi = 3

        self.servers = ["P02R{:02}C{:02}SRV{:02}".format(rack, chassi, srv) for rack in range(1, self.racks+1) for chassi in range(1, self.chassis_per_rack+1) for srv in range(1, self.servers_per_chassi+1)]
        self.sensors = ["sensors_racks_{:02}_to_{:02}_{}_{}".format(rack, rack + 2, direction, loc) for direction in ["in", "out"] for rack in [1, 4, 7, 10] for loc in ['b', 'm', 't']]
        self.crah = ["P02HDZ{:02}".format(i+1) for i in range(4)]

        self.n_servers = len(self.servers)
        self.n_sensors = len(self.sensors) 

        # Setup simulation constants
        self.setup_physical_constants()

        # Jobs
        self.load_generator = ConstantArrival(load=20, duration=200)

        # Gym environment stuff
        self.action_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.n_servers),
            gym.spaces.Box(-1, 1, shape=(2,))))
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(2 * self.n_servers + 2,))
        
        # Conversion variables for normalizing the state
        self.slow = np.concatenate((20 * np.ones(self.n_servers), self.server_idle_load * np.ones(self.n_servers), [0, 0]))
        self.shigh = np.concatenate((80 * np.ones(self.n_servers), self.server_max_load * np.ones(self.n_servers), [400, 3600]))

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

        cwd = os.getcwd()
        os.chdir(RAFSINE_BASE)
        self.sim = Simulation(f"{RAFSINE_BASE}/problems/ocp/project.lbm")
        os.chdir(cwd)

        self.server_load = self.server_idle_load * np.ones(self.n_servers)
        self.server_fan_flow = np.array([self.server_idle_flow for _ in range(self.n_servers)])

        self.sim.set_time_averaging_period(self.dt) # we take one second steps and average sensor readings over that time also
        self.sim.set_boundary_conditions(self.servers, [0 for _ in range(self.n_servers)], self.server_fan_flow)
        self.sim.run(1) # Run single step so temps exists when checked first time

        self.read_data()

        self.update_server()

        self.event_queue = []

        self.start_time = self.sim.get_time()
        self.target_time = 0

        self.job = self.load_generator.step(self.dt)

        state = np.concatenate((self.server_temp_out, self.server_load, self.job))
        return self.state_transform(state)

    def state_transform(self, state):
        state = (2 * state - self.slow - self.shigh) / (self.shigh - self.slow)
        return state
    
    def queue_load(self, srv_idx, power, start, dur):
        heapq.heappush(self.event_queue, (start, srv_idx, power, dur))

    def get_time(self):
        """Returns the time in second that has passed in the simulation since it started."""
        return (self.sim.get_time() - self.start_time).total_seconds()

    def update_crah(self, settings):
        flow, temp_out = settings
        delta_t = temp_out - self.crah_temp_in
        flow = flow * np.ones(len(self.crah))
        self.sim.set_boundary_conditions(self.crah, delta_t, flow)
        self.crah_fan_power = np.sum(self.crah_max_fan_power * (flow / self.crah_max_flow)**3)

        # For logging
        self.crah_flow = flow
        self.crah_temp_out = temp_out

        if self.ambient_temp < temp_out:
            self.compressor_power = 0
        else:
            self.compressor_power = np.sum(self.air_vol_heatcap * flow * delta_t)

    def update_server(self):
        self.cpu_temp = self.server_temp_in + self.R * self.server_load / self.server_fan_flow
        cpu_target_temp = self.server_idle_temp_cpu + (self.server_max_temp_cpu - self.server_idle_temp_cpu) * np.clip((self.server_load - self.server_idle_load) / (self.server_max_load - self.server_idle_load), 0, 1)
        #N = Nmin + (Nmax - Nmin) * clip(Tcpu / Tdes, 0, 1)
        self.server_flow = np.clip(self.server_fan_flow * self.cpu_temp / cpu_target_temp, self.server_idle_flow, self.server_max_flow)
        self.server_fan_power = np.sum(self.server_max_fan_power * (self.server_flow / self.server_max_flow)**3)

        self.sim.set_boundary_conditions(self.servers, self.server_temp_out - self.server_temp_in, self.server_fan_flow)

    def get_reward(self):
        total_energy = (self.server_fan_power + self.crah_fan_power + self.compressor_power) * self.dt
        energy_cost = 0.001 * total_energy
        return -energy_cost

    def step(self, action):
        placement, crah_settings = action
        (load, duration) = self.job

        # Increment time
        self.target_time += self.dt

        # Read new data from sim
        self.read_data()
        
        # Update CPU fans
        self.update_server()

        # Update CRAH fans
        self.update_crah(crah_settings)

        # Place jobs in queue
        self.queue_load(placement, load, self.get_time(), duration)
        
        # Run jobs if space and remove if done
        while len(self.event_queue) > 0 and self.event_queue[0][0] <= self.target_time:
            t, srv_idx, power, dur = heapq.heappop(self.event_queue)
            if t > self.get_time():
                self.sim.run(t - self.get_time())
            if dur < 0: # This is a finished job
                self.server_load[srv_idx] -= power
            elif self.server_load[srv_idx] < self.server_max_load: # This is a job that is placed on a server
                self.server_load[srv_idx] += power
                heapq.heappush(self.event_queue, (self.get_time() + dur, srv_idx, power, -1))
            else: # Server is full, job is queued
                heapq.heappush(self.event_queue, (self.get_time() + 1, srv_idx, power, dur))
        if self.target_time > self.get_time():
            self.sim.run(self.target_time - self.get_time())

        # Get new job, tuple of expected (load, duration)
        self.job = self.load_generator.step(self.dt)
        
        state = np.concatenate((self.server_temp_out, self.server_load, self.job))
        return self.state_transform(state), self.get_reward(), False, {}

    def read_data(self):
        df = self.sim.get_averages("temperature")
        self.server_temp_in = df[[*map(lambda x: x + "_inlet", self.servers)]].iloc[[-1]].to_numpy()[0]
        self.server_temp_out = df[[*map(lambda x: x + "_outlet", self.servers)]].iloc[[-1]].to_numpy()[0]
        self.crah_temp_in = df[[*map(lambda x: x + "_in", self.crah)]].iloc[[-1]].to_numpy()[0]
        #self.sensors = df[[*self.sensors]].iloc[[-1]].to_numpy()[0]

    def setup_physical_constants(self):
        self.R = 0.005

        # Kinematic viscosity of air (m^2/s)
        nu = 1.568e-5
        # Thermal conductivity (W/m K)
        k = 2.624e-2
        # Prandtl number of air
        Pr = 0.707
        self.air_vol_heatcap = Pr * k / nu 
        #self.air_vol_heatcap = 1000 * 1.225

        # Server constants
        self.server_max_load = 500  # W
        self.server_idle_load = 200
        self.server_max_temp_cpu = 85  # C
        self.server_idle_temp_cpu = 35
        # RPMs for fans
        server_fan_idle_rpm = 2000
        server_fan_max_rpm = 11000.0
        # Max air flow in CFM, 2 fans, converted to m3/s
        self.server_max_flow = 109.7 * 2 * 0.000471947443
        self.server_idle_flow = self.server_max_flow * server_fan_idle_rpm / server_fan_max_rpm
        # Max input power in W, 2 fans
        self.server_max_fan_power = 25.2 * 2 

        # CRAH constants
        self.crah_min_temp = 18
        self.crah_max_temp = 27
        self.crah_min_flow = self.n_servers * self.server_idle_flow # Minimal CRAH is minimal server
        self.crah_max_flow = self.n_servers * self.server_max_flow * 2 # Allow CRAH to be up to twice as strong as servers
        self.crah_max_fan_power = self.n_servers * self.server_max_fan_power # CRAH is twice as efficient (max is same energy as servers but double flow)

        # Env constants
        self.ambient_temp = 22
