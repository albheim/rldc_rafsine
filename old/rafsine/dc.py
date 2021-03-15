import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input

import pandas as pd
import os
import heapq
import numpy as np
import gym
import time
import sys

RAFSINE_BASE = "/home/ubuntu/rafsine"
sys.path.insert(0, RAFSINE_BASE)

from python.simulation import Simulation
from job import ConstantArrival

class RafsineDCEnv(gym.Env):
    def __init__(self, config={}):
        # Parameters with default values
        self.dt = config.get("dt", 1)
        self.seed = config.get("seed", 37)
        self.energy_cost = config.get("energy_cost", 0.00001)
        self.job_drop_cost = config.get("job_drop_cost", 1.0)

        # Server layout
        self.racks = 12
        self.chassis_per_rack = 10
        self.servers_per_chassi = 3

        self.servers = ["P02R{:02}C{:02}SRV{:02}".format(rack, chassi, srv) for rack in range(1, self.racks+1) for chassi in range(1, self.chassis_per_rack+1) for srv in range(1, self.servers_per_chassi+1)]
        self.crah = ["P02HDZ{:02}".format(i+1) for i in range(4)]

        self.n_servers = len(self.servers)
        self.n_crah = len(self.crah) 

        # Setup simulation constants
        self.setup_physical_constants()

        # Jobs
        # 360 servers with 200 idle load 
        # load * time * prob / 360 = avg added load to each server
        self.job_load = 20
        self.job_time = 3600
        self.load_generator = ConstantArrival(load=self.job_load, duration=self.job_time)
        #self.job_time = 3600
        #self.job_rate = 0.75 # Just tested to be around a nice number
        #self.job_load = 20
        #self.load_generator = RandomArrival(self.job_load, self.job_time, self.job_rate)


        # Gym environment stuff
        self.load_balanced = config.get("load_balanced", True)
        if self.load_balanced:
            self.action_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(self.n_servers), 
                gym.spaces.Box(-1.0, 1.0, shape=(2,))))
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(2 * self.n_servers + 2,))
        
        # Conversion variables for state/action mapping normalization
        self.alow = np.array((self.crah_min_flow, self.crah_min_temp))
        self.ahigh = np.array((self.crah_max_flow, self.crah_max_temp))
        self.slow = np.concatenate((20 * np.ones(self.n_servers), self.server_idle_load * np.ones(self.n_servers), [0, 0]))
        self.shigh = np.concatenate((80 * np.ones(self.n_servers), self.server_max_load * np.ones(self.n_servers), [self.job_load, self.job_time]))

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

        cwd = os.getcwd()
        os.chdir(RAFSINE_BASE)
        self.sim = Simulation(f"{RAFSINE_BASE}/problems/ocp/project.lbm")
        os.chdir(cwd)

        self.server_load = self.server_idle_load * np.ones(self.n_servers)
        self.server_flow = self.server_idle_flow * np.ones(self.n_servers)

        self.crah_flow = self.crah_min_flow * np.ones(self.n_crah)
        self.crah_temp_out = self.crah_min_temp * np.ones(self.n_crah)

        self.sim.set_time_averaging_period(self.dt) # we take one second steps and average sensor readings over that time also
        self.sim.set_boundary_conditions(self.servers, [0 for _ in range(self.n_servers)], self.server_flow)
        self.sim.set_boundary_conditions(self.crah, self.crah_temp_out, self.crah_flow)
        self.sim.run(1) # Run single step so temps exists when checked first time

        self.read_data()

        self.update_server()

        self.running_jobs = []

        self.start_time = self.sim.get_time()
        self.time = 0

        self.job = self.load_generator.step(self.dt)
        self.dropped_jobs = 0

        state = np.concatenate((self.server_temp_out, self.server_load, self.job))
        return self.state_transform(state)

    def state_transform(self, state):
        state = (2 * state - self.slow - self.shigh) / (self.shigh - self.slow)
        return state

    def action_transform(self, crah):
        return (crah * (self.ahigh - self.alow) + self.alow + self.ahigh) / 2
    
    def get_time(self):
        """Returns the time in second that has passed in the simulation since it started."""
        return (self.sim.get_time() - self.start_time).total_seconds()

    def update_crah(self, settings):
        flow, temp_out = settings

        # Maybe allow individual control?
        self.crah_flow = flow * np.ones(self.n_crah)
        self.crah_temp_out = temp_out * np.ones(self.n_crah)

        self.sim.set_boundary_conditions(self.crah, self.crah_temp_out, self.crah_flow)

        self.crah_fan_power = np.sum(self.crah_max_fan_power * (self.crah_flow / self.crah_max_flow)**3)

        # If Tamb < Tout compressor is off
        self.compressor_power = np.sum((self.ambient_temp > self.crah_temp_out) * self.air_vol_heatcap * self.crah_flow * (self.crah_temp_in - self.crah_temp_out))

    def update_server(self):
        self.server_temp_cpu = self.server_temp_in + self.R * self.server_load / self.server_flow
        cpu_target_temp = self.server_idle_temp_cpu + (self.server_max_temp_cpu - self.server_idle_temp_cpu) * np.clip((self.server_load - self.server_idle_load) / (self.server_max_load - self.server_idle_load), 0, 1)
        self.server_flow = np.clip(self.server_flow * self.server_temp_cpu / cpu_target_temp, self.server_idle_flow, self.server_max_flow)
        
        delta_t = self.server_load / (self.air_vol_heatcap * self.server_flow)
        self.sim.set_boundary_conditions(self.servers, delta_t, self.server_flow)

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
        #print("CRAH settings: ", crah_settings)

        # Increment time
        self.time += self.dt

        # Place jobs in queue and remove finished
        self.dropped_jobs = 0
        load, dur = self.job # Always here, but load and dur is zero if no job
        if self.server_load[placement] + load <= self.server_max_load:
            self.server_load[placement] += load
            heapq.heappush(self.running_jobs, (self.time + dur, load, placement))
        else:
            self.dropped_jobs = 1
        while len(self.running_jobs) > 0 and self.running_jobs[0][0] <= self.time:
            _, load, placement = heapq.heappop(self.running_jobs)
            self.server_load[placement] -= load

        # Read new data from sim
        self.read_data()
        
        # Update CPU fans
        self.update_server()

        # Update CRAH fans
        self.update_crah(self.action_transform(crah_settings))

        # Run simulation based on current boundary condition
        self.sim.run(self.time - self.get_time())
            
        # Get new job, tuple of expected (load, duration)
        self.job = self.load_generator.step(self.dt)
        
        state = np.concatenate((self.server_temp_out, self.server_load, self.job))
        return self.state_transform(state), self.reward(), False, {}

    def read_data(self):
        print("Step 1.1: ", time.time())
        df = self.sim.get_averages("temperature")
        print("Step 1.2: ", time.time())
        self.server_temp_in = df[[*map(lambda x: x + "_inlet", self.servers)]].iloc[[-1]].to_numpy()[0]
        self.server_temp_out = df[[*map(lambda x: x + "_outlet", self.servers)]].iloc[[-1]].to_numpy()[0]
        self.crah_temp_in = df[[*map(lambda x: x + "_in", self.crah)]].iloc[[-1]].to_numpy()[0]
        #print("Server load: ", np.max(self.server_load), "/", np.sum(self.server_load) / len(self.server_load))
        #print("Server out: ", np.max(self.server_temp_out))
        #print("Server in: ", np.max(self.server_temp_in))
        #print("CRAH in: ", np.max(self.crah_temp_in))
        # print("Server cpu: ", np.max(self.server_temp_cpu), "/", np.sum(self.server_temp_cpu) / len(self.server_temp_cpu))
        # print("Server flow: ", np.max(self.server_flow), "/", np.sum(self.server_flow))
        # print("CRAH flow: ", np.max(self.crah_flow), "/", np.sum(self.crah_flow))

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
        self.server_max_flow = 0.04 # If using 5 we get NaN from rafsine
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
        self.crah_max_flow = 2.5
        self.crah_max_fan_power = self.n_servers * self.server_max_fan_power / self.n_crah # CRAH is twice as efficient (max is same energy as servers but double flow)

        # Env constants
        self.ambient_temp = 17