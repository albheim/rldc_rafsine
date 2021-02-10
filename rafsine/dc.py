RAFSINE_BASE = "/home/ubuntu/rafsine"

import sys
sys.path.insert(0, RAFSINE_BASE)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, Input

from python.simulation import Simulation
import pandas as pd
import os
import heapq
import numpy as np


class DCEnv:
    def __init__(self):
        self.racks = 12
        self.chassis_per_rack = 10
        self.servers_per_chassi = 3

        self.servers = ["P02R{:02}C{:02}SRV{:02}".format(rack, chassi, srv) for rack in range(1, self.racks+1) for chassi in range(1, self.chassis_per_rack+1) for srv in range(1, self.servers_per_chassi+1)]
        self.sensors = ["sensors_racks_{:02}_to_{:02}_{}_{}".format(rack, rack + 2, direction, loc) for direction in ["in", "out"] for rack in [1, 4, 7, 10] for loc in ['b', 'm', 't']]

        self.seed = 37

        self.job_rate = 0.1
        self.job_power = 20
        self.job_duration = 200
        self.max_load = 500 # Max W per server
        self.rpm = 2000
        self.idle_load = 100

        # Kinematic viscosity of air (m^2/s)
        self.nu = 1.568e-5
        # Thermal conductivity (kW/m K)
        self.k = 2.624e-5
        # Prandtl number of air
        self.Pr = 0.707

        # Chassi fan specs from PFR0812DHE fan datasheet
        # Max input power in W
        Pmax = 25.2
        # Max RPM
        Nmax = 11000.0
        # Operational speed (RPM)
        Nop = 8000.0
        # Max air flow in CFM
        Qmax = 109.7
        # Fans per server
        Nfans = 2
        # Operational power from cube law of fans
        Pop = Pmax / (Nmax / Nop)**3 * Nfans
        # Assume fan volumetric flow is proportional to power 
        # around operating point
        Qop = Qmax / (Pmax / Pop)
        # Calculate flow in m^3/s per RPM at
        self.Q_per_RPM = Qop * 0.3048**3 / 60 / Nop

    def calc_flow(self):
        # Max input power in W, 2 fans
        Pmax = 25.2 * 2 
        # Max RPM
        Nmax = 11000.0
        # Max air flow in CFM, 2 fans
        Qmax = 109.7 * 2 

        Tcpu = Tin + R*p/Q
        N = Nmin + (Nmax - Nmin) * Tcpu / Tdes
        Q = Qmax * N / Nmax
        P = Pmax * (N / Nmax)**3
    
    def reset(self):
        self.loads = [self.idle_load for _ in range(len(self.servers))]
        self.vol_flows = [self.rpm * self.Q_per_RPM for _ in range(0, len(self.servers))]
        self.event_queue = []
        
        self.rng = np.random.default_rng(self.seed)

        cwd = os.getcwd()
        os.chdir(RAFSINE_BASE)
        self.sim = Simulation(f"{RAFSINE_BASE}/problems/ocp/project.lbm")
        os.chdir(cwd)
        self.start_time = self.sim.get_time()
        self.target_time = 0
        self.sim.set_time_averaging_period(1) # we take one second steps and average sensor readings over that time also

        self.sim.set_boundary_conditions(self.servers, [40 for _ in self.vol_flows], self.vol_flows)

        return {"time": self.get_time(), "temps":np.zeros(len(self.sensors)), "loads":np.copy(self.loads)}
    
    # Calculate the expected temperature jump across the servers (convert to kW)
    def deltaT(self, p, q):
        return (p / 1000 * self.nu) / (q * self.k * self.Pr)

    def queue_load(self, srv_idx, power, start, dur):
        heapq.heappush(self.event_queue, (start, srv_idx, power, dur))

    def get_time(self):
        return (self.sim.get_time() - self.start_time).total_seconds()

    def step(self, job, dt=1):
        self.target_time += dt

        placement, (load, duration) = job
        self.queue_load(placement, load, self.get_time(), duration)
        
        start_time = self.get_time()
        jobs_done = []
        while len(self.event_queue) > 0 and self.event_queue[0][0] <= self.target_time:
            t, srv_idx, power, dur = heapq.heappop(self.event_queue)
            if t > self.get_time():
                self.sim.run(t - self.get_time())
            if dur < 0: # This is a finished job
                self.loads[srv_idx] -= power
            elif self.loads[srv_idx] < self.max_load: # This is a job that is placed on a server
                self.loads[srv_idx] += power
                temp = self.deltaT(self.loads[srv_idx], self.vol_flows[srv_idx])
                #self.sim.set_boundary_conditions([self.servers[srv_idx]], [temp], [self.vol_flows[srv_idx]])
                heapq.heappush(self.event_queue, (self.get_time() + dur, srv_idx, power, -1))
            else: # Server is full, job is queued
                heapq.heappush(self.event_queue, (self.get_time() + 1, srv_idx, power, dur))
        if self.target_time > self.get_time():
            self.sim.run(self.target_time - self.get_time())

        temps = self.get_temps()
        state = {"time": self.get_time(), "temps":temps, "loads":np.copy(self.loads)}
        split_idx = len(self.sensors) // 2
        reward = -(np.var(temps[:split_idx]) + np.var(temps[split_idx:])) # Check variance of inlets and outlets seperately
        return state, reward

    def get_temps(self):
        df = self.sim.get_averages("temperature")[[*self.sensors]]
        return df.iloc[[-1]].to_numpy()[0]

    def show_hist(self, srv_idxs, start, stop):
        racks = [((idx // (self.chassis_per_rack * self.servers_per_chassi)) // self.racks) * self.racks + 1 for idx in srv_idxs] # Find lower index of rack in range for average
        sensor_names = ["sensors_racks_{:02}_to_{:02}_out_{}".format(rack, rack + 2, loc) for rack in racks for loc in ['b', 'm', 't']]
        df = self.sim.get_averages("temperature")[["time", *sensor_names]]
        df["time"] = (df.time - self.start_time).dt.total_seconds()
        return df.set_index('time').loc[start:stop]
