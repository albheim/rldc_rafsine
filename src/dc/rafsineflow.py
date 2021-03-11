import numpy as np
import sys
import os

RAFSINE_BASE = "/home/ubuntu/rafsine"
sys.path.insert(0, RAFSINE_BASE)
from python.simulation import Simulation

class RafsineFlow:
    def __init__(self, dt):
        # Server layout
        self.n_racks = 12
        self.chassis_per_rack = 10
        self.servers_per_chassi = 3
        
        self.servers_per_rack = self.chassis_per_rack * self.servers_per_chassi

        self.server_names = ["P02R{:02}C{:02}SRV{:02}".format(rack, chassi, srv) for rack in range(1, self.n_racks+1) for chassi in range(1, self.chassis_per_rack+1) for srv in range(1, self.servers_per_chassi+1)]
        self.crah_names = ["P02HDZ{:02}".format(i+1) for i in range(4)]

        self.n_servers = len(self.server_names)
        self.n_crah = len(self.crah_names)

        self.dt = dt

    def reset(self, servers, crah):
        cwd = os.getcwd()
        os.chdir(RAFSINE_BASE)
        self.sim = Simulation(f"{RAFSINE_BASE}/problems/ocp/project.lbm")
        os.chdir(cwd)

        self.sim.set_time_averaging_period(self.dt) # we take one second steps and average sensor readings over that time also
        self.sim.set_boundary_conditions(self.server_names, servers.delta_t, servers.flow)
        self.sim.set_boundary_conditions(self.crah_names, crah.temp_out, crah.min_flow / self.n_crah * np.ones(self.n_crah))

        self.sim.run(1) # Run single step so temps exists when checked first time
        self.start_time = self.sim.get_time()
        self.target_time = 0

        df = self.sim.get_averages("temperature")
        self.server_temp_in = df[[*map(lambda x: x + "_inlet", self.server_names)]].iloc[[-1]].to_numpy()[0]
        self.server_temp_out = df[[*map(lambda x: x + "_outlet", self.server_names)]].iloc[[-1]].to_numpy()[0]
        self.crah_temp_in = df[[*map(lambda x: x + "_in", self.crah_names)]].iloc[[-1]].to_numpy()[0]
    
    def step(self, servers, crah):
        self.sim.set_boundary_conditions(self.server_names, servers.delta_t, servers.flow)
        self.sim.set_boundary_conditions(self.crah_names, crah.temp_out, crah.flow / self.n_crah * np.ones(self.n_crah))
        self.target_time += self.dt
        duration = self.target_time - (self.sim.get_time() - self.start_time).total_seconds()
        self.sim.run(duration) 

        df = self.sim.get_averages("temperature")
        self.server_temp_in = df[[*map(lambda x: x + "_inlet", self.server_names)]].iloc[[-1]].to_numpy()[0]
        self.server_temp_out = df[[*map(lambda x: x + "_outlet", self.server_names)]].iloc[[-1]].to_numpy()[0]
        self.crah_temp_in = df[[*map(lambda x: x + "_in", self.crah_names)]].iloc[[-1]].to_numpy()[0]
